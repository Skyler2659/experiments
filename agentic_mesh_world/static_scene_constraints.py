"""Evidence-bound scene boundary/contact audit and static hidden-base completion."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from .mesh_world import MeshCodeWorld,pose_matrix,sha256
from .quality_reconstruction import points,write
from .static_surface_constraints import fit_plane,contact_metrics,extend_hidden_base
from .surface_quality import topology


def support_samples(capture,annotations,bounds):
    cap=Path(capture);ann=Path(annotations);data=json.loads(cap.read_text());packet=json.loads(ann.read_text())
    if sha256(cap)!=packet['capture_sha256']:raise ValueError('Capture binding changed')
    by={v['view_id']:v for v in data['views']};clouds=[];records=[]
    low,high=np.asarray(bounds)
    for row in packet['views']:
        v=by[row['view_id']]
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        depth=cap.parent/v['depth_uri']
        if sha256(depth)!=v['depth_sha256']:raise ValueError('Depth changed')
        d=np.load(depth);xyz=points(d,np.asarray(v['K']),pose_matrix(v['T_W_C']));exclude=np.zeros(d.shape,bool)
        for ref in row['masks'].values():
            file=ann.parent/ref['uri']
            if sha256(file)!=ref['sha256']:raise ValueError('Mask changed')
            exclude|=np.asarray(Image.open(file).convert('L'))>127
        valid=np.isfinite(xyz).all(2)&(d>0)&~exclude&(xyz[:,:,2]>low[2]-.045)&(xyz[:,:,2]<low[2]+.03)&np.all(xyz[:,:,:2]>=low[:2]-.16,2)&np.all(xyz[:,:,:2]<=high[:2]+.16,2)
        sample=xyz[valid][::10]
        if len(sample)<300:continue
        plane,receipt=fit_plane(sample);residual=sample@plane[:3]+plane[3];selected=sample[abs(residual)<.0015];clouds.append(selected)
        records.append({'view_id':v['view_id'],'depth_sha256':v['depth_sha256'],'plane_W':plane.tolist(),**receipt})
    if len(clouds)<3:raise ValueError('Need support evidence in three FIT views')
    return np.concatenate(clouds),records


def fit_error(mesh,xyz):
    import open3d as o
    ray=o.t.geometry.RaycastingScene();ray.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))))
    error=ray.compute_distance(o.core.Tensor(np.asarray(xyz,np.float32)),nthreads=2).numpy()
    return {'fit_point_to_surface_median_m':float(np.median(error)),'fit_point_to_surface_p95_m':float(np.quantile(error,.95))}


def boundary_audit(mesh,recipe):
    rows=[]
    for i,part in enumerate(recipe.get('parts',[])):
        if part['kind']!='profile_panel':continue
        probes=np.vstack([mesh.vertices,mesh.triangles_center,mesh.vertices[mesh.edges_unique].mean(1)])
        t=np.asarray(part['T_W_part']);p=(probes-t[:3,3])@t[:3,:3];axis=part['thin_axis'];span=1-axis;poly=np.asarray(part['polygon_span_z_m']);left,right=poly[0,0],poly[1,0]
        use=(abs(p[:,axis])<part['thickness_m']/2+.0015)&(p[:,span]>left+.005)&(p[:,span]<right-.005)
        upper=np.interp(p[use,span],poly[[3,2],0],poly[[3,2],1]);excess=p[use,2]-upper
        max_excess=float(np.max(excess)) if len(excess) else None
        rows.append({'part_index':i,'max_boundary_excess_m':max_excess,'status':'boundary_consistent' if max_excess is not None and max_excess<=.006 else 'boundary_review_required'})
    return rows


def silhouette_audit(mesh,oid,capture,annotations,output):
    """Check occupied/free-space errors, including unmodeled bevels and curves."""
    import open3d as o
    from scipy.ndimage import binary_dilation
    cap=Path(capture);ann=Path(annotations);data=json.loads(cap.read_text());packet=json.loads(ann.read_text());by={v['view_id']:v for v in data['views']}
    rows=[r for r in packet['views'] if oid in r['masks']]
    rows=[rows[i] for i in sorted(set([0,len(rows)//2,len(rows)-1]))] if rows else []
    ray=o.t.geometry.RaycastingScene();ray.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))))
    output=Path(output);output.mkdir(parents=True,exist_ok=True);reports=[]
    for row in rows:
        v=by[row['view_id']]
        if v['role']!='fit':raise ValueError('Silhouette fitting audit may use FIT only')
        depth=np.load(cap.parent/v['depth_uri']);t=pose_matrix(v['T_W_C']);p=points(np.ones_like(depth),np.asarray(v['K']),t)[::4,::4];origin=t[:3,3]
        rays=np.concatenate([np.broadcast_to(origin,p.shape),p-origin],2).astype('float32')
        hit=ray.cast_rays(o.core.Tensor(rays),nthreads=2)['t_hit'].numpy();d=depth[::4,::4]
        mask=np.asarray(Image.open(ann.parent/row['masks'][oid]['uri']).convert('L'))[::4,::4]>127
        valid=np.isfinite(d)&(d>0);visible=np.isfinite(hit)&valid&(hit<=d+.006)
        excess=visible&~binary_dilation(mask,iterations=1)&(hit<d-.006)
        missing=mask&valid&(~np.isfinite(hit)|(hit>d+.006))
        matched=mask&valid&np.isfinite(hit)&(abs(hit-d)<=.006)
        total=max(1,int(visible.sum()));observed=max(1,int((mask&valid).sum()))
        image=np.asarray(Image.open(cap.parent/v['rgb_uri']).convert('RGB'))[::4,::4].copy();image[excess]=[255,40,40];image[missing]=[40,80,255];image[matched]=(image[matched]*.5+np.array([20,220,70])*.5).astype('uint8')
        file=output/(v['view_id']+'.png');Image.fromarray(image).save(file)
        reports.append({'view_id':v['view_id'],'observed_mask_pixels':observed,'false_occupied_fraction':float(excess.sum()/total),'missing_observed_fraction':float(missing.sum()/observed),'overlay':str(file),'status':'silhouette_review_required' if excess.sum()/total>.05 or missing.sum()/observed>.10 else 'silhouette_consistent'})
    return reports


def table_resting_candidates(meshes, plane, max_gap):
    """Limit table completion to its existing evidence budget; retain other poses."""
    resting=[];unknown=[]
    for oid,mesh in meshes.items():
        contact=contact_metrics(mesh,plane)
        gap=max(contact['signed_min_gap_m'],contact['dominant_base_gap_m'])
        (resting if gap<=max_gap else unknown).append(oid)
    return resting,unknown


def run(config):
    cfg=json.loads(Path(config).read_text());source=MeshCodeWorld.load(cfg['source_scene']);out=Path(cfg['output'])
    ids=[n['object_id'] for n in source.source['nodes'] if n['object_id'] not in {'known_robot','measured_tabletop'}]
    meshes={oid:source.mesh(oid) for oid in ids};low=np.min([m.bounds[0] for m in meshes.values()],0);high=np.max([m.bounds[1] for m in meshes.values()],0)
    xyz,views=support_samples(cfg['capture'],cfg['annotations'],[low,high]);plane,plane_report=fit_plane(xyz)
    center=(low+high)/2;heights=[-(np.asarray(v['plane_W'])[:2]@center[:2]+v['plane_W'][3])/v['plane_W'][2] for v in views]
    spread=float(np.ptp(heights))
    if spread>.003:raise ValueError('Cross-view support planes disagree; do not snap objects')
    shutil.copytree(source.root,out);scene=copy.deepcopy(source.source);report=json.loads((out/'report.json').read_text());rows=[]
    table=source.mesh('measured_tabletop');table_h=table.vertices@plane[:3]+plane[3];table_offset=float(np.quantile(table_h,.95))
    if abs(table_offset)>.002:raise ValueError('Existing tabletop differs from measured support; rebuild support asset first')
    unknown_support=[]
    if cfg.get('resting_objects')=='auto_table_candidates':
        candidates,unknown_support=table_resting_candidates(meshes,plane,float(cfg.get('max_hidden_base_gap_m',.025)))
        resting=set(candidates)
    else:
        resting=set(ids if cfg.get('resting_objects')=='all_reconstructed' else cfg.get('resting_objects',[]))
    for oid in ids:
        original=meshes[oid];mesh=original;before=contact_metrics(mesh,plane);directory=out/'objects'/oid;recipe=json.loads((directory/'geometry_recipe.json').read_text())
        boundary=boundary_audit(mesh,recipe)
        if cfg.get('boundary_recipe_by_object',{}).get(oid):
            boundary=boundary_audit(mesh,json.loads(Path(cfg['boundary_recipe_by_object'][oid]).read_text()))
        evidence=np.load(out/'evidence'/f'{oid}_fit_points.npz');measured=evidence['xyz_W_m'];original_fit=fit_error(mesh,measured);completion={'status':'not_applied'}
        if before['status']=='floating' and oid in resting:
            mesh,completion=extend_hidden_base(mesh,plane,float(cfg.get('max_hidden_base_gap_m',.025)))
            # glTF uses float32 local positions. Verify that representation now,
            # before generating appearance or publishing a successful receipt.
            def encoded_topology(candidate):
                import trimesh
                local=candidate.copy();local.apply_transform(np.linalg.inv(source.object_transform(oid)))
                v,inv=np.unique(local.vertices.astype('float32'),axis=0,return_inverse=True)
                return topology(trimesh.Trimesh(v,inv[local.faces],process=False))[0]
            encoded=encoded_topology(mesh)
            if not encoded['watertight'] or encoded['components']!=1:
                mesh,completion=extend_hidden_base(original,plane,float(cfg.get('max_hidden_base_gap_m',.025)),force_implicit=True)
                encoded=encoded_topology(mesh)
            if not encoded['watertight'] or encoded['components']!=1:raise ValueError('Candidate fails glTF precision continuity '+oid)
            completion['float32_local_topology']=encoded
            new_fit=fit_error(mesh,measured)
            if new_fit['fit_point_to_surface_p95_m']>original_fit['fit_point_to_surface_p95_m']+.0015:
                completion={'status':'rejected_fit_degradation','candidate_completion':completion,
                            'prior_fit':original_fit,'rejected_fit':new_fit,'original_geometry_retained':True}
                mesh=original;new_fit=original_fit
            else:
                model=directory/'model.py';shutil.copy2(model,directory/'pre_support_model.py')
                write(directory/'support_completion.json',{'plane_W':plane.tolist(),**completion})
                from .static_surface_constraints import SUPPORT_REPLAY_SOURCE
                model.write_text(SUPPORT_REPLAY_SOURCE)
                recipe['support_completion']={'plane_W':plane.tolist(),**completion};write(directory/'geometry_recipe.json',recipe)
                _,near=cKDTree(measured).query(mesh.vertices);mesh.visual.vertex_colors=evidence['rgb'][near]
                local=mesh.copy();local.apply_transform(np.linalg.inv(source.object_transform(oid)))
                node=next(n for n in scene['nodes'] if n['object_id']==oid);file=out/node['asset']['mesh_ref'];local.export(file);local.export(file.with_suffix('.obj'))
                node['asset'].update(sha256=sha256(file),up_axis='Z',T_object_mesh=np.eye(4).tolist());node['asset']['provenance']['support_completion']=completion
                next(r for r in report['objects'] if r['object_id']==oid).update(**new_fit,topology=topology(mesh)[0],asset_faces=len(mesh.faces),bounds_W_m=mesh.bounds.tolist())
        else:new_fit=original_fit
        after=contact_metrics(mesh,plane)
        silhouettes=silhouette_audit(mesh,oid,cfg['capture'],cfg['annotations'],out/'quality'/'silhouettes'/oid)
        rows.append({'object_id':oid,'before':before,'after':after,'completion':completion,'fit_before':original_fit,'fit_after':new_fit,'boundary':boundary,'pose_changed':False,'pose_W_object':source.object_transform(oid).tolist(),'recipe_sha256':sha256(directory/'geometry_recipe.json'),
                     'silhouettes':silhouettes,'support_relation':'visually inferred resting hypothesis' if oid in resting else 'UNKNOWN: not a table resting candidate; pose retained'})
        print('constraint',oid,before['signed_min_gap_m'],after['signed_min_gap_m'],completion['status'],flush=True)
    scene['world_version']+=1;write(out/'scene.json',scene);write(out/'report.json',report)
    # Published receipts must refer to this candidate, not the copied old release.
    for name in ['manifest.json','blender_import_check.json','scene.glb','scene.blend']:
        (out/name).unlink(missing_ok=True)
    if (out/'previews').exists():shutil.rmtree(out/'previews')
    np.savez_compressed(out/'evidence'/'shared_support_points.npz',xyz_W_m=xyz)
    passed=all(r['after']['status']=='contact_geometry_consistent' for r in rows if r['object_id'] in resting) and all(b['status']=='boundary_consistent' for r in rows for b in r['boundary'])
    silhouette_passed=all(s['status']=='silhouette_consistent' for r in rows for s in r['silhouettes'])
    receipt={'status':'STATIC_GEOMETRY_CONSTRAINTS_PASSED' if passed and silhouette_passed and not unknown_support else 'GEOMETRY_REVIEW_REQUIRED','unknown_support_object_ids':unknown_support,'contact_and_detected_boundary_passed':passed,'silhouette_passed':silhouette_passed,'world_version':scene['world_version'],'capture_sha256':sha256(cfg['capture']),'source_scene_sha256':sha256(cfg['source_scene']),
             'support_plane_W':plane.tolist(),'support_plane_evidence':plane_report,'cross_view_height_spread_m':spread,'table_offset_m':table_offset,'views':views,'objects':rows,
             'scope':'support contact and detected panel boundaries; not full 3D acceptance or physical stability','holdout_used':False,'independent_acceptance':False}
    write(out/'geometry_constraints.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);print(json.dumps(run(p.parse_args().config)))
