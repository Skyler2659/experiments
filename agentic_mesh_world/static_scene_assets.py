"""Whole-object static appearance assets fitted to calibrated RGB-D observations."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .mesh_world import MeshCodeWorld,sha256,pose_matrix,matrix_pose
from .quality_reconstruction import points,write


def prepare(config):
    cfg=json.loads(Path(config).read_text());cap=Path(cfg['capture']);data=json.loads(cap.read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);views={v['view_id']:v for v in data['views']}
    reference=views[cfg['reference_view_id']];raw=Image.open(cap.parent/reference['rgb_uri']).convert('RGB');raw.save(out/'reference.png')
    atlas=Image.new('RGB',(1024,288*((len(cfg['objects'])+3)//4)),'white');draw=ImageDraw.Draw(atlas)
    for j,spec in enumerate(cfg['objects']):
        image=raw
        if spec.get('reference_view_id'):
            v=views[spec['reference_view_id']]
            if v['role']!='fit':raise ValueError('Identity reference must be FIT')
            image=Image.open(cap.parent/v['rgb_uri']).convert('RGB')
        crop=image.crop(spec['reference_crop_xyxy']);crop.thumbnail((248,250));x=(j%4)*256;y=(j//4)*288
        atlas.paste(crop,(x+(256-crop.width)//2,y+32+(250-crop.height)//2));draw.text((x+5,y+5),spec['object_id'],fill='black')
    atlas.save(out/'identity_atlas.jpg',quality=97);rows=[]
    for vid in cfg['view_ids']:
        v=views[vid]
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        rgb=cap.parent/v['rgb_uri']
        if sha256(rgb)!=v['rgb_sha256']:raise ValueError('Original RGB changed')
        Image.open(rgb).save(out/f'{vid}.png');rows.append(v)
    request={'capture_sha256':sha256(cap),'source_scene_sha256':sha256(cfg['source_scene']),
             'objects':cfg['objects'],'regions':cfg.get('regions',[]),'views':rows,'holdout_used':False}
    write(out/'request.json',request);write(out/'config.json',cfg);return {'status':'NEED_WHOLE_OBJECT_PROMPTS','views':len(rows)}


def collect(config):
    """Keep calibrated whole-instance samples; discard image-edge depth spill."""
    from scipy.ndimage import binary_erosion
    import open3d as o3d
    cfg=json.loads(Path(config).read_text());cap=Path(cfg['capture']);data=json.loads(cap.read_text())
    ann=Path(cfg['annotations']);packet=json.loads(ann.read_text())
    if packet['capture_sha256']!=sha256(cap):raise ValueError('Mask capture changed')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    views={v['view_id']:v for v in data['views']};samples={};bindings={}
    for row in packet['views']:
        v=views[row['view_id']]
        if v['role']!='fit':raise ValueError('Whole-body fitting requires FIT observations')
        rgb=cap.parent/v['rgb_uri'];dep=cap.parent/v['depth_uri']
        if sha256(rgb)!=row['rgb_sha256'] or sha256(dep)!=v['depth_sha256']:raise ValueError('RGB-D changed')
        depth=np.load(dep);xyz=points(depth,np.asarray(v['K']),pose_matrix(v['T_W_C']))
        color=np.asarray(Image.open(rgb).convert('RGB'))
        for oid,ref in row['masks'].items():
            mask_file=ann.parent/ref['uri']
            if sha256(mask_file)!=ref['sha256']:raise ValueError('Mask changed')
            mask=np.asarray(Image.open(mask_file).convert('L'))>127
            interior=binary_erosion(mask,iterations=1)
            valid=interior&np.isfinite(depth)&(depth>0)&np.isfinite(xyz).all(2)
            iy,ix=np.nonzero(valid);pick=np.linspace(0,len(ix)-1,min(len(ix),12000),dtype=int)
            if not len(pick):continue
            samples.setdefault(oid,[]).append((xyz[iy[pick],ix[pick]],color[iy[pick],ix[pick]]))
            bindings.setdefault(oid,[]).append({'view_id':v['view_id'],'rgb_sha256':sha256(rgb),'depth_sha256':sha256(dep),'mask_sha256':sha256(mask_file),'sample_count':len(pick)})
    stats={}
    for oid,rows in samples.items():
        xyz=np.concatenate([r[0] for r in rows]);rgb=np.concatenate([r[1] for r in rows])
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        _,index=cloud.remove_statistical_outlier(nb_neighbors=24,std_ratio=2.5)
        xyz=xyz[index];rgb=rgb[index]
        np.savez_compressed(out/f'{oid}.npz',xyz_W_m=xyz,rgb=rgb)
        stats[oid]={'points':len(xyz),'bounds_005_995_m':np.quantile(xyz,[.005,.995],axis=0).tolist(),'bindings':bindings[oid]}
    write(out/'samples.json',{'capture_sha256':sha256(cap),'annotations_sha256':sha256(ann),'objects':stats,'holdout_used':False})
    write(out/'config.json',cfg);return stats


def build(config):
    import trimesh,open3d as o3d
    from scipy.spatial import cKDTree
    from .static_asset_geometry import fit_recipe,make_recipe,box_part,rectangle_frame
    from .observed_texture import bake
    from .surface_quality import topology
    cfg=json.loads(Path(config).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    cap=Path(cfg['capture']);data=json.loads(cap.read_text());source=MeshCodeWorld.load(cfg['source_scene'])
    samples=Path(cfg['samples']);stats=json.loads((samples/'samples.json').read_text())
    ann=Path(cfg['annotations']);annotations={r['view_id']:r for r in json.loads(ann.read_text())['views']}
    if stats['capture_sha256']!=sha256(cap):raise ValueError('Sample capture changed')
    selected={b['view_id'] for v in stats['objects'].values() for b in v['bindings']}
    views=[v for v in data['views'] if v['view_id'] in selected]
    clouds={};reports=[];nodes=[];evidence=out/'evidence';evidence.mkdir()
    for oid in stats['objects']:
        z=np.load(samples/f'{oid}.npz');xyz=z['xyz_W_m'];rgb=z['rgb']
        workspace=cfg.get('workspace_bounds_W_m',{}).get(oid)
        parent='wine_rack_1' if oid.startswith('rack_') else oid
        if workspace is None and parent in {'wine_rack_1','wooden_cabinet_1','akita_black_bowl_1'}:
            workspace=(source.mesh(parent).bounds+np.array([[-.006]*3,[.006]*3])).tolist()
        if workspace is not None:
            b=np.asarray(workspace);keep=np.all(xyz>=b[0],1)&np.all(xyz<=b[1],1);xyz=xyz[keep];rgb=rgb[keep]
        if len(xyz)<100:raise ValueError('Insufficient whole-object samples '+oid)
        # Discard remote mask spill as an object-level cloud operation. This does
        # not attempt to assign every historical triangle or repair fragments.
        if oid not in cfg.get('keep_multiple_sample_clusters',[]) and not oid.startswith('rack_'):
            p=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz));labels=np.asarray(p.cluster_dbscan(eps=.012,min_points=15))
            ids,counts=np.unique(labels[labels>=0],return_counts=True)
            if len(ids):
                keep=labels==ids[np.argmax(counts)];xyz=xyz[keep];rgb=rgb[keep]
        clouds[oid]=(xyz,rgb)
        np.savez_compressed(evidence/f'{oid}_fit_points.npz',xyz_W_m=xyz,rgb=rgb)
    for spec in cfg['objects']:
        oid=spec['object_id'];family=spec['family'];xyz,rgb=clouds[oid]
        if family=='measured':
            from .projective_surface import reconstruct
            measured_views=[];masked=out/'measured_depth'/oid;masked.mkdir(parents=True)
            for v in views:
                ref=annotations[v['view_id']]['masks'].get(oid)
                if not ref:continue
                mask=np.asarray(Image.open(ann.parent/ref['uri']).convert('L'))>127
                depth=np.load(cap.parent/v['depth_uri']).copy();depth[~mask]=np.nan
                file=masked/(v['view_id']+'.npy');np.save(file,depth)
                measured_views.append({**v,'depth_uri':str(file)})
            mesh,receipt=reconstruct({'views':measured_views},cap.parent,np.array([xyz.min(0)-.003,xyz.max(0)+.003]),stride=1,max_edge_m=.006)
            origin=mesh.bounds.mean(0);t=np.eye(4);t[:3,3]=origin
            bounds=mesh.bounds.tolist();mesh.apply_translation(-origin)
            directory=out/'objects'/oid;directory.mkdir(parents=True);file=directory/'mesh.glb';mesh.export(file)
            write(directory/'geometry_recipe.json',{'family':'measured_surface','assumptions':[],'reconstruction':receipt})
            nodes.append({'object_id':oid,'parent_id':None,'pose_parent_object':matrix_pose(t),'state':'ESTIMATED','asset':{'mesh_ref':str(file.relative_to(out)),'sha256':sha256(file),'T_object_mesh':np.eye(4).tolist(),'up_axis':'Z','provenance':{'source':'calibrated masked RGB-D adjacent-pixel surface','generated_asset_used':False,'observed_inputs':stats['objects'][oid]['bindings'],'measured_evidence_ref':str((evidence/f'{oid}_fit_points.npz').relative_to(out)),'assumptions':[],'unseen_surface':'unknown'}}})
            world_mesh=mesh.copy();world_mesh.apply_translation(origin)
            ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world_mesh.vertices),o3d.utility.Vector3iVector(world_mesh.faces))))
            errors=ray.compute_distance(o3d.core.Tensor(xyz.astype('float32')),nthreads=2).numpy()
            reports.append({'object_id':oid,'family':family,'topology':topology(mesh)[0],'asset_faces':len(mesh.faces),'bounds_W_m':bounds,'fit_points':len(xyz),'fit_point_to_surface_p95_m':float(np.quantile(errors,.95)),'fit_point_to_surface_median_m':float(np.median(errors)),'independent_acceptance':False,'surface_receipt':receipt})
            print('measured asset',oid,'faces',len(mesh.faces),flush=True);continue
        recipe_regions=dict(clouds)
        for role,region_id in spec.get('shape_regions',{}).items():
            if region_id not in clouds:raise ValueError('Missing measured shape region '+oid+' '+region_id)
            recipe_regions[role]=clouds[region_id]
        recipe,rotation=fit_recipe(family,xyz,rgb,recipe_regions)
        mesh=make_recipe(recipe)
        before=topology(mesh)[0]
        if not before['watertight']:raise ValueError('Generated body is not a closed material shell '+oid)
        # Dense triangles allow original-image projections over planar bodies;
        # triangulation density is an appearance asset detail, not Agent input.
        edge=float(spec.get('texture_edge_m',.006))
        vertices,faces=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=edge,max_iter=8)
        mesh=trimesh.Trimesh(vertices,faces,process=False)
        distance,nearest=cKDTree(xyz).query(mesh.vertices);colors=rgb[nearest].copy()
        if family=='rack':
            shelf_parts=[make_recipe({'parts':[p]}) for p in recipe['parts'] if p['kind']=='field_slab']
            shelf_vertices=np.concatenate([p.vertices for p in shelf_parts]);sd,_=cKDTree(shelf_vertices).query(mesh.vertices)
            defaults=np.where((sd<.013)[:,None],np.array([139,118,85]),np.array([66,31,17]))
        else:defaults=np.broadcast_to(np.median(rgb,axis=0),colors.shape)
        colors[distance>.004]=defaults[distance>.004];mesh.visual.vertex_colors=colors
        ray=o3d.t.geometry.RaycastingScene();om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
        ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om));errors=ray.compute_distance(o3d.core.Tensor(xyz.astype('float32'))).numpy()
        # Clip identity masks into texture depth: a neighbor visible in a source
        # photo must never be baked onto a completed target face.
        obj_views=[];texture_depth=out/'_texture_depth';texture_depth.mkdir(exist_ok=True)
        for v in views:
            if v['view_id'] not in {b['view_id'] for b in stats['objects'][oid]['bindings']}:continue
            mask=np.asarray(Image.open(ann.parent/annotations[v['view_id']]['masks'][oid]['uri']).convert('L'))>127
            d=np.load(cap.parent/v['depth_uri']);d=d.copy();d[~mask]=np.nan
            f=texture_depth/(oid+'_'+v['view_id']+'.npy');np.save(f,d)
            obj_views.append({**v,'depth_uri':str(f)})
        textured,texture=bake(mesh,obj_views,cap.parent,tolerance_m=float(cfg.get('texture_tolerance_m',.006)),max_views=7)
        origin=(mesh.bounds[0]+mesh.bounds[1])/2;t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=origin
        textured.apply_transform(np.linalg.inv(t));directory=out/'objects'/oid;directory.mkdir(parents=True)
        file=directory/'mesh.glb';textured.export(file)
        # OBJ exporter writes the atlas and material alongside the independent mesh.
        textured.export(directory/'mesh.obj')
        write(directory/'geometry_recipe.json',recipe)
        (directory/'model.py').write_text('import json\nfrom pathlib import Path\nfrom agentic_mesh_world.static_asset_geometry import make_recipe\ndef make_mesh():\n    return make_recipe(json.loads(Path(__file__).with_name("geometry_recipe.json").read_text()))\n')
        provenance={'source':'whole-object static reconstruction fitted to calibrated FIT RGB-D','generated_asset_used':True,'observed_inputs':stats['objects'][oid]['bindings'],
                    'assumptions':recipe['assumptions'],'geometry_recipe_ref':str((directory/'geometry_recipe.json').relative_to(out)),'texture':texture,
                    'measured_evidence_ref':str((evidence/f'{oid}_fit_points.npz').relative_to(out))}
        nodes.append({'object_id':oid,'parent_id':None,'pose_parent_object':matrix_pose(t),'state':'ESTIMATED',
                      'asset':{'mesh_ref':str(file.relative_to(out)),'sha256':sha256(file),'T_object_mesh':np.eye(4).tolist(),'up_axis':'Z','provenance':provenance}})
        reports.append({'object_id':oid,'family':family,'topology':before,'asset_faces':len(textured.faces),'bounds_W_m':mesh.bounds.tolist(),'fit_points':len(xyz),
                        'fit_point_to_surface_median_m':float(np.median(errors)),'fit_point_to_surface_p95_m':float(np.quantile(errors,.95)),'texture':texture,'independent_acceptance':False})
        print('static asset',oid,'p95',reports[-1]['fit_point_to_surface_p95_m'],flush=True)
    # Support is a continuous slab of the observed table footprint. Robot is a
    # known public FK asset at recorded state, explicitly distinct provenance.
    if any(n['object_id']=='measured_tabletop' for n in source.source['nodes']):
        table=source.mesh('measured_tabletop');vertices=table.vertices
    else:
        # Large upward plane from FIT depth only; no native table asset/pose.
        table_samples=[]
        for v in views:
            d=np.load(cap.parent/v['depth_uri']);p=points(d,np.asarray(v['K']),pose_matrix(v['T_W_C']))
            valid=np.isfinite(p).all(2)&(d>0);p=p[valid][::20]
            table_samples.append(p)
        p=np.concatenate(table_samples)
        target=np.concatenate([clouds[s['object_id']][0] for s in cfg['objects']]);low,high=np.quantile(target,[.002,.998],axis=0)
        # Select the work surface near object bases, not a distant floor/wall.
        keep=np.all(p[:,:2]>=low[:2]-.25,1)&np.all(p[:,:2]<=high[:2]+.25,1)&(p[:,2]>low[2]-.04)&(p[:,2]<low[2]+.015)
        p=p[keep];pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        plane,idx=pc.segment_plane(distance_threshold=.0015,ransac_n=3,num_iterations=2000)
        if abs(plane[2])<.95:raise ValueError('Observed dominant plane is not horizontal support')
        vertices=p[idx]
    z=np.median(vertices[:,2]);vertices=vertices[np.abs(vertices[:,2]-z)<.0015]
    if cfg.get('preserve_table_outline',False):
        from scipy.spatial import ConvexHull
        hull=ConvexHull(vertices[:,:2]);part={'kind':'polygon_slab','dimensions':{'outline':vertices[hull.vertices,:2].tolist(),'z':float(z),'thickness':.025}}
    else:
        part,rotation=box_part(vertices);part['size_m'][2]=.025;tt=np.asarray(part['T_W_part']);tt[2,3]=z-.0125;part['T_W_part']=tt.tolist()
    recipe={'family':'table','units':'m','frame':'robot_base','parts':[part],'assumptions':['observed footprint; hidden underside and 25 mm thickness completed for static appearance']}
    mesh=make_recipe(recipe)
    vv,ff=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.025,max_iter=8)
    mesh=trimesh.Trimesh(vv,ff,process=False);mesh.visual.vertex_colors=[181,164,136,255]
    mesh,table_texture=bake(mesh,views,cap.parent,tolerance_m=.0025,max_views=7)
    directory=out/'objects'/'measured_tabletop';directory.mkdir();file=directory/'mesh.glb';mesh.export(file);mesh.export(directory/'mesh.obj');write(directory/'geometry_recipe.json',recipe)
    nodes.append({'object_id':'measured_tabletop','parent_id':None,'pose_parent_object':matrix_pose(np.eye(4)),'state':'ESTIMATED','asset':{'mesh_ref':str(file.relative_to(out)),'sha256':sha256(file),'T_object_mesh':np.eye(4).tolist(),'up_axis':'Z','provenance':{'generated_asset_used':True,'source':'continuous slab of measured table footprint','assumptions':recipe['assumptions']}}})
    if any(n['object_id']=='known_robot' for n in source.source['nodes']):
        robot=copy.deepcopy(source.node('known_robot'));directory=out/'objects'/'known_robot';directory.mkdir();file=directory/'mesh.glb';shutil.copy2(source.root/robot['asset']['mesh_ref'],file)
        robot['asset']['mesh_ref']=str(file.relative_to(out));nodes.append(robot)
    elif cfg.get('public_robot_snapshot'):
        from .mesh_context import public_robot_mesh
        snapshot=Path(cfg['public_robot_snapshot']);robot_mesh,provenance=public_robot_mesh(snapshot)
        directory=out/'objects'/'known_robot';directory.mkdir();file=directory/'mesh.glb';robot_mesh.export(file)
        provenance.update(snapshot_sha256=sha256(snapshot),snapshot_ref=str(snapshot),state_scope='historical recorded FK appearance; legacy archive lacks full synchronized joint-state contract')
        nodes.append({'object_id':'known_robot','parent_id':None,'pose_parent_object':matrix_pose(np.eye(4)),'state':'ESTIMATED','role':'known_robot',
                      'asset':{'mesh_ref':str(file.relative_to(out)),'sha256':sha256(file),'up_axis':'Z','T_object_mesh':np.eye(4).tolist(),'provenance':provenance}})
    scene={'schema_version':'mesh-code-v1','world_id':cfg.get('world_id','drawer_static_asset_release_v1'),'world_version':1,'frame':'robot_base','units':'m','nodes':nodes,'observations':[],
           'quality':{'reconstruction_acceptance':'PENDING','metric_contact_ready':False,'asset_release':'static appearance v1; generated continuity is a hypothesis'}}
    for v in views:
        target=evidence/(v['view_id']+'.png');shutil.copy2(cap.parent/v['rgb_uri'],target)
        scene['observations'].append({**v,'rgb_ref':str(target.relative_to(out))})
    write(out/'scene.json',scene);MeshCodeWorld.load(out/'scene.json').export_glb(out/'scene.glb')
    scene['robot_state']=source.source.get('robot_state',{})
    write(out/'scene.json',scene)
    write(out/'report.json',{'status':'STATIC_ASSETS_BUILT','objects':reports,'object_count':len(cfg['objects']),'support_count':len(nodes)-len(cfg['objects']),'holdout_used':False,'independent_acceptance':False,'code_world_schema_fixed':False})
    write(out/'config.json',cfg);write(out/'input_bindings.json',stats)
    shutil.rmtree(out/'_texture_depth')
    return {'scene':str(out/'scene.json'),'assets':len(nodes)}


def texture(config):
    """Refresh appearance from the closest depth-visible original FIT views."""
    from .instance_surface import project_samples
    from .observed_texture import bake
    cfg=json.loads(Path(config).read_text());source=MeshCodeWorld.load(cfg['source_scene'])
    out=Path(cfg['output']);shutil.copytree(source.root,out)
    cap=Path(cfg['capture']);data=json.loads(cap.read_text());scene=copy.deepcopy(source.source);used={};reports=[]
    candidates=[v for v in data['views'] if v['role']=='fit' and v.get('surface_enabled',True) and max(v['image_size'])<=1024]
    depths={}
    for v in candidates:
        if sha256(cap.parent/v['rgb_uri'])!=v['rgb_sha256'] or sha256(cap.parent/v['depth_uri'])!=v['depth_sha256']:raise ValueError('Texture original changed')
        depths[v['view_id']]=np.load(cap.parent/v['depth_uri'])
    for node in scene['nodes']:
        oid=node['object_id']
        if oid=='known_robot':continue
        mesh=source.mesh(oid);mesh.visual=mesh.visual.to_color() if hasattr(mesh.visual,'to_color') else mesh.visual
        index=np.linspace(0,len(mesh.faces)-1,min(600,len(mesh.faces)),dtype=int);centers=mesh.triangles_center[index];normals=mesh.face_normals[index]
        qualities=[]
        for v in candidates:
            _,_,visible=project_samples(centers[:,None,:],v,depths[v['view_id']],float(cfg.get('tolerance_m',.006)))
            delta=np.asarray(v['T_W_C']['translation_m'])-centers;dist=np.linalg.norm(delta,axis=1)
            angle=abs(np.sum(normals*delta/np.maximum(dist[:,None],1e-8),axis=1))
            qualities.append(visible[:,0]*angle*np.asarray(v['K'])[0,0]/np.maximum(dist,1e-8))
        qualities=np.asarray(qualities);current=np.zeros(len(index));chosen=[]
        for _ in range(7):
            score=np.maximum(qualities-current,0).sum(1);pick=int(np.argmax(score))
            if score[pick]<=0:break
            chosen.append(candidates[pick]);current=np.maximum(current,qualities[pick]);qualities[pick]=0
        if not chosen:raise ValueError('No original depth-visible texture '+oid)
        result,report=bake(mesh,chosen,cap.parent,tolerance_m=float(cfg.get('tolerance_m',.006)),max_views=7,max_atlas_size=4096)
        result.apply_transform(np.linalg.inv(source.object_transform(oid)))
        file=out/node['asset']['mesh_ref'];result.export(file);result.export(file.with_suffix('.obj'))
        node['asset']['up_axis']='Z'
        node['asset']['T_object_mesh']=np.eye(4).tolist()
        node['asset']['sha256']=sha256(file);node['asset']['provenance']['texture']=report
        node['asset']['provenance']['texture']['identity_basis']='candidate projection with original measured depth agreement; auxiliary FIT photos are not independent identity validation'
        for v in chosen:used[v['view_id']]=v
        reports.append({'object_id':oid,**report});print('texture',oid,[v['view_id'] for v in chosen],report['projected_face_fraction'],flush=True)
    existing={v['view_id'] for v in scene['observations']}
    for vid,v in used.items():
        file=out/'evidence'/(vid+'.png');shutil.copy2(cap.parent/v['rgb_uri'],file)
        if vid not in existing:scene['observations'].append({**v,'rgb_ref':str(file.relative_to(out))})
    scene['world_version']+=1;write(out/'scene.json',scene);textured=MeshCodeWorld.load(out/'scene.json')
    for node in scene['nodes']:
        oid=node['object_id']
        if not np.allclose(textured.mesh(oid).bounds,source.mesh(oid).bounds,atol=1e-6,rtol=0):
            raise ValueError('Texture export moved W geometry '+oid)
    textured.export_glb(out/'scene.glb')
    write(out/'texture_report.json',{'objects':reports,'geometry_changed':False,'holdout_used':False,'source_scene_sha256':sha256(cfg['source_scene'])})
    return {'scene':str(out/'scene.json'),'textured_assets':len(reports)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','build','texture']);p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps(globals()[a.stage](a.config)))
