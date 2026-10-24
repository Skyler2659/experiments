"""Measurement-first RGB-D surface workflow, with a shared executable mesh world.

No simulator dependency. Real scanners can provide the same capture contract.
Missing surfaces remain missing; generated candidates never count as measurements.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .mesh_world import MeshCodeWorld, pose_matrix, sha256, generated_scene_code, matrix_pose


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')


def validate_capture(data,root):
    root=Path(root);roles={'fit','holdout'}; ids=set(); states=set();files={}
    for v in data['views']:
        if v['view_id'] in ids or v['role'] not in roles:raise ValueError('Duplicate view or invalid split')
        ids.add(v['view_id']);states.add(v['state_id'])
        if v.get('surface_enabled',True) and v.get('depth_render_samples',0)>1:raise ValueError('Multisampled depth is not a calibrated point measurement; recapture without MSAA')
        if v['depth_units']!='m' or v['depth_convention']!='camera_z':raise ValueError('Require metric camera-z depth')
        if sha256(root/v['rgb_uri'])!=v['rgb_sha256']:raise ValueError('RGB changed')
        depth_path=root/v['depth_uri'];depth=np.load(depth_path,allow_pickle=False)
        depth_hash=sha256(depth_path)
        if v.get('depth_sha256') and v['depth_sha256']!=depth_hash:raise ValueError('Depth changed')
        if depth.ndim!=2 or not np.issubdtype(depth.dtype,np.number) or list(depth.shape[::-1])!=v['image_size']:
            raise ValueError('Depth dimensions differ from calibrated image size')
        if v.get('surface_enabled',True) and not (np.isfinite(depth)&(depth>0)).any():raise ValueError('No valid metric depth')
        for key in ('rgb_uri','depth_uri'):
            path=(root/v[key]).resolve();files[str(path)]=sha256(path)
        k=np.asarray(v['K']);t=pose_matrix(v['T_W_C'])
        if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid intrinsics')
        if not np.allclose(k[2],[0,0,1]) or k[0,1]!=0 or k[1,0]!=0:raise ValueError('Require rectified pinhole intrinsics without skew')
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(t[:3,:3]),1.):raise ValueError('Nonrigid camera')
    if len(states)!=1:raise ValueError('Moving objects require separate state segments, not static fusion')
    if sum(v['role']=='fit' for v in data['views'])<3 or not any(v['role']=='holdout' for v in data['views']):raise ValueError('Require multiview fit and reserved validation')
    return {'status':'VALID_INPUT_CONTRACT','calibration':data.get('calibration_status','UNVERIFIED'),
            'physical_accuracy_certified':False,'views':len(ids),'state_id':next(iter(states)),'files':files}


def points(depth,k,t):
    yy,xx=np.indices(depth.shape);z=depth
    p=np.stack([(xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z],-1)
    return p@t[:3,:3].T+t[:3,3]


def inside(p,b):return ((p>=b[0])&(p<=b[1])).all(axis=-1)


def fuse(data,root,bounds,voxel, *, truncation_voxels=4,backend='legacy_surface_crop',device='CPU:0'):
    if backend=='bounded_original_depth':
        from .bounded_tsdf import fuse as bounded
        return bounded(data,root,bounds,voxel,truncation_voxels=truncation_voxels,device=device)
    if backend!='legacy_surface_crop':raise ValueError('Unknown measured fusion backend')
    import open3d as o3d
    from PIL import Image
    vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=truncation_voxels*voxel,depth_sampling_stride=1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    counts=[]
    for v in data['views']:
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        d=np.load(root/v['depth_uri']).astype('float32');rgb=np.array(Image.open(root/v['rgb_uri']).convert('RGB'))
        if d.shape!=rgb.shape[:2] or list(rgb.shape[1::-1])!=v['image_size']:raise ValueError('RGB/depth size mismatch')
        k=np.asarray(v['K']);t=pose_matrix(v['T_W_C']);p=points(d,k,t)
        valid=np.isfinite(d)&(d>0)&inside(p,bounds);d=np.where(valid,d,0).astype('float32')
        if v.get('robot_mask_uri'):d[np.load(root/v['robot_mask_uri'])]=0
        rd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(d),depth_scale=1.,depth_trunc=5.,convert_rgb_to_intensity=False)
        intr=o3d.camera.PinholeCameraIntrinsic(d.shape[1],d.shape[0],k[0,0],k[1,1],k[0,2],k[1,2])
        vol.integrate(rd,intr,np.linalg.inv(t));counts.append({'view_id':v['view_id'],'valid_pixels':int(valid.sum())})
    mesh=vol.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    return mesh,counts


def coverage(mesh,data,root,tolerance):
    centers=mesh.triangles_center;count=np.zeros(len(centers),np.uint16)
    for v in data['views']:
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        t=pose_matrix(v['T_W_C']);p=(centers-t[:3,3])@t[:3,:3];k=np.asarray(v['K'])
        z=p[:,2];uv=p@k.T;uv=uv[:,:2]/np.maximum(z[:,None],1e-9);xy=np.rint(uv).astype(int)
        d=np.load(root/v['depth_uri']);ok=(z>0)&(xy[:,0]>=0)&(xy[:,0]<d.shape[1])&(xy[:,1]>=0)&(xy[:,1]<d.shape[0]);idx=np.flatnonzero(ok)
        count[idx]+=abs(d[xy[idx,1],xy[idx,0]]-z[idx])<tolerance
    return count


def evaluate(mesh,data,root,bounds,stride=4):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene()
    tri=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tri));rows=[]
    for v in data['views']:
        if v['role']!='holdout':continue
        d=np.load(root/v['depth_uri'])[::stride,::stride];k=np.asarray(v['K']);t=pose_matrix(v['T_W_C'])
        yy,xx=np.indices(d.shape);dirs=np.stack([(xx*stride-k[0,2])/k[0,0],(yy*stride-k[1,2])/k[1,1],np.ones_like(d)],-1)@t[:3,:3].T
        rays=np.concatenate([np.broadcast_to(t[:3,3],dirs.shape),dirs],-1).astype('float32')
        pred=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        p=dirs*d[...,None]+t[:3,3];target=np.isfinite(d)&(d>0)&inside(p,bounds)
        valid=target&np.isfinite(pred);err=abs(pred[valid]-d[valid])
        rows.append({'view_id':v['view_id'],'target_pixels':int(target.sum()),'predicted_pixels':int(valid.sum()),
            'coverage':float(valid.sum()/max(1,target.sum())),
            'median_error_m':float(np.median(err)) if len(err) else None,
            'p95_error_m':float(np.quantile(err,.95)) if len(err) else None,
            'within_3mm_fraction_of_all_target_pixels':float(np.sum(err<.003)/max(1,target.sum()))})
    return rows


def _run(config):
    import trimesh
    from PIL import Image,ImageDraw
    config=Path(config).resolve();cfg=json.loads(config.read_text());out=Path(cfg['output']).resolve()
    # Each run is immutable; no stale stage cache across changed data/configuration.
    if out.exists():
        saved=out/'status.json'
        if saved.exists() and (out/'config.json').read_bytes()==config.read_bytes():
            previous=json.loads(saved.read_text())
            if previous.get('status')=='RECONSTRUCTED_REVIEW_PENDING':
                MeshCodeWorld.load(out/'world/scene.json')
                audit=json.loads((out/'input_audit.json').read_text())
                if any(not Path(p).is_file() or sha256(Path(p))!=h for p,h in audit['files'].items()):
                    raise ValueError('Capture inputs changed; do not resume stale reconstruction')
                return previous
        raise ValueError('Existing incomplete or changed run: preserve it and choose a new output directory')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(config,out/'config.json');cap=Path(cfg['capture']).resolve();root=cap.parent
    if not cap.exists() and cfg.get('capture_command'):
        import subprocess
        command=cfg['capture_command']
        if not isinstance(command,list) or not all(isinstance(x,str) for x in command):raise ValueError('capture_command must be an argv list')
        with (out/'capture.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    data=json.loads(cap.read_text())
    audit=validate_capture(data,root)
    from .reconstruction_robot import prepare,attach,exclude_robot_pixels
    robot_binding=prepare(data,root,cfg.get('robot'))
    robot_tolerance=float(cfg.get('robot_exclusion_tolerance_m',.003))
    if not np.isfinite(robot_tolerance) or robot_tolerance<=0:raise ValueError('Positive robot exclusion tolerance required')
    robot_exclusion=exclude_robot_pixels(data,root,out,robot_binding,tolerance_m=robot_tolerance)
    audit['files']={str(root/v[key]):sha256(root/v[key]) for v in data['views'] for key in ('rgb_uri','depth_uri')}
    write(out/'input_audit.json',audit)
    import open3d
    write(out/'environment.json',{'numpy':np.__version__,'open3d':open3d.__version__,
        'implementation_sha256':sha256(Path(__file__))})
    boxes={oid:np.asarray(b,dtype=float) for oid,b in data['object_bounds_from_masked_depth'].items()}
    allbox=np.stack(list(boxes.values()));bounds=np.array([allbox[:,0].min(0),allbox[:,1].max(0)])
    bounds+=np.array([[-.08,-.08,-.012],[.08,.08,.03]])
    write(out/'status.json',{'stage':'measured_surface_fusion','status':'RUNNING'})
    voxel=cfg.get('voxel_m',.001);backend=cfg.get('surface_backend','tsdf')
    if backend=='tsdf':
        om,counts=fuse(data,root,bounds,voxel,backend=cfg.get('tsdf_backend','legacy_surface_crop'),device=cfg.get('fusion_device','CPU:0'))
        write(out/'surface_backend.json',{'backend':cfg.get('tsdf_backend','legacy_surface_crop'),
            'device':cfg.get('fusion_device','CPU:0'),'scope':'ROI selects allocated blocks; original measured rays update them, including returns behind the ROI; no closed-shape or free-space certificate'})
        mesh=trimesh.Trimesh(vertices=np.asarray(om.vertices),faces=np.asarray(om.triangles),
            vertex_colors=np.column_stack([np.clip(np.asarray(om.vertex_colors)*255,0,255).astype('uint8'),np.full(len(om.vertices),255,np.uint8)]),process=False)
    elif backend=='projective_surface':
        from .projective_surface import reconstruct
        mesh,backend_report=reconstruct(data,root,bounds,**cfg.get('projective_surface',{}))
        counts=backend_report['views'];write(out/'surface_backend.json',backend_report)
        if cfg.get('local_refinements'):raise ValueError('Projective surface already preserves separate measured sheets; local TSDF replacement requires a separate candidate')
    else:raise ValueError('Unknown measured surface backend')
    if not len(mesh.faces):raise ValueError('No measured surface reconstructed')
    refinements=[]
    for entry in cfg.get('local_refinements',[]):
        oid=entry['object_id'];local_voxel=float(entry['voxel_m'])
        if oid not in boxes or not 0 < local_voxel < voxel:raise ValueError('Invalid measured local refinement')
        box=boxes[oid]+np.array([[-.006,-.006,-.002],[.006,.006,.006]])
        fine,_=fuse(data,root,box,local_voxel,backend=cfg.get('tsdf_backend','legacy_surface_crop'),device=cfg.get('fusion_device','CPU:0'))
        fm=trimesh.Trimesh(vertices=np.asarray(fine.vertices),faces=np.asarray(fine.triangles),
            vertex_colors=np.column_stack([np.clip(np.asarray(fine.vertex_colors)*255,0,255).astype('uint8'),np.full(len(fine.vertices),255,np.uint8)]),process=False)
        if not len(fm.faces):raise ValueError('Local refinement produced no measured surfaces')
        keep=~inside(mesh.triangles_center,box)
        mesh=trimesh.util.concatenate([mesh.submesh([np.flatnonzero(keep)],append=True,repair=False),fm])
        refinements.append({'object_id':oid,'voxel_m':local_voxel,'faces':len(fm.faces),
            'boundary':'local TSDF patch; seam and part identity remain subject to review'})
    world=out/'world';(world/'assets').mkdir(parents=True);(world/'evidence').mkdir()
    mesh.export(world/'assets/measured_scene.glb')
    scene={'schema_version':'mesh-code-v1','world_id':cfg['world_id'],'world_version':cfg.get('world_version',1),'frame':'robot_base','units':'m','nodes':[],
           'observations':[],'canonical_camera':data['views'][0],
           'quality':{'metric_contact_ready':False,'sensor_model':data['sensor_model'],'unobserved_surfaces':'unknown, not filled',
                      'instance_partition':'measured mask depth ROI; candidate boundaries require visual review'}}
    for v in data['views']:
        dst=world/'evidence'/f'{v["view_id"]}.png';shutil.copy2(root/v['rgb_uri'],dst)
        visibility={}
        if v['role']=='fit':
            dep=np.load(root/v['depth_uri']);pts=points(dep,np.asarray(v['K']),pose_matrix(v['T_W_C']))
            visibility={oid:int(np.sum(inside(pts,b))) for oid,b in boxes.items()}
        scene['observations'].append({**v,'rgb_ref':str(dst.relative_to(world)),'mask_refs':{},'object_visible_pixels':visibility})
    labels=np.full(len(mesh.faces),-1,int);centers=mesh.triangles_center
    # Object ROIs come from segmented measured points, never generated mesh bounds.
    # Avoid growing down into the table. Ambiguous overlap stays explicitly unknown.
    membership=[]
    for oid,b in boxes.items():
        expanded=b+np.array([[-.004,-.004,0],[.004,.004,.004]])
        membership.append(inside(centers,expanded))
    member=np.stack(membership);single=member.sum(0)==1
    for i in range(len(boxes)):labels[single&member[i]]=i
    detail_reports={}
    names=list(boxes)+['unassigned_observed_context']
    for i,oid in enumerate(names):
        selected=labels==i if i<len(boxes) else labels==-1
        part=mesh.submesh([np.flatnonzero(selected)],append=True,repair=False)
        if not len(part.faces):continue
        from .surface_quality import clean,detail_gate
        part,detail=clean(part,data,root,**cfg.get('detail_cleaning',{}))
        detail['gate']=detail_gate(detail['after'],**cfg.get('detail_limits',{}))
        detail_reports[oid]=detail
        write(out/'detail_quality'/f'{oid}.json',detail)
        support=coverage(part,data,root,3*voxel);np.save(world/'assets'/f'{oid}_face_support.npy',support)
        frame=intrinsic_frame(part);part.apply_transform(np.linalg.inv(frame))
        file=world/'assets'/f'{oid}.glb';part.export(file)
        scene['nodes'].append({'object_id':oid,'parent_id':None,'pose_parent_object':matrix_pose(frame),
            'frame_source':'measured surface AABB center and area-weighted horizontal principal axes; semantic heading unverified',
            'state':'ESTIMATED','asset':{'mesh_ref':str(file.relative_to(world)),'sha256':sha256(file),'up_axis':'Z','T_object_mesh':np.eye(4).tolist(),
            'provenance':{'source':'measured RGBD '+backend,'generated_asset_used':False,'voxel_m':voxel if backend=='tsdf' else None,'appearance':'observed RGB vertex colors',
            'support_ref':f'assets/{oid}_face_support.npy','low_support_face_fraction':float(np.mean(support<2))}},
            'evidence_view_ids':[v['view_id'] for v in data['views'] if v['role']=='fit']})
    attach(scene,world,robot_binding)
    scene['robot_integration']['scan_exclusion']=robot_exclusion
    write(world/'scene.json',scene);(world/'scene.py').write_text(generated_scene_code());w=MeshCodeWorld.load(world/'scene.json');w.export_glb(world/'scene.glb')
    support_cfg=cfg.get('support_surface',{'enabled':True,'scale':1.6})
    if support_cfg.get('enabled',True):
        from .support_surface import expand
        write(out/'support_surface.json',expand(world/'scene.json',cap,observations=data,
            **{k:v for k,v in support_cfg.items() if k!='enabled'}))
        w=MeshCodeWorld.load(world/'scene.json');scene=w.source;w.export_glb(world/'scene.glb')
    mesh=trimesh.util.concatenate([w.mesh(n['object_id']) for n in w.source['nodes']])
    quality={'surface':{'faces':len(mesh.faces),'vertices':len(mesh.vertices),'surface_backend':backend,'voxel_m':voxel if backend=='tsdf' else None,'watertight':bool(mesh.is_watertight)},
        'local_refinements':refinements,'fit_views':counts,'holdout':evaluate(mesh,data,root,bounds) if not cfg.get('reserve_holdout',False) else [],'precision_scope':'ideal simulated RGBD sensor consistency, not real scanner accuracy',
        'pending':['unseen undersides','instance boundary review','real-device calibration','UV texture baking','structural and functional review']}
    if cfg.get('baseline_scene') and not cfg.get('reserve_holdout',False):
        baseline=MeshCodeWorld.load(cfg['baseline_scene']);bm=trimesh.util.concatenate([baseline.mesh(n['object_id']) for n in baseline.source['nodes']])
        quality['baseline_holdout']=evaluate(bm,data,root,bounds)
    if cfg.get('acceptance') and not cfg.get('reserve_holdout',False):
        instance={oid:{'measured':evaluate(mesh,data,root,b)} for oid,b in boxes.items()}
        write(out/'instance_region_audit.json',{'results':instance})
        assessment=assess_instance_regions(instance,**cfg['acceptance'])
        write(out/'acceptance.json',assessment)
        if assessment['failures']:
            write(out/'repair_request.json',{'status':'NEED_MEASUREMENT_OR_SURFACE_REPAIR',
                'failed_regions':assessment['failures'],
                'options':['new calibrated observations','local finer surface reconstruction','instance segmentation review','thin-surface alternative'],
                'validation_rule':'reviewed holdouts become development data; reserve new views before further tuning',
                'do_not':'declare structural code trustworthy before geometry review'})
        quality['observed_geometry_acceptance']=assessment['status']
    quality['detail_quality']=detail_reports
    write(out/'quality.json',quality)
    write(out/'observation_requests.json',{'status':'NEED_GEOMETRY_REVIEW',
        'requests':[{'object_id':n['object_id'],'reason':'single-view or unsupported mesh faces',
            'face_support_ref':n['asset']['provenance']['support_ref'],
            'low_support_face_fraction':n['asset']['provenance']['low_support_face_fraction'],
            'action':'acquire additional calibrated views of these surfaces; do not invent backside geometry'}
            for n in scene['nodes'] if n['asset']['provenance'].get('low_support_face_fraction',0)>.02],
        'unseen_regions':'not measurable from existing mesh coverage; underside inspection still required'})
    review_views=[data['views'][0],*[v for v in data['views'] if v['role']==('fit' if cfg.get('reserve_holdout',False) else 'holdout') and v.get('surface_enabled',True)][:2]]
    for v in review_views:
        dest=out/f'{v["view_id"]}_reconstruction.png';w.render(v,dest)
        imgs=[Image.open(root/v['rgb_uri']).convert('RGB')]
        if cfg.get('baseline_scene') and not cfg.get('reserve_holdout',False):
            bp=out/f'{v["view_id"]}_baseline.png';baseline.render(v,bp);imgs.append(Image.open(bp).convert('RGB'))
        imgs.append(Image.open(dest).convert('RGB'));panel=Image.new('RGB',(len(imgs)*768,800),'white');draw=ImageDraw.Draw(panel)
        for i,im in enumerate(imgs):panel.paste(im.resize((768,768)),(i*768,32))
        labels=['Observed RGB','Previous generated mesh','Measured RGBD mesh'] if len(imgs)==3 else ['Observed RGB','Measured RGBD mesh']
        for i,label in enumerate(labels):draw.text((i*768+10,10),label,fill='black')
        panel.save(out/f'{v["view_id"]}_comparison.jpg')
    from .surface_quality import render_detail_views
    for node in scene['nodes']:
        render_detail_views(w,node['object_id'],out/'detail_views'/node['object_id'])
    from .scene_structuring import run as structure
    stage2=structure(world/'scene.json',out/'structure',require_features=True)
    result={'status':'RECONSTRUCTED_REVIEW_PENDING','world':str(world/'scene.json'),'structure_status':stage2['status'],
            'note':'Not high-quality accepted until independent visual/geometry/structure review; no manipulation executed'}
    if quality.get('observed_geometry_acceptance')=='GEOMETRY_REPAIR_REQUIRED':
        result['status']='GEOMETRY_REPAIR_REQUIRED'
    if any(d['gate']['status']=='DETAIL_REPAIR_REQUIRED' for d in detail_reports.values()):
        result['status']='GEOMETRY_REPAIR_REQUIRED'
        result['detail_repair_required']=[oid for oid,d in detail_reports.items() if d['gate']['status']=='DETAIL_REPAIR_REQUIRED']
        write(out/'detail_repair_request.json',{'objects':result['detail_repair_required'],
            'sequence':['check sampling footprint against requested voxel size','acquire close multiview RGBD','try thin-surface reconstruction if signed-distance fusion fragments sheets','recheck geometry and turntable views'],
            'forbidden_shortcut':'remove all small components or fill unknown gaps solely to pass topology counts'})
    write(out/'status.json',result)
    # A deployment may provide closer measurements/adapter commands per object.
    # Without that evidence, emit the concrete request instead of fabricating it.
    repair_cfg=cfg.get('detail_repair')
    if result.get('detail_repair_required') and repair_cfg and repair_cfg.get('enabled',True):
        from .detail_repair import run as repair_details
        missing=[oid for oid in result['detail_repair_required'] if oid not in repair_cfg.get('objects',{})]
        if missing:
            result['detail_adapter_required']=missing
            write(out/'status.json',result)
        else:
            request={'schema_version':'detail-repair-workflow-v1','scope':'reconstruction_only',
                'base_scene':str(world/'scene.json'),'base_capture':str(cap),'output':str(out/'detail_repair'),
                'objects':{oid:repair_cfg['objects'][oid] for oid in result['detail_repair_required']},
                'acceptance':cfg.get('acceptance',{'max_p95_m':.005,'min_coverage':.98})}
            write(out/'detail_repair_workflow.json',request)
            result=repair_details(out/'detail_repair_workflow.json')
            result['initial_candidate']=str(world/'scene.json')
            write(out/'status.json',result)
    return result


def assess_instance_regions(results, *, max_p95_m, min_coverage):
    """Independent per-instance gate; background cannot hide thin-object errors."""
    failures=[]
    for oid,methods in results.items():
        for row in methods['measured']:
            if row['target_pixels'] < 30:
                failures.append({'object_id':oid,'view_id':row['view_id'],'reason':'insufficient validation pixels'})
            elif row['coverage'] < min_coverage or row['p95_error_m'] is None or row['p95_error_m'] > max_p95_m:
                failures.append({'object_id':oid,'view_id':row['view_id'],'reason':'coverage or depth residual exceeds configured target',
                    'coverage':row['coverage'],'p95_error_m':row['p95_error_m']})
    return {'status':'GEOMETRY_REPAIR_REQUIRED' if failures else 'OBSERVED_GEOMETRY_TARGET_MET',
        'limits':{'max_p95_m':max_p95_m,'min_coverage':min_coverage},'failures':failures,
        'scope':'visible instance regions in heldout RGBD only; not backside completeness, part semantics or real-device accuracy'}


def intrinsic_frame(mesh):
    """An explicitly geometric upright frame, not a recognized semantic pose."""
    center=mesh.bounds.mean(0);xy=mesh.triangles_center[:,:2]-center[:2]
    weights=mesh.area_faces;cov=(xy.T*weights)@xy/max(float(weights.sum()),1e-12)
    _,vectors=np.linalg.eigh(cov);x=vectors[:,-1]
    if x[np.argmax(abs(x))]<0:x=-x
    t=np.eye(4);t[:3,0]=[x[0],x[1],0];t[:3,1]=[-x[1],x[0],0];t[:3,3]=center
    return t


def run(config):
    cfg=json.loads(Path(config).read_text());out=Path(cfg['output']).resolve()
    existed=out.exists()
    try:return _run(config)
    except Exception as exc:
        if not existed and out.exists():
            write(out/'status.json',{'status':'FAILED','reason':str(exc),'evidence_preserved':True})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps(run(a.config)))
