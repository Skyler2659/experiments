"""Bounded whole-object completion: measured surfaces + explicit shape prior.

FIT depth carves witnessed free space, including container/handle openings.
Unknown cells retain the generated prior rather than being declared empty.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,gaussian_filter
from scipy.spatial import cKDTree
from .mesh_world import MeshCodeWorld,pose_matrix,sha256
from .quality_reconstruction import points,write
from .surface_quality import topology


def field_mesh(field,origin,pitch):
    import trimesh
    from skimage.measure import marching_cubes
    vertices,faces,_,_=marching_cubes(field,level=-pitch*1e-4,spacing=(pitch,pitch,pitch),allow_degenerate=False)
    mesh=trimesh.Trimesh(vertices+origin,faces,process=False);mesh.merge_vertices(digits_vertex=12)
    mesh.remove_unreferenced_vertices()
    if mesh.volume<0:mesh.invert()
    return mesh


def simplify_mesh(mesh,max_faces):
    import open3d as o,trimesh
    if len(mesh.faces)<=max_faces:return mesh
    p=o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))
    p=p.simplify_quadric_decimation(max_faces,maximum_error=2.5e-7,boundary_weight=10.)
    candidate=trimesh.Trimesh(np.asarray(p.vertices),np.asarray(p.triangles),process=False)
    if candidate.is_watertight and len(candidate.split())==1:return candidate
    return mesh


def run(config):
    import open3d as o,trimesh
    cfg=json.loads(Path(config).read_text());w=MeshCodeWorld.load(cfg['source_scene']);oid=cfg['object_id'];cap=Path(cfg['capture']);data=json.loads(cap.read_text());ann=Path(cfg['annotations']);packet=json.loads(ann.read_text());byid={v['view_id']:v for v in data['views']}
    e=np.load(w.root/'evidence'/f'{oid}_fit_points.npz');measured=e['xyz_W_m'];colors=e['rgb'];guide=MeshCodeWorld.load(cfg['guide_scene']).mesh(oid) if cfg.get('guide_scene') else w.mesh(oid)
    normals=[];samples=[];bound=np.quantile(measured,[.001,.999],axis=0)+[[-.003]*3,[.003]*3]
    for row in packet['views']:
        if oid not in row['masks']:continue
        v=byid[row['view_id']]
        if v['role']!='fit':raise ValueError('Bounded surface may use FIT only')
        d=np.load(cap.parent/v['depth_uri']);p=points(d,np.asarray(v['K']),pose_matrix(v['T_W_C']));ref=row['masks'][oid];maskfile=ann.parent/ref['uri']
        if sha256(maskfile)!=ref['sha256'] or sha256(cap.parent/v['depth_uri'])!=v['depth_sha256']:raise ValueError('Evidence changed')
        mask=binary_erosion(np.asarray(Image.open(maskfile).convert('L'))>127,iterations=1)
        dx=np.roll(p,-1,1)-np.roll(p,1,1);dy=np.roll(p,-1,0)-np.roll(p,1,0);n=np.cross(dx,dy);length=np.linalg.norm(n,axis=2);n/=np.maximum(length[:,:,None],1e-12)
        towards=np.asarray(v['T_W_C']['translation_m'])-p;n[(n*towards).sum(2)<0]*=-1
        valid=mask&np.isfinite(p).all(2)&(length>1e-12)&np.all(p>=bound[0],2)&np.all(p<=bound[1],2)&(np.linalg.norm(dx,axis=2)<.012)&(np.linalg.norm(dy,axis=2)<.012)
        iy,ix=np.nonzero(valid);samples.append(p[iy,ix]);normals.append(n[iy,ix])
    xyz=np.concatenate(samples);normal=np.concatenate(normals);tree=cKDTree(xyz)
    pitch=float(cfg.get('pitch_m',.001));lower=np.minimum(guide.bounds[0],bound[0])-4*pitch;upper=np.maximum(guide.bounds[1],bound[1])+4*pitch
    shape=np.ceil((upper-lower)/pitch).astype(int)+1;grid=np.indices(shape,dtype=np.float32).reshape(3,-1).T*pitch+lower;field=np.empty(len(grid),np.float32)
    ray=o.t.geometry.RaycastingScene();ray.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(guide.vertices),o.utility.Vector3iVector(guide.faces))))
    for start in range(0,len(grid),60000):
        p=grid[start:start+60000];base=ray.compute_signed_distance(o.core.Tensor(p.astype('float32')),nthreads=2).numpy()
        distance,near=tree.query(p);signed=((p-xyz[near])*normal[near]).sum(1)
        shell=np.maximum.reduce([signed,-signed-.004,distance-.0045]);field[start:start+len(p)]=np.minimum(base,shell)
    # Depth agreement has a finite tolerance; do not carve surfaces from tiny residuals.
    free=np.zeros(len(grid),np.float32);votes=np.zeros(len(grid),np.uint16)
    owned_views={row['view_id'] for row in packet['views'] if oid in row['masks']}
    for v in data['views']:
        if cfg.get('free_space_mode','depth_votes')=='disabled':break
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        if cfg.get('free_space_scope','masked_fit_views')=='masked_fit_views' and v['view_id'] not in owned_views:continue
        depth=np.load(cap.parent/v['depth_uri']);t=pose_matrix(v['T_W_C']);h,width=depth.shape
        for start in range(0,len(grid),100000):
            p=(grid[start:start+100000]-t[:3,3])@t[:3,:3];uv=p@np.asarray(v['K']).T;uv=uv[:,:2]/np.maximum(p[:,2,None],1e-12)
            inside=(p[:,2]>0)&(uv[:,0]>=1)&(uv[:,0]<width-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
            x=np.clip(np.rint(uv[:,0]),0,width-1).astype(int);y=np.clip(np.rint(uv[:,1]),0,h-1).astype(int);d=depth[y,x]
            margin=np.maximum(d-p[:,2]-.0025,0);margin[~(inside&np.isfinite(d)&(d>0))]=0
            free[start:start+len(p)]=np.maximum(free[start:start+len(p)],margin.astype('float32'))
            votes[start:start+len(p)]+=(margin>0).astype('uint16')
    distance,_=tree.query(grid);supported=distance<float(cfg.get('protect_observed_band_m',.003))
    carved=(votes>=int(cfg.get('free_space_min_views',2)))&~supported
    field[carved]=np.maximum(field[carved],free[carved]);field=field.reshape(shape)
    field=gaussian_filter(field,.45)
    for axis in range(3):
        sl=[slice(None)]*3;sl[axis]=0;field[tuple(sl)]=pitch;sl[axis]=-1;field[tuple(sl)]=pitch
    mesh=field_mesh(field,lower,pitch);components=mesh.split(only_watertight=False);mesh=max(components,key=lambda m:m.area)
    mesh=simplify_mesh(mesh,int(cfg.get('max_faces',120000)));top=topology(mesh)[0]
    if not top['watertight']:
        diagnostics=Path(cfg['output']+'_rejected');diagnostics.mkdir(exist_ok=False);np.savez_compressed(diagnostics/'implicit_field.npz',field=field,origin_W_m=lower,pitch_m=pitch);mesh.export(diagnostics/'candidate.ply');write(diagnostics/'topology.json',top)
        raise ValueError('Bounded completion is not closed')
    _,near=cKDTree(measured).query(mesh.vertices);mesh.visual.vertex_colors=colors[near]
    check=o.t.geometry.RaycastingScene();check.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))));errors=check.compute_distance(o.core.Tensor(measured.astype('float32')),nthreads=2).numpy()
    out=Path(cfg['output']);shutil.copytree(w.root,out);scene=copy.deepcopy(w.source);node=next(n for n in scene['nodes'] if n['object_id']==oid);directory=out/'objects'/oid
    prior_root=Path(cfg['guide_scene']).parent if cfg.get('guide_scene') else w.root
    shutil.copy2(prior_root/'objects'/oid/'geometry_recipe.json',directory/'prior_recipe.json');np.savez_compressed(directory/'implicit_field.npz',field=field,origin_W_m=lower,pitch_m=pitch)
    assumptions=['whole-object measured oriented surface shell with explicit continuous shape prior','unobserved material retains fitted parametric prior; thickness and hidden continuity are hypotheses',f'free-space mode: {cfg.get("free_space_mode","depth_votes")}; disabled carving preserves prior openings and walls',f'bounded grid pitch {pitch} m; sub-grid detail approximated; largest material component retained']
    write(directory/'geometry_recipe.json',{'family':'bounded_surface','frame':'robot_base','units':'m','config':cfg,'observed_normal_samples':len(xyz),'grid_shape':shape.tolist(),'assumptions':assumptions})
    (directory/'model.py').write_text('import numpy as np,json\nfrom pathlib import Path\nfrom agentic_mesh_world.static_bounded_asset import field_mesh,simplify_mesh\ndef make_mesh():\n    root=Path(__file__).parent\n    d=np.load(root/"implicit_field.npz")\n    cfg=json.loads((root/"geometry_recipe.json").read_text())["config"]\n    m=field_mesh(d["field"],d["origin_W_m"],float(d["pitch_m"]))\n    m=max(m.split(only_watertight=False),key=lambda part:part.area)\n    return simplify_mesh(m,int(cfg.get("max_faces",120000)))\n')
    bounds=mesh.bounds.tolist();mesh.apply_transform(np.linalg.inv(w.object_transform(oid)));file=out/node['asset']['mesh_ref'];mesh.export(file);mesh.export(file.with_suffix('.obj'))
    node['asset']['sha256']=sha256(file);node['asset']['provenance'].update(source='bounded whole-body surface/prior fusion',assumptions=assumptions)
    scene['world_version']+=1;write(out/'scene.json',scene)
    report={'object_id':oid,'family':'bounded_surface','topology':top,'bounds_W_m':bounds,'fit_point_to_surface_median_m':float(np.median(errors)),'fit_point_to_surface_p95_m':float(np.quantile(errors,.95)),'asset_faces':len(mesh.faces),'prior_component_count':len(components),'witnessed_free_voxels':int(carved.sum()),'holdout_used':False,'independent_acceptance':False}
    allreport=json.loads((out/'report.json').read_text());next(x for x in allreport['objects'] if x['object_id']==oid).update(report);write(out/'report.json',allreport);write(out/'bounded_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);print(json.dumps(run(p.parse_args().config)))
