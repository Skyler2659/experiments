"""Bind a measured, known-robot FK snapshot to the reconstructed mesh world.

Only the capture adapter loads the public robot model. This geometry-side
consumer never loads a task simulator, its object poses, or its native assets.
"""
import json,shutil
from pathlib import Path
import numpy as np
from .mesh_world import sha256,pose_matrix,matrix_pose
from .mesh_context import public_robot_mesh


def prepare(data,root,config=None):
    spec={**data.get('robot',{}),**(config or {})}
    if not spec.get('required',True):
        if spec.get('scope')!='environment_only':raise ValueError('Robot omission requires explicit environment_only scope')
        return None
    if not spec.get('state_uri') or not spec.get('snapshot_uri'):
        raise ValueError('ROBOT_INPUT_REQUIRED: measured arm/finger state and matching public-model FK snapshot')
    state_path=Path(root)/spec['state_uri'];snapshot=Path(root)/spec['snapshot_uri']
    state=json.loads(state_path.read_text());packet=np.load(snapshot,allow_pickle=False)
    inventory=json.loads(str(packet['inventory_json']))
    if inventory.get('source')!='public robosuite Panda only' or inventory.get('frame')!='robot_base' or inventory.get('units')!='m':
        raise ValueError('Require a declared public robot snapshot in robot_base metres')
    if inventory.get('state_sha256')!=sha256(state_path):raise ValueError('Robot snapshot/state mismatch')
    for key,length in [('qpos',7),('gripper_qpos',2)]:
        q=np.asarray(state.get(key));saved=np.asarray(inventory.get(key))
        if q.shape!=(length,) or saved.shape!=q.shape or not np.isfinite(q).all() or not np.array_equal(q,saved):
            raise ValueError('Robot joint state missing or differs from FK snapshot')
    states={v.get('robot_state_id',v['state_id']) for v in data.get('views',[]) if not v.get('robot')}
    if states and states!={state.get('state_id')}:raise ValueError('Robot state is not synchronized with captured scene')
    base=pose_matrix(state['robot_base_pose_W'])
    # W is the robot base in this scene contract. A nonidentity placement would
    # silently mix room/global coordinates with the reconstructed world.
    if not np.allclose(base,np.eye(4),atol=1e-8):raise ValueError('robot_base world requires identity robot_base_pose_W; transform capture into W first')
    per_view={}
    for view in data.get('views',[]):
        if view.get('robot'):
            single={**view,'state_id':view.get('robot_state_id',view['state_id'])};single.pop('robot')
            per_view[view['view_id']]=prepare({'robot':view['robot'],'views':[single]},root)
    if per_view and len(per_view)!=len(data.get('views',[])):
        raise ValueError('Moving-camera capture requires per-view robot bindings for every frame')
    return {'state':state,'state_path':state_path,'snapshot':snapshot,'inventory':inventory,'per_view':per_view}


def attach(scene,world_root,binding):
    if binding is None:
        scene['robot_integration']={'status':'ENVIRONMENT_ONLY_EXPLICIT'};return
    world_root=Path(world_root);(world_root/'robot').mkdir(parents=True,exist_ok=True)
    shutil.copy2(binding['state_path'],world_root/'robot/state.json')
    shutil.copy2(binding['snapshot'],world_root/'robot/fk_snapshot.npz')
    mesh,provenance=public_robot_mesh(binding['snapshot'])
    mesh_path=world_root/'robot/known_robot.glb';mesh.export(mesh_path)
    node={'object_id':'known_robot','parent_id':None,'pose_parent_object':binding['state']['robot_base_pose_W'],
        'state':'OBSERVED','role':'known_robot','asset':{'mesh_ref':'robot/known_robot.glb','sha256':sha256(mesh_path),
            'up_axis':'Z','T_object_mesh':np.eye(4).tolist(),'provenance':provenance},
        'robot_binding':{'state_ref':'robot/state.json','snapshot_ref':'robot/fk_snapshot.npz',
            'update_policy':'recompute public-model FK and replace this node on every new measured state; never edit qpos alone'}}
    scene['nodes']=[n for n in scene['nodes'] if n['object_id']!='known_robot']+[node]
    scene['robot_state']=binding['state']
    scene['robot_integration']={'status':'BOUND_TO_MEASURED_FK','state_id':binding['state']['state_id'],
        'source':'known public robot model; not reconstructed task-object geometry',
        'query_and_render_authority':'known_robot node in the same MeshCodeWorld',
        'state_ref':'robot/state.json','snapshot_ref':'robot/fk_snapshot.npz'}


def exclude_robot_pixels(data,root,output,binding,tolerance_m=.003):
    """Known-model depth agreement prevents a second scanned robot in TSDF."""
    if binding is None:return
    import open3d as o
    from .quality_reconstruction import points
    cache={}
    def robot_scene(b):
        key=str(b['snapshot'])
        if key not in cache:
            mesh,_=public_robot_mesh(b['snapshot'])
            tri=o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))
            result=o.t.geometry.RaycastingScene();result.add_triangles(o.t.geometry.TriangleMesh.from_legacy(tri));cache[key]=result
        return cache[key]
    folder=Path(output)/'robot_masks';folder.mkdir(parents=True,exist_ok=True)
    for v in data['views']:
        if not v.get('surface_enabled',True):continue
        scene=robot_scene(binding.get('per_view',{}).get(v['view_id'],binding))
        depth=np.load(Path(root)/v['depth_uri']);t=pose_matrix(v['T_W_C']);k=np.asarray(v['K'])
        dirs=points(np.ones_like(depth),k,t)-t[:3,3]
        rays=np.concatenate([np.broadcast_to(t[:3,3],dirs.shape),dirs],axis=-1).astype('float32')
        hit=scene.cast_rays(o.core.Tensor(rays),nthreads=8)['t_hit'].numpy()
        mask=np.isfinite(hit)&np.isfinite(depth)&(abs(hit-depth)<=tolerance_m)
        path=folder/f'{v["view_id"]}.npy';np.save(path,mask);v['robot_mask_uri']=str(path.resolve())
    return {'policy':'exclude only measured depth agreeing with known robot surface','tolerance_m':tolerance_m}


def augment(scene_path,capture_path,state_path,snapshot_path,output):
    """Add the robot to an existing version without regenerating task assets."""
    from .mesh_world import MeshCodeWorld,generated_scene_code
    scene_path=Path(scene_path);cap=Path(capture_path);out=Path(output)
    if out.exists():raise ValueError('Preserve old world; choose a new output directory')
    data=json.loads(cap.read_text())
    binding=prepare(data,cap.parent,{'state_uri':str(Path(state_path).resolve()),'snapshot_uri':str(Path(snapshot_path).resolve())})
    shutil.copytree(scene_path.parent,out)
    scene=json.loads(scene_path.read_text());scene['world_version']=scene.get('world_version',1)+1
    original={n['object_id']:n['asset']['sha256'] for n in scene['nodes']}
    attach(scene,out,binding)
    scene['robot_integration']['legacy_scanned_context']='preserved; may include partial robot surface. New captures exclude known robot before fusion.'
    (out/'scene.json').write_text(json.dumps(scene,indent=2));(out/'scene.py').write_text(generated_scene_code())
    world=MeshCodeWorld.load(out/'scene.json');world.export_glb(out/'scene.glb')
    if scene.get('canonical_camera'):world.render(scene['canonical_camera'],out/'with_robot.png')
    receipt={'status':'ROBOT_ASSEMBLED','environment_asset_hashes_unchanged':all(world.node(k)['asset']['sha256']==v for k,v in original.items()),
        'robot_bounds_W_m':world.bounds('known_robot') if hasattr(world,'bounds') else world.mesh('known_robot').bounds.tolist(),
        'state_id':binding['state']['state_id'],'scope':'known robot assembly; no new environment reconstruction or quality acceptance'}
    (out/'robot_integration.json').write_text(json.dumps(receipt,indent=2));return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--capture',required=True);p.add_argument('--state',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(augment(a.scene,a.capture,a.state,a.snapshot,a.output),indent=2))
