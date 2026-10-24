"""Import an existing object-mesh reconstruction as data, without its old runtime."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from .world import atomic

def import_scene(source,output):
    source=Path(source).resolve();out=Path(output).resolve();doc=json.loads((source/'scene.json').read_text())
    out.mkdir()
    for d in ['source','geometry','objects','state','scenarios','observations','work','task','tools']:(out/d).mkdir()
    for name in ['scene.json','scene.py','config.json']:
        if (source/name).exists():shutil.copy2(source/name,out/'source'/name)
    nodes={x['object_id']:x for x in doc['nodes']};transforms={}
    def matrix(oid):
        if oid in transforms:return transforms[oid]
        n=nodes[oid];p=n['pose_parent_object'];t=np.eye(4);t[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();t[:3,3]=p['translation_m']
        if n.get('parent_id'):t=matrix(n['parent_id'])@t
        transforms[oid]=t;return t
    objects={};state={};checks=[]
    for oid,n in nodes.items():
        t=matrix(oid);a=n['asset'];asset_file=source/a['mesh_ref']
        if a.get('sha256') and hashlib.sha256(asset_file.read_bytes()).hexdigest()!=a['sha256']:
            raise ValueError('Reconstruction asset changed: '+oid)
        # Keep a scene so multiple textured surfaces/materials survive import.
        asset=trimesh.load(asset_file,force='scene',process=False)
        mesh_transform=np.asarray(a.get('T_object_mesh',np.eye(4)))
        if a.get('up_axis')=='Y':
            y_to_z=np.eye(4);y_to_z[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
            mesh_transform=mesh_transform@y_to_z
        asset.apply_transform(mesh_transform)
        before=asset.to_geometry()
        dest=out/'geometry'/f'{oid}.glb';asset.export(dest)
        loaded=trimesh.load(dest,force='scene',process=False)
        m=loaded.to_geometry()
        error=float(max(cKDTree(before.vertices).query(m.vertices)[0].max(),
                        cKDTree(m.vertices).query(before.vertices)[0].max()))
        if len(before.faces)!=len(m.faces) or error>2e-6:
            raise ValueError('Imported geometry changed: '+oid)
        source_nodes=[]
        for name in asset.graph.nodes_geometry:
            local,gid=asset.graph.get(name)
            source_nodes.append({'path':str(name),'name':str(name),'type':'Mesh',
                'object_id':oid,'matrix_local_column_major':local.flatten(order='F').tolist(),
                'matrix_world_column_major':(t@local).flatten(order='F').tolist(),
                'vertices':len(asset.geometry[gid].vertices),'user_data':{}})
        declaration=dict(n)
        recipe=source/Path(a['mesh_ref']).parent/'geometry_recipe.json'
        if recipe.exists():
            snapshot=out/'source/reconstruction'/oid;snapshot.mkdir(parents=True)
            for f in recipe.parent.iterdir():
                if f.is_file() and (f.suffix in {'.json','.npz'} or f.name=='model.py'):
                    shutil.copy2(f,snapshot/f.name)
            declaration['reconstruction_recipe']=json.loads(recipe.read_text())
            declaration['reconstruction_recipe_ref']=str((snapshot/'geometry_recipe.json').relative_to(out))
        row={'object_id':oid,'movable':n.get('movable',oid not in ['known_robot','measured_tabletop','tabletop','environment']),'role':'robot_snapshot' if oid=='known_robot' else 'rigid_object',
            'geometry':{'local_mesh':str(dest.relative_to(out)),'local_bounds_m':m.bounds.tolist(),'vertices':len(m.vertices),'faces':len(m.faces)},'initial_T_W_object':t.tolist(),
            'source_nodes':source_nodes,'source_parameters':'source/scene.json','source_binding':{'node_id':oid,'asset':a},'declared':declaration}
        atomic(out/'objects'/f'{oid}.json',row);objects[oid]=f'objects/{oid}.json';state[oid]={'T_W_object':t.tolist(),'basis':'imported_reconstruction','evidence':'source/scene.json'}
        checks.append({'object_id':oid,'max_local_vertex_error_m':error,
                       'source_meshes':len(asset.geometry),'imported_meshes':len(loaded.geometry)})
    bound={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['source','geometry','objects'] for p in (out/d).rglob('*') if p.is_file()}
    revision=hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()
    atomic(out/'world.json',{'schema':'agent-codeworld-v1','geometry_revision':revision,'frame':doc['frame'],'units':'m','objects':objects,'bound_files':bound,
        'matrix_convention':'row-major T_W_object; column vectors',
        'source_kind':'reconstructed object meshes; independent of reconstruction backend','source_directory':str(source),
        'robot_execution_connected':False,'geometry_accuracy_verified':False})
    atomic(out/'state/current.json',{'geometry_revision':revision,'revision':0,'branch':'current','objects':state,'history':[]})
    atomic(out/'task/task.json',{'goal':None})
    atomic(out/'build_report.json',{'status':'PASS','objects':len(objects),'geometry_roundtrip':checks,
        'scope':'reconstruction-to-CodeWorld geometry preservation; not reconstruction accuracy or robot execution'})
    for name in ['AGENTS.md','README.md']:shutil.copy2(Path(__file__).parent/'templates'/name,out/name)
    shutil.copytree(Path(__file__).parent,out/'tools/codeworld_workspace',ignore=shutil.ignore_patterns('__pycache__','tests'))
    result={'workspace':str(out),'objects':len(objects),'source_kind':'reconstruction assets only'}
    print(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();import_scene(a.source,a.output)
