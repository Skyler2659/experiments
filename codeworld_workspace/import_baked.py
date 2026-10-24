"""Import an existing world-baked mesh export without rerunning reconstruction.

This preserves legacy export provenance and does not assert source-code binding.
"""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
import trimesh
from .world import atomic

def build(export,output):
    export=Path(export).resolve();out=Path(output).resolve()
    inventory=json.loads((export/'object_inventory.json').read_text())
    if any(r.get('coordinates')!='world-baked Z-up meters' for r in inventory):raise ValueError('Expected explicitly world-baked meter export')
    out.mkdir()
    for d in ['geometry','objects','state','task','work','source','observations','scenarios','tools']:(out/d).mkdir()
    objects={};states={};bound={}
    for row in inventory:
        oid=row['object_id']
        if Path(oid).name!=oid:raise ValueError('Unsafe object ID')
        src=export/row['mesh_ref'];mesh=trimesh.load(src,force='mesh')
        if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():raise ValueError('Invalid mesh '+oid)
        rel='geometry/'+oid+'.glb';shutil.copy2(src,out/rel);bound[rel]=hashlib.sha256((out/rel).read_bytes()).hexdigest()
        meta='objects/'+oid+'.json';objects[oid]=meta
        atomic(out/meta,{'object_id':oid,'role':'robot_snapshot' if 'robot' in oid else 'rigid_object','geometry':{'local_mesh':rel},'source_export':str(src)})
        states[oid]={'T_W_object':np.eye(4).tolist(),'provenance':'legacy_world_baked_reconstruction'}
    revision='legacy-'+hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()[:16]
    atomic(out/'world.json',{'schema':'agent-codeworld-v1','frame':'robot-base W; Z up','units':'m','matrix_convention':'row-major T_W_object; column vectors','objects':objects,'bound_files':bound,'geometry_revision':revision,'source':str(export),'source_program_binding_verified':False})
    atomic(out/'state/current.json',{'revision':0,'geometry_revision':revision,'objects':states,'history':[]})
    shutil.copy2(export/'object_inventory.json',out/'source/object_inventory.json')
    return {'workspace':str(out),'objects':len(objects),'source_program_binding_verified':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',required=True);p.add_argument('--output',required=True)
    print(json.dumps(build(**vars(p.parse_args()))))
