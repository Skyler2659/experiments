"""Integration check on a real compiled scene; moves only a hypothetical branch."""
import argparse,json,time
from pathlib import Path
import numpy as np
import trimesh
from .world import World,atomic

p=argparse.ArgumentParser();p.add_argument('workspace');p.add_argument('--object',default='akita_black_bowl_1');a=p.parse_args()
w=World(a.workspace);baseline=w.state();before={oid:w.mesh(oid).bounds for oid in w.objects()}
branch=w.fork('integration_'+str(time.time_ns()))
t=np.array(branch.object(a.object)['current']['T_W_object']);t[0,3]+=.1
branch.set_hypothesis({a.object:t.tolist()},expected_revision=baseline['revision'])
after=branch.mesh(a.object).bounds
np.testing.assert_allclose(after,before[a.object]+[.1,0,0],atol=2e-6)
assert w.state()==baseline
for oid in w.objects():
    if oid!=a.object:np.testing.assert_allclose(branch.mesh(oid).bounds,before[oid],atol=2e-6)
output=Path(a.workspace)/'work/hypothetical_bowl_move.glb'
branch.export_scene(output)
loaded=trimesh.load(output,force='scene',process=False)
expected=trimesh.Scene()
for oid in branch.objects(): expected.add_geometry(branch.mesh(oid))
np.testing.assert_allclose(loaded.bounds,expected.bounds,atol=2e-6)
report={'status':'PASS','objects':len(w.objects()),'branch':branch.branch,'current_state_unchanged':True,
        'changed_object':a.object,'hypothetical_translation_m':[.1,0,0],
        'other_object_bounds_unchanged':True,'updated_scene_export':str(output),
        'scope':'actual reconstructed asset compilation/query/update/export; no robot action'}
atomic(Path(a.workspace)/'integration_report.json',report)
print(json.dumps(report,indent=2))
