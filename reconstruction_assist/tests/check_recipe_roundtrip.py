"""Compare Three.js helper output to the numerical recipes used by the fitter."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import cKDTree
from reconstruction_assist.fit_tools import execute
from agentic_mesh_world.static_asset_geometry import make_recipe
root=Path(sys.argv[1]);ev=Path(sys.argv[2]);src=Path(__file__).resolve().parents[1];run=root/'runs/recipe_roundtrip_fixture';nd=run/'fractal/scene';nd.mkdir(parents=True,exist_ok=False);(run/'input/measurement').mkdir(parents=True)
shutil.copy2(src/'recipe_geometry.mjs',run/'input/measurement/recipe_geometry.mjs')
r=execute([{'id':'bowl','evidence_id':'akita_black_bowl_1','family':'bowl'},{'id':'shelf','evidence_id':'rack_upper_shelf','family':'tilted_slab'},{'id':'box','evidence_id':'blue_package_1','family':'upright_box'}],ev)
assert all(x['status']=='CANDIDATE' for x in r['results']);(nd/'recipes.json').write_text(json.dumps(r))
(nd/'component.js').write_text("import * as THREE from 'three';import {recipeGeometry} from '../../input/measurement/recipe_geometry.mjs';const recipes=await fetch(new URL('./recipes.json',import.meta.url)).then(r=>r.json());export function build(){const root=new THREE.Group();for(const r of recipes.results){const g=recipeGeometry(r.recipe);g.userData.object_id=r.id;root.add(g);}return root;}")
cap=json.loads((ev/'config.json').read_text())['capture'];out=root/'recipe_roundtrip_output';env=os.environ.copy();env['PLAYWRIGHT_BROWSERS_PATH']='/root/.cache/rcwmworld-browsers';env['LD_LIBRARY_PATH']=str(root/'.system-libs/usr/lib/x86_64-linux-gnu')
subprocess.run([str(root/'.render-tools/node/bin/node'),str(src/'render_node.mjs'),str(root),run.name,'scene',cap,str(out)],env=env,check=True)
report=[]
for r in r['results']:
 expected=make_recipe(r['recipe']);actual=trimesh.load(out/(r['id']+'.glb'),force='scene',process=False).to_geometry();error=max(cKDTree(expected.vertices).query(actual.vertices)[0].max(),cKDTree(actual.vertices).query(expected.vertices)[0].max());assert error<2e-6,(r['id'],error);report.append({'id':r['id'],'max_surface_vertex_error_m':error})
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
