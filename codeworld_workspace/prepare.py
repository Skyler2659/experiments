"""One-command preparation from the selected EXISTING reconstruction; never calls a model."""
import argparse,json,os,subprocess
from pathlib import Path
from .build import build
from .import_scene import import_scene
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/codeworld_workspace.json');p.add_argument('--reuse-export',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];cfg=json.loads((root/a.config).read_text())
v=cfg['validation']
if v['mode']!='reuse_existing_reconstruction' or v['run_model_inference'] is not False:
    raise ValueError('This entry point only compiles existing sources; no model inference is supported')
backend=cfg['reconstruction'].get('backend','rcwm')
if backend=='legacy_measured':
    source=root/cfg['reconstruction']['source_scene']
    import_scene(source.parent,root/v['workspace'])
    raise SystemExit(0)
if backend!='rcwm':raise ValueError('Unknown reconstruction backend: '+backend)
runtime=root/cfg['reconstruction']['runtime'];run=runtime/'runs'/v['source_run'];export=root/'static_runs'/v['source_run']/'codeworld_export'
if not a.reuse_export:
    env=os.environ.copy();env['RCWM_EXPORT_OBJECTS']='1'
    subprocess.run([str(runtime/'.render-tools/node/bin/node'),str(root/'scripts/export_codeworld.mjs'),str(runtime),v['source_run'],str(export)],env=env,check=True)
print(json.dumps(build(run,export,root/v['workspace'],runtime),indent=2))
