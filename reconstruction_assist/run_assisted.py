#!/usr/bin/env python3
"""Prepare measured evidence, run RCWM, and publish its actual program on Linux."""
import argparse,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=none','-o','ProxyCommand=ssh -S /tmp/rcwm-drawer-final -W %h:%p sz.2.gpu','l203']
RSYNC_SHELL='ssh -o ControlPath=none -o "ProxyCommand=ssh -S /tmp/rcwm-drawer-final -W %h:%p sz.2.gpu"'
def call(args,**kw):return subprocess.run([str(x) for x in args],check=True,**kw)
def remote(args):return call(SSH+[shlex.join([str(x) for x in args])])
def push(src,dst):return call(['rsync','-a','--exclude=.adaptive_best','--exclude=.best','-e',RSYNC_SHELL,src,'l203:'+dst])
def pull(src,dst):return call(['rsync','-a','-e',RSYNC_SHELL,'l203:'+src,dst])

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--runtime',required=True);p.add_argument('--publish-only',action='store_true');p.add_argument('--reuse-evidence',action='store_true');a=p.parse_args()
 config=Path(a.config).resolve();cfg=json.loads(config.read_text());transport=json.loads((ROOT/cfg['assistance']['transport_config']).read_text());runtime=Path(a.runtime).resolve();name=cfg['scene'];work=transport['remote_work'];rr=transport['remote_runtime'];source=transport['remote_source'];remote_run=rr+'/runs/'+name;run=runtime/'runs'/name;output=ROOT/'outputs'/name;output.mkdir(exist_ok=True)
 py='/private/lc/conda_envs/tabletopgen/bin/python';environment=['env','PYTHONPATH='+source+':/private/lc/research/agentic_mesh_world_deps/geometry','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','PYOPENGL_PLATFORM=egl']
 remote(['mkdir','-p',remote_run,source])
 setup="from pathlib import Path; r=Path("+repr(rr)+"); base=Path('/private/lc/research/RCWMWorld/runtime'); [(r/x).symlink_to(base/x) for x in ['.render-tools','.system-libs'] if not (r/x).exists()]"
 remote([py,'-c',setup]);push(str(ROOT/'reconstruction_assist'),source+'/');push(str(ROOT/'codeworld_workspace'),source+'/')
 for script in ['export_codeworld.mjs','codeworld_source_files.mjs','render_codeworld.mjs']:push(str(ROOT/'scripts'/script),source+'/'+script)
 if not a.publish_only:
  if run.exists():raise ValueError('Use a fresh run name; source programs are never overwritten')
  reference=(ROOT/cfg['reference']).parent
  if not a.reuse_evidence:
   evidence_cfg=ROOT/cfg['assistance']['evidence_config'];push(str(evidence_cfg),work+'/evidence_config.json');remote(environment+[py,'-m','reconstruction_assist.evidence',work+'/evidence_config.json'])
  pull(work+'/evidence/model/',str(reference/'measurement')+'/');shutil.copy2(ROOT/'reconstruction_assist/recipe_geometry.mjs',reference/'measurement/recipe_geometry.mjs')
  push(str(reference)+'/',remote_run+'/input/')
  env=os.environ.copy();env['RCWM_ROOT']=str(runtime)
  call([sys.executable,ROOT/'scripts/rcwm_world.py','run','--config',config],env=env)
 if not (run/'fractal/scene/part.json').exists():raise ValueError('Root program not delivered')
 push(str(run)+'/',remote_run+'/')
 node=rr+'/.render-tools/node/bin/node';renderenv=['env','PLAYWRIGHT_BROWSERS_PATH=/root/.cache/rcwmworld-browsers','LD_LIBRARY_PATH='+rr+'/.system-libs/usr/lib/x86_64-linux-gnu']
 exportenv=['RCWM_EXPORT_OBJECTS=1']
 if cfg['assistance'].get('canonical_export'):exportenv.append('RCWM_EXPORT_CANONICAL_VIEW=1')
 remote(renderenv+exportenv+[node,source+'/export_codeworld.mjs',rr,name,work+'/export'])
 remote(environment+[py,'-m','codeworld_workspace.build','--run',remote_run,'--export',work+'/export','--output',work+'/workspace','--runtime',rr,'--goal','Static reconstruction comparison only; no robot actions'])
 check=[py,'-m','codeworld_workspace.check_workspace',work+'/workspace']
 if cfg['assistance'].get('integration_check_object'):check+=['--object',cfg['assistance']['integration_check_object']]
 remote(environment+check)
 capture=cfg['assistance']['evaluation_capture']
 remote(renderenv+[node,source+'/render_codeworld.mjs',rr,work+'/workspace',capture,work+'/review'])
 for sub in ['export','review']:pull(work+'/'+sub+'/',str(output/sub)+'/')
 pull(work+'/workspace/',str(ROOT/'workspaces'/name)+'/')
 receipt={'status':'PUBLISHED_RECONSTRUCTION_REVIEW_REQUIRED','model':cfg['model'],'run':str(run),'remote_work':work,'workspace':str(ROOT/'workspaces'/name),'robot_actions':0,'metric_acceptance':False}
 (output/'publication.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
