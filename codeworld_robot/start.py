"""Start a fresh robot episode from an existing CodeWorld, without reconstruction inference."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from .io import atomic,read
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--episode',required=True);p.add_argument('--config',default='configs/codeworld_robot_milk_basket.json');p.add_argument('--manual-diagnostic',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=Path(a.workspace).resolve();out=Path(a.episode).resolve();config=(root/a.config).resolve();cfg=read(config)
backend=cfg.get('backend','libero')
if backend not in ('libero','robodojo','fetchbench'):raise ValueError('Unknown backend '+backend)
# The optional publication report is not part of the bound CodeWorld. Some NFS
# exports retain its stale directory entry after the atomic rename.
shutil.copytree(source,out,ignore=shutil.ignore_patterns('build_report.json'))
# Completion contracts belong to this fresh episode, never to reused assets.
for name in ('completion.json','completion_started.json','completion_closure.json','world_decision_uses.json','executor.claim'):
    (out/'task'/name).unlink(missing_ok=True)
from .fresh_state import reset_episode_evidence
reset_episode_evidence(out)
atomic(out/'task/world_flow.json',{'required':not a.manual_diagnostic,'version':1,'observation_to_world_to_decision':not a.manual_diagnostic,'mode':'manual_diagnostic' if a.manual_diagnostic else 'independent_strict'})
world=read(out/'world.json')
missing=[rel for rel in world.get('bound_files',{}) if not (out/rel).is_file()]
if missing:raise RuntimeError('Incomplete bound reconstruction copy: '+str(missing[:8]))
if not a.manual_diagnostic:
    atomic(out/'task/executor.json', {'status':'UNASSIGNED','role':'codex_cli'})
elif (out/'task/executor.json').exists():
    (out/'task/executor.json').unlink()
if backend=='robodojo':
    cfg['constraints']=list(cfg.get('constraints',[]))+['After completing object operations, return BOTH arms and grippers to the measured episode-initial configuration recorded in task/robot_home.json. Compute return motions from CodeWorld and verify the automatically required __robot_return_home goal before successful finish.']
atomic(out/'task/task.json',{'goal':cfg['goal'],'status':'ACTIVE','constraints':cfg.get('constraints',['rigid objects']),
    'success_criteria':cfg.get('success_criteria',[]),'robot_interface':'codeworld_robot.Robot','robot_backend':backend})
shutil.copytree(Path(__file__).parent,out/'tools/codeworld_robot',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','tests'))
from .backend_contract import execution_instructions, CONTRACTS
(out/'task/EXECUTION.md').write_text(execution_instructions(backend,(Path(__file__).parent/'README.md').read_text()))
(out/'task/ROBOT_BACKEND.md').write_text(CONTRACTS[backend])
shutil.copy2(Path(__file__).parent/'WORLD_FLOW.md',out/'task/WORLD_FLOW.md')
shutil.copytree(root/'codeworld_workspace',out/'tools/codeworld_workspace',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','tests'))
shutil.copy2(root/'codeworld_workspace/templates/AGENTS.md',out/'AGENTS.md')
with (out/'AGENTS.md').open('a') as f:
    f.write('\n## Robot episode\nRead task/EXECUTION.md and task/WORLD_FLOW.md. This fresh episode has no inherited robot measurements or decision receipts. Reconstructed geometry remains a CodeWorld prior available for planning; inspect its uncertainty and update relevant objects from RGB-D when the task requires current evidence. For independent execution use World.compute_decision, Robot.execute_decision, World.integrate_rgbd and World.derive_state. Successful finish requires a current verification receipt supported by world evidence. Physical failure uses a world-computed reason; budget, timeout or tool failure may close with a typed system_reason.\n')
env=os.environ.copy();env.update(PYTHONPATH=str(out/'tools')+':'+cfg.get('libero_root',''),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
if backend=='libero':env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
if cfg.get('libero_config_path'):env['LIBERO_CONFIG_PATH']=cfg['libero_config_path']
if cfg.get('native_library_path'):env['LD_LIBRARY_PATH']=cfg['native_library_path']+':'+env.get('LD_LIBRARY_PATH','')
if cfg.get('native_vulkan_icd'):env['VK_ICD_FILENAMES']=cfg['native_vulkan_icd']
python=cfg.get('python','/private/lc/research/vla-failure-factor/deps/openpi/.venv/bin/python' if backend=='libero' else ('/private/lc/research/RCWMWorld/fetchbench_env/bin/python' if backend=='fetchbench' else '/private/lc/research/RoboDojo/setup_logs/run_python.sh'))
module={'libero':'codeworld_robot.server','robodojo':'codeworld_robot.robodojo_server','fetchbench':'codeworld_robot.fetchbench_server'}[backend]
log=open(out/'server.log','w');proc=subprocess.Popen([python,'-m',module,'--workspace',str(out),'--config',str(config)],cwd=out,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
atomic(out/'launch.json',{'pid':proc.pid,'python':python,'source_workspace':str(source),'reconstruction_reused':True,'execution_mode':'manual_diagnostic' if a.manual_diagnostic else 'codex_cli','configuration':cfg})
print(json.dumps({'pid':proc.pid,'episode':str(out),'status':'STARTING'}))
