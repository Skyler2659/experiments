"""Durable server-only 20-scene scheduler. Restart never resets model budgets."""
import concurrent.futures,fcntl,json,os,re,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[2];BATCH=SOURCE.parent;BASE=Path('/private/lc/research/RCWMWorld/runtime');PY='/private/lc/conda_envs/tabletopgen/bin/python';SIM='/private/lc/research/vla-failure-factor/deps/openpi/.venv/bin/python'
def write(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def state(case,status,**kw):write(case/'status.json',{'scene':case.name,'status':status,'time':time.time(),'host':socket.gethostname(),'pid':os.getpid(),**kw})
def env():
 return {**os.environ,'RCWM_JDCLOUD_PROXY_KEY':'rcwm-local','PYTHONPATH':str(SOURCE)+':/private/lc/research/agentic_mesh_world_deps/geometry','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYOPENGL_PLATFORM':'egl','MUJOCO_GL':'egl','CUDA_VISIBLE_DEVICES':'0','PLAYWRIGHT_BROWSERS_PATH':'/root/.cache/rcwmworld-browsers','LD_LIBRARY_PATH':str(BASE/'.system-libs/usr/lib/x86_64-linux-gnu')}
def call(cmd,case,stage,e=None):
 state(case,stage,command=[str(x) for x in cmd]);
 with (case/(stage.lower()+'.log')).open('a') as f:subprocess.run([str(x) for x in cmd],env=e or env(),cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,check=True)
def manifest():
 root=Path('/private/lc/research/vla-failure-factor/deps/LIBERO-plus/libero/libero/bddl_files/libero_90');rows=[]
 for family,n in [('KITCHEN',10),('LIVING_ROOM',6),('STUDY',4)]:
  for i in range(1,n+1):
   files=sorted(root.glob(f'{family}_SCENE{i}_*.bddl'));assert files,(family,i);p=files[0];text=p.read_text();objects=[]
   for section in ['fixtures','objects']:
    block=re.search(r'\(:'+section+r'\s+(.*?)\)',text,re.S)
    if not block:continue
    for names,kind in re.findall(r'([\w\s]+?)\s+-\s+(\w+)',block[1]):
     for name in names.split():
      if kind.endswith('table') or name.endswith('table'):continue
      objects.append({'object_id':name,'description':kind.replace('_',' ')+'; instance '+name,'family':kind,'texture_edge_m':.004})
   assert objects,p
   rows.append({'id':f'{family.lower()}_scene{i}','task':p.stem,'suite':'libero_90','initial_state_id':0,'seed':832002,'objects':objects,'inventory_source':'official semantic object/fixture names and types only; no poses or region coordinates'})
 assert len(rows)==20;write(BATCH/'manifest.json',rows)
 for spec in rows:
  case=BATCH/'scenes'/spec['id'];case.mkdir(parents=True,exist_ok=True);write(case/'spec.json',spec)
  if not (case/'status.json').exists():state(case,'QUEUED')
 return rows
def worker(spec):
 case=BATCH/'scenes'/spec['id']
 try:
  with Path('/tmp/rcwm-libero20-gpu.lock').open('w') as lock:
   state(case,'WAITING_CAPTURE');fcntl.flock(lock,fcntl.LOCK_EX)
   if not (case/'capture/capture.json').exists():call([SIM,SOURCE/'scripts/libero20_server/capture.py',case],case,'CAPTURING')
  if not (case/'input/known_robot.glb').exists():call([PY,SOURCE/'scripts/libero20_server/prepare.py',case],case,'PREPARING_EVIDENCE')
  runtime=case/'runtime';runtime.mkdir(exist_ok=True)
  for name in ['.render-tools','.system-libs','.venv','.codex-cli']:
   if not (runtime/name).exists():(runtime/name).symlink_to(BASE/name)
  bindir=runtime/'bin';bindir.mkdir(exist_ok=True);wrapper=bindir/'codex';shutil.copy2(SOURCE/'scripts/libero20_server/codex_container.py',wrapper);wrapper.chmod(0o755)
  home=case/'codex_home';home.mkdir(exist_ok=True);shutil.copy2(SOURCE/'models.json',home/'models.json')
  (home/'config.toml').write_text('model = "gpt-5.6-sol"\nmodel_provider = "jdcloud_rcwm"\nmodel_reasoning_effort = "high"\nweb_search = "disabled"\nmodel_catalog_json = '+json.dumps(str(home/'models.json'))+'\n[model_providers.jdcloud_rcwm]\nname = "JDCloud server native Responses"\nbase_url = "http://127.0.0.1:8781/v1"\nwire_api = "responses"\nenv_key = "RCWM_JDCLOUD_PROXY_KEY"\nstream_idle_timeout_ms = 660000\nrequest_max_retries = 1\nstream_max_retries = 1\n[agents]\nenabled = false\n')
  transport={'transport':'local','remote_work':str(case),'remote_runtime':str(runtime),'remote_source':str(SOURCE),'max_reviews':4,'refinement':{'enabled':True,'strategy':'adaptive','patience':2,'min_relative_improvement':.03,'min_absolute_improvement':.005}};transport['agent_correction']={'enabled':True,'max_rounds':1,'extra_reviews':2,'extra_cycles':4,'timeout_seconds':660};write(case/'transport.json',transport)
  name='libero20_'+spec['id']+'_jdcloud_gpt56sol_20261019_v1'
  if (case/'infrastructure_restart.json').exists():name+='_infra_retry'+str(json.loads((case/'infrastructure_restart.json').read_text())['attempt'])
  inp=case/'input';cfg={'scene':name,'reference':str(inp/'reference.png'),'model':'gpt-5.6-sol','reasoning':'high','max_depth':4,'max_cycles':8,'solver_prompt':'prompts/libero20-adaptive.md','known_robot':{'asset':str(inp/'known_robot.glb'),'binding':str(inp/'robot_binding.json')},'assistance':{'transport_config':str(case/'transport.json'),'measurement':str(inp/'measurement/evidence.json')}};write(case/'config.json',cfg)
  e=env();e.update(RCWM_ROOT=str(runtime),CODEX_HOME=str(home),RCWM_CLEAN_CODEX_HOME='0',RCWM_JDCLOUD_PROXY_KEY='rcwm-local');run=runtime/'runs'/name
  if run.exists():raise RuntimeError('Existing run: refusing to reset cumulative budgets; explicit resume required')
  call([PY,SOURCE/'scripts/rcwm_world.py','run','--config',case/'config.json'],case,'RECONSTRUCTING',e)
  node=runtime/'.render-tools/node/bin/node';exportenv={**e,'RCWM_EXPORT_OBJECTS':'1','RCWM_EXPORT_CANONICAL_VIEW':'1'}
  call([node,SOURCE/'scripts/export_codeworld.mjs',runtime,name,case/'export'],case,'EXPORTING',exportenv)
  call([PY,'-m','codeworld_workspace.build','--run',run,'--export',case/'export','--output',case/'workspace','--runtime',runtime,'--goal','Static 20-scene reconstruction only; no robot actions'],case,'BUILDING_WORKSPACE',e)
  call([node,SOURCE/'scripts/render_codeworld.mjs',runtime,case/'workspace',case/'capture/capture.json',case/'review'],case,'RENDERING_REVIEW',e)
  events=[json.loads(x) for x in (run/'trace/events.jsonl').read_text().splitlines()];reviews=[]
  for p in (run/'fractal').glob('*/adaptive_state.json'):reviews.append({'node':p.parent.name,**json.loads(p.read_text())})
  write(case/'result.json',{'status':'DELIVERED_REVIEW_REQUIRED','model':'gpt-5.6-sol','actual_max_depth':max((r.get('depth',0) for r in events),default=0),'model_node_calls':sum(r.get('event')=='session_start' for r in events),'nodes':reviews,'robot_actions':0,'holdout_used_for_fitting':False,'run':str(run),'workspace':str(case/'workspace')});state(case,'DELIVERED_REVIEW_REQUIRED')
 except Exception as exc:
  (case/'error.txt').write_text(traceback.format_exc());state(case,'FAILED',error=str(exc));print(case.name,str(exc),flush=True)
def report():
 rows=[]
 for p in sorted((BATCH/'scenes').glob('*/status.json')):rows.append(json.loads(p.read_text()))
 write(BATCH/'status.json',{'host':socket.gethostname(),'scheduler_pid':os.getpid(),'time':time.time(),'scene_parallelism':4,'upstream_parallelism':8,'model':'gpt-5.6-sol','scenes':rows})
 html='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>LIBERO20 JDCloud server batch</title><style>body{font:16px system-ui;margin:40px;background:#fafafa}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}</style><h1>LIBERO 20 · JDCloud GPT-5.6 Sol</h1><p>全部计算在 l203；场景并发 4；接口并发 8。状态每 30 秒刷新。交付状态不等于质量通过。</p><table><tr><th>场景</th><th>状态</th><th>记录</th></tr>'
 import html as h
 for r in rows:html+='<tr><td>'+h.escape(r['scene'])+'</td><td>'+h.escape(r['status'])+'</td><td><a href="scenes/'+r['scene']+'/status.json">状态</a> · <a href="scenes/'+r['scene']+'/review/">重建对照</a> · <a href="scenes/'+r['scene']+'/result.json">实际递归记录</a>'+(' · '+h.escape(r.get('error','')) if r.get('error') else '')+'</td></tr>'
 (BATCH/'index.html').write_text(html+'</table>')
if __name__=='__main__':
 BATCH.mkdir(exist_ok=True);rows=manifest()
 if '--prepare-only' in sys.argv:report();sys.exit()
 write(BATCH/'scheduler.json',{'pid':os.getpid(),'host':socket.gethostname(),'started':time.time(),'compute_location':'l203 only','source':str(SOURCE)})
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(worker,r) for r in rows if json.loads((BATCH/'scenes'/r['id']/'status.json').read_text())['status']=='QUEUED']
  while any(not j.done() for j in jobs):report();time.sleep(15)
  for j in jobs:j.result()
 report()
