"""RoboDojo direct I/O: native X5 IK + joint servo, no task-policy gates."""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
from .io import atomic,read

def complete_receipt(result, robot_state, *, official_status=None, actor=None, decision_ref=None):
 """Include clock/budget provenance even when execution is rejected before motion."""
 if official_status is not None:result.update(official_status)
 result.update(robot_state=robot_state,actor=actor)
 if decision_ref is not None:result['decision_ref']=decision_ref
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--arm',type=int,default=0);p.add_argument('--config')
 a,extra=p.parse_known_args();root=Path(a.workspace).resolve();io=root/'robot_io';io.mkdir(parents=True)
 for d in ['requests','responses','captures']:(io/d).mkdir(exist_ok=True)
 settings=read(a.config) if a.config else {};official=bool(settings.get('official_evaluation',False))
 benchmark=settings.get('benchmark_protocol')=='robodojo-v1'
 baseline=benchmark and settings.get('strategy')=='roboprobe_l3'
 if benchmark and not official:raise ValueError('Benchmark episodes require the official evaluator')
 oracle=None
 if settings.get('motion_dependencies'):sys.path.insert(0,settings['motion_dependencies'])
 ROOT=Path(settings.get('robodojo_root','/private/lc/research/RoboDojo'));sys.path.insert(0,str(ROOT/'XPolicyLab'));sys.path.insert(0,str(ROOT))
 import warp
 from isaaclab.app import AppLauncher
 ap=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(ap);opts=ap.parse_args([])
 device=int(settings.get('sim_device',0));renderer=int(settings.get('renderer_gpu',0))
 opts.headless=True;opts.enable_cameras=True;opts.device=f'cuda:{device}';opts.kit_args=f'--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 --/renderer/multiGpu/enabled=False --/renderer/activeGpu={renderer}'
 from env.camera_manager.capture.render_sync import add_zero_delay_kit_args,wait_for_latest_cameras
 add_zero_delay_kit_args(opts);app=AppLauncher(opts).app
 from omegaconf import OmegaConf
 from utils.load_file import load_yaml
 from utils.pipeline_utils import process_randomization,process_config
 from env.seed_manager.seed_manager import SeedManager
 from task.RoboDojo.task_registry import load_task_class
 from scipy.spatial.transform import Rotation
 from PIL import Image
 import cv2
 task_name=settings.get('task_name','build_tower');seed=int(settings.get('seed',0));layout_id=int(settings.get('layout_id',seed))
 base=ROOT/'env_cfg';ec=load_yaml(str(base/settings.get('env_config','arx_x5.yml')));ec.update(task_name=task_name,num_envs=1,device_id=0,seed=seed)
 refs=ec['config'];cfg=OmegaConf.create({k:load_yaml(str(base/k/(refs[k]+'.yml'))) for k in ('sim','scene','camera','robot')});cfg.task_env=load_yaml(str(ROOT/f'task/RoboDojo/config/{task_name}.yml'));cfg.eval_cfg=ec;cfg.sim.scene.num_envs=1;cfg.sim.seed=[0] if benchmark else [seed];cfg.camera.default_frequency=25
 if not baseline:
  for _,row in cfg.camera.annotator.items():row.depth_capture={'type':'distance_to_image_plane','device':'cpu'}
 cfg=process_randomization(cfg);cfg,_=process_config(cfg,task_name=task_name)
 if not benchmark:cfg.sim.render=OmegaConf.create({'antialiasing_mode':'Off'})
 if official:
  from .robodojo_official import create_official_env,OfficialActions
  env=create_official_env(cfg,app,layout_id,managed_loop=benchmark,result_directory=settings.get('evaluator_directory'),
      policy_name='RoboDojo_Agent_L3_Inspect_EEF' if baseline else 'codeworld_file_policy')
  official_actions=OfficialActions(env,settings.get('evaluator_directory',io),expose_outcome=not benchmark)
  if baseline:
   # The released L3 deploy explicitly ignores model_client: its planner and
   # EEF executor run in this process. Only the unused WebSocket is replaced.
   if settings.get('probe_native_only'):
    # Initialization-only check: exercise the released observation decoder,
    # action binding and EEF planner import, with no model call or task action.
    from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect import deploy as native_deploy
    from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.deploy import _planner
    def baseline_probe():
     before=int(env.take_action_cnt[0]);spec=native_deploy._action_spec(env)
     observation=native_deploy._observation(env,0)
     assert callable(_planner(env))
     if not observation.images:raise ValueError('RoboProbe has no native RGB observation')
     if int(env.take_action_cnt[0])!=before:raise RuntimeError('Baseline observation probe advanced native actions')
     atomic(Path(settings['evaluator_directory'])/'baseline_interface_probe.json',
       {'rgb_images':len(observation.images),'state_fields':list(observation.state),'action_dimensions':len(spec.labels),'action_labels':list(spec.labels),
        'model_called':False,'actions':0,'policy':'RoboDojo_Agent_L3_Inspect_EEF'})
     env.success[0]=False;env.is_episode_end()
    env.eval_one_episode=baseline_probe
   from .recording import FrameTimeline
   timeline=FrameTimeline(root)
   baseline_video=cv2.VideoWriter(str(io/'continuous.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,480))
   original_take=env.take_action
   def recorded_take(*args,**kwargs):
    result=original_take(*args,**kwargs)
    env.render();wait_for_latest_cameras(env,capture_manager=env.capture_manager)
    data=env.capture_manager.step()
    baseline_video.write(np.asarray(data[0]['rgb'][0]['data'])[:,:,:3][:,:,::-1].copy())
    timeline.operation={'op':'native_take_action','args':str(args),'kwargs':str(kwargs)}
    timeline.record({'official_actions':int(env.take_action_cnt[0])})
    return result
   env.take_action=recorded_take
   try:env.run_eval()
   finally:
    baseline_video.release();official_actions.finalize();env.close();app.close()
   return
 else:
  sm=SeedManager(cfg.eval_cfg);sm.init_eval();layout=sm.get_seed_scene_info(seed);_,klass=load_task_class(task_name);env=klass(cfg,app)
  env.step_lim=float('inf');env.scene_manager.layout_manager.set_saved_layout(0,layout);env.reset(seed=[seed]);env.scene_manager.apply_saved_poses(env_idx_list=[0]);env.robot_manager.set_origin_endpose();env.robot_manager.set_robot_init_state()
 rm=env.robot_manager;origin=np.asarray(settings.get('world_origin_S_m',[0.,-.45,.765]),dtype=float);steps=0;captures=0;active=a.arm;stopped=False
 from .x5_ik import X5IK
 from .multiarm import arm_index,resolve_commands,joint_frames,CommandLatch
 from .pose import transform
 from .motion import joint_trajectory
 names=[r.arm_name for r in rm.robot_list];active=arm_index(active,names)
 solvers=[]
 for i,r in enumerate(rm.robot_list):
  base_pose=np.asarray(r.entity_origin_pose).reshape(-1,7)[0]
  base_T=np.eye(4);base_T[:3,3]=base_pose[:3]-origin;base_T[:3,:3]=Rotation.from_quat(np.r_[base_pose[4:7],base_pose[3]]).as_matrix()
  solvers.append(X5IK(ROOT/'Assets/Robots/x5/X5A.urdf',i,base=base_T,tcp_offset=[float(r.gripper_bias),0,0]))
 capabilities={'schema':'codeworld-robot-v2','robot_backend':'robodojo','robot_model':'dual_x5','arms':[{'arm':i,'arm_name':name,'base_T_W':solvers[i].base.tolist(),'tcp_offset_link6_m':solvers[i].tcp_offset.tolist(),'joint_limits_rad':solvers[i].limits,'approach_axis_tcp':[1,0,0]} for i,name in enumerate(names)],'synchronized_commands':True,'ik_query':True,'full_arm_collision_check':False,'joint_targets':True,'raw_osc':False}
 physics_dt=float(cfg.sim.dt)*int(cfg.sim.decimation);record_interval=max(1,round(1/(25*physics_dt)))
 if official:capabilities.update(official_actions.status(),official_evaluation=True)
 collision=None
 if settings.get('full_arm_collision_check',False) and not benchmark:
  from .collision import robodojo_collision
  from codeworld_workspace import World
  collision=robodojo_collision(ROOT/'Assets/Robots/x5/X5A.urdf',{i:s.base for i,s in enumerate(solvers)},World(root))
  capabilities.update(full_arm_collision_check=True,collision_coverage='sampled_URDF_joint_path',
    self_collision_check=True,inter_arm_collision_check=True,payload_collision_check=True,
    collision_contact_names='arm0/link7, arm0/link8, ...; object IDs from World',
    motion_profile=settings.get('motion_profile','ruckig'),max_velocity_rad_s=settings.get('max_velocity_rad_s',2.),
    max_acceleration_rad_s2=settings.get('max_acceleration_rad_s2',8.))
 from .recording import FrameTimeline
 timeline=FrameTimeline(root)
 video=cv2.VideoWriter(str(io/'continuous.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,480))
 # Initialize persistent actuator targets once; never reset the inactive arm.
 initial=[rm.get_joint(r)[0].copy() for r in rm.robot_list]
 initial_grippers=[rm.get_end_effector_real_val(r)[0].copy() for r in rm.robot_list]
 latch=CommandLatch({key:{'position':value.copy(),'velocity':np.zeros_like(value)} for i,r in enumerate(rm.robot_list) for key,value in [(rm.process_name(r.arm_name),initial[i]),(rm.process_name(r.gripper_name),initial_grippers[i])]})
 def state():
  rows=[]
  for i,r in enumerate(rm.robot_list):
   pose=rm.get_real_endpose(r)[0];rot=Rotation.from_quat(np.r_[pose[4:7],pose[3]]);tcp=pose[:3]+rot.apply([r.gripper_bias,0,0])-origin
   rows.append({'arm':i,'arm_name':r.arm_name,'eef_pose_W':{'translation_m':tcp.tolist(),'quaternion_xyzw':rot.as_quat().tolist()},'joint_positions':rm.get_joint(r)[0].tolist(),'gripper_qpos':rm.get_end_effector_real_val(r)[0].tolist(),'tcp_offset_link6_m':[float(r.gripper_bias),0,0]})
  for i,row in enumerate(rows):row.update(home_joint_positions=initial[i].tolist(),home_gripper_qpos=initial_grippers[i].tolist())
  s=dict(rows[active]);s.update(arms=rows,active_arm=active,robot_state_ref=f'{settings.get("run_id",root.name)}:{steps}',control_steps=steps,timestamp_ns=round(steps*physics_dt*1e9),clock_domain='simulation');s.update(official_actions.status() if official else {});atomic(io/'robot_state.json',s);return s
 
 def frame(record=True):
  env.render();wait_for_latest_cameras(env,capture_manager=env.capture_manager);data=env.capture_manager.step()
  if record:
   video.write(np.asarray(data[0]['rgb'][0]['data'])[:,:,:3][:,:,::-1].copy())
   timeline.record({'control_steps':steps,'simulation_time_s':steps*physics_dt})
  return data
 
 def capture():
  nonlocal captures
  if oracle is not None:oracle.sync(state())
  for _ in range(3):env.render()
  data=frame(False);d=io/'captures'/f'obs{captures:05d}';d.mkdir();views=[]
  for j,row in enumerate(data):
   rgb=np.asarray(row['rgb'][0]['data'])[:,:,:3];depth=np.asarray(row['distance_to_image_plane'][0]['data']).squeeze();Image.fromarray(rgb).save(d/f'cam{j}.png');np.save(d/f'cam{j}_depth.npy',depth)
   cam=env.camera_manager.cameras[0][j];pos,q=cam.get_world_pose(camera_axes='ros');T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix();T[:3,3]=np.asarray(pos)-origin
   views.append({'view_id':f'cam{j}','rgb_uri':str(d/f'cam{j}.png'),'depth_uri':str(d/f'cam{j}_depth.npy'),'K':env.camera_manager.get_camera_intrinsics(j).tolist(),'T_W_C':T.tolist(),'depth_convention':'camera_z_m'})
  c={'observation_id':d.name,'views':views,'robot_state':state()};atomic(d/'capture.json',c);atomic(io/'latest_capture.json',c);captures+=1;return str(d/'capture.json')
 
 def tick(control):
  nonlocal steps
  control=latch.merge(control)
  rm.control_manager.push([0],[[control]]);env.step(rm.control_manager.pop([0]));env.sim_step(render=False);steps+=1
  if steps%record_interval==0:frame()
 
 def official_take(joints=None,grippers=None):
  nonlocal steps
  if not official_actions.take(joints,grippers):return False
  steps=official_actions.status()['official_actions']*official_actions.interval
  frame()
  return True

 def planned_frames(targets,args):
  starts={i:rm.get_joint(r)[0].copy() for i,r in enumerate(rm.robot_list)}
  period=official_actions.interval*physics_dt if official else physics_dt
  if collision:
   duration=args.get('duration_s')
   if duration is None and args.get('duration_steps') is not None:duration=args['duration_steps']*physics_dt
   frames=joint_trajectory(starts,targets,period,args.get('max_velocity_rad_s',settings.get('max_velocity_rad_s',2.)),
       args.get('max_acceleration_rad_s2',settings.get('max_acceleration_rad_s2',8.)),duration,
       settings.get('motion_profile','ruckig'),args.get('max_jerk_rad_s3',settings.get('max_jerk_rad_s3',80.)))
  else:
   frames=list(joint_frames({i:starts[i] for i in targets},targets,args.get('duration_steps')))
  return starts,frames

 def validate_motion(targets,args):
  starts,frames=planned_frames(targets,args)
  fingers={i:rm.get_end_effector_real_val(r)[0].copy() for i,r in enumerate(rm.robot_list)}
  allowed=list(settings.get('allowed_contacts',[]))+list(args.get('allowed_contacts',[]))
  report=collision.check_path(starts,frames,fingers,allowed_contacts=allowed,
    attachments=args.get('attachments',[]),max_joint_increment=settings.get('collision_joint_increment_rad',.025))
  report.update(planned_official_actions=len(frames) if official else None,nominal_duration_s=len(frames)*(official_actions.interval*physics_dt if official else physics_dt))
  return frames,report

 def execute_joints(targets,n=None,args=None):
  args=args or {}
  starts={i:rm.get_joint(rm.robot_list[i])[0].copy() for i in targets}
  for i,q in targets.items():
   limits=np.asarray(solvers[i].limits)
   if np.asarray(q).shape!=(6,) or not np.isfinite(q).all() or np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('Joint target outside URDF limits')
  if collision:
   frames,report=validate_motion(targets,args)
   if report['status']!='CLEAR':return 0,report
   count=0
   for values in frames:
    if official:
     if not official_take(values):break
     count+=official_actions.interval
    else:
     tick({rm.process_name(rm.robot_list[i].arm_name):{'position':q,'velocity':np.zeros_like(q)} for i,q in values.items()});count+=1
   # Event-based settling, bounded in official periods. No unconditional tail.
   tolerance=args.get('tracking_tolerance_rad',.012)
   extra=0
   for _ in range(int(args.get('max_settle_actions',settings.get('max_settle_actions',4)))):
    if official and official_actions.ended:break
    if max(np.max(abs(rm.get_joint(rm.robot_list[i])[0]-q)) for i,q in targets.items())<=tolerance:break
    if official:
     if not official_take(targets):break
     count+=official_actions.interval
    else:tick({});count+=1
    extra+=1
   report['settle_actions']=extra
   return count,report
  if official:
   # Task controller supplies waypoints; each is executed and counted upstream.
   interval=official_actions.interval
   increment=float(settings.get('max_joint_increment_rad',.012))
   frames=list(joint_frames(starts,targets,n,max_increment=increment))
   count=0
   for k in range(interval-1,len(frames)+interval-1,interval):
    if not official_take(frames[min(k,len(frames)-1)]):break
    count+=interval
   for _ in range(2):
    if not official_take():break
    count+=interval
   return count,None
  frames=joint_frames(starts,targets,n);count=0
  for values in frames:
   tick({rm.process_name(rm.robot_list[i].arm_name):{'position':q,'velocity':np.zeros_like(q)} for i,q in values.items()});count+=1
  for _ in range(25):tick({});count+=1
  return count,None
 
 def dispatch(packet):
  nonlocal stopped,active
  from .ownership import authorize
  source=authorize(root,packet)
  op=packet['op'];args=packet.get('args',{});result={'actor':source}
  if op in ('move_tcp','move_tcp_group','joints','joints_group','gripper','gripper_group','check_ik','check_motion'):
   commands=resolve_commands(args['commands'] if op.endswith('_group') or op=='check_motion' else [args],names,active)
   result['commanded_arms']=list(commands)
   if op.startswith('move_tcp') or op in ('check_ik','check_motion'):
    targets={};solutions={};all_candidates={}
    for i,c in commands.items():
     if op=='check_motion' and 'joint_positions' in c:
      solutions[i]=np.asarray(c['joint_positions'],float);continue
     T=transform(c['pose_W']);targets[i]=T
     candidates=solvers[i].solve_candidates(T,rm.get_joint(rm.robot_list[i])[0])
     all_candidates[str(i)]=[q.tolist() for q in candidates]
     solutions[i]=candidates[0] if candidates else None
    failed=[i for i,q in solutions.items() if q is None]
    if failed:result.update(status='IK_NOT_FOUND',failed_arms=failed,steps=0)
    elif op=='check_ik':result.update(status='IK_FOUND',joint_targets={str(i):q.tolist() for i,q in solutions.items()},joint_candidates=all_candidates,collision_checked=False)
    elif op=='check_motion':
     if not collision:raise ValueError('Full-arm collision checking is not configured')
     _,report=validate_motion(solutions,args)
     result.update(status=report['status'],collision=report,joint_targets={str(i):q.tolist() for i,q in solutions.items()},steps=0)
    else:
     n,report=execute_joints(solutions,args.get('duration_steps'),args);rows=state()['arms'];errors={}
     for i,T in targets.items():
      actual=transform(rows[i]['eef_pose_W']);errors[str(i)]={'position_error_m':float(np.linalg.norm(actual[:3,3]-T[:3,3])),'rotation_error_rad':float(Rotation.from_matrix(actual[:3,:3].T@T[:3,:3]).magnitude())}
     arrived=all(e['position_error_m']<.004 and e['rotation_error_rad']<.06 for e in errors.values())
     result.update(status='COLLISION_BLOCKED' if report and report['status']=='COLLISION' else ('ARRIVED' if arrived else 'TARGET_NOT_REACHED'),steps=n,arm_errors=errors,collision=report)
     if len(errors)==1:result.update(next(iter(errors.values())))
   elif op.startswith('joints'):
    n,report=execute_joints({i:np.asarray(c['joint_positions'],dtype=float) for i,c in commands.items()},args.get('duration_steps'),args)
    result.update(status='COLLISION_BLOCKED' if report and report['status']=='COLLISION' else 'EXECUTED',steps=n,collision=report)
   else:
    n=args.get('steps',80)
    if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError('steps must be a positive integer')
    for c in commands.values():
     if c['mode'] not in ('OPEN','CLOSE','HOLD'):raise ValueError('Invalid gripper mode')
    if collision:
     starts={i:rm.get_joint(r)[0].copy() for i,r in enumerate(rm.robot_list)}
     fingers={i:rm.get_end_effector_real_val(r)[0].copy() for i,r in enumerate(rm.robot_list)}
     finger_targets={}
     for i,c in commands.items():
      if c['mode']=='HOLD':continue
      r=rm.robot_list[i];val=1. if c['mode']=='OPEN' else 0.
      if r.gripper_move['sign']!=1:val=1-val
      val=val*(r.gripper_scale[1]-r.gripper_scale[0])+r.gripper_scale[0]
      finger_targets[i]=np.array([val,val*r.gripper_move['mimic'][1]+r.gripper_move['mimic'][2]])
     samples=max(1,int(np.ceil(max([0.]+[np.max(abs(v-fingers[i])) for i,v in finger_targets.items()])/.002)))
     report=collision.check_path(starts,[starts]*samples,fingers,
      allowed_contacts=list(settings.get('allowed_contacts',[]))+list(args.get('allowed_contacts',[])),
      attachments=args.get('attachments',[]),finger_targets=finger_targets)
     result['collision']=report
     if report['status']=='COLLISION':
      result.update(status='COLLISION_BLOCKED',steps=0)
      return complete_receipt(result,state(),official_status=official_actions.status() if official else None,
        actor=source,decision_ref=args.get('decision_ref'))
    if official:
     import math
     values={i:1. if c['mode']=='OPEN' else 0. for i,c in commands.items() if c['mode']!='HOLD'}
     for _ in range(math.ceil(n/official_actions.interval)):
      if not official_take(grippers=values):break
    else:
     plans={i:list(rm.plan_endeffector_joint(0,rm.robot_list[i].arm_name,target_val=1. if c['mode']=='OPEN' else 0.,control_step_num=n)) for i,c in commands.items() if c['mode']!='HOLD'}
     for k in range(max([n]+[len(plan) for plan in plans.values()])):
      control={}
      for plan in plans.values():
       if k<len(plan):control.update(plan[k])
      tick(control)
    result.update(status='EXECUTED')
  elif op=='select_arm':active=arm_index(args['arm'],names);result={'status':'SELECTED','active_arm':active}
  elif op=='observe':result={'status':'OBSERVED'}
  elif op=='finish':stopped=True;result={'status':'CLOSED'}
  else:raise ValueError('Unknown direct I/O operation '+op)
  if official:
   result.update(official_actions.status())
   if official_actions.ended and op not in ('finish','observe','select_arm','check_ik'):
    result['status']='EPISODE_ENDED' if benchmark else ('EPISODE_SUCCESS' if env.success[0] else 'EPISODE_LIMIT')
  if op not in ('check_ik','check_motion'):result['capture']=capture()
  return complete_receipt(result,state(),official_status=official_actions.status() if official else None,
    actor=source,decision_ref=args.get('decision_ref'))
 if not official:
  for _ in range(50):env.sim_step(render=False)
 for _ in range(10):env.render()
 if benchmark:
  from robodojo_benchmark.sim_hooks import prepare_episode
  oracle=prepare_episode(env,root,settings,state,capture,frame)
  if settings.get('full_arm_collision_check',False):
   from .collision import robodojo_collision,RobotOnlyScene
   from codeworld_workspace import World
   world=RobotOnlyScene() if settings['representation']=='none' else World(root)
   collision=robodojo_collision(ROOT/'Assets/Robots/x5/X5A.urdf',{i:s.base for i,s in enumerate(solvers)},world)
   capabilities.update(full_arm_collision_check=True,self_collision_check=True,inter_arm_collision_check=True,
       scene_collision_check=settings['representation']!='none',payload_collision_check=settings['representation']!='none',
       motion_profile=settings.get('motion_profile','ruckig'),collision_coverage='sampled_URDF_joint_path')
 if collision is not None:capabilities['fixed_mount_contacts']=[list(pair) for pair in collision.mounting_contacts]
 atomic(root/'task/robot_connection.json',{'kind':'robodojo_direct_x5','io_directory':str(io),'active_arm':active,'episode_step_limit':int(env.step_lim) if official else None,'capabilities':capabilities})
 from codeworld_workspace.robot_home import capture_home
 home=capture_home(state())
 atomic(root/'task/robot_home.json',home)
 capabilities['return_home']={'required':True,'baseline':'task/robot_home.json','joint_tolerance_rad':home['joint_tolerance_rad'],'gripper_tolerance_m':home['gripper_tolerance_m']}
 atomic(root/'task/robot_connection.json',{'kind':'robodojo_direct_x5','io_directory':str(io),'active_arm':active,'episode_step_limit':int(env.step_lim) if official else None,'capabilities':capabilities})
 atomic(io/'ready.json',{'status':'READY','capture':capture(),'arm':active,'episode_step_limit':int(env.step_lim) if official else None});print('READY',flush=True)
 def serve():
  while app.is_running() and not stopped:
   if benchmark and (root/'task/stop_server.json').exists():break
   for p in sorted((io/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime_ns):
    dest=io/'responses'/p.name
    if dest.exists():continue
    try:
     request=read(p);timeline.operation={'request_id':p.stem,**request}
     result=dispatch(request)
    except Exception as ex:result={'status':'ERROR','error':str(ex),'traceback':traceback.format_exc()}
    result['request_id']=p.stem;atomic(dest,result);print(p.stem,result['status'],steps,flush=True)
    if stopped:break
   time.sleep(.05)
  if benchmark and not official_actions.ended:
   # Same early-stop path used by upstream policy capability failures. Native
   # final_check still evaluates all task predicates; no action is added.
   env.success[0]=False
   env.is_episode_end()
 try:
  if benchmark:
   # Replace only policy transport. Upstream run_eval owns reward registration,
   # terminal evaluation, unstable filtering, score normalization and _result.json.
   env.eval_one_episode=serve
   env.run_eval()
  else:serve()
 finally:
  video.release()
  if official:official_actions.finalize()
  atomic(io/'closed.json',{'status':'CLOSED','steps':steps,'captures':captures});env.close();app.close()

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  # Native SimulationApp teardown can otherwise hide initialization exceptions.
  try:
   index=sys.argv.index('--config');config=read(sys.argv[index+1])
   if config.get('evaluator_directory'):
    atomic(Path(config['evaluator_directory'])/'server_error.json',{'exception_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
  except Exception:pass
  traceback.print_exc()
  import os
  os._exit(1)
