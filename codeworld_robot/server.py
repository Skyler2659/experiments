"""LIBERO sensor/actuator bridge. Codex owns task decisions and world updates."""
import argparse,time,traceback,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,atomic
from .pose import transform,pose

class Server:
    joint_mode=False
    motion=None
    def __init__(self,workspace,config):
        import torch,imageio.v2 as imageio
        from libero.libero.envs import OffScreenRenderEnv
        self.root=Path(workspace).resolve();self.cfg=read(config);c=self.cfg
        self.joint_mode=c.get('robot_control','osc')=='joint_position'
        if c.get('full_arm_collision_check') and not self.joint_mode:
            raise ValueError('Full planned joint-path checks require explicit robot_control=joint_position; legacy OSC remains available separately')
        if c.get('motion_dependencies'):sys.path.insert(0,c['motion_dependencies'])
        self.io=self.root/'robot_io';self.io.mkdir()
        for d in ['requests','responses','captures']:(self.io/d).mkdir()
        self.cameras=c.get('cameras',['agentview','sideview','robot0_eye_in_hand'])
        base=Path(c['libero_root'])/'libero/libero'
        self.env=OffScreenRenderEnv(bddl_file_name=str(base/'bddl_files'/c['suite']/(c['scene_task']+'.bddl')),
            controller='JOINT_POSITION' if self.joint_mode else 'OSC_POSE',
            camera_names=self.cameras,camera_heights=512,camera_widths=512,control_freq=20,render_gpu_device_id=-1,
            horizon=int(c.get('horizon',1000)), ignore_done=bool(c.get('unlimited_episode',False)))
        self.env.seed(c['seed']);self.env.reset()
        states=torch.load(base/'init_files'/c['suite']/(c['scene_task']+'.pruned_init'),weights_only=False)
        self.obs=self.env.set_init_state(states[c['initial_state']])
        if c.get('settle_scene_fixed_robot'):
            from .libero_evaluation import settle_scene
            self.settling=settle_scene(self.env)
            self.obs=self.env.set_init_state(self.env.sim.get_state().flatten())
        else:
            for _ in range(c.get('settle_steps',10)):self.obs,*_=self.env.step([0.]*(7 if self.joint_mode else 6)+[-1.])
        if c.get('resume_state_path'):
            # Exact saved physics state; no task-object state is exposed to the agent.
            self.obs=self.env.set_init_state(np.load(c['resume_state_path']))
        # Calibration from the robot base is proprioceptive, not task-object state.
        if 'T_W_S' in c:self.T_W_S=np.asarray(c['T_W_S'])
        else:
            sim=self.env.sim;bid=sim.model.body_name2id('robot0_base');base=np.eye(4)
            base[:3,:3]=sim.data.body_xmat[bid].reshape(3,3);base[:3,3]=sim.data.body_xpos[bid]
            self.T_W_S=np.linalg.inv(base)
        self.steps=0;self.capture_index=0;self.stopped=False;self.official_first_success_step=None;self.evaluation_started_at=time.time();self.last_gripper_command='UNKNOWN' if c.get('resume_state_path') else 'OPEN'
        sim=self.env.sim
        for name,fov in c.get('camera_fov',{}).items():sim.model.cam_fovy[sim.model.camera_name2id(name)]=fov
        sim.render(camera_name='agentview',height=512,width=512);ctx=sim._render_context_offscreen
        if int(ctx.con.offSamples)!=0:
            sim.model.vis.quality.offsamples=0;ctx.gl_ctx.make_current();ctx.con.free();ctx._set_mujoco_context_and_buffers()
        if c.get('evaluation_output'):
            self.initial_official_success=bool(self.env.check_success())
            self.write_evaluation('READY')
        self.video=imageio.get_writer(str(self.io/'continuous.mp4'),fps=20,codec='libx264',quality=7)
        self.control_period=.05;self.motion=None
        caps={'schema':'codeworld-robot-v2','arms':[{'arm':0,'arm_name':'arm0'}],'synchronized_commands':False,'ik_query':False,'full_arm_collision_check':False,'joint_targets':False,'raw_osc':True}
        if self.joint_mode:
            self.joint_goal=np.asarray(self.obs['robot0_joint_pos']).copy()
            controller=self.env.robots[0].controller;native_goal=controller.set_goal
            def absolute_goal(action,set_qpos=None):
                return native_goal(action,set_qpos=self.joint_goal if set_qpos is None else set_qpos)
            controller.set_goal=absolute_goal
            from .robot_assets import export_mujoco_robot
            from .arm_motion import ArmMotion
            from codeworld_workspace import World
            robot=self.env.robots[0]
            self.model=export_mujoco_robot(sim,joint_names=robot.robot_model.joints,finger_names=robot.gripper.joints,
                root_body=robot.robot_model.root_body,tcp_site=robot.eef_site_id,output=self.root/'robot_io/public_robot',T_W_S=self.T_W_S,tcp_rotation_body=robot.robot_model.eef_name)
            self.motion=ArmMotion(self,self.model,World(self.root),c);caps=self.motion.capabilities()
            caps.update(robot_backend='libero',tcp_frame_convention='native_grip_site_position_with_eef_body_orientation',tool_axes_source='public_robot_urdf_and_T_link_tcp',robot_model='libero_panda',native_controller='JOINT_POSITION',control_protocol='absolute_joint_goal_native_PD',steps_unit='native_policy_cycle')
            atomic(self.io/'robot_binding.json',self.model.binding())
        atomic(self.root/'task/robot_connection.json',{'kind':'libero_direct_io','io_directory':str(self.io),'capabilities':caps})
        atomic(self.io/'ready.json',{'status':'READY','initial_capture':self.capture(),'policy':'native joint trajectory with sampled collision checks' if self.joint_mode else 'direct OSC'})
    def state(self):
        o=self.obs;t=transform({'translation_m':o['robot0_eef_pos'],'quaternion_xyzw':o['robot0_eef_quat']})
        s={'eef_pose_W':pose(self.T_W_S@t),'joint_positions':np.asarray(o['robot0_joint_pos']).tolist(),
           'gripper_qpos':np.asarray(o['robot0_gripper_qpos']).tolist(),'last_gripper_command':self.last_gripper_command,
           'robot_state_ref':f'{self.root.name}:{self.steps}','timestamp_ns':self.steps*50000000,'clock_domain':'simulation'}
        s.update(arm=0,arm_name='arm0');s['arms']=[dict(s)]
        atomic(self.io/'robot_state.json',s);return s
    def capture(self):
        from PIL import Image
        folder=self.io/'captures'/f'obs{self.capture_index:05d}';folder.mkdir();self.capture_index+=1
        sim=self.env.sim;views=[]
        for name in self.cameras:
            rgb,raw=sim.render(camera_name=name,height=512,width=512,depth=True)
            rgb_path=folder/f'{name}.png';Image.fromarray(np.flipud(rgb)).save(rgb_path)
            near=float(sim.model.vis.map.znear*sim.model.stat.extent);far=float(sim.model.vis.map.zfar*sim.model.stat.extent)
            raw=np.flipud(np.asarray(raw,float));depth=np.where((raw>=0)&(raw<1),near/(1-raw*(1-near/far)),np.nan)
            depth_path=folder/f'{name}_depth.npy';np.save(depth_path,depth.astype('float32'))
            cid=sim.model.camera_name2id(name);t=np.eye(4);t[:3,:3]=sim.data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);t[:3,3]=sim.data.cam_xpos[cid]
            f=256/np.tan(np.deg2rad(sim.model.cam_fovy[cid])/2)
            views.append({'view_id':name,'rgb_uri':str(rgb_path),'depth_uri':str(depth_path),'depth_convention':'camera_z_m',
                          'K':[[float(f),0,255.5],[0,float(f),255.5],[0,0,1]],'T_W_C':(self.T_W_S@t).tolist()})
        p={'observation_id':folder.name,'robot_state':self.state(),'views':views};atomic(folder/'capture.json',p);atomic(self.io/'latest_capture.json',p)
        return str(folder/'capture.json')
    def step(self,action):
        a=np.asarray(action,float)
        if a.shape!=((8,) if self.joint_mode else (7,)) or not np.isfinite(a).all():raise ValueError('Invalid native controller action dimensions')
        limit=self.cfg.get('max_action_steps')
        if limit is not None and self.steps>=limit:
            self.write_evaluation('BUDGET_EXHAUSTED')
            raise RuntimeError('Episode action budget exhausted; finish and report the observed outcome')
        self.obs,*_=self.env.step(np.clip(a,-1,1));self.steps+=1
        if self.cfg.get('evaluation_output'):
            success=bool(self.env.check_success())
            if success and self.official_first_success_step is None:self.official_first_success_step=self.steps
            if success or self.steps%20==0:self.write_evaluation('RUNNING')
        if a[-1]>0:self.last_gripper_command='CLOSE'
        elif a[-1]<0:self.last_gripper_command='OPEN'
        self.video.append_data(np.flipud(self.env.sim.render(camera_name='agentview',height=512,width=512)))
    def apply_joint_target(self,q):
        if self.steps>=self.cfg.get('max_action_steps',self.cfg.get('horizon',1000)):return False
        self.joint_goal=np.asarray(q,float).copy();self.step(np.r_[np.zeros(7),0.]);return True
    def apply_gripper_target(self,mode):
        if self.steps>=self.cfg.get('max_action_steps',self.cfg.get('horizon',1000)):return False
        self.step(np.r_[np.zeros(7),{'OPEN':-1.,'CLOSE':1.,'HOLD':0.}[mode]]);return True
    def finger_target(self,mode):
        return [sorted(self.model.limits_by_name[n],key=abs)[-1 if mode=='OPEN' else 0] for n in self.model.finger_names]
    def write_evaluation(self,status):
        destination=self.cfg.get('evaluation_output')
        if not destination:return
        final=bool(self.env.check_success())
        atomic(destination,{'status':status,'suite':self.cfg['suite'],'task':self.cfg['scene_task'],
            'seed':self.cfg['seed'],'initial_state':self.cfg['initial_state'],'steps':self.steps,
            'official_success_final':final,'official_success_ever':self.official_first_success_step is not None,
            'first_success_step':self.official_first_success_step,
            'initial_success':getattr(self,'initial_official_success',False),
            'max_action_steps':self.cfg.get('max_action_steps'),'elapsed_seconds':time.time()-self.evaluation_started_at,
            'judge':'unmodified LIBERO env.check_success; private evaluator, never sent to agent',
            'settling':getattr(self,'settling',None)})
    def servo(self,target,grip):
        current=transform(self.state()['eef_pose_W']);desired=transform(target)
        delta=self.T_W_S[:3,:3].T@(desired[:3,3]-current[:3,3])
        dr=Rotation.from_matrix(self.T_W_S[:3,:3].T@desired[:3,:3]@current[:3,:3].T@self.T_W_S[:3,:3]).as_rotvec()
        return np.r_[np.clip(delta/.05,-1,1),np.clip(dr/.5,-1,1),grip]
    def dispatch(self,packet):
        from .ownership import authorize
        source=authorize(self.root,packet)
        a=packet.get('args',{});op=packet['op'];result={'actor':source}
        if self.motion and op in self.motion.operations:
            result=self.motion.dispatch(op,a)
            result.update(actor=source,robot_state=self.state(),control_steps=self.steps,official_actions=self.steps,
                official_action_limit=self.cfg.get('max_action_steps',self.cfg.get('horizon',1000)))
            if a.get('decision_ref') is not None:result['decision_ref']=a['decision_ref']
            if op not in ('check_ik','check_motion'):result['capture']=self.capture()
            return result
        from .multiarm import arm_index,resolve_commands
        if op in ('move_tcp_group','gripper_group'):
            command=resolve_commands(a['commands'],['arm0'],0)[0]
            a={**a,**command};op=op.removesuffix('_group')
        if 'arm' in a:arm_index(a['arm'],['arm0'])
        if op=='select_arm':result={'status':'SELECTED','active_arm':0}
        elif op=='observe':result={'status':'OBSERVED'}
        elif op=='move_tcp':
            target=a['pose_W'];desired=transform(target);result={'status':'STEP_LIMIT','steps':0}
            count=a.get('max_steps',150)
            if isinstance(count,bool) or not isinstance(count,int) or count<1:raise ValueError('max_steps must be positive integer')
            for _ in range(count):
                current=transform(self.state()['eef_pose_W']);dp=float(np.linalg.norm(desired[:3,3]-current[:3,3]));dr=float(Rotation.from_matrix(desired[:3,:3]@current[:3,:3].T).magnitude())
                if dp<a.get('position_tolerance_m',.004) and dr<a.get('rotation_tolerance_rad',.06):result['status']='ARRIVED';break
                self.step(self.servo(target,0));result['steps']+=1
            current=transform(self.state()['eef_pose_W']);dp=float(np.linalg.norm(desired[:3,3]-current[:3,3]));dr=float(Rotation.from_matrix(desired[:3,:3]@current[:3,:3].T).magnitude())
            if dp<a.get('position_tolerance_m',.004) and dr<a.get('rotation_tolerance_rad',.06):result['status']='ARRIVED'
            result.update(position_error_m=dp,rotation_error_rad=dr,commanded_arms=[0])
        elif op=='gripper':
            mode=a['mode'];grip={'OPEN':-1,'CLOSE':1,'HOLD':0}[mode];target=self.state()['eef_pose_W']
            count=a.get('steps',12)
            if isinstance(count,bool) or not isinstance(count,int) or count<1:raise ValueError('steps must be positive integer')
            for _ in range(count):self.step(self.servo(target,grip))
            result={'status':'EXECUTED'}
        elif op=='step':
            if self.motion:raise ValueError('Raw OSC is unavailable in explicit joint-path mode; read capabilities')
            for action in a['actions']:self.step(action)
            result={'status':'EXECUTED','steps':len(a['actions'])}
        elif op=='finish':
            # Private terminal snapshot, never returned to the acting agent.
            if self.cfg.get('terminal_audit_dir'):
                audit=Path(self.cfg['terminal_audit_dir']);audit.mkdir(parents=True,exist_ok=True)
                np.save(audit/'terminal_sim_state.npy',self.env.sim.get_state().flatten())
                (audit/'model.xml').write_text(self.env.sim.model.get_xml())
            self.stopped=True;result={'status':'CLOSED'}
        else:raise ValueError('Unknown robot operation: '+op)
        if a.get('decision_ref') is not None:result['decision_ref']=a['decision_ref']
        result['actor']=source;result['robot_state']=self.state();result['capture']=self.capture();return result
    def run(self):
        try:
            while not self.stopped:
                for path in sorted((self.io/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime_ns):
                    dest=self.io/'responses'/path.name
                    if dest.exists():continue
                    try:result=self.dispatch(read(path))
                    except Exception as exc:result={'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()}
                    result['request_id']=path.stem;atomic(dest,result)
                    if self.stopped:break
                time.sleep(.05)
        finally:
            self.write_evaluation('CLOSED');self.video.close();self.env.close();atomic(self.io/'closed.json',{'status':'CLOSED','steps':self.steps})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--config',required=True);a=p.parse_args();Server(a.workspace,a.config).run()
