"""Direct native FetchBench FR3 I/O. No legacy planner, task programs or object poses."""
import argparse,json,os,sys,time,types,traceback
from pathlib import Path
import numpy as np
from .io import read,atomic
from .pose import pose


class FetchServer:
    def __init__(self,workspace,config):
        self.root=Path(workspace).resolve();self.cfg=read(config);c=self.cfg
        self.io=self.root/'robot_io';self.io.mkdir()
        for name in ['requests','responses','captures']:(self.io/name).mkdir()
        if c.get('motion_dependencies'):sys.path.insert(0,c['motion_dependencies'])
        root=Path(c['fetchbench_root']);vendor=root/'vendor/InfiniGym'
        if not vendor.exists():vendor=root/'InfiniGym'
        os.environ['ASSET_PATH']=c.get('asset_root',str(root/'assets'));sys.path.insert(0,str(vendor));os.chdir(vendor)
        from isaacgym import gymapi,gymtorch
        import torch
        torch._C._jit_set_texpr_fuser_enabled(False);torch._C._jit_override_can_fuse_on_gpu(False);torch._C._jit_set_nvfuser_enabled(False)
        import isaacgymenvs
        tasks=types.ModuleType('isaacgymenvs.tasks');tasks.__path__=[str(vendor/'isaacgymenvs/tasks')];sys.modules['isaacgymenvs.tasks']=tasks
        from isaacgymenvs.tasks.fetch.fetch_base import FetchBase
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from scipy.spatial.transform import Rotation
        import imageio.v2 as imageio
        self.gymapi=gymapi;self.gymtorch=gymtorch;self.torch=torch;self.Rotation=Rotation
        np.random.seed(c.get('seed',42));torch.manual_seed(c.get('seed',42))
        with initialize_config_dir(config_dir=str(vendor/'isaacgymenvs/config'),version_base='1.1'):
            cfg=compose(config_name='config',overrides=['task=FetchBase','scene='+c['scene_config'],'headless=True'])
        cfg.task.task.scene_config_path=cfg.scene.scene_list;cfg.task.viewer.enable=False;cfg.task.sim.physx.use_gpu=True
        cfg.task.env.robot.type=c.get('robot_asset','franka_r3_cvx');cfg.task.env.armControlType='joint'
        self.env=FetchBase(OmegaConf.to_container(cfg.task,resolve=True),'cuda:0','cuda:0',0,True,False,False)
        if self.env.num_envs!=1:raise ValueError('Direct bridge requires exactly one selected native scene')
        self.env.reset_task(int(c.get('task_index',3)));self.env.set_target_color()
        for _ in range(c.get('settle_physics_steps',30)):self.env.env_physics_step();self.env.post_phy_step()
        b=self.env._robot_base_state[0].cpu().numpy();base=self.matrix(b[:3],b[3:7]);self.T_W_S=np.linalg.inv(base)
        if 'T_W_S' in c:self.T_W_S=np.asarray(c['T_W_S'],float)
        self.physics_ticks_per_control=int(self.env.control_freq_inv)
        self.control_period=float(cfg.task.sim.dt)*self.physics_ticks_per_control;self.steps=0;self.capture_index=0;self.stopped=False
        self.count_domain='physics_tick' if self.physics_ticks_per_control==1 else 'native_control_cycle';self.gripper_mode='HOLD';self.joint_goal=self.env.states['q'][0,:7].cpu().numpy().copy()
        self.video=imageio.get_writer(str(self.io/'continuous.mp4'),fps=1/self.control_period,codec='libx264',quality=7)
        from .robot_assets import franka_model
        from .arm_motion import ArmMotion
        from codeworld_workspace import World
        self.model=franka_model(vendor/('assets/urdf/franka_description/robots/'+cfg.task.env.robot.type+'.urdf'),base=self.T_W_S@base,tcp_offset=c.get('tcp_offset_m',.1034))
        self.motion=ArmMotion(self,self.model,World(self.root),c);caps=self.motion.capabilities()
        caps.update(robot_model=cfg.task.env.robot.type,native_controller='joint',steps_unit=self.count_domain,physics_ticks_per_control=self.physics_ticks_per_control,
            counter_note='Native physics ticks; not a claimed paper-level policy-action budget',raw_osc=False)
        atomic(self.io/'robot_binding.json',self.model.binding())
        atomic(self.root/'task/robot_connection.json',{'kind':'fetchbench_direct_fr3','io_directory':str(self.io),'capabilities':caps})
        atomic(self.io/'ready.json',{'status':'READY','capture':self.capture(),'fk_check':self.motion.fk_check})
        print('READY',flush=True)

    def matrix(self,p,q):
        t=np.eye(4);t[:3,3]=p;t[:3,:3]=self.Rotation.from_quat(q).as_matrix();return t

    def state(self):
        q=self.env.states['q'][0].cpu().numpy();hand=self.matrix(self.env.states['eef_pos'][0].cpu().numpy(),self.env.states['eef_quat'][0].cpu().numpy())
        tool=np.eye(4);tool[2,3]=self.cfg.get('tcp_offset_m',.1034);tcp=self.T_W_S@hand@tool
        s={'arm':0,'arm_name':'arm0','joint_positions':q[:7].tolist(),'gripper_qpos':q[7:9].tolist(),
            'eef_pose_W':pose(tcp),'robot_state_ref':self.root.name+':'+str(self.steps),'control_steps':self.steps,
            'timestamp_ns':int(self.steps*self.control_period*1e9),'clock_domain':'simulation','counter_domain':self.count_domain}
        s['arms']=[dict(s)];atomic(self.io/'robot_state.json',s);return s

    def capture(self):
        from PIL import Image
        folder=self.io/'captures'/('obs%05d'%self.capture_index);folder.mkdir();self.capture_index+=1
        rgb,_=self.env.get_camera_image(rgb=True);views=[]
        for i,camera in enumerate(self.env.cameras[0][:2]):
            rp=folder/('cam%d.png'%i);Image.fromarray(rgb[0][i]).save(rp)
            depth=-np.asarray(self.env.gym.get_camera_image(self.env.sim,self.env.envs[0],camera,self.gymapi.IMAGE_DEPTH),np.float32)
            depth[~np.isfinite(depth)]=np.nan;dp=folder/('cam%d_depth.npy'%i);np.save(dp,depth)
            projection=np.asarray(self.env.gym.get_camera_proj_matrix(self.env.sim,self.env.envs[0],camera));view=np.asarray(self.env.gym.get_camera_view_matrix(self.env.sim,self.env.envs[0],camera))
            t=self.T_W_S@np.linalg.inv(view.T)@np.diag([1,-1,-1,1]);h,w=depth.shape
            views.append({'view_id':'cam%d'%i,'rgb_uri':str(rp),'depth_uri':str(dp),'depth_convention':'camera_z_m',
                'K':[[float(w*projection[0,0]/2),0,w/2],[0,float(h*projection[1,1]/2),h/2],[0,0,1]],'T_W_C':t.tolist()})
        p=folder/'capture.json';atomic(p,{'observation_id':folder.name,'views':views,'robot_state':self.state()});atomic(self.io/'latest_capture.json',read(p));return str(p)

    def finger_target(self,mode):
        return [self.model.limits_by_name[n][1 if mode=='OPEN' else 0] for n in self.model.finger_names]

    def native_step(self):
        if self.steps>=self.cfg.get('max_control_steps',1200):return False
        t=self.torch;mode={'OPEN':1.,'CLOSE':-1.,'HOLD':0.}[self.gripper_mode]
        self.env.pre_phy_step({'joint_state':t.tensor(self.joint_goal[None],dtype=t.float32,device=self.env.device),
            'gripper_state':t.tensor([mode],dtype=t.float32,device=self.env.device)})
        self.env.env_physics_step();self.env.post_phy_step();self.steps+=1
        rgb,_=self.env.get_camera_image(rgb=True);self.video.append_data(rgb[0][0]);return True

    def apply_joint_target(self,q):self.joint_goal=np.asarray(q,float);return self.native_step()
    def apply_gripper_target(self,mode):self.gripper_mode=mode;return self.native_step()

    def run(self):
        from .ownership import authorize
        try:
            while not self.stopped:
                for path in sorted((self.io/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime_ns):
                    dest=self.io/'responses'/path.name
                    if dest.exists():continue
                    try:
                        packet=read(path);actor=authorize(self.root,packet);op=packet['op'];args=packet.get('args',{})
                        if op in self.motion.operations:result=self.motion.dispatch(op,args)
                        elif op=='observe':result={'status':'OBSERVED'}
                        elif op=='select_arm':
                            if args['arm'] not in (0,'arm0'):raise ValueError('Only arm0 is available')
                            result={'status':'SELECTED','active_arm':0}
                        elif op=='finish':self.stopped=True;result={'status':'CLOSED'}
                        else:raise ValueError('Unsupported native FR3 operation '+op)
                        result.update(actor=actor,robot_state=self.state(),control_steps=self.steps,counter_domain=self.count_domain)
                        if op not in ('check_ik','check_motion'):result['capture']=self.capture()
                        if args.get('decision_ref') is not None:result['decision_ref']=args['decision_ref']
                    except Exception as exc:result={'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()}
                    result['request_id']=path.stem;atomic(dest,result)
                    if self.stopped:break
                time.sleep(.05)
        finally:
            self.video.close()
            # The unchanged native evaluator stays private; no object state is returned.
            evaluation=self.env.eval();atomic(self.io/'native_evaluation.json',{k:np.asarray(v).tolist() for k,v in evaluation.items() if k!='label'})
            atomic(self.io/'closed.json',{'status':'CLOSED','control_steps':self.steps,'counter_domain':self.count_domain})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--config',required=True);a=p.parse_args();FetchServer(a.workspace,a.config).run()
