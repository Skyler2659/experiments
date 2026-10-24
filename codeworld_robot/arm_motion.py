"""Single-arm robot-independent query/trajectory adapter for native backends."""
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .pose import transform
from .motion import joint_trajectory
from .collision import RobotCollision


class ArmMotion:
    operations={'check_ik','check_motion','joints','joints_group','move_tcp','move_tcp_group','gripper','gripper_group'}

    def __init__(self,driver,model,world,settings):
        self.driver=driver;self.model=model;self.settings=settings
        self.collision=RobotCollision(model.path,{0:model.base},world,models={0:model},
            disabled_self_pairs=settings.get('disabled_self_collision_pairs',[]))
        state=driver.state();q=np.asarray(state['joint_positions']);expected=transform(state['eef_pose_W']);actual=model.fk(q,state['gripper_qpos'])
        self.fk_check={'position_error_m':float(np.linalg.norm(expected[:3,3]-actual[:3,3])),
            'rotation_error_rad':float(Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude())}
        if self.fk_check['position_error_m']>.002 or self.fk_check['rotation_error_rad']>.01:
            raise ValueError('Public robot FK/TCP binding disagrees with native proprioception: '+str(self.fk_check))

    def capabilities(self):
        return {'schema':'codeworld-robot-v2','arms':[{'arm':0,'arm_name':'arm0',**self.model.binding()}],
            'synchronized_commands':False,'ik_query':True,'joint_targets':True,'full_arm_collision_check':True,
            'self_collision_check':True,'payload_collision_check':True,'gripper_sweep_check':True,
            'collision_coverage':'sampled_joint_fk','raw_osc':False,'control_period_s':self.driver.control_period,
            'motion_profile':self.settings.get('motion_profile','quintic'),'fk_binding_check':self.fk_check}

    def commands(self,op,args):
        from .multiarm import resolve_commands
        return resolve_commands(args['commands'] if op.endswith('_group') or op=='check_motion' else [args],['arm0'],0)

    def targets(self,op,args,state):
        commands=self.commands(op,args);c=commands[0]
        if 'joint_positions' in c:
            q=np.asarray(c['joint_positions'],float);limits=np.array(self.model.limits)
            if q.shape!=(len(self.model.joint_names),) or not np.isfinite(q).all() or np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('Invalid joint target or target outside robot limits')
            return q,None,[q]
        target=transform(c['pose_W']);candidates=self.model.solve_candidates(target,state['joint_positions'])
        return (candidates[0] if candidates else None),target,candidates

    def options(self,args):
        return {'allowed_contacts':list(self.settings.get('allowed_contacts',[]))+list(args.get('allowed_contacts',[])),
            'attachments':args.get('attachments',[]),'max_joint_increment':self.settings.get('collision_joint_increment_rad',.025)}

    def plan(self, op, args, state):
        # Reuse only an identical request, robot state and collision-world input.
        world=self.collision.world
        snapshot=world.state()
        geometry={oid:{k:row.get(k) for k in ('T_W_object','assembly','motion_epoch')}
                  for oid,row in snapshot['objects'].items()}
        key=json.dumps({'commands':self.commands(op,args),
            'options':{k:v for k,v in args.items() if k not in ('commands','arm','pose_W','joint_positions')},
            'q':state['joint_positions'],'f':state['gripper_qpos'],
            'robot_state_ref':state.get('robot_state_ref'),
            'geometry':geometry,'manifest':world.manifest,'settings':self.settings},sort_keys=True)
        if hasattr(world,'verify_binding'):world.verify_binding()
        if getattr(self,'_plan_cache',None) and self._plan_cache[0]==key:
            plan=copy.deepcopy(self._plan_cache[1]);plan[3]['plan_reused']=True
            plan[3]['state_revision']=snapshot.get('revision')
            return plan
        target,pose,candidates=self.targets(op,args,state)
        if target is None:return None,pose,[],{'status':'IK_NOT_FOUND'}
        q=np.asarray(state['joint_positions']);f=np.asarray(state['gripper_qpos'])
        rejected=[]
        for index,target in enumerate(candidates):
            frames=joint_trajectory({0:q},{0:target},self.driver.control_period,
                args.get('max_velocity_rad_s',self.settings.get('max_velocity_rad_s',1.2)),
                args.get('max_acceleration_rad_s2',self.settings.get('max_acceleration_rad_s2',4.)),
                args.get('duration_s'),self.settings.get('motion_profile','quintic'),
                args.get('max_jerk_rad_s3',self.settings.get('max_jerk_rad_s3',40.)))
            report=self.collision.check_path({0:q},frames,{0:f},**self.options(args))
            report.update(planned_control_steps=len(frames),counter_domain=getattr(self.driver,'count_domain','native_policy_cycle'),
                nominal_duration_s=len(frames)*self.driver.control_period,candidates_checked=index+1,
                candidate_count=len(candidates),plan_reused=False)
            if report['status']=='CLEAR':break
            rejected.append({'candidate':index,'status':report['status'],'hits':report.get('hits',[])})
        report['rejected_candidates']=rejected
        plan=[target,pose,frames,report]
        self._plan_cache=(key,copy.deepcopy(plan))
        return plan

    def dispatch(self,op,args):
        s=self.driver.state();q=np.asarray(s['joint_positions']);f=np.asarray(s['gripper_qpos']);before=self.driver.steps
        if op.startswith('gripper'):
            c=self.commands(op,args)[0];mode=c['mode']
            if mode not in ['OPEN','CLOSE','HOLD']:raise ValueError('Invalid gripper mode')
            ticks=args.get('steps',12)
            if isinstance(ticks,bool) or not isinstance(ticks,int) or ticks<1:raise ValueError('steps must be a positive integer')
            goal=f if mode=='HOLD' else np.asarray(self.driver.finger_target(mode))
            samples=max(1,int(np.ceil(np.max(abs(goal-f))/.002)))
            report=self.collision.check_path({0:q},[{0:q}]*samples,{0:f},finger_targets={0:goal},**self.options(args))
            if report['status']!='CLEAR':return {'status':'COLLISION_BLOCKED','steps':0,'collision':report}
            for _ in range(ticks):
                if not self.driver.apply_gripper_target(mode):break
            return {'status':'EXECUTED' if self.driver.steps-before==ticks else 'BUDGET_EXHAUSTED','steps':self.driver.steps-before,'collision':report}
        if op=='check_ik':
            target,pose,candidates=self.targets(op,args,s)
            if target is None:return {'status':'IK_NOT_FOUND','failed_arms':[0],'steps':0}
            return {'status':'IK_FOUND','joint_targets':{'0':target.tolist()},'joint_candidates':{'0':[x.tolist() for x in candidates]},'collision_checked':False,'steps':0}
        target,pose,frames,report=self.plan(op,args,s)
        if target is None:return {'status':'IK_NOT_FOUND','failed_arms':[0],'steps':0}
        result={'status':report['status'],'steps':0,'collision':report,'joint_targets':{'0':target.tolist()}}
        if op=='check_motion':return result
        if report['status']!='CLEAR':result['status']='COLLISION_BLOCKED';return result
        limit=args.get('max_steps')
        if limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit<1):raise ValueError('max_steps must be a positive integer')
        if limit is not None and len(frames)>limit:
            result['status']='MOTION_LIMIT_EXCEEDED';return result
        exhausted=False
        tolerance=float(args.get('tracking_tolerance_rad',self.settings.get('tracking_tolerance_rad',.015)))
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive tracking tolerance required')
        position_tolerance=float(args.get('position_tolerance_m',self.settings.get('position_tolerance_m',.001)))
        rotation_tolerance=float(args.get('rotation_tolerance_rad',self.settings.get('rotation_tolerance_rad',.01)))
        if not np.isfinite(position_tolerance) or position_tolerance<=0 or not np.isfinite(rotation_tolerance) or rotation_tolerance<=0:
            raise ValueError('Positive finite pose tolerances required')
        def reached(state):
            if pose is None:return np.max(abs(np.asarray(state['joint_positions'])-target))<=tolerance
            actual=transform(state['eef_pose_W'])
            return np.linalg.norm(actual[:3,3]-pose[:3,3])<=position_tolerance and Rotation.from_matrix(actual[:3,:3].T@pose[:3,:3]).magnitude()<=rotation_tolerance
        settle=args.get('max_settle_actions',self.settings.get('max_settle_actions',20))
        if isinstance(settle,bool) or not isinstance(settle,int) or settle<0:
            raise ValueError('max_settle_actions must be a nonnegative integer')
        for frame in frames:
            if not self.driver.apply_joint_target(frame[0]):exhausted=True;break
        extra=0
        if limit is not None:settle=min(settle,limit-len(frames))
        if exhausted:settle=0
        for _ in range(settle):
            if reached(self.driver.state()):break
            if not self.driver.apply_joint_target(target):exhausted=True;break
            extra+=1
        report['settle_actions']=extra;s=self.driver.state();error=float(np.max(abs(np.asarray(s['joint_positions'])-target)))
        result.update(status='EXECUTED' if error<=tolerance else 'TARGET_NOT_REACHED',steps=self.driver.steps-before,joint_error_rad=error)
        if pose is not None:
            actual=transform(s['eef_pose_W']);dp=float(np.linalg.norm(actual[:3,3]-pose[:3,3]));dr=float(Rotation.from_matrix(actual[:3,:3].T@pose[:3,:3]).magnitude())
            result.update(status='ARRIVED' if reached(s) else 'TARGET_NOT_REACHED',position_error_m=dp,rotation_error_rad=dr)
        if exhausted:result['status']='BUDGET_EXHAUSTED'
        return result
