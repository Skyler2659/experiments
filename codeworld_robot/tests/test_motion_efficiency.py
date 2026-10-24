import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from codeworld_robot.arm_motion import ArmMotion

class MotionEfficiencyTests(unittest.TestCase):
    def adapter(self):
        self.snapshot={'revision':0,'objects':{'object':{'T_W_object':np.eye(4).tolist(),'motion_epoch':0}}}
        world=SimpleNamespace(state=lambda:self.snapshot,manifest={'geometry_revision':'g'})
        self.state={'joint_positions':[0.], 'gripper_qpos':[0.,0.], 'robot_state_ref':'s0','eef_pose_W':{'translation_m':[0.,0.,0.],'quaternion_xyzw':[0,0,0,1]}}
        d=SimpleNamespace(state=lambda:self.state,steps=0,control_period=.05)
        a=ArmMotion.__new__(ArmMotion);a.driver=d;a.settings={};a.model=SimpleNamespace(limits=[[-2,2]],joint_names=['j'])
        a.collision=SimpleNamespace(world=world,check_path=lambda *a,**kw:{'status':'CLEAR'})
        a.targets=lambda *args:(np.array([.1]),np.eye(4),[np.array([.1]),np.array([.2])])
        return a
    def test_alternative_and_cached_plan(self):
        a=self.adapter();calls=[]
        def check(*args,**kw):
            calls.append(1);return {'status':'COLLISION' if len(calls)==1 else 'CLEAR'}
        a.collision.check_path=check
        args={'commands':[{'arm':0,'pose_W':self.state['eef_pose_W']}]}
        with patch('codeworld_robot.arm_motion.joint_trajectory',return_value=[{0:np.array([.2])}]):
            p=a.plan('check_motion',args,self.state);self.assertEqual(p[3]['candidates_checked'],2);self.assertEqual(p[0][0],.2)
            self.snapshot['revision']+=1
            p=a.plan('move_tcp',{'arm':0,'pose_W':self.state['eef_pose_W']},self.state)
            self.assertTrue(p[3]['plan_reused']);self.assertEqual(len(calls),2)
            self.snapshot['objects']['object']['motion_epoch']+=1
            a.plan('check_motion',args,self.state);self.assertEqual(len(calls),3)
            self.state['gripper_qpos']=[.02,.02]
            a.plan('check_motion',args,self.state);self.assertEqual(len(calls),4)
            a.plan('check_motion',{**args,'allowed_contacts':[['a','b']]},self.state);self.assertEqual(len(calls),5)
    def test_tcp_settles_even_when_joints_are_at_target(self):
        a=self.adapter();self.state['eef_pose_W']['translation_m']=[.01,0,0]
        def apply(q):
            a.driver.steps+=1;self.state['joint_positions']=q.tolist()
            if a.driver.steps>=4:self.state['eef_pose_W']['translation_m']=[0,0,0]
            return True
        a.driver.apply_joint_target=apply
        with patch('codeworld_robot.arm_motion.joint_trajectory',return_value=[{0:np.array([.1])}]):
            r=a.dispatch('move_tcp',{'arm':0,'pose_W':{'translation_m':[0,0,0],'quaternion_xyzw':[0,0,0,1]}})
        self.assertEqual(r['status'],'ARRIVED');self.assertEqual(r['steps'],4)
    def test_all_collisions_never_actuate(self):
        a=self.adapter();a.collision.check_path=lambda *args,**kw:{'status':'COLLISION'}
        a.driver.apply_joint_target=lambda q:self.fail('blocked path actuated')
        with patch('codeworld_robot.arm_motion.joint_trajectory',return_value=[{0:np.array([.1])}]):
            r=a.dispatch('move_tcp',{'pose_W':self.state['eef_pose_W']})
        self.assertEqual(r['status'],'COLLISION_BLOCKED');self.assertEqual(r['collision']['candidates_checked'],2)

    def test_settling_respects_total_step_limit(self):
        a=self.adapter();self.state['eef_pose_W']['translation_m']=[.01,0,0]
        def apply(q):
            a.driver.steps+=1;self.state['joint_positions']=q.tolist();return True
        a.driver.apply_joint_target=apply
        with patch('codeworld_robot.arm_motion.joint_trajectory',return_value=[{0:np.array([.1])}]):
            r=a.dispatch('move_tcp',{'pose_W':{'translation_m':[0,0,0],'quaternion_xyzw':[0,0,0,1]},'max_steps':2})
        self.assertEqual(r['steps'],2);self.assertEqual(r['status'],'TARGET_NOT_REACHED')

    def test_robot_motion_invalidates_cached_path(self):
        a=self.adapter();calls=[]
        def check(*args,**kw):calls.append(1);return {'status':'CLEAR'}
        a.collision.check_path=check
        args={'commands':[{'arm':0,'pose_W':self.state['eef_pose_W']}]}
        with patch('codeworld_robot.arm_motion.joint_trajectory',return_value=[{0:np.array([.1])}]):
            a.plan('check_motion',args,self.state)
            self.state['joint_positions']=[.001]
            a.plan('check_motion',args,self.state)
            self.state['robot_state_ref']='s1'
            a.plan('check_motion',args,self.state)
        self.assertEqual(len(calls),3)
