import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from codeworld_robot.robodojo_official import OfficialActions

class OfficialBudgetTest(unittest.TestCase):
    def test_blocked_gripper_receipt_retains_official_budget(self):
        from codeworld_robot.robodojo_server import complete_receipt
        status={'official_actions':107,'official_action_limit':1050,'remaining_actions':943,
                'terminated':False,'official_success':None}
        actor={'role':'codex_cli','run_id':'fixture'}
        result=complete_receipt({'status':'COLLISION_BLOCKED','steps':0},
            {'official_actions':107},official_status=status,actor=actor,decision_ref={'revision':3})
        self.assertEqual(result['official_actions'],107)
        self.assertEqual(result['remaining_actions'],943)
        self.assertEqual(result['steps'],0)
        self.assertEqual(result['status'],'COLLISION_BLOCKED')
        self.assertEqual(result['actor'],actor)
        self.assertEqual(result['robot_state']['official_actions'],107)
        self.assertEqual(result['decision_ref'],{'revision':3})

    def test_limit_stops_all_further_actuation(self):
        robot = SimpleNamespace(arm_name='left_arm',gripper_name='left_gripper',gripper_scale=[0,1],gripper_move={'sign':1})
        rm = SimpleNamespace(robot_list=[robot],get_joint=lambda r:[np.zeros(6)],get_end_effector_real_val=lambda r:[[1]],process_name=lambda s:s)
        env = SimpleNamespace(robot_manager=rm,dt=.004,obs_manager=SimpleNamespace(collect_interval=10),take_action_cnt=[0],step_lim=2,end_flag=[False],success=[True])
        seen=[]
        def take(action):
            seen.append(action);env.take_action_cnt[0]+=1
            if env.take_action_cnt[0]==env.step_lim:env.end_flag[0]=True;env.success[0]=False
        env.take_action=take
        with tempfile.TemporaryDirectory() as d:
            bridge=OfficialActions(env,d)
            self.assertTrue(bridge.take({0:np.ones(6)}))
            self.assertTrue(bridge.take(grippers={0:0}))
            self.assertFalse(bridge.take({0:np.zeros(6)}))
            self.assertEqual(len(seen),2)
            self.assertEqual(seen[-1]['left_arm'],[1]*6)
            self.assertEqual(bridge.status()['simulation_time_s'],.08)
            self.assertFalse(bridge.status()['official_success'])
if __name__=='__main__':unittest.main()
