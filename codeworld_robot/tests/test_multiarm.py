import unittest
import tempfile
from pathlib import Path
import numpy as np
from codeworld_robot.multiarm import arm_index, resolve_commands, joint_frames, CommandLatch
from codeworld_robot.server import Server

class MultiArmTests(unittest.TestCase):
    def test_identity_and_duplicates(self):
        self.assertEqual(arm_index('right',['left','right']),1)
        for value in (-1,2,True,'1'):
            with self.assertRaises(ValueError):arm_index(value,['left','right'])
        with self.assertRaises(ValueError):resolve_commands([{'arm':0},{'arm':'left'}],['left','right'],0)
    def test_shared_clock_and_endpoints(self):
        frames=list(joint_frames({0:np.zeros(6),1:np.ones(6)},{0:np.ones(6)*.1,1:np.ones(6)*.8},100))
        self.assertEqual(len(frames),100)
        for frame in frames:np.testing.assert_allclose(frame[1],1-2*frame[0])
        np.testing.assert_allclose(frames[-1][0],.1)
    def test_all_validated_before_execution(self):
        with self.assertRaises(ValueError):joint_frames({0:np.zeros(6),1:np.zeros(6)},{0:np.ones(6),1:[float('nan')]*6})
        with self.assertRaises(ValueError):joint_frames({0:np.zeros(6)},{0:np.ones(6)},1)
    def test_idle_arm_and_gripper_hold_last_target(self):
        latch=CommandLatch({'left':1,'right':2,'right_gripper':0})
        latch.merge({'right':3})
        self.assertEqual(latch.merge({'left':4}),{'left':4,'right':3,'right_gripper':0})
    def test_single_arm_rejects_group_before_motion(self):
        server=Server.__new__(Server)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);server.root=Path(temporary.name)
        with self.assertRaises(ValueError):server.dispatch({'op':'move_tcp_group','args':{'commands':[{'arm':0},{'arm':1}]}})
    def test_single_arm_group_and_legacy_same_path(self):
        server=Server.__new__(Server)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);server.root=Path(temporary.name)
        pose={'translation_m':[0,0,0],'quaternion_xyzw':[0,0,0,1]}
        server.state=lambda:{'eef_pose_W':pose};server.capture=lambda:'capture'
        for args in ({'pose_W':pose},{'commands':[{'arm':'arm0','pose_W':pose}]}):
            receipt=server.dispatch({'op':'move_tcp_group' if 'commands' in args else 'move_tcp','args':args})
            self.assertEqual(receipt['status'],'ARRIVED');self.assertEqual(receipt['steps'],0)

if __name__=='__main__':unittest.main()
