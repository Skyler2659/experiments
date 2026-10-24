import tempfile,unittest
from pathlib import Path
import numpy as np
from codeworld_robot.kinematics import URDFKinematics,mesh_path
from codeworld_robot.arm_motion import ArmMotion
from codeworld_robot.pose import pose

class EmptyWorld:
    def __init__(self,root):self.root=root;self.manifest={'objects':{}}
    def state(self):return {'revision':0,'objects':{}}
    def objects(self):return []

class BindingTest(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);p=Path(tmp.name)/'robot.urdf'
        links=['mount']+['segment%d'%i for i in range(7)]+['tool','left','right']
        xml='<robot name="seven">'+''.join('<link name="%s"/>'%n for n in links)
        for i in range(7):
            xml+='<joint name="axis%d" type="revolute"><parent link="%s"/><child link="segment%d"/><origin xyz="0 0 .1"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>'%(i,'mount' if i==0 else 'segment%d'%(i-1),i)
        xml+='<joint name="fixed" type="fixed"><parent link="segment6"/><child link="tool"/><origin xyz=".2 0 0"/></joint>'
        for n,sign in [('left',1),('right',-1)]:
            xml+='<joint name="%s_slide" type="prismatic"><parent link="tool"/><child link="%s"/><axis xyz="0 %d 0"/><limit lower="0" upper=".04"/></joint>'%(n,n,sign)
        p.write_text(xml+'</robot>');t=np.eye(4);t[2,3]=.13;base=np.eye(4);base[0,3]=1
        return URDFKinematics(p,joint_names=['axis%d'%i for i in range(7)],finger_names=['left_slide','right_slide'],tcp_link='tool',T_link_tcp=t,base=base)
    def test_seventh_joint_tool_base_and_finger_axes(self):
        m=self.fixture();q=np.zeros(7);q[-1]=np.pi/2;t=m.fk(q)
        np.testing.assert_allclose(t[:3,3],[1,.2,.83],atol=1e-12)
        tf=m.transforms(q,[.04,.04]);self.assertAlmostEqual(np.linalg.norm(tf['left'][:3,3]-tf['right'][:3,3]),.08)
        with self.assertRaises(ValueError):m.fk(np.zeros(6))
        candidate=m.solve_candidates(t,q);self.assertTrue(candidate);np.testing.assert_allclose(m.fk(candidate[0]),t,atol=1e-5)
    def test_finger_axes_follow_tcp_rotation(self):
        m=self.fixture();m.T_link_tcp[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        rows=m.binding()['finger_geometry']['joints']
        np.testing.assert_allclose(rows[0]['axis_tcp_increasing_coordinate'],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(rows[1]['axis_tcp_increasing_coordinate'],[-1,0,0],atol=1e-12)
        self.assertEqual(rows[0]['child_link'],'left')

    def test_query_does_not_act_and_budget_is_not_success(self):
        m=self.fixture()
        class Driver:
            control_period=.05;steps=0;count_domain='physics_tick'
            def state(self):return {'joint_positions':[0.]*7,'gripper_qpos':[0.,0.],'eef_pose_W':pose(m.fk(np.zeros(7)))}
            def apply_joint_target(self,q):return False
        d=Driver();a=ArmMotion(d,m,EmptyWorld(m.path.parent),{})
        target=[.01]*7;r=a.dispatch('check_motion',{'commands':[{'arm':0,'joint_positions':target}]})
        self.assertEqual(d.steps,0);self.assertEqual(r['collision']['counter_domain'],'physics_tick')
        self.assertEqual(a.dispatch('joints',{'arm':0,'joint_positions':target})['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(a.dispatch('joints',{'joint_positions':target,'max_steps':1})['status'],'MOTION_LIMIT_EXCEEDED')
        def accept_without_tracking(q):
            d.steps+=1
            return True
        d.apply_joint_target=accept_without_tracking
        receipt=a.dispatch('joints',{'joint_positions':target,'tracking_tolerance_rad':.001,'max_settle_actions':0})
        self.assertEqual(receipt['status'],'TARGET_NOT_REACHED')
        target[-1]=3
        with self.assertRaises(ValueError):a.dispatch('joints',{'joint_positions':target})
    def test_mujoco_mixed_position_orientation_convention(self):
        from types import SimpleNamespace
        from codeworld_robot.robot_assets import export_mujoco_robot
        from scipy.spatial.transform import Rotation
        m=SimpleNamespace(nbody=1,body_parentid=np.array([0]),body_geomadr=np.array([0]),body_geomnum=np.array([0]),
            body_name2id=lambda n:0,body_id2name=lambda i:'mount',site_name2id=lambda n:0,
            site_bodyid=np.array([0]),site_pos=np.array([[.1,0,0]]),
            site_quat=np.array([[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]]))
        d=SimpleNamespace(body_xpos=np.array([[1.,0,0]]),body_xmat=np.eye(3).reshape(1,9))
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        model=export_mujoco_robot(SimpleNamespace(model=m,data=d),joint_names=[],finger_names=[],root_body='mount',
            tcp_site='grip',output=tmp.name,T_W_S=np.eye(4),tcp_rotation_body='mount')
        t=model.fk([]);np.testing.assert_allclose(t[:3,3],[1.1,0,0]);np.testing.assert_allclose(t[:3,:3],np.eye(3))
    def test_package_mesh_resolution(self):
        m=self.fixture();self.assertEqual(mesh_path(m.path,'package://custom/mesh.stl',{'custom':m.path.parent}),m.path.parent/'mesh.stl')

if __name__=='__main__':unittest.main()
