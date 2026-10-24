import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from codeworld_robot.motion import joint_trajectory, densify_joint_path
from codeworld_robot.collision import RobotCollision


class FakeWorld:
    def __init__(self, root, mesh):
        self.root=root;mesh.export(root/'obstacle.glb')
        (root/'object.json').write_text(json.dumps({'role':'rigid_object','geometry':{'local_mesh':'obstacle.glb'}}))
        self.manifest={'objects':{'obstacle':'object.json'}}
    def objects(self):return ['obstacle']
    def state(self):return {'revision':3,'objects':{'obstacle':{'T_W_object':np.eye(4).tolist()}}}
    def _mesh(self, oid, state):return trimesh.load(self.root/'obstacle.glb',force='mesh')


class MotionCollisionTests(unittest.TestCase):
    def test_quintic_limits_and_no_fixed_floor(self):
        frames=joint_trajectory({0:np.zeros(6)},{0:np.ones(6)*.4},.04,2.,8.)
        q=np.vstack([np.zeros(6)]+[f[0] for f in frames])
        self.assertLessEqual(np.max(abs(np.diff(q,axis=0)/.04)),2.+1e-9)
        self.assertLessEqual(np.max(abs(np.diff(q,n=2,axis=0)/.04**2)),8.+1e-9)
        np.testing.assert_allclose(q[-1],.4)
        self.assertLess(len(joint_trajectory({0:np.zeros(6)},{0:np.ones(6)*.001},.004)),60)

    def test_ruckig_joint_limits_and_synchronized_endpoint(self):
        frames=joint_trajectory({0:np.zeros(6),1:np.zeros(6)},
            {0:np.ones(6)*.4,1:np.ones(6)*.8},.04,2.,8.,profile='ruckig')
        for arm,goal in [(0,.4),(1,.8)]:
            q=np.vstack([np.zeros(6)]+[f[arm] for f in frames])
            self.assertLessEqual(np.max(abs(np.diff(q,axis=0)/.04)),2.+1e-8)
            self.assertLessEqual(np.max(abs(np.diff(q,n=2,axis=0)/.04**2)),8.+1e-8)
            np.testing.assert_allclose(q[-1],goal,atol=1e-8)

    def fixture(self, obstacle_position=(.7,0,0), second_base=(0,3,0)):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        # One non-adjacent rotating tip plus a prismatic gripper surface.
        (root/'robot.urdf').write_text('''<robot name="fixture">
        <link name="base_link"/><link name="link1"/>
        <link name="link2"><collision><geometry><box size="0.12 0.12 0.12"/></geometry></collision></link>
        <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/><axis xyz="0 0 1"/></joint>
        <joint name="joint2" type="fixed"><parent link="link1"/><child link="link2"/><origin xyz="0.7 0 0"/></joint>
        </robot>''')
        mesh=trimesh.creation.box([.12,.12,.12]);mesh.apply_translation(obstacle_position)
        world=FakeWorld(root,mesh);b=np.eye(4);b[:3,3]=second_base
        return RobotCollision(root/'robot.urdf',{0:np.eye(4),1:b},world)

    def test_environment_collision_and_explicit_pair_exception(self):
        c=self.fixture();q={0:np.array([0.])};f={0:np.array([0.,0.])}
        hit=c.check_path(q,[q],f)
        self.assertEqual(hit['status'],'COLLISION');self.assertEqual(hit['hits'][0]['second'],'obstacle')
        clear=c.check_path(q,[q],f,allowed_contacts=[['arm0/link2','obstacle']])
        self.assertEqual(clear['status'],'CLEAR')

    def test_interarm_collision_includes_idle_arm(self):
        c=self.fixture(obstacle_position=(10,0,0),second_base=(0,0,0))
        q={0:np.array([0.]),1:np.array([0.])}
        report=c.check_path(q,[{0:np.array([.2])}],{0:[0,0],1:[0,0]})
        self.assertEqual(report['status'],'COLLISION')
        self.assertEqual({report['hits'][0]['first'],report['hits'][0]['second']},{'arm0/link2','arm1/link2'})

    def test_midpath_obstacle_even_when_endpoints_clear(self):
        c=self.fixture();q={0:np.array([-1.])};f={0:np.array([0.,0.])}
        report=c.check_path(q,[{0:np.array([1.])}],f)
        self.assertEqual(report['status'],'COLLISION');self.assertGreater(report['sample'],0)

    def test_nonadjacent_self_collision(self):
        c=self.fixture(obstacle_position=(10,0,0))
        path=Path(c.world.root/'robot.urdf')
        path.write_text('''<robot name="self"><link name="base_link"/>
        <link name="link1"><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
        <link name="link2"/>
        <link name="link3"><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
        <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/><axis xyz="0 0 1"/></joint>
        <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/><axis xyz="0 0 1"/></joint>
        <joint name="joint3" type="revolute"><parent link="link2"/><child link="link3"/><axis xyz="0 0 1"/></joint></robot>''')
        c=RobotCollision(path,{0:np.eye(4)},c.world);q={0:np.zeros(3)}
        report=c.check_path(q,[q],{0:[0,0]})
        self.assertEqual(report['status'],'COLLISION')
        self.assertEqual({report['hits'][0]['first'],report['hits'][0]['second']},{'arm0/link1','arm0/link3'})

    def test_gripper_sweep_checks_intermediate_finger_states(self):
        c=self.fixture(obstacle_position=(.7,.2,0));path=c.world.root/'robot.urdf'
        path.write_text(path.read_text().replace('name="joint2" type="fixed"','name="joint7" type="prismatic"').replace('<origin xyz="0.7 0 0"/>','<origin xyz="0.7 0 0"/><axis xyz="0 1 0"/>'))
        c=RobotCollision(path,{0:np.eye(4)},c.world);q={0:np.zeros(1)}
        report=c.check_path(q,[q]*20,{0:np.zeros(2)},finger_targets={0:np.array([.4,0])})
        self.assertEqual(report['status'],'COLLISION');self.assertGreater(report['sample'],0)

    def test_payload_checked_against_robot_with_explicit_grasp_contact(self):
        c=self.fixture();path=c.world.root/'robot.urdf'
        path.write_text(path.read_text().replace('link2','link6'))
        c=RobotCollision(path,{0:np.eye(4)},c.world);q={0:np.zeros(1)}
        t=np.eye(4);t[0,3]=-.7
        attachments=[{'object_id':'obstacle','arm':0,'T_tcp_object':t.tolist(),'tcp_offset_m':[0,0,0]}]
        report=c.check_path(q,[q],{0:[0,0]},attachments=attachments)
        self.assertEqual(report['status'],'COLLISION')
        self.assertEqual({report['hits'][0]['first'],report['hits'][0]['second']},{'obstacle','arm0/link6'})
        report=c.check_path(q,[q],{0:[0,0]},attachments=attachments,allowed_contacts=[['obstacle','arm0/link6']])
        self.assertEqual(report['status'],'CLEAR')


if __name__=='__main__':unittest.main()
