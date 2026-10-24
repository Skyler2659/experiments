import numpy as np
import trimesh
from codeworld_robot.tests.test_motion_collision import MotionCollisionTests
from codeworld_robot.collision import RobotCollision

class PartCollisionTests(MotionCollisionTests):
    def part_fixture(self, body_x, door_x):
        c=self.fixture();path=c.world.root/'robot.urdf';path.write_text(path.read_text().replace('link2','link6'))
        meshes={}
        for pid,x in [('body',body_x),('door',door_x)]:
            m=trimesh.creation.box([.12,.12,.12]);m.apply_translation([x,0,0]);meshes[pid]=m
        state={'revision':4,'objects':{'obstacle':{'T_W_object':np.eye(4).tolist(),'assembly':{'parts':{pid:{'T_object_part':np.eye(4).tolist()} for pid in meshes}}}}}
        c.world.state=lambda:state;c.world._part_mesh=lambda oid,pid,s:meshes[pid].copy()
        c.world._mesh=lambda oid,s:trimesh.util.concatenate(list(meshes.values()))
        return RobotCollision(path,{0:np.eye(4)},c.world)
    def test_part_exception_does_not_hide_fixed_body(self):
        c=self.part_fixture(.7,.7);q={0:np.zeros(1)}
        r=c.check_path(q,[q],{0:[0,0]},allowed_contacts=[['arm0/link6','obstacle::door']])
        self.assertEqual(r['status'],'COLLISION');self.assertTrue(any(h['second']=='obstacle::body' for h in r['hits']))
    def test_part_payload_moves_only_selected_part(self):
        c=self.part_fixture(2,.7);q={0:np.zeros(1)};t=np.eye(4);t[0,3]=10
        a=[{'object_id':'obstacle','part_id':'door','arm':0,'T_tcp_part':t.tolist(),'tcp_offset_m':[0,0,0]}]
        self.assertEqual(c.check_path(q,[q],{0:[0,0]})['status'],'COLLISION')
        self.assertEqual(c.check_path(q,[q],{0:[0,0]},attachments=a)['status'],'CLEAR')
        c=self.part_fixture(.7,2)
        r=c.check_path(q,[q],{0:[0,0]},attachments=a)
        self.assertEqual(r['status'],'COLLISION');self.assertTrue(any(h['second']=='obstacle::body' for h in r['hits']))
