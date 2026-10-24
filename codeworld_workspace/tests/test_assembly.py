import unittest
import numpy as np
from codeworld_workspace.tests import test_world as rigid_tests

class AssemblyTests(unittest.TestCase):
    setUp=rigid_tests.WorldTests.setUp
    tearDown=rigid_tests.WorldTests.tearDown
    def test_parts_motion_parent_pose_and_branch(self):
        w=self.w;t=np.eye(4);t[0,3]=.4
        w.publish_parts('box',{'body':{'mesh':self.root/'geometry/box.glb'},'drawer':{'mesh':self.root/'geometry/box.glb','T_object_part':t.tolist()}},parameters={'q':0},expected_revision=0)
        branch=w.fork('open');t[1,3]=.2
        branch.update_parts('box',{'drawer':t},parameters={'q':.2},basis='action_prediction',expected_revision=1)
        np.testing.assert_allclose(branch.part_mesh('box','drawer').centroid,[.4,.2,0],atol=1e-7)
        np.testing.assert_allclose(w.part_mesh('box','drawer').centroid,[.4,0,0],atol=1e-7)
        parent=np.eye(4);parent[2,3]=.5;branch.update({'box':parent},expected_revision=2)
        np.testing.assert_allclose(branch.part_mesh('box','drawer').centroid,[.4,.2,.5],atol=1e-7)
        self.assertEqual(branch.parts('box')['parameters']['q'],.2)
        branch.export_scene(self.root/'opened.glb');branch.verify_binding()
        with self.assertRaisesRegex(ValueError,'Stale'):branch.update_parts('box',{'drawer':t},expected_revision=1)
        with self.assertRaises(KeyError):branch.update_parts('box',{'missing':t},expected_revision=3)
        self.assertEqual(branch.state()['revision'],3)
    def test_rotation_and_geometry_replacement(self):
        w=self.w;mesh=self.root/'geometry/box.glb'
        w.publish_parts('box',{'knob':{'mesh':mesh}},expected_revision=0)
        t=np.eye(4);t[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        w.update_parts('box',{'knob':t},parameters={'angle_rad':np.pi/2},expected_revision=1)
        np.testing.assert_allclose(w.mesh('box').extents,[.2,.1,.3],atol=1e-7)
        old=w.state()['objects']['box']['assembly']['parts']['knob']['local_mesh']
        w.publish_parts('box',{'new':{'mesh':mesh}},expected_revision=2)
        self.assertTrue((self.root/old).exists());w.verify_binding()
