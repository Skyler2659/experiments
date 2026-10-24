import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import trimesh
from codeworld_workspace.world import World,atomic

class WorldTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        for d in ['geometry','objects','state','scenarios','observations']: (self.root/d).mkdir()
        trimesh.creation.box(extents=[.1,.2,.3]).export(self.root/'geometry/box.glb')
        for oid,movable in [('box',True),('table',False)]:
            atomic(self.root/f'objects/{oid}.json',{'movable':movable,'geometry':{'local_mesh':'geometry/box.glb'}})
        files={str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['geometry','objects'] for p in (self.root/folder).iterdir()}
        atomic(self.root/'world.json',{'geometry_revision':'test','frame':'W','objects':{x:f'objects/{x}.json' for x in ['box','table']},'bound_files':files})
        atomic(self.root/'state/current.json',{'geometry_revision':'test','revision':0,'objects':{x:{'T_W_object':np.eye(4).tolist()} for x in ['box','table']},'history':[]})
        self.w=World(self.root)
    def tearDown(self): self.tmp.cleanup()
    def test_hypothesis_isolation_and_rotation(self):
        branch=self.w.fork('candidate'); t=np.eye(4);t[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];t[0,3]=.5
        branch.set_hypothesis({'box':t.tolist()},0)
        np.testing.assert_allclose(branch.mesh('box').extents,[.2,.1,.3],atol=1e-7)
        np.testing.assert_allclose(self.w.mesh('box').centroid,[0,0,0],atol=1e-7)
        with self.assertRaisesRegex(ValueError,'Stale'):branch.set_hypothesis({'box':t.tolist()},0)
        with self.assertRaisesRegex(ValueError,'require fork'):self.w.set_hypothesis({'box':t.tolist()},0)
    def test_observation_commit_and_rejection(self):
        (self.root/'sensor.txt').write_text('fixture evidence, not a real robot observation')
        t=np.eye(4);t[2,3]=.2
        report={'observation_id':'test','timestamp':'test-time','estimator':'test_fixture','sensor_files':['sensor.txt'],'frame':'W','poses_W':{'box':t.tolist()}}
        atomic(self.root/'report.json',report);self.w.update_observed(self.root/'report.json',0)
        self.assertEqual(self.w.state()['revision'],1)
        np.testing.assert_allclose(self.w.mesh('box').centroid,[0,0,.2],atol=1e-7)
        with self.assertRaisesRegex(ValueError,'Stale'):self.w.update_observed(self.root/'report.json',0)
        report['poses_W']={'table':t.tolist()};atomic(self.root/'report.json',report)
        self.w.update_observed(self.root/'report.json',1)
        self.assertEqual(self.w.state()['revision'],2)
    def test_binding_and_geometry_query(self):
        result=self.w.nearest_surface('box',[[.2,0,0]])
        self.assertAlmostEqual(result['results'][0]['distance_m'],.15,places=6)
        (self.root/'geometry/box.glb').write_bytes(b'changed')
        self.assertEqual(World(self.root).state()['revision'],0)
        with self.assertRaisesRegex(ValueError,'changed'):self.w.verify_binding()
    def test_invalid_transform_atomicity(self):
        branch=self.w.fork('bad');t=np.eye(4);t[0,0]=2
        with self.assertRaisesRegex(ValueError,'rigid'):branch.set_hypothesis({'box':t.tolist()},0)
        self.assertEqual(branch.state()['revision'],0)

if __name__=='__main__': unittest.main()
