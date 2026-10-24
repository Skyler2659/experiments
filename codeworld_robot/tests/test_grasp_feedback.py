import json,tempfile,unittest
from pathlib import Path
import numpy as np
import trimesh
from codeworld_workspace.grasp import check_grasp_geometry
from codeworld_robot.grasp import assess_grasp


class BoxWorld:
    manifest={'geometry_revision':'box_fixture'}
    def __init__(self):self.box=trimesh.creation.box([.06,.12,.10])
    def state(self):return {'revision':3,'geometry_revision':'box_fixture'}
    def mesh(self,oid):return self.box


class GraspGeometryTests(unittest.TestCase):
    def candidate(self,**kw):
        return dict(center_W=[0,0,0],closing_axis_W=[1,0,0],approach_axis_W=[0,0,-1],max_opening_m=.08,finger_depth_m=.02,finger_pad_height_m=.02,**kw)
    def test_clipped_triangle_band_and_orientation_change(self):
        w=BoxWorld();args=self.candidate();r=check_grasp_geometry(w,'box',**args)
        self.assertEqual(r['status'],'WIDTH_COMPATIBLE');self.assertAlmostEqual(r['span_m'],.06)
        args['closing_axis_W']=[0,1,0]
        self.assertEqual(check_grasp_geometry(w,'box',**args)['status'],'INCOMPATIBLE')
    def test_offset_grasp_rejected_despite_small_span(self):
        args=self.candidate();args['center_W']=[.02,0,0]
        self.assertEqual(check_grasp_geometry(BoxWorld(),'box',**args)['status'],'INCOMPATIBLE')
    def test_no_surface_stale_and_invalid_axes(self):
        args=self.candidate();args['center_W']=[0,0,1]
        self.assertEqual(check_grasp_geometry(BoxWorld(),'box',**args)['status'],'INCONCLUSIVE')
        with self.assertRaisesRegex(ValueError,'Stale'):check_grasp_geometry(BoxWorld(),'box',**self.candidate(expected_revision=2))
        args=self.candidate();args['approach_axis_W']=[1,0,0]
        with self.assertRaises(ValueError):check_grasp_geometry(BoxWorld(),'box',**args)


class LiftFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'work').mkdir()
    def capture(self,name,object_z,tcp_z,*,camera_shift=0,views=2,closed=True,duplicate_camera=False):
        folder=self.root/'robot_io/captures'/name;folder.mkdir(parents=True);rows=[];masks={}
        for i in range(views):
            cx=0 if duplicate_camera else i*.10;T=np.eye(4);T[0,3]=cx+camera_shift
            center_x=80-round(80*cx/object_z);mask=np.zeros((160,160),bool);mask[70:90,center_x-10:center_x+10]=True
            depth=np.full((160,160),object_z);np.save(folder/f'v{i}_depth.npy',depth);mp=self.root/'work'/f'{name}_v{i}.npy';np.save(mp,mask)
            rows.append({'view_id':f'v{i}','depth_uri':f'v{i}_depth.npy','depth_convention':'camera_z_m','K':[[80,0,80],[0,80,80],[0,0,1]],'T_W_C':T.tolist()});masks[f'v{i}']=str(mp.relative_to(self.root))
        state={'arm':0,'robot_state_ref':name,'last_gripper_command':'CLOSE' if closed else 'OPEN','eef_pose_W':{'translation_m':[0,0,tcp_z],'quaternion_xyzw':[0,0,0,1]}}
        p=folder/'capture.json';p.write_text(json.dumps({'robot_state':state,'views':rows}));return str(p.relative_to(self.root)),masks
    def assess(self,a,b,**kw):
        return assess_grasp(self.root,before_capture=a[0],after_capture=b[0],before_masks=a[1],after_masks=b[1],object_id='test_object',**kw)
    def test_object_follows_measured_tcp_without_side_effects(self):
        a=self.capture('before',1.,1.2);b=self.capture('after',1.04,1.24)
        files={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        r=self.assess(a,b);self.assertEqual(r['status'],'LIFT_EVIDENCE');self.assertFalse(r['holding_certified'])
        self.assertEqual(files,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
    def test_commanded_lift_but_stationary_object_is_not_held(self):
        a=self.capture('before',1.,1.2);b=self.capture('after',1.,1.24)
        self.assertEqual(self.assess(a,b)['status'],'NOT_LIFTED')
    def test_one_view_and_open_fingers_cannot_assert_lift(self):
        a=self.capture('before',1.,1.2,views=1);b=self.capture('after',1.04,1.24,views=1)
        self.assertEqual(self.assess(a,b)['status'],'INCONCLUSIVE')
        a=self.capture('before_open',1.,1.2);b=self.capture('after_open',1.04,1.24,closed=False)
        self.assertEqual(self.assess(a,b)['status'],'INCONCLUSIVE')
    def test_moving_camera_and_duplicate_view_rejected(self):
        a=self.capture('before',1.,1.2);b=self.capture('after',1.04,1.24,camera_shift=.1)
        self.assertEqual(self.assess(a,b)['status'],'INCONCLUSIVE')
        a=self.capture('before_dup',1.,1.2,duplicate_camera=True);b=self.capture('after_dup',1.04,1.24,duplicate_camera=True)
        self.assertEqual(self.assess(a,b)['status'],'INCONCLUSIVE')
    def test_receipt_for_backend_without_last_command_and_reopen(self):
        a=self.capture('before:10',1.,1.2);b=self.capture('after:20',1.04,1.24)
        for capture in [a,b]:
            p=self.root/capture[0];d=json.loads(p.read_text());del d['robot_state']['last_gripper_command'];p.write_text(json.dumps(d))
        req=self.root/'robot_io/requests';res=self.root/'robot_io/responses';req.mkdir();res.mkdir()
        (req/'closed.json').write_text(json.dumps({'id':'closed','op':'gripper','args':{'arm':0,'mode':'CLOSE'}}))
        (res/'closed.json').write_text(json.dumps({'status':'EXECUTED','robot_state':{'control_steps':8}}))
        self.assertEqual(self.assess(a,b,closure_request='robot_io/requests/closed.json')['status'],'LIFT_EVIDENCE')
        (req/'opened.json').write_text(json.dumps({'id':'opened','op':'gripper','args':{'arm':0,'mode':'OPEN'}}))
        (res/'opened.json').write_text(json.dumps({'status':'EXECUTED','robot_state':{'control_steps':15}}))
        self.assertEqual(self.assess(a,b,closure_request='robot_io/requests/closed.json')['status'],'INCONCLUSIVE')

    def test_no_probe_and_occluded_depth_are_inconclusive(self):
        a=self.capture('before',1.,1.2);b=self.capture('after',1.04,1.2)
        self.assertEqual(self.assess(a,b)['status'],'INCONCLUSIVE')
        with self.assertRaises(ValueError):self.assess(a,a)
        with self.assertRaises(ValueError):self.assess(a,b,min_positive_views=1)
        with self.assertRaises(ValueError):assess_grasp(self.root,before_capture='/tmp/elsewhere',after_capture=b[0],before_masks=a[1],after_masks=b[1],object_id='test_object')

if __name__=='__main__':unittest.main()
