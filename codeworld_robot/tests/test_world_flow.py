import copy,json,tempfile,unittest,os
from unittest.mock import patch
from pathlib import Path
import numpy as np
import trimesh
from codeworld_robot.io import atomic,read
from codeworld_robot.flow import enforce
from codeworld_workspace import World
from codeworld_workspace.completion import digest
from codeworld_workspace.flow import validate_program


class WorldFlowTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'CODEWORLD_ROLE':'codex_cli','CODEWORLD_EXECUTOR_ID':'fixture'});self.env.start();self.addCleanup(self.env.stop)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        atomic(self.root/'task/world_flow.json',{'required':True});atomic(self.root/'task/task.json',{'goal':'fixture'})
        atomic(self.root/'task/robot_connection.json',{'io_directory':str(self.root/'robot_io')})
        self.robot={'robot_state_ref':'fixture:0','timestamp_ns':0,'eef_pose_W':{'translation_m':[0,0,1]}}
        atomic(self.root/'robot_io/robot_state.json',self.robot)
        (self.root/'geometry').mkdir();trimesh.creation.box([.1,.2,.3]).export(self.root/'geometry/box.ply')
        atomic(self.root/'objects/box.json',{'geometry':{'local_mesh':'geometry/box.ply'},'declared':{'unknowns':['friction']}})
        T=np.eye(4).tolist();atomic(self.root/'world.json',{'geometry_revision':'fixture','objects':{'box':'objects/box.json'},'bound_files':{}})
        atomic(self.root/'state/current.json',{'revision':0,'history':[],'geometry_revision':'fixture','objects':{'box':{'T_W_object':T,'basis':'observation_estimate','review':{'pose':{'status':'observed'}},'sensor_validation':{'pose_digest':digest(T),'orientation_observed':True,'translation_observed':True}}}})
        self.world=World(self.root)
    def program(self,body):
        p=self.root/'work/compute.py';p.parent.mkdir(exist_ok=True);p.write_text(body);return str(p)
    def decision(self):
        return self.world.compute_decision(self.program("def compute(ctx):\n p=ctx.position('box')\n return {'actions':[{'op':'move_tcp','args':{'pose_W':{'translation_m':(p+np.array([0,0,.2])).tolist(),'quaternion_xyzw':Rotation.from_matrix(ctx.transform('box')[:3,:3]).as_quat().tolist()}},'affected_objects':['box']}],'judgments':{'height':float(p[2]),'on':bool(p[2]>.1)}}\n"),purpose='manipulation',object_ids=['box'])
    def packet(self,r):
        a=r['result']['actions'][0];return {'id':'request','op':a['op'],'args':{**copy.deepcopy(a['args']),'world_decision':{'file':r['decision'],'index':0}},'actor':{'role':'codex_cli','run_id':'fixture'}}
    def test_raw_commands_and_payload_changes_rejected(self):
        with self.assertRaises(ValueError):enforce(self.root,{'op':'move_tcp','args':{},'actor':{'role':'codex_cli'}})
        r=self.decision();p=self.packet(r);enforce(self.root,p)
        p['args']['pose_W']['translation_m'][0]=9
        with self.assertRaises(ValueError):enforce(self.root,p)
    def test_failure_judgment_also_requires_world_computation(self):
        from codeworld_robot.completion import enforce as finish_enforce
        p={'id':'finish','op':'finish','args':{'outcome':'failed','reason':'unsupported'},'actor':{'role':'codex_cli','run_id':'fixture'}}
        with self.assertRaises(ValueError):finish_enforce(self.root,p)
        r=self.world.compute_decision(self.program("def compute(ctx):\n u=ctx.uncertainty('box')\n return {'judgments':{'reason':'unsupported' if u else 'unknown'}}"),purpose='judgment',object_ids=['box'])
        p['args']['judgment']=r['decision'];finish_enforce(self.root,p)
    def test_controller_receipt_is_consumed_through_world_measurement(self):
        r=self.world.compute_decision(self.program("def compute(ctx):\n p=ctx.position('box')\n return {'actions':[{'op':'check_ik','args':{'pose_W':{'translation_m':p.tolist(),'quaternion_xyzw':Rotation.from_matrix(ctx.transform('box')[:3,:3]).as_quat().tolist()}},'affected_objects':[]}]}\n"),purpose='judgment',object_ids=['box']);p=self.packet(r);enforce(self.root,p)
        # A completed read-only controller query, with its actual world decision provenance.
        atomic(self.root/'robot_io/requests/request.json',p)
        atomic(self.root/'robot_io/responses/request.json',{'status':'IK_FOUND','robot_state':self.robot})
        self.world.sync_robot_feedback()
        r=self.world.compute_decision(self.program("def compute(ctx):\n m=ctx.measurement('request')\n return {'judgments':{'reachable':bool(m['status']=='IK_FOUND')}}"),purpose='judgment',object_ids=['box'])
        self.assertTrue(r['result']['judgments']['reachable'])
    def test_receipt_cannot_be_reused_by_other_executor(self):
        p=self.packet(self.decision());p['actor']['run_id']='other'
        with self.assertRaises(ValueError):enforce(self.root,p)
    def test_no_query_and_wrong_relevant_objects_rejected(self):
        with self.assertRaises(ValueError):self.world.compute_decision(self.program("def compute(ctx):\n return {'judgments':{'success':True}}"),purpose='judgment',object_ids=['box'])
        with self.assertRaises(ValueError):self.world.compute_decision(self.program("def compute(ctx):\n s=ctx.proprioception()\n return {'judgments':{'success':True}}"),purpose='judgment',object_ids=['box'])
    def test_decision_code_cannot_read_images_or_external_files(self):
        for code in ["import os", "x=open('image.png')", "x=np.load('depth.npy')", "x=ctx.world.state()"]:
            with self.assertRaises(ValueError):validate_program(code)
    def test_stale_world_and_robot_rejected(self):
        baseline=self.world.state();r=self.decision();self.world.update({'box':np.eye(4)},basis='action_prediction')
        with self.assertRaises(ValueError):enforce(self.root,self.packet(r))
        atomic(self.world.state_path,baseline)
        r=self.decision();atomic(self.root/'robot_io/robot_state.json',{**self.robot,'robot_state_ref':'fixture:1','timestamp_ns':1})
        with self.assertRaises(ValueError):enforce(self.root,self.packet(r))
    def test_cannot_relabel_visual_pose_or_boolean_as_observed(self):
        with self.assertRaises(ValueError):self.world.update({'box':np.eye(4)},basis='observation_estimate',evidence=['image'])
        with self.assertRaises(ValueError):self.world.assess({'box':{'pose':'observed'}},evidence=['image'])
        with self.assertRaises(ValueError):self.world.update_state({'box':{'on':True}},evidence=['image'])
        r=self.decision();self.world.derive_state(r['decision'],'box',{'on':'on'})
        self.assertFalse(self.world.state()['objects']['box']['observed_state']['values']['on'])
    def test_completion_uses_derived_state_and_rejects_changed_dependencies(self):
        self.world.define_completion({'g':{'description':'measured state','conditions':[{'kind':'state','object':'box','field':'on','equals':False}]}})
        r=self.decision();self.world.derive_state(r['decision'],'box',{'on':'on'})
        self.assertEqual(self.world.verify_completion(robot_state=self.robot)['status'],'VERIFIED')
        self.world.assess({'box':{'pose':'pending'}})
        self.assertEqual(self.world.verify_completion(robot_state=self.robot)['status'],'INCONCLUSIVE')
    def test_executed_motion_invalidates_object_and_decision_cannot_replay(self):
        r=self.decision();p=self.packet(r);enforce(self.root,p)
        p2=copy.deepcopy(p);p2['id']='duplicate'
        with self.assertRaises(ValueError):enforce(self.root,p2)
        atomic(self.root/'robot_io/requests/request.json',p)
        new={**self.robot,'robot_state_ref':'fixture:1','timestamp_ns':1};atomic(self.root/'robot_io/robot_state.json',new)
        atomic(self.root/'robot_io/responses/request.json',{'status':'EXECUTED','steps':1,'robot_state':new})
        self.world.sync_robot_feedback()
        self.assertEqual(self.world.state()['objects']['box']['review']['pose']['status'],'pending')
        # Planning may use a pending estimate; it has not become observed truth.
        self.assertTrue(self.decision()['result']['actions'])
    def test_ambiguous_rotation_remains_queryable_as_world_estimate(self):
        s=self.world.state();s['objects']['box']['sensor_validation']['orientation_observed']=False;atomic(self.world.state_path,s)
        result=self.world.compute_decision(self.program("def compute(ctx):\n T=ctx.transform('box')\n return {'judgments':{'turned':bool(T[0,0]<.5)}}"),purpose='judgment',object_ids=['box'])
        self.assertFalse(result['result']['judgments']['turned'])
        self.assertFalse(self.world.state()['objects']['box']['sensor_validation']['orientation_observed'])
        r=self.world.compute_decision(self.program("def compute(ctx):\n u=ctx.uncertainty('box')\n return {'judgments':{'need_observation':bool(u)}}"),purpose='sensing',object_ids=['box'])
        self.assertTrue(r['result']['judgments']['need_observation'])
    def capture(self,z=1.):
        folder=self.root/'robot_io/captures/test';folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'depth.npy',np.full((64,64),z));mask=np.zeros((64,64),bool);mask[27:37,27:37]=True;np.save(self.root/'mask.npy',mask)
        atomic(folder/'capture.json',{'robot_state':self.robot,'views':[{'view_id':'cam','depth_uri':str(folder/'depth.npy'),'depth_convention':'camera_z_m','K':[[100,0,32],[0,100,32],[0,0,1]],'T_W_C':np.eye(4).tolist()}]})
        return str(folder/'capture.json')
    def test_rgbd_numerical_fit_updates_world_and_bad_target_is_rejected(self):
        # Visible front plane at z=1 is consistent with this box centered at z=1.15.
        cap=self.capture();T=np.eye(4);T[2,3]=1.15
        r=self.world.integrate_rgbd(cap,{'box':{'T_W_object':T.tolist(),'masks':{'cam':'mask.npy'}}})
        self.assertEqual(r['objects']['box']['status'],'SURFACE_CONSISTENT')
        self.assertEqual(self.world.state()['objects']['box']['T_W_object'],T.tolist())
        self.assertFalse(self.world.state()['objects']['box']['sensor_validation']['orientation_observed'])
        bad=T.copy();bad[2,3]=2
        r=self.world.integrate_rgbd(cap,{'box':{'T_W_object':bad.tolist(),'masks':{'cam':'mask.npy'}}})
        self.assertEqual(r['objects']['box']['status'],'INCONCLUSIVE')
        self.assertEqual(self.world.state()['objects']['box']['T_W_object'],T.tolist())
        self.assertEqual(self.world.state()['objects']['box']['review']['pose']['status'],'pending')
    def test_insufficient_depth_and_old_capture_do_not_update_pose(self):
        cap=self.capture(float('nan'));T=np.eye(4)
        r=self.world.integrate_rgbd(cap,{'box':{'T_W_object':T.tolist(),'masks':{'cam':'mask.npy'}}})
        self.assertEqual(r['objects']['box']['status'],'INCONCLUSIVE')
        atomic(self.root/'robot_io/robot_state.json',{**self.robot,'robot_state_ref':'fixture:1'})
        with self.assertRaises(ValueError):self.world.integrate_rgbd(cap,{'box':{'T_W_object':T.tolist(),'masks':{'cam':'mask.npy'}}})

if __name__=='__main__':unittest.main()
