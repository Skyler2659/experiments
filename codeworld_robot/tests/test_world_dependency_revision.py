"""Decision freshness follows physical inputs, not append-only bookkeeping."""
import copy
import hashlib
import unittest
from unittest.mock import patch
from codeworld_robot.io import atomic,read
from codeworld_robot.flow import enforce
from codeworld_robot.tests import test_world_flow
from codeworld_robot.tests import test_world_completion_audit
from codeworld_workspace import flow
from codeworld_workspace.completion import digest
from codeworld_workspace.world import _BOUND_FILE_DIGESTS

class WorldDependencyTests(unittest.TestCase):
    def setUp(self):
        self.f=test_world_flow.WorldFlowTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root;self.world=self.f.world

    def reply(self,receipt,reply,index=0,request_id='request'):
        action=receipt['result']['actions'][index]
        packet={'id':request_id,'op':action['op'],'args':{**copy.deepcopy(action['args']),
            'world_decision':{'file':receipt['decision'],'index':index}},'actor':receipt['actor']}
        enforce(self.root,packet)
        atomic(self.root/f'robot_io/requests/{request_id}.json',packet)
        atomic(self.root/f'robot_io/responses/{request_id}.json',reply)
        self.world.sync_robot_feedback()
        return packet

    def test_readonly_controller_reply_does_not_expire_unexecuted_action(self):
        program=self.f.program("""def compute(ctx):
 p=ctx.position('box')
 pose={'translation_m':p.tolist(),'quaternion_xyzw':[0,0,0,1]}
 return {'actions':[{'op':'check_ik','args':{'pose_W':pose},'affected_objects':[]},
                    {'op':'move_tcp','args':{'pose_W':pose},'affected_objects':['box']}]}
""")
        r=self.world.compute_decision(program,purpose='manipulation',object_ids=['box'])
        self.reply(r,{'status':'IK_FOUND','robot_state':self.f.robot})
        self.assertGreater(self.world.state()['revision'],r['state_revision'])
        action=r['result']['actions'][1]
        enforce(self.root,{'id':'move','op':action['op'],'args':{**action['args'],
            'world_decision':{'file':r['decision'],'index':1}},'actor':r['actor']})

    def test_unqueried_object_and_history_changes_do_not_expire_decision(self):
        r=self.f.decision();s=self.world.state()
        s['objects']['background']=copy.deepcopy(s['objects']['box'])
        s['objects']['background']['T_W_object'][0][3]=12
        s['history'].append({'kind':'unrelated observation'});s['revision']+=1
        atomic(self.world.state_path,s);enforce(self.root,self.f.packet(r))

    def test_new_capture_of_same_pose_does_not_expire_geometry_only_decision(self):
        r=self.f.decision();s=self.world.state()
        s['objects']['box']['sensor_validation']['capture']='another capture'
        s['objects']['box']['review']['pose']['evidence']=['another capture']
        s['revision']+=1;atomic(self.world.state_path,s)
        enforce(self.root,self.f.packet(r))

    def test_queried_input_and_robot_state_still_expire_decision(self):
        r=self.f.decision();s=self.world.state();s['objects']['box']['T_W_object'][0][3]=.2
        atomic(self.world.state_path,s)
        with self.assertRaisesRegex(ValueError,'queried input'):enforce(self.root,self.f.packet(r))
        r=self.f.decision();atomic(self.root/'robot_io/robot_state.json',{**self.f.robot,'robot_state_ref':'new'})
        with self.assertRaisesRegex(ValueError,'robot state'):enforce(self.root,self.f.packet(r))

    def test_dispatch_does_not_repeat_computation(self):
        with patch.object(flow,'run_computation',wraps=flow.run_computation) as compute:
            r=self.f.decision();enforce(self.root,self.f.packet(r))
            self.assertEqual(compute.call_count,1)

    def test_registered_result_still_binds_source_queries_and_parameters(self):
        r=self.f.decision();saved=read(self.root/r['decision']);saved['result']['actions'][0]['args']['pose_W']['translation_m']=[8,8,8]
        atomic(self.root/r['decision'],saved)
        with self.assertRaisesRegex(ValueError,'registered'):enforce(self.root,self.f.packet(r))

    def test_sensing_motion_does_not_invalidate_untouched_objects(self):
        r=self.world.compute_decision(self.f.program("""def compute(ctx):
 p=ctx.position('box')
 return {'actions':[{'op':'move_tcp','args':{'pose_W':{'translation_m':p.tolist(),'quaternion_xyzw':[0,0,0,1]}},'affected_objects':[]}]}
"""),purpose='sensing',object_ids=['box'])
        new={**self.f.robot,'robot_state_ref':'new'}
        atomic(self.root/'robot_io/robot_state.json',self.f.robot)
        # Authorize first, then ingest a completed synthetic motion receipt.
        self.reply(r,{'status':'EXECUTED','steps':1,'robot_state':new})
        self.assertEqual(self.world.state()['objects']['box']['review']['pose']['status'],'observed')

    def test_zero_step_error_preserves_object_but_unknown_error_invalidates_it(self):
        r=self.f.decision();self.reply(r,{'status':'ERROR','steps':0,'robot_state':self.f.robot})
        self.assertIn('sensor_validation',self.world.state()['objects']['box'])
        r=self.f.decision();self.reply(r,{'status':'ERROR','robot_state':self.f.robot},request_id='unknown')
        self.assertNotIn('sensor_validation',self.world.state()['objects']['box'])

    def test_derived_object_fact_survives_unrelated_robot_motion_and_can_be_queried(self):
        r=self.f.decision();self.world.derive_state(r['decision'],'box',{'on':'on'})
        self.world.define_completion({'g':{'description':'fixture state','conditions':[{'kind':'state','object':'box','field':'on','equals':False}]}})
        new={**self.f.robot,'robot_state_ref':'new','timestamp_ns':1};atomic(self.root/'robot_io/robot_state.json',new)
        self.assertEqual(self.world.verify_completion(robot_state=new)['status'],'VERIFIED')
        result=self.world.compute_decision(self.f.program("def compute(ctx):\n return {'judgments':{'off':not ctx.state('box','on')}}"),purpose='judgment',object_ids=['box'])
        self.assertTrue(result['result']['judgments']['off'])
        self.world.assess({'box':{'pose':'pending'}})
        with self.assertRaisesRegex(ValueError,'pending'):self.world.compute_decision(self.f.program("def compute(ctx):\n return {'judgments':{'off':not ctx.state('box','on')}}"),purpose='judgment',object_ids=['box'])

    def test_helper_alias_can_derive_and_reuse_numeric_state(self):
        source="def height(world):\n return float(world.position('box')[2])\ndef compute(ctx):\n return {'judgments':{'height':height(ctx)}}"
        r=self.world.compute_decision(self.f.program(source),purpose='judgment',object_ids=['box'])
        self.world.derive_state(r['decision'],'box',{'height':'height'})
        result=self.world.compute_decision(self.f.program("def compute(ctx):\n return {'judgments':{'height':ctx.state('box','height')}}"),purpose='judgment',object_ids=['box'])
        self.assertAlmostEqual(result['result']['judgments']['height'],0.)

    def test_surface_fit_transform_height_can_be_derived_without_uniqueness_diagnostics(self):
        s=self.world.state();s['objects']['box']['sensor_validation'].update(
            orientation_observed=None,translation_observed=None,
            orientation_observed_axes=None,translation_observed_axes=None,
            report={'observability':{'status':'NOT_EVALUATED'}})
        atomic(self.world.state_path,s)
        r=self.world.compute_decision(self.f.program("def compute(ctx):\n T=ctx.transform('box')\n return {'judgments':{'height':float(T[2,3])}}"),purpose='judgment',object_ids=['box'])
        self.world.derive_state(r['decision'],'box',{'height':'height'})
        result=self.world.compute_decision(self.f.program("def compute(ctx):\n return {'judgments':{'height':ctx.state('box','height')}}"),purpose='judgment',object_ids=['box'])
        self.assertAlmostEqual(result['result']['judgments']['height'],0.)

    def test_known_orientation_ambiguity_does_not_certify_absolute_angle(self):
        s=self.world.state();s['objects']['box']['sensor_validation'].update(
            orientation_observed=False,orientation_observed_axes=[True,True,False],
            orientation_geometry_equivalent_axes=[False,False,True])
        atomic(self.world.state_path,s)
        r=self.world.compute_decision(self.f.program("def compute(ctx):\n T=ctx.transform('box')\n return {'judgments':{'yaw':float(Rotation.from_matrix(T[:3,:3]).as_euler('xyz')[2])}}"),purpose='judgment',object_ids=['box'])
        self.world.derive_state(r['decision'],'box',{'yaw':'yaw'})
        with self.assertRaisesRegex(ValueError,'unsupported'):
            self.world.compute_decision(self.f.program("def compute(ctx):\n return {'judgments':{'yaw':ctx.state('box','yaw')}}"),purpose='judgment',object_ids=['box'])

    def test_height_fact_does_not_inherit_an_unrelated_orientation_requirement(self):
        s=self.world.state();s['objects']['box']['sensor_validation']['orientation_observed']=None
        atomic(self.world.state_path,s)
        r=self.f.decision();self.world.derive_state(r['decision'],'box',{'height':'height'})
        result=self.world.compute_decision(self.f.program("def compute(ctx):\n return {'judgments':{'height':ctx.state('box','height')}}"),purpose='judgment',object_ids=['box'])
        self.assertAlmostEqual(result['result']['judgments']['height'],0.)

    def test_prediction_is_queryable_but_cannot_complete_a_task(self):
        self.world.define_completion({'g':{'description':'fixture state','conditions':[{'kind':'state','object':'box','field':'on','equals':False}]}})
        self.world.update({'box':self.world.state()['objects']['box']['T_W_object']},basis='action_prediction')
        r=self.f.decision();self.world.derive_state(r['decision'],'box',{'on':'on'})
        self.assertEqual(self.world.verify_completion(robot_state=self.f.robot)['status'],'INCONCLUSIVE')

    def test_immutable_asset_hash_is_cached_but_detects_changed_file(self):
        path=self.root/'geometry/box.ply';expected=hashlib.sha256(path.read_bytes()).hexdigest()
        self.world.manifest['bound_files']={'geometry/box.ply':expected};_BOUND_FILE_DIGESTS.clear()
        from codeworld_workspace import world as module
        with patch.object(module.hashlib,'sha256',wraps=hashlib.sha256) as hashing:
            self.world.verify_binding();self.world.verify_binding();self.assertEqual(hashing.call_count,1)
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'changed'):self.world.verify_binding()
            self.assertEqual(hashing.call_count,2)

class CompletionDegreeTests(unittest.TestCase):
    def setUp(self):
        self.f=test_world_completion_audit.WorldCompletionAuditTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
    def test_geometrically_equivalent_rotation_does_not_block_bounds_goal(self):
        self.f.geometry_goal();s=self.f.world.state();stamp=s['objects']['box']['sensor_validation']
        stamp.update(orientation_observed=False,orientation_observed_axes=[True,True,False],orientation_geometry_equivalent_axes=[False,False,True])
        atomic(self.f.world.state_path,s);self.assertEqual(self.f.verify(),'VERIFIED')
    def test_ordinary_surface_fit_does_not_require_unique_six_dof_certificate(self):
        self.f.geometry_goal();s=self.f.world.state();s['objects']['box']['sensor_validation'].update(translation_observed=None,orientation_observed=None)
        atomic(self.f.world.state_path,s);self.assertEqual(self.f.verify(),'VERIFIED')
    def test_height_goal_ignores_unobservable_planar_translation(self):
        self.f.geometry_goal();(self.f.root/'task/completion.json').unlink()
        self.f.world.define_completion({'g':{'description':'same height','conditions':[{'kind':'relative_center','object':'box','reference':'target','axis':[0,0,1],'frame':'W','min':-.01,'max':.01}]}})
        s=self.f.world.state();s['objects']['box']['sensor_validation'].update(translation_observed=False,translation_observed_axes=[False,False,True],orientation_observed=False)
        atomic(self.f.world.state_path,s);self.assertEqual(self.f.verify(),'VERIFIED')
    def test_loose_fit_threshold_cannot_certify_a_precise_goal(self):
        self.f.geometry_goal();s=self.f.world.state();s['objects']['box']['sensor_validation'].update(surface_error_m=.04,acceptance_tolerance_m=.1)
        atomic(self.f.world.state_path,s);self.assertEqual(self.f.verify(),'INCONCLUSIVE')

if __name__=='__main__':unittest.main()
