"""Offline regressions for physical completion evidence and fresh CLI episodes."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from codeworld_robot.io import atomic,read
from codeworld_robot.tests import test_world_flow
from codeworld_workspace.completion import digest
from codeworld_robot.fresh_state import reset_episode_evidence


class WorldCompletionAuditTests(unittest.TestCase):
    setUp=test_world_flow.WorldFlowTests.setUp
    program=test_world_flow.WorldFlowTests.program

    def geometry_goal(self):
        cap='robot_io/captures/current/capture.json'
        atomic(self.root/cap,{'robot_state':self.robot,'views':[{'view_id':'fixture'}]})
        s=self.world.state();row=s['objects']['box']
        row['review']['pose'].update(revision=0,evidence=[cap])
        row['sensor_validation']['translation_observed']=True
        target=copy.deepcopy(row);target['T_W_object'][0][3]=.4
        target['sensor_validation']['pose_digest']=digest(target['T_W_object'])
        s['objects']['target']=target;atomic(self.world.state_path,s)
        manifest=read(self.root/'world.json');manifest['objects']['target']='objects/box.json'
        atomic(self.root/'world.json',manifest);self.world.manifest=manifest
        self.world.define_completion({'g':{'description':'object extent near target', 'conditions':[
            {'kind':'bounds_gap','object':'box','reference':'target','axis':[1,0,0],
             'frame':'W','object_anchor':'max','reference_anchor':'min','min':-.31,'max':-.29}]}})

    def verify(self):
        return self.world.verify_completion(robot_state=self.robot)['status']

    def state_claim(self,queries,dependencies=None):
        self.world.sync_robot_feedback();s=self.world.state();row=s['objects']['box']
        d={'robot_state_ref':{'robot_state_ref':self.robot['robot_state_ref'],'timestamp_ns':0},
           'queries':queries,'dependencies':dependencies or {}}
        row['observed_state']={'values':{'on':True},'fields':{'on':{
            'status':'observed','derivation':d,'evidence':['decisions/fixture.json']}}}
        atomic(self.world.state_path,s)
        self.world.define_completion({'g':{'description':'physical state', 'conditions':[
            {'kind':'state','object':'box','field':'on','equals':True}]}})

    def test_orientation_unknown_cannot_verify_geometry_relation(self):
        self.geometry_goal();self.assertEqual(self.verify(),'VERIFIED')
        s=self.world.state();s['objects']['box']['sensor_validation']['orientation_observed']=False
        atomic(self.world.state_path,s);self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_translation_unknown_cannot_verify_geometry_relation(self):
        self.geometry_goal();s=self.world.state()
        s['objects']['box']['sensor_validation']['translation_observed']=False
        atomic(self.world.state_path,s);self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_only_validated_assembly_can_support_geometry_completion(self):
        self.geometry_goal();s=self.world.state();row=s['objects']['box']
        mesh=self.root/'geometry/box.ply'
        assembly={'basis':'action_prediction','parts':{'body':{
            'local_mesh':'geometry/box.ply','sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),
            'T_object_part':row['T_W_object']}}}
        row['assembly']=assembly;atomic(self.world.state_path,s)
        self.assertEqual(self.verify(),'INCONCLUSIVE')
        # Accepted RGB-D validation is the authority; a proposal's old basis label is not.
        s=self.world.state();s['objects']['box']['sensor_validation']['assembly_digest']=digest(assembly)
        atomic(self.world.state_path,s);self.assertEqual(self.verify(),'VERIFIED')

    def test_metadata_or_uncertainty_does_not_prove_physical_state(self):
        self.state_claim([{'op':'metadata','object':'box'}])
        self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_robot_proprioception_does_not_prove_unrelated_object_state(self):
        self.state_claim([{'op':'metadata','object':'box'},{'op':'proprioception'}])
        self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_pending_geometry_dependency_does_not_prove_state(self):
        self.world.sync_robot_feedback();s=self.world.state();row=s['objects']['box']
        row.pop('sensor_validation');row['review']['pose']['status']='pending';atomic(self.world.state_path,s)
        deps={'box':{'pose':digest(row['T_W_object']),'validation':digest(None),'review':digest(row['review']['pose'])}}
        self.state_claim([{'op':'position','object':'box'}],deps)
        self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_measurement_must_match_current_robot_state(self):
        s=self.world.state();s['measurements']={'lift':{'object_id':'box','robot_state_ref':{'robot_state_ref':'old'},'value':{'status':'LIFT_EVIDENCE'}}}
        atomic(self.world.state_path,s);self.state_claim([{'op':'measurement','id':'lift','object':'box'}])
        self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_current_measurement_can_support_state_without_full_pose(self):
        s=self.world.state();s['objects']['box'].pop('sensor_validation')
        s['objects']['box']['review']['pose']['status']='pending'
        s['measurements']={'lift':{'object_id':'box','robot_state_ref':{'robot_state_ref':'fixture:0','timestamp_ns':0},'value':{'status':'LIFT_EVIDENCE'}}}
        atomic(self.world.state_path,s);self.state_claim([{'op':'measurement','id':'lift','object':'box'}])
        self.assertEqual(self.verify(),'VERIFIED')

    def test_missing_query_provenance_cannot_verify_a_legacy_derived_boolean(self):
        self.state_claim([]);self.assertEqual(self.verify(),'INCONCLUSIVE')

    def test_fresh_episode_drops_old_observations_and_field_derivations(self):
        self.state_claim([{'op':'position','object':'box'}]);s=self.world.state()
        s['runtime']={'robot_state':self.robot};s['measurements']={'old':{}};atomic(self.world.state_path,s)
        atomic(self.root/'decisions/old.json',{});atomic(self.root/'work/completion/old.json',{})
        reset_episode_evidence(self.root);fresh=self.world.state()
        self.assertNotIn('runtime',fresh);self.assertNotIn('measurements',fresh)
        self.assertNotIn('sensor_validation',fresh['objects']['box'])
        self.assertEqual(fresh['objects']['box']['observed_state']['fields']['on']['status'],'pending')
        self.assertFalse((self.root/'robot_io').exists());self.assertFalse((self.root/'decisions').exists())

    def test_cli_refuses_a_real_world_without_strict_protocol(self):
        from codeworld_robot.agent_cli import policy
        atomic(self.root/'task/executor.json',{'status':'UNASSIGNED'})
        (self.root/'task/world_flow.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'fresh strict CodeWorld'):
            policy({'episode':str(self.root),'python':sys.executable},'fixture','ACTIVE',claim=True)
        self.assertEqual(read(self.root/'task/executor.json')['status'],'UNASSIGNED')

if __name__=='__main__':unittest.main()
