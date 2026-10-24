"""Regression cases for nominal queries, stale raw sources, and feedback recovery."""
import copy
import unittest
import numpy as np
import trimesh
from codeworld_robot.io import atomic,read
from codeworld_robot.flow import enforce
from codeworld_robot.tests import test_world_flow
from codeworld_workspace.flow import validate_program
from codeworld_workspace.provenance import analyze_program

class WorldProvenanceAuditTests(unittest.TestCase):
    setUp=test_world_flow.WorldFlowTests.setUp
    program=test_world_flow.WorldFlowTests.program
    decision=test_world_flow.WorldFlowTests.decision
    packet=test_world_flow.WorldFlowTests.packet

    def compute(self,source,purpose='judgment',objects=None):
        return self.world.compute_decision(self.program(source),purpose=purpose,object_ids=['box'] if objects is None else objects)

    def test_constant_judgment_is_diagnostic_but_cannot_become_measured_state(self):
        for query in ["ctx.metadata('box')", "ctx.position('box')", "ctx.uncertainty('box')"]:
            source="def compute(ctx):\n ignored="+query+"\n return {'judgments':{'on':True}}"
            r=self.compute(source)
            self.assertTrue(analyze_program(source,r['result'])['warnings'])
            with self.assertRaises(ValueError):self.world.derive_state(r['decision'],'box',{'on':'on'})

    def test_one_valid_output_does_not_launder_constant_other_output(self):
        r=self.compute("def compute(ctx):\n p=ctx.position('box')\n return {'judgments':{'height':float(p[2]),'on':True}}")
        with self.assertRaises(ValueError):self.world.derive_state(r['decision'],'box',{'on':'on'})

    def test_metadata_cannot_be_laundered_into_physical_state(self):
        r=self.compute("def compute(ctx):\n m=ctx.metadata('box')\n return {'judgments':{'on':bool(m)}}")
        with self.assertRaisesRegex(ValueError,'Physical state requires'):
            self.world.derive_state(r['decision'],'box',{'on':'on'})

    def test_constant_action_parameters_are_recorded_as_diagnostics(self):
        source="def compute(ctx):\n ignored=ctx.position('box')\n return {'actions':[{'op':'move_tcp','args':{'pose_W':{'translation_m':[1,2,3]}},'affected_objects':['box']}]}"
        r=self.compute(source)
        self.assertTrue(analyze_program(source,r['result'])['warnings'])

    def test_world_based_budget_does_not_hide_constant_target_diagnostic(self):
        source="def compute(ctx):\n p=ctx.position('box')\n return {'actions':[{'op':'move_tcp','args':{'max_steps':int(p[0])+150,'pose_W':{'translation_m':[1,2,3],'quaternion_xyzw':[0,0,0,1]}},'affected_objects':['box']}]}"
        r=self.compute(source)
        self.assertTrue(any('pose_W.translation_m' in w for w in analyze_program(source,r['result'])['warnings']))

    def test_fixed_tool_orientation_and_gripper_mode_are_allowed(self):
        r=self.compute("def compute(ctx):\n p=ctx.position('box')\n return {'actions':[{'op':'move_tcp','args':{'pose_W':{'translation_m':p.tolist(),'quaternion_xyzw':[0,0,0,1]}},'affected_objects':['box']},{'op':'gripper','args':{'mode':'close'},'affected_objects':['box']}]}" )
        self.assertEqual(r['result']['actions'][1]['args']['mode'],'close')

    def test_zero_multiplication_is_a_diagnostic_not_a_semantic_proof(self):
        r=self.compute("def compute(ctx):\n p=ctx.position('box')\n return {'judgments':{'height':float(p[2])*0+1.2}}")
        diagnostics=analyze_program(r['source'],r['result'])
        self.assertTrue(diagnostics['warnings'])
        self.assertNotIn('height',diagnostics['constant_judgments'])

    def test_conditional_and_helper_computations_retain_dependencies(self):
        r=self.compute("def high(p):\n return float(p[2])>.1\ndef compute(ctx):\n p=ctx.position('box')\n return {'judgments':{'on':high(p),'reason':'high' if high(p) else 'low'}}")
        self.assertFalse(r['result']['judgments']['on'])
        self.world.derive_state(r['decision'],'box',{'on':'on'})

    def test_pure_context_helper_and_comprehension_are_supported(self):
        r=self.compute("def height(ctx):\n return float(ctx.position('box')[2])\ndef compute(ctx):\n p=ctx.position('box')\n return {'judgments':{'height':height(ctx)},'actions':[{'op':'gripper','args':{'mode':'close'},'affected_objects':['box']} for i in range(2)]}")
        self.assertEqual(len(r['result']['actions']),2)
        self.assertAlmostEqual(r['result']['judgments']['height'],0)
        self.world.derive_state(r['decision'],'box',{'height':'height'})

    def test_helper_context_parameter_name_does_not_restrict_derived_state(self):
        source="def height(world):\n return float(world.position('box')[2])\ndef compute(ctx):\n return {'judgments':{'height':height(ctx)}}"
        r=self.compute(source)
        self.assertNotIn('height',analyze_program(source,r['result'])['constant_judgments'])
        self.world.derive_state(r['decision'],'box',{'height':'height'})

    def test_metadata_excludes_stale_initial_world_coordinates(self):
        p=self.root/'objects/box.json';d=read(p);d.update(initial_T_W_object=np.eye(4).tolist(),source_nodes=[{'position':[1,2,3]}]);atomic(p,d)
        r=self.compute("def compute(ctx):\n m=ctx.metadata('box')\n return {'judgments':{'metadata':m}}")
        self.assertNotIn('initial_T_W_object',r['result']['judgments']['metadata'])
        self.assertNotIn('source_nodes',r['result']['judgments']['metadata'])

    def test_position_returns_mesh_center_not_root_transform_delta(self):
        mesh=trimesh.creation.box([.1,.2,.3]);mesh.apply_translation([.5,.2,.3]);mesh.export(self.root/'geometry/box.ply')
        r=self.compute("def compute(ctx):\n return {'judgments':{'center':ctx.position('box').tolist()}}")
        np.testing.assert_allclose(r['result']['judgments']['center'],[.5,.2,.3],atol=1e-7)

    def test_ambiguous_translation_remains_explicit_for_planning(self):
        s=self.world.state();s['objects']['box']['sensor_validation']['translation_observed']=False;atomic(self.world.state_path,s)
        r=self.compute("def compute(ctx):\n p=ctx.position('box')\n u=ctx.uncertainty('box')\n return {'judgments':{'estimate':p.tolist(),'uncertainty':u}}")
        self.assertFalse(self.world.state()['objects']['box']['sensor_validation']['translation_observed'])
        self.assertEqual(len(r['result']['judgments']['estimate']),3)

    def test_rejected_bad_receipt_does_not_poison_subsequent_sync(self):
        atomic(self.root/'robot_io/requests/bad.json',{'id':'bad','op':'move_tcp','args':{'world_decision':{'file':'decisions/missing.json','index':0}}})
        atomic(self.root/'robot_io/responses/bad.json',{'status':'ERROR','error':'bad ref'})
        self.world.sync_robot_feedback();self.world.sync_robot_feedback()
        self.assertIn('bad.json',self.world.state()['runtime']['rejected_feedback'])
        enforce(self.root,self.packet(self.decision()))

    def test_sensing_motion_preserves_unaffected_scene_evidence(self):
        r=self.compute("def compute(ctx):\n s=ctx.proprioception()\n return {'actions':[{'op':'move_tcp','args':{'pose_W':s['eef_pose_W']},'affected_objects':[]}]}" ,purpose='sensing',objects=[])
        packet=self.packet(r);enforce(self.root,packet)
        atomic(self.root/'robot_io/requests/request.json',packet)
        atomic(self.root/'robot_io/responses/request.json',{'status':'EXECUTED','steps':1,'robot_state':self.robot})
        self.world.sync_robot_feedback()
        self.assertIn('sensor_validation',self.world.state()['objects']['box'])

    def test_other_actor_cannot_derive_state(self):
        r=self.decision();p=self.root/r['decision'];d=read(p);d['actor']['run_id']='other';atomic(p,d)
        with self.assertRaisesRegex(ValueError,'another executor'):self.world.derive_state(r['decision'],'box',{'on':'on'})

if __name__=='__main__':unittest.main()
