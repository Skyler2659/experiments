"""Offline regression checks for controller receipt and transport boundaries."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from codeworld_robot import client
from codeworld_robot.flow import enforce
from codeworld_robot.completion import enforce as finish_enforce
from codeworld_robot.io import atomic, read
from codeworld_robot.multiarm import resolve_commands
from codeworld_robot.tests import test_world_flow


class WorldDispatchAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_world_flow.WorldFlowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def test_editing_receipt_and_request_together_is_rejected(self):
        decision = self.fixture.decision()
        receipt = read(self.root / decision['decision'])
        receipt['result']['actions'][0]['args']['pose_W']['translation_m'] = [8, 7, 6]
        atomic(self.root / decision['decision'], receipt)
        decision['result'] = receipt['result']
        with self.assertRaises(ValueError):
            enforce(self.root, self.fixture.packet(decision))

    def test_removing_actual_query_history_is_rejected(self):
        decision = self.fixture.decision()
        receipt = read(self.root / decision['decision'])
        receipt['queries'] = []
        atomic(self.root / decision['decision'], receipt)
        with self.assertRaises(ValueError):
            enforce(self.root, self.fixture.packet(decision))

    def test_authorization_retains_original_receipt_after_artifact_edit(self):
        decision = self.fixture.decision()
        packet = self.fixture.packet(decision)
        enforce(self.root, packet)
        original = read(self.root / decision['decision'])
        modified = copy.deepcopy(original)
        modified['result']['actions'][0]['affected_objects'] = []
        modified['queried_objects'] = []
        atomic(self.root / decision['decision'], modified)
        ledger = read(self.root / 'task/world_decision_uses.json')
        saved = ledger[original['id'] + ':0']
        self.assertEqual(saved['request_id'], packet['id'])
        self.assertEqual(saved['receipt'], original)
        atomic(self.root / 'robot_io/requests/request.json', packet)
        atomic(self.root / 'robot_io/responses/request.json',
               {'status': 'EXECUTED', 'steps': 1, 'robot_state': self.fixture.robot})
        self.fixture.world.sync_robot_feedback()
        self.assertEqual(self.fixture.world.state()['objects']['box']['review']['pose']['status'],
                         'pending')

    def exchange(self, op, result, **args):
        """Simulate a completed file transport; no backend or robot is started."""
        request_id = 'dispatchaudit'

        def publish_request(path, packet):
            atomic(path, packet)
            enforce(self.root, packet)
            atomic(self.root / 'robot_io/responses' / (request_id + '.json'), result)

        with patch.object(client.uuid, 'uuid4', return_value=SimpleNamespace(hex=request_id)):
            with patch.object(client, 'atomic', side_effect=publish_request):
                return client.Robot(self.root).call(op, **args)

    def test_strict_controller_reply_returns_the_published_world_measurement(self):
        decision = self.fixture.decision()
        action = decision['result']['actions'][0]
        reply = {'status': 'ARRIVED', 'steps': 1,
                 'position_error_m': .001, 'robot_state': self.fixture.robot}
        result = self.exchange(action['op'], reply, **copy.deepcopy(action['args']),
                               world_decision={'file': decision['decision'], 'index': 0})
        self.assertEqual(result['status'], 'MEASUREMENT_IN_WORLD')
        self.assertNotIn('robot_state', result)
        self.assertNotIn('position_error_m', result)
        measurement = self.fixture.world.state()['measurements'][result['measurement_id']]
        self.assertEqual(measurement['value'], reply)
        self.assertEqual(result['value'], measurement['value'])
        self.assertEqual(result['state_revision'], self.fixture.world.state()['revision'])

    def test_observation_capture_remains_available_to_perception(self):
        reply = {'status': 'OBSERVED', 'capture': 'robot_io/captures/test/capture.json'}
        self.assertEqual(self.exchange('observe', reply), reply)

    def test_partial_execution_error_is_a_world_measurement(self):
        decision = self.fixture.decision()
        action = decision['result']['actions'][0]
        reply = {'status': 'ERROR', 'error': 'stopped after partial motion',
                 'robot_state': self.fixture.robot}
        result = self.exchange(action['op'], reply, **copy.deepcopy(action['args']),
                               world_decision={'file': decision['decision'], 'index': 0})
        self.assertEqual(result['status'], 'MEASUREMENT_IN_WORLD')
        state = self.fixture.world.state()
        self.assertEqual(state['measurements'][result['measurement_id']]['value']['status'], 'ERROR')
        self.assertEqual(result['value']['status'], 'ERROR')
        self.assertNotIn('sensor_validation', state['objects']['box'])

    def test_system_termination_needs_no_physical_world_judgment(self):
        for kind in ('budget_exhausted','timeout','tool_error','process_error','interrupted'):
            packet={'id':'finish','op':'finish','actor':{'role':'codex_cli','run_id':'fixture'},
                    'args':{'outcome':'aborted','system_reason':{'kind':kind,'detail':'runtime fixture'}}}
            finish_enforce(self.root,packet)
            closed=read(self.root/'task/completion_closure.json')
            self.assertEqual(closed['category'],'system_termination')
            self.assertEqual(closed['outcome'],'aborted')
            self.assertEqual(closed['system_reason']['kind'],kind)

    def test_system_reason_cannot_claim_success_or_unknown_physical_result(self):
        for outcome,kind in [('success','timeout'),('failed','object_fell')]:
            packet={'id':'finish','op':'finish','actor':{'role':'codex_cli','run_id':'fixture'},
                    'args':{'outcome':outcome,'system_reason':{'kind':kind,'detail':'fixture'}}}
            with self.assertRaises(ValueError):finish_enforce(self.root,packet)

    def test_physical_failure_still_requires_world_judgment(self):
        packet={'id':'finish','op':'finish','actor':{'role':'codex_cli','run_id':'fixture'},
                'args':{'outcome':'failed','reason':'object fell'}}
        with self.assertRaisesRegex(ValueError,'World computation|CodeWorld computation'):
            finish_enforce(self.root,packet)

    def completed_fixture(self):
        self.fixture.world.define_completion({'height':{'description':'box at measured height',
            'conditions':[{'kind':'state','object':'box','field':'height','min':-.01,'max':.01}]}})
        decision=self.fixture.decision()
        self.fixture.world.derive_state(decision['decision'],'box',{'height':'height'})
        report=client.Robot(self.root).verify_completion()
        self.assertEqual(report['status'],'VERIFIED')
        return report

    def test_finish_survives_read_only_ik_feedback(self):
        report=self.completed_fixture()
        source="def compute(ctx):\n p=ctx.position('box')\n return {'actions':[{'op':'check_ik','args':{'pose_W':{'translation_m':p.tolist(),'quaternion_xyzw':[0,0,0,1]}},'affected_objects':[]}]}"
        decision=self.fixture.world.compute_decision(self.fixture.program(source),purpose='planning',object_ids=['box'])
        action=decision['result']['actions'][0]
        self.exchange('check_ik',{'status':'IK_FOUND','robot_state':self.fixture.robot},
                      **action['args'],world_decision={'file':decision['decision'],'index':0})
        self.assertGreater(self.fixture.world.state()['revision'],report['state_revision'])
        finish_enforce(self.root,{'op':'finish','actor':{'role':'codex_cli','run_id':'fixture'},
                                 'args':{'verification':report['verification']}})
        self.assertEqual(read(self.root/'task/completion_closure.json')['outcome'],'world_verified')

    def test_finish_still_rejects_related_object_pending(self):
        report=self.completed_fixture()
        self.fixture.world.assess({'box':{'pose':'pending'}})
        with self.assertRaisesRegex(ValueError,'not verified'):
            finish_enforce(self.root,{'op':'finish','actor':{'role':'codex_cli','run_id':'fixture'},
                                     'args':{'verification':report['verification']}})

    def test_implicit_arm_is_frozen_before_later_active_arm_change(self):
        robot={**self.fixture.robot,'active_arm':0,'arms':[{'arm_name':'left'},{'arm_name':'right'}]}
        atomic(self.root/'robot_io/robot_state.json',robot)
        decision=self.fixture.decision()
        self.assertEqual(decision['result']['actions'][0]['args']['arm'],0)
        robot['active_arm']=1
        atomic(self.root/'robot_io/robot_state.json',robot)
        enforce(self.root,self.fixture.packet(decision))
        command=decision['result']['actions'][0]['args']
        self.assertEqual(list(resolve_commands([command],['left','right'],robot['active_arm'])),[0])

    def test_group_defaults_and_explicit_selection_are_frozen_in_payload(self):
        robot={**self.fixture.robot,'active_arm':0}
        atomic(self.root/'robot_io/robot_state.json',robot)
        source="def compute(ctx):\n p=ctx.position('box')\n return {'actions':[{'op':'select_arm','args':{'arm':1}},{'op':'gripper_group','args':{'commands':[{'mode':'close'},{'arm':0,'mode':'open'}]},'affected_objects':['box']}]}"
        d=self.fixture.world.compute_decision(self.fixture.program(source),purpose='manipulation',object_ids=['box'])
        self.assertEqual([c['arm'] for c in d['result']['actions'][1]['args']['commands']],[1,0])

    def test_legacy_controller_reply_retains_compatibility(self):
        atomic(self.root / 'task/world_flow.json', {'required': False})
        reply = {'status': 'ARRIVED', 'position_error_m': .001}
        self.assertEqual(self.exchange('move_tcp', reply, pose_W={}), reply)

    def test_strict_raw_robot_state_requires_world_query(self):
        with self.assertRaisesRegex(ValueError, 'ctx.proprioception'):
            client.Robot(self.root).state()
        atomic(self.root / 'task/world_flow.json', {'required': False})
        self.assertEqual(client.Robot(self.root).state(), self.fixture.robot)


if __name__ == '__main__':
    unittest.main()
