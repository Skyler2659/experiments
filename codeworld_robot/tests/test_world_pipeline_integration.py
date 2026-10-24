"""Offline flow integration: prior -> RGB-D -> decision -> state -> completion."""
import unittest

import numpy as np

from codeworld_robot.tests import test_world_flow
from codeworld_robot.fresh_state import reset_episode_evidence
from codeworld_robot.io import atomic
from codeworld_robot.flow import enforce


class WorldPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_world_flow.WorldFlowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.world = self.fixture.world

    def test_fresh_prior_then_single_view_update_supports_world_state_reuse(self):
        f = self.fixture
        reset_episode_evidence(f.root)
        atomic(f.root / 'robot_io/robot_state.json', f.robot)
        source = """def height(context):
    return float(context.position('box')[2])
def compute(ctx):
    return {'judgments': {'height': height(ctx)}}
"""
        decision = self.world.compute_decision(f.program(source), purpose='judgment', object_ids=['box'])
        self.assertAlmostEqual(decision['result']['judgments']['height'], 0.)
        self.world.define_completion({'height': {'description': 'box center height is observed',
            'conditions': [{'kind': 'state', 'object': 'box', 'field': 'height', 'min': 1.14, 'max': 1.16}]}})
        self.assertEqual(self.world.verify_completion(robot_state=f.robot)['status'], 'INCONCLUSIVE')

        transform = np.eye(4); transform[2, 3] = 1.15
        update = self.world.integrate_rgbd(f.capture(), {'box': {
            'T_W_object': transform.tolist(), 'masks': {'cam': 'mask.npy'},
        }})
        self.assertIsNone(update['objects']['box']['orientation_observed'])
        self.assertEqual(update['objects']['box']['surface_queries'], 1)
        decision = self.world.compute_decision(f.program(source), purpose='judgment', object_ids=['box'])
        self.world.derive_state(decision['decision'], 'box', {'height': 'height'})
        self.assertEqual(self.world.verify_completion(robot_state=f.robot)['status'], 'VERIFIED')
        later = self.world.compute_decision(f.program("""def compute(ctx):
    return {'judgments': {'height': ctx.state('box', 'height')}}
"""), purpose='judgment', object_ids=['box'])
        self.assertAlmostEqual(later['result']['judgments']['height'], 1.15)

        # An actual movement of that object invalidates the observation and its
        # derived field, while its estimate stays available for replanning.
        moving = self.world.compute_decision(f.program("""def compute(ctx):
    p = ctx.position('box')
    return {'actions': [{'op': 'move_tcp', 'args': {'pose_W': {
        'translation_m': p.tolist(), 'quaternion_xyzw': [0, 0, 0, 1]}},
        'affected_objects': ['box']}]}
"""), purpose='manipulation', object_ids=['box'])
        packet = f.packet(moving); enforce(f.root, packet)
        atomic(f.root / 'robot_io/requests/request.json', packet)
        robot = {**f.robot, 'robot_state_ref': 'fixture:1', 'timestamp_ns': 1}
        atomic(f.root / 'robot_io/robot_state.json', robot)
        atomic(f.root / 'robot_io/responses/request.json', {'status': 'EXECUTED', 'steps': 1, 'robot_state': robot})
        self.world.sync_robot_feedback()
        self.assertEqual(self.world.verify_completion(robot_state=robot)['status'], 'INCONCLUSIVE')
        replanning = self.world.compute_decision(f.program(source), purpose='judgment', object_ids=['box'])
        self.assertAlmostEqual(replanning['result']['judgments']['height'], 1.15)


if __name__ == '__main__':
    unittest.main()
