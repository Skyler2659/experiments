import json
import tempfile
import unittest
from pathlib import Path

from codeworld_robot.fresh_state import reset_episode_evidence


class FreshWorldPriorTests(unittest.TestCase):
    def test_reconstruction_survives_without_inheriting_episode_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'state').mkdir()
            for name in ('robot_io', 'decisions', 'work/completion'):
                path = root / name
                path.mkdir(parents=True)
                (path / 'old.json').write_text('{}')
            transform = [[1, 0, 0, .2], [0, 1, 0, .3], [0, 0, 1, .4], [0, 0, 0, 1]]
            state = {
                'revision': 4,
                'runtime': {'robot_state': {'robot_state_ref': 'old:8'}},
                'measurements': {'old': {'value': 'held'}},
                'objects': {'cup': {
                    'T_W_object': transform,
                    'basis': 'observation_estimate',
                    'sensor_validation': {'capture': 'old.png'},
                    'review': {'pose': {'status': 'observed'}},
                    'observed_state': {'fields': {'held': {
                        'status': 'observed', 'value': True, 'derivation': {'receipt': 'old.json'},
                    }}},
                }},
            }
            (root / 'state/current.json').write_text(json.dumps(state))
            reset_episode_evidence(root)
            result = json.loads((root / 'state/current.json').read_text())
            cup = result['objects']['cup']
            self.assertEqual(cup['T_W_object'], transform)
            self.assertEqual(cup['basis'], 'reconstruction_prior')
            self.assertEqual(cup['review']['pose']['status'], 'prior')
            self.assertNotIn('sensor_validation', cup)
            self.assertNotIn('measurements', result)
            self.assertNotIn('runtime', result)
            self.assertEqual(cup['observed_state']['fields']['held']['status'], 'pending')
            self.assertNotIn('derivation', cup['observed_state']['fields']['held'])
            self.assertFalse((root / 'decisions').exists())


if __name__ == '__main__':
    unittest.main()
