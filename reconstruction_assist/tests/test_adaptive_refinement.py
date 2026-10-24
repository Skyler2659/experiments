import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from reconstruction_assist import adaptive_refinement as adaptive
from reconstruction_assist.refinement_policy import decomposition_required


class AdaptiveRefinementTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'runs/sample'
        self.node = self.run / 'fractal/object'
        self.node.mkdir(parents=True)
        self.write(self.node / 'part.json', {'module': 'component.js', 'children': []})
        (self.node / 'component.js').write_text('first candidate')
        self.config = {'max_reviews': 6, 'refinement': {
            'enabled': True, 'strategy': 'adaptive', 'patience': 2,
            'structural_objects': ['object'], 'minimum_parts': 2}}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def review(self, score, passed=False, inherited=False):
        result = {'score': score, 'status': 'FIT_CHECKS_PASSED' if passed else 'FIT_REVIEW_REQUIRED',
                  'objects': [{'object_id': 'object', 'needs_revision': not passed}]}
        return adaptive.finish_review(self.node, self.run, self.config, result, 'feedback/review', inherited)

    def test_first_pass_stops_without_forced_decomposition(self):
        self.assertFalse(decomposition_required(self.node, self.config))
        retry, state = self.review(.2, passed=True)
        self.assertFalse(retry)
        self.assertEqual(state['reviews'], 1)
        self.assertTrue(adaptive.dispatch(self.node, self.run, 'scene'))

    def test_object_descent_needs_a_failed_review(self):
        self.write(self.node / 'node_context.json', {'depth': 1, 'parent_id': 'scene'})
        self.assertFalse(adaptive.can_delegate(self.node, self.run))
        self.review(.5)
        self.assertTrue(adaptive.can_delegate(self.node, self.run))
        self.review(.1, passed=True)
        self.assertFalse(adaptive.can_delegate(self.node, self.run))

    def test_subpart_cannot_bypass_a_stopped_owner_budget(self):
        self.config['max_reviews'] = 1
        self.review(.5)
        child = self.run / 'fractal/detail'
        self.write(child / 'node_context.json', {'depth': 2, 'parent_id': 'object'})
        self.write(child / 'node_scope.json', {'owner_object_id': 'object'})
        self.assertFalse(adaptive.can_delegate(child, self.run))

    def test_scene_cannot_rename_objects_to_reset_their_budget(self):
        scene = self.run / 'fractal/scene'
        self.write(scene / 'node_context.json', {'depth': 0, 'parent_id': '-'})
        self.write(self.run / 'input/objects.json', {'objects': [{'object_id': 'object'}]})
        self.write(scene / 'children.json', ['object'])
        self.assertTrue(adaptive.can_delegate(scene, self.run))
        self.write(scene / 'children.json', ['object_retry_2'])
        self.assertFalse(adaptive.can_delegate(scene, self.run))

    def test_only_real_delivery_or_observation_change_reopens_a_pass(self):
        self.review(.2, passed=True)
        (self.node / 'brief.md').write_text('Please try again anyway')
        self.write(self.node / 'children.json', [])
        self.assertTrue(adaptive.dispatch(self.node, self.run, 'scene'))
        (self.node / 'component.js').write_text('changed geometry')
        self.assertFalse(adaptive.dispatch(self.node, self.run, 'scene'))
        (self.node / 'component.js').write_text('first candidate')
        self.write(self.run / 'input/capture.json', {'new_observation': True})
        self.assertFalse(adaptive.dispatch(self.node, self.run, 'scene'))

    def test_measured_parent_regression_can_reopen_pass_once(self):
        self.review(.2, passed=True)
        self.write(self.run / 'fractal/scene/adaptive_targets.json', {
            'object': {'feedback': 'new_parent_review', 'revision': adaptive.revision(self.node, self.run)}})
        self.assertFalse(adaptive.dispatch(self.node, self.run, 'scene'))
        self.assertTrue(adaptive.dispatch(self.node, self.run, 'scene'))

    def test_two_small_improvements_stop_local_work(self):
        self.assertTrue(self.review(1.)[0])
        self.assertTrue(self.review(.99)[0])
        retry, state = self.review(.985)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'NO_PROGRESS')
        self.assertEqual(state['reviews'], 3)

    def test_meaningful_improvement_resets_patience(self):
        self.review(1.)
        self.review(.99)
        retry, state = self.review(.8)
        self.assertTrue(retry)
        self.assertEqual(state['stagnant_reviews'], 0)

    def test_regression_restores_complete_subtree_and_external_assets(self):
        child = self.run / 'fractal/detail'
        self.write(child / 'part.json', {'children': []})
        (child / 'component.js').write_text('good detail')
        (child / 'texture.png').write_bytes(b'good texture')
        self.write(self.node / 'part.json', {'children': ['detail']})
        self.review(1.)
        (self.node / 'component.js').write_text('bad parent')
        (child / 'component.js').write_text('bad detail')
        (child / 'texture.png').write_bytes(b'bad texture')
        self.review(1.2)
        retry, state = self.review(1.3)
        self.assertFalse(retry)
        self.assertEqual((self.node / 'component.js').read_text(), 'first candidate')
        self.assertEqual((child / 'component.js').read_text(), 'good detail')
        self.assertEqual((child / 'texture.png').read_bytes(), b'good texture')
        self.assertEqual(state['reviewed_revision'], adaptive.revision(self.node, self.run))

    def test_stopped_child_residual_stops_parent_without_claiming_pass(self):
        retry, state = self.review(1., inherited=True)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'BLOCKED_BY_STOPPED_CHILD')
        selected = json.loads((self.node / 'selection.json').read_text())
        self.assertEqual(selected['selected_status'], 'FIT_REVIEW_REQUIRED')

    def test_budget_is_cumulative_across_parent_revisits(self):
        self.config['max_reviews'] = 2
        self.review(1.)
        retry, state = self.review(.5)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'REVIEW_BUDGET_EXHAUSTED')
        self.assertTrue(adaptive.dispatch(self.node, self.run, 'scene'))
        (self.node / 'component.js').write_text('parent attempts to restart the object')
        self.assertTrue(adaptive.dispatch(self.node, self.run, 'scene'))
        self.assertEqual((self.node / 'component.js').read_text(), 'first candidate')

    def test_passed_runner_revisit_does_not_call_model_or_remote_tools(self):
        self.review(.2, passed=True)
        root = Path(__file__).resolve().parents[2]
        self.config.update(remote_runtime='unused', remote_work='unused', remote_source='unused')
        transport = Path(self.temp.name) / 'transport.json'
        self.write(transport, self.config)
        runtime = Path(self.temp.name)
        fake = runtime / 'bin'
        fake.mkdir()
        for name in ('codex', 'ssh', 'rsync'):
            p = fake / name
            p.write_text('#!/bin/bash\ntouch "'+str(runtime/'unexpected_call')+'"\nexit 99\n')
            p.chmod(0o755)
        env = dict(os.environ, PATH=str(fake)+os.pathsep+os.environ['PATH'],
                   RCWM_ROOT=str(runtime), RCWM_CODE=str(root/'vendor/RecursiveCWM'),
                   RCWM_ADAPTIVE_DISPATCH='1',
                   RCWM_FEEDBACK_HOOK=str(root/'reconstruction_assist/feedback_hook.py'),
                   RCWM_ASSIST_CONFIG=str(transport))
        result = subprocess.run(['bash', str(root/'vendor/RecursiveCWM/runner/solve_recursive.sh'),
                                 'runs/sample', 'object', 'scene', '1', 'sample'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((runtime/'unexpected_call').exists())
        events = (self.run/'trace/events.jsonl').read_text()
        self.assertIn('"event":"reuse"', events)
        self.assertNotIn('"event":"session_start"', events)

    def test_parent_stops_when_only_residual_is_an_exhausted_object(self):
        from reconstruction_assist.feedback_hook import inherited_residuals
        self.config['max_reviews'] = 1
        feedback = 'fractal/object/feedback/first'
        failed = {'score': 1., 'status': 'FIT_REVIEW_REQUIRED', 'missing_object_ids': [],
                  'objects': [{'object_id': 'object', 'needs_revision': True, 'score': 1.}]}
        self.write(self.run / feedback / 'feedback.json', failed)
        adaptive.finish_review(self.node, self.run, self.config, failed, feedback)
        scene = self.run / 'fractal/scene'
        self.write(scene / 'part.json', {'children': ['object']})
        (scene / 'component.js').write_text('compose child unchanged')
        self.assertEqual(inherited_residuals(failed, scene, self.run), ['object'])
        retry, state = adaptive.finish_review(scene, self.run, self.config, failed,
                                              'fractal/scene/feedback/first', only_inherited=True)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'BLOCKED_BY_STOPPED_CHILD')
        self.assertEqual(json.loads((scene / 'adaptive_targets.json').read_text()), {})
        self.assertTrue(adaptive.dispatch(scene, self.run, '-'))


if __name__ == '__main__':
    unittest.main()
