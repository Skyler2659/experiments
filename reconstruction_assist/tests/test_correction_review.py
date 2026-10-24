import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reconstruction_assist import adaptive_refinement as adaptive
from reconstruction_assist import correction_review as correction


class CorrectionReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'runs/sample'
        self.node = self.run / 'fractal/butter_1'
        self.node.mkdir(parents=True)
        self.write(self.node / 'part.json', {'children': []})
        (self.node / 'component.js').write_text('bad stretched butter')
        self.write(self.node / 'node_context.json', {'depth': 1})
        self.write(self.run / 'input/objects.json', {'objects': [{'object_id': 'butter_1'}, {'object_id': 'butter_2'}]})
        self.write(self.run / 'input/measurement/evidence.json', {'objects': {'butter_1': {}, 'butter_2': {}}})
        self.write(self.run / 'input/capture.json', {'views': [
            {'view_id': 'agentview', 'role': 'fit', 'rgb_uri': 'agentview.png'},
            {'view_id': 'secret', 'role': 'holdout', 'rgb_uri': 'secret.png'}]})
        (self.run / 'input/agentview.png').write_bytes(b'original image')
        (self.run / 'input/secret.png').write_bytes(b'holdout')
        self.feedback = 'fractal/butter_1/feedback/first'
        render = self.run / self.feedback / 'agentview_render.png'
        render.parent.mkdir(parents=True)
        render.write_bytes(b'render')
        self.cfg = {'remote_work': str(self.run), 'transport': 'local', 'max_reviews': 1,
                    'refinement': {'strategy': 'adaptive'},
                    'agent_correction': {'enabled': True, 'max_rounds': 1, 'extra_reviews': 2, 'extra_cycles': 4}}
        result = {'score': .5, 'status': 'FIT_REVIEW_REQUIRED', 'objects': [{'object_id': 'butter_1', 'needs_revision': True}]}
        _, self.state = adaptive.finish_review(self.node, self.run, self.cfg, result, self.feedback)
        self.diagnosis = {'verdict': 'repair', 'issue_types': ['geometry', 'instance_association'], 'diagnosis': 'Two boxes bridged by one slab',
                          'affected_object_ids': ['butter_1'], 'visual_evidence': ['agentview has a visible gap'],
                          'repair_instructions': 'Split the wrongly bridged measured regions and preserve instance ownership.',
                          'observation_requests': [], 'mask_assignments': []}

    def write(self, path, obj):
        correction.write(path, obj)

    def review(self, diagnosis=None):
        with patch.object(correction, 'diagnose', return_value=diagnosis or self.diagnosis):
            return correction.review(self.node, self.run, self.cfg, self.state, self.feedback)

    def test_disabled_never_calls_diagnostic_agent(self):
        self.cfg['agent_correction']['enabled'] = False
        with patch.object(correction, 'diagnose', side_effect=AssertionError('unexpected call')):
            retry, _, _ = correction.review(self.node, self.run, self.cfg, self.state, self.feedback)
        self.assertFalse(retry)

    def test_packet_excludes_holdout_and_contains_cross_instance_context(self):
        out = self.node / 'packet'
        value = correction.packet(self.node, self.run, self.cfg, self.feedback, out)
        self.assertEqual([v['view_id'] for v in value['views']], ['agentview'])
        self.assertNotIn('secret.png', json.dumps(value))
        self.assertEqual(set(value['all_instance_bounds']), {'butter_1', 'butter_2'})
        self.assertEqual(len(value['images_in_order']), 2)

    def test_stagnation_reopens_only_diagnosed_scope_without_resetting_reviews(self):
        retry, state, note = self.review()
        self.assertTrue(retry)
        self.assertEqual(state['reviews'], 1)
        self.assertEqual(state['decision'], 'REFINE')
        self.assertNotIn('best_score', state)  # visually rejected baseline must not win by old score
        self.assertEqual(correction.effective_config(self.node, self.cfg)['max_reviews'], 3)
        self.assertEqual(correction.cycle_allowance(self.node), 4)
        self.assertFalse((self.run / 'fractal/butter_2').exists())
        self.assertIn('correction_instructions.md', note)
        self.assertFalse(adaptive.dispatch(self.node, self.run, 'scene'))

    def test_visual_pass_does_not_override_numeric_failure(self):
        value = {**self.diagnosis, 'verdict': 'accept', 'affected_object_ids': []}
        retry, state, _ = self.review(value)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'REVIEW_BUDGET_EXHAUSTED')

    def test_numeric_pass_never_calls_visual_agent(self):
        self.state['decision'] = 'ACCEPTED'
        self.write(self.node / 'adaptive_state.json', self.state)
        with patch.object(correction, 'diagnose', side_effect=AssertionError('must skip passing object')):
            retry, state, _ = correction.review(self.node, self.run, self.cfg, self.state, self.feedback)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'ACCEPTED')

    def test_missing_information_records_pending_capture(self):
        value = {**self.diagnosis, 'verdict': 'needs_observation',
                 'observation_requests': ['A closer oblique RGB-D view of the gap between the butter boxes']}
        retry, _, _ = self.review(value)
        self.assertFalse(retry)
        receipts = list((self.node / 'correction').glob('*/observation_request.json'))
        request = correction.load(receipts[0])
        self.assertFalse(request['capture_performed'])
        self.assertEqual(request['status'], 'PENDING_ACQUISITION')

    def test_correction_metadata_does_not_change_program_revision(self):
        before = adaptive.revision(self.node, self.run)
        self.write(self.node / 'correction_state.json', {'rounds': 1})
        (self.node / 'correction_instructions.md').write_text('repair')
        self.assertEqual(before, adaptive.revision(self.node, self.run))

    def test_extra_budget_is_bounded_and_final_visual_rejection_stays_unresolved(self):
        self.review()
        self.state = correction.load(self.node / 'adaptive_state.json')
        self.state.update(decision='REVIEW_BUDGET_EXHAUSTED', reviewed_revision='new_geometry')
        self.write(self.node / 'adaptive_state.json', self.state)
        retry, state, _ = self.review()
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'REVIEW_BUDGET_EXHAUSTED')
        self.assertEqual(correction.cycle_allowance(self.node), 4)

    def test_numeric_pass_after_repair_skips_final_visual_audit(self):
        self.review()
        self.state = correction.load(self.node / 'adaptive_state.json')
        self.state['decision'] = 'ACCEPTED'
        with patch.object(correction, 'diagnose', side_effect=AssertionError('must skip passing repair')):
            retry, state, _ = correction.review(self.node, self.run, self.cfg, self.state, self.feedback)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'ACCEPTED')
        self.assertEqual(correction.load(self.node / 'correction_state.json')['status'], 'REPAIRED_NUMERICALLY')

    def test_agent_error_cannot_be_counted_as_visual_acceptance(self):
        with patch.object(correction, 'diagnose', side_effect=TimeoutError('API timeout')):
            retry, state, _ = correction.review(self.node, self.run, self.cfg, self.state, self.feedback)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'REVIEW_BUDGET_EXHAUSTED')
        self.assertEqual(correction.load(self.node / 'correction_state.json')['status'], 'DIAGNOSIS_ERROR')

    def test_object_reviewer_cannot_reopen_sibling(self):
        value = {**self.diagnosis, 'affected_object_ids': ['butter_2']}
        self.assertFalse(self.review(value)[0])
        self.assertEqual(correction.load(self.node / 'correction_state.json')['status'], 'DIAGNOSIS_ERROR')

    def test_scene_review_reopens_named_child_and_parent(self):
        scene = self.run / 'fractal/scene'
        self.write(scene / 'node_context.json', {'depth': 0})
        self.write(scene / 'part.json', {'children': ['butter_1']})
        (scene / 'component.js').write_text('compose')
        target = self.run / 'fractal/scene/feedback/first'
        target.mkdir(parents=True)
        (target / 'agentview_render.png').write_bytes(b'render')
        state = {**self.state, 'decision': 'BLOCKED_BY_STOPPED_CHILD'}
        self.write(scene / 'adaptive_state.json', state)
        with patch.object(correction, 'diagnose', return_value=self.diagnosis):
            retry, state, _ = correction.review(scene, self.run, self.cfg, state, str(target.relative_to(self.run)))
        self.assertTrue(retry)
        self.assertEqual(state['decision'], 'REFINE')
        self.assertEqual(correction.load(self.node / 'adaptive_state.json')['decision'], 'REFINE')
        self.assertIn('butter_1', (scene / 'correction_instructions.md').read_text())


    def test_accepted_identity_partner_is_reused_without_spending_recovery(self):
        partner = self.run / 'fractal/butter_2'
        self.write(partner / 'adaptive_state.json', {'decision':'ACCEPTED'})
        self.write(partner / 'correction_state.json', {'rounds':1})
        before = {'butter_1':{'x':1},'butter_2':{'x':2}}
        after = {'butter_1':{'x':3},'butter_2':{'x':2}}
        targets = correction.repair_targets(self.run, {'butter_1','butter_2'},before,after)
        self.assertEqual(targets, {'butter_1'})

    def test_reciprocal_identity_evidence_allows_joint_scope(self):
        masks=[{'view_id':'agentview','object_id':a,'source_object_id':b,'crop_xyxy':[]}
               for a,b in [('butter_1','butter_2'),('butter_2','butter_1')]]
        value={**self.diagnosis,'affected_object_ids':['butter_1','butter_2'],'mask_assignments':masks}
        self.assertEqual(correction.validate_diagnosis(value,['butter_1','butter_2'],['butter_1']),{'butter_1','butter_2'})
        value['issue_types']=['geometry']
        with self.assertRaises(ValueError):
            correction.validate_diagnosis(value,['butter_1','butter_2'],['butter_1'])

    def test_diagnosis_releases_shared_lock_and_commit_reacquires_it(self):
        import fcntl
        shared = self.run / '.feedback.lock'
        with shared.open('a') as first:
            fcntl.flock(first, fcntl.LOCK_EX)
            def diagnose(*args):
                with shared.open('a') as second:
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self.diagnosis
            with patch.object(correction, 'diagnose', side_effect=diagnose):
                retry, _, _ = correction.review(self.node, self.run, self.cfg, self.state,
                                                self.feedback, shared_lock=first)
            self.assertTrue(retry)
            with shared.open('a') as second:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_stale_diagnosis_cannot_commit_or_refund_reviews(self):
        def diagnose(*args):
            self.write(self.run / 'input/measurement/evidence.json', {'objects': {'butter_1': {'new': True}}})
            return self.diagnosis
        with patch.object(correction, 'diagnose', side_effect=diagnose), patch.object(correction, 'reopen') as reopen:
            retry, state, _ = correction.review(self.node, self.run, self.cfg, self.state, self.feedback)
        self.assertTrue(retry)
        reopen.assert_not_called()
        self.assertEqual(state['reviews'], self.state['reviews'])
        self.assertEqual(correction.load(self.node / 'correction_state.json')['status'], 'STALE_DIAGNOSIS_REMEASURE')

    def test_evidence_change_keeps_consumed_adaptive_budget(self):
        self.write(self.run / 'input/measurement/evidence.json', {'objects': {'butter_1': {'new': True}}})
        result = {'status':'FIT_REVIEW_REQUIRED', 'score':.4, 'objects':[]}
        _, state = adaptive.finish_review(self.node,self.run,self.cfg,result,self.feedback)
        self.assertEqual(state['reviews'], 2)

    def test_texture_only_diagnosis_cannot_reopen_object(self):
        value = {**self.diagnosis, 'issue_types': ['texture'],
                 'repair_instructions': 'Make the wrapper more photorealistic'}
        retry, state, _ = self.review(value)
        self.assertFalse(retry)
        self.assertEqual(state['decision'], 'REVIEW_BUDGET_EXHAUSTED')
        self.assertFalse((self.node / 'correction_instructions.md').exists())

    def test_repair_requires_a_structural_issue(self):
        with self.assertRaises(ValueError):
            correction.validate_diagnosis({**self.diagnosis, 'issue_types': []}, ['butter_1'], ['butter_1'])

    def test_cli_image_flags_do_not_consume_the_prompt(self):
        output = self.node / 'cli'
        output.mkdir()
        seen = []
        def execute(command, **kwargs):
            seen.extend(command)
            self.write(output / 'diagnosis.json', self.diagnosis)
        with patch.object(correction.subprocess, 'run', side_effect=execute):
            correction.diagnose({'images_in_order': ['source.png', 'render.png']}, output, self.cfg)
        self.assertEqual(seen[-2], '--')
        self.assertIn('independent reconstruction', seen[-1])
        self.assertIn('Do not evaluate or repair texture', seen[-1])
        self.assertNotIn('even when numerical checks passed', seen[-1])

    def test_runner_can_use_recorded_extra_cycles_after_base_limit(self):
        import os
        import subprocess
        root = Path(__file__).resolve().parents[2]
        runtime = self.run.parents[1]
        bindir = runtime / 'bin'; bindir.mkdir()
        codex = bindir / 'codex'
        codex.write_text('#!/bin/bash\necho "session id: abcdef"\necho "{\\"children\\":[]}" > "$RCWM_ROOT/runs/sample/fractal/butter_1/part.json"\n')
        codex.chmod(0o755)
        hook = bindir / 'hook'
        hook.write_text('#!/bin/bash\nif [ "$4" = dispatch ]; then exit 0; fi\n'
                        'P="$RCWM_ROOT/runs/sample/fractal/butter_1/correction_state.json"\n'
                        "if [ ! -f \"$P\" ]; then echo '{\"extra_cycles\":2}' > \"$P\"; exit 10; fi\nexit 0\n")
        hook.chmod(0o755)
        env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'], RCWM_ROOT=str(runtime),
                   RCWM_CODE=str(root/'vendor/RecursiveCWM'), RCWM_FEEDBACK_HOOK=str(hook),
                   RCWM_ADAPTIVE_DISPATCH='1', RCWM_MAXCYC='1')
        result = subprocess.run(['bash', str(root/'vendor/RecursiveCWM/runner/solve_recursive.sh'),
                                 'runs/sample', 'butter_1', 'scene', '1', 'sample'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(x) for x in (self.run/'trace/events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['event']=='session_start' for e in events), 2)

    def test_mask_swap_is_simultaneous_and_only_new_evidence_is_activated(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow available in server geometry runtime')
        old = self.run / 'evidence'
        old.mkdir()
        for name, pixel in [('butter_1', (0, 0)), ('butter_2', (3, 3))]:
            im = Image.new('L', (4, 4)); im.putpixel(pixel, 255); im.save(old / (name + '.png'))
        ann = {'views': [{'view_id': 'agentview', 'masks': {
            name: {'uri': name + '.png'} for name in ['butter_1', 'butter_2']}}]}
        self.write(old / 'annotations.json', ann)
        self.write(old / 'config.json', {'annotations': str(old / 'annotations.json'),
                   'capture': str(self.run / 'input/capture.json'), 'output': str(old)})
        dest = self.node / 'correction/mask_test'
        assignments = [{'view_id': 'agentview', 'object_id': a, 'source_object_id': b, 'crop_xyxy': []}
                       for a, b in [('butter_1', 'butter_2'), ('butter_2', 'butter_1')]]
        def regenerate(*args, **kwargs):
            model = dest / 'evidence/model'
            self.write(model / 'evidence.json', {'objects': {'butter_1': {'corrected': True}}})
        before = (old / 'annotations.json').read_bytes()
        with patch.object(correction.subprocess, 'run', side_effect=regenerate):
            correction.rebuild_evidence(self.run, self.cfg, assignments, {'butter_1', 'butter_2'}, dest, _worker=True)
        new = correction.load(dest / 'annotations.json')['views'][0]['masks']
        self.assertEqual(Image.open(new['butter_1']['uri']).getpixel((3, 3)), 255)
        self.assertEqual(Image.open(new['butter_2']['uri']).getpixel((0, 0)), 255)
        self.assertEqual(before, (old / 'annotations.json').read_bytes())
        self.assertEqual(correction.active_evidence(self.run, self.cfg), str(dest / 'evidence'))
        self.assertTrue(correction.load(self.run / 'input/measurement/evidence.json')['objects']['butter_1']['corrected'])
        # A later single-object correction uses the immutable original label,
        # not butter_2's already-remapped mask from the active version.
        model_cfg = {**correction.load(old / 'config.json'), 'annotations':str(dest / 'annotations.json')}
        self.write(dest / 'evidence/config.json', model_cfg)
        second = self.node / 'correction/second'
        def regenerate_second(*args, **kwargs):
            self.write(second / 'evidence/model/evidence.json', {'objects': {'butter_1': {'corrected': True}}})
        with patch.object(correction.subprocess, 'run', side_effect=regenerate_second):
            correction.rebuild_evidence(self.run,self.cfg,[assignments[0]],{'butter_1'},second,_worker=True)
        newer = correction.load(second / 'annotations.json')['views'][0]['masks']
        self.assertEqual(Image.open(newer['butter_1']['uri']).getpixel((3,3)),255)
        self.assertEqual(Image.open(newer['butter_2']['uri']).getpixel((0,0)),255)


    def test_holdout_mask_correction_is_rejected_before_activation(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow available in server geometry runtime')
        old = self.run / 'evidence'
        old.mkdir()
        self.write(old / 'annotations.json', {'views': []})
        self.write(old / 'config.json', {'annotations': str(old / 'annotations.json'),
                   'capture': str(self.run / 'input/capture.json')})
        assignments = [{'view_id': 'secret', 'object_id': 'butter_1',
                        'source_object_id': 'butter_1', 'crop_xyxy': []}]
        with self.assertRaises(ValueError):
            correction.rebuild_evidence(self.run, self.cfg, assignments, {'butter_1'},
                                        self.node / 'correction/bad', _worker=True)
        self.assertFalse((self.run / 'correction/active_evidence.json').exists())


if __name__ == '__main__':
    unittest.main()
