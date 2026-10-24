"""Agent-owned, aspect-specific state review. No automatic perception or motion gate."""
import os
import copy
import time
import fcntl

STATUSES = {'pending', 'occluded', 'observed', 'hypothesized', 'unreviewed'}


def stamp_review(row, aspect, status, revision, evidence, note=None):
    row.setdefault('review', {})[aspect] = {
        'status': status, 'revision': revision, 'actor': {'role': os.environ.get('CODEWORLD_ROLE','supervisor_manual'), 'run_id': os.environ.get('CODEWORLD_EXECUTOR_ID')}, 'recorded_at_ns': time.time_ns(),
        'evidence': copy.deepcopy(evidence or []), 'note': note,
    }


def record_pose_observation(row, revision, basis, evidence):
    # A predicted pose must not clear a concern raised by an actual disturbance.
    refs = (evidence or {}).get('references', [])
    if basis == 'observation_estimate' and refs:
        stamp_review(row, 'pose', 'observed', revision, refs, evidence.get('note'))


class ReviewMixin:
    def assess(self, assessments, *, evidence=None, note=None, expected_revision=None):
        """Record selected object/aspect assessments without altering geometry or poses.

        Example: {'plate': {'pose': 'pending'}, 'cabinet':
                  {'geometry:drawer_interior': 'unreviewed'}}.
        Aspect names are agent-defined. Observed assessments require evidence references.
        This is an agent assertion, not an independent geometry/pose certification.
        """
        from .world import atomic
        if not assessments:
            raise ValueError('Empty assessment')
        from .flow import strict
        if strict(self.root) and any(v=='observed' for a in assessments.values() for v in a.values()):raise ValueError('Confirm observed state through integrate_rgbd, not a visual assertion')
        refs = list(evidence or [])
        if isinstance(evidence, str):
            raise ValueError('Evidence must be a list of references')
        with (self.root/'state/update.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.state()
            if expected_revision is not None and state['revision'] != expected_revision:
                raise ValueError('Stale state revision; reread and recompute')
            for oid, aspects in assessments.items():
                if oid not in state['objects']:
                    raise KeyError(oid)
                if not aspects:
                    raise ValueError('Empty aspect assessment')
                for aspect, status in aspects.items():
                    if not isinstance(aspect, str) or not aspect or status not in STATUSES:
                        raise ValueError('Invalid aspect or review status')
                    if status == 'observed' and not refs:
                        raise ValueError('Observed assessment requires evidence references')
                    stamp_review(state['objects'][oid], aspect, status, state['revision']+1, refs, note)
            state['revision'] += 1
            state['history'].append({'revision': state['revision'], 'kind': 'state_assessment',
                'assessments': copy.deepcopy(assessments), 'evidence': refs, 'note': note,
                'actor': {'role': os.environ.get('CODEWORLD_ROLE','supervisor_manual'), 'run_id': os.environ.get('CODEWORLD_EXECUTOR_ID')}, 'recorded_at_ns': time.time_ns()})
            atomic(self.state_path, state)
        return state['revision']

    def review_status(self, object_ids=None):
        """Return last assessments, including unresolved objects outside the task target.

        No assessment means unreviewed, never implicitly unchanged. Does not inspect RGB-D.
        Predictions and geometry publication do not confirm pending observations.
        """
        state = self.state()
        ids = list(state['objects']) if object_ids is None else list(object_ids)
        from .world import read
        objects = {}
        for oid in ids:
            row = state['objects'][oid]
            metadata = read(self.root/self.manifest['objects'][oid])
            aspects = {'pose': {'status': 'unreviewed'}, 'geometry': {'status': 'unreviewed'}}
            aspects.update(copy.deepcopy(row.get('review', {})))
            objects[oid] = {'basis': row.get('basis'), 'aspects': aspects,
                            'source_unknowns': metadata.get('declared', {}).get('unknowns', [])}
        return {'state_revision': state['revision'], 'objects': objects}
