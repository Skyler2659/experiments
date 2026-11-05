"""Read-only multi-stage access forecasts over reconstructed geometry."""
import copy
import numpy as np
from .world import rigid


def preview_plan(world, stages, *, access_margin_m=.04):
    margin = float(access_margin_m)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('access_margin_m must be finite and nonnegative')
    state = copy.deepcopy(world.state())
    names = set(state['objects'])
    reports = []
    for index, stage in enumerate(stages):
        for oid, pose in stage.get('poses', {}).items():
            meta = world.object(oid)
            if meta.get('role') in {'robot_snapshot', 'fixed_obstacle'} or meta.get('movable') is False:
                raise ValueError('Cannot plan relocation of fixed object: ' + oid)
            state['objects'][oid]['T_W_object'] = rigid(pose).tolist()
        # Reserve access not just for this stage, but for later requested objects.
        needed = set(stage.get('access', []))
        for later in stages[index+1:]:
            needed.update(later.get('access', []))
            needed.update(later.get('poses', {}))
        if not needed <= names:
            raise ValueError('Unknown future access objects')
        bounds = {oid: world._mesh(oid, state).bounds for oid in names
                  if world.object(oid).get('role') != 'robot_snapshot'}
        risks = []
        for target in sorted(needed):
            box = bounds[target]
            zone = box + np.array([[-margin]*3, [margin]*3])
            for obstacle, other in bounds.items():
                if obstacle == target:
                    continue
                # Support below the object is not itself evidence of blocked side/top access.
                if other[1, 2] <= box[0, 2] + 1e-6:
                    continue
                if np.all(zone[0] <= other[1]) and np.all(other[0] <= zone[1]):
                    risks.append({'object': target, 'obstacle': obstacle,
                                  'reason': 'future_access_envelope_overlap'})
        reports.append({'stage': stage.get('name', str(index)), 'needed_later': sorted(needed),
                        'access_risks': risks})
    return {'world_revision': state['revision'], 'stages': reports,
            'coverage': 'hypothetical_AABB_access_envelopes',
            'access_margin_m': margin, 'metric_validation': False,
            'limitations': 'Conservative access warning, not visibility, IK, grasp stability or collision proof. '
                           'Use calibrated views and check_joint_path for candidate approaches.'}
