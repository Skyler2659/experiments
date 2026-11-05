"""Evidence consistency for grasps and placement claims, using allowed observations only."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .tracking import arm_pose


def angle(a, b):
    return float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def review(state, packet, poses, unchanged, unresolved, held, relations):
    history = state.setdefault('observation_pairs', {})
    checks = {}
    for arm, oid in held.items():
        if oid is None:
            continue
        old, new = history.get(oid), poses.get(oid)
        result = {'object_id': oid, 'status': 'unconfirmed',
                  'reason': 'Need two current observed poses during object/hand motion'}
        if old and new and oid not in unresolved and old['observation_id'] != packet['observation_id']:
            observed = all(new.get('translation_observed_axes', [])) and all(new.get('orientation_observed_axes', []))
            observed = observed and old['fully_observed'] and new.get('time_basis', 'current') == 'current'
            if observed:
                ee0, ee1 = arm_pose(old['robot_state'], arm), arm_pose(packet['robot_state'], arm)
                obj0, obj1 = np.asarray(old['pose']), np.asarray(new['T_W_object'])
                travel = float(np.linalg.norm(ee1[:3, 3] - ee0[:3, 3]))
                rotation = angle(ee0, ee1)
                delta = np.linalg.inv(ee0) @ obj0
                expected = ee1 @ delta
                error = float(np.linalg.norm(obj1[:3, 3] - expected[:3, 3]))
                rotation_error = angle(obj1, expected)
                if travel >= .005 or rotation >= np.deg2rad(5):
                    consistent = error <= .015 and rotation_error <= np.deg2rad(15)
                    moved = np.linalg.norm(obj1[:3, 3]-obj0[:3, 3]) >= .003 or angle(obj0, obj1) >= np.deg2rad(3)
                    result.update(status='motion_consistent' if consistent and moved else 'contradicted',
                                  reason='Paired RGB pose estimates compared with measured hand motion',
                                  translation_residual_m=error, rotation_residual_deg=float(np.rad2deg(rotation_error)),
                                  observations=[old['observation_id'], packet['observation_id']],
                                  metric_validation=False)
        checks[arm] = result
    # Store only actual pose estimates, never kinematic predictions or unchanged text.
    for oid, proposal in poses.items():
        if (oid in unresolved or proposal.get('time_basis', 'current') != 'current'
                or history.get(oid, {}).get('observation_id') == packet['observation_id']):
            continue
        history[oid] = {'observation_id': packet['observation_id'],
                        'host_step': packet.get('metadata', {}).get('host_step'),
                        'pose': copy.deepcopy(proposal['T_W_object']),
                        'robot_state': copy.deepcopy(packet['robot_state']),
                        'fully_observed': all(proposal['translation_observed_axes']) and all(proposal['orientation_observed_axes'])}
    claims = state.setdefault('relations', {})
    # A changed object or target invalidates prior stability claims until reviewed again.
    for oid, claim in claims.items():
        if oid in poses or oid in unresolved or claim['target'] in poses or claim['target'] in unresolved:
            claim['status'] = 'needs_review'
    for oid, value in relations.items():
        if oid not in state['objects'] or value.get('target') not in state['objects']:
            raise ValueError('relations must name known object and target IDs')
        if value.get('kind') not in {'supported', 'inserted'} or not str(value.get('note', '')).strip():
            raise ValueError('relations require supported/inserted kind and observation note')
        # Text claims cannot certify contact topology, stability, or evaluator success.
        claims[oid] = {**copy.deepcopy(value), 'status': 'observed_claim',
                       'observation_id': packet['observation_id'], 'metric_validation': False,
                       'verification': 'Contact/support is agent interpreted, not independently established'}
    state['grasp_review'] = checks
    return checks
