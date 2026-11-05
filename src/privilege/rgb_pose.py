"""Fit existing object geometry to agent-identified points in recorded RGB.

No detector, depth, simulator state, extra image capture or model call is used.
Pixel correspondences identify the SAME local landmark in two or more views.
"""

import numpy as np


def fit_pose(packet, specification, previous):
    views = {v['view_id']: v for v in packet['views']}
    local, observed, pixels, correspondences = [], [], [], []
    mode = specification.get('mode', 'rigid')
    if mode not in {'rigid', 'translation'}:
        raise ValueError('RGB fit mode must be rigid or translation')
    used = set()
    for point in specification['points']:
        xyz = np.asarray(point['local_xyz'], float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('local_xyz must be a finite object-local 3-vector')
        equations, offsets = [], []
        if len(point['pixels']) < 2:
            raise ValueError('A landmark needs at least two calibrated views')
        point_pixels = []
        saved_pixels = {}
        for name, uv in point['pixels'].items():
            view = views[name]
            # Moving/slipped geometry cannot be triangulated across different times.
            if view.get('timing', {}).get('pictures_behind', 0):
                raise ValueError('Use current views for RGB fitting; this view is delayed: ' + name)
            expected_step = packet.get('metadata', {}).get('host_step')
            view_step = view.get('timing', {}).get('host_step', expected_step)
            if expected_step is not None and view_step != expected_step:
                raise ValueError('RGB views must come from the current captured host step')
            uv = np.asarray(uv, float)
            if uv.shape != (2,) or not np.isfinite(uv).all():
                raise ValueError('pixels use finite [column, row] coordinates')
            if np.any(uv < 0) or np.any(uv >= view['image_size']):
                raise ValueError('Landmark lies outside the recorded image')
            camera = np.asarray(view['T_W_C'])
            ray = camera[:3, :3] @ np.linalg.solve(view['K'], [*uv, 1.])
            ray /= np.linalg.norm(ray)
            projector = np.eye(3) - np.outer(ray, ray)
            equations.append(projector)
            offsets.append(projector @ camera[:3, 3])
            point_pixels.append((view, uv))
            saved_pixels[name] = uv.tolist()
            used.add(name)
        a, b = np.concatenate(equations), np.concatenate(offsets)
        position, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
        if rank < 3:
            raise ValueError('View rays do not constrain landmark depth')
        local.append(xyz)
        observed.append(position)
        pixels.append(point_pixels)
        correspondences.append({'local_xyz': xyz.tolist(), 'pixels': saved_pixels})
    if not local:
        raise ValueError('Provide image landmarks for the changed object')
    local, observed = np.asarray(local), np.asarray(observed)
    transform = np.asarray(previous, float).copy()
    if mode == 'rigid':
        if len(local) < 3 or np.linalg.matrix_rank(local - local.mean(0)) < 2:
            raise ValueError('Rigid fitting needs three non-collinear local landmarks')
        if np.linalg.matrix_rank(observed - observed.mean(0)) < 2:
            raise ValueError('Observed landmarks are degenerate')
        u, _, vt = np.linalg.svd((local - local.mean(0)).T @ (observed - observed.mean(0)))
        transform[:3, :3] = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    transform[:3, 3] = observed.mean(0) - transform[:3, :3] @ local.mean(0)
    predicted = local @ transform[:3, :3].T + transform[:3, 3]
    errors = []
    for xyz, measurements in zip(predicted, pixels):
        for view, uv in measurements:
            camera_xyz = (np.linalg.inv(view['T_W_C']) @ [*xyz, 1.])[:3]
            if camera_xyz[2] <= 0:
                raise ValueError('Fitted landmark is behind a camera')
            projected = np.asarray(view['K']) @ camera_xyz
            errors.append(float(np.linalg.norm(projected[:2] / projected[2] - uv)))
    return {
        'T_W_object': transform.tolist(), 'time_basis': 'current',
        'translation_observed_axes': [True] * 3,
        'orientation_observed_axes': [mode == 'rigid'] * 3,
        'rgb_fit': {'mode': mode, 'views': sorted(used), 'landmarks': len(local),
                    'rms_reprojection_px': float(np.sqrt(np.mean(np.square(errors)))),
                    'rms_landmark_m': float(np.sqrt(np.mean(np.sum((predicted - observed)**2, axis=1)))),
                    'correspondences': correspondences,
                    'identity_source': 'agent_rgb_correspondences'},
    }


def resolve_proposals(world, packet, proposals):
    """Resolve image_points inside the ordinary update, without a second API cycle."""
    result, failed = {}, {}
    snapshot = world.snapshot()
    for oid, proposal in proposals.items():
        if 'image_points' not in proposal:
            result[oid] = proposal
            continue
        if 'T_W_object' in proposal:
            raise ValueError('Provide image_points or T_W_object, not both')
        # Unknown IDs and malformed proposals are still caller errors.
        previous = snapshot.transform(oid)
        try:
            result[oid] = fit_pose(packet, proposal['image_points'], previous)
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            failed[oid] = 'RGB fit unresolved: ' + str(exc)
    return result, failed
