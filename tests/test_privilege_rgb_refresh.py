"""Post-slip correction closes the actual review/read/action path, offline."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_privilege_multiview import scene, action, commit
from tests.test_privilege_tracking_updates import grab, move, pending, pose
from privilege.observations import load
from privilege.world import atomic


def image_points(packet, transform, *, mode='rigid'):
    points = [[-.05, -.05, .05], [.05, -.05, .05], [.05, .05, .05], [-.05, .05, .05]]
    if mode == 'translation':
        points = points[:1]
    measurements = []
    for local in points:
        xyz = transform @ [*local, 1.]
        pixels = {}
        for view in packet['views']:
            camera = np.linalg.inv(view['T_W_C']) @ xyz
            projected = np.asarray(view['K']) @ camera[:3]
            pixels[view['view_id']] = (projected[:2] / projected[2]).tolist()
        measurements.append({'local_xyz': local, 'pixels': pixels})
    return {'image_points': {'mode': mode, 'points': measurements}}


@pytest.mark.parametrize('empty', [False, True])
def test_slip_fit_updates_next_plan_and_stays_after_retreat(scene, empty):
    world, robot, api = scene
    grab(api, robot)
    if empty:
        original = robot.state
        def state():
            value = original()
            value['left']['gripper'] = 0.
            return value
        robot.state = state
    move(api, robot, [.2, .1, .9])
    cap, rev = pending(api)
    _, packet, _ = load(world.root, cap)
    landed = np.eye(4)
    landed[:3, :3] = Rotation.from_euler('xyz', [18, -13, 37], degrees=True).as_matrix()
    landed[:3, 3] = [.22, -.05, .91]
    receipt = api.commit_step(cap, poses={'block': image_points(packet, landed)},
        held={'left': None}, expected_revision=rev, note='Board slipped and tilted in RGB')
    assert receipt['updated'] == ['block'] and not receipt['pose_refresh']
    assert api._loop.pending is None and world.state()['attachments'] == {}
    context = api.read_context()
    assert np.allclose(next(o for o in context['objects'] if o['object_id'] == 'block')['T_W_object'],
                       landed, atol=1e-6)
    validation = world.state()['objects']['block']['sensor_validation']
    assert validation['fit_performed'] and validation['rgb_fit']['rms_reprojection_px'] < 1e-8
    move(api, robot, [.5, .3, 1.1]); commit(api)
    assert np.allclose(world.snapshot().transform('block'), landed)


def test_same_review_fresh_image_correction_is_not_deferred_by_detach(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=0)
    result = commit(api, held={'left': None}, poses=pose(.23, .91, time_basis='image'))
    assert not result['deferred'] and result['updated'] == ['block']
    assert np.allclose(world.snapshot().position('block'), [.23, 0, .91])


def test_fresh_image_at_release_host_step_is_accepted(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=-1)
    cap, rev = pending(api)
    path, packet, _ = load(world.root, cap)
    packet['metadata']['image_host_step'] = robot.step
    atomic(path, packet)
    result = api.commit_step(cap, held={'left': None}, poses=pose(.25, .9, time_basis='image'),
                            expected_revision=rev, note='Fresh post-release image')
    assert not result['deferred']
    assert np.allclose(world.snapshot().position('block'), [.25, 0, .9])


def test_missing_landing_persists_but_does_not_add_action_gate(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=0)
    result = commit(api, held={'left': None})
    assert 'block' in result['pose_refresh']
    old = world.snapshot().transform('block')
    move(api, robot, [.5, .3, 1.1]); commit(api)
    assert 'block' in api.read_context()['pose_refresh']
    assert 'block' in api._loop.context()['pose_refresh']
    assert np.allclose(world.snapshot().transform('block'), old)
    action(api, robot, arm='left', gripper=0)
    assert 'block' in api._loop.pending['focus_objects']
    commit(api, poses=pose(.23, .92))
    assert not api.read_context()['pose_refresh']


def test_failed_fit_is_persistent_unresolved_and_retry_uses_same_capture(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=-1)
    cap, rev = pending(api)
    _, packet, _ = load(world.root, cap)
    target = np.eye(4); target[:3, 3] = [.2, .03, .93]
    spec = image_points(packet, target)
    bad = {'image_points': {**spec['image_points'], 'points': spec['image_points']['points'][:1]}}
    result = api.commit_step(cap, poses={'block': bad}, expected_revision=rev, note='Too few points')
    assert 'block' in result['fit_failed'] and not result['updated']
    assert 'block' in result['unresolved'] and 'block' in result['pose_refresh']
    step = robot.step
    result = api.integrate_rgb(cap, {'block': spec}, expected_revision=world.state()['revision'], note='More visible corners')
    assert result['objects'] == ['block'] and robot.step == step
    assert np.allclose(world.snapshot().transform('block'), target)
    assert not api.read_context()['pose_refresh']


def test_delayed_views_cannot_create_mixed_time_landing_fit(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=0)
    cap, rev = pending(api)
    path, packet, _ = load(world.root, cap)
    packet['views'][1]['timing']['pictures_behind'] = 1
    atomic(path, packet)
    before = world.snapshot().transform('block')
    result = api.commit_step(cap, poses={'block': image_points(packet, before)},
                            expected_revision=rev, note='Mixed-time views')
    assert 'delayed' in result['fit_failed']['block']
    assert np.allclose(world.snapshot().transform('block'), before)
    action(api, robot, arm='left', gripper=0)  # review closed without an extra wait


def test_disturbed_neighbour_retains_refresh_until_corrected(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=0)
    commit(api, disturbed={'block': 'Hit by the carried object'})
    assert world.state()['objects']['block']['review']['pose']['status'] == 'unresolved'
    action(api, robot, arm='left', gripper=0); commit(api)
    assert 'block' in api.read_context()['pose_refresh']
    action(api, robot, arm='left', gripper=0)
    commit(api, poses=pose(.2, .93))
    assert not api.read_context()['pose_refresh']


def test_translation_mode_preserves_declared_orientation(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=0)
    cap, rev = pending(api); _, packet, _ = load(world.root, cap)
    target = world.snapshot().transform('block'); target[:3, 3] = [.21, .03, .9]
    api.commit_step(cap, poses={'block': image_points(packet, target, mode='translation')},
                    expected_revision=rev, note='Translated without visible rotation')
    assert np.allclose(world.snapshot().transform('block'), target)
    assert world.state()['objects']['block']['sensor_validation']['orientation_observed_axes'] == [False]*3


def test_integrate_then_detach_same_capture_preserves_visual_provenance(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=0)
    cap, rev = pending(api)
    api.integrate_rgb(cap, pose(.2, .9), expected_revision=rev, note='Observed landing')
    commit(api, held={'left': None})
    row = world.state()['objects']['block']
    assert row['basis'] == 'rgb_agent_estimate'
    assert row['review']['pose']['status'] == 'observed'
    assert not row.get('pose_refresh') and world.state()['attachments'] == {}
    move(api, robot, [.5, .3, 1.1]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.2, 0, .9])
