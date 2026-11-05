"""Offline motion/correction sequences; no simulator or external perception calls."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_privilege_multiview import scene, action, commit
from privilege.observations import load
from privilege.world import atomic


def move(api, robot, xyz, *, gripper=0, ok=True):
    api.read_context()
    api._before_action('move_pose', {'arm': 'left', 'gripper': gripper})
    robot.pos = list(xyz)
    robot.step += 1
    api._after_action(SimpleNamespace(to_dict=lambda: {'ok': ok}))


def pose(x=.15, z=1., time_basis='current'):
    t = np.eye(4); t[:3, 3] = [x, 0., z]
    return {'block': {'T_W_object': t.tolist(), 'time_basis': time_basis,
                      'translation_observed_axes': [True]*3, 'orientation_observed_axes': [True]*3}}


def pending(api):
    ctx = api.read_context()
    return ctx['interaction']['pending_update']['capture'], ctx['revision']


def grab(api, robot):
    action(api, robot, arm='left', gripper=1)
    commit(api, held={'left': 'block'})


@pytest.mark.parametrize('route', ['commit', 'integrate_rgb', 'update', 'location'])
def test_visual_correction_survives_next_translation_and_rotation(scene, route):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=0)
    cap, rev = pending(api)
    if route == 'commit':
        api.commit_step(cap, poses=pose(), expected_revision=rev, note='Visual correction')
    elif route == 'integrate_rgb':
        api.integrate_rgb(cap, pose(), expected_revision=rev, note='Visual correction')
        commit(api)
    elif route == 'update':
        api.update({'block': pose()['block']['T_W_object']}, note='Current pose estimate')
        commit(api)
    else:
        api.integrate_location(cap, 'block', {'center_xyz': [.15, 0, 99], 'top_z': 1.05},
                               expected_revision=rev, note='Identity checked')
        commit(api)
    old = world.snapshot()
    move(api, robot, [0, 0, .9]); commit(api, held={'left': 'block'})
    assert np.allclose(world.snapshot().position('block'), [.15, 0, 1.1])
    assert np.allclose(old.position('block'), [.15, 0, 1.])
    robot.quat = [2**-.5, 0, 0, 2**-.5]
    move(api, robot, [.2, .1, .9]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.2, .25, 1.1])
    assert np.allclose(world.snapshot().transform('block')[:3, :3], [[0,-1,0],[1,0,0],[0,0,1]])


def test_grasp_confirmation_after_multiple_lift_segments_uses_close_time(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=1); commit(api)
    move(api, robot, [0, 0, .9], gripper=1); commit(api)
    move(api, robot, [0, 0, 1.0], gripper=1); commit(api, held={'left': 'block'})
    assert np.allclose(world.snapshot().position('block'), [.1, 0, 1.2])
    assert world.state()['attachments']['left']['initialization'] == 'paired_grasp_candidate'
    move(api, robot, [.2, 0, 1.1]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.3, 0, 1.3])


def test_current_visual_pose_takes_precedence_over_grasp_candidate(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=1); commit(api)
    move(api, robot, [0, 0, 1.0])
    commit(api, held={'left': 'block'}, poses=pose(.13, 1.18))
    move(api, robot, [0, 0, 1.1]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.13, 0, 1.28])


def test_correction_before_late_confirmation_refreshes_candidate(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=1); commit(api)
    move(api, robot, [0, 0, 1.0])
    cap, rev = pending(api)
    api.integrate_rgb(cap, pose(.13, 1.18), expected_revision=rev, note='Measured after lift')
    commit(api, held={'left': 'block'})
    assert np.allclose(world.snapshot().position('block'), [.13, 0, 1.18])


@pytest.mark.parametrize('route', ['commit', 'integrate_rgb'])
def test_delayed_rgb_correction_is_propagated_and_persists(scene, route):
    world, robot, api = scene
    grab(api, robot)
    move(api, robot, [.2, 0, .8])
    cap, rev = pending(api); path, packet, _ = load(world.root, cap)
    packet['metadata']['image_robot_state'] = {'left_ee_pose': [0, 0, .8, 1, 0, 0, 0]}
    atomic(path, packet)
    if route == 'commit':
        api.commit_step(cap, poses=pose(.15, time_basis='image'), expected_revision=rev, note='Delayed image')
    else:
        api.integrate_rgb(cap, pose(.15, time_basis='image'), expected_revision=rev, note='Delayed image')
        commit(api)
    assert np.allclose(world.snapshot().position('block'), [.35, 0, 1.])
    move(api, robot, [.3, 0, .8]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.45, 0, 1.])


@pytest.mark.parametrize('route', ['commit', 'integrate_rgb', 'location'])
def test_release_old_image_does_not_overwrite_landing_or_block_next_action(scene, route):
    world, robot, api = scene
    grab(api, robot)
    move(api, robot, [.2, 0, .8]); commit(api)
    action(api, robot, arm='left', gripper=-1)
    before = world.snapshot().transform('block').copy()
    cap, rev = pending(api); path, packet, _ = load(world.root, cap)
    packet['metadata']['image_robot_state'] = {'left_ee_pose': [0, 0, .8, 1, 0, 0, 0]}
    packet['views'][0]['timing']['pictures_behind'] = 1
    atomic(path, packet)
    if route == 'commit':
        result = api.commit_step(cap, poses=pose(.1, time_basis='image'), expected_revision=rev, note='Pre-release image')
    elif route == 'integrate_rgb':
        result = api.integrate_rgb(cap, pose(.1, time_basis='image'), expected_revision=rev, note='Pre-release image')
        commit(api)
    else:
        result = api.integrate_location(cap, 'block', {'center_xyz': [.1, 0, 99]}, expected_revision=rev, note='Old image estimate')
        commit(api)
    assert 'block' in result['deferred']
    assert np.allclose(world.snapshot().transform('block'), before)
    assert world.state()['objects']['block']['review']['pose']['status'] != 'observed'
    move(api, robot, [.5, 0, 1.]); commit(api, poses=pose(.31, .98))
    assert world.state()['attachments'] == {}
    move(api, robot, [.8, 0, 1.]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.31, 0, .98])


def test_open_and_retreat_does_not_drag_released_object_to_final_hand_pose(scene):
    world, robot, api = scene
    grab(api, robot)
    before = world.snapshot().position('block')
    move(api, robot, [.8, .4, 1.2], gripper=-1); commit(api)
    assert np.allclose(world.snapshot().position('block'), before)
    assert world.state()['attachments'] == {}
    assert world.state()['objects']['block']['basis'] == 'release_prediction'


def test_rejected_open_preserves_grasp_and_future_prediction(scene):
    world, robot, api = scene
    grab(api, robot)
    api.read_context(); api._before_action('grip', {'arm':'left', 'gripper':-1})
    api._after_action(SimpleNamespace(to_dict=lambda: {'ok': False, 'env_steps': 0}))
    commit(api)
    assert 'left' in world.state()['attachments']
    move(api, robot, [.2, 0, .8]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.3, 0, 1.])


def test_explicit_release_with_current_landing_correction_keeps_observation(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=0)
    commit(api, held={'left': None}, poses=pose(.3, .98))
    row = world.state()['objects']['block']
    assert row['basis'] == 'rgb_agent_estimate'
    assert row['review']['pose']['status'] == 'observed'
    assert 'sensor_validation' in row
    move(api, robot, [.5, 0, .9]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.3, 0, .98])


def test_measured_closed_fingers_override_successful_open_command(scene):
    world, robot, api = scene
    grab(api, robot)
    state = robot.state
    robot.state = lambda: {**state(), 'left': {**state()['left'], 'gripper': .2}}
    action(api, robot, arm='left', gripper=-1); commit(api)
    assert 'left' in world.state()['attachments']


def test_conflict_does_not_prefer_stale_world_and_location_needs_no_new_capture(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=0)
    commit(api, unresolved={'block': 'Moved, position not yet estimated'})
    assert api.reconcile_location('block', {'center_xyz': [.3, 0, 99]})['preferred'] == 'no_default'
    cap = api.observe()['capture']
    before_step = robot.step
    robot.image = lambda *a, **kw: pytest.fail('Unexpected new camera request')
    result = api.integrate_location(cap, 'block', {'center_xyz': [.3, 0, 99], 'top_z': 1.05},
                                   expected_revision=world.state()['revision'], note='Same captured object')
    assert result['objects'] == ['block'] and robot.step == before_step
    assert np.allclose(world.snapshot().position('block'), [.3, 0, 1.])
    assert world.state()['objects']['block']['sensor_validation']['source'] == 'pyrualean_locate'


def test_empty_gripper_stops_ghost_carry_and_repeated_held_does_not_rebind(scene):
    world, robot, api = scene
    state = robot.state
    robot.opening = .2
    robot.state = lambda: {**state(), 'left': {**state()['left'], 'gripper': robot.opening}}
    grab(api, robot)
    before = world.snapshot().transform('block').copy()
    robot.opening = 0.
    move(api, robot, [.4, 0, 1.1], gripper=1)
    assert world.state()['attachments'] == {}
    assert np.allclose(world.snapshot().transform('block'), before)
    assert world.state()['releases']['block']['reason'] == 'measured_empty_gripper'
    assert world.state()['objects']['block']['review']['pose']['status'] == 'unresolved'
    result = commit(api, held={'left': 'block'})
    assert result['held'] == {'left': None}
    assert result['rejected_held']['left']['reason'] == 'measured_empty_gripper'
    assert world.state()['attachments'] == {}
    move(api, robot, [.5, 0, 1.2], gripper=1)
    commit(api, held={'left': 'block'}, poses=pose(.3, .98))
    assert world.state()['attachments'] == {}
    assert np.allclose(world.snapshot().position('block'), [.3, 0, .98])
    assert world.state()['objects']['block']['review']['pose']['status'] == 'observed'
    # A later contact can establish a new grasp without an added gate/wait.
    robot.opening = .2
    action(api, robot, arm='left', gripper=1)
    commit(api, held={'left': 'block'})
    assert world.state()['attachments']['left']['object_id'] == 'block'


@pytest.mark.parametrize('opening', [None, .000001, .2])
def test_nonempty_or_unavailable_opening_does_not_infer_grasp_loss(scene, opening):
    world, robot, api = scene
    state = robot.state
    robot.state = lambda: {**state(), 'left': {**state()['left'], 'gripper': opening}}
    grab(api, robot)
    move(api, robot, [.2, 0, .9], gripper=1)
    commit(api, held={'left': 'block'})
    assert 'left' in world.state()['attachments']
    assert np.allclose(world.snapshot().position('block'), [.3, 0, 1.1])


def test_empty_gripper_cannot_propagate_delayed_image_as_if_still_held(scene):
    world, robot, api = scene
    state = robot.state
    robot.opening = .2
    robot.state = lambda: {**state(), 'left': {**state()['left'], 'gripper': robot.opening}}
    grab(api, robot)
    robot.opening = 0.
    move(api, robot, [.4, 0, 1.1], gripper=1)
    cap, rev = pending(api); path, packet, _ = load(world.root, cap)
    packet['metadata']['image_robot_state'] = {'left_ee_pose': [0, 0, .8, 1, 0, 0, 0]}
    packet['views'][0]['timing']['pictures_behind'] = 1
    atomic(path, packet)
    before = world.snapshot().transform('block').copy()
    result = api.commit_step(cap, poses=pose(.2, time_basis='image'), held={'left':'block'},
                             expected_revision=rev, note='Old image must not recreate grasp')
    assert 'block' in result['deferred']
    assert result['held'] == {'left': None}
    assert np.array_equal(world.snapshot().transform('block'), before)
    move(api, robot, [.5, 0, 1.2]); commit(api, poses=pose(.3, .98))
    assert np.allclose(world.snapshot().position('block'), [.3, 0, .98])


@pytest.mark.parametrize('feedback,cleared', [
    ({'ok': True, 'env_steps': 3}, True),
    ({'ok': False, 'env_steps': 2}, True),
    ({'ok': False, 'env_steps': 0}, False),
])
def test_partial_reopen_ends_pending_attempt_only_if_executed(scene, feedback, cleared):
    world, robot, api = scene
    state = robot.state
    robot.opening = .3
    robot.state = lambda: {**state(), 'left': {**state()['left'], 'gripper': robot.opening}}
    action(api, robot, arm='left', gripper=1); commit(api)
    before = world.state()['tracking']['grasp_candidates']['left']
    api.read_context()
    api._before_action('grip', {'arm': 'left', 'gripper': -1, 'opening': .52})
    if cleared:
        robot.opening = .52
    robot.step += 1
    api._after_action(SimpleNamespace(to_dict=lambda: feedback)); commit(api)
    candidates = world.state()['tracking']['grasp_candidates']
    assert ('left' not in candidates) == cleared
    if not cleared:
        assert candidates['left'] == before
    assert world.state()['attachments'] == {}


def test_partial_reopen_does_not_invent_release_of_confirmed_payload(scene):
    world, robot, api = scene
    state = robot.state
    robot.state = lambda: {**state(), 'left': {**state()['left'], 'gripper': .3}}
    grab(api, robot)
    action(api, robot, arm='left', gripper=-1, opening=.52); commit(api)
    assert 'left' in world.state()['attachments']
    move(api, robot, [.2, 0, .9]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.3, 0, 1.1])


def test_invalidated_candidate_cannot_return_through_action_before():
    from privilege import tracking
    state = {'objects': {'egg': {'T_W_object': np.eye(4).tolist()}}, 'revision': 0}
    robot = {'left': {'eef_pos': [0, 0, 1], 'eef_quat': [1, 0, 0, 0], 'gripper': .3}}
    tracking.begin_action(state, robot, 'grip', {'arm': 'left', 'gripper': 1})
    robot['left']['gripper'] = 0.
    tracking.predict(state, robot)
    assert not state['tracking']['grasp_candidates']
    robot['left'].update(gripper=.3, eef_pos=[0, 0, 2])
    tracking.reconcile(state, robot, {'left': 'egg'}, set())
    assert np.allclose(np.array(state['objects']['egg']['T_W_object']), np.eye(4))
    assert state['attachments']['left']['initialization'] == 'current_pose'


def test_recorded_empty_close_reopen_and_egg_regrasp_use_new_anchor():
    """Replay public calls 1432..1615; no simulator, truth or perception service."""
    import json
    from pathlib import Path
    from privilege import tracking
    trace = json.loads((Path(__file__).parent / 'fixtures/egg_stale_grasp_20261104.json').read_text())
    state = {'objects': {'egg_3': {'T_W_object': trace['actions'][0]['object_before']}}, 'revision': 0}
    for item in trace['actions']:
        assert np.allclose(state['objects']['egg_3']['T_W_object'], item['object_before'])
        tracking.begin_action(state, item['before'], item['tool'], item['kwargs'])
        reset = ['right'] if item['kwargs'].get('gripper') == -1 else []
        tracking.predict(state, item['after'], reset_candidates=reset)
        if item['after']['right']['gripper'] == 0:
            assert 'right' not in state['tracking']['grasp_candidates']
    last = trace['actions'][-1]
    tracking.reconcile(state, last['after'], {'right': 'egg_3'}, set())
    expected = (tracking.arm_pose(last['after'], 'right') @
                np.linalg.inv(tracking.arm_pose(last['before'], 'right')) @
                np.asarray(last['object_before']))
    assert np.allclose(state['objects']['egg_3']['T_W_object'], expected)
    assert np.linalg.norm(expected[:3, 3] - np.asarray(last['object_before'])[:3, 3]) < .001
    assert expected[2, 3] > .765  # Recorded table height; the old anchor put it below.
