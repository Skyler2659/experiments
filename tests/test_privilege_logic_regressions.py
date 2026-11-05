"""Deterministic state lifecycle/integration regressions, no live robot or inference."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from tests.test_privilege_multiview import scene, action, commit
from tests.test_privilege_tracking_updates import grab, move, pose, pending
from privilege.world import atomic


def test_rgb_update_poses_alias_keeps_revision_and_capture_checks(scene):
    world, robot, api = scene
    capture = api.observe()['capture']
    revision = world.state()['revision']
    api.integrate_rgb(capture, poses=pose(.25, .9), expected_revision=revision, note='Visible block')
    assert np.allclose(world.snapshot().position('block'), [.25, 0, .9])
    before = world.state()
    with pytest.raises(ValueError, match='revision|changed|Revision'):
        api.integrate_rgb(capture, poses=pose(.5, .9), expected_revision=revision, note='Stale write')
    assert world.state() == before
    with pytest.raises(ValueError, match='not both'):
        api.integrate_rgb(capture, pose(.5, .9), poses=pose(.5, .9),
                          expected_revision=before['revision'], note='Ambiguous')


def test_intermediate_pose_and_unresolved_survive_release_and_next_review(scene):
    world, robot, api = scene
    grab(api, robot)
    action(api, robot, arm='left', gripper=-1)
    receipt = commit(api, poses=pose(.3, .95), held={'left': None},
                     unresolved={'block': 'Falling; current pose usable, landing unknown'})
    row = world.state()['objects']['block']
    assert np.allclose(world.snapshot().position('block'), [.3, 0, .95])
    assert row['review']['pose']['status'] == 'unresolved'
    assert 'block' in receipt['pose_refresh'] and not world.state()['attachments']
    move(api, robot, [.5, 0, 1.]); commit(api)
    assert np.allclose(world.snapshot().position('block'), [.3, 0, .95])
    assert 'pose_refresh' in world.state()['objects']['block']
    action(api, robot, arm='left', gripper=0)
    commit(api, poses=pose(.32, .8))
    assert 'pose_refresh' not in world.state()['objects']['block']


def test_disturbed_fixed_object_not_auto_confirmed_by_omission(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=0)
    commit(api, disturbed={'wall': 'Visible part moved'})
    row = world.state()['objects']['wall']
    assert row['review']['pose']['status'] == 'unresolved'
    assert 'pose_refresh' in row


def test_nonexecuted_close_cannot_seed_later_grasp(scene):
    world, robot, api = scene
    original = robot.state
    robot.state = lambda: {**original(), 'left': {**original()['left'], 'gripper': 1.}}
    api.read_context(); api._before_action('move_pose', {'arm': 'left', 'gripper': 1})
    api._after_action(SimpleNamespace(to_dict=lambda: {'ok': False, 'env_steps': 0}))
    commit(api)
    assert 'left' not in world.state()['tracking'].get('grasp_candidates', {})


def test_failed_close_keeps_preexisting_real_candidate(scene):
    world, robot, api = scene
    action(api, robot, arm='left', gripper=1); commit(api)
    before = copy.deepcopy(world.state()['tracking']['grasp_candidates'])
    api.read_context(); api._before_action('move_pose', {'arm': 'left', 'gripper': 1})
    api._after_action(SimpleNamespace(to_dict=lambda: {'ok': False, 'env_steps': 0}))
    commit(api)
    assert world.state()['tracking']['grasp_candidates'] == before


def test_target_refreshes_only_changed_dependencies_and_keeps_snapshot_frozen(scene):
    world, robot, api = scene
    target = api.target(lambda s: s.position('block') + [0, 0, .1])
    frozen = api.snapshot()
    assert np.allclose(list(target), [.1, 0, 1.1])
    wall = world.snapshot().transform('wall'); wall[0, 3] += .2
    api.update({'wall': wall})
    assert np.allclose(list(target), [.1, 0, 1.1]) and target.evaluations == 1
    api.update({'block': pose(.3, 1.2)['block']['T_W_object']})
    assert np.allclose(list(target), [.3, 0, 1.3]) and target.evaluations == 2
    assert np.allclose(frozen.position('block'), [.1, 0, 1.])
    api.update({'block': pose(.3, 1.2)['block']['T_W_object']})
    assert np.allclose(list(target), [.3, 0, 1.3]) and target.evaluations == 2


def test_target_follows_prediction_and_visual_correction_without_extra_motion(scene):
    world, robot, api = scene
    target = api.target(lambda s: s.position('block'))
    assert np.allclose(target.value(), [.1, 0, 1.])
    grab(api, robot)
    move(api, robot, [0, 0, 1.]); commit(api, poses=pose(.2, 1.3))
    step = robot.step
    assert np.allclose(list(target), [.2, 0, 1.3])
    move(api, robot, [0, 0, 1.1]); commit(api)
    assert np.allclose(list(target), [.2, 0, 1.4])
    assert robot.step == step + 1


def test_target_refreshes_geometry_but_pose_only_target_ignores_shape(scene, tmp_path):
    world, robot, api = scene
    top = api.target(lambda s: s.bounds('block')[1])
    origin = api.target(lambda s: s.transform('block')[:3, 3])
    before = top.value(); origin.value()
    path = tmp_path/'large.glb'; trimesh.creation.box([.2]*3).export(path)
    world.publish_parts('block', {'body': {'mesh': path}})
    assert not np.allclose(top.value(), before) and top.evaluations == 2
    origin.value(); assert origin.evaluations == 1
    transform = np.eye(4); transform[2, 3] = .2
    world.update_parts('block', {'body': transform})
    assert np.isclose(top.value()[2], before[2]+.25) and top.evaluations == 3


def test_target_is_resolved_by_native_xyz_argument_conversion(scene):
    # Use the concrete official RGB adapter; capture dispatch without robot IO.
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    world, _, api = scene
    calls = []
    robot = RobodojoOfficialColourRobot(SimpleNamespace(), primitives='no-vla')
    robot._stateful = lambda tool, kwargs, build: calls.append(kwargs)
    target = api.target(lambda s: s.position('block'))
    robot.move_to(target)
    api.update({'block': pose(.4, 1.2)['block']['T_W_object']})
    robot.move_to(target)
    assert np.allclose(calls[0]['xyz'], [.1, 0, 1.])
    assert np.allclose(calls[1]['xyz'], [.4, 0, 1.2])


@pytest.mark.parametrize('task', ['build_tower', 'hang_mugs', 'fill_egg_holder'])
def test_shared_fixture_setup_independent_of_task_and_existing_model_preserved(scene, task):
    from privilege.fixtures import setup_fixed_obstacles
    import privilege.fixtures as fixtures
    world, _, _ = scene
    profile = json.loads((Path(fixtures.__file__).parent/'resources/fixed_obstacles/robodojo_x5.json').read_text())
    packet = {'frame': profile['frame'], 'views': [profile['reference_camera']], 'task': task}
    result = setup_fixed_obstacles(world, {}, packet, 'robodojo_xpl_rgb')
    assert result['added'] == ['camera_support']
    assert world.object('camera_support')['role'] == 'fixed_obstacle'
    before = world.state()
    result = setup_fixed_obstacles(world, {}, packet, 'robodojo_xpl_rgb')
    assert result['retained_existing'] == ['camera_support'] and world.state() == before


def test_other_camera_does_not_receive_wrong_fixed_fixture(scene):
    from privilege.fixtures import setup_fixed_obstacles
    world, _, _ = scene
    result = setup_fixed_obstacles(world, {}, {'frame':'pyrualean_world', 'views':[
        {'view_id':'cam_head','T_W_C':np.eye(4).tolist()}]}, 'robodojo_xpl_rgb')
    assert not result['matched_calibration'] and result['added'] == []


def test_target_tracks_requested_uncertainty_and_attachment_changes(scene):
    world, robot, api = scene
    def choose(s):
        state = s.uncertainty('block')
        dz = .1 if (state.get('review') or {}).get('pose', {}).get('status') == 'pending' else 0
        return s.position('block') + [0, 0, dz]
    target = api.target(choose)
    assert np.allclose(target.value(), [.1, 0, 1.])
    api.record_feedback(['block'], note='Object moved')
    assert np.allclose(target.value(), [.1, 0, 1.1]) and target.evaluations == 2
    s = world.state()
    s['attachments'] = {'left': {'object_id': 'block', 'T_E_object': np.eye(4).tolist()}}
    atomic(world.state_path, s)
    target.value(); assert target.evaluations == 3


def test_read_context_returns_the_same_snapshot_it_records(scene):
    _, _, api = scene
    original = api._loop.scene.packet
    calls = []
    def packet(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    api._loop.scene.packet = packet
    result = api.read_context()
    assert len(calls) == 1
    assert result['revision'] == api._loop.read_at['revision']


def test_live_arithmetic_pose_orientation_and_numpy_path_refresh(scene):
    world, robot, api = scene
    position = api.live.position('block') + [0, 0, .08]
    transform = api.live.transform('block')
    rotation = transform[:3, :3]
    path = np.stack([position, position + [0, 0, .1]])
    relative = np.linalg.inv(transform) @ api.live.transform('wall')
    before = path.value(); old_relative = relative.value()
    api.update({'block': pose(.4, 1.3)['block']['T_W_object']})
    assert np.allclose(path.value(), [[.4, 0, 1.38], [.4, 0, 1.48]])
    assert not np.allclose(relative.value(), old_relative)
    assert np.allclose(rotation.value(), np.eye(3))
    assert not np.allclose(path.value(), before)
    # Cross-action expression is resolved by the concrete native argument path.
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    calls=[]
    adapter=RobodojoOfficialColourRobot(SimpleNamespace(), primitives='no-vla')
    adapter._stateful=lambda tool, kwargs, build: calls.append(kwargs)
    adapter.move_to(position)
    assert np.allclose(calls[-1]['xyz'], [.4,0,1.38])


def test_composed_expression_uses_single_snapshot_and_unrelated_changes_reuse(scene):
    world, robot, api = scene
    expr=api.live.position('block') - api.live.transform('block')[:3,3]
    original=world.state
    calls=[]
    def state():
        calls.append(True)
        return original()
    world.state=state
    assert np.allclose(expr.value(), [0,0,0])
    assert len(calls)==1
    wall=api.snapshot().transform('wall');wall[0,3]+=.2
    api.update({'wall':wall});expr.value()
    assert expr.evaluations==1


def test_compute_supports_dynamic_branch_and_constant_snapshots(scene):
    world,robot,api=scene
    expr=api.compute(lambda s: np.stack([s.position('block'), s.position('wall')])
                     if s.position('block')[0] < .2 else np.array([[0.,0.,1.]]))
    assert expr.value().shape==(2,3)
    frozen=np.asarray(api.live.position('block'))
    api.update({'block':pose(.3,1.2)['block']['T_W_object']})
    assert expr.value().shape==(1,3) and expr.evaluations==2
    assert np.allclose(frozen,[.1,0,1.])
