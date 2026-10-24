from types import SimpleNamespace
import numpy as np
import pytest
from codeworld_robot.robodojo_dispatch import X5Dispatcher, SessionState


@pytest.fixture
def dispatcher(tmp_path, monkeypatch):
    monkeypatch.setattr('codeworld_robot.ownership.authorize', lambda *a: {'role': 'codex_cli', 'run_id': 'test'})
    session = SessionState()
    def must_not_act(*a, **k):raise AssertionError('Unexpected robot action')
    status = {'official_actions': 17, 'official_action_limit': 1050, 'remaining_actions': 1033}
    robot = SimpleNamespace(arm_name='left', gripper_name='gripper', gripper_move={'sign': 1, 'mimic': [0, 1, 0]}, gripper_scale=[0, 1])
    rm = SimpleNamespace(robot_list=[robot], get_joint=lambda r: [np.zeros(6)], get_end_effector_real_val=lambda r: [np.zeros(2)])
    return X5Dispatcher(session=session, root=tmp_path, names=['left'], solvers=[], rm=rm,
        validate_motion=must_not_act, execute_joints=must_not_act,
        state=lambda: {'active_arm': session.active}, settings={}, official=True,
        official_actions=SimpleNamespace(ended=False, status=lambda: status, interval=10),
        official_take=must_not_act, tick=must_not_act, benchmark=True,
        env=SimpleNamespace(success=[False]), capture=lambda: 'capture.json')


def test_select_observe_and_finish_share_session(dispatcher):
    assert dispatcher({'op': 'select_arm', 'args': {'arm': 'left'}})['active_arm'] == 0
    assert dispatcher({'op': 'observe'})['capture'] == 'capture.json'
    assert dispatcher({'op': 'finish'})['status'] == 'CLOSED'
    assert dispatcher.session.stopped


def test_collision_blocked_gripper_does_not_act_and_retains_budget(dispatcher):
    dispatcher.session.collision = SimpleNamespace(check_path=lambda *a, **k: {'status': 'COLLISION'})
    result = dispatcher({'op': 'gripper', 'args': {'mode': 'CLOSE', 'decision_ref': {'revision': 3}}})
    assert result['status'] == 'COLLISION_BLOCKED' and result['steps'] == 0
    assert result['official_actions'] == 17 and result['remaining_actions'] == 1033
    assert result['decision_ref'] == {'revision': 3}


def test_joint_motion_uses_existing_executor_and_measured_budget(dispatcher):
    calls = []
    dispatcher.execute_joints = lambda targets, duration, args: (calls.append((targets, duration)) or (20, {'status': 'CLEAR'}))
    result = dispatcher({'op': 'joints', 'args': {'joint_positions': [0.1]*6, 'duration_steps': 20}})
    assert result['status'] == 'EXECUTED' and result['steps'] == 20
    assert result['official_actions'] == 17
    assert len(calls) == 1 and calls[0][1] == 20
    np.testing.assert_allclose(calls[0][0][0], [.1]*6)


def test_motion_query_never_executes_robot(dispatcher):
    dispatcher.session.collision = object()
    calls = []
    dispatcher.validate_motion = lambda targets, args: (calls.append(targets) or (None, {'status': 'CLEAR'}))
    result = dispatcher({'op': 'check_motion', 'args': {'commands': [{'arm': 0, 'joint_positions': [0]*6}]}})
    assert result['status'] == 'CLEAR' and result['steps'] == 0
    assert result['official_actions'] == 17 and len(calls) == 1


def test_gripper_consumes_native_action_intervals(dispatcher):
    calls = []
    dispatcher.official_take = lambda **kwargs: calls.append(kwargs) or True
    result = dispatcher({'op': 'gripper', 'args': {'mode': 'OPEN', 'steps': 21}})
    assert result['status'] == 'EXECUTED'
    assert calls == [{'grippers': {0: 1.}}]*3  # ceil(21/10), unchanged native action accounting.
