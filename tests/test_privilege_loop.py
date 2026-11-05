"""Mandatory CodeWorld ordering on the unmodified fake host's motion primitives."""

import json

import numpy as np
import pytest

from pyrualean.arms import MAX_TEXT_CHARS, ArmResult, ArmToolkit
from pyrualean.openai_runtime import _tool_result_text


@pytest.fixture
def integrated(tmp_path, robo, monkeypatch):
    adapter = pytest.importorskip("privilege.pyrualean")
    import trimesh
    from privilege.storage import from_meshes

    path = tmp_path / "box.glb"
    trimesh.creation.box([0.1, 0.1, 0.1]).export(path)
    world = from_meshes({"box": {"mesh": path}, "rack": {"mesh": path}}, tmp_path / "world")

    def capture(robot, root, frame):
        from privilege.observations import register

        state = adapter.plain(robot.state())
        return register(
            root,
            np.zeros((12, 16, 3), dtype=np.uint8),
            K=[[10, 0, 8], [0, 10, 6], [0, 0, 1]],
            T_W_C=np.eye(4),
            frame=frame,
            robot_state=state,
            metadata={"host_step": state["step"]},
        )

    monkeypatch.setattr(adapter, "capture", capture)
    api = adapter.AgentWorld(world, robo, tmp_path)
    kit = ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "ws",
        feedback="pure",
        images="none",
        bindings={"world": api},
        interaction=api,
    )
    return kit, api, world


def execute(kit, code):
    return kit.execute_tool("python", {"code": code})


def commit(api, *, unresolved=None):
    context = api.read_context()
    return api.commit_step(
        context["interaction"]["pending_update"]["capture"],
        unchanged={"box": "RGB shows same placement", "rack": "RGB shows fixed rack"}
        if unresolved is None
        else {},
        unresolved=unresolved,
        expected_revision=context["revision"],
        note="post-action review",
    )


def test_read_required_and_no_second_motion_before_update(integrated, robo):
    kit, api, world = integrated
    denied = execute(kit, "robo.move_to([.05,-.1,.6])")
    assert "read_context" in denied.result["exception"]
    assert robo._ledger.stateful_calls() == 0
    before = world.snapshot()
    answer = execute(kit, "world.read_context(); robo.move_to([.05,-.1,.6]); robo.set_gripper(1)")
    assert "update required" in answer.result["exception"]
    assert robo._ledger.stateful_calls() == 1
    assert answer.result["codeworld"]["interaction"]["pending_update"]["capture"]
    assert len(answer.images) == 1  # required new RGB also with images=none, feedback=pure
    assert "stdout" in answer.result and "state" not in answer.result
    assert world.state()["revision"] == 1
    assert np.allclose(before.transform("box"), world.snapshot().transform("box"))
    assert "error" in kit.execute_tool("finish", {"status": "stuck", "summary": "pending"}).result
    assert kit.finish is None
    commit(api)
    assert world.state()["revision"] == 2
    assert "read_context" in execute(kit, "robo.set_gripper(1)").result["exception"]
    moved = execute(kit, "world.read_context(); robo.set_gripper(1)")
    assert "exception" not in moved.result
    commit(api, unresolved={"box": "hidden by fingers", "rack": "outside view"})
    state = world.state()
    assert state["objects"]["box"]["review"]["pose"]["status"] == "unresolved"
    assert "sensor_validation" not in state["objects"]["box"]
    assert api._loop.status()["updated_count"] == 2
    assert kit.execute_tool("finish", {"status": "stuck", "summary": "reviewed"}).result["error"] == "early_finish_requires_recovery"
    assert kit.finish is None
    assert kit.execute_tool("finish", {"status": "success", "summary": "reviewed"}).result["ok"]


def test_commit_requires_new_capture_complete_coverage_and_revision(integrated):
    kit, api, world = integrated
    old = api.observe()
    execute(kit, "world.read_context(); robo.move_to([.05,-.1,.6])")
    with pytest.raises(ValueError, match="after this action"):
        api.commit_step(
            old["capture"],
            unchanged={"box": "same", "rack": "same"},
            expected_revision=1,
            note="old",
        )
    cap = api.read_context()["interaction"]["pending_update"]["capture"]
    with pytest.raises(ValueError, match="known non-robot"):
        api.commit_step(cap, unchanged={"unknown": "same"}, expected_revision=1, note="bad object")
    with pytest.raises(ValueError, match="Stale"):
        api.commit_step(
            cap, unchanged={"box": "same", "rack": "same"}, expected_revision=0, note="stale"
        )
    assert world.state()["revision"] == 1 and api._loop.pending is not None
    t = np.eye(4)
    t[0, 3] = 0.03
    receipt = api.commit_step(
        cap,
        poses={
            "box": {
                "T_W_object": t.tolist(),
                "translation_observed_axes": [True, False, False],
                "orientation_observed_axes": [False, False, False],
            }
        },
        unchanged={"rack": "fixed in new RGB"},
        expected_revision=1,
        note="box shifted",
    )
    assert receipt["status"] == "WORLD_STEP_UPDATED"
    assert world.snapshot().transform("box")[0, 3] == 0.03


def test_action_failure_and_capture_failure_still_block_motion(integrated, backend):
    kit, api, world = integrated

    def bad_capture():
        raise ValueError("camera disconnected")

    api._loop.observe = bad_capture
    backend.error_next = {"error": "partial motion failed"}
    answer = execute(kit, "world.read_context(); robo.move_to([.05,-.1,.6])")
    assert "partial motion failed" in answer.result["exception"]
    pending = answer.result["codeworld"]["interaction"]["pending_update"]
    assert api._loop.pending["record"]["ok"] is False and "camera disconnected" in pending["capture_error"]
    assert pending["phase"] == "awaiting_update"  # unrelated objects are no longer all marked pending
    assert "update required" in execute(kit, "robo.set_gripper(1)").result["exception"]
    api._loop.observe = api._observe_raw
    api.observe()
    commit(api)


def test_prediction_and_branch_do_not_complete_update(integrated):
    kit, api, world = integrated
    execute(kit, "world.read_context(); robo.move_to([.05,-.1,.6])")
    api.update({"box": np.eye(4)}, basis="action_prediction")
    branch = api.fork("try")
    branch.read_context()
    assert "update required" in execute(kit, "robo.set_gripper(1)").result["exception"]
    commit(api)
    api.read_context()
    api.update({"box": np.eye(4)}, basis="agent_estimate")
    assert "read_context" in execute(kit, "robo.set_gripper(1)").result["exception"]


def test_program_cannot_batch_motion_without_world_update(integrated, robo, tmp_path):
    _, api, _ = integrated
    kit = ArmToolkit(
        robo,
        arm="program",
        workspace=tmp_path / "program",
        bindings={"world": api},
        interaction=api,
    )
    code = (
        "def run(robo):\n    world.read_context()\n"
        "    robo.move_to([.05,-.1,.6])\n    robo.set_gripper(1)\n"
    )
    result = kit.execute_tool("run_program", {"code": code})
    assert "update required" in result.result["exception"] and robo._ledger.stateful_calls() == 1
    commit(api)


def test_large_original_feedback_cannot_hide_codeworld_input():
    context = {"revision": 5, "objects": [{"object_id": "last_object", "description": "geometry"}]}
    result = ArmResult({"stdout": "x" * (MAX_TEXT_CHARS + 1), "codeworld": context})
    blocks = result.content_blocks
    assert "truncated" in blocks[0]["text"]
    assert json.loads(blocks[1]["text"])["codeworld"] == context
    assert "last_object" in _tool_result_text(result)


def test_inflight_motion_remains_visible_if_host_interrupts(integrated, backend):
    import threading

    kit, api, _ = integrated
    entered, release = threading.Event(), threading.Event()
    original = backend._do_move_to

    def wait_then_move(**kwargs):
        entered.set()
        assert release.wait(3)
        return original(**kwargs)

    backend._do_move_to = wait_then_move
    worker = threading.Thread(
        target=execute, args=(kit, "world.read_context(); robo.move_to([.05,-.1,.6])")
    )
    worker.start()
    try:
        assert entered.wait(3)
        assert api._loop.status()["pending_update"]["phase"] == "executing"
        with pytest.raises(ValueError, match="still executing"):
            api.commit_step("unused", unchanged={}, expected_revision=0, note="too early")
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert api._loop.status()["pending_update"]["phase"] == "awaiting_update"
    commit(api)
