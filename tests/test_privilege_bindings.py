"""Optional geometry bindings leave the host action/finish surface intact."""

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.runner import load_policy


@pytest.fixture
def geometry(tmp_path):
    storage = pytest.importorskip("privilege.storage")
    import trimesh

    path = tmp_path / "object.glb"
    trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(path)
    return storage.from_meshes({"object": {"mesh": path}}, tmp_path / "world", frame="test")


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_geometry_and_original_robot_in_same_execution(robo, tmp_path, geometry, arm):
    from privilege.pyrualean import AgentWorld

    api = AgentWorld(geometry, robo, tmp_path)
    kit = ArmToolkit(robo, arm=arm, workspace=tmp_path / "ws", bindings={"world": api})
    body = "print(world.snapshot().position('object')); robo.move_to([0.05, -0.10, 0.6])"
    if arm == "cells":
        answer = kit.execute_tool("python", {"code": body})
    else:
        answer = kit.execute_tool("run_program", {"code": "def run(robo):\n    " + body + "\n"})
    assert "exception" not in answer.result, answer.result
    assert answer.result["moved"] is True
    assert any(call["tool"] == "move_to" for call in answer.result["primitive_calls"])
    assert "[" in answer.result["stdout"]
    if arm == "cells":
        receipt = kit.execute_tool(
            "python",
            {
                "code": (
                    "world.record_feedback(['object'], note='moved by host'); "
                    "world.state()['revision']"
                )
            },
        )
        assert receipt.result["value"] == "1" and not receipt.result["moved"]
    finished = kit.execute_tool("finish", {"status": "stuck", "summary": "test only"})
    assert finished.result["environment_terminated"] is False


def test_reserved_binding_cannot_replace_robot(robo, tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        ArmToolkit(robo, arm="cells", workspace=tmp_path, bindings={"robo": object()})
    with pytest.raises(ValueError, match="reserved"):
        ArmToolkit(robo, arm="program", workspace=tmp_path, bindings={"run": object()})


def test_program_binding_persists_while_module_variables_do_not(tmp_path):
    namespace = {"store": []}
    path = tmp_path / "policy.py"
    path.write_text("store.append(len(store))\ndef run(robo):\n    return list(store)\n")
    assert load_policy(path, namespace=namespace)(None) == [0]
    assert load_policy(path, namespace=namespace)(None) == [0, 1]


@pytest.mark.parametrize("mode", ["enabled", "disabled", "failure", "pending"])
def test_play_setup_prompt_and_result_contract(tmp_path, monkeypatch, backend, geometry, mode):
    from types import SimpleNamespace

    import privilege.pyrualean as extension

    from pyrualean import play
    from pyrualean.robots import get_adapter

    monkeypatch.chdir(tmp_path)
    card = SimpleNamespace(
        robot="libero",
        max_env_steps=600,
        task_language="test",
        images={},
        render=lambda: "# Task\n test",
    )
    host = SimpleNamespace(
        boot=lambda *args, **kwargs: (backend, []),
        dump_card=lambda *args: card,
        make_backend=lambda toolkit: toolkit,
    )
    adapter = SimpleNamespace(**vars(get_adapter("libero")), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "resolve_rpent_root", lambda explicit=None: tmp_path)
    seen = []

    def prepare(path, **kwargs):
        seen.append("prepare")
        assert path == str(tmp_path / "config.json")
        if mode == "failure":
            raise ValueError("invalid calibration")
        api = extension.AgentWorld(geometry, kwargs["robo"], kwargs["scratch"])
        return {"world": api}, "\nPRIVILEGE TEST PROMPT", {"mode": "test"}

    monkeypatch.setattr(extension, "prepare", prepare)

    def runtime(prompt, images, kit, **kwargs):
        seen.append("agent")
        if mode in {"enabled", "pending"}:
            assert "PRIVILEGE TEST PROMPT" in prompt
            result = kit.execute_tool("python", {"code": "world.snapshot().objects()"})
            assert result.result["value"] == "['object']"
            assert "codeworld" in result.result
            if mode == "pending":
                kit.execute_tool(
                    "python", {"code": "world.read_context(); robo.move_to([.05,-.1,.6])"}
                )
        else:
            assert "PRIVILEGE TEST PROMPT" not in prompt
            assert (
                kit.execute_tool("python", {"code": "'world' in globals()"}).result["value"]
                == "False"
            )
        finish = kit.execute_tool("finish", {"status": "stuck", "summary": "offline test"})
        if mode == "enabled":
            assert finish.result["error"] == "early_finish_requires_recovery"
        return {"status": "completed", "returncode": 0, "errors": [], "runtime": "openai"}

    monkeypatch.setattr(play, "run_openai_agent", runtime)
    cell = SimpleNamespace(suite="test", task=0, seed=0, max_episode_steps=600, cuda_device=0)
    out = tmp_path / "episode"
    result = play.play_episode(
        cell,
        out,
        arm="cells",
        model="fake",
        runtime="openai",
        base_url="http://unused.invalid",
        budget_s=10,
        privilege_config="config.json" if mode != "disabled" else None,
    )
    assert result["environment_success"] is False
    if mode == "failure":
        assert seen == ["prepare"] and result["status"] == "privilege_error"
        assert result["error"]["message"] == "invalid calibration"
    else:
        assert seen == (["prepare", "agent"] if mode in {"enabled", "pending"} else ["agent"])
        if mode == "pending":
            assert result["status"] == "world_update_incomplete" and result["finish"] is None
            assert result["privilege"]["final_interaction"]["pending_update"] is not None
        elif mode == "enabled":
            assert result["status"] == "completed" and result["finish"] is None
        else:
            assert result["status"] == "completed" and result["finish"]["status"] == "stuck"
        assert ("privilege" in result) == (mode != "disabled")
