import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.finish_recovery import FinishRecovery, resume_argv
from pyrualean.play import _codex_argv


def check(policy, turns, **args):
    return policy.check(dict(status="stuck", **args), turns, terminal=False, remaining_calls=260)


def trial(index, strategy):
    return dict(turn_index=index, strategy=strategy, observation="The grasp still missed the object.")


def motion():
    return dict(tool="python", ok=True, env_steps=5)


def test_repeated_finish_old_trials_and_duplicate_trials_do_not_pass():
    p = FinishRecovery()
    turns = [motion()] * 40
    assert check(p, turns)["checkpoint_turn_index"] == 40
    turns.append(dict(tool="finish", ok=False))
    assert check(p, turns, recovery_trials=[trial(1, "a"), trial(2, "b")])
    turns.extend([motion(), motion()])
    assert check(p, turns, recovery_trials=[trial(41, "a"), trial(41, "b")])
    assert check(p, turns, recovery_trials=[trial(41, "a"), trial(42, " A ")])
    assert check(p, turns, recovery_trials=[trial(41, "a"), trial(99, "b")])
    assert check(p, turns, recovery_trials=[trial(41, "change approach"), trial(42, "regrasp from other side")]) is None


def test_read_only_turns_cannot_be_presented_as_executed_recovery():
    p = FinishRecovery()
    turns = []
    check(p, turns)
    turns.extend([dict(tool="finish", ok=False), dict(tool="python", ok=True, env_steps=0), motion()])
    assert check(p, turns, recovery_trials=[trial(1, "a"), trial(2, "b")])
    assert check(p, turns, unrecoverable=dict(reason="object outside reachable workspace", observation="fresh image shows it below table", turn_index=1, alternatives_ruled_out=["other arm cannot reach", "gripper cannot reach below table"])) is None
    assert p.audit[-1]["basis"] == "agent_observed_irrecoverability"


@pytest.mark.parametrize("status,terminal,remaining", [("success", False, 260), ("failure", True, 260), ("stuck", False, 0)])
def test_valid_exit_paths_remain_open(status, terminal, remaining):
    assert FinishRecovery().check(dict(status=status), [], terminal=terminal, remaining_calls=remaining) is None


def test_gate_preserves_live_namespace_and_budget(robo, tmp_path):
    interaction = SimpleNamespace(_before_finish=lambda: None, _tool_feedback=lambda: ({}, None), _before_action=lambda *a: None, _after_action=lambda *a: None, _action_rejected=lambda *a: None)
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", max_turns=4, interaction=interaction)
    initial = tk.execute_tool("python", {"code": "marker=17"})
    assert initial.result["tool_turn_index"] == 0
    stop = tk.execute_tool("finish", dict(status="stuck", summary="missed grasp"))
    assert stop.result["accepted"] is False and tk.finish is None
    assert tk._remaining_decision_calls() == 3
    assert tk.execute_tool("python", {"code": "marker"}).result["value"] == "17"
    a = tk.execute_tool("python", {"code": "robo.move_to([0.01, 0, 0.65])"})
    b = tk.execute_tool("python", {"code": "robo.move_to([0.02, 0, 0.65])"})
    assert b.result["tool_turn_index"] == 4
    assert tk._remaining_decision_calls() == 0
    assert tk.execute_tool("finish", dict(status="stuck", summary="budget exhausted")).result["ok"]


def test_original_baseline_finish_is_unchanged(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    assert tk.execute_tool("finish", dict(status="stuck")).result["ok"]
    assert "recovery_trials" not in tk.get_tools_spec()[-1]["input_schema"]["properties"]


def test_resume_keeps_model_and_live_mcp_but_drops_stale_initial_images():
    argv = _codex_argv(codex_bin="codex", model="gpt-6-astra", reasoning_effort="high", mcp_url="http://127.0.0.1:123/mcp", workspace=Path("/same/workspace"), last_message=Path("/last.md"), images=[Path("/initial.png")], base_url=None, api_key_env=None)
    resumed = resume_argv(argv, "same-session")
    assert resumed[:3] == ["codex", "exec", "resume"]
    assert resumed[-2:] == ["same-session", "-"]
    assert "/initial.png" not in resumed and "-C" not in resumed and "-s" not in resumed
    assert "gpt-6-astra" in resumed
    assert any("127.0.0.1:123/mcp" in x for x in resumed)
    assert any('model_reasoning_effort="high"' in x for x in resumed)


@pytest.mark.parametrize("cooperates", [True, False])
def test_codex_early_return_resumes_same_episode_with_bounded_continuations(monkeypatch, backend, robo, tmp_path, cooperates):
    import io
    import pyrualean.play as play
    import pyrualean.mcp_server as mcp

    monkeypatch.chdir(tmp_path)
    interaction = SimpleNamespace(_before_finish=lambda: None, _tool_feedback=lambda: ({}, None), _before_action=lambda *a: None, _after_action=lambda *a: None, _action_rejected=lambda *a: None)
    captured = {}
    def toolkit(*args, **kwargs):
        kwargs["interaction"] = interaction
        captured["toolkit"] = ArmToolkit(*args, **kwargs)
        return captured["toolkit"]
    class Server:
        def __init__(self, *args, **kwargs): pass
        def start(self): return "http://127.0.0.1:123/mcp"
        def stop(self): pass
    boots = []
    host = SimpleNamespace(boot=lambda *a, **k: (boots.append(1) or backend, []), make_backend=lambda t: t, dump_card=lambda *a: SimpleNamespace(task_language="test task", images={}))
    adapter = SimpleNamespace(name="libero", host_module=lambda: host, robot_cls=type(robo))
    monkeypatch.setattr(play, "get_adapter", lambda _: adapter)
    monkeypatch.setattr(play, "ArmToolkit", toolkit)
    monkeypatch.setattr(play, "build_prompt", lambda *a, **k: "task")
    monkeypatch.setattr(play, "stop_all", lambda *a: None)
    monkeypatch.setattr(mcp, "ArmMcpServer", Server)
    commands, prompts, timeouts = [], [], []
    class Pipe(io.BytesIO):
        def close(self): prompts.append(self.getvalue()); super().close()
    class Process:
        def __init__(self, argv, **kwargs):
            commands.append(argv)
            self.stdin = Pipe()
            self.events = kwargs["stdout"]
        def wait(self, timeout):
            timeouts.append(timeout)
            tk = captured["toolkit"]
            self.events.write(b'{"type":"thread.started","thread_id":"same-session"}\n')
            self.events.flush()
            if len(commands) == 1:
                tk.execute_tool("python", {"code": "marker=31"})
                if cooperates:
                    tk.execute_tool("finish", dict(status="stuck"))
            elif cooperates:
                assert tk.execute_tool("python", {"code": "marker"}).result["value"] == "31"
                indices = []
                for x in [0.01, 0.02]:
                    indices.append(tk.execute_tool("python", {"code": f"robo.move_to([{x}, 0, 0.65])"}).result["tool_turn_index"])
                end = tk.execute_tool("finish", dict(status="stuck", recovery_trials=[trial(indices[0], "new approach"), trial(indices[1], "other side")]))
                assert end.result.get("ok"), (end.result, tk.turns)
            return 0
    monkeypatch.setattr(play.subprocess, "Popen", Process)
    cell = SimpleNamespace(suite="test", task="test", seed=0, max_episode_steps=1000, cuda_device=None)
    result = play.play_episode(cell, tmp_path / "episode", arm="cells", model="gpt-6-astra", budget_s=300, max_turns=30)
    assert result["error"] is None, result
    assert len(boots) == 1
    assert len(commands) == (2 if cooperates else 3)
    assert all(c[:3] == ["codex", "exec", "resume"] for c in commands[1:])
    assert all(c[-2:] == ["same-session", "-"] for c in commands[1:])
    assert all(b"SAME live episode" in p for p in prompts[1:])
    assert timeouts == sorted(timeouts, reverse=True)
    assert result["finish_recovery"]["accepted_finish"] is cooperates
    assert captured["toolkit"]._cell_index == (4 if cooperates else 1)
