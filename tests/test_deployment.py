"""Catch setup omissions without a GPU, credentials or simulator process."""

import subprocess
from pathlib import Path

from worldrua import preflight
from worldrua.config import load, save


def fixture(tmp_path):
    sim = tmp_path / "sim"
    (sim / "env_cfg").mkdir(parents=True)
    (sim / "env_cfg/arx_x5.yml").write_text("config_name: arx_x5\n")
    path = tmp_path / "batch.json"
    save(
        path,
        dict(
            output="out",
            robodojo_root=str(sim),
            runtime="render",
            auth_home="login",
            python="/fake/agent/bin/python",
            codex="/fake/codex",
            environment={"PATH": "/fake/simulator/bin"},
            tasks=[dict(task="stack_bowls", samples=[0, 2])],
        ),
    )
    return load(path)


def test_missing_layout_is_reported_before_launch(tmp_path):
    cfg = fixture(tmp_path)
    layouts = Path(cfg["robodojo_root"]) / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    layouts.mkdir(parents=True)
    (layouts / "stack_bowls_0.json").write_text("{}")
    errors = []
    preflight.check_layouts(cfg, errors)
    assert len(errors) == 1 and "stack_bowls_2.json" in errors[0]


def test_probes_use_configured_environments_and_detect_missing_checkpoint(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name, path: "/fake/simulator/bin/python3" if name == "python3" else None,
    )
    monkeypatch.delenv("SAM3_CHECKPOINT_PATH", raising=False)

    def probe(label, command, env, failures, timeout=45):
        calls.append((label, command, env))
        if label == "Agent dependencies":
            failures.append("Agent dependencies: SAM3 missing")

    monkeypatch.setattr(preflight, "probe", probe)
    errors = preflight.check(cfg)
    assert any("SAM3_CHECKPOINT_PATH" in e for e in errors)
    assert any("SAM3 missing" in e for e in errors)
    host = next(c for c in calls if c[0] == "Agent dependencies")
    sim = next(c for c in calls if c[0].startswith("Simulator Python"))
    assert host[1][0] == "/fake/agent/bin/python"
    assert sim[1][0] == "/fake/simulator/bin/python3"
    assert sim[2]["PATH"] == "/fake/simulator/bin"


def test_failed_import_and_timeout_are_diagnostics(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "ModuleNotFoundError: cv2"),
    )
    errors = []
    assert preflight.probe("Agent", ["python", "-c", "pass"], {}, errors) is None
    assert "cv2" in errors[0]

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 45)

    monkeypatch.setattr(preflight.subprocess, "run", timeout)
    preflight.probe("Renderer", ["node"], {}, errors)
    assert "timed out" in errors[-1]
