"""Portable batch configuration. Paths are relative to the configuration file."""

import json
import os
from pathlib import Path
import re
import shutil
import sys

RESOURCE = Path(__file__).parent / "resources"


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if re.search(r"\$\{[^}]+\}", result):
            raise ValueError("Unset environment variable in configuration: " + value)
        return result
    return value


def load(path):
    path = Path(path).expanduser().resolve()
    cfg = expand(json.loads(path.read_text()))
    for key in ("output", "robodojo_root", "runtime", "auth_home"):
        if not cfg.get(key):
            raise ValueError("Missing configuration field: " + key)
        cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    cfg.setdefault("xpolicylab_root", str(Path(cfg["robodojo_root"]) / "XPolicyLab"))
    cfg["xpolicylab_root"] = str(
        (path.parent / Path(cfg["xpolicylab_root"]).expanduser()).resolve()
    )
    cfg.setdefault("python", sys.executable)
    cfg.setdefault("codex", "codex")
    for key in ("python", "codex"):
        value = cfg[key]
        cfg[key] = shutil.which(value) or str((path.parent / Path(value).expanduser()).resolve())
    cfg.setdefault("gpus", [0, 1, 2])
    cfg.setdefault("slots_per_gpu", 2)
    cfg.setdefault("min_free_mib", 14500)
    cfg.setdefault("launch_interval_s", 35)
    cfg.setdefault("model", "gpt-6-astra")
    cfg.setdefault("reasoning", "high")
    cfg.setdefault("max_turns", 300)
    cfg.setdefault("eval_seed", 0)
    cfg.setdefault("truth_audit", False)
    cfg.setdefault("environment", {})
    if not cfg["gpus"] or len(set(cfg["gpus"])) != len(cfg["gpus"]):
        raise ValueError("gpus must be a nonempty list of distinct GPU indices")
    if any(type(g) is not int or g < 0 for g in cfg["gpus"]):
        raise ValueError("Invalid GPU index")
    if type(cfg["slots_per_gpu"]) is not int or cfg["slots_per_gpu"] < 1:
        raise ValueError("slots_per_gpu must be positive")
    if cfg["min_free_mib"] <= 0 or cfg["launch_interval_s"] < 0:
        raise ValueError("Invalid resource reservation or launch interval")
    jobs = []
    for row in cfg.get("tasks", []):
        task = row["task"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
            raise ValueError("Invalid task identifier")
        for seed in row["samples"]:
            if type(seed) is not int or seed < 0:
                raise ValueError("Sample indices must be nonnegative integers")
            jobs.append(dict(id=f"{task}_s{seed}", task=task, sample=seed))
    if not jobs or len({r["id"] for r in jobs}) != len(jobs):
        raise ValueError("Provide nonempty, unique task/sample pairs")
    cfg["jobs"] = jobs
    return cfg


def check(cfg):
    """Check the configured deployment without starting a robot or contacting the model."""
    from .preflight import check as deployment_check

    return deployment_check(cfg)
