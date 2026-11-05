"""GPU queue, durable owners and coordinator-only recovery for WorldRUA."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .config import RESOURCE, read, save
from .episode import process_identity

TERMINAL = {"FINISHED", "INTERRUPTED", "CANCELLED"}


def stage(cfg):
    root = Path(cfg["output"])
    if root.exists():
        raise FileExistsError(
            "Use a new output directory, or resume the existing batch: " + str(root)
        )
    if Path(cfg["auth_home"]).is_relative_to(root):
        raise ValueError("Keep the source login outside the batch output")
    root.mkdir(parents=True)
    wrapper = Path(cfg["python"]).parent / "worldrua-codex"
    rows = []
    for job in cfg["jobs"]:
        folder = root / job["id"]
        folder.mkdir()
        config = read(RESOURCE / "robodojo.json")
        urdf = str(Path(cfg["robodojo_root"]) / "Assets/Robots/x5/X5A.urdf")
        config["robot_asset"]["urdf"] = config["collision"]["urdf"] = urdf
        config["reconstruction"].update(
            runtime=cfg["runtime"],
            codex=str(wrapper),
            codex_home=str(folder / "agent-home"),
            model=cfg["model"],
            reasoning=cfg["reasoning"],
        )
        save(folder / "privilege.json", config)
        args = [
            "--robot",
            "robodojo_xpl_rgb",
            "--arm",
            "cells",
            "--suite",
            "arx_x5",
            "--task",
            job["task"],
            "--seed",
            str(job["sample"]),
            "--eval-seed",
            str(cfg["eval_seed"]),
            "--cuda-device",
            "0",
            "--primitives",
            "no-vla",
            "--source-root",
            cfg["robodojo_root"],
            "--xpolicylab-root",
            cfg["xpolicylab_root"],
            "--runtime",
            "codex",
            "--model",
            cfg["model"],
            "--reasoning",
            cfg["reasoning"],
            "--codex-bin",
            str(wrapper),
            "--codex-home",
            str(folder / "agent-home"),
            "--images",
            "on-motion",
            "--feedback",
            "rich",
            "--max-turns",
            str(cfg["max_turns"]),
            "--budget-s",
            "0",
            "--privilege-config",
            str(folder / "privilege.json"),
            "--out",
            str(folder / "episode"),
        ]
        save(folder / "command.json", {"args": args})
        rows.append(dict(job, status="QUEUED"))
    save(
        root / "manifest.json", dict(schema="worldrua-batch-v1", created_at=time.time(), config=cfg)
    )
    save(root / "status.json", dict(time=time.time(), all_finished=False, runs=rows, gpu=[]))
    return root


def gpu_state():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    return [
        dict(zip(("index", "used", "total", "util"), map(int, line.split(","))))
        for line in out.splitlines()
    ]


def eligible_gpus(cfg, rows, gpus):
    return [
        g
        for g in gpus
        if g["index"] in cfg["gpus"]
        and g["total"] - g["used"] >= cfg["min_free_mib"]
        and sum(r["status"] == "RUNNING" and r.get("gpu") == g["index"] for r in rows)
        < cfg["slots_per_gpu"]
    ]


def reconcile(root, rows):
    """Adopt a live owner, including a launch before the coordinator saved status."""
    for row in rows:
        if row["status"] in TERMINAL:
            continue
        folder = root / row["id"]
        owner = read(folder / "owner.json")
        if not owner:
            if row["status"] == "RUNNING":
                # The coordinator can die just after Popen, before owner.json exists.
                # A live recorded PID must not be launched a second time.
                if process_identity(row.get("pid", -1)) == row.get("process_start") and row.get(
                    "process_start"
                ):
                    continue
                row.update(status="INTERRUPTED", error="Owner disappeared before registration")
            continue
        row.update(
            pid=owner["pid"],
            process_start=owner["process_start"],
            gpu=owner["gpu"],
            started_at=owner["started_at"],
        )
        if owner.get("process_start") and process_identity(owner["pid"]) == owner["process_start"]:
            row["status"] = "RUNNING"
            row["stage"] = (
                "POLICY"
                if (folder / "episode/privilege/integration.json").exists()
                else "RECONSTRUCTION"
                if (folder / "episode/privilege").exists()
                else "SIMULATOR_STARTUP"
            )
            continue
        result = read(folder / "episode/result.json")
        complete = result.get("status") == "completed" and isinstance(
            result.get("environment_success"), bool
        )
        guard = read(folder / "episode/resource_guard/interruption.json")
        row.update(
            status="FINISHED" if complete and not guard else "INTERRUPTED",
            finished_at=owner.get("finished_at", time.time()),
            result=result,
        )
        if row["status"] == "FINISHED":
            row["success"] = result["environment_success"]
        else:
            row["error"] = (
                guard.get("reason")
                or result.get("error")
                or "Owner exited without a complete evaluation"
            )
        native = []
        for path in (folder / "episode/bench/eval_result").rglob("_result.json"):
            native.append(dict(path=str(path.relative_to(root)), result=read(path)))
        row["native_results"] = native


def copy_login(cfg, folder):
    home = folder / "agent-home"
    home.mkdir(exist_ok=True, mode=0o700)
    shutil.copy2(Path(cfg["auth_home"]) / "auth.json", home / "auth.json")
    (home / "auth.json").chmod(0o600)


def cleanup_logins(root):
    """Remove only batch-owned copies; never revoke or touch the source login."""
    removed = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            d
            for d in dirs
            if not (Path(base) / d).is_symlink() and d not in ("bench", "node_modules")
        ]
        if "auth.json" in files:
            path = Path(base) / "auth.json"
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            path.unlink()
            removed.append(str(path.relative_to(root)))
    save(
        root / "login_cleanup.json",
        dict(time=time.time(), removed=removed, source_login_untouched=True),
    )


def supervise(root):
    cfg = read(root / "manifest.json")["config"]
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = read(root / "status.json")["runs"]
        children = []
        last_launch = time.time()  # allow preexisting owners to register after a crash
        while True:
            for child in children:
                child.poll()  # reap finished children before testing /proc identities
            reconcile(root, rows)
            try:
                gpus, error = gpu_state(), None
            except Exception as exc:
                gpus, error = [], str(exc)
            queued = next((r for r in rows if r["status"] == "QUEUED"), None)
            eligible = eligible_gpus(cfg, rows, gpus)
            if (
                queued
                and eligible
                and not (root / "STOP").exists()
                and time.time() - last_launch >= cfg["launch_interval_s"]
            ):
                gpu = min(
                    eligible,
                    key=lambda g: (
                        sum(r["status"] == "RUNNING" and r.get("gpu") == g["index"] for r in rows),
                        g["used"],
                    ),
                )
                folder = root / queued["id"]
                copy_login(cfg, folder)
                if cfg["truth_audit"]:
                    from .prepare import prepare_truth_audit

                    prepare_truth_audit(folder, cfg)
                with (folder / "episode.launch.log").open("ab") as log:
                    child = subprocess.Popen(
                        [
                            cfg["python"],
                            "-m",
                            "worldrua.episode",
                            "own",
                            str(folder),
                            str(gpu["index"]),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                children.append(child)
                queued.update(
                    status="RUNNING",
                    stage="SIMULATOR_STARTUP",
                    gpu=gpu["index"],
                    pid=child.pid,
                    process_start=process_identity(child.pid),
                    started_at=time.time(),
                )
                last_launch = time.time()
            done = all(r["status"] in TERMINAL for r in rows)
            save(
                root / "status.json",
                dict(
                    time=time.time(),
                    all_finished=done,
                    runs=rows,
                    gpu=gpus,
                    gpu_error=error,
                    paused=(root / "STOP").exists(),
                ),
            )
            if done:
                cleanup_logins(root)
                return
            time.sleep(5)


def service(root):
    with (root / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root / "STOP_SERVICE").exists():
            with (root / "supervisor.log").open("ab") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "worldrua.batch", "supervise", str(root)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                rc = proc.wait()
            with (root / "coordinator_restarts.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(time=time.time(), pid=proc.pid, returncode=rc)) + "\n")
            if read(root / "status.json").get("all_finished"):
                return
            time.sleep(10)


if __name__ == "__main__":
    target = Path(sys.argv[2]).resolve()
    if sys.argv[1] == "supervise":
        supervise(target)
    else:
        service(target)
