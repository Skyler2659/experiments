"""An independent episode owner. Coordinator death does not stop its simulator."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import read, save


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def policy(folder):
    from .recorder import install

    install(folder / "audit")
    from pyrualean.play import main

    args = read(folder / "command.json")["args"]
    rc = main(args)
    result_path = folder / "episode/result.json"
    if result_path.exists():
        result = read(result_path)
        result.update(system="worldrua", method="WorldRUA", robot_interface="pyrualean")
        save(result_path, result)
    return rc


def own(folder, gpu):
    with (folder / "execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 73
        # Never restart a previous episode implicitly, even if its owner died.
        if (folder / "owner.json").exists():
            return 74
        owner = dict(
            pid=os.getpid(),
            process_start=process_identity(os.getpid()),
            gpu=gpu,
            started_at=time.time(),
        )
        save(folder / "owner.json", owner)
        cfg = read(folder.parent / "manifest.json")["config"]
        env = dict(os.environ, **cfg["environment"])
        env.update(
            WORLD_RUA_CODEX=cfg["codex"],
            WORLD_RUA_AUTH_HOME=cfg["auth_home"],
            WORLD_RUA_RUN=str(folder),
            CODEX_HOME=str(folder / "agent-home"),
            PYRUA_RECOVERY_DIR=str(folder / "recovery"),
            PYRUA_AUDIT_ROOT=str(folder / "audit"),
            RUN_GPU=str(gpu),
            ROBODOJO_MAX_BASH_RETRIES="0",
        )
        env.setdefault("OMP_NUM_THREADS", "8")
        # The GPU is selected by the host flag, not by remapping global device IDs.
        cmd = read(folder / "command.json")
        args = cmd["args"]
        args[args.index("--cuda-device") + 1] = str(gpu)
        save(folder / "command.json", cmd)
        proc = subprocess.Popen(
            [cfg["python"], "-m", "worldrua.episode", "policy", str(folder)], env=env, cwd=folder
        )
        owner["child_pid"] = proc.pid
        save(folder / "owner.json", owner)
        rc = proc.wait()
        owner.update(returncode=rc, finished_at=time.time())
        save(folder / "owner.json", owner)
        return rc


if __name__ == "__main__":
    folder = Path(sys.argv[2]).resolve()
    raise SystemExit(policy(folder) if sys.argv[1] == "policy" else own(folder, int(sys.argv[3])))
