"""Linux out-of-process limits for an episode's in-process Python tools.

A Python thread cannot safely be killed after an RPC timeout. On a limit breach
the watcher records an infrastructure interruption and kills this host and its
descendants, rather than allowing an abandoned cell to keep moving or allocating.
RSS is sampled, not a kernel memory sandbox; GPU virtual address space is not
limited. The watcher's interpreter is independent of the policy's GIL.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "ppid": int(fields[1]), "start": int(fields[19]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE")}
    except (OSError, ValueError, IndexError):
        return None


def descendants(root, excluded):
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            row = process_info(int(path.name))
            if row:
                rows[row["pid"]] = row
    owned = {root}
    while True:
        more = {pid for pid, r in rows.items() if r["ppid"] in owned and pid != excluded}
        if more <= owned:
            break
        owned |= more
    return [rows[pid] for pid in owned if pid in rows and pid != excluded]


def watch(config):
    root = config["pid"]
    control = Path(config["control"])
    ready = control.with_name("ready.json")
    atomic(ready, {"watcher_pid": os.getpid()})
    active = None
    while True:
        owner = process_info(root)
        if owner is None or owner["start"] != config["start"]:
            return
        try:
            state = json.loads(control.read_text())
            if state.get("stop"):
                return
            active = state.get("active")
        except (OSError, ValueError):
            pass  # retain the current deadline if the control file is temporarily unreadable
        tree = descendants(root, os.getpid())
        tree_rss = sum(r["rss_bytes"] for r in tree)
        reason = None
        if owner["rss_bytes"] > config["host_memory_bytes"]:
            reason = "host_memory_limit"
        elif tree_rss > config["tree_memory_bytes"]:
            reason = "episode_memory_limit"
        elif active and time.monotonic() >= active["deadline"]:
            reason = "tool_wall_timeout"
        if reason:
            receipt = {"status": "resource_interrupted", "reason": reason,
                       "time": time.time(), "active_tool": active,
                       "host_rss_bytes": owner["rss_bytes"], "tree_rss_bytes": tree_rss,
                       "host_memory_bytes": config["host_memory_bytes"],
                       "tree_memory_bytes": config["tree_memory_bytes"],
                       "processes": tree, "signal": "SIGKILL",
                       "evaluation_failure": False}
            atomic(config["receipt"], receipt)
            # Check start times so a recycled PID can never identify another job.
            for row in sorted(tree, key=lambda r: r["pid"] != root):
                live = process_info(row["pid"])
                if live and live["start"] == row["start"]:
                    try:
                        os.kill(row["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            return
        time.sleep(config.get("poll_s", 0.25))


class ExecutionGuard:
    def __init__(self, directory, *, tool_timeout_s=240.0,
                 host_memory_mb=8192, tree_memory_mb=20480):
        values = (tool_timeout_s, host_memory_mb, tree_memory_mb)
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
            raise ValueError("Execution limits must be finite and positive")
        self.directory = Path(directory).resolve()
        self.timeout = float(tool_timeout_s)
        self.host_memory_mb = float(host_memory_mb)
        self.tree_memory_mb = float(tree_memory_mb)
        self.process = None
        self.serial = 0

    def start(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("The execution guard requires Linux /proc")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.control = self.directory / "control.json"
        ready = self.directory / "ready.json"
        ready.unlink(missing_ok=True)
        atomic(self.control, {"active": None})
        owner = process_info(os.getpid())
        config = {"pid": os.getpid(), "start": owner["start"],
                  "control": str(self.control),
                  "receipt": str(self.directory / "interruption.json"),
                  "host_memory_bytes": int(self.host_memory_mb * 1024**2),
                  "tree_memory_bytes": int(self.tree_memory_mb * 1024**2)}
        atomic(self.directory / "config.json", {**config, "tool_timeout_s": self.timeout})
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(Path(__file__).resolve()), json.dumps(config)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        deadline = time.monotonic() + 10
        while not ready.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError("Execution guard did not start")
            time.sleep(0.02)

    def begin(self, tool):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Execution guard is unavailable; refusing to start a tool")
        self.serial += 1
        atomic(self.control, {"active": {"id": self.serial, "tool": tool,
                                         "started_at": time.time(),
                                         "deadline": time.monotonic() + self.timeout}})

    def end(self):
        atomic(self.control, {"active": None})

    def stop(self):
        if self.process is None:
            return
        atomic(self.control, {"active": None, "stop": True})
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None


if __name__ == "__main__":
    watch(json.loads(sys.argv[1]))
