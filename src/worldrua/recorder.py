"""Run-only call/state tracing. No ground-truth reads and no extra robot calls."""

import dataclasses
import functools
import hashlib
import itertools
import json
from pathlib import Path
import threading
import time


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        if getattr(value, "size", 0) > 10000:
            return {"omitted_array_shape": list(value.shape)}
        return value.tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def install(root):
    from pyrualean._robot import RobotBase
    from pyrualean.hosts.xpolicylab_bridge import Bridge
    from privilege.pyrualean import AgentWorld
    from privilege.query import Snapshot

    root = Path(root)
    (root / "states").mkdir(parents=True, exist_ok=True)
    stream = (root / "calls.jsonl").open("a", buffering=1)
    serial = itertools.count()
    lock = threading.RLock()
    local = threading.local()

    def state_ref(world, frozen=None):
        if world is None:
            return None
        blob = json.dumps(frozen if frozen is not None else world.state(), sort_keys=True).encode()
        key = hashlib.sha256(blob).hexdigest()
        path = root / "states" / f"{key}.json"
        if not path.exists():
            path.write_bytes(blob)
        return key

    def context(obj):
        if isinstance(obj, RobotBase):
            api = obj._action_observer
            world = getattr(api, "_world", None)
            robo = obj
        else:
            world = obj._world
            robo = getattr(obj, "_robo", None)
        return {
            "world_state": state_ref(world),
            "snapshot_state": state_ref(world, obj._state) if isinstance(obj, Snapshot) else None,
            "robot_state": plain(getattr(robo, "_last_state", None)),
            "ledger_length": len(robo._ledger) if hasattr(robo, "_ledger") else None,
        }

    def write(row):
        with lock:
            stream.write(json.dumps(plain(row), allow_nan=False) + "\n")

    def wrap(cls, name, kind):
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            call = next(serial)
            parent = getattr(local, "call", None)
            local.call = call
            error = None
            result = None
            try:
                write(
                    {
                        "event": "begin",
                        "id": call,
                        "parent": parent,
                        "time_ns": time.time_ns(),
                        "kind": kind,
                        "method": name,
                        "args": args,
                        "kwargs": kwargs,
                        **context(self),
                    }
                )
            except Exception as exc:
                write({"event": "logger_error", "id": call, "error": repr(exc)})
            try:
                result = original(self, *args, **kwargs)
                return result
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    write(
                        {
                            "event": "end",
                            "id": call,
                            "time_ns": time.time_ns(),
                            "error": error,
                            "result": result
                            if name not in {"state", "mesh", "part_mesh", "snapshot"}
                            else None,
                            **context(self),
                        }
                    )
                except Exception as exc:
                    write({"event": "logger_error", "id": call, "error": repr(exc)})
                local.call = parent

        setattr(cls, name, wrapped)

    for name in ("_readonly", "_stateful"):
        wrap(RobotBase, name, "robot")
    for cls, kind in ((AgentWorld, "world"), (Snapshot, "snapshot")):
        for name, value in list(vars(cls).items()):
            if not name.startswith("_") and callable(value):
                wrap(cls, name, kind)
    original = Bridge.update_obs

    @functools.wraps(original)
    def update_obs(self, obs):
        result = original(self, obs)
        import numpy as np

        cameras = {}
        for name, cam in (obs.get("vision") or {}).items():
            if cam.get("color") is not None:
                a = np.ascontiguousarray(np.asarray(cam["color"])[..., :3], dtype=np.uint8)
                cameras[name] = hashlib.sha256(a.tobytes()).hexdigest()
        with (root / "bridge.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    plain(
                        {
                            "time_ns": time.time_ns(),
                            "status": self._status,
                            "state": obs.get("state"),
                            "image_state": obs.get("image_state"),
                            "rgb_sha256": cameras,
                        }
                    )
                )
                + "\n"
            )
        return result

    Bridge.update_obs = update_obs
