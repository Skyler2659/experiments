"""One-time scene height alignment to an explicitly supplied host table prior."""

import copy
import fcntl
import time

import numpy as np

from .world import atomic, rigid


def align_table_height(world, *, table_object_id, target_z_m, source):
    """Translate reconstructed geometry together; calibrated robot/cameras stay fixed.

    Only current object transforms change. Bound source files, local geometry,
    shapes, rotations and relative scene placement are preserved.
    """
    target = float(target_z_m)
    if not np.isfinite(target):
        raise ValueError("Table target height must be finite")
    if world.object(table_object_id).get("role") == "robot_snapshot":
        raise ValueError("The table reference cannot be a calibrated robot")
    with (world.root / "state/update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = world.state()
        previous = state.get("height_alignment")
        if previous:
            if previous["table_object_id"] != table_object_id or previous["target_z_m"] != target:
                raise ValueError("This world already has a different initial height alignment")
            return copy.deepcopy(previous)
        table_top = float(world._mesh(table_object_id, state).bounds[1, 2])
        delta = target - table_top
        transform = np.eye(4)
        transform[2, 3] = delta
        moved, fixed = [], []
        for oid, row in state["objects"].items():
            if world.object(oid).get("role") == "robot_snapshot":
                fixed.append(oid)
                continue
            row["T_W_object"] = (transform @ rigid(row["T_W_object"])).tolist()
            row["basis_before_height_alignment"] = row.get("basis")
            row["basis"] = "host_table_height_aligned_prior"
            row.pop("sensor_validation", None)
            row.pop("oracle_validation", None)
            row.setdefault("review", {})["pose"] = {"status": "unreviewed"}
            moved.append(oid)
        receipt = {"table_object_id": table_object_id, "source_top_z_m": table_top,
                   "target_z_m": target, "delta_z_m": delta, "source": source,
                   "translated_objects": moved, "fixed_robot_objects": fixed,
                   "T_aligned_reconstructed": transform.tolist(),
                   "kind": "host_table_height_prior", "metric_validation": False}
        state["height_alignment"] = receipt
        state["revision"] += 1
        state["history"].append({"kind": "initial_table_height_alignment", "revision": state["revision"],
                                 **copy.deepcopy(receipt), "recorded_at_ns": time.time_ns()})
        atomic(world.state_path, state)
    return receipt


def table_object(world, configured=None):
    """Resolve the named table; ambiguous scenes should name it in host config."""
    aliases = {"tabletop", "table", "work_table", "worktable", "table_top"}
    ids = world.objects()
    if configured in ids:
        return configured
    if configured is not None and configured not in aliases:
        raise ValueError(f"Configured table object does not exist: {configured}")
    candidates = [oid for oid in ids if oid in aliases
                  or world.object(oid).get("role") in {"table", "tabletop", "work_surface"}]
    if len(candidates) != 1:
        raise ValueError("Set height_alignment.table_object_id to one unambiguous reconstructed table surface; "
                         f"candidates={candidates}")
    return candidates[0]
