"""Absolute robot-frame command validation for the current mesh planner."""
import math
from typing import Any
from .errors import Real2CodeError
from .math3d import transform

def validate_open_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict) or set(action) != {"kind", "reason", "expected_observation", "target_pose_W", "gripper", "object_id"}:
        raise Real2CodeError("Micro action has an invalid interface")
    if action["object_id"] is not None and (not isinstance(action["object_id"], str) or not action["object_id"].strip()):
        raise Real2CodeError("contact object ID must be a nonempty string")
    if action["kind"] not in {"MOVE_TCP", "SET_GRIPPER", "GOAL_DONE", "TASK_DONE", "REPLAN"} or any(not isinstance(action[key], str) or not action[key].strip() for key in ("reason", "expected_observation")):
        raise Real2CodeError("Micro action needs kind, reason and expected observation")
    if action["kind"] == "MOVE_TCP":
        if action["gripper"] is not None or not isinstance(action["target_pose_W"], dict):
            raise Real2CodeError("MOVE_TCP needs only an absolute W-frame pose")
        pose = action["target_pose_W"]
        if set(pose) != {"translation_m", "quaternion_xyzw"} or len(pose["translation_m"]) != 3 or len(pose["quaternion_xyzw"]) != 4:
            raise Real2CodeError("invalid W-frame TCP pose")
        if any(type(x) not in (int, float) or not math.isfinite(x) for values in pose.values() for x in values):
            raise Real2CodeError("TCP pose contains nonfinite coordinates")
        transform(pose)
    elif action["kind"] == "SET_GRIPPER":
        if action["target_pose_W"] is not None or action["gripper"] not in {"OPEN", "CLOSE"}:
            raise Real2CodeError("SET_GRIPPER needs only OPEN or CLOSE")
    elif action["target_pose_W"] is not None or action["gripper"] is not None or action["object_id"] is not None:
        raise Real2CodeError("control-free decision must not carry a robot command")
    return action
