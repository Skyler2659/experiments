"""Read-only RGB-D lift assessment. No motion, world edits, GT or automatic masks."""

from pathlib import Path
import json
import numpy as np


def _inside(p, root):
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _path(root, path, base=None):
    p = Path(path)
    if not p.is_absolute():
        p = (root if str(p).startswith(("robot_io/", "work/")) else (base or root)) / p
    p = p.resolve()
    if not _inside(p, root):
        raise ValueError("Evidence must be inside the episode")
    return p


def load_capture(root, path):
    p = _path(root, path)
    if not _inside(p, root / "robot_io/captures"):
        raise ValueError("Expected a robot RGB-D capture")
    return p, json.loads(p.read_text())


def _tcp(cap, arm):
    s = cap["robot_state"]
    rows = s.get("arms", [s])
    choices = [r for r in rows if r.get("arm", 0) == arm or r.get("arm_name") == arm]
    if len(choices) != 1:
        raise ValueError("Unknown or ambiguous arm")
    row = choices[0]
    p = row["eef_pose_W"]
    xyz = np.asarray(p["translation_m"], float)
    q = np.asarray(p["quaternion_xyzw"], float)
    if (
        xyz.shape != (3,)
        or q.shape != (4,)
        or not np.isfinite(xyz).all()
        or not np.isfinite(q).all()
        or not np.isclose(np.linalg.norm(q), 1, atol=0.01)
    ):
        raise ValueError("Invalid measured TCP pose")
    return (
        xyz,
        q / np.linalg.norm(q),
        row.get("robot_state_ref", s.get("robot_state_ref")),
        row.get("last_gripper_command", s.get("last_gripper_command")),
    )


def masked_cloud(root, cap_path, view, mask_path, min_points):
    mask_path = _path(root, mask_path)
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path, allow_pickle=False)
    else:
        from PIL import Image

        mask = np.asarray(Image.open(mask_path))
    depth = np.load(_path(root, view["depth_uri"], cap_path.parent), allow_pickle=False)
    if mask.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Object mask and depth shape must match")
    if mask.dtype != bool and not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("Expected binary object mask")
    if view.get("depth_convention", "camera_z_m") != "camera_z_m":
        raise ValueError("Only calibrated camera-z depth in meters is supported")
    K = np.asarray(view["K"], float)
    T = np.asarray(view["T_W_C"], float)
    if (
        K.shape != (3, 3)
        or T.shape != (4, 4)
        or not np.isfinite(K).all()
        or not np.isfinite(T).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1])
    ):
        raise ValueError("Invalid camera calibration")
    if (
        not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1)
    ):
        raise ValueError("Expected rigid camera transform")
    y, x = np.where(mask.astype(bool) & np.isfinite(depth) & (depth > 0))
    if len(x) < min_points:
        return None
    points = (np.column_stack([x, y, np.ones_like(x)]) @ np.linalg.inv(K).T) * depth[
        y, x, None
    ]
    points = points @ T[:3, :3].T + T[:3, 3]
    return {
        "center": np.median(points, axis=0),
        "extent": np.quantile(points, 0.9, axis=0) - np.quantile(points, 0.1, axis=0),
        "points": len(points),
        "T": T,
        "mask": str(mask_path),
    }


def assess_grasp(
    root,
    *,
    before_capture,
    after_capture,
    before_masks,
    after_masks,
    object_id,
    arm=0,
    object_before_capture=None,
    closure_request=None,
    min_tcp_lift_m=0.02,
    min_object_lift_m=0.015,
    motion_tolerance_m=0.008,
    stationary_tolerance_m=0.005,
    min_points=32,
    min_positive_views=2
):
    """Compare agent-selected object pixels to measured TCP motion.

    before_capture is after closure and before the agent-chosen lift. Optional
    object_before_capture supplies an earlier unobstructed object baseline.
    Positive evidence needs >=2 stationary camera views. Relative rotation,
    occlusion/visibility changes or inconsistent views yield INCONCLUSIVE.
    Masks/identity are explicitly agent-supplied, not native segmentation.
    """
    root = Path(root).resolve()
    for name, value in [
        ("min_tcp_lift_m", min_tcp_lift_m),
        ("min_object_lift_m", min_object_lift_m),
        ("motion_tolerance_m", motion_tolerance_m),
        ("stationary_tolerance_m", stationary_tolerance_m),
    ]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(name + " must be positive and finite")
    if (
        isinstance(min_points, bool)
        or not isinstance(min_points, int)
        or min_points < 3
    ):
        raise ValueError("Invalid min_points")
    if (
        isinstance(min_positive_views, bool)
        or not isinstance(min_positive_views, int)
        or min_positive_views < 2
    ):
        raise ValueError("Positive lift evidence needs at least two views")
    bp, before = load_capture(root, before_capture)
    ap, after = load_capture(root, after_capture)
    if bp == ap:
        raise ValueError("Need distinct before and after captures")
    op, objbefore = (
        load_capture(root, object_before_capture)
        if object_before_capture
        else (bp, before)
    )
    p0, q0, ref0, cmd0 = _tcp(before, arm)
    p1, q1, ref1, cmd1 = _tcp(after, arm)
    delta = p1 - p0
    rotation = float(2 * np.arccos(np.clip(abs(q0 @ q1), 0, 1)))
    closure_source = "measured capture command" if cmd1 in ["CLOSE", "HOLD"] else None
    if cmd1 is None and closure_request is not None:
        request_path = _path(root, closure_request)
        if not _inside(request_path, root / "robot_io/requests"):
            raise ValueError("Expected a public gripper request")
        request = json.loads(request_path.read_text())
        args = request["args"]
        receipt_path = root / "robot_io/responses" / (request["id"] + ".json")
        receipt = json.loads(receipt_path.read_text())
        requested_arm = args.get("arm", 0)
        requested = (
            [args]
            if request["op"] == "gripper"
            else args.get("commands", []) if request["op"] == "gripper_group" else []
        )
        closed = any(
            c.get("mode") == "CLOSE" and c.get("arm", requested_arm) == arm
            for c in requested
        )

        def counter(value):
            state = value.get("robot_state", value)
            ref = state.get("robot_state_ref", "")
            if "control_steps" in state:
                return state["control_steps"]
            try:
                return int(ref.rsplit(":", 1)[1])
            except (ValueError, IndexError):
                return None

        end = counter(after["robot_state"])
        closed_at = counter(receipt)
        before_at = counter(before["robot_state"])
        if (
            closed
            and receipt.get("status") in ["EXECUTED", "EPISODE_SUCCESS"]
            and all(v is not None for v in [end, closed_at, before_at])
            and closed_at <= before_at <= end
        ):
            closure_source = str(request_path)
            # An explicit later open supersedes the old close receipt.
            for path in (root / "robot_io/requests").glob("*.json"):
                r = json.loads(path.read_text())
                a = r.get("args", {})
                commands = (
                    [a]
                    if r.get("op") == "gripper"
                    else a.get("commands", []) if r.get("op") == "gripper_group" else []
                )
                if not any(
                    c.get("mode") == "OPEN" and c.get("arm", a.get("arm", 0)) == arm
                    for c in commands
                ):
                    continue
                rp = root / "robot_io/responses" / (r["id"] + ".json")
                if rp.exists():
                    when = counter(json.loads(rp.read_text()))
                    if when is not None and closed_at < when <= end:
                        closure_source = None
                        break
    result = {
        "schema": "codeworld-grasp-observation-v1",
        "object_id": object_id,
        "arm": arm,
        "status": "INCONCLUSIVE",
        "evidence": {
            "before_capture": str(bp),
            "after_capture": str(ap),
            "object_before_capture": str(op),
        },
        "mask_identity": "agent_selected_object_pixels",
        "native_object_state_used": False,
        "closure_evidence": closure_source,
        "tcp_delta_W_m": delta.tolist(),
        "tcp_rotation_rad": rotation,
        "views": [],
        "holding_certified": False,
        "robot_actions_issued": 0,
        "world_updated": False,
    }
    if ref0 == ref1 or delta[2] < min_tcp_lift_m or rotation > np.deg2rad(5):
        return {
            **result,
            "reason": "Need a measured upward probe with less than 5 degrees TCP rotation",
        }
    aviews = {v["view_id"]: v for v in after["views"]}
    bviews = {v["view_id"]: v for v in objbefore["views"]}
    for vid in sorted(set(before_masks) & set(after_masks) & set(aviews) & set(bviews)):
        b = masked_cloud(root, op, bviews[vid], before_masks[vid], min_points)
        a = masked_cloud(root, ap, aviews[vid], after_masks[vid], min_points)
        row = {"view_id": vid, "usable": False}
        if b is None or a is None:
            row["reason"] = "Too few valid target depth pixels"
        elif not np.allclose(b["T"], a["T"], atol=1e-5):
            row["reason"] = "Camera moved; visible-centroid tracking is not reliable"
        elif np.any(
            abs(a["extent"] - b["extent"])
            > np.maximum(0.01, 0.5 * np.maximum(a["extent"], b["extent"]))
        ):
            row["reason"] = "Object visibility/shape changed too much"
        else:
            d = a["center"] - b["center"]
            row.update(
                usable=True,
                object_delta_W_m=d.tolist(),
                tcp_follow_error_m=float(np.linalg.norm(d - delta)),
                before_points=b["points"],
                after_points=a["points"],
                before_mask=b["mask"],
                after_mask=a["mask"],
            )
        result["views"].append(row)
    good = [r for r in result["views"] if r["usable"]]
    if not good:
        return {**result, "reason": "No reliable stationary-camera target measurement"}
    ds = np.asarray([r["object_delta_W_m"] for r in good])
    median = np.median(ds, axis=0)
    result["object_delta_W_m"] = median.tolist()
    if np.max(np.linalg.norm(ds - median, axis=1)) > motion_tolerance_m:
        return {**result, "reason": "Selected views disagree about object motion"}
    if np.all(abs(ds[:, 2]) <= stationary_tolerance_m):
        return {
            **result,
            "status": "NOT_LIFTED",
            "reason": "TCP rose but the selected object surfaces stayed at the previous height",
        }
    distinct = []
    for row in good:
        T = np.asarray(bviews[row["view_id"]]["T_W_C"], float)
        if all(np.linalg.norm(T[:3, 3] - other[:3, 3]) >= 0.02 for other in distinct):
            distinct.append(T)
    result["independent_view_count"] = len(distinct)
    if (
        len(distinct) >= min_positive_views
        and closure_source is not None
        and np.all(ds[:, 2] >= min_object_lift_m)
        and all(r["tcp_follow_error_m"] <= motion_tolerance_m for r in good)
    ):
        return {
            **result,
            "status": "LIFT_EVIDENCE",
            "reason": "Object pixels rose with the measured TCP in at least two views; this does not establish long-term stability or force closure",
        }
    return {
        **result,
        "reason": "Lift/co-motion/closure evidence is incomplete; do not infer an attachment",
    }
