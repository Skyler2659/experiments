"""Run-only simulator observer. Never inserts privileged fields into observations."""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def transform(position, quaternion):
    p = np.asarray(plain(position), dtype=float).reshape(3)
    q = np.asarray(plain(quaternion), dtype=float).reshape(4)
    w, x, y, z = q / np.linalg.norm(q)
    t = np.eye(4)
    t[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    t[:3, 3] = p
    return t


def local_vertices(stage, path):
    """Visible USD vertices in a metric rigid root frame, including root scale."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_t = np.asarray(cache.GetLocalToWorldTransform(root), float).T
    scales = np.linalg.norm(root_t[:3, :3], axis=0)
    axes = root_t[:3, :3] / scales
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("Sheared USD root")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("Expected metre USD stage")
    scale = np.diag([*scales, 1.0])
    inverse = np.linalg.inv(root_t)
    pieces = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        visible = UsdGeom.Imageable(prim)
        if visible and (
            visible.ComputeVisibility() == UsdGeom.Tokens.invisible
            or visible.GetPurposeAttr().Get() in ("guide", "proxy")
        ):
            continue
        kind = prim.GetTypeName()
        if kind == "Mesh":
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        elif kind == "Cube":
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            vertices = (
                np.array([[a, b, c] for a in (-0.5, 0.5) for b in (-0.5, 0.5) for c in (-0.5, 0.5)])
                * size
            )
        else:
            continue
        t = scale @ inverse @ np.asarray(cache.GetLocalToWorldTransform(prim), float).T
        pieces.append(vertices @ t[:3, :3].T + t[:3, 3])
    if not pieces:
        raise ValueError(f"No visible mesh vertices: {path}")
    return np.concatenate(pieces)


class Recorder:
    def __init__(self, env):
        import omni.usd

        self.root = Path(os.environ["PYRUA_AUDIT_ROOT"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = (self.root / "truth.jsonl").open("a", buffering=1)
        self.counter = 0
        self.items = {}
        stage = omni.usd.get_context().get_stage()
        lm = env.scene_manager.layout_manager
        metadata = {}
        for env_idx in range(env.num_envs):
            for label in lm.get_labels_by_prefix(env_idx, ""):
                name = lm.get_instance_name(env_idx, label)
                if name is None:
                    raise ValueError(f"Missing scene label {label}")
                obj = lm.get_scene_object(env_idx, name)
                vertices = local_vertices(stage, obj.prim_path)
                key = f"env{env_idx}_{label}"
                self.items[key] = (env_idx, label, name, vertices)
                np.savez_compressed(self.root / f"{key}_vertices.npz", vertices=vertices)
                metadata[key] = {
                    "env_idx": env_idx,
                    "label": label,
                    "instance": name,
                    "prim_path": obj.prim_path,
                    "description": plain(lm.get_label_descriptions(env_idx, label=label)),
                    "local_bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                    "bounds_basis": "initial mesh transformed by current root; articulated bounds not live",
                }
        (self.root / "truth_objects.json").write_text(json.dumps(metadata, indent=2))

    def record(self, env, observations, phase):
        start = time.perf_counter()
        row = {
            "index": self.counter,
            "phase": phase,
            "time_ns": time.time_ns(),
            "native_steps": plain(env.take_action_cnt),
            "objects": {},
            "observations": {},
        }
        lm = env.scene_manager.layout_manager
        for key, (env_idx, label, name, vertices) in self.items.items():
            p, q = lm.get_instance_pose(env_idx, inst_name=name, relative=True)
            t = transform(p, q)
            points = vertices @ t[:3, :3].T + t[:3, 3]
            row["objects"][key] = {
                "T_env_object": t.tolist(),
                "env_origin": plain(env.scene_manager.env_origins[env_idx]),
                "bounds_env": [points.min(0).tolist(), points.max(0).tolist()],
            }
            obj = lm.get_scene_object(env_idx, name)
            if hasattr(obj, "get_current_joint_positions"):
                row["objects"][key]["joint_positions"] = plain(obj.get_current_joint_positions())
        for idx, obs in (observations or {}).items():
            cameras = {}
            for name, camera in (obs.get("vision") or {}).items():
                rgb = camera.get("color")
                if rgb is not None:
                    a = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
                    cameras[name] = {
                        "rgb_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "shape": list(a.shape),
                    }
            row["observations"][str(idx)] = {"state": plain(obs.get("state")), "cameras": cameras}
        row["sampling_ms"] = (time.perf_counter() - start) * 1000
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.counter += 1


def record(env, observations=None, phase="observation"):
    # Logging failures are explicit; never change the policy's result or motion.
    try:
        if not hasattr(env, "_pyrualean_audit_recorder"):
            env._pyrualean_audit_recorder = Recorder(env)
        env._pyrualean_audit_recorder.record(env, observations, phase)
    except Exception as exc:
        root = Path(os.environ["PYRUA_AUDIT_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "truth_errors.jsonl").open("a") as f:
            f.write(
                json.dumps({"time_ns": time.time_ns(), "error": repr(exc), "phase": phase}) + "\n"
            )
