"""Compile an executed RCWM export into a persistent geometry workspace."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .world import atomic, rigid


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot_ignore(directory, names):
    ignored = set(shutil.ignore_patterns(
        "*.log", ".sid", "events.jsonl", "correction", "correction_*",
        "feedback", "trace", ".adaptive_best", ".best",
    )(directory, names))
    # Match the exporter's rglob traversal: keep source-file links, but do not
    # descend into directory links created by temporary render/validation rigs.
    # Such links can point back to an ancestor or to the entire shared runtime.
    ignored.update(name for name in names
                   if (Path(directory) / name).is_symlink()
                   and (Path(directory) / name).is_dir())
    return ignored


def object_pose(matrix):
    """Extract a rigid pose; executed shape scale stays baked into local geometry."""
    affine = np.asarray(matrix, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("Invalid executed object transform")
    u, singular, vt = np.linalg.svd(affine[:3, :3])
    if singular.min() <= 1e-12:
        raise ValueError("Singular executed object transform")
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, affine[:3, 3]
    return rigid(pose)


def build(run, export, output, runtime, frame="world"):
    run, export, out, runtime = map(lambda x: Path(x).resolve(), (run, export, output, runtime))
    if out.exists():
        raise ValueError("Use a new output directory")
    binding = json.loads((export / "executed_binding.json").read_text())
    for rel, h in binding["source_files"].items():
        if digest(run / rel) != h:
            raise ValueError("Export no longer matches source: " + rel)
    inventory = json.loads((export / "object_inventory.json").read_text())
    if {r["object_id"] for r in inventory} != set(binding["objects"]):
        raise ValueError("Object export/tree mismatch")
    params_rel = "fractal/scene/parameters.json"
    if not (run / params_rel).exists():
        params_rel = "parameters.json"
    params = json.loads((run / params_rel).read_text())
    if set(params["objects"]) != set(binding["objects"]):
        raise ValueError("Declared/executed IDs differ")
    out.mkdir(parents=True)
    for d in ["objects", "geometry", "state", "scenarios", "observations", "work", "source"]:
        (out / d).mkdir()
    snapshot = out / "source/runtime/runs" / run.name
    shutil.copytree(
        run,
        snapshot,
        ignore=snapshot_ignore,
    )
    three = runtime / ".render-tools/node_modules/three"
    shutil.copytree(three, out / "source/runtime/.render-tools/node_modules/three")
    shutil.copy2(export / "executed_binding.json", out / "source/executed_binding.json")
    shutil.copy2(export / "scene.glb", out / "geometry/initial_scene.glb")
    objects = {}
    state = {}
    checks = []
    for item in inventory:
        oid = item["object_id"]
        if "/" in oid or oid in [".", ".."]:
            raise ValueError("Unsafe object ID")
        # Three.js stores matrices column-major. Its executed matrix wins over textual declarations.
        t = object_pose(
            np.array(binding["objects"][oid]["matrix_world_column_major"]).reshape(4, 4, order="F")
        )
        asset = trimesh.load(export / item["mesh_ref"], force="scene", process=False)
        before = asset.to_geometry()
        if not len(before.vertices) or not np.isfinite(before.vertices).all():
            raise ValueError("Invalid geometry: " + oid)
        # Three.js permits RGB COLOR_0; trimesh's GLB writer expects RGBA.
        # Preserve the original RGB and add opaque alpha, without changing geometry/materials.
        for geom in asset.geometry.values():
            attrs = getattr(getattr(geom, "visual", None), "vertex_attributes", {})
            if "color" in attrs and attrs["color"].ndim == 2 and attrs["color"].shape[1] == 3:
                colors = attrs["color"]
                alpha = (
                    np.iinfo(colors.dtype).max if np.issubdtype(colors.dtype, np.integer) else 1.0
                )
                attrs["color"] = np.column_stack(
                    [colors, np.full(len(colors), alpha, dtype=colors.dtype)]
                )
        asset.apply_transform(np.linalg.inv(t))
        dest = out / "geometry" / oid
        dest.mkdir()
        asset.export(dest / "local.glb")
        local = trimesh.load(dest / "local.glb", force="mesh", process=False)
        rebuilt = local.copy()
        rebuilt.apply_transform(t)
        # Exporters may reorder vertices; compare both directions without sort tie instability.
        if len(before.vertices) != len(rebuilt.vertices) or len(before.faces) != len(rebuilt.faces):
            raise ValueError("Geometry cardinality changed: " + oid)
        # Duplicate GLB vertices do not change the nearest-surface equivalence check.
        # Deduplicate before KD queries to avoid quadratic searches in coincident leaves.
        original_unique = np.unique(before.vertices, axis=0)
        rebuilt_unique = np.unique(rebuilt.vertices, axis=0)
        err = float(
            max(
                cKDTree(original_unique).query(rebuilt_unique)[0].max(),
                cKDTree(rebuilt_unique).query(original_unique)[0].max(),
            )
        )
        bound_err = float(np.max(np.abs(before.bounds - rebuilt.bounds)))
        if bound_err > 2e-6 or err > 2e-6:
            raise ValueError(f"Geometry roundtrip mismatch {oid}: {err}")
        declaration = params["objects"][oid]
        nodes = [n for n in binding["nodes"] if n["object_id"] == oid]
        row = {
            "object_id": oid,
            "movable": declaration.get("movable", False) if oid != "known_robot" else False,
            "role": "robot_snapshot" if oid == "known_robot" else (
                "fixed_obstacle" if declaration.get("role") == "fixed_obstacle" else "rigid_object"),
            "geometry": {
                "local_mesh": str((dest / "local.glb").relative_to(out)),
                "local_bounds_m": local.bounds.tolist(),
                "vertices": len(local.vertices),
                "faces": len(local.faces),
            },
            "initial_T_W_object": t.tolist(),
            "source_binding": binding["objects"][oid],
            "source_nodes": nodes,
            "declared": declaration,
            "declaration_status": (
                "source hypotheses; not geometry-certified; initial declarations are not live state"
            ),
            "source_parameters": str((snapshot / params_rel).relative_to(out)),
        }
        atomic(out / "objects" / f"{oid}.json", row)
        objects[oid] = f"objects/{oid}.json"
        state[oid] = {
            "T_W_object": t.tolist(),
            "basis": "executed_reconstruction",
            "evidence": "source/executed_binding.json",
        }
        checks.append(
            {"object_id": oid, "max_vertex_error_m": err, "max_bounds_error_m": bound_err}
        )
    bound = {
        str(p.relative_to(out)): digest(p)
        for folder in ["source", "geometry", "objects"]
        for p in sorted((out / folder).rglob("*"))
        if p.is_file()
    }
    revision = hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
    atomic(
        out / "world.json",
        {
            "schema": "agent-codeworld-v1",
            "geometry_revision": revision,
            "frame": frame,
            "units": "m",
            "matrix_convention": "row-major T_W_object; column vectors",
            "objects": objects,
            "bound_files": bound,
            "scope": "rigid static scene; no articulation or dynamics",
            "robot_execution_connected": False,
            "geometry_accuracy_verified": False,
        },
    )
    atomic(
        out / "state/current.json",
        {
            "geometry_revision": revision,
            "revision": 0,
            "branch": "current",
            "objects": state,
            "history": [],
        },
    )
    atomic(
        out / "build_report.json",
        {
            "status": "PASS",
            "objects": len(objects),
            "geometry_roundtrip": checks,
            "scope": "source binding and executed geometry equivalence, not task success",
        },
    )
    return {"workspace": str(out), "objects": len(objects), "geometry_revision": revision}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--export", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--frame", default="world")
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
