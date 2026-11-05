"""Create/import world data without copying an agent or robot runtime."""

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh

from .world import World, atomic, rigid


def from_meshes(objects, output, *, frame="world"):
    """Build from {id: {mesh: local_mesh_path, T_W_object: matrix, ...metadata}}."""
    if not objects or not frame:
        raise ValueError("Objects and their metric frame are required")
    out = Path(output).resolve()
    if out.exists():
        raise ValueError("Use a new output directory")
    prepared = {}
    for oid, spec in objects.items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", oid):
            raise ValueError("Invalid object ID")
        source = Path(spec["mesh"]).resolve()
        asset = trimesh.load(source, force="scene", process=False)
        mesh = asset.to_geometry()
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Invalid local geometry")
        prepared[oid] = (asset, rigid(spec.get("T_W_object", np.eye(4))).tolist(), spec)
    for name in ("objects", "geometry", "state", "source", "observations", "scenarios", "work"):
        (out / name).mkdir(parents=True, exist_ok=True)
    index, states = {}, {}
    for oid, (asset, transform, spec) in prepared.items():
        relative = f"geometry/{oid}.glb"
        asset.export(out / relative)
        meta = {
            "object_id": oid,
            "role": spec.get("role", "rigid_object"),
            "geometry": {"local_mesh": relative},
            "initial_T_W_object": transform,
            "source_nodes": [],
            "declared": copy.deepcopy(spec.get("declared", {})),
            "label": spec.get("label", oid),
            "movable": spec.get("movable", spec.get("declared", {}).get("movable")),
        }
        index[oid] = f"objects/{oid}.json"
        atomic(out / index[oid], meta)
        states[oid] = {"T_W_object": transform, "basis": "imported_reconstruction"}
    bound = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("objects", "geometry")
        for p in (out / folder).rglob("*")
        if p.is_file()
    }
    revision = hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
    atomic(
        out / "world.json",
        {
            "schema": "agent-codeworld-v1",
            "objects": index,
            "frame": frame,
            "units": "m",
            "matrix_convention": "row-major T_W_object; column vectors",
            "bound_files": bound,
            "geometry_revision": revision,
            "geometry_accuracy_verified": False,
        },
    )
    atomic(
        out / "state/current.json",
        {
            "revision": 0,
            "geometry_revision": revision,
            "branch": "current",
            "objects": states,
            "history": [],
        },
    )
    return World(out)


def import_world(source, output, *, frame, T_host_world=None):
    source = Path(source).resolve()
    original = World(source)
    original.verify_binding()
    if T_host_world is None and original.manifest["frame"] != frame:
        raise ValueError("Imported world frame differs; provide explicit T_host_world")
    t = rigid(T_host_world if T_host_world is not None else np.eye(4))
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("world.json", "objects", "geometry", "geometry_edits", "source"):
        item = source / name
        if item.is_dir():
            shutil.copytree(item, destination / name)
        elif item.is_file():
            shutil.copy2(item, destination / name)
    for name in ("state", "observations", "scenarios", "work"):
        (destination / name).mkdir()
    manifest = copy.deepcopy(original.manifest)
    state = original.state()
    for row in state["objects"].values():
        row["T_W_object"] = (t @ rigid(row["T_W_object"])).tolist()
        for key in (
            "sensor_validation",
            "oracle_validation",
            "last_observation_recorded_at_ns",
            "observed_state",
        ):
            row.pop(key, None)
        row.update(basis="imported_prior", review={"pose": {"status": "unreviewed"}})
    state = {
        "revision": 0,
        "branch": "current",
        "history": [],
        "objects": state["objects"],
        "geometry_revision": manifest["geometry_revision"],
    }
    manifest.update(frame=frame, import_transform=t.tolist())
    atomic(destination / "world.json", manifest)
    atomic(destination / "state/current.json", state)
    world = World(destination)
    world.verify_binding()
    return world
