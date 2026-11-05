"""Compact, current scene input for a host agent; full meshes stay queryable."""

import copy
import json

from .world import read


class SceneContext:
    def __init__(self, world):
        self.world = world
        self._bounds = {}

    def packet(self, *, detailed=False):
        snapshot = self.world.snapshot()
        state = snapshot._state
        objects = []
        for oid in snapshot.objects():
            meta = read(self.world.root / self.world.manifest["objects"][oid])
            current = state["objects"][oid]
            key = json.dumps([current["T_W_object"], current.get("assembly")], sort_keys=True)
            if oid not in self._bounds or self._bounds[oid][0] != key:
                mesh = snapshot.mesh(oid)
                self._bounds[oid] = (
                    key,
                    mesh.bounds.tolist(),
                    mesh.centroid.tolist(),
                    len(mesh.vertices),
                    len(mesh.faces),
                )
            assembly = current.get("assembly")
            parts = (
                list(assembly["parts"])
                if assembly
                else [n.get("name") for n in meta.get("source_nodes", []) if n.get("name")]
            )
            objects.append(
                {
                    "object_id": oid,
                    "label": meta.get("label", oid),
                    "role": meta.get("role", "rigid_object"),
                    "movable": meta.get("movable", meta.get("declared", {}).get("movable")),
                    "T_W_object": copy.deepcopy(current["T_W_object"]),
                    "bounds_W_m": copy.deepcopy(self._bounds[oid][1]),
                    "centroid_W_m": copy.deepcopy(self._bounds[oid][2]),
                    "geometry": {
                        "local_mesh": None
                        if assembly
                        else meta.get("geometry", {}).get("local_mesh"),
                        "representation": "agent_assembly" if assembly else "reconstructed_mesh",
                        "vertices": self._bounds[oid][3],
                        "faces": self._bounds[oid][4],
                        "geometry_revision": assembly["geometry_revision"]
                        if assembly
                        else self.world.manifest["geometry_revision"],
                    },
                    "part_count": len(parts),
                    "part_names": parts[:32],
                    "part_names_truncated": len(parts) > 32,
                    "parameters": copy.deepcopy(assembly.get("parameters", {})) if assembly else {},
                    "basis": current.get("basis"),
                    "pose_refresh": copy.deepcopy(current.get("pose_refresh")),
                    "review": copy.deepcopy(current.get("review", {})),
                    "sensor_validation": copy.deepcopy(current.get("sensor_validation")),
                    "source_unknowns": meta.get("declared", {}).get("unknowns", []),
                    "source_declarations": copy.deepcopy(meta.get("declared", {})),
                }
            )
        if not detailed:
            # Full precision geometry and source declarations remain available through
            # object()/snapshot(). Deliver ALL current objects, not a truncated prefix.
            compact = []
            for obj in objects:
                if obj["role"] == "robot_snapshot":
                    continue
                row = {k: obj[k] for k in ("object_id", "role", "movable", "T_W_object", "bounds_W_m", "basis")}
                if obj["source_unknowns"]:
                    row["geometry_unknowns"] = obj["source_unknowns"]
                row["review"] = {k: v.get("status") for k, v in obj["review"].items()}
                row["geometry_revision"] = obj["geometry"]["geometry_revision"][:12]
                if obj['pose_refresh']:
                    row['pose_refresh'] = obj['pose_refresh']
                if obj["parameters"]:
                    row["parameters"] = obj["parameters"]
                compact.append(row)
            objects = rounded(compact)
        return {
            "schema": "privilege-agent-context-v1",
            "revision": snapshot.revision,
            "geometry_revision": self.world.manifest["geometry_revision"],
            "frame": snapshot.frame,
            "units": "m",
            "objects": objects,
            "attachments": rounded(copy.deepcopy(state.get("attachments", {}))),
            "grasp_review": rounded(copy.deepcopy(state.get("grasp_review", {}))),
            "relations": copy.deepcopy(state.get("relations", {})),
            "recovery_history": copy.deepcopy(state.get("recovery_history", [])),
            "pose_refresh": {oid: copy.deepcopy(row['pose_refresh']) for oid, row in
                             state['objects'].items() if row.get('pose_refresh')},
            "height_alignment": rounded({k: state["height_alignment"][k] for k in
                ("table_object_id", "target_z_m", "delta_z_m", "source")}) if state.get("height_alignment") else None,
            "geometry_accuracy_verified": False,
            "robot_geometry": "initial frozen reference; use robo.state() for live robot",
            "declarations": "source hypotheses, not live measurements",
        }


def rounded(value):
    """Six decimal places for communication only; stored metric data is unchanged."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value
