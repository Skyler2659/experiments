"""Verify that the exported robot is the provided geometry in the same world frame."""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def verify_robot_asset(source, exported, tolerance_m=2e-5):
    original = trimesh.load(source, force="scene", process=False).to_geometry()
    actual = trimesh.load(exported, force="scene", process=False).to_geometry()
    if not len(original.vertices) or len(original.faces) != len(actual.faces):
        raise ValueError("Known robot mesh cardinality changed")
    first = np.unique(original.vertices, axis=0)
    second = np.unique(actual.vertices, axis=0)
    error = float(max(cKDTree(first).query(second)[0].max(), cKDTree(second).query(first)[0].max()))
    if not np.isfinite(error) or error > tolerance_m:
        raise ValueError("Known robot geometry or world transform changed: " + str(error))
    return {"max_vertex_error_m": error, "faces": len(actual.faces), "geometry_reused": True}
