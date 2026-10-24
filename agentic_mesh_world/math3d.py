"""Coordinate conversions used by the public contract.

T_A_B maps a point expressed in B into A. External quaternions are xyzw.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import Real2CodeError


def _vec(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise Real2CodeError(f"{name} must be {size} finite numbers")
    return array


def normalize_quaternion(xyzw: Any) -> np.ndarray:
    q = _vec(xyzw, 4, "quaternion xyzw")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise Real2CodeError("zero quaternion")
    return q / norm


def rotation_from_xyzw(xyzw: Any) -> np.ndarray:
    x, y, z, w = normalize_quaternion(xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def xyzw_from_rotation(rotation: Any) -> list[float]:
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise Real2CodeError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(max(0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) * 2
            q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif i == 1:
            s = math.sqrt(max(0, 1 + r[1, 1] - r[0, 0] - r[2, 2])) * 2
            q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = math.sqrt(max(0, 1 + r[2, 2] - r[0, 0] - r[1, 1])) * 2
            q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    result = normalize_quaternion(q)
    if result[3] < 0:  # deterministic representation; q and -q are equivalent
        result = -result
    return result.tolist()


def transform(pose: dict[str, Any]) -> np.ndarray:
    translation = _vec(pose.get("translation_m"), 3, "translation_m")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_from_xyzw(pose.get("quaternion_xyzw"))
    matrix[:3, 3] = translation
    return matrix


def pose_from_transform(matrix: np.ndarray) -> dict[str, list[float]]:
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise Real2CodeError("transform must be finite 4x4")
    return {"translation_m": matrix[:3, 3].tolist(), "quaternion_xyzw": xyzw_from_rotation(matrix[:3, :3])}


def inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def apply(matrix: np.ndarray, point: Any) -> np.ndarray:
    return matrix[:3, :3] @ _vec(point, 3, "point") + matrix[:3, 3]
