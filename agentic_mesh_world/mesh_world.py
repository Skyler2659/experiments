"""Executable, mesh-backed scene program for the Real2Code reconstruction stage.

The program is the authority for asset transforms, articulated state, rendering,
and geometric queries.  A rendered scene and a query can therefore never use
independent object placements.  Coordinates are metres in the robot-base W frame.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import Real2CodeError
from .store import safe_id


def nearest_mesh_surface(mesh, point, *, return_normal=False):
    """Query valid triangles without modifying the authoritative textured mesh."""
    import trimesh
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise Real2CodeError("surface query needs a finite W-frame point")
    triangles = mesh.triangles
    areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = np.isfinite(triangles).all(axis=(1, 2)) & (areas2 > 1e-20)
    if not valid.any():
        raise Real2CodeError("mesh has no nondegenerate surface triangles")
    query_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[valid], process=False)
    closest, distances, triangle = trimesh.proximity.closest_point_naive(query_mesh, point[None, :])
    if not np.isfinite(closest).all() or not np.isfinite(distances).all():
        raise Real2CodeError("mesh surface query returned nonfinite geometry")
    if return_normal:
        return closest[0], float(distances[0]), query_mesh.face_normals[triangle[0]]
    return closest[0], float(distances[0])


def sha256(path: Path) -> str:
    path = Path(path)
    block = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            block.update(data)
    return block.hexdigest()


def pose_matrix(pose: dict[str, Any]) -> np.ndarray:
    """T_parent_child, with an xyzw quaternion."""
    translation = np.asarray(pose["translation_m"], dtype=float)
    quaternion = np.asarray(pose["quaternion_xyzw"], dtype=float)
    if translation.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise Real2CodeError("pose needs finite translation_m[3] and quaternion_xyzw[4]")
    norm = np.linalg.norm(quaternion)
    if abs(norm - 1.0) > 1e-4:
        raise Real2CodeError("pose quaternion must be normalized")
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def matrix_pose(matrix: np.ndarray) -> dict[str, list[float]]:
    from scipy.spatial.transform import Rotation
    return {"translation_m": matrix[:3, 3].tolist(), "quaternion_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()}


def _unit_axis(value: Any) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or abs(np.linalg.norm(axis) - 1) > 1e-4:
        raise Real2CodeError("joint axis_parent must be a unit 3-vector")
    return axis


class MeshCodeWorld:
    """Read one validated scene program and execute its geometry/state queries."""

    def __init__(self, source: dict[str, Any], root: str | Path):
        self.source = copy.deepcopy(source)
        self.root = Path(root).resolve()
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "MeshCodeWorld":
        path = Path(path).resolve()
        return cls(json.loads(path.read_text()), path.parent)

    def _validate(self) -> None:
        scene = self.source
        if scene.get("schema_version") != "mesh-code-v1" or scene.get("frame") != "robot_base" or scene.get("units") != "m":
            raise Real2CodeError("expected mesh-code-v1 scene in robot_base metres")
        if not isinstance(scene.get("nodes"), list) or not scene["nodes"]:
            raise Real2CodeError("scene needs at least one mesh-backed node")
        safe_id(scene.get("world_id"), "world_id")
        seen: set[str] = set()
        for node in scene["nodes"]:
            oid = safe_id(node.get("object_id"), "object_id")
            if oid in seen:
                raise Real2CodeError(f"duplicate object: {oid}")
            parent = node.get("parent_id")
            if parent is not None and parent not in seen:
                raise Real2CodeError(f"parent must precede child: {oid}")
            seen.add(oid)
            pose_matrix(node["pose_parent_object"])
            asset = node["asset"]
            relative = Path(asset["mesh_ref"])
            if relative.is_absolute() or ".." in relative.parts:
                raise Real2CodeError("mesh_ref must stay inside the scene package")
            path = self.root / relative
            if not path.is_file() or sha256(path) != asset.get("sha256"):
                raise Real2CodeError(f"mesh missing or hash mismatch: {oid}")
            up = asset.get("up_axis")
            if up not in {"Y", "Z"}:
                raise Real2CodeError("mesh up_axis must be Y or Z")
            transform = np.asarray(asset.get("T_object_mesh"), dtype=float)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
                raise Real2CodeError("asset needs finite 4x4 T_object_mesh")
            if np.linalg.det(transform[:3, :3]) <= 1e-10:
                raise Real2CodeError("T_object_mesh must preserve orientation and nonzero scale")
            joint = node.get("joint")
            if joint:
                if parent is None or joint.get("type") not in {"fixed", "prismatic", "revolute"}:
                    raise Real2CodeError("joint needs a parent and supported type")
                if joint["type"] != "fixed":
                    _unit_axis(joint.get("axis_parent"))
                    low, high = joint.get("limits", [None, None])
                    value = joint.get("value")
                    if any(not isinstance(x, (float, int)) or not np.isfinite(x) for x in (low, high, value)) or low > value or value > high:
                        raise Real2CodeError("joint value must lie inside finite limits")
            if "state" in node and node["state"] not in {"OBSERVED", "ESTIMATED", "UNKNOWN"}:
                raise Real2CodeError("invalid node state")
            regions = node.get("regions", [])
            if not isinstance(regions, list):
                raise Real2CodeError("regions must be a list")
            region_ids: set[str] = set()
            for region in regions:
                region_id = safe_id(region.get("region_id"), "region_id")
                if region_id in region_ids:
                    raise Real2CodeError(f"duplicate region: {region_id}")
                region_ids.add(region_id)
                if region.get("pose_O_R"):
                    pose_matrix(region["pose_O_R"])

        if scene.get('robot_integration',{}).get('status')=='BOUND_TO_MEASURED_FK':
            from .reconstruction_robot import prepare
            refs=scene['robot_integration']
            bound=prepare({'views':[]},self.root,{'state_uri':refs['state_ref'],'snapshot_uri':refs['snapshot_ref']})
            if scene.get('robot_state')!=bound['state']:raise Real2CodeError('Robot state changed without updating FK geometry')
            robot=self.node('known_robot')
            if not np.allclose(pose_matrix(robot['pose_parent_object']),np.eye(4)):raise Real2CodeError('Known robot base placement differs from W frame')
            provenance=robot['asset'].get('provenance',{})
            if provenance.get('state_sha256')!=sha256(bound['state_path']):raise Real2CodeError('Robot mesh is bound to a different measured state')

    def node(self, object_id: str) -> dict[str, Any]:
        for node in self.source["nodes"]:
            if node["object_id"] == object_id:
                return node
        raise Real2CodeError(f"unknown object: {object_id}")

    def object_transform(self, object_id: str) -> np.ndarray:
        node = self.node(object_id)
        parent = node.get("parent_id")
        base = self.object_transform(parent) if parent else np.eye(4)
        joint = node.get("joint")
        motion = np.eye(4)
        if joint and joint["type"] == "prismatic":
            motion[:3, 3] = _unit_axis(joint["axis_parent"]) * float(joint["value"])
        elif joint and joint["type"] == "revolute":
            from scipy.spatial.transform import Rotation
            motion[:3, :3] = Rotation.from_rotvec(_unit_axis(joint["axis_parent"]) * float(joint["value"])).as_matrix()
        return base @ motion @ pose_matrix(node["pose_parent_object"])

    def mesh_pieces(self, object_id: str):
        import trimesh
        node = self.node(object_id)
        asset = node["asset"]
        loaded = trimesh.load(self.root / asset["mesh_ref"], force="scene", process=False)
        pieces = [piece.copy() for piece in loaded.dump() if isinstance(piece, trimesh.Trimesh) and len(piece.faces)]
        if not pieces:
            raise Real2CodeError(f"mesh contains no triangles: {object_id}")
        transform = np.asarray(asset["T_object_mesh"], dtype=float)
        if asset["up_axis"] == "Y":
            y_to_z = np.eye(4)
            y_to_z[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
            transform = transform @ y_to_z
        world = self.object_transform(object_id) @ transform
        for piece in pieces:
            piece.apply_transform(world)
        return pieces

    def mesh(self, object_id: str):
        import trimesh
        return trimesh.util.concatenate(self.mesh_pieces(object_id))

    def bounds(self, object_id: str) -> list[list[float]]:
        return self.mesh(object_id).bounds.tolist()

    def surface_distance(self, object_id: str, point_W_m: list[float]) -> dict[str, Any]:
        """Unsigned nearest surface query on the actual visual mesh.

        This does not decide inside/outside, contact force, or collision safety.
        """
        closest, distance = nearest_mesh_surface(self.mesh(object_id), point_W_m)
        return {"distance_m": distance, "nearest_W_m": closest.tolist(),
                "kind": "unsigned_mesh_surface"}

    def joint_state(self, object_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.node(object_id).get("joint"))

    def region(self, object_id: str, region_id: str) -> dict[str, Any]:
        node = self.node(object_id)
        region = next((item for item in node.get("regions", []) if item.get("region_id") == region_id), None)
        if region is None:
            raise Real2CodeError(f"unknown region: {object_id}/{region_id}")
        result = copy.deepcopy(region)
        if region.get("pose_O_R"):
            result["pose_W_R"] = matrix_pose(self.object_transform(object_id) @ pose_matrix(region["pose_O_R"]))
        return result

    def set_joint(self, object_id: str, value: float, *, evidence_ref: str) -> "MeshCodeWorld":
        if not evidence_ref:
            raise Real2CodeError("joint update needs observation evidence")
        source = copy.deepcopy(self.source)
        node = next(item for item in source["nodes"] if item["object_id"] == object_id)
        joint = node.get("joint")
        if not joint or joint["type"] == "fixed" or not joint["limits"][0] <= value <= joint["limits"][1]:
            raise Real2CodeError("joint update outside declared model or limits")
        joint["value"] = float(value)
        joint["evidence_ref"] = evidence_ref
        node["state"] = "ESTIMATED"
        return MeshCodeWorld(source, self.root)

    def export_glb(self, path: str | Path) -> None:
        import trimesh
        scene = trimesh.Scene()
        for node in self.source["nodes"]:
            oid = node["object_id"]
            for index, piece in enumerate(self.mesh_pieces(oid)):
                scene.add_geometry(piece, node_name=f"{oid}:part{index}")
        scene.export(path)

    def render(self, camera: dict[str, Any], output: str | Path) -> None:
        """Render the exact same geometry and transforms used by mesh queries."""
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from PIL import Image
        import pyrender
        size = camera["image_size"]
        width, height = int(size[0]), int(size[1])
        scene = pyrender.Scene(bg_color=[235, 239, 244, 255], ambient_light=[0.45, 0.45, 0.45])
        for node in self.source["nodes"]:
            for piece in self.mesh_pieces(node["object_id"]):
                rendered = pyrender.Mesh.from_trimesh(piece, smooth=bool(camera.get('smooth_shading',False)))
                if node['asset'].get('provenance',{}).get('surface_type') == 'thin_sheet':
                    for primitive in rendered.primitives: primitive.material.doubleSided = True
                scene.add(rendered)
        k = np.asarray(camera["K"], dtype=float)
        cam = pyrender.IntrinsicsCamera(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2], znear=.001)
        # Input T_W_C uses OpenCV x-right/y-down/z-forward. pyrender uses OpenGL.
        cv_to_gl = np.diag([1, -1, -1, 1])
        scene.add(cam, pose=pose_matrix(camera["T_W_C"]) @ cv_to_gl)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2), pose=pose_matrix(camera["T_W_C"]) @ cv_to_gl)
        renderer = pyrender.OffscreenRenderer(width, height)
        try:
            color, _ = renderer.render(scene)
        finally:
            renderer.delete()
        Image.fromarray(color).save(output)


def generated_scene_code() -> str:
    """Small executable view of the sole scene.json authority."""
    return (
        "# Generated from scene.json; edit with a verified patch.\n"
        "from pathlib import Path\nfrom agentic_mesh_world.mesh_world import MeshCodeWorld\n"
        "WORLD = MeshCodeWorld.load(Path(__file__).with_name('scene.json'))\n"
        "def object_pose(object_id): return WORLD.object_transform(object_id)\n"
        "def object_mesh(object_id): return WORLD.mesh(object_id)\n"
        "def object_bounds(object_id): return WORLD.bounds(object_id)\n"
        "def surface_distance(object_id, point_W_m): return WORLD.surface_distance(object_id, point_W_m)\n"
        "def joint_state(object_id): return WORLD.joint_state(object_id)\n"
        "def region(object_id, region_id): return WORLD.region(object_id, region_id)\n"
        "def render(camera, output): return WORLD.render(camera, output)\n"
    )
