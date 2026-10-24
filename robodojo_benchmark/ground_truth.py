"""Simulator-side oracle geometry/state adapter; never used by the other arms.

Exports local triangle meshes with holes and disconnected components intact.
The only privileged fields are geometry, object identity and current rigid pose.
Reward functions, goal poses, future trajectories and task-success flags are absent.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from codeworld_robot.io import atomic, read


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def pose_of(obj, origin):
    pos, quat = obj.get_world_pose()
    pos, quat = array(pos).reshape(3), array(quat).reshape(4)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
    T[:3, 3] = pos - origin
    if not np.isfinite(T).all():
        raise ValueError("Non-finite native object pose")
    return T


def triangulate(points, counts, indices, holes=()):
    """Ear clipping handles concave USD faces without filling holes or convexifying."""
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(counts < 0) or counts.sum() != len(indices):
        raise ValueError('Invalid USD topology')
    # Native assets are predominantly already triangulated. Avoid running a
    # polygon-normal/ear-clipping loop once per triangle in dense clutter scenes.
    if np.all(counts == 3):
        faces = indices.reshape(-1, 3)
        if holes:
            faces = faces[~np.isin(np.arange(len(faces)), list(holes))]
        return faces
    triangles, offset = [], 0
    for face_index, count in enumerate(counts):
        face = list(map(int, indices[offset:offset + count])); offset += count
        if face_index in holes or count < 3:
            continue
        if count == 3:
            triangles.append(face)
            continue
        p = points[face]
        normal = np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0)
        axes = [i for i in range(3) if i != np.argmax(np.abs(normal))]
        xy = p[:, axes]
        signed_area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
        if abs(signed_area) < 1e-16:
            continue
        winding = 1 if signed_area > 0 else -1
        pending = list(range(count))
        def cross(a, b, c):
            u, v = b - a, c - a
            return winding * (u[0] * v[1] - u[1] * v[0])
        while len(pending) > 3:
            for j, b in enumerate(pending):
                a, c = pending[j - 1], pending[(j + 1) % len(pending)]
                if cross(xy[a], xy[b], xy[c]) <= 1e-14:
                    continue
                inside = any(all(v >= -1e-14 for v in (cross(xy[a], xy[b], xy[k]),
                             cross(xy[b], xy[c], xy[k]), cross(xy[c], xy[a], xy[k])))
                             for k in pending if k not in (a, b, c))
                if not inside:
                    triangles.append([face[a], face[b], face[c]])
                    pending.pop(j)
                    break
            else:
                raise ValueError("USD polygon cannot be triangulated without changing its surface")
        triangles.append([face[k] for k in pending])
    if offset != len(indices):
        raise ValueError("Invalid USD topology")
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def prim_mesh(prim):
    from pxr import UsdGeom
    kind = prim.GetTypeName()
    if kind == "Mesh":
        m = UsdGeom.Mesh(prim)
        p = np.asarray(m.GetPointsAttr().Get(), dtype=float)
        faces = triangulate(p, list(m.GetFaceVertexCountsAttr().Get()),
                            list(m.GetFaceVertexIndicesAttr().Get()), set(m.GetHoleIndicesAttr().Get() or []))
        if m.GetOrientationAttr().Get() == "leftHanded":
            faces = faces[:, ::-1]
        return trimesh.Trimesh(p, faces, process=False)
    if kind == "Cube":
        return trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())] * 3)
    if kind == "Sphere":
        return trimesh.creation.icosphere(subdivisions=4, radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
    if kind in ("Cylinder", "Cone", "Capsule"):
        schema = getattr(UsdGeom, kind)(prim)
        radius, height = float(schema.GetRadiusAttr().Get()), float(schema.GetHeightAttr().Get())
        if kind == "Cylinder":
            mesh = trimesh.creation.cylinder(radius, height, sections=96)
        elif kind == "Cone":
            mesh = trimesh.creation.cone(radius, height, sections=96)
            mesh.apply_translation([0, 0, -height / 2])
        else:
            mesh = trimesh.creation.capsule(height=height, radius=radius, count=[32, 32])
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
        axis = str(schema.GetAxisAttr().Get())
        if axis != "Z":
            mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0] if axis == "X" else [0, 1, 0]))
        return mesh
    return None


def local_mesh(stage, prim_path, *, exclude_roots=(), allow_empty=False):
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise ValueError(f"Missing object prim {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_T = np.asarray(cache.GetLocalToWorldTransform(root), float).T
    scales = np.linalg.norm(root_T[:3, :3], axis=0)
    normalized = root_T[:3, :3] / scales
    if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-5):
        raise ValueError("Sheared object-root transforms need an explicit geometry adapter")
    unscale = np.eye(4); unscale[:3, :3] = np.diag(scales)
    unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(unit, 1.):
        raise ValueError(f"Expected RoboDojo meter stage, got metersPerUnit={unit}")
    pieces = []
    inverse = np.linalg.inv(root_T)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if any(path == p or path.startswith(p + '/') for p in exclude_roots):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and (imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
                          or imageable.GetPurposeAttr().Get() in ("guide", "proxy")):
            continue
        mesh = prim_mesh(prim)
        if mesh is None:
            continue
        transform = unscale @ inverse @ np.asarray(cache.GetLocalToWorldTransform(prim), float).T
        mesh.apply_transform(transform)
        pieces.append(mesh)
    if not pieces:
        if allow_empty:
            return None
        raise ValueError(f"No supported visible surface for {prim_path}; no bounding-box fallback")
    mesh = trimesh.util.concatenate(pieces)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid native surface {prim_path}")
    return mesh


class ArticulationSurface:
    """One rigid link's surface, with the native USD link frame at each observation.

    RoboDojo's ArticulationObject.get_link_pose uses this same USD transform.
    Full prim paths avoid ambiguity when separate branches reuse link names.
    Joint targets, limits, physics properties and evaluator state are not read.
    """
    def __init__(self, stage, prim_path, category, parent, role, mesh):
        self.stage, self.prim_path = stage, prim_path
        self.category_name, self.parent_articulation = category, parent
        self.role, self.mesh = role, mesh

    def get_world_pose(self):
        from pxr import Usd, UsdGeom
        prim = self.stage.GetPrimAtPath(self.prim_path)
        T = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float).T
        rotation = T[:3, :3] / np.linalg.norm(T[:3, :3], axis=0)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError('Articulation link frame must have an orthogonal, proper rotation')
        q = Rotation.from_matrix(rotation).as_quat()
        return T[:3, 3], np.r_[q[3], q[:3]]


def articulation_surfaces(stage, key, obj):
    """Partition visible surfaces by rigid body; moving links never share a mesh."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(obj.prim_path)
    links = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
             if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not links:
        raise ValueError(f'Articulation {key} has no native rigid link frames')
    paths = links + ([obj.prim_path] if obj.prim_path not in links else [])
    for path in paths:
        children = [p for p in links if p != path and p.startswith(path + '/')]
        mesh = local_mesh(stage, path, exclude_roots=children, allow_empty=True)
        if mesh is None:
            continue  # Empty inertial/link frames have no surface to represent.
        name = path[len(obj.prim_path):].strip('/') or 'root'
        yield f'{key}/{name}', ArticulationSurface(stage, path, getattr(obj, 'category_name', 'native_object'),
                  str(key), 'articulation_link' if path in links else 'articulation_fixture', mesh)


class GroundTruthWorld:
    def __init__(self, env, root, origin, mode="live_state", reuse_episode=None):
        self.env, self.root = env, Path(root)
        self.origin, self.mode = np.asarray(origin, float), mode
        sm = env.scene_manager
        for name in ("_garment_objects", "_fluid_objects"):
            if getattr(sm, name, [{}])[0]:
                raise ValueError(f"GT rigid adapter cannot represent {name}; register a task adapter")
        objects, roles = {}, {}
        for name in ("_rigid_and_dynamic_objects", "_geometry_objects"):
            for key, obj in getattr(sm, name)[0].items():
                if key in objects:
                    raise ValueError(f"Duplicate native identity {key}")
                objects[key] = obj
                roles[key] = 'fixture' if name == '_geometry_objects' else 'rigid_object'
        for key, obj in getattr(sm, '_articulation_objects', [{}])[0].items():
            surfaces = dict(articulation_surfaces(sm.stage, key, obj))
            if not surfaces:
                raise ValueError(f'Articulation {key} has no supported visible surface')
            if objects.keys() & surfaces.keys():
                raise ValueError(f'Duplicate native articulation identity {key}')
            objects.update(surfaces)
            roles.update({name: surface.role for name, surface in surfaces.items()})
        table = getattr(sm, "_tables", [None])[0]
        if table is not None:
            objects["tabletop"] = table
            roles['tabletop'] = 'fixture'
        if not objects:
            raise ValueError("Native object inventory is empty")
        cached = Path(reuse_episode) if reuse_episode else None
        if cached:
            from codeworld_workspace import World
            World(cached).verify_binding()
        self.objects = {}
        meta, states, bound = {}, {}, {}
        for folder in ("objects", "geometry", "source", "state", "scenarios", "observations", "work", "task"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for index, (key, obj) in enumerate(sorted(objects.items())):
            oid = re.sub(r"[^A-Za-z0-9_]", "_", str(key)).strip("_")
            if not oid or oid in self.objects:
                oid = f"native_{index:04d}"
            self.objects[oid] = obj
            rel = f"geometry/{oid}.glb"
            if cached:
                old = read(cached / f"objects/{oid}.json")
                if old["name"] != str(key) or old["role"] != roles[key]:
                    raise ValueError("Cached GT object identity mismatch: " + oid)
                shutil.copy2(cached / old["geometry"]["local_mesh"], self.root / rel)
            else:
                mesh = obj.mesh if isinstance(obj, ArticulationSurface) else local_mesh(sm.stage, obj.prim_path)
                mesh.export(self.root / rel)
            T = pose_of(obj, self.origin).tolist()
            path = f"objects/{oid}.json"
            meta[oid] = path
            atomic(self.root / path, {"object_id": oid, "name": str(key), "role": roles[key],
                   "geometry": {"local_mesh": rel}, "initial_T_W_object": T, "source_nodes": [],
                   **({'parent_articulation': obj.parent_articulation} if isinstance(obj, ArticulationSurface) else {}),
                   "category": getattr(obj, "category_name", "native_object"), "geometry_source": "native_usd_visual_surface"})
            states[oid] = {"T_W_object": T, "basis": "reconstruction_prior", "provenance": "gt_initial_geometry",
                           "review": {"pose": {"status": "prior", "revision": 0}}}
            for file in (rel, path):
                bound[file] = hashlib.sha256((self.root / file).read_bytes()).hexdigest()
        if cached and set(read(cached / "world.json")["objects"]) != set(meta):
            raise ValueError("Cached GT object inventory mismatch")
        atomic(self.root / "world.json", {"schema": "agent-codeworld-v1", "frame": "robot-base W; Z up", "units": "m",
               "matrix_convention": "row-major T_W_object; column vectors", "objects": meta, "bound_files": bound,
               "geometry_revision": "native_usd_v1", "source": "ground_truth", "source_program_binding_verified": False,
               "oracle_mode": mode, "surface": "USD visual triangle meshes; analytic primitives tessellated"})
        atomic(self.root / "state/current.json", {"revision": 0, "geometry_revision": "native_usd_v1", "objects": states,
               "history": [], "oracle": {"mode": mode}})
        atomic(self.root / "source/gt_provenance.json", {"mode": mode, "object_count": len(meta),
               "exporter": "robodojo_benchmark.ground_truth", "geometry": "native USD visual surface",
               "state": "native rigid and articulated-link transforms" if mode == "live_state" else "initial geometry and pose only"})

    def sync(self, robot_state):
        if self.mode != "live_state":
            return
        import fcntl
        from codeworld_workspace.completion import digest, state_ref
        with (self.root / "state/update.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            s = read(self.root / "state/current.json")
            version = state_ref(robot_state)
            if s.get("oracle", {}).get("robot_state_ref") == version:
                return
            s["revision"] += 1
            for oid, obj in self.objects.items():
                row = s["objects"][oid]
                row["T_W_object"] = pose_of(obj, self.origin).tolist()
                row["basis"] = "oracle_state"
                row["oracle_validation"] = {"pose_digest": digest(row["T_W_object"]), "robot_state_ref": version}
                row.setdefault("review", {})["pose"] = {"status": "oracle", "revision": s["revision"]}
                row.pop("sensor_validation", None)
                for field in row.get("observed_state", {}).get("fields", {}).values():
                    field["status"] = "pending"
            s["oracle"] = {"mode": "live_state", "robot_state_ref": version}
            s.setdefault("runtime", {})["robot_state"] = robot_state
            s.setdefault("history", []).append({"revision": s["revision"], "kind": "oracle_state", "robot_state_ref": version,
                                                  "poses": {o: r["T_W_object"] for o, r in s["objects"].items()}})
            atomic(self.root / "state/current.json", s)
