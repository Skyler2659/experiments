"""Indexed distance queries over reusable object-local reconstruction geometry.

Pose proposals transform measured points into the same local surface. They do not
copy a complete mesh or rebuild its spatial index for each candidate transform.
"""
import copy
from collections import OrderedDict

import numpy as np
import trimesh

from .completion import digest, path_in


_CACHE = OrderedDict()
_CACHE_SIZE = 16


def _signature(path):
    info = path.stat()
    return (str(path), info.st_mtime_ns, info.st_size)


class SurfaceIndex:
    def __init__(self, mesh):
        self.mesh = mesh
        # Trimesh caches this tree on the unchanged mesh. Build once, before the
        # sequence of poses, instead of paying its construction in every query.
        self.mesh.triangles_tree
        self.queries = 0

    def distances(self, points_W, T_W_object):
        points = np.asarray(points_W, dtype=float)
        transform = np.asarray(T_W_object, dtype=float)
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        self.queries += 1
        return trimesh.proximity.closest_point(self.mesh, local)[1]

    def equivalent(self, first, second, tolerance_m=2e-4):
        """Sampled whole-surface equivalence diagnostic, not a symmetry proof."""
        vertices = np.asarray(self.mesh.vertices)
        vertices = vertices[np.linspace(0, len(vertices)-1, min(len(vertices), 1024)).astype(int)]
        centers = np.asarray(self.mesh.triangles_center)
        centers = centers[np.linspace(0, len(centers)-1, min(len(centers), 1024)).astype(int)]
        local = np.concatenate((vertices, centers))
        a = np.asarray(first); b = np.asarray(second)
        points_a = local @ a[:3, :3].T + a[:3, 3]
        points_b = local @ b[:3, :3].T + b[:3, 3]
        error = max(float(self.distances(points_a, b).max()),
                    float(self.distances(points_b, a).max()))
        return bool(error <= tolerance_m), error


def geometry_key(world, object_id, state):
    """Identify only the local geometry inputs, independently of object pose."""
    from .world import read
    row = state['objects'][object_id]
    if row.get('assembly'):
        geometry_key = digest(row['assembly'])
        files = [path_in(world.root, p['local_mesh']) for p in row['assembly']['parts'].values()]
    else:
        metadata_path = path_in(world.root, world.manifest['objects'][object_id])
        metadata = read(metadata_path)
        geometry_key = world.manifest.get('geometry_revision')
        files = [metadata_path, path_in(world.root, metadata['geometry']['local_mesh'])]
    return (str(world.root), object_id, geometry_key, tuple(_signature(p) for p in files))


def object_surface(world, object_id, state):
    """Reuse immutable local geometry; a geometry change creates a fresh index."""
    key = geometry_key(world, object_id, state)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    local_state = copy.deepcopy(state)
    local_state['objects'][object_id]['T_W_object'] = np.eye(4).tolist()
    result = SurfaceIndex(world._mesh(object_id, local_state))
    result.geometry_key = key
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result


def pose_observability(surface, T, points, camera_origins, max_p90_m):
    """Optional diagnostics. Ambiguity is recorded, never a generic read gate."""
    from scipy.spatial.transform import Rotation
    T = np.asarray(T)
    points = np.asarray(points)
    points = points[np.linspace(0, len(points)-1, min(len(points), 128)).astype(int)]
    baseline = float(np.quantile(surface.distances(points, T), .9))
    threshold = max(max_p90_m, baseline + .002)
    translations = []
    for axis in range(3):
        for offset in (-.02, .02):
            candidate = T.copy(); candidate[axis, 3] += offset
            error = float(np.quantile(surface.distances(points, candidate), .9))
            translations.append({'axis':axis, 'offset_m':offset, 'p90_m':error})
    translation_axes = [all(np.isfinite(v['p90_m']) and v['p90_m'] > threshold
                            for v in translations if v['axis'] == axis) for axis in range(3)]
    rotations = []
    orientation_axes = [None, None, None]
    equivalent_axes = [None, None, None]
    multi_view = (len(camera_origins) >= 2 and
                  max(np.linalg.norm(a-b) for a in camera_origins for b in camera_origins) >= .02)
    if multi_view:
        center = surface.mesh.centroid @ T[:3, :3].T + T[:3, 3]
        for axis in range(3):
            for angle in (-30., 30., 90., 180.):
                rotation = Rotation.from_rotvec(np.eye(3)[axis] * np.deg2rad(angle)).as_matrix()
                candidate = T.copy()
                candidate[:3, :3] = rotation @ T[:3, :3]
                candidate[:3, 3] = rotation @ (T[:3, 3] - center) + center
                error = float(np.quantile(surface.distances(points, candidate), .9))
                entry = {'axis':axis, 'angle_deg':angle, 'p90_m':error}
                if error <= threshold:
                    equivalent, geometry_error = surface.equivalent(T, candidate)
                    entry.update(geometry_equivalent=equivalent, geometry_sample_error_m=geometry_error)
                rotations.append(entry)
        orientation_axes = [all(np.isfinite(v['p90_m']) and v['p90_m'] > threshold
                                for v in rotations if v['axis'] == axis) for axis in range(3)]
        equivalent_axes = [all(v.get('geometry_equivalent', False)
                               for v in rotations if v['axis'] == axis and v['p90_m'] <= threshold)
                           for axis in range(3)]
    return {'status':'EVALUATED', 'baseline_p90_m':baseline, 'alternative_threshold_m':threshold,
            'translation_observed':all(translation_axes),
            'orientation_observed':all(orientation_axes) if multi_view else None,
            'translation_observed_axes':translation_axes, 'orientation_observed_axes':orientation_axes,
            'orientation_geometry_equivalent_axes':equivalent_axes,
            'translation_alternatives':translations, 'orientation_alternatives':rotations,
            'translation_probe_offset_m':.02, 'alternative_acceptance_margin_m':.002,
            'geometry_equivalence_tolerance_m':2e-4,
            'limitations':'Discrete alternatives and sampled surface equivalence are diagnostics, not proof of a unique pose. Symmetric geometry can remain useful while orientation is ambiguous.'}
