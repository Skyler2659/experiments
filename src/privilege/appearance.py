"""Publish original surface subsets without carrying textures through mesh calculations."""
import numpy as np
import trimesh


def source_surface(world, object_id, faces, transform):
    """Face indices refer to the bound object's original object-frame render mesh."""
    path = world.root / world.object(object_id)['geometry']['local_mesh']
    mesh = trimesh.load(path, force='mesh', process=False)
    faces = np.asarray(faces)
    if faces.ndim != 1 or faces.dtype.kind not in 'iu' or not len(faces):
        raise ValueError('source_faces must be a nonempty integer list')
    if faces.min() < 0 or faces.max() >= len(mesh.faces) or len(np.unique(faces)) != len(faces):
        raise ValueError('source_faces contains invalid or duplicate face indices')
    part = mesh.submesh([faces], append=True, repair=False)
    part.apply_transform(np.linalg.inv(transform))
    return part


def restore_surface(world, object_id, mesh, transform):
    """Restore only exact source triangles; never invent texture for new geometry.

    Match in object frame, including UV seams from the original render asset.
    A changed/deformed surface must supply its own appearance.
    """
    original = trimesh.load(world.root / world.object(object_id)['geometry']['local_mesh'],
                            force='mesh', process=False, skip_materials=True)
    from scipy.spatial import cKDTree
    from itertools import permutations
    tolerance = 1e-7  # 0.1 micrometre: GLB float32 roundoff, not shape fitting.
    tree = cKDTree(original.triangles_center)
    orders = np.asarray(list(permutations(range(3))))
    # Most split parts retain the object's original local frame, then articulate
    # via T_object_part. Also support a rigidly recentered part-local mesh.
    for source_frame in (np.eye(4), transform):
        moved = mesh.copy(); moved.apply_transform(source_frame)
        candidates = tree.query_ball_point(moved.triangles_center, tolerance)
        indices = []
        for triangle, nearby in zip(moved.triangles, candidates):
            hits = []
            for index in nearby:
                residual = original.triangles[index][orders] - triangle[None, :, :]
                if np.any(np.max(np.linalg.norm(residual, axis=2), axis=1) <= tolerance):
                    hits.append(index)
            if len(hits) != 1:
                break
            indices.append(hits[0])
        if len(indices) == len(mesh.faces) and len(set(indices)) == len(indices):
            return source_surface(world, object_id, indices, source_frame)
    return None
