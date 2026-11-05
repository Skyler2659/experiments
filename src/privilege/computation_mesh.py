"""Geometry for calculations; rendering continues to use the bound GLB assets."""

import numpy as np
import trimesh


def computation_mesh(mesh):
    """Drop render materials and weld exactly coincident vertices, without moving faces.

    GLB exporters often duplicate a vertex for each face/UV seam. Splitting that
    render mesh can produce thousands of disconnected triangles and deep-copy
    its textures thousands of times. Calculations need the triangle surface,
    not those render attributes. No tolerance, simplification or hole repair is
    applied here; even nearby but distinct vertices remain distinct.
    """
    vertices = np.asarray(mesh.vertices)
    _, first, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return trimesh.Trimesh(
        vertices=vertices[first[order]].copy(),
        faces=remap[inverse][np.asarray(mesh.faces)],
        process=False,
    )
