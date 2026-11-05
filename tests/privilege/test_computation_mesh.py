import hashlib

import numpy as np
from PIL import Image
import trimesh

from privilege.computation_mesh import computation_mesh
from privilege.storage import from_meshes


def textured_triangle_soup():
    box = trimesh.creation.box()
    vertices = box.triangles.reshape(-1, 3).copy()
    faces = np.arange(len(vertices)).reshape(-1, 3)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGBA", (128, 64), "red"),
        metallicRoughnessTexture=Image.new("RGB", (128, 64), "blue"),
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False,
                          visual=trimesh.visual.TextureVisuals(
                              uv=np.zeros((len(vertices), 2)), material=material))


def test_split_welds_exact_vertices_without_textures_or_surface_changes():
    source = textured_triangle_soup()
    result = computation_mesh(source)
    assert np.array_equal(result.triangles, source.triangles)
    assert len(result.vertices) == 8
    assert len(result.split(only_watertight=False)) == 1
    assert result.visual.kind != "texture"
    assert source.visual.material.baseColorTexture.size == (128, 64)
    # Numerical proximity alone must never move or merge a vertex.
    source.vertices[0, 0] += 1e-10
    near = computation_mesh(source)
    assert len(near.vertices) == 9
    assert np.array_equal(near.triangles, source.triangles)


def test_world_and_snapshot_queries_leave_bound_render_asset_unchanged(tmp_path):
    source = tmp_path / "textured.glb"
    textured_triangle_soup().export(source)
    world = from_meshes({"object": {"mesh": source}}, tmp_path / "world", frame="test")
    local = world.root / world.object("object")["geometry"]["local_mesh"]
    before = hashlib.sha256(local.read_bytes()).hexdigest()
    for query in [world.mesh("object"), world.snapshot().mesh("object")]:
        assert query.visual.kind != "texture"
        assert len(query.split(only_watertight=False)) == 1
    assert hashlib.sha256(local.read_bytes()).hexdigest() == before
    render = trimesh.load(local, force="mesh", process=False)
    assert render.visual.material.baseColorTexture is not None
    assert world.verify_binding()["status"] == "BOUND"
