import numpy as np
import pytest
import trimesh
from privilege.storage import from_meshes


@pytest.fixture
def world(tmp_path):
    mesh = tmp_path / "object.glb"
    trimesh.creation.box(extents=[0.2, 0.1, 0.3]).export(mesh)
    pose = np.eye(4)
    pose[:3, 3] = [0.3, 0.4, 0.5]
    return from_meshes(
        {"object": {"mesh": mesh, "T_W_object": pose}, "other": {"mesh": mesh}},
        tmp_path / "world",
        frame="pyrualean_world",
    )


@pytest.fixture
def observation(world):
    return world.record_observation(
        np.zeros((12, 16, 3), dtype=np.uint8),
        K=[[10, 0, 8], [0, 10, 6], [0, 0, 1]],
        T_W_C=np.eye(4),
    )
