import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from privilege.alignment import align_table_height
from privilege.pyrualean import prepare
from privilege.storage import from_meshes


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box([.1, .1, .05]).export(path)
    def pose(x, y, z):
        t = np.eye(4); t[:3, 3] = [x, y, z]
        return t
    return from_meshes({
        "tabletop": {"mesh": path, "T_W_object": pose(0, 0, .725), "movable": False},
        "block": {"mesh": path, "T_W_object": pose(.1, .2, .775)},
        "floor": {"mesh": path, "T_W_object": pose(0, 0, -.025), "movable": False},
        "known_robot": {"mesh": path, "T_W_object": pose(.3, -.45, .765), "role": "robot_snapshot"},
    }, tmp_path / "source", frame="pyrualean_world")


def test_global_translation_preserves_contacts_robot_geometry_and_repeat(scene):
    before = scene.snapshot()
    receipt = align_table_height(scene, table_object_id="tabletop", target_z_m=.765, source="test host prior")
    after = scene.snapshot()
    assert receipt["delta_z_m"] == pytest.approx(.015, abs=1e-8)
    assert after.bounds("tabletop")[1, 2] == pytest.approx(.765)
    for oid in ("tabletop", "block", "floor"):
        assert np.allclose(after.mesh(oid).vertices - before.mesh(oid).vertices, [0, 0, receipt["delta_z_m"]])
        assert np.array_equal(after.transform(oid)[:3, :3], before.transform(oid)[:3, :3])
    assert np.allclose(after.transform("block")[:3, 3] - after.transform("tabletop")[:3, 3],
                       before.transform("block")[:3, 3] - before.transform("tabletop")[:3, 3])
    assert np.array_equal(after.transform("known_robot"), before.transform("known_robot"))
    scene.verify_binding()  # original source/geometry files were not rewritten
    assert align_table_height(scene, table_object_id="tabletop", target_z_m=.765, source="test host prior") == receipt
    assert scene.state()["revision"] == after.revision  # no double shift


@pytest.mark.parametrize("fresh", [False, True])
def test_prepare_aligns_before_agent_and_import_never_reconstructs(scene, tmp_path, monkeypatch, fresh):
    import privilege.pyrualean as adapter
    robot = SimpleNamespace(
        state=lambda: {"step": 0, "joints": [0.]},
        image=lambda camera, step: np.zeros((12, 16, 3), np.uint8),
        camera_meta=lambda camera, step: {"intrinsic_matrix": [[10, 0, 8], [0, 10, 6], [0, 0, 1]],
                                         "extrinsic_matrix": np.eye(4).tolist()},
    )
    source_top = scene.snapshot().bounds("tabletop")[1, 2]
    calls = []
    def reconstruct(*args, **kwargs):
        calls.append(True)
        assert fresh, "Import rerun must not reconstruct"
        return scene
    monkeypatch.setattr(adapter, "reconstruct", reconstruct)
    if fresh:
        state_file = tmp_path / "robot_state.json"
        state_file.write_text(json.dumps(robot.state()))
        config = {"robot_glb": str(scene.root / "geometry/known_robot.glb"),
                  "robot_state_file": str(state_file), "reconstruction": {}}
    else:
        config = {"world": str(scene.root)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out = tmp_path / "episode"; out.mkdir()
    bindings, prompt, record = prepare(config_path, robo=robot, robot="robodojo_xpl_rgb", out_dir=out, scratch=out)
    assert bindings["world"].snapshot().bounds("tabletop")[1, 2] == pytest.approx(.765)
    assert record["height_alignment"]["delta_z_m"] == pytest.approx(.765-source_top)
    assert 'height_alignment' in prompt and 'host_table_height_aligned_prior' in prompt
    assert len(calls) == int(fresh)
    if not fresh:
        assert scene.snapshot().bounds("tabletop")[1, 2] == source_top  # only run copy changes
