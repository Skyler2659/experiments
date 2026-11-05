import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import trimesh
from privilege.storage import import_world
from privilege.world import World


def proposal(pose):
    return {
        "object": {
            "T_W_object": pose,
            "translation_observed_axes": [True, True, False],
            "orientation_observed_axes": [False, False, False],
        }
    }


def test_geometry_snapshot_branch_and_queries(world):
    assert not any(
        name.startswith(("codeworld_robot", "robodojo_benchmark")) for name in sys.modules
    )
    snap = world.snapshot()
    assert np.allclose(snap.bounds("object"), [[0.2, 0.35, 0.35], [0.4, 0.45, 0.65]])
    assert np.allclose(snap.local_point("object", [0, 0, 0]), [0.3, 0.4, 0.5])
    branch = world.fork("alternative")
    pose = np.eye(4)
    pose[0, 3] = 1
    branch.set_hypothesis({"object": pose}, expected_revision=0)
    assert world.state()["revision"] == 0
    assert np.allclose(branch.mesh("object").centroid, [1, 0, 0])
    world.update({"object": pose})
    assert np.allclose(snap.position("object"), [0.3, 0.4, 0.5])
    assert world.aabb_relation("object", "other")["boxes_overlap"] is False
    result = world.nearest_surface("other", [[0.2, 0, 0]])
    assert result["results"][0]["distance_m"] == pytest.approx(0.1)
    assert world.verify_binding()["status"] == "BOUND"


def test_feedback_rgb_estimate_and_prediction_are_distinct(world, observation):
    before = world.mesh("object").vertices.copy()
    rev = world.record_feedback(
        ["object"], note="host reported an interaction", result={"reached": True}
    )
    assert np.array_equal(before, world.mesh("object").vertices)
    assert world.review_status()["objects"]["object"]["aspects"]["pose"]["status"] == "pending"
    result = world.integrate_rgb(
        observation["capture"],
        proposal(np.eye(4)),
        expected_revision=rev,
        note="image-based estimate; z ambiguous",
    )
    assert result["metric_validation"] is False
    state = world.state()
    assert state["objects"]["object"]["sensor_validation"]["translation_observed_axes"] == [
        True,
        True,
        False,
    ]
    assert "sensor_validation" not in state["objects"]["other"]
    world.update({"object": np.eye(4)}, basis="action_prediction")
    row = world.state()["objects"]["object"]
    assert "sensor_validation" not in row and row["review"]["pose"]["status"] == "hypothesized"


def test_revision_lock_rejects_competing_updates(world, observation):
    def update():
        try:
            world.integrate_rgb(
                observation["capture"], proposal(np.eye(4)), expected_revision=0, note="candidate"
            )
            return "ok"
        except ValueError:
            return "stale"

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: update(), range(2))) == ["ok", "stale"]
    assert world.state()["revision"] == 1


def test_invalid_feedback_is_atomic(world, observation):
    bad = proposal(np.eye(4))
    bad["other"] = {**bad["object"], "translation_observed_axes": [1, 1, 0]}
    with pytest.raises(ValueError, match="booleans"):
        world.integrate_rgb(observation["capture"], bad, expected_revision=0, note="invalid")
    assert world.state()["revision"] == 0
    path = world.root / observation["capture"]
    packet = json.loads(path.read_text())
    packet["frame"] = "another_frame"
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="frames differ"):
        world.integrate_rgb(
            observation["capture"], proposal(np.eye(4)), expected_revision=0, note="invalid"
        )


def test_parts_update_geometry_and_snapshot_without_reconstruction(world, tmp_path):
    mesh = tmp_path / "part.glb"
    trimesh.creation.box(extents=[0.05, 0.05, 0.05]).export(mesh)
    body = np.eye(4)
    moving = np.eye(4)
    moving[0, 3] = 0.2
    world.publish_parts(
        "object",
        {
            "body": {"mesh": mesh, "T_object_part": body},
            "drawer": {"mesh": mesh, "T_object_part": moving},
        },
    )
    snap = world.snapshot()
    old_body = world.part_mesh("object", "body").vertices.copy()
    moving[0, 3] = 0.4
    world.update_parts("object", {"drawer": moving}, parameters={"opening_m": 0.2})
    assert np.array_equal(old_body, world.part_mesh("object", "body").vertices)
    assert np.allclose(snap.position("object", "drawer"), [0.5, 0.4, 0.5])
    assert np.allclose(world.part_mesh("object", "drawer").centroid, [0.7, 0.4, 0.5])
    assert world.verify_binding()["status"] == "BOUND"


def test_import_changes_frame_without_copying_old_harness(world, tmp_path):
    (world.root / "task").mkdir()
    (world.root / "task/completion.json").write_text('{"required":true}')
    t = np.eye(4)
    t[2, 3] = 0.765
    migrated = import_world(world.root, tmp_path / "imported", frame="new", T_host_world=t)
    assert not (migrated.root / "task").exists()
    assert np.allclose(migrated.mesh("object").centroid, [0.3, 0.4, 1.265])
    assert np.allclose(world.mesh("object").centroid, [0.3, 0.4, 0.5])
    assert migrated.state()["objects"]["object"]["basis"] == "imported_prior"
    assert World(migrated.root).verify_binding()["status"] == "BOUND"
