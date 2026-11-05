"""Regression coverage for camera timing, held geometry, and complete model input."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("privilege")
import trimesh
from privilege.pyrualean import AgentWorld, capture
from privilege.storage import from_meshes
from privilege.observations import load
from privilege.comparison import render
from pyrualean.arms import ArmResult
from pyrualean.hosts import robodojo_native, robodojo_sight
from pyrualean.prompt import TaskCard
from pyrualean.task_descriptions import public_description, _tasks


class Robot:
    CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    def __init__(self):
        self.step = 0
        self.pos = [0., 0., .8]
        self.quat = [1., 0., 0., 0.]
    def state(self):
        return {"step": self.step, "left": {"eef_pos": self.pos, "eef_quat": self.quat},
                "right": {"eef_pos": [0., 0., .8], "eef_quat": [1., 0., 0., 0.]}}
    def image(self, camera, step):
        assert step == self.step
        return np.full((80, 100, 3), self.CAMERAS.index(camera) * 50, np.uint8)
    def camera_meta(self, camera, step):
        t = np.diag([1., -1., -1., 1.]); t[0, 3] = .05 * self.CAMERAS.index(camera)
        return {"intrinsic_matrix": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                "extrinsic_matrix": t, "shape": [80, 100, 3],
                "timing": {"host_step": step, "pose_state": "image_state"}}


@pytest.fixture
def scene(tmp_path):
    mesh = tmp_path / "cube.glb"
    trimesh.creation.box([.1, .1, .1]).export(mesh)
    t = np.eye(4); t[:3, 3] = [.1, 0, 1]
    world = from_meshes({"block": {"mesh": mesh, "T_W_object": t, "movable": True},
                         "wall": {"mesh": mesh, "movable": False}}, tmp_path / "world")
    robot = Robot()
    return world, robot, AgentWorld(world, robot, tmp_path / "scratch")


def action(api, robot, **kwargs):
    api.read_context(); api._before_action("move_pose", kwargs)
    robot.step += 1
    api._after_action(SimpleNamespace(to_dict=lambda: {"tool": "move_pose", "ok": True}))


def commit(api, **kwargs):
    ctx = api.read_context()
    return api.commit_step(ctx["interaction"]["pending_update"]["capture"],
                           expected_revision=ctx["revision"], note="Review changed objects; rest stable", **kwargs)


def test_three_views_preserve_camera_geometry_and_validate_each_image(scene):
    world, robot, api = scene
    packet = api.observe()
    assert [v["view_id"] for v in packet["views"]] == list(robot.CAMERAS)
    assert np.allclose(packet["views"][2]["T_W_C"], [[1,0,0,.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    path, _, _ = load(world.root, packet["capture"])
    (path.parent / packet["views"][2]["rgb_uri"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="image changed"):
        load(world.root, packet["capture"])


def test_grasp_translation_rotation_release_and_background_retention(scene):
    world, robot, api = scene
    action(api, robot, arm="left", gripper=1)
    commit(api, held={"left": "block"})
    background = world.snapshot().transform("wall").copy()
    # A 90 degree arm rotation must rotate the grasp offset as well as the object.
    api.read_context(); api._before_action("move_pose", {"arm": "left", "gripper": 0})
    robot.pos = [.2, .1, .9]; robot.quat = [2**-.5, 0., 0., 2**-.5]; robot.step += 1
    api._after_action(SimpleNamespace(to_dict=lambda: {"ok": True}))
    assert np.allclose(world.snapshot().position("block"), [.2, .2, 1.1])
    assert np.allclose(world.snapshot().transform("block")[:3, :3], [[0,-1,0],[1,0,0],[0,0,1]])
    commit(api)  # finger occlusion needs no repeated unresolved explanation
    assert np.allclose(world.snapshot().transform("wall"), background)
    assert world.state()["objects"]["wall"]["review"]["pose"]["status"] == "observed"
    action(api, robot, arm="left", gripper=-1)
    commit(api)
    released = world.snapshot().transform("block").copy()
    robot.pos = [.8, .8, .8]
    action(api, robot, arm="left", gripper=0); commit(api)
    assert np.allclose(world.snapshot().transform("block"), released)
    assert world.state()["attachments"] == {}


def test_matched_renderer_projects_to_expected_pixels(scene):
    world, robot, api = scene
    view = api.observe()["views"][0]
    _, labels = render(world.snapshot(), view, ["block"])
    yy, xx = np.where(labels == 0)
    assert len(xx) > 80
    assert 58 < np.mean(xx) < 61 and 38 < np.mean(yy) < 41
    action(api, robot, arm="left", gripper=0)
    ctx, panels = api._tool_feedback()
    assert len(panels["images"]) == 3
    assert not ctx.get("comparison_error")
    assert len(ctx["post_action_observation"]["views"]) == 3
    assert api._tool_feedback()[1] is None  # read-only calls don't resend identical panels


def test_image_time_pose_not_latest_robot_pose(tmp_path):
    shown = {"left_ee_pose": [0, 0, 1, 1, 0, 0, 0]}
    latest = {"left_ee_pose": [.4, .2, 1, 1, 0, 0, 0]}
    steps = robodojo_native.Steps(tmp_path)
    steps.record({"state": latest, "image_state": shown, "vision": {
        "cam_left_wrist": {"color": np.zeros((480, 640, 3), np.uint8)}}},
        {"step": 5, "pictures_behind": 1})
    meta = steps.records[0]["cameras"]["cam_left_wrist"]
    assert np.allclose(meta["extrinsic_matrix"], robodojo_sight.wrist_view(shown["left_ee_pose"]).pose)
    assert not np.allclose(meta["extrinsic_matrix"], robodojo_sight.wrist_view(latest["left_ee_pose"]).pose)
    disk = json.loads((tmp_path / "steps/0000/observation.json").read_text())
    assert disk["image_state"] == shown and disk["state"] == latest


def test_public_goals_shared_and_complete():
    assert len(_tasks()) == 42
    for key, item in _tasks().items():
        assert item["description"] and item["scoring"] and item["source"].startswith("https://robodojo-benchmark.com/")
    for robot in ("robodojo", "robodojo_xpl_rgb"):
        card = TaskCard(suite="", task="build_tower", seed=0, task_language="Build a tower", robot=robot, object_names=[])
        text = card.render()
        assert "100 points" in text and "all eight pieces" in text and "top pieces" in text
    assert public_description("robodojo_xpl_rgb", "build_tower_random") == _tasks()["build_tower"]
    assert public_description("libero", "build_tower") is None


def test_current_geometry_survives_verbose_stdout(scene):
    world, robot, api = scene
    context = api.read_context()
    blocks = ArmResult({"stdout": "x" * 25000, "codeworld": context}).content_blocks
    assert sum(len(b["text"]) for b in blocks) < 12000
    decoded = json.loads(blocks[1]["text"])["codeworld"]
    assert {o["object_id"] for o in decoded["objects"]} == {"block", "wall"}
    assert decoded["objects"][0]["T_W_object"] == np.round(world.snapshot().transform("block"), 6).tolist()


def test_geometry_conflict_does_not_privilege_either_estimate_or_move(scene):
    world, robot, api = scene
    report = api.reconcile_location("block", {"center_xyz": [.1, 0, 1.2], "top_z": 1.3, "height_m": .535})
    assert report["conflict"] and report["preferred"] == "no_default"
    assert robot.step == 0 and world.state()["revision"] == 0
    # PyRUA center Z is halfway from TABLE_Z to the top, not physical center Z.
    assert not api.reconcile_location("block", {"center_xyz": [.1, 0, .88]})["conflict"]
    with pytest.raises(ValueError, match="predates"):
        api.reconcile_location("block", {"top_z": 1.05}, estimate_step=-1)
    with pytest.raises(ValueError, match="same frame"):
        api.reconcile_location("block", {"top_z": 1.05}, frame="different_frame")


def test_delayed_image_pose_correction_advances_held_object(scene):
    from privilege.world import atomic
    world, robot, api = scene
    action(api, robot, arm="left", gripper=1)
    cap = api.read_context()["interaction"]["pending_update"]["capture"]
    path, packet, _ = load(world.root, cap)
    packet["metadata"]["image_robot_state"] = {
        "left_ee_pose": [-.2, 0, .8, 1, 0, 0, 0]}
    atomic(path, packet)
    measured = np.eye(4); measured[:3, 3] = [-.1, 0, 1]
    commit(api, held={"left": "block"}, poses={"block": {
        "T_W_object": measured.tolist(), "time_basis": "image",
        "translation_observed_axes": [True]*3, "orientation_observed_axes": [True]*3}})
    assert np.allclose(world.snapshot().position("block"), [.1, 0, 1])
    assert np.allclose(np.asarray(world.state()["attachments"]["left"]["T_E_object"])[:3, 3], [.1, 0, .2])
    original_meta = robot.camera_meta
    robot.camera_meta = lambda camera, step: {
        **original_meta(camera, step), "image_robot_state": packet["metadata"]["image_robot_state"]}
    report = api.reconcile_location("block", {"camera": "cam_left_wrist", "center_xyz": [-.1, 0, .88]})
    assert not report["conflict"]
    assert report["geometry_time_alignment"] == "held_prediction_at_image_time"
    assert np.allclose(world.snapshot().position("block"), [.1, 0, 1])  # comparison is read-only
