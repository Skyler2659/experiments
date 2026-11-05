import json

import numpy as np
import pytest
from privilege.pyrualean import AgentWorld, capture, prepare


class ReadOnlyRobot:
    def __init__(self):
        self.step = 0
        self.calls = []

    def state(self):
        self.calls.append("state")
        return {"step": self.step, "joints": [0.0]}

    def image(self, camera, step):
        self.calls.append("image")
        assert step == self.step and camera == "cam_head"
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def camera_meta(self, camera, step):
        self.calls.append("camera_meta")
        return {
            "intrinsic_matrix": [[10, 0, 8], [0, 10, 6], [0, 0, 1]],
            "extrinsic_matrix": np.eye(4).tolist(),
        }


def test_camera_axes_and_host_step_are_bound(world, tmp_path):
    robo = ReadOnlyRobot()
    api = AgentWorld(world, robo, tmp_path)
    cap = api.observe()
    assert np.allclose(cap["views"][0]["T_W_C"], np.diag([1, -1, -1, 1]))
    assert cap["metadata"]["host_step"] == 0
    robo.step = 1
    with pytest.raises(ValueError, match="predates host motion"):
        api.integrate_rgb(cap["capture"], {}, expected_revision=0, note="old image")
    assert set(robo.calls) <= {"state", "image", "camera_meta"}


def test_prepare_imports_data_and_injects_world_only(world, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"world": "world", "frame": "pyrualean_world", "height_alignment": False}))
    out = tmp_path / "episode"
    out.mkdir()
    bindings, prompt, record = prepare(
        config,
        robo=ReadOnlyRobot(),
        robot="robodojo_xpl_rgb",
        out_dir=out,
        scratch=out / "workspace",
    )
    assert set(bindings) == {"world"}
    assert bindings["world"].objects() == world.objects()
    assert "original `robo`" in prompt
    assert record["mode"] == "imported_prior"
    assert record['perception'] == 'codeworld'
    assert record['disabled_perception_tools'] == ['back_project', 'locate', 'segment']
    assert bindings['world']._robo._disabled_perception_tools == {'back_project', 'locate', 'segment'}
    assert not (out / "privilege/world/task").exists()


@pytest.mark.parametrize('mismatch', [None, 'rgb', 'state', 'camera'])
def test_recovery_checks_observation_before_world_reuse(tmp_path, mismatch):
    from privilege.pyrualean import check_recovery_observation

    source, target = tmp_path / 'source', tmp_path / 'target'
    old = capture(ReadOnlyRobot(), source, 'pyrualean_world')

    class RestartedRobot(ReadOnlyRobot):
        def image(self, camera, step):
            rgb = super().image(camera, step)
            if mismatch == 'rgb':
                rgb[3:8, 4:10] = 255
            return rgb

        def camera_meta(self, camera, step):
            meta = super().camera_meta(camera, step)
            if mismatch == 'camera':
                meta['extrinsic_matrix'][0][3] += 0.1
            return meta

    robo = RestartedRobot()
    if mismatch == 'state':
        robo.step = 1
    new = capture(robo, target, 'pyrualean_world')
    if mismatch:
        with pytest.raises(ValueError):
            check_recovery_observation(source / old['capture'], target, new['capture'])
    else:
        report = check_recovery_observation(source / old['capture'], target, new['capture'])
        assert report['matched'] and report['rgb_mae_255'] == 0


def test_prepare_fresh_reconstruction_receives_calibrated_public_inputs(
    tmp_path, world, monkeypatch
):
    import privilege.pyrualean as adapter

    robot = ReadOnlyRobot()
    state_file = tmp_path / "robot_state.json"
    state_file.write_text(json.dumps(robot.state()))
    glb = world.root / "geometry/object.glb"
    config = tmp_path / "fresh.json"
    config.write_text(
        json.dumps(
            {
                "robot_glb": str(glb),
                "robot_state_file": str(state_file),
                "reconstruction": {"runtime": "runtime"},
                "height_alignment": False,
            }
        )
    )

    def reconstruct(root, capture_path, robot_glb, output, cfg, **kwargs):
        packet = json.loads((root / capture_path).read_text())
        assert packet["frame"] == "pyrualean_world"
        assert len(packet["views"]) == 1 and "K" in packet["views"][0]
        assert packet["robot_state"] == robot.state()
        assert robot_glb == glb and cfg["runtime"] == str(tmp_path / "runtime")
        return world

    monkeypatch.setattr(adapter, "reconstruct", reconstruct)
    out = tmp_path / "episode"
    out.mkdir()
    bindings, _, record = prepare(
        config, robo=robot, robot="robodojo_xpl_rgb", out_dir=out, scratch=out
    )
    assert record["mode"] == "fresh_calibrated_rgb" and bindings["world"].objects()


def test_agent_part_files_stay_in_scratch(world, tmp_path):
    api = AgentWorld(world, ReadOnlyRobot(), tmp_path / "scratch")
    with pytest.raises(ValueError):
        api.publish_parts("object", {"part": {"mesh": "../outside.glb"}})


def test_capture_rejects_resolution_mismatch(tmp_path):
    class WrongCalibration(ReadOnlyRobot):
        def camera_meta(self, camera, step):
            return {**super().camera_meta(camera, step), "shape": [120, 160, 3]}

    with pytest.raises(ValueError, match="resolution differs"):
        capture(WrongCalibration(), tmp_path, "pyrualean_world")
