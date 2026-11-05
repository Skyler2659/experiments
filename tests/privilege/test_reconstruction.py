"""The model is stubbed; export, robot binding and compilation run for real."""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from privilege.reconstruction import reconstruct, run_process


def test_subprocess_budget_and_failure(tmp_path):
    kwargs = dict(cwd=tmp_path, env=dict(os.environ), log=tmp_path / "stage.log")
    with pytest.raises(TimeoutError):
        run_process([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1, **kwargs)
    with pytest.raises(RuntimeError, match="Host episode ended"):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout=10,
            cancelled=lambda: True,
            **kwargs,
        )
    with pytest.raises(RuntimeError, match="exited 7"):
        run_process([sys.executable, "-c", "raise SystemExit(7)"], timeout=10, **kwargs)


def test_failed_reconstruction_records_stage_and_does_not_compile(world, observation, tmp_path):
    runtime = tmp_path / "minimal-runtime"
    for name in (
        ".venv/bin/python",
        ".render-tools/node/bin/node",
        ".render-tools/node_modules/three/build/three.module.js",
        ".render-tools/node_modules/playwright/index.mjs",
    ):
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fail(argv, **kwargs):
        assert "RCWM_FEEDBACK_HOOK" not in kwargs["env"]
        raise RuntimeError("model delivery failed")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="model delivery failed"):
        reconstruct(
            world.root,
            observation["capture"],
            world.root / "geometry/object.glb",
            output,
            {"runtime": runtime, "codex": sys.executable},
            process_runner=fail,
        )
    receipt = json.loads((output / "reconstruction.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["stage"] == "reconstruct"
    assert not (output / "world").exists()


def test_multiview_delivers_every_rgb_and_calibration(world, observation, tmp_path):
    from privilege.observations import load, register_views

    _, packet, image = load(world.root, observation["capture"])
    v = packet["views"][0]
    views = [{"view_id": name, "image": image, "K": v["K"], "T_W_C": v["T_W_C"]}
             for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")]
    capture = register_views(world.root, views, frame=packet["frame"])
    runtime = tmp_path / "runtime"
    for name in (".venv/bin/python", ".render-tools/node/bin/node",
                 ".render-tools/node_modules/three/build/three.module.js",
                 ".render-tools/node_modules/playwright/index.mjs"):
        p = runtime / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    def inspect(argv, **kwargs):
        run = Path(kwargs["cwd"]) / "runs/scene"
        delivered = json.loads((run / "input/views.json").read_text())["views"]
        assert [v["view_id"] for v in delivered] == [v["view_id"] for v in views]
        for actual, source in zip(delivered, views):
            assert (run / "input" / actual["rgb_uri"]).read_bytes() == image.read_bytes()
            assert actual["K"] == source["K"] and actual["T_W_C"] == source["T_W_C"]
        prompt = (run / "solver_prompt.md").read_text()
        assert "Read EVERY assigned reference image" in prompt
        assert "Do not request or use depth, masks, additional RGB views" not in prompt
        assert "geometry_origin='multiview_rgb_rcwm'" in prompt
        raise RuntimeError("inspected inputs")

    with pytest.raises(RuntimeError, match="inspected inputs"):
        reconstruct(world.root, capture["capture"], world.root / "geometry/object.glb",
                    tmp_path / "multiview", {"runtime": runtime, "codex": sys.executable,
                    "view_ids": [v["view_id"] for v in views]}, process_runner=inspect)
    receipt = json.loads((tmp_path / "multiview/reconstruction.json").read_text())
    assert receipt["input_view_count"] == 3


def deliver_scene(private):
    run = Path(private) / "runs/scene"
    scene = run / "fractal/scene"
    scene.mkdir(parents=True)
    shutil.copy2(run / "input/reference.png", scene / "target.png")
    (scene / "part.json").write_text(
        json.dumps({"module": "component.js", "export": "build", "children": []})
    )
    (scene / "parameters.json").write_text(
        json.dumps({"objects": {"known_robot": {}, "block": {"movable": True}}})
    )
    (scene / "component.js").write_text("""
import * as THREE from 'three';
export function build() {
 const group = new THREE.Group();
 const block = new THREE.Mesh(new THREE.BoxGeometry(.2,.1,.3), new THREE.MeshStandardMaterial());
 block.position.set(.3,.4,.5); block.userData.object_id='block'; group.add(block);
 return group;
}
""")
    (scene / "index.html").write_text("""<html><head></head><body><script type="module">
import * as THREE from 'three';
import {GLTFLoader} from '/.render-tools/node_modules/three/examples/jsm/loaders/GLTFLoader.js';
import {build} from './component.js';
window.ready=(async()=>{
 window.scene=new THREE.Scene(); scene.add(build());
 const asset=await new GLTFLoader().loadAsync('/runs/scene/input/known_robot.glb');
 asset.scene.userData={object_id:'known_robot', geometry_origin:'public_urdf_measured_joints'};
 scene.add(asset.scene); scene.add(new THREE.AmbientLight(0xffffff,2));
 window.camera=new THREE.PerspectiveCamera(45,16/12,.01,10);
 camera.up.set(0,0,1); camera.position.set(1,-1,1); camera.lookAt(0,0,.3);
 window.renderer=new THREE.WebGLRenderer(); renderer.setSize(16,12);
 document.body.appendChild(renderer.domElement); renderer.render(scene,camera);
 return true;
})();</script></body></html>""")


@pytest.mark.skipif(
    not os.getenv("PRIVILEGE_TEST_RUNTIME"), reason="set PRIVILEGE_TEST_RUNTIME for browser export"
)
@pytest.mark.parametrize("multiview", [False, True])
def test_calibrated_export_compile_roundtrip(world, observation, tmp_path, multiview):
    robot = trimesh.creation.box(extents=[0.05, 0.05, 0.1])
    robot.apply_translation([0, 0, 0.2])
    robot_file = tmp_path / "robot.glb"
    robot.export(robot_file)
    template = tmp_path / "model-delivery"
    (template / "runs/scene/input").mkdir(parents=True)
    from privilege.observations import load

    _, _, image = load(world.root, observation["capture"])
    if multiview:
        from privilege.observations import register_views
        _, packet, _ = load(world.root, observation["capture"])
        v = packet["views"][0]
        observation = register_views(world.root, [
            {"view_id": name, "image": image, "K": v["K"], "T_W_C": v["T_W_C"]}
            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")], frame=packet["frame"])
    shutil.copy2(image, template / "runs/scene/input/reference.png")
    deliver_scene(template)
    codex = tmp_path / "fake-codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "import sys, shutil\nfrom pathlib import Path\n"
        "if '--version' in sys.argv:\n    print('codex-cli 0.154.0')\n"
        "else:\n"
        f"    shutil.copytree({str(template / 'runs/scene/fractal/scene')!r}, "
        "Path.cwd() / 'runs/scene/fractal/scene', dirs_exist_ok=True)\n"
        "    print('session id: 00000000-0000-0000-0000-000000000000')\n"
        "    print('tokens used\\n0')\n"
    )
    codex.chmod(0o755)
    home = tmp_path / "synthetic-codex-home"
    home.mkdir()
    (home / "auth.json").write_text('{"test_stub": true}')
    seen = []

    def stage(argv, **kwargs):
        seen.append(Path(kwargs["log"]).stem)
        if seen[-1] == "reconstruct":
            meta = json.loads(
                (Path(kwargs["cwd"]) / "runs/scene/input/camera_metadata.json").read_text()
            )
            assert meta["frame"] == "pyrualean_world" and meta["camera_axes"] == "opencv"
        run_process(argv, **kwargs)

    result = reconstruct(
        world.root,
        observation["capture"],
        robot_file,
        tmp_path / "reconstructed",
        {
            "runtime": os.environ["PRIVILEGE_TEST_RUNTIME"],
            "codex": str(codex),
            "codex_home": str(home),
            "max_depth": 0,
            "max_cycles": 1,
            "timeout_s": 90,
            **({"view_ids": ["cam_head", "cam_left_wrist", "cam_right_wrist"]} if multiview else {}),
        },
        process_runner=stage,
    )
    assert seen == ["reconstruct", "check_program", "export", *(["render_views"] if multiview else []), "compile"]
    if multiview:
        from PIL import Image
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            with Image.open(tmp_path / "reconstructed/calibrated_renders" / (camera + ".png")) as im:
                assert list(im.size) == v["image_size"]
    assert result.verify_binding()["status"] == "BOUND"
    assert np.allclose(result.snapshot().position("block"), [0.3, 0.4, 0.5], atol=1e-6)
    assert np.allclose(result.mesh("known_robot").bounds, robot.bounds, atol=1e-6)
    assert not (result.root / "task").exists() and not (result.root / "tools").exists()
    assert json.loads((result.root.parent / "reconstruction.json").read_text())["status"] == "BUILT"
