"""Native simulator initialization and released baseline lifecycle."""

import argparse
import sys
from pathlib import Path
import numpy as np
from .io import atomic


def initialize(settings, io, *, official, benchmark, baseline):
    official_actions = None
    ROOT = Path(settings["robodojo_root"])
    sys.path.insert(0, str(ROOT / "XPolicyLab"))
    sys.path.insert(0, str(ROOT))
    import warp
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    opts = ap.parse_args([])
    device = int(settings.get("sim_device", 0))
    renderer = int(settings.get("renderer_gpu", 0))
    opts.headless = True
    opts.enable_cameras = True
    opts.device = f"cuda:{device}"
    opts.kit_args = f"--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 --/renderer/multiGpu/enabled=False --/renderer/activeGpu={renderer}"
    from env.camera_manager.capture.render_sync import (
        add_zero_delay_kit_args,
        wait_for_latest_cameras,
    )

    add_zero_delay_kit_args(opts)
    app = AppLauncher(opts).app
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_randomization, process_config
    from env.seed_manager.seed_manager import SeedManager
    from task.RoboDojo.task_registry import load_task_class

    task_name = settings.get("task_name", "build_tower")
    seed = int(settings.get("seed", 0))
    layout_id = int(settings.get("layout_id", seed))
    base = ROOT / "env_cfg"
    ec = load_yaml(str(base / settings.get("env_config", "arx_x5.yml")))
    ec.update(task_name=task_name, num_envs=1, device_id=0, seed=seed)
    refs = ec["config"]
    cfg = OmegaConf.create(
        {
            k: load_yaml(str(base / k / (refs[k] + ".yml")))
            for k in ("sim", "scene", "camera", "robot")
        }
    )
    cfg.task_env = load_yaml(str(ROOT / f"task/RoboDojo/config/{task_name}.yml"))
    cfg.eval_cfg = ec
    cfg.sim.scene.num_envs = 1
    cfg.sim.seed = [0] if benchmark else [seed]
    cfg.camera.default_frequency = 25
    if not baseline:
        for _, row in cfg.camera.annotator.items():
            row.depth_capture = {"type": "distance_to_image_plane", "device": "cpu"}
    cfg = process_randomization(cfg)
    cfg, _ = process_config(cfg, task_name=task_name)
    if not benchmark:
        cfg.sim.render = OmegaConf.create({"antialiasing_mode": "Off"})
    if official:
        from .robodojo_official import create_official_env, OfficialActions

        env = create_official_env(
            cfg,
            app,
            layout_id,
            managed_loop=benchmark,
            result_directory=settings.get("evaluator_directory"),
            policy_name=(
                "RoboDojo_Agent_L3_Inspect_EEF" if baseline else "codeworld_file_policy"
            ),
        )
        official_actions = OfficialActions(
            env, settings.get("evaluator_directory", io), expose_outcome=not benchmark
        )
    else:
        sm = SeedManager(cfg.eval_cfg)
        sm.init_eval()
        layout = sm.get_seed_scene_info(seed)
        _, klass = load_task_class(task_name)
        env = klass(cfg, app)
        env.step_lim = float("inf")
        env.scene_manager.layout_manager.set_saved_layout(0, layout)
        env.reset(seed=[seed])
        env.scene_manager.apply_saved_poses(env_idx_list=[0])
        env.robot_manager.set_origin_endpose()
        env.robot_manager.set_robot_init_state()
    return ROOT, app, env, cfg, official_actions, wait_for_latest_cameras


def run_baseline(
    env, app, root, io, settings, official_actions, wait_for_latest_cameras
):
    import cv2

    # The released L3 deploy explicitly ignores model_client: its planner and
    # EEF executor run in this process. Only the unused WebSocket is replaced.
    if settings.get("probe_native_only"):
        # Initialization-only check: exercise the released observation decoder,
        # action binding and EEF planner import, with no model call or task action.
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect import deploy as native_deploy
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.deploy import _planner

        def baseline_probe():
            before = int(env.take_action_cnt[0])
            spec = native_deploy._action_spec(env)
            observation = native_deploy._observation(env, 0)
            assert callable(_planner(env))
            if not observation.images:
                raise ValueError("RoboProbe has no native RGB observation")
            if int(env.take_action_cnt[0]) != before:
                raise RuntimeError("Baseline observation probe advanced native actions")
            atomic(
                Path(settings["evaluator_directory"]) / "baseline_interface_probe.json",
                {
                    "rgb_images": len(observation.images),
                    "state_fields": list(observation.state),
                    "action_dimensions": len(spec.labels),
                    "action_labels": list(spec.labels),
                    "model_called": False,
                    "actions": 0,
                    "policy": "RoboDojo_Agent_L3_Inspect_EEF",
                },
            )
            env.success[0] = False
            env.is_episode_end()

        env.eval_one_episode = baseline_probe
    from .recording import FrameTimeline

    timeline = FrameTimeline(root)
    baseline_video = cv2.VideoWriter(
        str(io / "continuous.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480)
    )
    original_take = env.take_action

    def recorded_take(*args, **kwargs):
        result = original_take(*args, **kwargs)
        env.render()
        wait_for_latest_cameras(env, capture_manager=env.capture_manager)
        data = env.capture_manager.step()
        baseline_video.write(
            np.asarray(data[0]["rgb"][0]["data"])[:, :, :3][:, :, ::-1].copy()
        )
        timeline.operation = {
            "op": "native_take_action",
            "args": str(args),
            "kwargs": str(kwargs),
        }
        timeline.record({"official_actions": int(env.take_action_cnt[0])})
        return result

    env.take_action = recorded_take
    try:
        env.run_eval()
    finally:
        baseline_video.release()
        official_actions.finalize()
        env.close()
        app.close()
    return
