"""RoboDojo direct I/O: native X5 IK + joint servo, no task-policy gates."""

import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np
from .io import atomic, read
from .request_loop import complete_receipt
from .robodojo_dispatch import SessionState, X5Dispatcher


def main():
    session = SessionState()
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--arm", type=int, default=0)
    p.add_argument("--config")
    a, extra = p.parse_known_args()
    root = Path(a.workspace).resolve()
    io = root / "robot_io"
    io.mkdir(parents=True)
    for d in ["requests", "responses", "captures"]:
        (io / d).mkdir(exist_ok=True)
    settings = read(a.config) if a.config else {}
    official = bool(settings.get("official_evaluation", False))
    benchmark = settings.get("benchmark_protocol") == "robodojo-v1"
    baseline = benchmark and settings.get("strategy") == "roboprobe_l3"
    if benchmark and not official:
        raise ValueError("Benchmark episodes require the official evaluator")
    oracle = None
    if settings.get("motion_dependencies"):
        sys.path.insert(0, settings["motion_dependencies"])
    from .robodojo_runtime import initialize, run_baseline
    from scipy.spatial.transform import Rotation
    from PIL import Image
    import cv2

    ROOT, app, env, cfg, official_actions, wait_for_latest_cameras = initialize(
        settings, io, official=official, benchmark=benchmark, baseline=baseline
    )
    if baseline:
        run_baseline(
            env, app, root, io, settings, official_actions, wait_for_latest_cameras
        )
        return
    rm = env.robot_manager
    origin = np.asarray(
        settings.get("world_origin_S_m", [0.0, -0.45, 0.765]), dtype=float
    )
    steps = 0
    captures = 0
    session.active = a.arm
    session.stopped = False
    from .x5_ik import X5IK
    from .multiarm import arm_index, resolve_commands, joint_frames, CommandLatch
    from .pose import transform
    from .motion import joint_trajectory

    names = [r.arm_name for r in rm.robot_list]
    session.active = arm_index(session.active, names)
    solvers = []
    for i, r in enumerate(rm.robot_list):
        base_pose = np.asarray(r.entity_origin_pose).reshape(-1, 7)[0]
        base_T = np.eye(4)
        base_T[:3, 3] = base_pose[:3] - origin
        base_T[:3, :3] = Rotation.from_quat(
            np.r_[base_pose[4:7], base_pose[3]]
        ).as_matrix()
        solvers.append(
            X5IK(
                ROOT / "Assets/Robots/x5/X5A.urdf",
                i,
                base=base_T,
                tcp_offset=[float(r.gripper_bias), 0, 0],
            )
        )
    capabilities = {
        "schema": "codeworld-robot-v2",
        "robot_backend": "robodojo",
        "robot_model": "dual_x5",
        "arms": [
            {
                "arm": i,
                "arm_name": name,
                "base_T_W": solvers[i].base.tolist(),
                "tcp_offset_link6_m": solvers[i].tcp_offset.tolist(),
                "joint_limits_rad": solvers[i].limits,
                "approach_axis_tcp": [1, 0, 0],
            }
            for i, name in enumerate(names)
        ],
        "synchronized_commands": True,
        "ik_query": True,
        "full_arm_collision_check": False,
        "joint_targets": True,
        "raw_osc": False,
    }
    physics_dt = float(cfg.sim.dt) * int(cfg.sim.decimation)
    record_interval = max(1, round(1 / (25 * physics_dt)))
    if official:
        capabilities.update(official_actions.status(), official_evaluation=True)
    session.collision = None
    if settings.get("full_arm_collision_check", False) and not benchmark:
        from .collision import robodojo_collision
        from codeworld_workspace import World

        session.collision = robodojo_collision(
            ROOT / "Assets/Robots/x5/X5A.urdf",
            {i: s.base for i, s in enumerate(solvers)},
            World(root),
        )
        capabilities.update(
            full_arm_collision_check=True,
            collision_coverage="sampled_URDF_joint_path",
            self_collision_check=True,
            inter_arm_collision_check=True,
            payload_collision_check=True,
            collision_contact_names="arm0/link7, arm0/link8, ...; object IDs from World",
            motion_profile=settings.get("motion_profile", "ruckig"),
            max_velocity_rad_s=settings.get("max_velocity_rad_s", 2.0),
            max_acceleration_rad_s2=settings.get("max_acceleration_rad_s2", 8.0),
        )
    from .recording import FrameTimeline

    timeline = FrameTimeline(root)
    video = cv2.VideoWriter(
        str(io / "continuous.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480)
    )
    # Initialize persistent actuator targets once; never reset the inactive arm.
    initial = [rm.get_joint(r)[0].copy() for r in rm.robot_list]
    initial_grippers = [
        rm.get_end_effector_real_val(r)[0].copy() for r in rm.robot_list
    ]
    latch = CommandLatch(
        {
            key: {"position": value.copy(), "velocity": np.zeros_like(value)}
            for i, r in enumerate(rm.robot_list)
            for key, value in [
                (rm.process_name(r.arm_name), initial[i]),
                (rm.process_name(r.gripper_name), initial_grippers[i]),
            ]
        }
    )

    def state():
        rows = []
        for i, r in enumerate(rm.robot_list):
            pose = rm.get_real_endpose(r)[0]
            rot = Rotation.from_quat(np.r_[pose[4:7], pose[3]])
            tcp = pose[:3] + rot.apply([r.gripper_bias, 0, 0]) - origin
            rows.append(
                {
                    "arm": i,
                    "arm_name": r.arm_name,
                    "eef_pose_W": {
                        "translation_m": tcp.tolist(),
                        "quaternion_xyzw": rot.as_quat().tolist(),
                    },
                    "joint_positions": rm.get_joint(r)[0].tolist(),
                    "gripper_qpos": rm.get_end_effector_real_val(r)[0].tolist(),
                    "tcp_offset_link6_m": [float(r.gripper_bias), 0, 0],
                }
            )
        for i, row in enumerate(rows):
            row.update(
                home_joint_positions=initial[i].tolist(),
                home_gripper_qpos=initial_grippers[i].tolist(),
            )
        s = dict(rows[session.active])
        s.update(
            arms=rows,
            active_arm=session.active,
            robot_state_ref=f'{settings.get("run_id",root.name)}:{steps}',
            control_steps=steps,
            timestamp_ns=round(steps * physics_dt * 1e9),
            clock_domain="simulation",
        )
        s.update(official_actions.status() if official else {})
        atomic(io / "robot_state.json", s)
        return s

    def frame(record=True):
        env.render()
        wait_for_latest_cameras(env, capture_manager=env.capture_manager)
        data = env.capture_manager.step()
        if record:
            video.write(
                np.asarray(data[0]["rgb"][0]["data"])[:, :, :3][:, :, ::-1].copy()
            )
            timeline.record(
                {"control_steps": steps, "simulation_time_s": steps * physics_dt}
            )
        return data

    def capture():
        nonlocal captures
        if oracle is not None:
            oracle.sync(state())
        for _ in range(3):
            env.render()
        data = frame(False)
        d = io / "captures" / f"obs{captures:05d}"
        d.mkdir()
        views = []
        for j, row in enumerate(data):
            rgb = np.asarray(row["rgb"][0]["data"])[:, :, :3]
            depth = np.asarray(row["distance_to_image_plane"][0]["data"]).squeeze()
            Image.fromarray(rgb).save(d / f"cam{j}.png")
            np.save(d / f"cam{j}_depth.npy", depth)
            cam = env.camera_manager.cameras[0][j]
            pos, q = cam.get_world_pose(camera_axes="ros")
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
            T[:3, 3] = np.asarray(pos) - origin
            views.append(
                {
                    "view_id": f"cam{j}",
                    "rgb_uri": str(d / f"cam{j}.png"),
                    "depth_uri": str(d / f"cam{j}_depth.npy"),
                    "K": env.camera_manager.get_camera_intrinsics(j).tolist(),
                    "T_W_C": T.tolist(),
                    "depth_convention": "camera_z_m",
                }
            )
        c = {"observation_id": d.name, "views": views, "robot_state": state()}
        atomic(d / "capture.json", c)
        atomic(io / "latest_capture.json", c)
        captures += 1
        return str(d / "capture.json")

    def tick(control):
        nonlocal steps
        control = latch.merge(control)
        rm.control_manager.push([0], [[control]])
        env.step(rm.control_manager.pop([0]))
        env.sim_step(render=False)
        steps += 1
        if steps % record_interval == 0:
            frame()

    def official_take(joints=None, grippers=None):
        nonlocal steps
        if not official_actions.take(joints, grippers):
            return False
        steps = (
            official_actions.status()["official_actions"] * official_actions.interval
        )
        frame()
        return True

    def planned_frames(targets, args):
        starts = {i: rm.get_joint(r)[0].copy() for i, r in enumerate(rm.robot_list)}
        period = official_actions.interval * physics_dt if official else physics_dt
        if session.collision:
            duration = args.get("duration_s")
            if duration is None and args.get("duration_steps") is not None:
                duration = args["duration_steps"] * physics_dt
            frames = joint_trajectory(
                starts,
                targets,
                period,
                args.get("max_velocity_rad_s", settings.get("max_velocity_rad_s", 2.0)),
                args.get(
                    "max_acceleration_rad_s2",
                    settings.get("max_acceleration_rad_s2", 8.0),
                ),
                duration,
                settings.get("motion_profile", "ruckig"),
                args.get("max_jerk_rad_s3", settings.get("max_jerk_rad_s3", 80.0)),
            )
        else:
            frames = list(
                joint_frames(
                    {i: starts[i] for i in targets}, targets, args.get("duration_steps")
                )
            )
        return starts, frames

    def validate_motion(targets, args):
        starts, frames = planned_frames(targets, args)
        fingers = {
            i: rm.get_end_effector_real_val(r)[0].copy()
            for i, r in enumerate(rm.robot_list)
        }
        allowed = list(settings.get("allowed_contacts", [])) + list(
            args.get("allowed_contacts", [])
        )
        report = session.collision.check_path(
            starts,
            frames,
            fingers,
            allowed_contacts=allowed,
            attachments=args.get("attachments", []),
            max_joint_increment=settings.get("collision_joint_increment_rad", 0.025),
        )
        report.update(
            planned_official_actions=len(frames) if official else None,
            nominal_duration_s=len(frames)
            * (official_actions.interval * physics_dt if official else physics_dt),
        )
        return frames, report

    def execute_joints(targets, n=None, args=None):
        args = args or {}
        starts = {i: rm.get_joint(rm.robot_list[i])[0].copy() for i in targets}
        for i, q in targets.items():
            limits = np.asarray(solvers[i].limits)
            if (
                np.asarray(q).shape != (6,)
                or not np.isfinite(q).all()
                or np.any(q < limits[:, 0])
                or np.any(q > limits[:, 1])
            ):
                raise ValueError("Joint target outside URDF limits")
        if session.collision:
            frames, report = validate_motion(targets, args)
            if report["status"] != "CLEAR":
                return 0, report
            count = 0
            for values in frames:
                if official:
                    if not official_take(values):
                        break
                    count += official_actions.interval
                else:
                    tick(
                        {
                            rm.process_name(rm.robot_list[i].arm_name): {
                                "position": q,
                                "velocity": np.zeros_like(q),
                            }
                            for i, q in values.items()
                        }
                    )
                    count += 1
            # Event-based settling, bounded in official periods. No unconditional tail.
            tolerance = args.get("tracking_tolerance_rad", 0.012)
            extra = 0
            for _ in range(
                int(
                    args.get(
                        "max_settle_actions", settings.get("max_settle_actions", 4)
                    )
                )
            ):
                if official and official_actions.ended:
                    break
                if (
                    max(
                        np.max(abs(rm.get_joint(rm.robot_list[i])[0] - q))
                        for i, q in targets.items()
                    )
                    <= tolerance
                ):
                    break
                if official:
                    if not official_take(targets):
                        break
                    count += official_actions.interval
                else:
                    tick({})
                    count += 1
                extra += 1
            report["settle_actions"] = extra
            return count, report
        if official:
            # Task controller supplies waypoints; each is executed and counted upstream.
            interval = official_actions.interval
            increment = float(settings.get("max_joint_increment_rad", 0.012))
            frames = list(joint_frames(starts, targets, n, max_increment=increment))
            count = 0
            for k in range(interval - 1, len(frames) + interval - 1, interval):
                if not official_take(frames[min(k, len(frames) - 1)]):
                    break
                count += interval
            for _ in range(2):
                if not official_take():
                    break
                count += interval
            return count, None
        frames = joint_frames(starts, targets, n)
        count = 0
        for values in frames:
            tick(
                {
                    rm.process_name(rm.robot_list[i].arm_name): {
                        "position": q,
                        "velocity": np.zeros_like(q),
                    }
                    for i, q in values.items()
                }
            )
            count += 1
        for _ in range(25):
            tick({})
            count += 1
        return count, None

    dispatch = X5Dispatcher(
        session=session,
        root=root,
        names=names,
        solvers=solvers,
        rm=rm,
        validate_motion=validate_motion,
        execute_joints=execute_joints,
        state=state,
        settings=settings,
        official=official,
        official_actions=official_actions,
        official_take=official_take,
        tick=tick,
        benchmark=benchmark,
        env=env,
        capture=capture,
    )

    if not official:
        for _ in range(50):
            env.sim_step(render=False)
    for _ in range(10):
        env.render()
    if benchmark:
        from .hooks import load_hook

        oracle = load_hook(settings["episode_preparer"])(
            env, root, settings, state, capture, frame
        )
        if settings.get("full_arm_collision_check", False):
            from .collision import robodojo_collision, RobotOnlyScene
            from codeworld_workspace import World

            world = (
                RobotOnlyScene()
                if settings["representation"] == "none"
                else World(root)
            )
            session.collision = robodojo_collision(
                ROOT / "Assets/Robots/x5/X5A.urdf",
                {i: s.base for i, s in enumerate(solvers)},
                world,
            )
            capabilities.update(
                full_arm_collision_check=True,
                self_collision_check=True,
                inter_arm_collision_check=True,
                scene_collision_check=settings["representation"] != "none",
                payload_collision_check=settings["representation"] != "none",
                motion_profile=settings.get("motion_profile", "ruckig"),
                collision_coverage="sampled_URDF_joint_path",
            )
    if session.collision is not None:
        capabilities["fixed_mount_contacts"] = [
            list(pair) for pair in session.collision.mounting_contacts
        ]
    atomic(
        root / "task/robot_connection.json",
        {
            "kind": "robodojo_direct_x5",
            "io_directory": str(io),
            "active_arm": session.active,
            "episode_step_limit": int(env.step_lim) if official else None,
            "capabilities": capabilities,
        },
    )
    from codeworld_workspace.robot_home import capture_home

    home = capture_home(state())
    atomic(root / "task/robot_home.json", home)
    capabilities["return_home"] = {
        "required": True,
        "baseline": "task/robot_home.json",
        "joint_tolerance_rad": home["joint_tolerance_rad"],
        "gripper_tolerance_m": home["gripper_tolerance_m"],
    }
    atomic(
        root / "task/robot_connection.json",
        {
            "kind": "robodojo_direct_x5",
            "io_directory": str(io),
            "active_arm": session.active,
            "episode_step_limit": int(env.step_lim) if official else None,
            "capabilities": capabilities,
        },
    )
    atomic(
        io / "ready.json",
        {
            "status": "READY",
            "capture": capture(),
            "arm": session.active,
            "episode_step_limit": int(env.step_lim) if official else None,
        },
    )
    print("READY", flush=True)

    def serve():
        from .request_loop import serve_requests

        serve_requests(
            app,
            root,
            dispatch,
            timeline,
            should_stop=lambda: session.stopped
            or (benchmark and (root / "task/stop_server.json").exists()),
            steps=lambda: steps,
        )
        if benchmark and not official_actions.ended:
            # Same early-stop path used by upstream policy capability failures. Native
            # final_check still evaluates all task predicates; no action is added.
            env.success[0] = False
            env.is_episode_end()

    try:
        if benchmark:
            # Replace only policy transport. Upstream run_eval owns reward registration,
            # terminal evaluation, unstable filtering, score normalization and _result.json.
            env.eval_one_episode = serve
            env.run_eval()
        else:
            serve()
    finally:
        video.release()
        if official:
            official_actions.finalize()
        atomic(
            io / "closed.json",
            {"status": "CLOSED", "steps": steps, "captures": captures},
        )
        env.close()
        app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Native SimulationApp teardown can otherwise hide initialization exceptions.
        try:
            index = sys.argv.index("--config")
            config = read(sys.argv[index + 1])
            if config.get("evaluator_directory"):
                atomic(
                    Path(config["evaluator_directory"]) / "server_error.json",
                    {
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
        except Exception:
            pass
        traceback.print_exc()
        import os

        os._exit(1)
