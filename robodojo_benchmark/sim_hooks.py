"""Episode preparation inside the same frozen official reset used for actions."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from codeworld_robot.io import atomic, read


def public_robot_mesh(env, destination, settings, robot_state):
    import trimesh
    from codeworld_robot.kinematics import URDFKinematics, mesh_path
    from codeworld_robot.collision import origin
    urdf = Path(settings["robodojo_root"]) / "Assets/Robots/x5/X5A.urdf"
    xml = ET.parse(urdf).getroot()
    meshes, checks = [], []
    for i, robot in enumerate(env.robot_manager.robot_list):
        base_pose = np.asarray(robot.entity_origin_pose).reshape(-1, 7)[0]
        base = np.eye(4)
        base[:3, 3] = base_pose[:3] - settings["world_origin_S_m"]
        base[:3, :3] = Rotation.from_quat(np.r_[base_pose[4:7], base_pose[3]]).as_matrix()
        tool = np.eye(4); tool[0, 3] = float(robot.gripper_bias)
        model = URDFKinematics(urdf, base=base, tcp_link="link6", T_link_tcp=tool)
        row = robot_state["arms"][i]
        frames = model.transforms(row["joint_positions"], row["gripper_qpos"])
        T = frames["link6"] @ tool
        actual = np.asarray(row["eef_pose_W"]["translation_m"])
        error = float(np.linalg.norm(T[:3, 3] - actual))
        checks.append({"arm": i, "fk_translation_error_m": error})
        if error > .003:
            raise ValueError(f"Public robot FK mismatch: {checks[-1]}")
        for link in xml.findall("link"):
            for visual in link.findall("visual"):
                spec = visual.find("geometry/mesh")
                if spec is None:
                    continue
                mesh = trimesh.load(mesh_path(urdf, spec.get("filename")), force="mesh", process=False)
                mesh.apply_scale(np.fromstring(spec.get("scale", "1 1 1"), sep=" "))
                mesh.apply_transform(frames[link.get("name")] @ origin(visual.find("origin")))
                meshes.append(mesh)
    trimesh.util.concatenate(meshes).export(destination)
    return checks


def scan_scene(env, root, settings, state, frame):
    """The fixed scan never calls sim_step and restores the original camera."""
    from scripts.robodojo42.render_ready import wait_ready
    directory = root / "preparation/capture"
    directory.mkdir(parents=True)
    initial_count = int(env.take_action_cnt[0])
    views, readiness = [], {}
    origin = np.asarray(settings["world_origin_S_m"], float)
    camera = env.camera_manager.cameras[0][0]
    native_pose = tuple(np.asarray(v).copy() for v in camera.get_world_pose(camera_axes="ros"))
    data, readiness["initial"] = wait_ready(lambda: frame(False), env.capture_manager.step, initial=True)

    def record(view_id, j=0, scanner=False, role="fit"):
        data, readiness[view_id] = wait_ready(lambda: frame(False), env.capture_manager.step)
        rgb = np.asarray(data[j]["rgb"][0]["data"])[:, :, :3].copy()
        depth = np.asarray(data[j]["distance_to_image_plane"][0]["data"]).squeeze().astype("float32")
        if rgb.std() < 1 or not np.isfinite(depth).any():
            raise ValueError(f"Empty camera frame: {view_id}")
        rgb_path, depth_path = directory / f"{view_id}.png", directory / f"{view_id}_depth.npy"
        Image.fromarray(rgb).save(rgb_path); np.save(depth_path, depth)
        cam = env.camera_manager.cameras[0][j]
        pos, q = cam.get_world_pose(camera_axes="ros")
        T = np.eye(4); T[:3, 3] = np.asarray(pos) - origin
        T[:3, :3] = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        views.append({"view_id": view_id, "role": role, "rgb_uri": rgb_path.name, "depth_uri": depth_path.name,
                      "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                      "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                      "K": np.asarray(env.camera_manager.get_camera_intrinsics(j)).tolist(), "T_W_C": T.tolist(),
                      "image_size": [rgb.shape[1], rgb.shape[0]], "depth_units": "m", "depth_convention": "camera_z_m",
                      "surface_enabled": True, "external_scanner": scanner})

    try:
        for j in range(len(env.camera_manager.cameras[0])):
            record("agentview" if j == 0 else f"wrist{j}", j)
        center, radius = np.array([0., 0., .16]), 1.12
        for i in range(18):
            angle, elevation = 2 * np.pi * (i % 6) / 6 + .15, np.deg2rad([25, 45, 65][i // 6])
            pos = center + radius * np.array([np.cos(angle) * np.cos(elevation), np.sin(angle) * np.cos(elevation), np.sin(elevation)])
            z = (center - pos) / radius; x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
            q = Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat()
            camera.set_world_pose(position=pos + origin, orientation=np.r_[q[3], q[:3]], camera_axes="ros")
            record(f"scanner_{i:06d}", scanner=True, role="holdout" if i in (3, 9, 15) else "fit")
    finally:
        camera.set_world_pose(position=native_pose[0], orientation=native_pose[1], camera_axes="ros")
    data, readiness["native_revisit"] = wait_ready(lambda: frame(False), env.capture_manager.step, initial=True)
    revisit = np.asarray(data[0]["rgb"][0]["data"])[:, :, :3]
    first = np.asarray(Image.open(directory / "agentview.png"))
    h, w = first.shape[:2]
    def blocks(a):
        return a[:h-h%8, :w-w%8].astype(float).reshape(h//8, 8, w//8, 8, 3).mean((1, 3))
    mae = float(np.mean(abs(blocks(revisit) - blocks(first))))
    readiness["native_revisit_mae"] = mae
    atomic(directory / "render_readiness.json", readiness)
    if mae > 3:
        raise ValueError(f"Appearance changed during frozen scan: block RGB MAE={mae:.3f}")
    if int(env.take_action_cnt[0]) != initial_count:
        raise RuntimeError("Acquisition advanced the official action counter")
    robot_state = state()
    checks = public_robot_mesh(env, directory / "known_robot.glb", settings, robot_state)
    cap = {"observation_id": settings["run_id"] + "_initial", "frame": "robot_base_W",
           "world_origin_S_m": origin.tolist(), "views": views, "robot_state": robot_state,
           "render_readiness_verified": True, "robot_task_actions": initial_count, "fk_checks": checks,
           "state_policy": "same official reset as execution; physics frozen during acquisition",
           "scanner": "external fixed virtual scanner; augmented observation protocol"}
    atomic(directory / "capture.json", cap)
    return cap


def prepare_episode(env, root, settings, state, capture, frame):
    root = Path(root)
    instruction = env.obs_manager.instruction[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Native public task instruction is missing")
    task = read(root / "task/task.json")
    brief = read(root / "task/public_task_brief.json")
    from .task_brief import apply_task_brief
    apply_task_brief(task, instruction, brief)
    atomic(root / "task/task.json", task)
    # Probes can use only native RGB-D and skip the expensive scanner.
    if settings.get("probe_native_only"):
        capture()
    else:
        scan_scene(env, root, settings, state, frame)
    oracle = None
    if settings["representation"] == "ground_truth":
        from .ground_truth import GroundTruthWorld
        oracle = GroundTruthWorld(env, root, settings["world_origin_S_m"], settings["gt_mode"], reuse_episode=settings.get("reuse_gt_episode"))
        oracle.sync(state())
    atomic(root / "preparation/ready.json", {"status": "CAPTURED", "official_actions": int(env.take_action_cnt[0])})
    deadline = time.monotonic() + settings.get("preparation_timeout_s", 14400)
    while not (root / "task/world_ready.json").exists():
        if (root / "task/stop_server.json").exists():
            raise InterruptedError("Preparation stopped")
        if time.monotonic() >= deadline:
            raise TimeoutError("Reconstruction did not publish a world before its preparation deadline")
        time.sleep(.2)
    if settings["representation"] != "none":
        from codeworld_workspace import World
        World(root).verify_binding()
    elif (root / "world.json").exists():
        raise ValueError("CodeWorld was supplied to the no-CodeWorld strategy")
    return oracle
