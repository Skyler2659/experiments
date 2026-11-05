"""Calibrated FastCodeWorld reconstruction, without an episode/robot harness."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .observations import load
from .robot_asset_check import verify_robot_asset
from .reconstruction_inputs import snapshot_inputs
from pyrualean.resilient_cli import waiting_seconds
from .world import World, atomic, read

RESOURCES = Path(__file__).parent / "resources"


def run_process(argv, *, cwd, env, log, timeout, cancelled=None):
    """Bound the whole reconstruction subprocess group, including recursive children."""
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            [str(a) for a in argv],
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        wait_started = time.time()
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    raise RuntimeError("Host episode ended during reconstruction")
                if time.monotonic() >= deadline + waiting_seconds(env.get("PYRUA_RECOVERY_DIR"), wait_started):
                    raise TimeoutError("Reconstruction wall budget exhausted")
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"Stage exited {process.returncode}; inspect {log}")
        finally:
            # Children may survive even after their parent exits on an error.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def reconstruct(
    observation_root,
    capture,
    robot_glb,
    output,
    config,
    *,
    cancelled=None,
    process_runner=run_process,
):
    """Build a world from selected calibrated RGB views and a public robot snapshot.

    config: runtime, model, reasoning, max_depth, max_cycles, timeout_s, codex,
    optional codex_home. The supplied runtime contains Node/Three/Playwright and
    the RCWM Python/rendering dependencies. No simulator or task metadata is used.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new reconstruction output directory")
    capture_path, packet, image = load(observation_root, capture)
    cfg = dict(config)
    requested = cfg.get("view_ids", [packet["views"][0]["view_id"]])
    available = {v["view_id"]: v for v in packet["views"]}
    if not requested or len(set(requested)) != len(requested) or any(v not in available for v in requested):
        raise ValueError("view_ids must select unique views from the same capture")
    views = [available[v] for v in requested]
    image = capture_path.parent / views[0]["rgb_uri"]
    multiview = len(views) > 1
    runtime = Path(cfg["runtime"]).expanduser().resolve()
    depth, cycles, timeout = (
        cfg.get("max_depth", 4),
        cfg.get("max_cycles", 3),
        cfg.get("timeout_s", 14400),
    )
    if type(depth) is not int or depth < 0 or type(cycles) is not int or cycles < 1 or timeout <= 0:
        raise ValueError("Invalid recursion or wall budget")
    for rel in (
        ".venv/bin/python",
        ".render-tools/node/bin/node",
        ".render-tools/node_modules/three/build/three.module.js",
        ".render-tools/node_modules/playwright/index.mjs",
    ):
        if not (runtime / rel).exists():
            raise ValueError("Missing reconstruction runtime dependency: " + str(runtime / rel))
    robot_glb = Path(robot_glb).resolve()
    if not robot_glb.is_file():
        raise ValueError("Provide a robot GLB posed in the observation's world frame")
    codex = shutil.which(cfg.get("codex", "codex"))
    if codex is None:
        raise ValueError("Reconstruction Codex executable was not found")
    output.mkdir(parents=True)
    private = output / "runtime"
    private.mkdir()
    for name in (".render-tools", ".system-libs", ".venv", ".browsers", "platform.json"):
        if (runtime / name).exists():
            (private / name).symlink_to(
                runtime / name, target_is_directory=(runtime / name).is_dir()
            )
    (private / "bin").mkdir()
    (private / "bin/codex").symlink_to(Path(codex).resolve())
    if (runtime / "bin/date").exists():
        (private / "bin/date").symlink_to((runtime / "bin/date").resolve())
    (private / "AGENTS.md").write_text(
        "Reconstruct the scene only from the assigned RGB, calibration and public robot asset. "
        "Do not operate a robot or inspect simulator objects, evaluator state or other episodes. "
        "Work inside this reconstruction run. Follow the recursive solver's node contracts.\n"
    )
    name = "scene"
    run = private / "runs" / name
    run.mkdir(parents=True)
    shutil.copytree(RESOURCES / "recursive_cwm", run / "runner_source")
    prompt = (RESOURCES / "fast-single-rgb.md").read_text()
    if multiview:
        prompt = prompt.replace(
            "The target is one RGB image. Do not request or use depth, masks, additional RGB views,\nsimulator object poses or prior reconstructions.",
            "The targets are the synchronized calibrated RGB views listed in input/views.json. "
            "Read EVERY assigned reference image, including both wrist views. Do not request or use "
            "depth, masks, unassigned RGB views, simulator object poses or prior reconstructions.",
        ).replace("not extra scene views", "shared support metadata")
        prompt = prompt.replace("Use the supplied single\ncamera calibration for framing; monocular", "Use each supplied\ncamera calibration for framing; remaining")
        prompt = prompt.replace("geometry_origin='single_rgb_rcwm'", "geometry_origin='multiview_rgb_rcwm'")
        prompt += ("\nMulti-view fitting contract: reference.png is the primary head view, NOT the only "
                   "target. At every node inspect all useful assigned views with their fixed K and T_W_C. "
                   "Keep one shared metric scene; never fit a separate shape or camera per view. "
                   "Occlusion and cropping are not missing geometry. Do not invent details behind the gripper. "
                   "Children inherit input/views.json and may use those full-frame references to resolve "
                   "their assigned objects. Before finishing, render the delivered whole scene at ALL "
                   "assigned cameras and compare each with its reference. The supplied helper is "
                   "input/render_calibrated.mjs; invoke Node with arguments RUNTIME RUN_NAME OUTPUT_DIR. "
                   "It reads the delivered viewer and its original materials, not identity colors. "
                   "Record which views constrained each object's dimensions/pose and remaining ambiguities "
                   "in account.md. Do not spend repair cycles matching fine floor/wall texture.\n")
    prompt += (
        "\nPrivilege host frame contract: W is the supplied observation frame "
        + repr(packet["frame"])
        + ", in meters, Z-up. Preserve the camera's metric placement and "
        "the robot GLB coordinates; do not recenter the scene at a robot base. "
        "input/camera_metadata.json uses OpenCV camera axes (+x right, +y down, +z forward); "
        "a Three.js camera uses +x right, +y up, -z forward. Convert the camera axes explicitly.\n"
    )
    (run / "solver_prompt.md").write_text(prompt)
    inp = run / "input"
    inp.mkdir()
    shutil.copy2(image, inp / "reference.png")
    shutil.copy2(robot_glb, inp / "known_robot.glb")
    selected = []
    for i, v in enumerate(views):
        filename = "reference.png" if i == 0 else f"reference_{i}.png"
        shutil.copy2(capture_path.parent / v["rgb_uri"], inp / filename)
        selected.append({**v, "rgb_uri": filename})
    assigned_metadata = {"views.json": {"frame": packet["frame"], "views": selected}}
    shutil.copy2(RESOURCES / "render_calibrated.mjs", inp / "render_calibrated.mjs")
    import hashlib

    view = views[0]
    assigned_metadata.update({
        "input.json": {
            "reference": "reference.png", "sha256": view["rgb_sha256"],
            "image_size": view["image_size"],
            "input_kind": "multiview_rgb" if multiview else "single_rgb",
            "view_ids": requested, "previous_reconstruction_used": False,
        },
        "camera_metadata.json": {
            "K": view["K"], "T_W_C": view["T_W_C"],
            "image_size": view["image_size"], "camera_axes": "opencv", "frame": packet["frame"],
        },
        "robot_binding.json": {
            "reference_sha256": view["rgb_sha256"],
            "robot_node": {"asset": {"sha256": hashlib.sha256(robot_glb.read_bytes()).hexdigest()}},
            "frame": packet["frame"], "robot_state": packet["robot_state"],
        },
    })
    snapshot_inputs(inp, assigned_metadata)
    shutil.copy2(Path(__file__).with_name("reconstruction_inputs.py"), inp / "ensure_inputs.py")
    env = dict(os.environ)
    for key in ("RCWM_FEEDBACK_HOOK", "RCWM_ASSIST_CONFIG", "RCWM_ADAPTIVE_DISPATCH"):
        env.pop(key, None)
    env.update(
        RCWM_ROOT=str(private),
        RCWM_PROMPT=str(run / "solver_prompt.md"),
        RCWM_MODEL=cfg.get("model", "gpt-6-astra"),
        RCWM_REASONING=cfg.get("reasoning", "high"),
        RCWM_CLEAN_CODEX_HOME="1",
        RCWM_EXPORT_OBJECTS="1",
        RCWM_EXPORT_CANONICAL_VIEW="0",
        RCWM_SANDBOX=cfg.get("sandbox", "workspace-write"),
    )
    if cfg.get("codex_home"):
        env["CODEX_HOME"] = str(Path(cfg["codex_home"]).expanduser().resolve())
    if cfg.get("codex_config"):
        env["RCWM_CODEX_CONFIG"] = str(Path(cfg["codex_config"]).expanduser().resolve())
    env["PATH"] = os.pathsep.join(
        [str(private / "bin"), str(private / ".render-tools/node/bin"), env.get("PATH", "")]
    )
    if (private / ".browsers").is_dir():
        env["PLAYWRIGHT_BROWSERS_PATH"] = str((private / ".browsers").resolve())
    libraries = private / ".system-libs/usr/lib/x86_64-linux-gnu"
    if libraries.is_dir():
        env["LD_LIBRARY_PATH"] = str(libraries) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    started = time.monotonic()
    started_wall = time.time()
    receipt = {
        "status": "RUNNING",
        "frame": packet["frame"],
        "model": env["RCWM_MODEL"],
        "reasoning": env["RCWM_REASONING"],
        "observation_id": packet["observation_id"],
        "max_depth": depth,
        "max_cycles": cycles,
        "timeout_s": timeout,
        "metric_validation": False,
        "source": "calibrated_fastcodeworld",
        "input_view_ids": requested,
        "input_view_count": len(views),
    }
    atomic(output / "reconstruction.json", receipt)

    def stage(label, argv):
        receipt["stage"] = label
        atomic(output / "reconstruction.json", receipt)
        remaining = timeout - (time.monotonic() - started) + waiting_seconds(env.get("PYRUA_RECOVERY_DIR"), started_wall)
        if remaining <= 0:
            raise TimeoutError("Reconstruction wall budget exhausted")
        for retry in range(4):
            log = output / (label + (f".retry{retry}" if retry else "") + ".log")
            try:
                process_runner(argv, cwd=private, env=env, log=log,
                               timeout=remaining, cancelled=cancelled)
                receipt.setdefault("completed_stages", []).append(label)
                atomic(output / "reconstruction.json", receipt)
                return
            except RuntimeError:
                detail = log.read_text(errors="replace")[-16000:] if log.exists() else ""
                transient = any(x in detail for x in ("ENOENT", "FileNotFoundError", "Stale file handle", "Input/output error"))
                if not transient or retry == 3:
                    raise
                # Stay in the same host and capture; restore only assigned metadata.
                from .reconstruction_inputs import ensure_inputs
                ensure_inputs(inp)
                receipt.setdefault("stage_retries", []).append({"stage": label, "attempt": retry+1, "reason": "transient_file_access"})
                atomic(output / "reconstruction.json", receipt)
                env["RCWM_RESUME"] = "1"
                if label == "compile" and (output / "world").exists():
                    (output / "world").rename(output / f"world.incomplete.{retry}")
                time.sleep(5)
                remaining = timeout - (time.monotonic() - started) + waiting_seconds(env.get("PYRUA_RECOVERY_DIR"), started_wall)
                if remaining <= 0: raise TimeoutError("Reconstruction active wall budget exhausted")

    try:
        stage(
            "reconstruct",
            ["bash", run / "runner_source/rcwm.sh", inp / "reference.png", name, depth, cycles],
        )
        stage(
            "check_program",
            [
                private / ".venv/bin/python",
                run / "runner_source/runner/check_part.py",
                run / "fractal/scene/part.json",
            ],
        )
        export = output / "export"
        stage(
            "export",
            [
                private / ".render-tools/node/bin/node",
                RESOURCES / "export_codeworld.mjs",
                private,
                name,
                export,
            ],
        )
        binding = read(export / "executed_binding.json")
        if multiview:
            stage("render_views", [private / ".render-tools/node/bin/node",
                                   inp / "render_calibrated.mjs", private, name,
                                   output / "calibrated_renders"])
        robot = binding.get("objects", {}).get("known_robot", {})
        if robot.get("user_data", {}).get("geometry_origin") != "public_urdf_measured_joints":
            raise ValueError("Export did not preserve the known robot identity")
        atomic(
            output / "robot_asset_check.json",
            verify_robot_asset(robot_glb, export / "objects/known_robot/mesh.glb"),
        )
        if cancelled is not None and cancelled():
            raise RuntimeError("Host episode ended before world compilation")
        stage(
            "compile",
            [
                os.sys.executable,
                "-m",
                "privilege.build",
                "--run",
                run,
                "--export",
                export,
                "--output",
                output / "world",
                "--runtime",
                private,
                "--frame",
                packet["frame"],
            ],
        )
        world = World(output / "world")
        world.verify_binding()
        receipt.update(status="BUILT", world="world")
        return world
    except BaseException as exc:
        receipt.update(status="FAILED", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        receipt["elapsed_s"] = time.monotonic() - started
        receipt["provider_wait_s"] = waiting_seconds(env.get("PYRUA_RECOVERY_DIR"), started_wall)
        atomic(output / "reconstruction.json", receipt)
