"""Portable invocation of the full existing adaptive RCWM method."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from .catalog import ROOT, read
from .campaign import write


class ReconstructionFailure(RuntimeError):
    pass


def infrastructure_failure(log_path):
    """Recognize setup/transport failures; don't score them as geometry failures."""
    text = Path(log_path).read_text(errors='replace')
    markers = ('ModelTransportFailure:', 'ModuleNotFoundError:', 'ImportError:',
               'cannot open shared object file', 'No module named',
               'HTTP Error 401', 'HTTP Error 403', 'HTTP Error 429', 'HTTP Error 502',
               'HTTP Error 503', 'URLError:', 'refresh_token_reused',
               'Missing runtime dependency:', 'Missing command:', 'No space left on device',
               "browserType.launch: Executable doesn't exist", 'Host system is missing dependencies')
    return next((marker for marker in markers if marker in text), None)


def reconstruction_infrastructure_failure(case, stage):
    marker = infrastructure_failure(case / (stage + '.log'))
    if marker or stage != 'reconstruction':return marker
    for path in (case / 'runtime/runs').glob('*/fractal/*/codex-run.log'):
        text = path.read_text(errors='replace')
        for token in ("bwrap: Can't find source path", 'Failed to create user namespace',
                      'stream disconnected before completion', 'refresh_token_reused',
                      'Your access token could not be refreshed'):
            if token in text:return token
    return None


def reconstruct(episode, runtime, spec, job, process_runner):
    episode = Path(episode)
    case = episode / "preparation"
    cfg = spec["reconstruction"]
    write(case / "spec.json", {"id": job["job_id"], "task": job["native_task"],
                              "eval_seed": job["eval_seed"], "layout_id": job["layout_id"]})
    shared = Path(runtime["reconstruction_runtime"]).resolve()
    private_runtime = case / "runtime"
    private_runtime.mkdir()
    (private_runtime / 'AGENTS.md').write_text(
        'You are the reconstruction worker for this fresh episode. Build the scene from the supplied FIT RGB-D evidence and per-node task/brief files. '
        'The parent episode robot-executor instructions apply only to the later action agent. Do not control the robot or call its interface. '
        'Do not inspect native simulator assets, layouts, evaluator files, private poses, historical episodes or other reconstruction runs. '
        'Use the fixed reconstruction toolkit and current measured input. Preserve camera calibration and evidence provenance. '
        'Write only this run and its assigned node; follow its refinement and correction protocol.\n')
    for name in (".render-tools", ".system-libs", ".venv", ".codex-cli", ".browsers", "platform.json"):
        if (shared / name).exists():
            (private_runtime / name).symlink_to(shared / name, target_is_directory=(shared / name).is_dir())
    (private_runtime / "bin").mkdir()
    (private_runtime / "bin/codex").symlink_to(Path(shutil.which(runtime.get("codex", "codex")) or runtime["codex"]).resolve())
    env = {**os.environ, **runtime.get("env", {}), "PYTHONPATH": str(ROOT), "RCWM_ROOT": str(private_runtime),
           "RCWM_CODEX": runtime.get("codex", "codex"), "RCWM_RECON_MODEL": cfg["model"],
           "RCWM_RECON_REASONING": cfg["reasoning_effort"], "RCWM_VISION_TRANSPORT": runtime.get("vision_transport", "codex"),
           "RCWM_CLEAN_CODEX_HOME": "0", "RCWM_ISOLATED_HOME": "1", "RCWM_GPU_LOCK": str(case / "sam.lock"),
           "RCWM_SANDBOX": runtime.get('reconstruction_sandbox','workspace-write')}
    if env['RCWM_SANDBOX'] not in ('workspace-write','danger-full-access'):
        raise ValueError('reconstruction_sandbox must be workspace-write or danger-full-access')
    # Export is a sibling process of the recursive runner, so it must receive
    # the same browser cache and shared libraries instead of Playwright defaults.
    env['PLAYWRIGHT_BROWSERS_PATH'] = str((private_runtime / '.browsers').resolve())
    libraries = private_runtime / '.system-libs/usr/lib/x86_64-linux-gnu'
    if libraries.is_dir():
        env['LD_LIBRARY_PATH'] = str(libraries) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    if runtime.get("reconstruction_codex_home"):
        env["CODEX_HOME"] = runtime["reconstruction_codex_home"]
    if runtime.get("reconstruction_base_url"):
        env["RCWM_RESPONSES_URL"] = runtime["reconstruction_base_url"].rstrip("/") + "/responses"
        env["RCWM_RECON_API_KEY_ENV"] = runtime["reconstruction_api_key_env"]
    python = runtime["reconstruction_python"]
    deadline = time.monotonic() + cfg["wall_timeout_s"]

    def call(stage, argv, timeout=None):
        write(case / "stage.json", {"stage": stage, "started": time.time()})
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReconstructionFailure("Reconstruction wall budget exhausted")
        try:
            code = process_runner(argv, cwd=ROOT, env=env, log=case / (stage + ".log"),
                                  timeout=min(remaining, timeout or remaining))
        except subprocess.TimeoutExpired as exc:
            marker = reconstruction_infrastructure_failure(case, stage)
            if marker:raise RuntimeError(f'{stage} infrastructure failure ({marker}); inspect preparation logs') from exc
            raise ReconstructionFailure(f"{stage} exceeded reconstruction wall budget") from exc
        if code:
            marker = reconstruction_infrastructure_failure(case, stage)
            if marker:
                raise RuntimeError(f'{stage} infrastructure failure ({marker}); inspect preparation/{stage}.log')
            raise ReconstructionFailure(f"{stage} exited {code}; inspect preparation/{stage}.log")

    call("inventory", [python, str(ROOT / "scripts/robodojo42/inventory.py"), str(case)], 1000)
    call("evidence", [python, str(ROOT / "scripts/robodojo42/prepare.py"), str(case)], 1800)
    transport = {"transport": "local", "remote_work": str(case), "remote_runtime": str(private_runtime), "remote_source": str(ROOT),
                 "evidence_python": python,
                 "max_reviews": 4, "refinement": {"enabled": True, "strategy": "adaptive", "patience": 2,
                 "min_relative_improvement": .03, "min_absolute_improvement": .005},
                 "agent_correction": {"enabled": True, "max_rounds": 1, "extra_reviews": 2, "extra_cycles": 4, "timeout_seconds": 660,
                                      "evidence_python": python}}
    write(case / "transport.json", transport)
    scene = "episode_" + job["job_id"]
    # Runner scene IDs allow letters, numbers, underscore and hyphen.
    inp = case / "input"
    config = {"scene": scene, "reference": str(inp / "reference.png"), "model": cfg["model"],
              "reasoning": cfg["reasoning_effort"], "max_depth": cfg["max_depth"], "max_cycles": cfg["max_cycles"],
              "solver_prompt": "prompts/robodojo42-adaptive.md", "known_robot": {"asset": str(inp / "known_robot.glb"),
              "binding": str(inp / "robot_binding.json")}, "assistance": {"transport_config": str(case / "transport.json"),
              "measurement": str(inp / "measurement/evidence.json")}}
    write(case / "reconstruction.json", config)
    call("reconstruction", [python, str(ROOT / "scripts/rcwm_world.py"), "run", "--config", str(case / "reconstruction.json")])
    node = str(private_runtime / ".render-tools/node/bin/node")
    env.update(RCWM_EXPORT_OBJECTS="1", RCWM_EXPORT_CANONICAL_VIEW="1")
    call("export", [node, str(ROOT / "scripts/export_codeworld.mjs"), str(private_runtime), scene, str(case / "export")])
    call("world", [python, "-m", "codeworld_workspace.build", "--run", str(private_runtime / "runs" / scene),
                   "--export", str(case / "export"), "--output", str(case / "workspace"), "--runtime", str(private_runtime)])
    world = case / "workspace"
    # Copy only world assets. Keep the current official episode's goal, I/O and ownership.
    for name in ("world.json", "geometry", "objects", "source", "state", "scenarios", "observations"):
        src = world / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, episode / name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, episode / name)
    # No previous episode is copied. The reconstructed prior still needs normal
    # observation-backed updates before completion, as in the existing method.
    state = read(episode / "state/current.json")
    state["source_episode"] = env.get('ROBODOJO_RUN_ID', job['job_id'])
    state['source_sample'] = job['job_id']
    write(episode / "state/current.json", state)
    write(case / "reconstruction_receipt.json", {"status": "BUILT", "model": cfg["model"], "reasoning_effort": cfg["reasoning_effort"],
          "reconstruction_sandbox":env['RCWM_SANDBOX'],
          "historical_assets_reused": False, "same_reset_as_execution": True,
          "source_program_binding_verified": (read(world / 'build_report.json')['status'] == 'PASS'
                                               and (episode / 'source/executed_binding.json').is_file()),
          "build_report": "preparation/workspace/build_report.json"})
