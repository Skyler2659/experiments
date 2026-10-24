"""Run one selected layout with isolated artifacts and native scoring."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from .catalog import ROOT, read, layout_path, asset_revision
from .campaign import write


def runtime_config(path):
    path = Path(path).resolve()
    config = read(path)
    for key in ("robodojo_root", "roboprobe_root", "reconstruction_runtime", "codex_auth_home", "codex_provider_config", "motion_dependencies"):
        if config.get(key):
            value = Path(os.path.expandvars(config[key])).expanduser()
            config[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    for key in ("sim_python", "roboprobe_sim_python", "control_python", "reconstruction_python", "codex"):
        if config.get(key):
            value = os.path.expandvars(config[key])
            config[key] = str((path.parent / value).resolve()) if "/" in value and not Path(value).is_absolute() else value
    return config


def stop_process(process, grace=10):
    if process.poll() is not None:
        return
    # Model/render subprocesses may create their own sessions. Stopping only the
    # outer shell group would leave them running after a timed-out reconstruction.
    groups = {process.pid}
    snapshot = subprocess.run(['ps', '-eo', 'pid=,ppid=,pgid='], text=True, capture_output=True)
    rows = [tuple(map(int, line.split())) for line in snapshot.stdout.splitlines() if len(line.split()) == 3]
    descendants = {process.pid}
    while True:
        expanded = descendants | {pid for pid, parent, _ in rows if parent in descendants}
        if expanded == descendants:break
        descendants = expanded
    groups.update(group for pid, _, group in rows if pid in descendants and group in descendants)
    for group in groups:
        try:os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        for group in groups:
            try:os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:pass
        process.wait()


def run_process(argv, *, cwd, env, log, timeout):
    with Path(log).open("w") as output:
        process = subprocess.Popen([str(v) for v in argv], cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            stop_process(process)


def wait_file(path, process, timeout):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if process.poll() is not None:
            raise RuntimeError(f"Simulator exited {process.returncode} before {Path(path).name}; inspect server.log")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Waiting for {path} timed out")
        time.sleep(.25)


@contextlib.contextmanager
def fresh_codex_home(runtime, *, reconstruction=False):
    """Share authentication only. No copied chats, global instructions or skills."""
    base = Path(runtime.get("codex_auth_home", os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))))
    sessions = Path.home() / '.cache/codeworld-benchmark/sessions'
    sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="episode-", dir=sessions) as folder:
        path = Path(folder)
        shared_auth = (base / 'auth.json').resolve()
        local_auth = path / 'auth.json'
        if shared_auth.is_file():local_auth.symlink_to(shared_auth)
        content = ('[memories]\ngenerate_memories = false\nuse_memories = false\n'
                   '[features]\nmemories = false\nmulti_agent = false\nplugins = false\napps = false\n'
                   'chronicle = false\nskill_search = false\nskip_host_skill_discovery = true\n')
        if runtime.get("codex_provider_config"):
            # Optional provider-only TOML is supplied by the operator, outside the package.
            content = Path(runtime["codex_provider_config"]).read_text() + "\n" + content
        content = 'web_search = "disabled"\n' + content
        (path / "config.toml").write_text(content)
        if reconstruction:
            # A fixed, inspectable toolkit used by the existing reconstruction method.
            # No user skills or learned artifacts are imported between samples.
            source = ROOT / "vendor/RecursiveCWM/skills/worldgen-techniques"
            if source.is_dir():
                shutil.copytree(source, path / "skills/worldgen-techniques")
        # Some CLI releases replace auth.json atomically, breaking a symlink.
        # Preserve refreshed credentials at the operator's auth home promptly;
        # otherwise the next episode would replay an already-used refresh token.
        stop = threading.Event()
        previous = shared_auth.read_bytes() if shared_auth.is_file() else None
        def sync_auth():
            nonlocal previous
            if previous is None or not local_auth.is_file() or local_auth.is_symlink():return
            import fcntl
            with (base/'benchmark-auth.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                current, refreshed = shared_auth.read_bytes(), local_auth.read_bytes()
                if current == previous and refreshed != previous:
                    fd, tmp = tempfile.mkstemp(prefix='.benchmark-auth-',dir=shared_auth.parent)
                    try:
                        with os.fdopen(fd,'wb') as file:file.write(refreshed)
                        os.replace(tmp,shared_auth)
                    finally:
                        if os.path.exists(tmp):os.unlink(tmp)
                previous=shared_auth.read_bytes()
                local_auth.unlink();local_auth.symlink_to(shared_auth)
        def watch():
            while not stop.wait(.5):
                try:sync_auth()
                except OSError:pass
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        try:yield str(path)
        finally:
            stop.set();watcher.join();sync_auth()


def prepare_files(episode, job, runtime=None):
    episode.mkdir()
    for directory in ("task", "work", "tools", "state", "preparation"):
        (episode / directory).mkdir()
    for package in ("codeworld_robot", "codeworld_workspace", "robodojo_benchmark"):
        shutil.copytree(ROOT / package, episode / "tools" / package,
                        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"))
    scanner = episode / "tools/scripts/robodojo42"
    scanner.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/robodojo42/render_ready.py", scanner / "render_ready.py")
    from codeworld_robot.backend_contract import CONTRACTS, execution_instructions
    from .instructions import NO_WORLD, NO_WORLD_BACKEND, GT_LIVE, COMMON
    from .task_brief import load_task_brief
    brief = load_task_brief((runtime or {}).get("roboprobe_root", ROOT / "vendor/RoboProbe"), job["native_task"])
    write(episode / "task/public_task_brief.json", brief)
    (episode / "task/PUBLIC_TASK.md").write_text(brief["text"])
    representation = job["representation"]
    text = NO_WORLD if representation == "none" else (ROOT / "codeworld_workspace/templates/AGENTS.md").read_text()
    if representation == "ground_truth" and job["gt_mode"] == "live_state":
        text += GT_LIVE
    (episode / "AGENTS.md").write_text(text + COMMON)
    (episode / "task/ROBOT_BACKEND.md").write_text(NO_WORLD_BACKEND if representation == 'none' else CONTRACTS["robodojo"])
    (episode / "task/EXECUTION.md").write_text(NO_WORLD if representation == "none" else
        execution_instructions("robodojo", (ROOT / "codeworld_robot/README.md").read_text()))
    flow = NO_WORLD if representation == 'none' else (ROOT / "codeworld_robot/WORLD_FLOW.md").read_text()
    if representation == 'ground_truth' and job['gt_mode'] == 'live_state':
        flow = GT_LIVE + '\nThe flow below uses oracle observations instead of RGB-D pose fitting in this arm.\n' + flow
    (episode / 'task/WORLD_FLOW.md').write_text(flow)
    write(episode / "task/task.json", {"goal": "Awaiting official public instruction", "status": "ACTIVE",
          "robot_backend": "robodojo", "robot_interface": "codeworld_robot.Robot",
          "constraints": ["Use the official action budget. Return arms as required by the public task."]})
    write(episode / "task/executor.json", {"status": "UNASSIGNED", "role": "codex_cli"})
    write(episode / "task/world_flow.json", {"required": representation != "none", "version": 1,
          "mode": "representation_ablation" if representation == "none" else "independent_strict"})
    write(episode / "task/representation.json", {"representation": representation, "gt_mode": job["gt_mode"]})


def native_result(path, layout_id):
    result = read(path)
    details = list(result.get("details", {}).values())
    if len(details) != 1 or details[0].get("layout_id") != layout_id:
        raise ValueError("Native result does not contain exactly the requested layout")
    d = details[0]
    if type(d.get("success")) is not bool or type(d.get("score")) not in (int, float) or not 0 <= d["score"] <= 1:
        raise ValueError("Malformed native success/score")
    return {"official_success": d["success"], "score_0_100": float(d["score"]) * 100,
            "native_result": str(Path(path).resolve()), "layout_id": layout_id}


def run_ours(attempt, runtime, spec, job, env, *, probe=False):
    from .reconstruct import reconstruct, ReconstructionFailure
    episode, evaluator = attempt / "episode", attempt / "evaluator"
    evaluator.mkdir()
    prepare_files(episode, job, runtime)
    from .reuse import validate_reuse, restore_reconstruction
    reuse = validate_reuse(runtime, job, attempt)
    settings = {"backend": "robodojo", "benchmark_protocol": "robodojo-v1", "official_evaluation": True,
                "robodojo_root": runtime["robodojo_root"], "env_config": "arx_x5.yml", "task_name": job["native_task"],
                "seed": job["eval_seed"], "layout_id": job["layout_id"], "world_origin_S_m": [0., -.45, .765],
                "representation": job["representation"], "gt_mode": job["gt_mode"], "run_id": env["ROBODOJO_RUN_ID"],
                "evaluator_directory": str(evaluator), "full_arm_collision_check": True, "motion_profile": "ruckig",
                "preparation_timeout_s": spec["reconstruction"]["wall_timeout_s"] + 300,
                "probe_native_only": probe, "max_settle_actions": 4}
    if reuse and job['representation'] == 'ground_truth':
        settings['reuse_gt_episode'] = str(reuse / 'episode')
    settings['renderer_gpu'] = int(runtime.get('renderer_gpu', env['CUDA_VISIBLE_DEVICES']))
    if runtime.get("motion_dependencies"):
        settings["motion_dependencies"] = runtime["motion_dependencies"]
    write(evaluator / "server_config.json", settings)
    server_env = {**env, "PYTHONPATH": str(episode / "tools") + os.pathsep + env.get("PYTHONPATH", "")}
    method_failure = None
    with (attempt / "server.log").open("w") as log:
        server = subprocess.Popen([runtime["sim_python"], "-m", "codeworld_robot.robodojo_server", "--workspace", str(episode),
                   "--config", str(evaluator / "server_config.json")], cwd=episode, env=server_env,
                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(attempt / "process.json", {"server_pid": server.pid, "started_at": time.time()})
        try:
            wait_file(episode / "preparation/ready.json", server, runtime.get("startup_timeout_s", 900))
            if reuse and job["representation"] == "reconstructed":
                restore_reconstruction(reuse, episode, job, env["ROBODOJO_RUN_ID"])
            elif job["representation"] == "reconstructed":
                if probe:
                    raise ValueError("Use a full reconstruction check for Ours, not a native-only probe")
                try:
                    with fresh_codex_home(runtime, reconstruction=True) as reconstruction_home:
                        reconstruct(episode, {**runtime, "reconstruction_codex_home": reconstruction_home,
                                              "env": {**env, "EGL_DEVICE_ID": env['CUDA_VISIBLE_DEVICES'], "PYOPENGL_PLATFORM": "egl"}},
                                    spec, job, run_process)
                except ReconstructionFailure as exc:
                    method_failure = str(exc)
                    raise
            write(episode / "task/world_ready.json", {"status": "READY"})
            wait_file(episode / "robot_io/ready.json", server, 180)
            if probe:
                write(episode / "task/stop_server.json", {"reason": "native initialization probe; no agent or actions"})
            else:
                with fresh_codex_home(runtime) as codex_home:
                    transport = {"episode": str(episode), "python": runtime["control_python"], "benchmark_protocol": "robodojo-v1",
                                 "representation": job["representation"], "gt_mode": job["gt_mode"], "codex_home": codex_home,
                                 "agent_timeout_s": spec["agent"]["wall_timeout_s"]}
                    write(attempt / "transport.json", transport)
                    command = [runtime["control_python"], "-m", "codeworld_robot.agent_cli", "run", "--config", str(attempt / "transport.json"),
                               "--run-dir", str(attempt / "agent"), "--model", spec["agent"]["model"],
                               "--reasoning-effort", spec["agent"]["reasoning_effort"], "--codex", runtime.get("codex", "codex")]
                    code = run_process(command, cwd=ROOT, env=env, log=attempt / "agent_launcher.log",
                                       timeout=spec["agent"]["wall_timeout_s"] + 120)
                    write(episode / "task/stop_server.json", {"reason": "CLI lifecycle ended", "returncode": code})
            server.wait(timeout=120)
            result = native_result(evaluator / "_result.json", job["layout_id"])
            actions = read(evaluator / "official_result.json")
            result.update(status="SCORED", official_actions=actions["official_actions"],
                          official_action_limit=actions["official_action_limit"], probe_only=probe)
            if not probe:
                life = read(attempt / "agent/lifecycle.json")
                result["agent_lifecycle"] = life
                if life.get("termination_reason") == "wall_timeout":
                    result["termination"] = "agent_wall_budget"
                elif life['status'] == 'STOPPED':
                    result['status'] = 'INTERRUPTED'
                elif life["status"] == "FAILED" or life.get("exit_code", 0) not in (0, None):
                    result["status"] = "INFRA_ERROR"
            return result
        except ReconstructionFailure:
            return {"status": "METHOD_FAILURE", "official_success": None, "score_0_100": None,
                    "end_to_end_success": False, "failure_stage": "reconstruction", "error": method_failure}
        finally:
            write(episode / "task/stop_server.json", {"reason": "worker teardown"})
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop_process(server)


def run_roboprobe(attempt, runtime, spec, job, env, *, probe=False):
    cfg = spec["roboprobe"]
    rp = Path(runtime["roboprobe_root"])
    base_url = runtime.get("roboprobe_base_url") or os.environ.get("L3_INSPECT_BASE_URL")
    key_env = runtime.get("roboprobe_api_key_env", "OPENAI_API_KEY")
    if not probe and (not base_url or not os.environ.get(key_env)):
        raise RuntimeError("RoboProbe requires roboprobe_base_url and its API-key environment variable; Codex login is separate")
    evaluator = attempt / "evaluator"; evaluator.mkdir()
    episode = attempt / "episode"; episode.mkdir()
    # Native task loading/reset/scoring plus the released unmodified EEF policy.
    # This policy's deploy.py discards its no-op WebSocket model_client.
    # A shell left over from a probe must not silently switch the released arm
    # into an embodiment diagnostic, different prompt, horizon or action budget.
    baseline_env = {**{k:v for k,v in env.items() if not k.startswith('L3_INSPECT_') or k == key_env},
        "ROBODOJO_ROOT": runtime["robodojo_root"],
        "L3_INSPECT_USE_RECIPE": "1", "L3_INSPECT_PLANNER": "astra", "L3_INSPECT_MODEL": cfg["model"], "L3_INSPECT_REASONING_EFFORT": cfg["reasoning_effort"],
        "L3_INSPECT_MAX_LLM_CALLS": str(cfg["max_llm_calls"]), "L3_INSPECT_IMAGE_HORIZON": str(cfg["image_horizon"]),
        "L3_INSPECT_KEEP_ALL_IMAGES": "0", "L3_INSPECT_DEPTH": "off", "L3_INSPECT_ACTION_TYPE": "joint",
        "L3_INSPECT_BASE_URL": base_url or '', "L3_INSPECT_API_KEY_ENV": key_env,
        "L3_INSPECT_TRACE_DIR": str(attempt / "trace"), "L3_INSPECT_HARD_TIMEOUT_S": "90"}
    baseline_env.pop('L3_INSPECT_ENV_STEP_LIMIT', None)
    baseline_env.pop('ROBODOJO_ENABLE_METRIC_DEPTH', None)
    settings = {"benchmark_protocol":"robodojo-v1", "strategy":"roboprobe_l3", "official_evaluation":True,
                "robodojo_root":runtime['robodojo_root'], "env_config":"arx_x5.yml", "task_name":job['native_task'],
                "seed":job['eval_seed'], "layout_id":job['layout_id'], "evaluator_directory":str(evaluator),
                "probe_native_only":probe,
                "renderer_gpu":int(runtime.get('renderer_gpu',env['CUDA_VISIBLE_DEVICES']))}
    write(evaluator / 'server_config.json', settings)
    command = [runtime.get('roboprobe_sim_python', runtime['sim_python']), '-m', 'codeworld_robot.robodojo_server',
               '--workspace', str(episode), '--config', str(evaluator/'server_config.json')]
    try:
        code = run_process(command, cwd=ROOT, env=baseline_env, log=attempt/'roboprobe.log',
                           timeout=runtime.get('startup_timeout_s',900) if probe else spec['agent']['wall_timeout_s'])
    except subprocess.TimeoutExpired:
        if probe:raise
        # A declared policy wall-budget exhaustion is a method outcome. There
        # is no finalized native score, so do not manufacture a zero score or
        # permit retries as if this were a credential/dependency error.
        return {'status':'METHOD_FAILURE','official_success':None,'score_0_100':None,
                'end_to_end_success':False,'failure_stage':'agent','termination':'agent_wall_budget',
                'error':'RoboProbe exceeded the declared wall budget before native finalization'}
    if code:
        raise RuntimeError(f"RoboProbe exited {code}; inspect roboprobe.log and the original planner trace")
    result = native_result(evaluator/'_result.json', job['layout_id'])
    status = read(evaluator/'official_result.json')
    return {"status": "SCORED", **result, "official_actions":status['official_actions'],
            "official_action_limit":status['official_action_limit'], "policy":"RoboDojo_Agent_L3_Inspect_EEF",
            "probe_only":probe}


def execute_job(campaign, job, runtime, gpu, *, probe=False):
    attempt = campaign.root / "runs" / job["job_id"] / f"attempt_{job['attempt']:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "job.json", job)
    run_id = campaign.manifest["name"] + "-" + job["job_id"] + "-" + uuid.uuid4().hex[:10]
    env = {**os.environ, **runtime.get("env", {}), "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT),
           "ROBODOJO_RUN_ID": run_id, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    started = time.time()
    result = None
    try:
        actual_assets = asset_revision(runtime['robodojo_root'])
        expected_assets = campaign.manifest.get('platform_sources', {}).get('Assets', {}).get('commit')
        write(attempt / 'source_versions.json', {'upstreams':campaign.manifest['upstreams'],
              'platform_sources':campaign.manifest.get('platform_sources', {}),
              'observed_assets_revision':actual_assets})
        if expected_assets and actual_assets and expected_assets != actual_assets:
            raise RuntimeError(f'Native Assets revision differs from the campaign: {actual_assets} != {expected_assets}')
        layout = layout_path(runtime["robodojo_root"], job)
        # Private input record stays outside the agent's working directory.
        shutil.copy2(layout, attempt / "native_layout.json")
        runner = run_roboprobe if job["strategy"] == "roboprobe_l3" else run_ours
        result = runner(attempt, runtime, campaign.manifest["spec"], job, env, probe=probe)
    except (KeyboardInterrupt, InterruptedError) as exc:
        result = {"status": "INTERRUPTED", "error": str(exc)}
    except Exception as exc:
        error_file = attempt / "evaluator/server_error.json"
        server_error = read(error_file) if error_file.exists() else {}
        status = "UNSTABLE" if server_error.get("exception_type") in ("UnStableError", "PhysXBrokenError") else "INFRA_ERROR"
        result = {"status": status, "error": str(exc), "server_error": server_error}
    finally:
        if result is not None:
            result.update(run_id=run_id, artifact_dir=str(attempt), elapsed_s=time.time() - started,
                          strategy=job["strategy"], task=job["task"], sample=job["sample"], layout_id=job["layout_id"],
                          native_task=job["native_task"], eval_seed=job["eval_seed"], probe_only=probe)
            write(attempt / "result.json", result)
    return result


def finish_job(campaign, job, result):
    """Commit the experiment outcome before optional, independently retryable media."""
    campaign.finish(job, result)
    if not result.get('probe_only'):
        from .artifacts import export
        try:
            export(result['artifact_dir'])
        except Exception as exc:
            # The exporter persists its own failure manifest. A video failure
            # must not change the native score or make a scored episode retryable.
            print(f"Artifact export failed: {exc}; see {result['artifact_dir']}/artifacts/manifest.json", flush=True)
    # KeyboardInterrupt intentionally propagates after the score is committed.
