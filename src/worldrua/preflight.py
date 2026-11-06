"""Deployment checks in the actual Agent, simulator and rendering environments.

No model request, model weight load, simulator startup or robot action is issued.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HOST_PROBE = r"""
import importlib, json, sys
errors=[]
for name in ('numpy','scipy','trimesh','PIL','fcl','rtree','mcp','uvicorn','httpx',
             'msgpack_numpy','websockets.asyncio.server','yaml','cv2','h5py','imageio_ffmpeg',
             'einops','pycocotools','psutil','torchvision',
             'sam3.model.sam3_image_processor','sam3.model_builder'):
    try: importlib.import_module(name)
    except Exception as exc: errors.append(name+': '+str(exc))
try:
    import torch
    if not torch.cuda.is_available(): errors.append('torch: CUDA is unavailable')
except Exception as exc: errors.append('torch: '+str(exc))
try:
    from pyrualean.hosts.xpolicylab_robodojo import _import_server
    from pathlib import Path
    xpl=Path(sys.argv[1]); _import_server(xpl.parent)
    sys.path.insert(0,str(xpl))
    importlib.import_module('client_server.ws.model_server')
except Exception as exc: errors.append('XPolicyLab policy server: '+str(exc))
print(json.dumps(errors))
sys.exit(bool(errors))
"""
SIM_PROBE = r"""
import importlib, importlib.util, json, sys
errors=[]
for name in ('yaml','numpy','scipy','cv2','msgpack_numpy','transforms3d'):
    try: importlib.import_module(name)
    except Exception as exc: errors.append(name+': '+str(exc))
for name in ('isaacsim','isaaclab'):
    if importlib.util.find_spec(name) is None: errors.append('Missing '+name)
print(json.dumps(errors))
sys.exit(bool(errors))
"""


def probe(label, command, env, failures, timeout=45):
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        failures.append(f"{label}: {exc}")
        return None
    if result.returncode:
        detail = (result.stderr + "\n" + result.stdout).strip()[-1600:]
        failures.append(f"{label}: {detail or 'exit ' + str(result.returncode)}")
        return None
    return result.stdout


def check_layouts(cfg, failures):
    import yaml

    root = Path(cfg["robodojo_root"])
    env_config = root / "env_cfg/arx_x5.yml"
    if not env_config.is_file():
        return
    try:
        suite = (yaml.safe_load(env_config.read_text()) or {}).get("config_name") or "arx_x5"
        layouts = root / "Assets/Eval_Layout/RoboDojo" / suite / str(cfg["eval_seed"])
        for job in cfg["jobs"]:
            path = layouts / f"{job['task']}_{job['sample']}.json"
            if not path.is_file():
                failures.append("Missing task layout: " + str(path))
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        failures.append("Invalid simulator configuration: " + str(exc))


def check(cfg):
    failures = []
    env = dict(os.environ, **cfg["environment"])
    root, runtime = Path(cfg["robodojo_root"]), Path(cfg["runtime"])
    if sys.platform != "linux":
        failures.append("Robot scheduling requires Linux; macOS supports offline development only")
    paths = [
        Path(cfg["python"]),
        Path(cfg["codex"]),
        Path(cfg["python"]).parent / "worldrua-codex",
        root / "src/eval_client/main.py",
        root / "scripts/eval_policy.sh",
        root / "env_cfg/arx_x5.yml",
        root / "Assets/Robots/x5/X5A.urdf",
        Path(cfg["xpolicylab_root"]) / "client_server/ws/model_server.py",
        Path(cfg["auth_home"]) / "auth.json",
    ]
    paths += [
        runtime / p
        for p in (
            ".venv/bin/python",
            ".render-tools/node/bin/node",
            ".render-tools/node_modules/three/build/three.module.js",
            ".render-tools/node_modules/playwright/index.mjs",
        )
    ]
    paths += [
        root / "task/RoboDojo/config" / (t + ".yml")
        for t in dict.fromkeys(j["task"] for j in cfg["jobs"])
    ]
    for path in paths:
        if not path.is_file():
            failures.append("Missing: " + str(path))
    for part in ("env", "src", "task", "utils", "scripts", "env_cfg"):
        if not (root / part).is_dir():
            failures.append("Missing RoboDojo source directory: " + str(root / part))
    checkpoint = env.get("SAM3_CHECKPOINT_PATH")
    if not checkpoint:
        failures.append("SAM3_CHECKPOINT_PATH is not set; download the checkpoint (docs/SETUP.md)")
    elif (
        not Path(checkpoint).expanduser().is_file()
        or Path(checkpoint).expanduser().stat().st_size == 0
    ):
        failures.append("SAM3 checkpoint is missing or empty: " + checkpoint)
    check_layouts(cfg, failures)
    if Path(cfg["python"]).is_file():
        probe(
            "Agent dependencies",
            [cfg["python"], "-c", HOST_PROBE, cfg["xpolicylab_root"]],
            env,
            failures,
        )
    simulator_python = shutil.which("python3", path=env.get("PATH", ""))
    if simulator_python:
        probe(
            "Simulator Python (" + simulator_python + ")",
            [simulator_python, "-c", SIM_PROBE],
            env,
            failures,
        )
    else:
        failures.append("Simulator python3 is missing from PATH; set ROBODOJO_PYTHON_BIN")
    if Path(cfg["codex"]).is_file():
        probe(
            "Codex CLI compatibility",
            [cfg["codex"], "exec", "--ignore-user-config", "--help"],
            env,
            failures,
        )
    smi = shutil.which("nvidia-smi", path=env.get("PATH", ""))
    if smi:
        output = probe(
            "NVIDIA driver",
            [smi, "--query-gpu=index", "--format=csv,noheader,nounits"],
            env,
            failures,
        )
        if output is not None:
            indices = {int(x.strip()) for x in output.splitlines() if x.strip().isdigit()}
            for gpu in cfg["gpus"]:
                if gpu not in indices:
                    failures.append(f"Configured GPU {gpu} is not available")
    else:
        failures.append("nvidia-smi is not available on PATH")
    node = runtime / ".render-tools/node/bin/node"
    if node.is_file() and (runtime / ".render-tools/node_modules/playwright/index.mjs").is_file():
        render_env = dict(env, PLAYWRIGHT_BROWSERS_PATH=str(runtime / ".browsers"))
        from privilege.reconstruction import RESOURCES

        probe(
            "Chromium/WebGL rendering",
            [str(node), str(RESOURCES / "recursive_cwm/setup/smoke_test.mjs"), str(runtime)],
            render_env,
            failures,
            timeout=90,
        )
    return failures
