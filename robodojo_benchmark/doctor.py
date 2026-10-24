"""Read-only deployment checks. GPU rendering is checked by the probe command."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from .catalog import ROOT, layout_path, read, asset_revision


def doctor(runtime, jobs):
    errors, warnings, checks = [], [], []
    ours = any(j["strategy"] != "roboprobe_l3" for j in jobs)
    reconstruction_needed = any(j["strategy"] == "ours" for j in jobs)
    baseline = any(j["strategy"] == "roboprobe_l3" for j in jobs)
    keys = ["sim_python", "control_python"] + (["codex"] if ours else []) + (["reconstruction_python"] if reconstruction_needed else [])
    if baseline and runtime.get('roboprobe_sim_python'):keys.append('roboprobe_sim_python')
    executables = {}
    for key in keys:
        value = runtime.get(key, "")
        exists = bool(value and (shutil.which(value) or (Path(value).is_file() and os.access(value, os.X_OK))))
        checks.append({"name": key, "ok": exists, "path": value})
        executables[key] = exists
        if not exists:
            errors.append(f"Executable missing: {key}={value}")
    # Inspect the configured interpreters, not whichever Python launched doctor.
    # Importing open3d/FCL catches missing native libraries that find_spec misses.
    def imports(key, modules):
        if not executables.get(key):return
        env = {**os.environ, **runtime.get('env', {})}
        if runtime.get('motion_dependencies'):
            env['PYTHONPATH'] = runtime['motion_dependencies'] + os.pathsep + env.get('PYTHONPATH', '')
        script = ('import importlib,json\nfailures=[]\nversions={}\n'
                  'for name in '+repr(modules)+':\n'
                  ' try:\n'
                  '  module=importlib.import_module(name)\n'
                  '  versions[name]=str(getattr(module,"__version__","unknown"))\n'
                  ' except Exception as exc: failures.append(name+": "+type(exc).__name__+": "+str(exc))\n'
                  'print("DEPENDENCY_CHECK="+json.dumps({"failures":failures,"versions":versions}))\n')
        versions = {}
        try:
            p = subprocess.run([runtime[key], '-c', script], env=env, capture_output=True, text=True, timeout=60)
            marker = [s for s in p.stdout.splitlines() if s.startswith('DEPENDENCY_CHECK=')]
            detail = json.loads(marker[-1].split('=', 1)[1]) if marker else {'failures':['Interpreter import probe failed']}
            failures, versions = detail['failures'], detail.get('versions', {})
            if key in ('sim_python','roboprobe_sim_python') and versions.get('numpy','').startswith('2.'):
                failures.append('Isaac Sim 5.1 needs NumPy 1.x; install the pinned native overlay before starting Kit')
            if p.returncode and not failures:failures=['Interpreter exited '+str(p.returncode)]
        except (OSError, subprocess.TimeoutExpired) as exc:failures=[type(exc).__name__+': '+str(exc)]
        checks.append({'name':key+'_dependencies','ok':not failures,'modules':modules,'versions':versions})
        errors.extend(key+': '+failure for failure in failures)
    imports('control_python', ['numpy','scipy','trimesh','fcl','ruckig','yaml','cv2','PIL','imageio_ffmpeg'])
    imports('sim_python', ['numpy','scipy','trimesh','fcl','ruckig','omegaconf'])
    if baseline:imports('roboprobe_sim_python' if runtime.get('roboprobe_sim_python') else 'sim_python', ['openai','PIL','websockets'])
    if reconstruction_needed:imports('reconstruction_python', ['numpy','scipy','trimesh','open3d','torch','sam2','cv2','skimage','pyrender','manifold3d'])
    seen = set()
    for job in jobs:
        key = (job["native_task"], job["eval_seed"], job["layout_id"])
        if key in seen:
            continue
        seen.add(key)
        try:
            path = layout_path(runtime["robodojo_root"], job)
            # Git-LFS pointer files are not usable layouts.
            read(path)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{key}: {exc}")
    checks.append({"name": "official_layouts", "requested": len(seen), "errors": sum(1 for e in errors if e.startswith("("))})
    for repo in ("RoboDojo", "RoboProbe"):
        folder = Path(runtime.get("robodojo_root" if repo == "RoboDojo" else "roboprobe_root", ""))
        pinned = read(ROOT / "experiments/robodojo/upstreams.json")[repo]["commit"]
        from .provenance import upstream_state
        observed = upstream_state(folder, pinned)
        checks.append({'name': repo + '_source', **observed})
        errors.extend(repo + ': ' + error for error in observed['errors'])
        if observed['commit'] is None and not observed['distributed_digest_verified']:
            warnings.append(repo + ': exported tree has no verified distribution digest')
    rb = Path(runtime.get("robodojo_root", ""))
    asset_pin = read(ROOT / 'experiments/robodojo/platform_sources.json')['Assets']['commit']
    actual_assets = asset_revision(rb)
    checks.append({'name':'asset_revision','expected':asset_pin,'actual':actual_assets})
    if actual_assets and actual_assets != asset_pin:
        errors.append(f'Native Assets revision mismatch: expected {asset_pin}, got {actual_assets}')
    elif actual_assets is None:
        warnings.append('Assets is an exported folder without a readable revision; confirm it is the pinned official asset release before comparing results')
    urdf = rb / "Assets/Robots/x5/X5A.urdf"
    if not urdf.is_file() or urdf.read_text()[:30].startswith("version https://git-lfs"):
        errors.append("Public X5 URDF is missing or still a Git-LFS pointer")
    if any(j["strategy"] == "roboprobe_l3" for j in jobs):
        rp = Path(runtime.get("roboprobe_root", ""))
        if not (rb / "XPolicyLab/policy/RoboDojo_Agent_L3_Inspect_EEF/model.py").is_file():
            errors.append("RoboDojo/XPolicyLab must point to the supplied RoboProbe checkout")
        if not (rp.parent / "env_cfg/robot/_robot_info.json").is_file():
            errors.append("RoboProbe parent env_cfg/robot/_robot_info.json is missing")
        key = runtime.get("roboprobe_api_key_env", "OPENAI_API_KEY")
        if not (runtime.get("roboprobe_base_url") or os.environ.get("L3_INSPECT_BASE_URL")) or not os.environ.get(key):
            errors.append("RoboProbe planner endpoint/key is not configured (Codex subscription login is a separate route)")
    if any(j["strategy"] == "ours" for j in jobs):
        reconstruction = Path(runtime.get("reconstruction_runtime", ""))
        for name in (".venv/bin/python", ".render-tools/node/bin/node", ".render-tools/node_modules/playwright/index.mjs"):
            if not (reconstruction / name).exists():
                errors.append(f"Missing reconstruction runtime {name}; run scripts/setup.sh")
        if not (reconstruction / '.browsers').is_dir():
            errors.append('Reconstruction browser cache .browsers is missing; run scripts/setup.sh')
    if shutil.which("nvidia-smi"):
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True)
        checks.append({"name": "gpus", "value": p.stdout.strip()})
        if any(token in p.stdout for token in ("H100", "H200", "H800", "H20", "A100", "A800")):
            warnings.append("NVIDIA does not officially support GPUs without RT cores for Isaac Sim. RoboProbe ships A100/A800 workarounds; require a successful native RGB-D probe on this exact host.")
    else:
        warnings.append("No NVIDIA runtime detected on this host; simulator rendering has not been verified here")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks,
            "next_validation": "Run a native probe, then one full fresh episode for each strategy before a campaign"}
