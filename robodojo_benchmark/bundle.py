"""Source-only handoff: no credentials, historical episodes or reconstructed assets."""
import io
import json
from pathlib import Path
import tarfile
from .catalog import ROOT, read

DIRECTORIES = ("robodojo_benchmark", "codeworld_robot", "codeworld_workspace", "reconstruction_assist", "agentic_mesh_world",
               "vendor/RecursiveCWM", "vendor/RoboProbe", "vendor/RoboDojo")
FILES = ("upstream.json", "prompts/robodojo42-adaptive.md", "scripts/setup.sh",
         "scripts/install_linux_libraries.sh", "scripts/rcwm_world.py", "scripts/export_codeworld.mjs",
         "scripts/jdcloud_responses_proxy.py",
         "scripts/render_codeworld.mjs", "scripts/codeworld_source_files.mjs",
         "scripts/robodojo42/inventory.py", "scripts/robodojo42/prepare.py", "scripts/robodojo42/render_ready.py")
EXPERIMENT_FILES = ("suite.json", "upstreams.json", "platform_sources.json", "requirements.txt", "requirements-reconstruction.txt",
                    "runtime.example.json", "setup_links.py", "install.sh", "install.py", "run_one.sh", "configure_api.py", "README.zh-CN.md", "PROTOCOL.md", "VALIDATION.md", "validation-evidence.json")


def bundle(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    candidates = [ROOT / f for f in FILES]
    required = candidates + [ROOT / 'experiments/robodojo' / name for name in EXPERIMENT_FILES]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:raise FileNotFoundError('Missing handoff sources: '+', '.join(missing))
    for directory in DIRECTORIES:
        candidates.extend((ROOT / directory).rglob("*"))
    candidates.extend(ROOT / "experiments/robodojo" / name for name in EXPERIMENT_FILES)
    excluded = {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules", "eval_result", "Assets", "third_party"}
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(set(candidates)):
            rel = path.relative_to(ROOT)
            if (not path.is_file() or path.is_symlink() or excluded.intersection(rel.parts)
                or path.suffix in ('.pyc','.pem','.key') or path.name.startswith('.env')
                or path.name in ('auth.json','runtime.local.json')):
                continue
            archive.add(path, arcname=str(Path("CodeWorldv1") / rel), recursive=False)
        # Publish the concise operator guide at the archive root; the workspace
        # supervisor AGENTS.md is not an instruction for the receiving operator.
        archive.add(ROOT / "experiments/robodojo/README.zh-CN.md",
                    arcname="CodeWorldv1/README.md", recursive=False)
        for repo, data in read(ROOT / "experiments/robodojo/upstreams.json").items():
            payload = json.dumps(data, indent=2).encode()
            info = tarfile.TarInfo(f"CodeWorldv1/vendor/{repo}/SOURCE_REVISION.json"); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return str(output)
