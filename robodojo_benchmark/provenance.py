"""Identify and archive the actual software used by an attempt."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

SOURCE_DIRS = (
    "robodojo_benchmark",
    "codeworld_robot",
    "codeworld_workspace",
    "reconstruction_assist",
    "agentic_mesh_world",
    "scripts",
    "prompts",
    "experiments/robodojo",
    ".github",
    "vendor/RecursiveCWM",
    "vendor/RoboDojo",
    "vendor/RoboProbe",
)
EXCLUDE = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
    "Assets",
    "third_party",
    "runtime",
    "eval_result",
    ".browsers",
    ".render-tools",
    ".cache",
    ".ruff_cache",
    ".secrets",
    ".sweep",
    ".worktrees",
    "checkpoints",
    "runs",
    "logs",
    "Log",
    "exports",
    "train_out",
    "build",
    "dist",
}


def source_files(root, directories=None):
    root = Path(root).resolve()
    candidates = []
    for name in directories or SOURCE_DIRS:
        path = root / name
        if path.is_dir():
            # Prune generated trees instead of walking large installed assets.
            import os

            for folder, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if d not in EXCLUDE
                    and not d.startswith(".venv")
                    and not d.endswith(".egg-info")
                    and not Path(folder, d).is_symlink()
                )
                candidates.extend(Path(folder, f) for f in files)
    if directories is None:
        candidates.extend(
            root / n
            for n in ("README.md", "ARCHITECTURE.md", ".gitignore", "upstream.json")
        )
    return sorted(
        p
        for p in candidates
        if p.is_file()
        and not p.is_symlink()
        and not p.name.startswith(".env")
        and p.suffix not in (".pyc", ".pem", ".key")
        and p.name
        not in (
            "auth.json",
            "runtime.local.json",
            "provider.local.toml",
            "SOURCE_REVISION.json",
        )
    )


def tree_digest(root, directories=None):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in source_files(root, directories):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def git_state(root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], text=True, capture_output=True
        )

    top = git("rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        return {"commit": None, "dirty": None}
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": head.stdout.strip() if not head.returncode else None,
        "dirty": bool(status.stdout) if not status.returncode else None,
    }


def upstream_state(path, expected):
    path = Path(path).resolve()
    state = git_state(path)
    marker = path / "SOURCE_REVISION.json"
    declared = json.loads(marker.read_text()) if marker.is_file() else {}
    # A checksum binds the shipped tree; it is not independent proof of upstream authenticity.
    digest = tree_digest(path, (".",))
    actual = state["commit"] or declared.get("commit")
    errors = []
    if actual != expected:
        errors.append(f"revision mismatch: expected {expected}, got {actual}")
    if state["dirty"]:
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            capture_output=True,
        )

        def installation_only(line):
            name = line[3:]
            if line[:2] != "??":
                return False
            item = path / name
            if name == "XPolicyLab" and item.is_symlink():
                return (
                    item.resolve() / "policy/RoboDojo_Agent_L3_Inspect_EEF/model.py"
                ).is_file()
            if name == "client_server" and item.is_symlink():
                return item.resolve() == (path / "utils/client_server").resolve()
            if name == "env_cfg/robot/_robot_info.json":
                try:
                    return json.loads(item.read_text()) == {
                        "dual_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]}
                    }
                except (OSError, ValueError):
                    return False
            return False

        if status.returncode or any(
            not installation_only(line) for line in status.stdout.splitlines()
        ):
            errors.append(
                "tracked or untracked source modifications in upstream checkout"
            )
    if declared.get("tree_sha256") and digest != declared["tree_sha256"]:
        errors.append("source contents differ from the distributed upstream snapshot")
    return {
        **state,
        "revision": actual,
        "tree_sha256": digest,
        "distributed_digest_verified": bool(declared.get("tree_sha256"))
        and digest == declared["tree_sha256"],
        "errors": errors,
    }


def archive_sources(root, campaign_root):
    root = Path(root).resolve()
    digest = tree_digest(root)
    directory = Path(campaign_root) / "sources"
    directory.mkdir(exist_ok=True)
    destination = directory / (digest + ".tar.gz")
    if not destination.exists():
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".tar.gz", delete=False
        ) as temp:
            temporary = Path(temp.name)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for path in source_files(root):
                    archive.add(
                        path,
                        arcname=str(Path("CodeWorld") / path.relative_to(root)),
                        recursive=False,
                    )
                for path in (root / "vendor").glob("*/SOURCE_REVISION.json"):
                    archive.add(
                        path,
                        arcname=str(Path("CodeWorld") / path.relative_to(root)),
                        recursive=False,
                    )
            if tree_digest(root) != digest:
                raise RuntimeError(
                    "Source changed while capturing the attempt; restart with a stable checkout"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        **git_state(root),
        "tree_sha256": digest,
        "source_archive": str(destination.relative_to(campaign_root)),
    }
