"""Offline export of recorded assets and media. Never replays robot actions."""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from .config import RESOURCE, read, save


def hidden(oid, record):
    return (
        record.get("role") == "robot_snapshot"
        or oid in {"mounting_plate", "mounting_bracket", "mounting_post"}
        or any(
            x in oid.lower()
            for x in (
                "known_robot",
                "camera",
                "sensor",
                "tripod",
                "cam_head",
                "cam_left",
                "cam_right",
            )
        )
    )


def scene_glb(world, output):
    """Assemble original visual assets at recorded poses, retaining textures/parts."""
    import numpy as np
    import trimesh

    spec, state = read(world / "world.json"), read(world / "state/current.json")
    scene = trimesh.Scene()
    for oid, record_path in spec["objects"].items():
        record = read(world / record_path)
        if hidden(oid, record):
            continue
        live = state["objects"][oid]
        parts = live.get("assembly", {}).get("parts") or {
            "root": dict(
                local_mesh=record["geometry"]["local_mesh"], T_object_part=np.eye(4).tolist()
            )
        }
        for part_id, part in parts.items():
            source = trimesh.load(world / part["local_mesh"], force="scene", process=False)
            parent = np.asarray(live["T_W_object"]) @ np.asarray(part["T_object_part"])
            for node in source.graph.nodes_geometry:
                transform, geometry = source.graph[node]
                name = f"{oid}/{part_id}/{node}"
                scene.add_geometry(
                    source.geometry[geometry].copy(),
                    node_name=name,
                    geom_name=name,
                    transform=parent @ transform,
                )
    scene.export(str(output))
    return dict(
        path=str(output),
        revision=state["revision"],
        appearance="original visual materials",
        robot_and_cameras_hidden=True,
    )


def archive(run, world, output):
    """Explicit artifact roots; login files, symlinks and runtime dependencies excluded."""
    skip = {
        "auth.json",
        ".codex-home",
        "agent-home",
        ".venv",
        ".render-tools",
        ".browsers",
        ".system-libs",
        "node_modules",
        "__pycache__",
        ".git",
    }
    roots = [(run / "audit", "audit"), (run / "recovery", "recovery"), (world, "world")]
    for name in (
        "prompt.txt",
        "calls.jsonl",
        "codex_events.jsonl",
        "tool_turns.json",
        "codex_stderr.txt",
        "result.json",
        "last_message.md",
        "steps",
    ):
        roots.append((run / "episode" / name, "episode/" + name))
    roots.append((run / "episode/privilege/reconstruction/runtime/runs", "reconstruction_runs"))
    with tarfile.open(output, "w:gz") as tar:
        for path, prefix in roots:
            files = path.rglob("*") if path.is_dir() else [path]
            for item in files:
                relative = item.relative_to(path) if path.is_dir() else Path()
                if (
                    not item.is_file()
                    or item.is_symlink()
                    or any(x in skip for x in relative.parts)
                    or item.name in skip
                    or any(p.is_symlink() for p in item.parents if p != path.parent)
                ):
                    continue
                tar.add(item, arcname=str(Path(prefix) / relative), recursive=False)


def export(run, *, runtime=None, videos=False, bundle=False):
    integration = read(run / "episode/privilege/integration.json")
    if not integration.get("world"):
        raise ValueError("No completed CodeWorld integration in this episode")
    world = Path(integration["world"])
    if not world.is_absolute():
        world = (run / "episode/privilege" / world).resolve()
    out = run / "exports"
    out.mkdir(exist_ok=True)
    result = {"scene": scene_glb(world, out / "current_scene.glb")}
    initial = run / "episode/privilege/reconstruction/export/scene.glb"
    if initial.exists():
        shutil.copy2(initial, out / "initial_scene.glb")
    for src in (run / "episode/bench/eval_result").rglob("*.mp4"):
        if ".tmp." not in src.name and "_stream" not in src.parts:
            shutil.copy2(src, out / src.name)
    if bundle:
        archive(run, world, out / "artifacts.tar.gz")
        result["bundle"] = str(out / "artifacts.tar.gz")
    if videos:
        if not runtime:
            raise ValueError("--runtime is required for comparison video rendering")
        subprocess.run(
            [
                sys.executable,
                str(RESOURCE / "export_videos.py"),
                str(run),
                "--runtime",
                str(runtime),
            ],
            check=True,
        )
        result["videos"] = str(run / "videos")
    save(out / "manifest.json", result)
    return result
