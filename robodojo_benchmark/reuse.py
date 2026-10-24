"""Explicit initial-artifact reuse for execution reruns; never copy execution history."""
from pathlib import Path
import shutil
from .catalog import read
from .campaign import write


def validate_reuse(runtime, job, attempt):
    source = runtime.get("reuse_initial_artifacts", {}).get(job["strategy"])
    if not source:
        return None
    source = Path(source).resolve()
    old = read(source / "job.json")
    for field in ("native_task", "layout_id", "eval_seed", "representation", "gt_mode"):
        if old[field] != job[field]:
            raise ValueError("Reuse episode mismatch: " + field)
    if read(source / "native_layout.json") != read(attempt / "native_layout.json"):
        raise ValueError("Reuse native layout contents changed")
    write(attempt / "artifact_reuse.json", {"source_attempt": str(source),
          "mode": "initial_geometry_only_fresh_execution", "historical_assets_reused": True,
          "prior_actions_or_final_state_copied": False})
    return source


def restore_reconstruction(source, episode, job, run_id):
    world = source / "episode/preparation/workspace"
    report = read(world / "build_report.json")
    if report["status"] != "PASS":
        raise ValueError("Initial reconstructed world did not pass source binding")
    from codeworld_workspace import World
    World(world).verify_binding()
    for name in ("world.json", "geometry", "objects", "source"):
        src, dst = world / name, episode / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    state = read(world / "state/current.json")
    if state["revision"] != 0 or state.get("history"):
        raise ValueError("Reuse requires untouched initial reconstruction state")
    state.update(source_episode=run_id, source_sample=job["job_id"])
    write(episode / "state/current.json", state)
    for name in ("scenarios", "observations"):
        (episode / name).mkdir(exist_ok=True)
    write(episode / "preparation/reconstruction_receipt.json", {
          "status": "REUSED_INITIAL", "historical_assets_reused": True,
          "same_reset_as_execution": False, "same_layout_seed_verified": True,
          "source_attempt": str(source), "source_program_binding_verified": True,
          "prior_actions_or_final_state_copied": False})
