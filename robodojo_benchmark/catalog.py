"""Build samples from the published slot inventory, never from policy outcomes."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {
    "roboprobe_l3": {"label": "RoboProbe-L3", "representation": None,
                     "executor": "RoboDojo_Agent_L3_Inspect_EEF", "observation": "native_rgb"},
    "ours_no_codeworld": {"label": "Ours − CodeWorld", "representation": "none",
                          "executor": "codex_cli", "observation": "external_multiview_rgbd"},
    "ours": {"label": "Ours", "representation": "reconstructed",
             "executor": "codex_cli", "observation": "external_multiview_rgbd"},
    "gt_codeworld": {"label": "GT CodeWorld", "representation": "ground_truth",
                     "executor": "codex_cli", "observation": "external_multiview_rgbd_plus_oracle"},
}
DEFAULT_TASKS = ["insert_tubes", "build_tower", "pack_objects_into_box", "hang_mugs",
                 "fill_egg_holder", "organize_table", "arrange_largest_number", "swap_T",
                 "store_tools_in_toolbox", "sweep_blocks", "play_stacking_toy", "fill_pen_holder", "stack_bowls", "deposit_coin"]


def read(path):
    return json.loads(Path(path).read_text())


def published_catalog(root=ROOT):
    path = Path(root) / "vendor/RoboProbe/results/l3_inspect_eef_official_2100/astra_task_layouts.json"
    return {r["task"]: {"dimension": r["dimension"].lower(), "samples": [
        {"sample": s["slot"], "variant": s["variant"], "layout_id": s["layout_id"],
         "native_task": r["task"] + ("_random" if s["variant"] == "random" else "")}
        for s in r["slots"]]} for r in read(path)["rows"]}


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def build_manifest(spec, root=ROOT):
    # A small custom suite inherits the same frozen implementation/budget defaults.
    defaults = read(Path(root) / "experiments/robodojo/suite.json")
    supplied = copy.deepcopy(spec)
    spec = {**defaults, **supplied}
    for section in ("agent", "roboprobe", "reconstruction"):
        spec[section] = {**defaults[section], **supplied.get(section, {})}
    for section in ("agent", "reconstruction"):
        positive_int(spec[section]["wall_timeout_s"], section + ".wall_timeout_s")
        if not isinstance(spec[section]["model"], str) or not spec[section]["model"].strip():
            raise ValueError(section + ".model must be explicit")
    for name in ("max_depth", "max_cycles"):
        positive_int(spec["reconstruction"][name], "reconstruction." + name)
    positive_int(spec["roboprobe"]["max_llm_calls"], "roboprobe.max_llm_calls")
    positive_int(spec["roboprobe"]["image_horizon"], "roboprobe.image_horizon")
    catalog = published_catalog(root)
    strategies = spec.get("strategies", list(STRATEGIES))
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies) - STRATEGIES.keys():
        raise ValueError("Select distinct registered strategies")
    eval_seed = positive_int(spec.get("eval_seed", 0), "eval_seed", 0)
    gt_mode = spec.get("gt_mode", "live_state")
    if gt_mode not in ("live_state", "initial_geometry"):
        raise ValueError("gt_mode must be live_state or initial_geometry")
    tasks = spec.get("tasks", DEFAULT_TASKS)
    if not tasks:
        raise ValueError("No tasks selected")
    jobs, seen = [], set()
    for entry in tasks:
        entry = {"task": entry} if isinstance(entry, str) else entry
        task = identifier(entry["task"])
        if task in seen:
            raise ValueError(f"Duplicate task {task}")
        seen.add(task)
        if "layouts" in entry:
            samples = []
            for idx, row in enumerate(entry["layouts"], 1):
                if type(row) is int:
                    row = {"layout_id": row}
                native = identifier(row.get("native_task", task))
                samples.append({"sample": idx, "native_task": native,
                                "variant": "random" if native.endswith("_random") else "standard",
                                "layout_id": positive_int(row["layout_id"], "layout_id", 0)})
            dimension = entry.get("dimension", catalog.get(task, {}).get("dimension", "custom"))
        else:
            if task not in catalog:
                raise ValueError(f"New task {task}: supply layouts and dimension in the suite")
            samples = copy.deepcopy(catalog[task]["samples"])
            dimension = catalog[task]["dimension"]
        selection = entry.get("samples", spec.get("samples", "all"))
        if selection != "all":
            if type(selection) is int:
                positive_int(selection, "samples")
                selection = list(range(1, selection + 1))
            if not isinstance(selection, list) or not selection or len(set(selection)) != len(selection):
                raise ValueError("samples must be 'all', a positive count, or distinct slot numbers")
            for sample in selection:
                positive_int(sample, "sample")
            if set(selection) - {r["sample"] for r in samples}:
                raise ValueError(f"Unknown samples for {task}: {selection}")
            samples = [s for s in samples if s["sample"] in selection]
        if not samples or len({(s["native_task"], s["layout_id"]) for s in samples}) != len(samples):
            raise ValueError(f"Empty or duplicate layouts for {task}")
        for sample in samples:
            for strategy in strategies:
                job_id = f"{task}__{sample['sample']:03d}__{strategy}"
                jobs.append({"job_id": job_id, "task": task, "dimension": dimension,
                             **sample, "eval_seed": eval_seed, "strategy": strategy,
                             **STRATEGIES[strategy], "gt_mode": gt_mode if strategy == "gt_codeworld" else None})
    return {"schema": "codeworld-robodojo-campaign-v1", "name": identifier(spec.get("name", "robodojo")),
            "protocol": "official_tasks_layouts_actions_scorer_with_declared_observation_augmentation",
            "sample_indexing": "1-based published slot; layout_id is the native sorted-file index",
            "spec": copy.deepcopy(spec), "jobs": jobs, "job_count": len(jobs),
            "upstreams": read(Path(root) / "experiments/robodojo/upstreams.json"),
            "platform_sources": read(Path(root) / "experiments/robodojo/platform_sources.json")}


def layout_path(robodojo_root, job, env_config="arx_x5"):
    # SeedManager enumerates sorted files. The index need not equal the filename suffix.
    folder = Path(robodojo_root) / "Assets/Eval_Layout/RoboDojo" / env_config / str(job["eval_seed"])
    pattern = re.compile(re.escape(job["native_task"]) + r"_\d+\.json")
    files = sorted((p for p in folder.iterdir() if pattern.fullmatch(p.name)),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    try:
        return files[job["layout_id"]]
    except IndexError:
        raise ValueError(f"Layout {job['layout_id']} unavailable: {job['native_task']} ({len(files)} files)")


def asset_revision(robodojo_root):
    """Read the asset checkout's revision without requiring ownership of its Git repo.

    An exported Assets folder can lack Git metadata; do not confuse the enclosing
    RoboDojo code checkout's HEAD with the dataset revision in that case.
    """
    rd = Path(robodojo_root).resolve()
    assets = (rd / 'Assets').resolve()
    for folder in (assets, assets.parent):
        if folder == rd:
            continue
        git = folder / '.git'
        if git.is_file() and git.read_text().startswith('gitdir: '):
            git = (folder / git.read_text().strip()[8:]).resolve()
        if not (git / 'HEAD').is_file():
            continue
        head = (git / 'HEAD').read_text().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        packed = git / 'packed-refs'
        if packed.exists():
            for line in packed.read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    return fields[0]
    return None
