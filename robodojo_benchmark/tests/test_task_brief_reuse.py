import importlib.util
from pathlib import Path
import pytest
from robodojo_benchmark.catalog import ROOT, read
from robodojo_benchmark.campaign import write
from robodojo_benchmark.task_brief import load_task_brief, apply_task_brief
from robodojo_benchmark.reuse import validate_reuse, restore_reconstruction
from robodojo_benchmark.worker import prepare_files


def test_all_ours_arms_receive_released_recipe(tmp_path):
    recipe_dir = ROOT / "vendor/RoboProbe/policy/RoboDojo_Agent_L3_Inspect/recipes"
    spec = importlib.util.spec_from_file_location("released_recipes", recipe_dir / "__init__.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for representation in ("none", "reconstructed", "ground_truth"):
        episode = tmp_path / representation
        job = {"representation": representation, "gt_mode": "live_state", "native_task": "build_tower"}
        prepare_files(episode, job)
        brief = read(episode / "task/public_task_brief.json")
        assert brief["text"] == module.task_recipe("build_tower")[1]
        task = apply_task_brief(read(episode / "task/task.json"), "Build a tower.", brief)
        assert "all eight pieces" in task["goal"] and "two upright supports" in task["goal"]
        assert task["native_instruction"] == "Build a tower."
    assert load_task_brief(ROOT / "vendor/RoboProbe", "build_tower_random")["text"] == brief["text"]
    assert not load_task_brief(tmp_path, "new_task")["available"]
    with pytest.raises(ValueError):load_task_brief(tmp_path, "../private")


def test_reuse_rejects_different_instance(tmp_path):
    source, attempt = tmp_path / "old", tmp_path / "new"
    job = dict(strategy="ours", native_task="build_tower", layout_id=0, eval_seed=0, representation="reconstructed", gt_mode="live_state")
    write(source / "job.json", job);write(source / "native_layout.json", {"layout": 0})
    write(attempt / "native_layout.json", {"layout": 0})
    runtime = {"reuse_initial_artifacts": {"ours": str(source)}}
    assert validate_reuse(runtime, job, attempt) == source
    with pytest.raises(ValueError):validate_reuse(runtime, {**job, "eval_seed": 1}, attempt)
    write(attempt / "native_layout.json", {"layout": 2})
    with pytest.raises(ValueError):validate_reuse(runtime, job, attempt)


def test_reuse_copies_initial_world_only(tmp_path, monkeypatch):
    import codeworld_workspace
    monkeypatch.setattr(codeworld_workspace.World, "verify_binding", lambda self: None)
    source, episode = tmp_path / "old", tmp_path / "new"
    world = source / "episode/preparation/workspace"
    write(world / "build_report.json", {"status": "PASS"})
    write(world / "world.json", {})
    for name in ("geometry", "objects", "source"):(world / name).mkdir()
    write(world / "state/current.json", {"revision": 0, "history": []})
    write(source / "episode/state/current.json", {"revision": 99, "history": ["prior actions"]})
    write(source / "episode/task/completion.json", {"success": True})
    episode.mkdir()
    restore_reconstruction(source, episode, {"job_id": "test"}, "fresh-run")
    assert read(episode / "state/current.json")["revision"] == 0
    assert not (episode / "task/completion.json").exists()
    assert read(episode / "preparation/reconstruction_receipt.json")["historical_assets_reused"]


def test_gt_cache_reads_fresh_pose_and_drops_history(tmp_path, monkeypatch):
    import numpy as np
    import trimesh
    from types import SimpleNamespace
    from robodojo_benchmark.ground_truth import GroundTruthWorld
    from codeworld_workspace import World
    pos = np.array([0., 0., 0.])
    obj = SimpleNamespace(prim_path='/cube', get_world_pose=lambda: (pos.copy(), np.array([1., 0., 0., 0.])))
    env = SimpleNamespace(scene_manager=SimpleNamespace(stage=None,
        _rigid_and_dynamic_objects=[{'cube': obj}], _geometry_objects=[{}]))
    monkeypatch.setattr('robodojo_benchmark.ground_truth.local_mesh', lambda *args: trimesh.creation.box())
    source = tmp_path / 'old'
    gt = GroundTruthWorld(env, source, [0, 0, 0])
    pos[0] = 9.
    gt.sync({'robot_state_ref': 'old:99', 'official_actions': 99})
    pos[0] = 2.
    def no_reexport(*args):raise AssertionError('GT geometry must be reused')
    monkeypatch.setattr('robodojo_benchmark.ground_truth.local_mesh', no_reexport)
    fresh = tmp_path / 'new'
    cached = GroundTruthWorld(env, fresh, [0, 0, 0], reuse_episode=source)
    cached.sync({'robot_state_ref': 'new:0', 'official_actions': 0})
    world = World(fresh); world.verify_binding()
    state = world.state()
    assert state['objects']['cube']['T_W_object'][0][3] == 2.
    assert len(state['history']) == 1 and state['oracle']['robot_state_ref']['robot_state_ref'] == 'new:0'


def test_public_briefs_cover_entire_published_catalog():
    from robodojo_benchmark.catalog import published_catalog, DEFAULT_TASKS
    recipe_dir = ROOT / 'vendor/RoboProbe/policy/RoboDojo_Agent_L3_Inspect/recipes'
    spec = importlib.util.spec_from_file_location('all_released_recipes', recipe_dir / '__init__.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    catalog = published_catalog()
    assert set(DEFAULT_TASKS) <= set(catalog)
    for name in catalog:
        for task in (name, name + '_random'):
            brief = load_task_brief(ROOT / 'vendor/RoboProbe', task)
            released = module.task_recipe(task)
            assert released is not None, task
            assert brief['available'] and brief['text'].strip(), task
            assert brief['text'] == released[1], task
            goal = apply_task_brief({}, 'instance-specific instruction', brief)
            assert goal['goal'] == 'instance-specific instruction\n\nTASK RECIPE:\n' + released[1].rstrip()
