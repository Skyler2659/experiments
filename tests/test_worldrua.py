import numpy as np
import pytest
import trimesh
from PIL import Image

from worldrua import batch
from worldrua.config import load, read, save
from worldrua.dashboard import public_file
from worldrua.media import archive, scene_glb


def config_file(tmp_path):
    cfg = dict(
        output="out",
        robodojo_root="sim",
        runtime="render",
        auth_home="login",
        tasks=[dict(task="stack_bowls", samples=[0, 1])],
    )
    path = tmp_path / "batch.json"
    save(path, cfg)
    return path


def test_stage_is_portable_and_does_not_boot(tmp_path):
    cfg = load(config_file(tmp_path))
    root = batch.stage(cfg)
    assert read(root / "status.json")["runs"][0]["status"] == "QUEUED"
    assert not (root / "stack_bowls_s0/owner.json").exists()
    args = read(root / "stack_bowls_s0/command.json")["args"]
    assert args[args.index("--source-root") + 1] == str(tmp_path / "sim")
    assert read(root / "stack_bowls_s0/privilege.json")["reconstruction"]["view_ids"] == [
        "cam_head",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    with pytest.raises(FileExistsError):
        batch.stage(cfg)


def test_reject_duplicate_samples_and_missing_environment(tmp_path):
    path = config_file(tmp_path)
    cfg = read(path)
    cfg["tasks"][0]["samples"] = [0, 0]
    save(path, cfg)
    with pytest.raises(ValueError, match="unique"):
        load(path)
    cfg["tasks"][0]["samples"] = [0]
    cfg["runtime"] = "${WORLD_RUA_NONEXISTENT_TEST_VAR}"
    save(path, cfg)
    with pytest.raises(ValueError, match="Unset"):
        load(path)


def test_adopt_owner_after_coordinator_died_before_saving_status(tmp_path, monkeypatch):
    root = batch.stage(load(config_file(tmp_path)))
    rows = read(root / "status.json")["runs"]
    save(
        root / rows[0]["id"] / "owner.json", dict(pid=123, process_start="456", gpu=2, started_at=9)
    )
    monkeypatch.setattr(batch, "process_identity", lambda pid: "456" if pid == 123 else None)
    batch.reconcile(root, rows)
    assert rows[0]["status"] == "RUNNING"
    assert rows[0]["gpu"] == 2
    assert rows[0]["started_at"] == 9
    batch.reconcile(root, rows)
    assert rows[0]["started_at"] == 9


def test_pid_reuse_is_interruption_not_a_new_episode(tmp_path, monkeypatch):
    root = batch.stage(load(config_file(tmp_path)))
    rows = read(root / "status.json")["runs"]
    save(
        root / rows[0]["id"] / "owner.json", dict(pid=123, process_start="old", gpu=0, started_at=1)
    )
    monkeypatch.setattr(batch, "process_identity", lambda _: "new")
    batch.reconcile(root, rows)
    assert rows[0]["status"] == "INTERRUPTED"
    assert "success" not in rows[0]


def test_queue_reserves_memory_and_counts_episodes():
    cfg = dict(gpus=[0, 1], min_free_mib=14500, slots_per_gpu=2)
    rows = [dict(status="RUNNING", gpu=0), dict(status="RUNNING", gpu=0)]
    gpus = [dict(index=0, total=46068, used=23000), dict(index=1, total=46068, used=35000)]
    assert batch.eligible_gpus(cfg, rows, gpus) == []
    gpus[1]["used"] = 12000
    assert batch.eligible_gpus(cfg, rows, gpus) == [gpus[1]]


def test_cleanup_only_copies_not_source_or_symlink(tmp_path):
    root = tmp_path / "batch"
    home = root / "sample/agent-home"
    home.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    (source / "auth.json").write_text("dummy")
    (root / "linked-home").symlink_to(source, target_is_directory=True)
    (home / "auth.json").write_text("dummy-copy")
    batch.cleanup_logins(root)
    assert (source / "auth.json").read_text() == "dummy"
    assert not (home / "auth.json").exists()


def test_visual_export_preserves_texture_and_hides_camera(tmp_path):
    from privilege.storage import from_meshes

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), image=Image.new("RGB", (4, 4), "red")
    )
    mesh.export(tmp_path / "mesh.glb")
    pose = np.eye(4)
    pose[0, 3] = 2
    world = from_meshes(
        {
            "object": {"mesh": tmp_path / "mesh.glb", "T_W_object": pose},
            "camera_support": {"mesh": tmp_path / "mesh.glb"},
        },
        tmp_path / "world",
    )
    out = tmp_path / "current.glb"
    scene_glb(world.root, out)
    scene = trimesh.load(out, force="scene", process=False)
    assert len(scene.geometry) == 1
    assert scene.bounds[:, 0].tolist() == pytest.approx([1.5, 2.5])
    assert next(iter(scene.geometry.values())).visual.kind == "texture"


def test_bundle_excludes_login_and_dependencies(tmp_path):
    import tarfile

    run = tmp_path / "run"
    world = run / "world"
    world.mkdir(parents=True)
    (world / "world.json").write_text("{}")
    home = world / "source/.codex-home"
    home.mkdir(parents=True)
    (home / "auth.json").write_text("dummy")
    (world / "external").symlink_to(home, target_is_directory=True)
    out = tmp_path / "bundle.tar.gz"
    archive(run, world, out)
    with tarfile.open(out) as t:
        assert t.getnames() == ["world/world.json"]


def test_dashboard_cannot_serve_private_files(tmp_path):
    with pytest.raises(ValueError):
        public_file(tmp_path, "sample/agent-home/auth.json")
    with pytest.raises(ValueError):
        public_file(tmp_path, "../other.png")
    assert public_file(tmp_path, "sample/episode/steps/0001/cam_head.png").is_relative_to(tmp_path)
