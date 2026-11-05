import shutil
import json

import numpy as np
import trimesh

from privilege.build import build, digest, snapshot_ignore
from privilege.world import World


def test_snapshot_does_not_follow_recursive_validation_runtime(tmp_path):
    run = tmp_path / "run"
    child = run / "fractal" / "child"
    rig = child / "validation_runtime"
    rig.mkdir(parents=True)
    source = child / "component.js"
    source.write_text("export function build() {}")
    (rig / "child").symlink_to(child, target_is_directory=True)
    (rig / "environment.js").symlink_to(source)
    external = tmp_path / "shared_runtime"
    external.mkdir()
    (external / "unneeded.bin").write_bytes(b"runtime")
    (rig / ".render-tools").symlink_to(external, target_is_directory=True)
    (child / "session.log").write_text("excluded")

    snapshot = tmp_path / "snapshot"
    shutil.copytree(run, snapshot, ignore=snapshot_ignore)
    copied = snapshot / "fractal" / "child"
    assert (copied / "component.js").read_text() == source.read_text()
    assert (copied / "validation_runtime/environment.js").read_text() == source.read_text()
    assert not (copied / "validation_runtime/child").exists()
    assert not (copied / "validation_runtime/.render-tools").exists()
    assert not (copied / "session.log").exists()


def test_build_bakes_nonuniform_scale_without_changing_executed_geometry(tmp_path):
    run, export, runtime = [tmp_path / name for name in ["run", "export", "runtime"]]
    run.mkdir()
    export.mkdir()
    three = runtime / ".render-tools/node_modules/three"
    three.mkdir(parents=True)
    (three / "package.json").write_text("{}")
    (run / "parameters.json").write_text(json.dumps({"objects": {"mug": {"movable": True}}}))
    affine = trimesh.transformations.rotation_matrix(0.16, [0, 0, 1])
    affine[:3, :3] = affine[:3, :3] @ np.diag([1.0, 1.0, 1.1])
    affine[:3, 3] = [-0.443, 0.443, -0.006]
    mesh = trimesh.creation.box([0.08, 0.08, 0.06])
    mesh.apply_transform(affine)
    mesh.export(export / "scene.glb")
    node = {"object_id": "mug", "matrix_world_column_major": affine.flatten(order="F").tolist()}
    (export / "executed_binding.json").write_text(json.dumps({
        "source_files": {"parameters.json": digest(run / "parameters.json")},
        "objects": {"mug": node}, "nodes": [node],
    }))
    (export / "object_inventory.json").write_text(json.dumps([
        {"object_id": "mug", "mesh_ref": "scene.glb"},
    ]))
    output = tmp_path / "world"
    build(run, export, output, runtime)
    world = World(output)
    world.verify_binding()
    actual = world._mesh("mug", world.state())
    assert np.allclose(actual.bounds, mesh.bounds, atol=2e-6)
    assert np.allclose(np.sort(actual.vertices, axis=0), np.sort(mesh.vertices, axis=0), atol=2e-6)
    pose = np.array(world.state()["objects"]["mug"]["T_W_object"])
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
