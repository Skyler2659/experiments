"""Preserve assigned public metadata and check it before spending solver cycles."""

import hashlib
import json
import os
from pathlib import Path
import sys


SNAPSHOT = "assigned_public_inputs.json"
METADATA = ("input.json", "views.json", "camera_metadata.json", "robot_binding.json")


def write_synced(path, value):
    # These immutable inputs have no concurrent writers. Direct publication avoids
    # relying on rename visibility on the experiment's shared filesystem.
    with Path(path).open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def snapshot_inputs(directory, files=None):
    directory = Path(directory)
    if files is None:
        files = {name: json.loads((directory / name).read_text()) for name in METADATA}
    write_synced(directory / SNAPSHOT, {"schema": "assigned-public-inputs-v1", "files": files})
    return ensure_inputs(directory)


def ensure_inputs(directory):
    directory = Path(directory)
    snapshot = json.loads((directory / SNAPSHOT).read_text())
    if snapshot.get("schema") != "assigned-public-inputs-v1" or set(snapshot["files"]) != set(METADATA):
        raise ValueError("Invalid assigned-input snapshot")
    files = snapshot["files"]
    # Verify the saved metadata belongs to these RGBs and this posed public robot.
    for view in files["views.json"]["views"]:
        name = view["rgb_uri"]
        if Path(name).name != name:
            raise ValueError("Reference must be an input-directory file")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != view["rgb_sha256"]:
            raise ValueError("Assigned RGB differs from metadata snapshot")
    binding = files["robot_binding.json"]
    if hashlib.sha256((directory / "known_robot.glb").read_bytes()).hexdigest() != binding["robot_node"]["asset"]["sha256"]:
        raise ValueError("Public robot differs from metadata snapshot")
    missing = []
    for name, expected in files.items():
        try:
            actual = json.loads((directory / name).read_text())
        except FileNotFoundError:
            missing.append(name)
        else:
            if actual != expected:
                raise ValueError("Assigned metadata changed: " + name)
    # Do not repair anything until every readable input passes validation.
    for name in missing:
        write_synced(directory / name, files[name])
        if json.loads((directory / name).read_text()) != files[name]:
            raise ValueError("Restored input did not read back: " + name)
    return {"validated": True, "restored": missing}


if __name__ == "__main__":
    print(json.dumps(ensure_inputs(sys.argv[1])), flush=True)
