"""Wire the released repositories without modifying their Python/scoring source."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[2]
rd, rp = root / "vendor/RoboDojo", root / "vendor/RoboProbe"


def link(path, target):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Keep existing {path}; resolve the workspace layout explicitly")
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


link(rd / "XPolicyLab", rp)
link(rp.parent / "env_cfg", rd / "env_cfg")
# The current RoboDojo imports client_server at package root; RoboProbe places
# the same transport under utils/client_server. This is an import-path shim.
link(rp / "client_server", rp / "utils/client_server")
info = rd / "env_cfg/robot/_robot_info.json"
expected = {"dual_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]}}
if info.exists():
    if json.loads(info.read_text()).get("dual_x5") != expected["dual_x5"]:
        raise ValueError("Existing ARX X5 action dimensions disagree with the official binding")
else:
    info.write_text(json.dumps(expected, indent=2) + "\n")
print("RoboProbe import, shared env_cfg and public dual-X5 action dimensions ready")
