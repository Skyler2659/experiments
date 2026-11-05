"""Use the selected login and the tested same-session recovery wrapper."""

import os
from pathlib import Path
import shutil
import sys

from pyrualean.resilient_cli import run


def main():
    binary = os.environ["WORLD_RUA_CODEX"]
    source = Path(os.environ["WORLD_RUA_AUTH_HOME"]).resolve()
    root = Path(os.environ["WORLD_RUA_RUN"]).resolve()
    home = Path(os.environ.get("CODEX_HOME", root / "agent-home")).resolve()
    if not home.is_relative_to(root) or home == source:
        raise ValueError("CLI sessions must use a private home inside this episode")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (home / "auth.json").exists():
        shutil.copy2(source / "auth.json", home / "auth.json")
        (home / "auth.json").chmod(0o600)
    os.environ["CODEX_HOME"] = str(home)
    args = sys.argv[1:]
    if not args or args[0] != "exec":
        os.execv(binary, [binary, *args])
    args[1:1] = ["--ignore-user-config", "-c", "notify=[]", "-c", "features.chronicle=false"]
    control = (
        root / "episode/resource_guard/control.json"
        if any("mcp_servers.pyrualean" in a for a in args)
        else None
    )
    payload = sys.stdin.buffer.read() if args[-1] == "-" else b""
    return run(binary, args, input_data=payload, directory=root / "recovery", control=control)
