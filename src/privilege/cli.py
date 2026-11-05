"""Standalone reconstruction/compiler commands. Never starts a robot."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from .world import World, read


def main(argv=None):
    parser = argparse.ArgumentParser(prog="privilege")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "setup-runtime", help="install separate Node/Chromium/Python rendering dependencies"
    )
    p.add_argument("output")
    p.add_argument("--python", default="python3")
    p.add_argument("--node-from", help="existing Node installation prefix (required on macOS)")
    p = sub.add_parser("reconstruct", help="one calibrated RGB + known robot GLB to world")
    for name in ("image", "camera", "robot-glb", "config", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("compile", help="compile an already exported RCWM scene")
    for name in ("run", "export", "runtime", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--frame", default="world")
    p = sub.add_parser("import-world", help="copy geometry/state without a legacy harness")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--frame", required=True)
    p.add_argument("--transform", help="JSON file containing row-major T_host_world")
    p = sub.add_parser("inspect")
    p.add_argument("world")
    args = parser.parse_args(argv)
    if args.command == "setup-runtime":
        import sys

        from .reconstruction import RESOURCES

        if sys.platform == "darwin" and not args.node_from:
            parser.error("On macOS provide --node-from with a native Node installation prefix")
        root = Path(args.output).expanduser().resolve()
        argv = [
            "bash",
            str(RESOURCES / "recursive_cwm/setup/setup_runtime.sh"),
            str(root),
            "--python",
            args.python,
        ]
        if args.node_from:
            argv += ["--node-from", str(Path(args.node_from).expanduser().resolve())]
        env = dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=str(root / ".browsers"))
        subprocess.run(argv, env=env, check=True)
        result = {"runtime": str(root)}
    elif args.command == "reconstruct":
        from .observations import register
        from .reconstruction import reconstruct

        config_path = Path(args.config).resolve()
        config = read(config_path)
        for key in ("runtime", "codex_home", "codex_config"):
            if config.get(key):
                config[key] = str((config_path.parent / Path(config[key]).expanduser()).resolve())
        camera = read(args.camera)
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=False)
        cap = register(
            out,
            args.image,
            K=camera["K"],
            T_W_C=camera["T_W_C"],
            frame=camera["frame"],
            robot_state=camera.get("robot_state"),
        )
        world = reconstruct(out, cap["capture"], args.robot_glb, out / "reconstruction", config)
        result = {"world": str(world.root)}
    elif args.command == "compile":
        from .build import build

        result = build(args.run, args.export, args.output, args.runtime, frame=args.frame)
    elif args.command == "import-world":
        from .storage import import_world

        world = import_world(
            args.source,
            args.output,
            frame=args.frame,
            T_host_world=read(args.transform) if args.transform else None,
        )
        result = {"world": str(world.root)}
    else:
        world = World(args.world)
        result = {
            "objects": world.objects(),
            "frame": world.manifest["frame"],
            "revision": world.state()["revision"],
            "binding": world.verify_binding(),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
