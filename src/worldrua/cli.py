"""WorldRUA's portable experiment interface."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import check, load, read


def launch(root, detach):
    cmd = [sys.executable, "-m", "worldrua.batch", "service", str(root)]
    if detach:
        with (root / "service.log").open("ab") as log:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps(dict(root=str(root), service_pid=proc.pid)))
        return 0
    return subprocess.call(cmd)


def main(argv=None):
    p = argparse.ArgumentParser(prog="worldrua", description="WorldRUA experiments and artifacts")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "stage", "run"):
        s = sub.add_parser(name)
        s.add_argument("config")
        if name == "run":
            s.add_argument("--detach", action="store_true")
    s = sub.add_parser("resume", help="recover the coordinator and adopt live episodes")
    s.add_argument("root")
    s.add_argument("--detach", action="store_true")
    for name in ("status", "pause-queue", "unpause-queue"):
        sub.add_parser(name).add_argument("root")
    s = sub.add_parser("dashboard")
    s.add_argument("root")
    s.add_argument("--port", type=int, default=5191)
    s = sub.add_parser(
        "comparisons", help="render live observed/CodeWorld images from saved states"
    )
    s.add_argument("root")
    s = sub.add_parser("export")
    s.add_argument("run")
    s.add_argument("--runtime")
    s.add_argument("--videos", action="store_true")
    s.add_argument("--bundle", action="store_true")
    sub.add_parser("setup-runtime", add_help=False)
    args, rest = p.parse_known_args(argv)
    if args.command == "setup-runtime":
        from privilege.cli import main as privilege

        privilege(["setup-runtime", *rest])
        return 0
    if rest:
        p.error("unrecognized arguments: " + " ".join(rest))
    if args.command in ("doctor", "stage", "run"):
        cfg = load(args.config)
        if args.command != "stage":
            failures = check(cfg)
            if failures:
                print(json.dumps(dict(ready=False, errors=failures), indent=2))
                return 2
        if args.command == "doctor":
            print(
                json.dumps(
                    dict(
                        ready=True,
                        samples=len(cfg["jobs"]),
                        gpus=cfg["gpus"],
                        slots_per_gpu=cfg["slots_per_gpu"],
                    )
                )
            )
            return 0
        from .batch import stage

        root = stage(cfg)
        if args.command == "stage":
            print(root)
            return 0
        return launch(root, args.detach)
    if args.command == "export":
        from .media import export

        run = Path(args.run).expanduser().resolve()
        runtime = args.runtime or read(run.parent / "manifest.json").get("config", {}).get(
            "runtime"
        )
        print(
            json.dumps(
                export(run, runtime=runtime, videos=args.videos, bundle=args.bundle), indent=2
            )
        )
        return 0
    root = Path(args.root).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        p.error("No staged WorldRUA batch: " + str(root))
    if args.command == "resume":
        (root / "STOP_SERVICE").unlink(missing_ok=True)
        return launch(root, args.detach)
    if args.command == "status":
        from .dashboard import snapshot

        print(json.dumps(snapshot(root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "pause-queue":
        (root / "STOP").touch()
        print("Queue paused. Existing episodes continue.")
        return 0
    if args.command == "unpause-queue":
        (root / "STOP").unlink(missing_ok=True)
        return 0
    if args.command == "comparisons":
        from .config import RESOURCE

        return subprocess.call([sys.executable, str(RESOURCE / "watch_comparisons.py"), str(root)])
    from .dashboard import serve

    serve(root, args.port)
    return 0
