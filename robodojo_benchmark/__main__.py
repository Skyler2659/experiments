from __future__ import annotations

import argparse
import collections
import fcntl
import json
from pathlib import Path
import signal
import sys
from .catalog import ROOT, STRATEGIES, read
from .campaign import Campaign, write


def main():
    parser = argparse.ArgumentParser(description="RoboDojo: select strategy, task and 1-based sample")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("init")
    p.add_argument("--suite", default=str(ROOT / "experiments/robodojo/suite.json"))
    p.add_argument("--output", required=True)
    for command in ("list", "status", "summary"):
        p = commands.add_parser(command); p.add_argument("--campaign", required=True)
        if command == "list":
            p.add_argument("--strategy", choices=STRATEGIES); p.add_argument("--task"); p.add_argument("--sample", type=int)
            p.add_argument('--state', choices=('PENDING','RUNNING','SCORED','METHOD_FAILURE','INFRA_ERROR','UNSTABLE','INTERRUPTED'))
    for command in ("run", "worker", "doctor"):
        p = commands.add_parser(command); p.add_argument("--campaign", required=True); p.add_argument("--runtime", required=True)
        p.add_argument("--strategy", choices=STRATEGIES); p.add_argument("--task"); p.add_argument("--sample", type=int)
        if command != "doctor":
            p.add_argument("--gpu", default="0"); p.add_argument("--shard"); p.add_argument("--max-jobs", type=int)
    p = commands.add_parser("probe")
    p.add_argument("--runtime", required=True); p.add_argument("--output", required=True)
    p.add_argument("--suite", help="Optional initialization-only suite; never included in task success statistics")
    p.add_argument("--task", default="insert_tubes"); p.add_argument("--sample", type=int, default=1)
    p.add_argument("--strategy", choices=("ours_no_codeworld", "gt_codeworld", "roboprobe_l3"), default="gt_codeworld")
    p.add_argument("--gpu", default="0")
    for command in ("retry", "recover"):
        p = commands.add_parser(command); p.add_argument("--campaign", required=True)
        p.add_argument("--job", required=True); p.add_argument("--reason", required=True)
    p = commands.add_parser("bundle"); p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "init":
        c = Campaign.create(args.output, read(args.suite))
        print(json.dumps({"campaign": str(c.root), "jobs": len(c.jobs)}, indent=2)); return
    if args.command == "bundle":
        from .bundle import bundle
        print(bundle(args.output)); return
    if args.command == "probe":
        spec = read(args.suite or ROOT / "experiments/robodojo/suite.json")
        spec.update(name="native_probe", purpose="validation", strategies=[args.strategy])
        if not args.suite:spec['tasks']=[{"task": args.task, "samples": [args.sample]}]
        c = Campaign.create(args.output, spec)
    else:
        c = Campaign(args.campaign)
    if args.command == "status":
        print(json.dumps(dict(collections.Counter(r["status"] for r in c.rows())), indent=2)); return
    if args.command == "summary":
        from .report import summarize
        print(json.dumps(summarize(c), indent=2)); return
    if args.command in ("retry", "recover"):
        getattr(c, args.command)(args.job, args.reason); print("Recorded " + args.command); return
    candidates = (c.select() if args.command == 'probe' and args.suite else
                  c.select(args.strategy, args.task, args.sample, getattr(args, "shard", None)))
    if args.command == "list":
        states = {row['job_id']:row for row in c.rows()}
        for key in candidates:
            j = c.jobs[key]
            row = states[key]
            if args.state and row['status'] != args.state:continue
            print(f"{key}\t{row['status']}\tattempt={row['attempt']}\t{j['native_task']}\tlayout={j['layout_id']}\teval_seed={j['eval_seed']}")
        return
    from .worker import runtime_config, execute_job
    runtime = runtime_config(args.runtime)
    if args.command == "doctor":
        from .doctor import doctor
        result = doctor(runtime, [c.jobs[key] for key in candidates])
        print(json.dumps(result, indent=2)); raise SystemExit(0 if result["ok"] else 1)
    if args.command == "run" and (not args.strategy or not args.task or args.sample is None):
        parser.error("run requires --strategy, --task, and --sample; use worker for batches")
    def interrupted(*_):
        raise KeyboardInterrupt("Worker interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    maximum = 1 if args.command == 'run' else (len(candidates) if args.command == 'probe' else (args.max_jobs or len(candidates)))
    # One worker per physical GPU; other GPUs and CPU policy services run independently.
    lock_path = Path(runtime.get("gpu_lock_directory", "/tmp")) / ("codeworld-benchmark-gpu-" + str(args.gpu).replace("/", "_") + ".lock")
    count, failed = 0, False
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while count < maximum:
            job = c.claim(candidates)
            if job is None:
                break
            print(json.dumps({"started": job["job_id"], "attempt": job["attempt"]}), flush=True)
            result = execute_job(c, job, runtime, args.gpu, probe=args.command == "probe")
            c.finish(job, result)
            print(json.dumps(result), flush=True)
            failed |= result["status"] in ("INFRA_ERROR", "UNSTABLE", "INTERRUPTED")
            count += 1
            if result["status"] in ("INFRA_ERROR", "INTERRUPTED"):
                break  # Fix a broken worker/credential rather than failing the remaining queue.
    if not count:
        print("No pending job matches; completed attempts are preserved")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
