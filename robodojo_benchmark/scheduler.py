"""A GPU reservation with one, two or four independent episode workers."""
import fcntl
import json
import multiprocessing
from pathlib import Path
import signal

from .worker import execute_job, finish_job


def _interrupt(*_):
    raise KeyboardInterrupt("Worker interrupted")


def _drain(campaign, candidates, runtime, gpu, slot, maximum, claimed, gate, stop, probe):
    signal.signal(signal.SIGTERM, _interrupt)
    failed = False
    try:
        while not stop.is_set():
            # max_jobs applies to the whole invocation, not to each slot.
            with gate:
                if stop.is_set() or claimed.value >= maximum:
                    break
                job = campaign.claim(candidates)
                if job is None:
                    break
                claimed.value += 1
            print(json.dumps({"started": job["job_id"], "attempt": job["attempt"],
                              "gpu": str(gpu), "slot": slot}), flush=True)
            result = execute_job(campaign, job, runtime, gpu, probe=probe)
            if result["status"] in ("INFRA_ERROR", "INTERRUPTED"):
                stop.set()  # Let active episodes finish, but stop claiming new work.
            finish_job(campaign, job, result)
            print(json.dumps(result), flush=True)
            failed |= result["status"] in ("INFRA_ERROR", "UNSTABLE", "INTERRUPTED")
    except BaseException:
        stop.set()
        raise
    return int(failed)


def _child(*args):
    # The launcher handles terminal Ctrl-C, then sends one SIGTERM per worker.
    # Avoid a second interruption cutting short simulator cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(_drain(*args))


def run_workers(campaign, candidates, runtime, gpu, *, tasks_per_gpu=1, max_jobs=None, probe=False):
    if tasks_per_gpu not in (1, 2, 4):
        raise ValueError("tasks_per_gpu must be 1, 2 or 4")
    maximum = len(candidates) if max_jobs is None else max_jobs
    if maximum < 0:
        raise ValueError("max_jobs must not be negative")
    # Linux simulation workers fork before creating any simulator/model processes.
    # Inherited lock descriptors also keep the reservation alive during cleanup.
    ctx = multiprocessing.get_context("fork")
    claimed, gate, stop = ctx.Value("i", 0), ctx.Lock(), ctx.Event()
    path = Path(runtime.get("gpu_lock_directory", "/tmp")) / (
        "codeworld-benchmark-gpu-" + str(gpu).replace("/", "_") + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as reservation:
        # Same lock as older versions: another launcher cannot add extra slots.
        fcntl.flock(reservation, fcntl.LOCK_EX)
        print(json.dumps({"gpu": str(gpu), "tasks_per_gpu": tasks_per_gpu}), flush=True)
        common = (maximum, claimed, gate, stop, probe)
        if tasks_per_gpu == 1:
            code = _drain(campaign, candidates, runtime, gpu, 0, *common)
        else:
            children = []
            try:
                for slot in range(min(tasks_per_gpu, maximum, len(candidates))):
                    child = ctx.Process(target=_child, args=(campaign, candidates, runtime, gpu, slot, *common))
                    child.start()
                    children.append(child)
                for child in children:
                    child.join()
                code = int(any(child.exitcode != 0 for child in children))
            except BaseException:
                stop.set()
                # SIGTERM follows execute_job's existing interruption/cleanup path.
                for child in children:
                    if child.is_alive():
                        child.terminate()
                for child in children:
                    child.join()
                raise
    if not claimed.value:
        print("No pending job matches; completed attempts are preserved")
    return code
