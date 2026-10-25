import fcntl
import multiprocessing
import os
import signal
import time

import pytest

from robodojo_benchmark.campaign import Campaign
from robodojo_benchmark import scheduler


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_parallel_claims_limit_and_completed_preservation(tmp_path, monkeypatch, slots):
    ctx = multiprocessing.get_context("fork")
    active, peak = ctx.Value("i", 0), ctx.Value("i", 0)
    barrier = ctx.Barrier(slots)
    campaign = Campaign.create(tmp_path / "campaign", {
        "tasks": ["insert_tubes"], "samples": 12, "strategies": ["ours"]})

    def execute(c, job, runtime, gpu, **kwargs):
        assert gpu == "3"
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
        barrier.wait(timeout=10)
        time.sleep(0.02)
        return {"status": "SCORED"}

    def finish(c, job, result):
        c.finish(job, result)
        with active.get_lock():
            active.value -= 1

    monkeypatch.setattr(scheduler, "execute_job", execute)
    monkeypatch.setattr(scheduler, "finish_job", finish)
    runtime = {"gpu_lock_directory": str(tmp_path)}
    assert scheduler.run_workers(campaign, list(campaign.jobs), runtime, "3",
                                 tasks_per_gpu=slots, max_jobs=8) == 0
    assert peak.value == slots
    rows = campaign.rows()
    assert sum(r["status"] == "SCORED" for r in rows) == 8
    assert sum(r["status"] == "PENDING" for r in rows) == 4
    assert all(r["attempt"] <= 1 for r in rows)
    assert scheduler.run_workers(campaign, list(campaign.jobs), runtime, "3",
                                 tasks_per_gpu=slots) == 0
    assert all(r["status"] == "SCORED" and r["attempt"] == 1 for r in campaign.rows())


def test_infra_error_stops_claims_and_returns_failure(tmp_path, monkeypatch):
    campaign = Campaign.create(tmp_path / "campaign", {
        "tasks": ["insert_tubes"], "samples": 12, "strategies": ["ours"]})
    monkeypatch.setattr(scheduler, "execute_job", lambda *a, **kw: {"status": "INFRA_ERROR"})
    monkeypatch.setattr(scheduler, "finish_job", lambda c, j, r: c.finish(j, r))
    assert scheduler.run_workers(campaign, list(campaign.jobs),
                                 {"gpu_lock_directory": str(tmp_path)}, "0", tasks_per_gpu=4) == 1
    rows = campaign.rows()
    assert 1 <= sum(r["status"] == "INFRA_ERROR" for r in rows) <= 4
    assert not any(r["status"] == "RUNNING" for r in rows)


def test_existing_gpu_lock_blocks_new_pool(tmp_path, monkeypatch):
    ctx = multiprocessing.get_context("fork")
    entered = ctx.Event()
    campaign = Campaign.create(tmp_path / "campaign", {
        "tasks": ["insert_tubes"], "samples": 1, "strategies": ["ours"]})

    def execute(*a, **kw):
        entered.set()
        return {"status": "SCORED"}

    monkeypatch.setattr(scheduler, "execute_job", execute)
    monkeypatch.setattr(scheduler, "finish_job", lambda c, j, r: c.finish(j, r))
    with (tmp_path / "codeworld-benchmark-gpu-0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child = ctx.Process(target=scheduler.run_workers, args=(campaign, list(campaign.jobs),
                            {"gpu_lock_directory": str(tmp_path)}, "0"), kwargs={"tasks_per_gpu": 4})
        child.start()
        try:
            assert not entered.wait(0.2)
            fcntl.flock(lock, fcntl.LOCK_UN)
            assert entered.wait(10)
            child.join(10)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.kill()
                child.join()


def test_stop_launcher_finishes_active_workers_and_releases_gpu(tmp_path, monkeypatch):
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Barrier(3)
    campaign = Campaign.create(tmp_path / "campaign", {
        "tasks": ["insert_tubes"], "samples": 6, "strategies": ["ours"]})

    def execute(*args, **kwargs):
        try:
            ready.wait(timeout=10)
            time.sleep(30)
        except KeyboardInterrupt:
            return {"status": "INTERRUPTED"}
        raise AssertionError("Expected termination")

    def launch():
        signal.signal(signal.SIGTERM, scheduler._interrupt)
        try:
            scheduler.run_workers(campaign, list(campaign.jobs),
                                  {"gpu_lock_directory": str(tmp_path)}, "0", tasks_per_gpu=2)
        except KeyboardInterrupt:
            pass

    monkeypatch.setattr(scheduler, "execute_job", execute)
    monkeypatch.setattr(scheduler, "finish_job", lambda c, j, r: c.finish(j, r))
    child = ctx.Process(target=launch)
    child.start()
    try:
        ready.wait(timeout=10)
        os.kill(child.pid, signal.SIGTERM)
        child.join(10)
        assert child.exitcode == 0
        rows = campaign.rows()
        assert sum(r["status"] == "INTERRUPTED" for r in rows) == 2
        assert sum(r["status"] == "PENDING" for r in rows) == 4
        with (tmp_path / "codeworld-benchmark-gpu-0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
