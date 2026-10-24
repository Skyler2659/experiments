"""Immutable campaign manifest, exclusive claims, and append-only attempts."""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import uuid
from pathlib import Path
from .catalog import build_manifest, read


def write(path, value):
    from codeworld_robot.io import atomic
    atomic(Path(path), value)


class Campaign:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read(self.root / "manifest.json")
        self.jobs = {j["job_id"]: j for j in self.manifest["jobs"]}

    @classmethod
    def create(cls, root, spec):
        manifest = build_manifest(spec)
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        write(root / "manifest.json", manifest)
        with sqlite3.connect(root / "queue.sqlite") as db:
            db.executescript("""
                CREATE TABLE jobs (job_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0, owner TEXT, updated REAL, result TEXT);
                CREATE TABLE attempts (job_id TEXT, attempt INTEGER, owner TEXT, status TEXT,
                    started REAL, finished REAL, result TEXT, PRIMARY KEY(job_id, attempt));
                CREATE TABLE events (time REAL, job_id TEXT, kind TEXT, detail TEXT);
            """)
            db.executemany("INSERT INTO jobs(job_id,status) VALUES (?, 'PENDING')",
                           [(j["job_id"],) for j in manifest["jobs"]])
        return cls(root)

    def connect(self):
        db = sqlite3.connect(self.root / "queue.sqlite", timeout=60)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=60000")
        return db

    def claim(self, candidates):
        owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:10]}"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for job_id in candidates:
                row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if row is None or row["status"] != "PENDING":
                    continue
                attempt, now = row["attempt"] + 1, time.time()
                db.execute("UPDATE jobs SET status='RUNNING',attempt=?,owner=?,updated=? WHERE job_id=?",
                           (attempt, owner, now, job_id))
                db.execute("INSERT INTO attempts VALUES(?,?,?,'RUNNING',?,NULL,NULL)",
                           (job_id, attempt, owner, now))
                return {**self.jobs[job_id], "attempt": attempt, "owner": owner}
        return None

    def finish(self, job, result):
        status = result["status"]
        if status not in ("SCORED", "METHOD_FAILURE", "INFRA_ERROR", "UNSTABLE", "INTERRUPTED"):
            raise ValueError("Invalid terminal state")
        payload = json.dumps(result, allow_nan=False)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)).fetchone()
            if row["owner"] != job["owner"] or row["status"] != "RUNNING":
                raise ValueError("Claim no longer owned by this worker")
            db.execute("UPDATE jobs SET status=?,updated=?,result=? WHERE job_id=?",
                       (status, time.time(), payload, job["job_id"]))
            db.execute("UPDATE attempts SET status=?,finished=?,result=? WHERE job_id=? AND attempt=?",
                       (status, time.time(), payload, job["job_id"], job["attempt"]))

    def retry(self, job_id, reason):
        if not reason.strip():
            raise ValueError("A retry needs an infrastructure reason")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in ("INFRA_ERROR", "INTERRUPTED"):
                raise ValueError("Only infrastructure/interrupted attempts can be retried; scored/method failures are final")
            attempt = self.root / 'runs' / job_id / f"attempt_{row['attempt']:03d}"
            if (attempt / 'evaluator/_result.json').exists():
                raise ValueError('Native result exists; recover its score instead of retrying the episode')
            db.execute("UPDATE jobs SET status='PENDING',owner=NULL,result=NULL WHERE job_id=?", (job_id,))
            db.execute("INSERT INTO events VALUES(?,?,?,?)", (time.time(), job_id, "retry", reason))

    def recover(self, job_id, reason):
        # Explicit operator action after checking the recorded host/PID is gone.
        if not reason.strip():
            raise ValueError("Explain why the recorded worker is no longer running")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in ('RUNNING', 'INFRA_ERROR', 'INTERRUPTED'):
                raise ValueError("Only RUNNING or failed infrastructure jobs need recovery")
            from .results import saved_native_result
            attempt = self.root / 'runs' / job_id / f"attempt_{row['attempt']:03d}"
            native = saved_native_result(attempt, self.jobs[job_id])
            if row['status'] != 'RUNNING' and native is None:
                raise ValueError('No native score to recover; use retry for infrastructure failures')
            # Recover the already-persisted outcome if the worker died between
            # writing result.json and committing SQLite (including older exports).
            result_path = self.root / 'runs' / job_id / f"attempt_{row['attempt']:03d}" / 'result.json'
            if result_path.is_file() or native is not None:
                saved = read(result_path) if result_path.is_file() else {}
                if native is not None:
                    if saved.get('status') not in (None, 'SCORED'):
                        saved['recovered_previous_status'] = saved['status']
                    saved.update(native)
                    write(result_path, saved)
                status = saved.get('status')
                if status not in ('SCORED', 'METHOD_FAILURE', 'INFRA_ERROR', 'UNSTABLE', 'INTERRUPTED'):
                    raise ValueError('Saved result has an invalid terminal state; inspect before recovering')
                payload = json.dumps(saved, allow_nan=False)
                now = time.time()
                db.execute('UPDATE jobs SET status=?,updated=?,result=? WHERE job_id=?',
                           (status, now, payload, job_id))
                db.execute('UPDATE attempts SET status=?,finished=?,result=? WHERE job_id=? AND attempt=?',
                           (status, now, payload, job_id, row['attempt']))
                db.execute('INSERT INTO events VALUES(?,?,?,?)', (now, job_id, 'recover_saved_result', reason))
                return
            db.execute("UPDATE jobs SET status='INTERRUPTED',updated=? WHERE job_id=?", (time.time(), job_id))
            db.execute("UPDATE attempts SET status='INTERRUPTED',finished=? WHERE job_id=? AND attempt=?",
                       (time.time(), job_id, row["attempt"]))
            db.execute("INSERT INTO events VALUES(?,?,?,?)", (time.time(), job_id, "recover", reason))

    def rows(self):
        with self.connect() as db:
            return [{**self.jobs[r["job_id"]], **dict(r), "result": json.loads(r["result"]) if r["result"] else None}
                    for r in db.execute("SELECT * FROM jobs ORDER BY rowid")]

    def select(self, strategy=None, task=None, sample=None, shard=None):
        jobs = [j for j in self.manifest["jobs"] if (not strategy or j["strategy"] == strategy)
                and (not task or j["task"] == task) and (sample is None or j["sample"] == sample)]
        if not jobs:
            raise ValueError("No job matches the selector")
        if shard:
            index, count = (int(v) for v in shard.split("/"))
            if not 0 <= index < count:
                raise ValueError("shard must be INDEX/COUNT, with 0 <= INDEX < COUNT")
            jobs = [j for i, j in enumerate(jobs) if i % count == index]
        return [j["job_id"] for j in jobs]
