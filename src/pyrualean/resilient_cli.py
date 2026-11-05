"""Retry provider transport inside the live episode, retaining its Codex session.

This is not simulator crash recovery. The parent owns the simulator and MCP
worker; this wrapper only replaces the failed planner process after tools drain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid


CONTINUE = (
    "The provider connection was interrupted. Continue this SAME session and live episode. "
    "The host, simulator, Python namespace, completed actions and remaining budgets were retained. "
    "A tool may have completed while its reply was lost. Inspect the current observations, "
    "CodeWorld and action history before deciding the next action. Never reset the scene or "
    "automatically replay an earlier tool call. Complete any pending world update first. "
    "For reconstruction, inspect the existing files and continue from the last completed work."
)


def classify(text):
    text = text.lower()
    statuses = set(re.findall(r'(?:http(?:/\d(?:\.\d)?)?\s+|(?:status(?:\s+code)?|status_code)[\s:=\"\']+)([45]\d\d)\b', text))
    if '429' in statuses or any(x in text for x in ("usage limit", "usage_limit", "quota exceeded", "quota_exceeded", "insufficient_quota", "exceeded your current quota", "rate limit", "rate_limit", "too many requests", "at capacity")):
        return "quota_or_capacity"
    if '401' in statuses or any(x in text for x in ("unauthorized", "authentication", "token expired", "refresh token")):
        return "authentication"
    if statuses.intersection({'502', '503', '504'}) or any(x in text for x in ("stream disconnected", "connection reset", "connection refused", "connection closed", "network", "timed out", "timeout", "failed to send", "error sending request", "transport error", "service unavailable", "internal server error")):
        return "transport"
    return None


def session_id(text):
    found = re.findall(r'(?:session id:\s*|"thread_id"\s*:\s*")([0-9a-f-]{36})', text, re.I)
    return found[-1] if found else None


def resume_args(args, sid):
    existing = "resume" in args
    options = list(args[1:-2] if existing else args[1:-1])
    options = [x for x in options if x != "resume"]
    kept = []; i = 0
    while i < len(options):
        if options[i] in ("-s", "--sandbox"):
            # exec resume has no --sandbox flag; preserve the mode via config.
            kept.extend(["-c", "sandbox_mode=" + json.dumps(options[i+1])])
            i += 2
        elif options[i] in ("-C", "--cd", "-i", "--image"):
            i += 2
        else:
            kept.append(options[i]); i += 1
    return ["exec", "resume", *kept, sid, "-"]


def waiting_seconds(directory, since, now=None):
    """Union of concurrent provider waits; never multiply by child count."""
    if not directory:
        return 0.0
    now = time.time() if now is None else now
    intervals = []
    for path in Path(directory).glob('*.events.jsonl'):
        start = None
        try:
            for line in path.read_text().splitlines():
                try: row = json.loads(line)
                except ValueError: continue
                if row.get('event') == 'wait_start': start = row['time']
                elif row.get('event') == 'wait_end' and start is not None:
                    intervals.append((max(since, start), min(now, row['time']))); start = None
            if start is not None:
                # Dead wrappers must not stop a later stage's clock indefinitely.
                state = json.loads(path.with_name(path.name.replace('.events.jsonl', '.json')).read_text())
                live = False
                try:
                    os.kill(state['pid'], 0)
                    live = now-state['time'] < 10 and state.get('status') in ('WAITING_PROVIDER','WAITING_TOOL','NEEDS_ATTENTION')
                except (OSError, KeyError): pass
                intervals.append((max(since, start), now if live else min(now, state['time'])))
        except (OSError, ValueError, KeyError):
            continue
    end = since; total = 0.0
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, end)); end = max(end, b)
    return total


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / (str(os.getpid()) + '-' + uuid.uuid4().hex[:8] + '.json')
        self.events = self.path.with_suffix('.events.jsonl')
        self.state = dict(pid=os.getpid(), cwd=os.getcwd(), attempt=0)
        self.lock = threading.Lock()

    def update(self, **data):
        with self.lock:
            self.state.update(data, time=time.time())
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.state)); temp.replace(self.path)

    def event(self, event, **data):
        with self.events.open('a') as stream:
            stream.write(json.dumps(dict(event=event, time=time.time(), **data)) + '\n')

    def wait(self, seconds, *, status, control=None):
        self.event('wait_start', reason=status)
        deadline = time.monotonic() + seconds
        try:
            while True:
                active = False
                if control:
                    try: active = bool(json.loads(Path(control).read_text()).get('active'))
                    except (OSError, ValueError): active = True
                self.update(status='WAITING_TOOL' if active else status,
                            retry_at=time.time() + max(0, deadline-time.monotonic()))
                if not active and time.monotonic() >= deadline: break
                time.sleep(min(1.0, max(0.01, deadline-time.monotonic())) if not active else 1.0)
        finally:
            self.event('wait_end')


def run(binary, args, *, input_data=b'', directory, control=None, base_delay=15, max_delay=900):
    journal = Journal(directory)
    current = list(args); sid = None; attempt = 0
    if 'resume' in args: sid = args[-2]
    while True:
        attempt += 1
        journal.update(status='RUNNING', attempt=attempt, session_id=sid, retry_at=None)
        captured = [[], []]
        observed_sid = [sid]
        proc = subprocess.Popen([binary, *current], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        journal.update(client_pid=proc.pid)
        def drain(source, target, parts):
            for line in iter(source.readline, b''):
                target.write(line); target.flush(); parts.append(line)
                found_sid = session_id(line.decode(errors="replace"))
                if found_sid and found_sid != observed_sid[0]:
                    observed_sid[0] = found_sid
                    journal.update(session_id=found_sid)
                if sum(map(len, parts)) > 524288: del parts[:len(parts)//2]
        streams = [threading.Thread(target=drain, args=(proc.stdout, sys.stdout.buffer, captured[0])),
                   threading.Thread(target=drain, args=(proc.stderr, sys.stderr.buffer, captured[1]))]
        for t in streams: t.start()
        try:
            proc.stdin.write(input_data); proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = proc.wait()
        for t in streams: t.join()
        output = b''.join(captured[0]).decode(errors='replace')
        diagnostic = output + b''.join(captured[1]).decode(errors='replace')
        sid = observed_sid[0] or session_id(diagnostic) or sid
        errors = []; terminal = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
                if event.get('type') in ('turn.failed', 'error'): errors.append(line)
                if event.get('type') in ('turn.failed', 'turn.completed'): terminal.append(event['type'])
            except (ValueError, AttributeError): pass
        failed = rc != 0 or (terminal[-1] == 'turn.failed' if terminal else bool(errors))
        if not errors:
            fatal = [line for line in diagnostic.splitlines() if line.startswith('ERROR:')]
            errors = fatal[-1:] or [line for line in diagnostic.splitlines() if 'codex_models_manager' not in line and any(x in line.lower() for x in ('error', 'failed', 'limit', 'capacity', 'disconnected'))][-3:]
        reason = classify('\n'.join(errors)) if failed else None
        journal.event('cli_exit', returncode=rc, session_id=sid, reason=reason)
        if not reason:
            journal.update(status='COMPLETED' if not failed else 'FAILED', returncode=rc, session_id=sid)
            return rc if rc else (1 if failed else 0)
        # Existing in-flight tools are owned by the host and finish exactly once.
        # Never issue another planner turn until the guard reports idle.
        delay = min(max_delay, base_delay * 2 ** min(attempt-1, 8))
        journal.update(reason=reason, session_id=sid)
        journal.wait(delay, status='WAITING_PROVIDER', control=control)
        if sid:
            current = resume_args(current, sid); input_data = CONTINUE.encode()
        elif 'mcp_tool_call' in output or 'command_execution' in output:
            # Losing session identity after work is not permission to start fresh.
            journal.update(status='NEEDS_ATTENTION', reason='missing_session_after_tool')
            while not (journal.directory/'ALLOW_FRESH_SESSION').exists():
                journal.wait(30, status='NEEDS_ATTENTION', control=control)


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1], sys.argv[2:], input_data=sys.stdin.buffer.read(),
                         directory=os.environ['PYRUA_RECOVERY_DIR'],
                         control=os.environ.get('PYRUA_TOOL_CONTROL')))
