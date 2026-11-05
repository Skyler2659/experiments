import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from pyrualean.mcp_server import _execute_tool


def test_rpc_cancellation_does_not_abandon_worker_or_allow_overlap():
    entered, release = threading.Event(), threading.Event()

    class Toolkit:
        def execute_tool(self, name, arguments):
            entered.set()
            assert release.wait(5)
            return "completed"

    async def scenario():
        lock = asyncio.Lock()

        async def call():
            async with lock:
                return await _execute_tool(Toolkit(), "python", {})

        task = asyncio.create_task(call())
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert lock.locked()
        release.set()
        assert await task == "completed"
        assert not lock.locked()

    asyncio.run(scenario())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process guard")
@pytest.mark.parametrize("mode", ["timeout", "memory", "child_memory", "normal"])
def test_real_process_limits_preserve_unrelated_process(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    script = r'''
import json,sys,time,subprocess
from pathlib import Path
from pyrualean.execution_guard import ExecutionGuard
folder=Path(sys.argv[1]);mode=sys.argv[2]
g=ExecutionGuard(folder, tool_timeout_s=.5 if mode=='timeout' else 10,
                 host_memory_mb=70,tree_memory_mb=100)
g.start()
g.begin('python')
if mode=='normal':
 g.end();time.sleep(.8);g.stop();sys.exit(0)
if mode=='child_memory':
 child=subprocess.Popen([sys.executable,'-c','import time; x=bytearray(160*1024*1024); time.sleep(20)'])
 (folder/'child.pid').write_text(str(child.pid))
elif mode=='memory':
 x=bytearray(160*1024*1024)
time.sleep(20)
'''
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
    proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), mode], env=env)
    try:
        code = proc.wait(timeout=12)
        assert other.poll() is None
        receipt = tmp_path / "interruption.json"
        if mode == "normal":
            assert code == 0
            assert not receipt.exists()
        else:
            assert code == -signal.SIGKILL
            data = json.loads(receipt.read_text())
            assert data["status"] == "resource_interrupted"
            assert data["reason"] == {"timeout": "tool_wall_timeout",
                                      "memory": "host_memory_limit",
                                      "child_memory": "episode_memory_limit"}[mode]
            assert not data["evaluation_failure"]
            if mode == "child_memory":
                from pyrualean.execution_guard import process_info
                # Watcher kills the host first, then its captured descendants.
                import time
                pid = int((tmp_path / "child.pid").read_text())
                deadline = time.monotonic() + 2
                while process_info(pid) is not None and time.monotonic() < deadline:
                    time.sleep(.02)
                assert process_info(pid) is None
    finally:
        for p in [proc, other]:
            if p.poll() is None:
                p.kill()
            p.wait()
