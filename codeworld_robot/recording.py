"""Passive frame timeline. States are held at their last observed revision."""
import json
from pathlib import Path

class FrameTimeline:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / 'robot_io'
        (self.directory / 'world_snapshots').mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'timeline.jsonl'
        self.index = 0
        self.operation = None
        self._stamp = None
        self.snapshot = None

    def record(self, clock):
        state = self.root / 'state/current.json'
        if state.exists():
            stamp = state.stat().st_mtime_ns
            if stamp != self._stamp:
                payload = json.loads(state.read_text())
                target = self.directory / 'world_snapshots' / f'{self.index:08d}.json'
                target.write_text(json.dumps(payload))
                self.snapshot = str(target.relative_to(self.directory))
                self._stamp = stamp
        row = dict(frame=self.index, clock=clock, operation=self.operation, world_snapshot=self.snapshot)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, default=str) + '\n')
        self.index += 1
