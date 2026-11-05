import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

MODULE = Path(__file__).resolve().parents[1]/'src/pyrualean/resilient_cli.py'
spec = importlib.util.spec_from_file_location('resilient_cli', MODULE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ResilienceTest(unittest.TestCase):
    def test_resume_preserves_model_mcp_and_existing_session(self):
        a = ['exec','-m','gpt-6-astra','-c','mcp_servers.pyrualean.url="http://live"','-C','/ws','-i','old.png','-s','workspace-write','-']
        b = module.resume_args(a,'session')
        self.assertEqual(b[-2:], ['session','-'])
        self.assertIn('gpt-6-astra', b); self.assertIn(a[4], b)
        self.assertNotIn('old.png', b)
        self.assertEqual(module.resume_args(b,'session'), b)

    def test_classifier_does_not_retry_programming_errors(self):
        self.assertEqual(module.classify('You have hit your usage limit'), 'quota_or_capacity')
        self.assertEqual(module.classify('stream disconnected before completion'), 'transport')
        self.assertIsNone(module.classify('TypeError: geometry.split is not a function'))

    def test_wait_intervals_are_union_not_sum(self):
        with tempfile.TemporaryDirectory() as d:
            for i,(a,b) in enumerate([(10,30),(20,40)]):
                (Path(d)/f'{i}.events.jsonl').write_text(json.dumps(dict(event='wait_start',time=a))+'\n'+json.dumps(dict(event='wait_end',time=b))+'\n')
            self.assertEqual(module.waiting_seconds(d,15,50),25)

    def test_inflight_tool_drains_before_resume(self):
        with tempfile.TemporaryDirectory() as d:
            control=Path(d)/'control.json';control.write_text('{"active":{"id":17}}')
            timer=threading.Timer(.15,lambda: control.write_text('{"active":null}'));timer.start()
            journal=module.Journal(Path(d)/'recovery');start=time.monotonic()
            journal.wait(.01,status='WAITING_PROVIDER',control=control)
            timer.join()
            self.assertGreaterEqual(time.monotonic()-start,.15)
            self.assertIsNone(json.loads(control.read_text())['active'])

    def test_disconnect_after_action_resumes_without_replaying_action(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);fake=p/'fake';counter=p/'actions';trace=p/'trace'
            fake.write_text('#!'+sys.executable+'\n'+f'''
import sys,json,pathlib
p=pathlib.Path({d!r}); a=sys.argv[1:]; text=sys.stdin.read()
with (p/'trace').open('a') as f: f.write(json.dumps(a)+'\\n')
sid='12345678-1234-1234-1234-123456789abc'
if not (p/'actions').exists():
 (p/'actions').write_text('1')
 print(json.dumps(dict(type='thread.started',thread_id=sid)),flush=True)
 print(json.dumps(dict(type='turn.failed',error=dict(message='stream disconnected before completion'))),flush=True)
 sys.exit(1)
assert a[-2:]==[sid,'-'] and a.count('resume')==1
assert 'Never reset' in text and (p/'actions').read_text()=='1'
print(json.dumps(dict(type='turn.completed',usage=dict(input_tokens=1))),flush=True)
''');fake.chmod(0o755)
            harness=p/'test.py'
            harness.write_text(f'''import importlib.util
s=importlib.util.spec_from_file_location('r',{str(MODULE)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
raise SystemExit(m.run({str(fake)!r},['exec','--json','-'],input_data=b'original',directory={str(p/'recovery')!r},base_delay=.01,max_delay=.01))
''')
            r=subprocess.run([sys.executable,str(harness)],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(counter.read_text(),'1');self.assertEqual(len(trace.read_text().splitlines()),2)
            receipt=json.loads(next((p/'recovery').glob('*.json')).read_text())
            self.assertEqual(receipt['status'],'COMPLETED');self.assertEqual(receipt['attempt'],2)


if __name__ == '__main__': unittest.main()
