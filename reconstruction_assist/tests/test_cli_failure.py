"""A failed CLI must never be hidden by an old valid scene delivery."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

class CliFailureTest(unittest.TestCase):
    def test_new_and_resumed_cli_fail_before_fit_or_reusing_old_delivery(self):
        for resume in [False,True]:
            with self.subTest(resume=resume),tempfile.TemporaryDirectory() as d:
                root=Path(d);nd=root/'runs/test/fractal/object';nd.mkdir(parents=True)
                (nd/'part.json').write_text(json.dumps({'module':'component.js','children':[]}))
                (nd/'component.js').write_text('export function build(){}')
                (nd/'fit_requests.json').write_text('{"requests":[]}')
                if resume:(nd/'.sid').write_text('abcdef')
                bindir=root/'bin';bindir.mkdir()
                codex=bindir/'codex';codex.write_text('#!/bin/bash\necho "Missing environment variable" >&2\nexit 77\n');codex.chmod(0o755)
                hook=bindir/'hook';hook.write_text('#!/bin/bash\ntouch "'+str(root/'unexpected_review')+'"\nexit 0\n');hook.chmod(0o755)
                env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
                     'RCWM_ROOT':str(root),'RCWM_CODE':str(ROOT/'vendor/RecursiveCWM'),
                     'RCWM_FEEDBACK_HOOK':str(hook),'RCWM_ADAPTIVE_DISPATCH':'0','RCWM_MAXCYC':'1'}
                r=subprocess.run(['bash',str(ROOT/'vendor/RecursiveCWM/runner/solve_recursive.sh'),
                                  'runs/test','object','scene','1','test'],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,77,r.stderr)
                events=[json.loads(x) for x in (root/'runs/test/trace/events.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['event'],'session_error')
                self.assertEqual(events[-1]['returncode'],77)
                self.assertFalse(any(x['event']=='session_end' for x in events))
                self.assertFalse((root/'unexpected_review').exists())
                self.assertTrue((nd/'part.json').exists())

if __name__=='__main__':unittest.main()
