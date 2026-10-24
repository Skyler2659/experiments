import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from codeworld_robot.io import atomic, read
from codeworld_robot.ownership import authorize
ROOT = Path(__file__).resolve().parents[2]

class OwnershipTest(unittest.TestCase):
    def test_cli_waits_for_robot_before_claiming_episode(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'episode';episode.mkdir()
            atomic(episode/'task/executor.json', {'status':'UNASSIGNED'})
            config=root/'config.json';atomic(config, {'episode':str(episode),'python':sys.executable,'ready_timeout_s':10})
            fake=root/'fake-codex'
            fake.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("-C")+1]);c=json.loads((p/"transport.json").read_text());e=Path(c["episode"]);assert (e/"robot_io/ready.json").is_file();(e/"robot_io/closed.json").write_text("{}");print(json.dumps({"type":"thread.started","thread_id":"ready-fixture"}))\n');fake.chmod(0o755)
            worker=subprocess.Popen([sys.executable,'-m','codeworld_robot.agent_cli','run','--config',str(config),'--run-dir',str(root/'run'),'--model','fixture','--codex',str(fake)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if (root/'run/lifecycle.json').exists() and read(root/'run/lifecycle.json')['status']=='WAITING_FOR_ROBOT':break
                    time.sleep(.05)
                else:self.fail('Never waited for robot')
                self.assertEqual(read(episode/'task/executor.json')['status'],'UNASSIGNED')
                bridge=root/'run/remote.py'
                help_result=subprocess.run([sys.executable,str(bridge),'--help'],capture_output=True,text=True)
                self.assertEqual(help_result.returncode,0,help_result.stderr)
                for args in ([],['-']):
                    result=subprocess.run([sys.executable,str(bridge),*args],input='from pathlib import Path\nprint(Path.cwd())',capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(Path(result.stdout.strip()).resolve(),episode.resolve())
                atomic(episode/'task/robot_connection.json',{})
                atomic(episode/'robot_io/ready.json',{})
                _,err=worker.communicate(timeout=10)
                self.assertEqual(worker.returncode,0,err)
                self.assertEqual(read(root/'run/lifecycle.json')['thread_id'],'ready-fixture')
            finally:
                if worker.poll() is None:worker.kill();worker.wait()

    def test_remote_python_wrapper_cannot_change_episode_working_directory(self):
        from codeworld_robot.agent_cli import execute
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'episode';episode.mkdir()
            ssh=root/'fake-ssh';ssh.write_text('#!/bin/sh\nexec /bin/sh -c "$1"\n');ssh.chmod(0o755)
            wrapper=root/'python-wrapper';wrapper.write_text('#!/bin/sh\ncd /\nexec '+sys.executable+' "$@"\n');wrapper.chmod(0o755)
            result=execute({'episode':str(episode),'python':str(wrapper),'ssh':[str(ssh)]},'from pathlib import Path\nprint(Path.cwd())','fixture')
            self.assertEqual(Path(result.strip()).resolve(),episode.resolve())
    def test_only_owner_can_act_and_supervisor_can_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic(root/'task/executor.json', {'status':'ACTIVE','run_id':'right'})
            for source in ({}, {'role':'supervisor_stop'}, {'role':'codex_cli','run_id':'wrong'}):
                with self.assertRaises(PermissionError): authorize(root, {'op':'move_tcp','actor':source})
            authorize(root, {'op':'move_tcp','actor':{'role':'codex_cli','run_id':'right'}})
            authorize(root, {'op':'finish','actor':{'role':'supervisor_stop'}})
            atomic(root/'task/executor.json', {'status':'INACTIVE','run_id':'right'})
            with self.assertRaises(PermissionError):
                authorize(root, {'op':'move_tcp','actor':{'role':'codex_cli','run_id':'right'}})

    def test_cli_lifecycle_and_no_relabel_of_existing_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'episode';episode.mkdir()
            atomic(episode/'task/executor.json', {'status':'UNASSIGNED'})
            config=root/'config.json';atomic(config, {'episode':str(episode),'python':sys.executable,'wait_for_ready':False})
            fake=root/'fake-codex'
            fake.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
from pathlib import Path
args=sys.argv
assert args[args.index('model_reasoning_effort="max"')-1]=='-c'
run=Path(args[args.index('-C')+1])
cfg=json.loads((run/'transport.json').read_text())
assert os.environ['CODEWORLD_ROLE']=='codex_cli'
owner=json.loads((Path(cfg['episode'])/'task/executor.json').read_text())
assert owner['run_id']==os.environ['CODEWORLD_EXECUTOR_ID']
closed=Path(cfg['episode'])/'robot_io/closed.json'
closed.parent.mkdir();closed.write_text('{}')
Path(args[args.index('-o')+1]).write_text('Fixture only; no robot connected')
print(json.dumps({'type':'thread.started','thread_id':'fixture-thread'}))
''');fake.chmod(0o755)
            command=[sys.executable,'-m','codeworld_robot.agent_cli','run','--config',str(config),
                     '--run-dir',str(root/'run'),'--model','fixture-model','--reasoning-effort','max','--codex',str(fake)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            state=read(root/'run/lifecycle.json')
            self.assertEqual(state['thread_id'],'fixture-thread')
            self.assertEqual(state['status'],'EXITED')
            self.assertEqual(state['reasoning_effort'],'max')
            self.assertNotIn('cleanup_error',state)
            self.assertEqual(read(episode/'task/executor.json')['status'],'INACTIVE')
            command[command.index('--run-dir')+1]=str(root/'second')
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Closed episode',result.stderr)

    def test_stop_terminates_cli_and_revokes_owner(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'episode';episode.mkdir()
            atomic(episode/'task/executor.json', {'status':'UNASSIGNED'})
            config=root/'config.json';atomic(config, {'episode':str(episode),'python':sys.executable,'wait_for_ready':False})
            fake=root/'fake-codex'
            fake.write_text('#!'+sys.executable+'\nimport time\ntime.sleep(60)\n');fake.chmod(0o755)
            command=[sys.executable,'-m','codeworld_robot.agent_cli','run','--config',str(config),
                     '--run-dir',str(root/'run'),'--model','fixture-model','--codex',str(fake)]
            worker=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    statefile=root/'run/lifecycle.json'
                    if statefile.exists() and read(statefile)['status']=='RUNNING': break
                    time.sleep(.05)
                else: self.fail('Worker never started')
                # Stand-in for an already stopped simulator; no robot is connected.
                atomic(episode/'robot_io/closed.json', {})
                subprocess.run([sys.executable,'-m','codeworld_robot.agent_cli','stop','--run-dir',str(root/'run')],cwd=ROOT,check=True)
                _,err=worker.communicate(timeout=10)
                self.assertEqual(worker.returncode,0,err)
                self.assertEqual(read(root/'run/lifecycle.json')['status'],'STOPPED')
                self.assertEqual(read(episode/'task/executor.json')['status'],'INACTIVE')
            finally:
                if worker.poll() is None: worker.kill(); worker.wait()

if __name__=='__main__': unittest.main()
