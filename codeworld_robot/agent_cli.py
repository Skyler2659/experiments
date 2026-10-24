"""Independent CLI lifecycle. No manipulation policy lives in this module."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid
from .io import atomic, read


def execute(config, script, run_id):
    env = {'CODEWORLD_ROLE': 'codex_cli', 'CODEWORLD_EXECUTOR_ID': run_id,
           'PYTHONPATH': str(Path(config['episode']) / 'tools')}
    try:
        if config.get('ssh'):
            # Simulator Python wrappers may cd to their own installation directory.
            # Restore the episode inside Python before resolving any relative path.
            bootstrap = 'import os,sys; os.chdir(' + repr(config['episode']) + '); exec(compile(sys.stdin.read(), "<episode>", "exec"))'
            command = 'cd ' + shlex.quote(config['episode']) + ' && env ' + ' '.join(
                shlex.quote(k + '=' + v) for k, v in env.items()) + ' ' + shlex.quote(config['python']) + ' -c ' + shlex.quote(bootstrap)
            return subprocess.run(config['ssh'] + [command], input=script, text=True, capture_output=True, check=True).stdout
        return subprocess.run([config.get('python', sys.executable), '-'], cwd=config['episode'],
                              env={**os.environ, **env}, input=script, text=True,
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Episode transport failed ({exc.returncode}): {exc.stderr}') from exc


def policy(config, run_id, status, claim=False):
    script = """
from pathlib import Path
import json, os
p=Path('task/executor.json')
p.parent.mkdir(parents=True,exist_ok=True)
"""
    if claim:
        script += """
if Path('robot_io/closed.json').exists(): raise RuntimeError('Closed episode; create a fresh episode')
if not p.exists(): raise RuntimeError('Unmanaged episode; create a fresh CLI-managed episode')
if Path('world.json').exists():
 flow=Path('task/world_flow.json')
 if not flow.exists() or json.loads(flow.read_text()).get('required') is not True:
  raise RuntimeError('Independent CLI requires a fresh strict CodeWorld episode; legacy/manual episodes cannot be claimed')
old=json.loads(p.read_text())
if old.get('status')!='UNASSIGNED': raise RuntimeError('Episode already assigned; never silently reuse an assisted run')
# Atomic exclusive claim prevents two launchers from owning the same episode.
fd=os.open('task/executor.claim', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
os.close(fd)
"""
    script += "\nvalue=" + repr({'role': 'codex_cli', 'run_id': run_id, 'status': status})
    script += "\ntmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(p)\n"
    execute(config, script, run_id)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', required=True, help='JSON: episode, python, optional ssh argv')
    run.add_argument('--run-dir', required=True)
    run.add_argument('--model', required=True)
    run.add_argument('--reasoning-effort', choices=('low','medium','high','xhigh','max','ultra'))
    run.add_argument('--codex', default='codex')
    for name in ('status', 'stop'):
        sub.add_parser(name).add_argument('--run-dir', required=True)
    a = p.parse_args(); directory = Path(a.run_dir).resolve()
    if a.command == 'status':
        print(json.dumps(read(directory/'lifecycle.json'), indent=2)); return
    if a.command == 'stop':
        state = read(directory/'lifecycle.json')
        if state['status'] not in ('RUNNING','WAITING_FOR_ROBOT','STARTING'): raise RuntimeError('CLI is not running')
        # Avoid signalling a stale/reused PID: worker watches this file.
        atomic(directory/'stop.json', {'requested_at': time.time()}); return
    config = read(a.config)
    if not Path(config['episode']).is_absolute(): raise ValueError('episode must be absolute')
    directory.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    state = {'run_id': run_id, 'model': a.model, 'status': 'STARTING',
             'execution_agent': 'codex_cli', 'started_at': time.time(), 'episode': config['episode']}
    if a.reasoning_effort:state['reasoning_effort']=a.reasoning_effort
    atomic(directory/'lifecycle.json', state)
    atomic(directory/'transport.json', config)
    # Agent-authored scripts run beside the remote CodeWorld, with actor identity propagated.
    module_root = str(Path(__file__).resolve().parents[1])
    (directory/'remote.py').write_text(
        'import sys,json,argparse\nfrom pathlib import Path\nsys.path.insert(0,' + repr(module_root) + ')\n'
        'from codeworld_robot.agent_cli import execute\n'
        'parser=argparse.ArgumentParser(description="Execute your Python script in the episode workspace.")\n'
        'parser.add_argument("script",nargs="?",default="-",help="local script path; omit or use - for stdin")\n'
        'args=parser.parse_args()\n'
        'cfg=json.loads(Path(__file__).with_name("transport.json").read_text())\n'
        'script=sys.stdin.read() if args.script=="-" else Path(args.script).read_text()\n'
        'print(execute(cfg,script,' + repr(run_id) + '),end="")\n')
    instructions = '''You are the independent CodeWorld execution agent, not the supervising chat.
Read the episode AGENTS.md, task/task.json, task/ROBOT_BACKEND.md, task/EXECUTION.md and CodeWorld interfaces first.
Use remote.py to run YOUR Python scripts in the episode; it propagates your execution identity.
The run-directory filesystem and episode filesystem are distinct. Use remote.py to write
work/decision.py inside the episode before calling World.compute_decision('work/decision.py',...).
Passing an agent_runs local path does not copy that file into the episode automatically.
ALL physical decisions and judgments must be computed from CodeWorld, not image inspection.
Read task/WORLD_FLOW.md. Perception and decision are separate phases:
1. RGB-D may be used only to identify objects and estimate pose/part candidates and masks.
2. Use the reconstructed CodeWorld prior for initial planning. World.integrate_rgbd publishes
   RGB-D pose/part estimates with their fit and uncertainty. Observe/update relevant objects
   when their state may have changed or the next decision needs information that is missing.
3. Write a computation-only program compute(ctx), then World.compute_decision(program,
   purpose=...,object_ids=[...]). It queries a recorded world snapshot and returns actions
   or judgments. Coordinates, directions, grasp/placement/retreat/rotation choices,
   collision/reachability decisions, recovery, progress and task/subtask completion ALL
   follow this route. Do not inspect images in the computation program or copy image
   coordinates into commands. Thresholds, public tool dimensions and task semantics must
   be declared. Fixed tool orientations, actuator modes and skill parameters are legitimate
   policy constants; they need not numerically vary with a measurement. Normal pure Python
   helpers and comprehensions are supported. Objects may change internal configuration or local shape through interaction. Do not treat every reconstructed object as permanently rigid. Infer a useful motion/deformation hypothesis from CodeWorld, act without requiring a full joint or material model, then update only the affected parts or local shape from observations. See task/WORLD_FLOW.md; scalar parameters alone do not change geometry.
4. Robot.execute_decision(receipt['decision'], index) submits the exact calculated command;
   direct robot commands without a current world decision are rejected at server dispatch.
5. Robot motion invalidates potentially affected object state. Refresh those objects before
   relying on changed state; reuse unchanged objects. Neither every motion nor a read-only
   query requires re-fitting the full scene. Predicted/commanded poses are not measurements.
6. Physical scalar/boolean fields must use World.derive_state from a recorded CodeWorld
   computation. Never set on=True, held=True, success=True, an angle or a relation simply
   because an image looks plausible. Read numerical world queries and uncertainty.
Controller/grasp reports are stored in CodeWorld before being returned as value plus
measurement_id. This value is available for inspection without an extra CLI call. Use
ctx.measurement inside subsequent computations. Grasp geometry/lift checks are optional
diagnostics when needed, not mandatory steps for every action. Query valid derived fields
through ctx.state(object_id, field). Images are not a parallel judgment channel.
Every progress report must cite its world decision/verification and distinguish unknown
from false. A query followed by unrelated constants is not world-derived reasoning.
For multistage assembly, consider reachability of the final configuration and continued access
to objects still needed before committing early placements. A collision-free placement alone
does not establish that later grasps or releases remain possible. Validate grasp/release outcomes
from measured observations; a predicted carried pose can be wrong after slip or incomplete release.
Use only public robot observations/proprioception and reconstructed geometry. Never inspect simulator
object poses, task XML, private reward predicates, or change server/configuration/ownership files.
Author your own task programs; do not read scripts from earlier runs or ask the supervisor for actions.
Read Robot.capabilities() and obey the episode's budget. Never reset or teleport the environment.
Before the first robot motion, translate EVERY public task requirement into a named subgoal
with World.define_completion. Read codeworld_workspace.completion and the episode documentation.
Declare frames, reference objects, acceptance intervals and any ambiguity explicitly. Do not
replace "right of" with image-right or a single signed-axis check when proximity/support also
matters. Conditions are your interpretation of the public goal, never private reward predicates.
All task/subtask completion decisions MUST come from Robot.verify_completion(goal_ids), after
integrating validated RGB-D poses/states into CodeWorld. Images are evidence for updating
the world, not an alternative completion judge. Unchanged objects retain usable evidence;
a planned transform, move receipt or free-text success is not observed state.
A VERIFIED subgoal can become stale after later actions; recheck all goals at final completion.
Use Robot.finish(verification=report['verification']) only with a VERIFIED full-task report.
If conditions fail or are inconclusive, continue observation/update/planning within budget.
For physical inability, use Robot.finish(outcome='failed', reason=..., judgment=decision_path).
For runtime termination use Robot.finish(outcome='aborted', system_reason={'kind':
'timeout', 'detail':'...'}). Kinds: budget_exhausted, timeout, tool_error, process_error,
interrupted. This does not assert a physical outcome. Official benchmark scoring is separate.
On interruption stop; never claim completion merely because the CLI exits successfully.
'''
    if config.get('benchmark_protocol')=='robodojo-v1':
        from .hooks import load_hook
        instructions=load_hook(config['instruction_transform'])(instructions,config['representation'],config.get('gt_mode'))
    (directory/'AGENTS.md').write_text(instructions)
    prompt = 'Execute the goal in task/task.json of the configured episode. Start by reading its instructions through remote.py. Plan and act independently using '+('the direct RGB-D ablation interface.' if config.get('representation')=='none' else 'CodeWorld.')
    (directory/'prompt.txt').write_text(prompt)
    proc = None; claimed = False; interrupted = False
    def request_stop(*_):
        atomic(directory/'stop.json', {'requested_at': time.time()})
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    try:
        if config.get('wait_for_ready',True):
            state['status']='WAITING_FOR_ROBOT';atomic(directory/'lifecycle.json',state)
            deadline=time.monotonic()+config.get('ready_timeout_s',300)
            while True:
                if (directory/'stop.json').exists():raise InterruptedError('Stopped before robot readiness')
                ready=execute(config,"from pathlib import Path\nprint(Path('robot_io/ready.json').is_file() and Path('task/robot_connection.json').is_file())\n",run_id).strip()
                if ready=='True':break
                if time.monotonic()>=deadline:raise TimeoutError('Robot did not become ready before executor startup')
                time.sleep(3)
        policy(config, run_id, 'ACTIVE', claim=True); claimed = True
        command = [a.codex, 'exec', '--json', '--skip-git-repo-check', '-C', str(directory),
                   '-m', a.model, '-s', 'danger-full-access', '-c', 'approval_policy="never"',
                   '-c', 'shell_environment_policy.inherit="all"', '-o', str(directory/'final.txt'), prompt]
        if a.reasoning_effort:command[-1:-1]=['-c','model_reasoning_effort='+json.dumps(a.reasoning_effort)]
        if config.get('benchmark_protocol')=='robodojo-v1':
            command[-1:-1]=['--ephemeral','--disable','memories','--disable','multi_agent',
                           '--disable','plugins','--disable','apps','--disable','chronicle',
                           '--disable','skill_search','--enable','skip_host_skill_discovery',
                           '-c','memories.generate_memories=false','-c','memories.use_memories=false',
                           '-c','web_search="disabled"']
        child_env={**os.environ, 'CODEWORLD_ROLE':'codex_cli', 'CODEWORLD_EXECUTOR_ID':run_id}
        if config.get('codex_home'):child_env['CODEX_HOME']=config['codex_home']
        deadline=time.monotonic()+config.get('agent_timeout_s',float('inf'))
        with (directory/'events.jsonl').open('w') as events, (directory/'stderr.log').open('w') as err:
            proc = subprocess.Popen(command, stdout=events, stderr=err, start_new_session=True,
                env=child_env)
            state.update(status='RUNNING', pid=proc.pid); atomic(directory/'lifecycle.json',state)
            while proc.poll() is None:
                if time.monotonic()>=deadline:
                    state['termination_reason']='wall_timeout'
                    atomic(directory/'stop.json', {'requested_at':time.time(),'reason':'wall_timeout'})
                if (directory/'stop.json').exists():
                    interrupted = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    break
                time.sleep(.3)
        state.update(status='STOPPED' if interrupted else ('EXITED' if proc.returncode==0 else 'FAILED'), exit_code=proc.returncode)
    except InterruptedError as exc:
        state.update(status='STOPPED',error=str(exc))
    except Exception as exc:
        state.update(status='FAILED', error=str(exc)); raise
    finally:
        if claimed:
            try:
                policy(config, run_id, 'INACTIVE')
                # Lifecycle stop is the only supervisor command; never an extra task action.
                execute(config, "import os\nos.environ['CODEWORLD_ROLE']='supervisor_stop'\nfrom pathlib import Path\nif not Path('robot_io/closed.json').exists():\n from codeworld_robot import Robot\n Robot('.').finish()\n", run_id)
            except Exception as exc: state['cleanup_error'] = str(exc)
        events = directory/'events.jsonl'
        if events.exists():
            for line in events.read_text().splitlines():
                try: event=json.loads(line)
                except ValueError: continue
                if event.get('type')=='thread.started': state['thread_id']=event.get('thread_id')
        state['ended_at']=time.time(); atomic(directory/'lifecycle.json',state)


if __name__ == '__main__': main()
