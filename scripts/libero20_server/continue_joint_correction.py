"""Explicitly authorized local continuation of an ended reconstruction episode."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE/'reconstruction_assist'))
from reconstruction_assist import correction_review as correction
from reconstruction_assist import adaptive_refinement as adaptive
sys.path.insert(0, str(Path(__file__).parent))
import batch
import run_single_correction as monitoring


def main():
    base = Path(sys.argv[1]); case = base/'scenes/kitchen_scene10'
    run_name = json.loads((case/'config.json').read_text())['scene']
    runtime = case/'runtime'; run = runtime/'runs'/run_name
    work = base/'recovery'; work.mkdir(exist_ok=False)
    before = work/'before'; before.mkdir()
    for f in [base/'status.json',case/'status.json',case/'error.txt']:
        if f.exists():shutil.copy2(f,before/(('batch_' if f.parent==base else '')+f.name))
    shutil.copytree(run/'input/measurement',before/'measurement')
    for node in (run/'fractal').iterdir():
        if not node.is_dir():continue
        out=before/'fractal'/node.name;out.mkdir(parents=True)
        for f in node.iterdir():
            if f.is_file() and not f.name.endswith('.log'):shutil.copy2(f,out/f.name)
    cfg=json.loads((run/'assist_transport.json').read_text())
    cfg['remote_source']=str(SOURCE)
    correction.write(work/'transport.json',cfg)
    env=batch.env();env.update(RCWM_JDCLOUD_PROXY_KEY='rcwm-local',RCWM_ROOT=str(runtime),CODEX_HOME=str(case/'codex_home'),
        RCWM_MODEL='gpt-5.6-sol',RCWM_REASONING='max',RCWM_ADAPTIVE_DISPATCH='1',
        RCWM_ASSIST_CONFIG=str(work/'transport.json'),
        RCWM_FEEDBACK_HOOK=str(SOURCE/'reconstruction_assist/feedback_hook.py'),
        RCWM_CODE=str(run/'runner_source'),RCWM_PROMPT=str(run/'solver_prompt.md'),
        RCWM_MAXCYC='8',RCWM_MAXD='4')
    env['PATH']=str(runtime/'bin')+':'+str(runtime/'.render-tools/node/bin')+':'+env['PATH']
    accepted=['akita_black_bowl_1','wooden_cabinet_1','chocolate_pudding_1','butter_2']
    revisions={oid:adaptive.revision(run/'fractal'/oid,run) for oid in accepted}
    node=run/'fractal/butter_1'
    diagnosis_path=sorted(node.glob('correction/*/diagnosis.json'))[-1]
    diagnosis=json.loads(diagnosis_path.read_text())
    known=[o['object_id'] for o in json.loads((run/'input/objects.json').read_text())['objects']]
    affected=correction.validate_diagnosis(diagnosis,known,['butter_1'])
    version=adaptive.revision(node,run)
    old_state=json.loads((node/'adaptive_state.json').read_text())
    if version!=old_state['reviewed_revision']:raise RuntimeError('Program changed since selected diagnosis; do not replay')
    output=node/'correction'/('continued_'+str(time.time_ns()));output.mkdir()
    correction.write(output/'diagnosis.json',diagnosis)
    correction.write(output/'replay.json',{'source_diagnosis':str(diagnosis_path),'authorized_continuation':True,'program_revision':version})
    old_evidence=json.loads((run/'input/measurement/evidence.json').read_text())['objects']
    correction.rebuild_evidence(run,cfg,diagnosis['mask_assignments'],affected,output)
    new_evidence=json.loads((run/'input/measurement/evidence.json').read_text())['objects']
    targets=correction.repair_targets(run,affected,old_evidence,new_evidence)
    if targets!={'butter_1'}:raise RuntimeError('Unexpected local recovery targets: '+repr(targets))
    correction.write(work/'plan.json',{'replayed_model_diagnosis':str(diagnosis_path),'mask_identity_partners':sorted(affected),
        'geometry_repair_targets':sorted(targets),'reused_accepted_objects':accepted,'prior_reviews':old_state['reviews'],
        'bounds':{oid:new_evidence[oid]['visible_bounds_W_m'] for oid in affected}})
    correction.reopen(node,run,cfg,diagnosis,output)
    if (case/'error.txt').exists():(case/'error.txt').rename(before/'original_error.txt')
    def call(command,stage):
        batch.state(case,stage)
        with (work/(stage.lower()+'.log')).open('a') as log:
            subprocess.run([str(x) for x in command],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    call(['bash',run/'runner_source/runner/solve_recursive.sh','runs/'+run_name,'butter_1','scene','1',run_name], 'REPAIRING_BUTTER_1')
    for oid,rev in revisions.items():
        if adaptive.revision(run/'fractal'/oid,run)!=rev:raise RuntimeError('Accepted object changed: '+oid)
    root=run/'fractal/scene';state=json.loads((root/'adaptive_state.json').read_text())
    state.update(decision='REFINE',evidence_revision=adaptive.evidence_revision(run),stagnant_reviews=0)
    for key in ('best_score','best_status','best_revision','best_feedback','progress_best'):state.pop(key,None)
    correction.write(root/'adaptive_state.json',state)
    call([batch.PY,SOURCE/'reconstruction_assist/feedback_hook.py','runs/'+run_name,'scene','3'], 'REVIEWING_REPAIRED_ASSEMBLY')
    if (case/'export').exists():(case/'export').rename(before/'export_incomplete')
    exporter=runtime/'.render-tools/node/bin/node'
    env.update(RCWM_EXPORT_OBJECTS='1',RCWM_EXPORT_CANONICAL_VIEW='1')
    call([exporter,SOURCE/'scripts/export_codeworld.mjs',runtime,run_name,case/'export'], 'EXPORTING')
    call([batch.PY,'-m','codeworld_workspace.build','--run',run,'--export',case/'export','--output',case/'workspace','--runtime',runtime,'--goal','Static reconstruction local continuation; no robot actions'], 'BUILDING_WORKSPACE')
    call([exporter,SOURCE/'scripts/render_codeworld.mjs',runtime,case/'workspace',case/'capture/capture.json',case/'review'], 'RENDERING_REVIEW')
    nodes=[{'node':f.parent.name,**json.loads(f.read_text())} for f in (run/'fractal').glob('*/adaptive_state.json')]
    correction.write(case/'result.json',{'status':'DELIVERED_REVIEW_REQUIRED','model':'gpt-5.6-sol','reasoning':'max','run':str(run),
        'workspace':str(case/'workspace'),'nodes':nodes,'authorized_local_continuation':True,'reused_accepted_objects':accepted,
        'actual_max_depth':max(json.loads(x).get('depth',0) for x in (run/'trace/events.jsonl').read_text().splitlines()),'robot_actions':0})
    batch.state(case,'DELIVERED_REVIEW_REQUIRED')


if __name__=='__main__':
    failure=[]
    def execute():
        try:main()
        except Exception as error:
            import traceback
            base=Path(sys.argv[1]);(base/'recovery_error.txt').write_text(traceback.format_exc())
            batch.state(base/'scenes/kitchen_scene10','FAILED_RECOVERY',error=repr(error));failure.append(error)
    worker=threading.Thread(target=execute);worker.start()
    while worker.is_alive():
        try:monitoring.report()
        except (OSError,ValueError):pass
        worker.join(10)
    monitoring.report()
    if failure:raise SystemExit(1)
