import concurrent.futures
import json
from pathlib import Path
import pytest
from robodojo_benchmark.catalog import build_manifest, layout_path, STRATEGIES
from robodojo_benchmark.campaign import Campaign, write
from robodojo_benchmark.report import summarize
from robodojo_benchmark.worker import native_result, execute_job, prepare_files, fresh_codex_home


def test_published_matrix_and_pairing():
    m = build_manifest({})
    assert len(m['jobs']) == 2800
    assert m['platform_sources']['Assets']['commit']=='35efbc7dedfdbeeb6e95fb749bd885d73d483e41'
    pairs = {}
    for j in m['jobs']:
        key = (j['task'], j['sample'])
        pairs.setdefault(key, []).append(j)
    assert len(pairs) == 700
    for rows in pairs.values():
        assert {j['strategy'] for j in rows} == set(STRATEGIES)
        assert len({(j['native_task'], j['eval_seed'], j['layout_id']) for j in rows}) == 1
        assert all('success' not in j and 'score' not in j for j in rows)
    hang = [j for j in m['jobs'] if j['task'] == 'hang_mugs' and j['strategy'] == 'ours']
    assert sum(j['variant'] == 'random' for j in hang) == 25
    assert hang[25]['native_task'] == 'hang_mugs_random' and hang[25]['layout_id'] == 0
    egg = [j for j in m['jobs'] if j['task'] == 'fill_egg_holder' and j['strategy'] == 'ours']
    assert [j['layout_id'] for j in egg[-4:]] == [50, 51, 52, 53]


def test_extension_and_sorted_layout_index(tmp_path):
    m = build_manifest({'strategies':['ours'], 'tasks':[{'task':'new_task', 'layouts':[0, {'layout_id':1,'native_task':'new_task_random'}]}]})
    assert len(m['jobs']) == 2 and m['spec']['agent']['wall_timeout_s'] > 0
    folder = tmp_path/'Assets/Eval_Layout/RoboDojo/arx_x5/0'; folder.mkdir(parents=True)
    for name in ['new_task_random_9.json','new_task_random_2.json','new_task_random_100.json','new_task_1.json']:
        (folder/name).write_text('{}')
    assert layout_path(tmp_path, m['jobs'][1]).name == 'new_task_random_9.json'
    for bad in ({'tasks':['unknown']}, {'tasks':['hang_mugs','hang_mugs']}, {'samples':[True]}, {'samples':[0]}, {'samples':51}, {'agent':{'wall_timeout_s':0}}):
        with pytest.raises(ValueError): build_manifest(bad)


def test_concurrent_claims_retries_and_incomplete_summary(tmp_path):
    c = Campaign.create(tmp_path/'campaign', {'tasks':['insert_tubes'], 'samples':5})
    def drain(_):
        claimed = []
        while (job := c.claim(list(c.jobs))) is not None:
            claimed.append(job['job_id'])
            c.finish(job, {'status':'SCORED','official_success':False,'score_0_100':0})
        return claimed
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        keys = sum(pool.map(drain, range(6)), [])
    assert len(keys) == len(set(keys)) == 20
    report = summarize(c)
    assert report['complete'] and report['paired_samples_complete'] == 5
    with pytest.raises(ValueError): c.retry(keys[0], 'try until success')
    c2 = Campaign.create(tmp_path/'retry', {'tasks':['insert_tubes'], 'samples':1})
    j = c2.claim(list(c2.jobs)); c2.finish(j, {'status':'INFRA_ERROR','error':'missing dependency'})
    c2.retry(j['job_id'], 'installed dependency')
    j2 = c2.claim([j['job_id']]); assert j2['attempt'] == 2
    with pytest.raises(ValueError): c2.finish(j, {'status':'METHOD_FAILURE'})
    c2.finish(j2, {'status':'METHOD_FAILURE'})
    report = summarize(c2)
    assert not report['complete'] and report['tasks'][0]['success_rate'] == 0
    assert report['tasks'][1]['success_rate'] is None
    assert report['tasks'][0]['completed_success_rate'] == 0


def test_native_result_and_probe_exclusion(tmp_path):
    f = tmp_path/'_result.json'
    write(f, {'details':{'0':{'layout_id':7,'success':False,'score':.4}}})
    assert native_result(f,7)['score_0_100'] == 40
    with pytest.raises(ValueError): native_result(f,0)
    for score in (float('nan'),float('inf'),1.1,-.1,'0.5',True):
        f.write_text(json.dumps({'details':{'0':{'layout_id':7,'success':True,'score':score}}}))
        with pytest.raises(ValueError): native_result(f,7)
    c = Campaign.create(tmp_path/'probe', {'tasks':['insert_tubes'], 'samples':1,'strategies':['gt_codeworld']})
    j=c.claim(list(c.jobs)); c.finish(j, {'status':'SCORED','official_success':True,'score_0_100':100,'probe_only':True})
    report=summarize(c)
    assert not report['complete'] and report['paired_samples_complete']==0 and report['tasks'][0]['scored']==0


def test_prelaunch_error_is_recorded_and_ablation_has_no_world(tmp_path):
    c=Campaign.create(tmp_path/'c', {'tasks':['insert_tubes'],'samples':1,'strategies':['ours_no_codeworld']})
    job=c.claim(list(c.jobs))
    result=execute_job(c,job,{'robodojo_root':str(tmp_path/'absent')},'0')
    assert result['status']=='INFRA_ERROR'
    c.finish(job,result)
    assert Path(result['artifact_dir'],'result.json').is_file()
    episode=tmp_path/'episode'; prepare_files(episode,job)
    assert not (episode/'world.json').exists()
    assert not json.loads((episode/'task/world_flow.json').read_text())['required']
    assert 'Ours minus CodeWorld' in (episode/'AGENTS.md').read_text()
    assert 'World.' not in (episode/'task/ROBOT_BACKEND.md').read_text()


def test_fresh_auth_home_never_imports_history(tmp_path):
    base=tmp_path/'auth';base.mkdir();(base/'auth.json').write_text('{}')
    (base/'AGENTS.md').write_text('secret task guidance');(base/'history.jsonl').write_text('history')
    with fresh_codex_home({'codex_auth_home':str(base)},reconstruction=True) as home:
        p=Path(home)
        assert (p/'auth.json').is_symlink()
        assert not (p/'AGENTS.md').exists() and not (p/'history.jsonl').exists()
        assert 'use_memories = false' in (p/'config.toml').read_text()
        assert (p/'skills/worldgen-techniques/SKILL.md').is_file()
        replacement=p/'new-auth.json';replacement.write_text('{"new_access_token":"test-only"}')
        replacement.replace(p/'auth.json')
    assert not p.exists()
    assert json.loads((base/'auth.json').read_text())['new_access_token']=='test-only'


def test_timeout_stops_detached_model_process(tmp_path):
    import subprocess, sys, time, os, signal
    from robodojo_benchmark.worker import stop_process
    pidfile=tmp_path/'child.pid'
    script=('import subprocess,sys,time\n'
            'from pathlib import Path\n'
            'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(90)"],start_new_session=True)\n'
            f'Path({str(pidfile)!r}).write_text(str(child.pid))\n'
            'time.sleep(90)\n')
    parent=subprocess.Popen([sys.executable,'-c',script],start_new_session=True)
    child=None
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
        child=int(pidfile.read_text());stop_process(parent,grace=1)
        # An adopted zombie is terminated too; it no longer runs or holds a GPU.
        state=subprocess.run(['ps','-p',str(child),'-o','stat='],text=True,capture_output=True).stdout.strip()
        assert not state or state.startswith('Z')
        assert parent.poll() is not None
    finally:
        for pid in (parent.pid,child):
            if pid:
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        parent.wait()


def test_model_and_setup_errors_are_not_geometry_failures(tmp_path):
    from robodojo_benchmark.reconstruct import infrastructure_failure, reconstruction_infrastructure_failure
    path=tmp_path/'stage.log'
    for message in ['ModelTransportFailure: request exited 1','ModuleNotFoundError: open3d','Missing runtime dependency: node',
                    "browserType.launch: Executable doesn't exist at /cache/chrome"]:
        path.write_text(message);assert infrastructure_failure(path)
    path.write_text('No RGB-derived mask for inventory objects: ["mug"]')
    assert infrastructure_failure(path) is None
    (tmp_path/'reconstruction.log').write_text('RCWM did not deliver a valid program')
    detail=tmp_path/'runtime/runs/episode/fractal/scene/codex-run.log'
    detail.parent.mkdir(parents=True);detail.write_text("bwrap: Can't find source path runtime: Permission denied")
    assert reconstruction_infrastructure_failure(tmp_path,'reconstruction')


def test_baseline_wall_budget_and_frozen_environment(tmp_path,monkeypatch):
    import subprocess
    from robodojo_benchmark import worker
    m=build_manifest({'strategies':['roboprobe_l3'],'tasks':['build_tower'],'samples':1})
    monkeypatch.setenv('TEST_BENCH_KEY','unit-test-placeholder')
    seen={}
    def timeout(argv,**kwargs):
        seen.update(kwargs['env']);raise subprocess.TimeoutExpired(argv,kwargs['timeout'])
    monkeypatch.setattr(worker,'run_process',timeout)
    runtime={'roboprobe_root':str(tmp_path),'robodojo_root':str(tmp_path),'sim_python':'unused',
             'roboprobe_base_url':'http://unused.invalid/v1','roboprobe_api_key_env':'TEST_BENCH_KEY'}
    result=worker.run_roboprobe(tmp_path,runtime,m['spec'],m['jobs'][0],
        {'CUDA_VISIBLE_DEVICES':'0','L3_INSPECT_EMBODIMENT_PROBE':'1','L3_INSPECT_ENV_STEP_LIMIT':'9999','L3_INSPECT_MAX_LLM_CALLS':'1'})
    assert result['status']=='METHOD_FAILURE' and result['score_0_100'] is None
    assert result['termination']=='agent_wall_budget'
    assert 'L3_INSPECT_EMBODIMENT_PROBE' not in seen and 'L3_INSPECT_ENV_STEP_LIMIT' not in seen
    assert seen['L3_INSPECT_MAX_LLM_CALLS']=='170'
    assert seen['L3_INSPECT_HARD_TIMEOUT_S']=='90'  # Released astra launcher default.


def test_asset_revision_does_not_confuse_native_code_head(tmp_path):
    from robodojo_benchmark.catalog import asset_revision
    rd=tmp_path/'RoboDojo';(rd/'Assets').mkdir(parents=True)
    (rd/'.git').mkdir();(rd/'.git/HEAD').write_text('code-revision')
    assert asset_revision(rd) is None
    (rd/'Assets').rmdir()
    repo=tmp_path/'dataset';(repo/'Assets').mkdir(parents=True)
    (rd/'Assets').symlink_to(repo/'Assets',target_is_directory=True)
    (repo/'.git/refs/heads').mkdir(parents=True)
    (repo/'.git/HEAD').write_text('ref: refs/heads/main\n')
    (repo/'.git/refs/heads/main').write_text('asset-revision\n')
    assert asset_revision(rd)=='asset-revision'
    (repo/'.git/refs/heads/main').unlink()
    (repo/'.git/packed-refs').write_text('# pack-refs\npacked-revision refs/heads/main\n')
    assert asset_revision(rd)=='packed-revision'
