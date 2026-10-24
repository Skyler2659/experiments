import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
from robodojo_benchmark.campaign import Campaign, write
from robodojo_benchmark.catalog import ROOT, build_manifest
from robodojo_benchmark import worker
from robodojo_benchmark.provenance import archive_sources, git_state, tree_digest, upstream_state


def native_score(attempt, job):
    write(attempt/'evaluator/_result.json', {'details': {'0': {
        'layout_id': job['layout_id'], 'success': False, 'score': .3}}})
    write(attempt/'evaluator/official_result.json', {'official_actions': 17, 'official_action_limit': 1050})


@pytest.mark.parametrize('lifecycle', [
    {'status': 'FAILED', 'exit_code': 1},
    {'status': 'STOPPED', 'exit_code': -15},
    {'status': 'STOPPED', 'termination_reason': 'wall_timeout', 'exit_code': -15},
])
def test_agent_exit_does_not_overwrite_native_score(tmp_path, monkeypatch, lifecycle):
    manifest = build_manifest({'tasks': ['build_tower'], 'samples': 1, 'strategies': ['ours_no_codeworld']})
    job = manifest['jobs'][0]
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=123, wait=lambda **k: 0))
    monkeypatch.setattr(worker, 'wait_file', lambda *a: None)
    def agent(*a, **k):
        native_score(tmp_path, job)
        write(tmp_path/'agent/lifecycle.json', lifecycle)
        return 1
    monkeypatch.setattr(worker, 'run_process', agent)
    result = worker.run_ours(tmp_path, {'robodojo_root': str(ROOT/'vendor/RoboDojo'),
        'sim_python': 'unused', 'control_python': 'unused', 'codex_auth_home': str(tmp_path/'no_auth')},
        manifest['spec'], job, {'CUDA_VISIBLE_DEVICES': '0', 'ROBODOJO_RUN_ID': 'test'})
    assert result['status'] == 'SCORED' and result['score_0_100'] == 30
    assert result['official_success'] is False and result['agent_lifecycle'] == lifecycle


@pytest.mark.parametrize('failure', [RuntimeError('cleanup failed'), KeyboardInterrupt('stopped')])
def test_exception_after_native_score_is_final(tmp_path, monkeypatch, failure):
    campaign = Campaign.create(tmp_path/'campaign', {'tasks': ['build_tower'], 'samples': 1, 'strategies': ['ours_no_codeworld']})
    job = campaign.claim(list(campaign.jobs))
    monkeypatch.setattr(worker, 'asset_revision', lambda _: None)
    layout = tmp_path/'layout.json'; write(layout, {})
    monkeypatch.setattr(worker, 'layout_path', lambda *a: layout)
    def execute(attempt, *a, **k):
        native_score(attempt, job)
        raise failure
    monkeypatch.setattr(worker, 'run_ours', execute)
    result = worker.execute_job(campaign, job, {'robodojo_root': str(ROOT/'vendor/RoboDojo')}, '0')
    campaign.finish(job, result)
    assert result['status'] == 'SCORED' and result['official_success'] is False
    assert 'post_score_error' in result
    with pytest.raises(ValueError): campaign.retry(job['job_id'], 'cleanup error')
    versions = json.loads((Path(result['artifact_dir'])/'source_versions.json').read_text())
    assert versions['framework']['tree_sha256']
    assert (campaign.root/versions['framework']['source_archive']).is_file()
    assert versions['actual_upstreams']['RoboDojo']['distributed_digest_verified']


def test_historical_misclassification_cannot_be_retried(tmp_path):
    campaign = Campaign.create(tmp_path/'campaign', {'tasks': ['build_tower'], 'samples': 1, 'strategies': ['ours']})
    job = campaign.claim(list(campaign.jobs))
    attempt = campaign.root/'runs'/job['job_id']/'attempt_001'
    native_score(attempt, job)
    saved = {'status': 'INFRA_ERROR', 'error': 'late CLI failure', 'probe_only': False}
    write(attempt/'result.json', saved); campaign.finish(job, saved)
    with pytest.raises(ValueError, match='Native result exists'): campaign.retry(job['job_id'], 'try again')
    campaign.recover(job['job_id'], 'restore finalized native score')
    row = campaign.rows()[0]
    assert row['status'] == 'SCORED' and row['result']['score_0_100'] == 30
    assert row['result']['recovered_previous_status'] == 'INFRA_ERROR'
    assert row['attempt'] == 1


def test_actual_sources_archived_once_without_local_credentials(tmp_path):
    source = tmp_path/'source'; (source/'codeworld_robot').mkdir(parents=True)
    (source/'codeworld_robot/a.py').write_text('x = 1\n')
    (source/'codeworld_robot/auth.json').write_text('secret')
    (source/'codeworld_robot/runtime.local.json').write_text('secret')
    campaign = tmp_path/'campaign'; campaign.mkdir()
    first = archive_sources(source, campaign)
    assert archive_sources(source, campaign) == first
    with tarfile.open(campaign/first['source_archive']) as archive:
        assert archive.getnames() == ['CodeWorld/codeworld_robot/a.py']
        assert archive.extractfile(archive.getnames()[0]).read() == b'x = 1\n'
    (source/'codeworld_robot/a.py').write_text('x = 2\n')
    second = archive_sources(source, campaign)
    assert first['tree_sha256'] != second['tree_sha256']
    assert len(list((campaign/'sources').glob('*.tar.gz'))) == 2


def test_upstream_content_change_detected_with_same_declared_commit(tmp_path):
    (tmp_path/'policy.py').write_text('original')
    pin = {'commit': 'upstream-pin', 'tree_sha256': tree_digest(tmp_path, ('.',))}
    write(tmp_path/'SOURCE_REVISION.json', pin)
    assert not upstream_state(tmp_path, 'upstream-pin')['errors']
    (tmp_path/'policy.py').write_text('modified')
    assert upstream_state(tmp_path, 'upstream-pin')['errors']


def test_git_identity_reports_dirty_checkout(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)
    git('init'); git('config', 'user.name', 'Test'); git('config', 'user.email', 'test@example.invalid')
    (tmp_path/'policy.py').write_text('original'); git('add', '.'); git('commit', '-m', 'initial')
    initial = git_state(tmp_path)
    assert initial['commit'] and initial['dirty'] is False
    (tmp_path/'policy.py').write_text('modified')
    assert git_state(tmp_path) == {**initial, 'dirty': True}
    assert upstream_state(tmp_path, initial['commit'])['errors']


def test_installation_outputs_do_not_change_source_digest(tmp_path):
    (tmp_path/'policy.py').write_text('original')
    before = tree_digest(tmp_path, ('.',))
    for directory in ('RoboProbe.egg-info', '__pycache__', '.cache', '.secrets'):
        (tmp_path/directory).mkdir()
        (tmp_path/directory/'generated').write_text('not source')
    assert tree_digest(tmp_path, ('.',)) == before


def test_recover_native_probe_remains_excluded(tmp_path):
    campaign = Campaign.create(tmp_path/'campaign', {'tasks': ['build_tower'], 'samples': 1, 'strategies': ['gt_codeworld']})
    job = campaign.claim(list(campaign.jobs))
    attempt = campaign.root/'runs'/job['job_id']/'attempt_001'
    native_score(attempt, job)
    write(attempt/'evaluator/server_config.json', {'probe_native_only': True})
    campaign.recover(job['job_id'], 'probe process stopped after scoring')
    from robodojo_benchmark.report import summarize
    assert campaign.rows()[0]['result']['probe_only'] is True
    assert summarize(campaign)['tasks'][0]['scored'] == 0
