import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import trimesh
from codeworld_robot.recording import FrameTimeline
from robodojo_benchmark.artifacts import export


def dump(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload))


@pytest.mark.parametrize('strategy',['ours','gt_codeworld','ours_no_codeworld','roboprobe_l3'])
def test_synchronized_export(tmp_path,strategy):
    root=tmp_path/'episode';io=root/'robot_io'
    recorder=FrameTimeline(root)
    has_world=strategy in ('ours','gt_codeworld')
    dump(tmp_path/'job.json',{'strategy':strategy})
    if has_world:
        (root/'geometry').mkdir()
        trimesh.creation.box(extents=[.1,.1,.1]).export(root/'geometry/cube.ply')
        dump(root/'world.json',{'objects':{'cube':'objects/cube.json'}})
        dump(root/'objects/cube.json',{'geometry':{'local_mesh':'geometry/cube.ply'}})
        dump(root/'state/current.json',{'revision':1,'objects':{'cube':{'T_W_object':np.eye(4).tolist()}}})
    video=cv2.VideoWriter(str(io/'continuous.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,480))
    assert video.isOpened()
    for i in range(4):
        recorder.operation={'request_id':str(i),'op':'fixture_no_robot_action'}
        video.write(np.full((480,640,3),i*40,np.uint8))
        recorder.record({'control_steps':i*10})
    video.release()
    result=export(tmp_path)
    assert result['status']=='COMPLETE' and result['frames']==4
    assert len(result['videos'])==(3 if has_world else 2)
    for name,relative in result['videos'].items():
        cap=cv2.VideoCapture(str(tmp_path/relative))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==4
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==(640 if name=='task' else 1280)
        cap.release()
    rows=[json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [r['operation']['request_id'] for r in rows]==['0','1','2','3']
    if has_world:assert len(set(r['world_snapshot'] for r in rows))==1


def test_historical_video_without_timeline_not_fabricated(tmp_path):
    assert export(tmp_path)['status']=='UNAVAILABLE'


def test_api_configuration_does_not_save_key(tmp_path,monkeypatch):
    import importlib.util
    import tomllib
    from robodojo_benchmark.catalog import ROOT
    spec=importlib.util.spec_from_file_location('configure_api',ROOT/'experiments/robodojo/configure_api.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    runtime=tmp_path/'runtime.local.json';dump(runtime,{'sim_python':'/example/python'})
    monkeypatch.setenv('CUSTOM_KEY','secret-fixture-value')
    module.configure(runtime,'https://example.com/v1','CUSTOM_KEY')
    config=json.loads(runtime.read_text())
    provider=Path(config['codex_provider_config'])
    parsed=tomllib.loads(provider.read_text())
    assert parsed['model_providers']['experiment']['env_key']=='CUSTOM_KEY'
    assert config['vision_transport']=='responses'
    assert config['reconstruction_base_url']==config['roboprobe_base_url']
    assert 'secret-fixture-value' not in runtime.read_text()+provider.read_text()
    with pytest.raises(ValueError):module.configure(runtime,'https://example.com/v1/responses')
