"""Keep reconstructed geometry as a prior; discard previous execution evidence."""
import shutil
from pathlib import Path
from .io import atomic,read


def reset_episode_evidence(root):
    root=Path(root)
    for rel in ('robot_io','decisions','work/completion'):
        path=root/rel
        if path.exists():shutil.rmtree(path)
    state_path=root/'state/current.json'
    if not state_path.exists():return
    state=read(state_path)
    state.pop('runtime',None);state.pop('measurements',None)
    for row in state.get('objects',{}).values():
        row.pop('sensor_validation',None)
        row['basis']='reconstruction_prior'
        row.setdefault('review',{})['pose']={
            'status':'prior',
            'reason':'Reconstructed geometry is available for planning; current episode observation is not yet confirmed',
        }
        for field in row.get('observed_state',{}).get('fields',{}).values():
            field['status']='pending';field.pop('derivation',None)
    state['revision']=state.get('revision',0)+1
    state.setdefault('history',[]).append({'revision':state['revision'],'kind':'fresh_episode_evidence_reset'})
    atomic(state_path,state)
