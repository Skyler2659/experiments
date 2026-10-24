"""Bind every external robot decision to an executable CodeWorld computation."""
from pathlib import Path
from .io import read,atomic
from codeworld_workspace.flow import strict
from codeworld_workspace.completion import path_in

OPERATIONS={'move_tcp','move_tcp_group','joints','joints_group','gripper','gripper_group','step','select_arm','check_ik','check_motion'}


def enforce(root,packet):
    root=Path(root).resolve()
    if not strict(root) or packet.get('actor',{}).get('role')!='codex_cli':return
    op=packet['op'];args=packet.get('args',{})
    if op not in OPERATIONS:return
    from codeworld_workspace import World
    world=World(root);world.sync_robot_feedback()
    ref=args.get('world_decision')
    if not isinstance(ref,dict):raise ValueError('Every robot decision must use World.compute_decision -> Robot.execute_decision')
    receipt=read(path_in(root,ref['file']));i=ref.get('index')
    if type(i)!=int or i<0:raise ValueError('Invalid decision action index')
    # Bind to the registered computation and its relevant inputs. Dispatch does
    # not rerun the agent's calculation or expire it for append-only bookkeeping.
    from codeworld_workspace.flow import verify_decision
    verify_decision(world,receipt,actor=packet.get('actor'))
    try:action=receipt['result']['actions'][i]
    except (KeyError,IndexError,TypeError):raise ValueError('Decision has no such action')
    payload={k:v for k,v in args.items() if k!='world_decision'}
    if action['op']!=op or action['args']!=payload:raise ValueError('Robot command differs from world-computed payload')
    ledger=root/'task/world_decision_uses.json';used=read(ledger) if ledger.exists() else {}
    key=receipt['id']+':'+str(i)
    previous=used.get(key)
    previous_id=previous.get('request_id') if isinstance(previous,dict) else previous
    if previous is not None and previous_id!=packet['id']:raise ValueError('Decision action already submitted; inspect original receipt before retrying')
    if isinstance(previous,dict) and previous.get('receipt')!=receipt:
        raise ValueError('Decision receipt changed after this request was authorized')
    # Feedback must consume the exact computation authorized for this request,
    # even if the agent subsequently rewrites its editable decision artifact.
    if not isinstance(previous,dict):
        used[key]={'request_id':packet['id'],'receipt':receipt}
        atomic(ledger,used)
