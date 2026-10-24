"""Completion enforcement shared by LIBERO, RoboDojo and FetchBench transports."""
from pathlib import Path
from .io import read,atomic

MOTION_OPS={'move_tcp','move_tcp_group','joints','joints_group','gripper','gripper_group','step'}
SYSTEM_TERMINATIONS={'budget_exhausted','timeout','tool_error','process_error','interrupted'}


def enforce(root,packet):
    root=Path(root).resolve()
    if packet.get('actor',{}).get('role')!='codex_cli' or not (root/'world.json').exists():return
    op=packet['op'];args=packet.get('args',{})
    if op in MOTION_OPS:
        if not (root/'task/completion.json').exists():raise ValueError('Define all task/subtask completion conditions in CodeWorld before acting')
        atomic(root/'task/completion_started.json',{'actor':packet['actor']})
    if op!='finish':return
    outcome=args.get('outcome','success')
    system_reason=args.get('system_reason')
    if system_reason is not None:
        if outcome not in ('failed','aborted'):raise ValueError('System termination cannot claim task success')
        if (not isinstance(system_reason,dict) or system_reason.get('kind') not in SYSTEM_TERMINATIONS
                or not isinstance(system_reason.get('detail'),str) or not system_reason['detail'].strip()):
            raise ValueError('System termination requires a supported kind and nonempty detail')
        if args.get('judgment') is not None:raise ValueError('Choose a physical judgment or a system termination reason')
        atomic(root/'task/completion_closure.json',{'outcome':outcome,'category':'system_termination',
            'reason':system_reason['detail'],'system_reason':system_reason,'actor':packet['actor']})
        return
    if outcome in ('failed','aborted'):
        if not isinstance(args.get('reason'),str) or not args['reason'].strip():raise ValueError('Non-success closure needs a reason')
        from codeworld_workspace.flow import strict
        if strict(root):
            from codeworld_workspace import World
            from codeworld_workspace.completion import path_in
            from codeworld_workspace.flow import verify_decision
            w=World(root);w.sync_robot_feedback()
            if not args.get('judgment'):raise ValueError('Failure/inability judgment must also reference a CodeWorld computation')
            judgment=read(path_in(root,args['judgment']))
            verify_decision(w,judgment,actor=packet['actor'])
            if args['reason'] not in judgment.get('result',{}).get('judgments',{}).values():raise ValueError('Unsupported failure judgment')
        atomic(root/'task/completion_closure.json',{'outcome':outcome,'reason':args['reason'],'actor':packet['actor']})
        return
    if outcome!='success':raise ValueError('Unknown completion outcome')
    from codeworld_workspace import World
    from codeworld_workspace.completion import path_in
    if not args.get('verification'):raise ValueError('Completion requires a CodeWorld verification receipt; image/text self-assessment is insufficient')
    report=read(path_in(root,args['verification']))
    connection=read(root/'task/robot_connection.json');io=Path(connection['io_directory'])
    live=World(root).verify_completion(robot_state=read(io/'robot_state.json'))
    same_contract=(report.get('schema')==live['schema'] and report.get('scope')=='task'
        and report.get('status')=='VERIFIED' and report.get('contract_digest')==live['contract_digest']
        and report.get('geometry_revision')==live['geometry_revision']
        and set(report.get('goals',{}))==set(live['goals']))
    # A read-only receipt or unrelated metadata may change the world revision.
    # Re-evaluate the physical conditions now instead of comparing whole reports.
    if not same_contract or live['scope']!='task' or live['status']!='VERIFIED':
        raise ValueError('Completion is stale, incomplete or not verified; inspect CodeWorld goal checks')
    atomic(root/'task/completion_closure.json',{'outcome':'world_verified','verification':live,'actor':packet['actor']})
