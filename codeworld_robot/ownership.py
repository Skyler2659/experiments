"""Accidental cross-session control guard, not an OS security boundary."""
import os
from pathlib import Path
from .io import read


def actor():
    return {'role': os.environ.get('CODEWORLD_ROLE', 'supervisor_manual'),
            'run_id': os.environ.get('CODEWORLD_EXECUTOR_ID')}


def authorize(root, packet):
    policy = Path(root) / 'task/executor.json'
    source = packet.get('actor', {'role': 'legacy_unattributed'})
    if policy.exists():
        owner = read(policy)
        stopping = packet['op'] == 'finish' and source.get('role') == 'supervisor_stop'
        owned = (owner.get('status') == 'ACTIVE' and source.get('role') == 'codex_cli'
                 and source.get('run_id') == owner.get('run_id') and bool(owner.get('run_id')))
        if not (stopping or owned):
            raise PermissionError('Episode belongs to independent Codex CLI; use its launcher. Supervisor may only stop.')
    from .flow import enforce as enforce_flow
    enforce_flow(root,packet)
    from .completion import enforce
    enforce(root, packet)
    return source
