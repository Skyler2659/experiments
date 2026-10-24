"""Hierarchy rules for measured reconstruction; subparts are accepted only in an assembly."""
import json
import hashlib
from pathlib import Path


def resolve_children(node,run):
    """Resolve the reference forms accepted by RecursiveCWM's check_part.py."""
    node,run=Path(node),Path(run);result=[]
    children=json.loads((node/'part.json').read_text()).get('children',[])
    if not isinstance(children,list):raise ValueError('part.json children must be a list')
    for ref in children:
        name=None;path=None
        if isinstance(ref,dict):
            name=next((ref[k] for k in ('id','name','node') if isinstance(ref.get(k),str)),None)
            path=next((ref[k] for k in ('part','path') if isinstance(ref.get(k),str)),None)
            if not path and not name:raise ValueError('Child reference needs id/name or part/path')
        elif isinstance(ref,str):
            if ref.endswith('.json') or ref.startswith('.'):path=ref
            else:name=ref
        else:raise ValueError('Unsupported child reference: '+repr(ref))
        candidates=[node/path] if path else [base/n/'part.json' for n in (name,Path(name).name) for base in (run/'fractal',node,node.parent)]
        part=next((p if p.is_file() else p/'part.json' for p in candidates if p.is_file() or (p/'part.json').is_file()),None)
        if part is None:raise ValueError('Missing child delivery: '+str(ref))
        directory=part.resolve().parent
        if not directory.is_relative_to((run/'fractal').resolve()):raise ValueError('Child delivery outside run fractal tree')
        result.append((name or directory.name,directory))
    return result


def subpart_scope(node, run):
    node, run = Path(node), Path(run)
    path = node / 'node_scope.json'
    if not path.exists():
        context_path = node / 'node_context.json'
        if context_path.exists() and json.loads(context_path.read_text())['depth'] >= 2:
            raise ValueError('Subpart node is missing its own node_scope.json; read the actual parent node_context and prepare the assigned scope in this flat node directory')
        return None
    scope = json.loads(path.read_text())
    if scope.get('kind') != 'subpart':
        raise ValueError('node_scope.kind must be subpart')
    context = json.loads((node / 'node_context.json').read_text())
    parent = context['parent_id']
    if parent in ('-', 'scene') or scope.get('parent_node') != parent:
        raise ValueError('Only an object or subpart can own a subpart')
    owner = scope.get('owner_object_id')
    if parent != owner:
        inherited = json.loads((run / 'fractal' / parent / 'node_scope.json').read_text())
        if inherited.get('owner_object_id') != owner:
            raise ValueError('Subpart owner differs from parent owner')
    return scope


def decomposition_required(node, config):
    policy = config.get('refinement', {})
    if policy.get('strategy') == 'adaptive':
        return False  # Evaluate the object before deciding whether any decomposition is necessary.
    if not policy.get('enabled') or node.name not in policy.get('structural_objects', []):
        return False
    context = json.loads((node / 'node_context.json').read_text())
    if context['depth'] >= context['max_depth']:
        return False
    part = json.loads((node / 'part.json').read_text())
    return len(part.get('children', [])) < policy.get('minimum_parts', 2)


def should_retry(result, state, config, only_inherited):
    strict = config.get('refinement', {}).get('enabled', False)
    return (result['status'] != 'FIT_CHECKS_PASSED'
            and state['reviews'] < config.get('max_reviews', 3)
            and (strict or not only_inherited))


def child_source_revision(node,run,sources):
    """A parent-only snapshot cannot restore a historical child program."""
    node,run=Path(node).resolve(),Path(run).resolve()
    digest=hashlib.sha256();visited=set()
    def visit(child):
        if child in visited:return
        visited.add(child)
        for path in sorted(sources(child)):
            digest.update(str(path.relative_to(run)).encode());digest.update(path.read_bytes())
        try:children=resolve_children(child,run)
        except (OSError,ValueError):digest.update(b'missing-or-invalid-child-manifest');return
        for _,sub in children:visit(sub)
    try:children=resolve_children(node,run)
    except (OSError,ValueError):children=[];digest.update(b'missing-or-invalid-child-manifest')
    for _,child in children:visit(child)
    return digest.hexdigest()


def update_child_revision(state,revision):
    changed='child_source_revision' in state and state['child_source_revision']!=revision
    if changed:
        state['best_score']=float('inf')
        state.pop('best_feedback',None);state.pop('best_status',None)
    state['child_source_revision']=revision
    return changed
