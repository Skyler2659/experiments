"""Per-object acceptance, local progress limits, and versioned subtree reuse."""
import hashlib
import json
import shutil
from pathlib import Path

try:
    from .refinement_policy import resolve_children
except ImportError:
    from refinement_policy import resolve_children


def enabled(config):
    return config.get('refinement', {}).get('strategy') == 'adaptive'


def program_files(node):
    controls = ('feedback', 'fit_requests', 'selection', 'review_state',
                'refinement_pending', 'node_context', 'adaptive_', 'correction_')
    return sorted(p for p in Path(node).iterdir() if p.is_file()
                  and (p.suffix in {'.js', '.mjs', '.json', '.html', '.png', '.jpg', '.jpeg', '.webp', '.glb', '.bin'} or p.name == 'account.md')
                  and not p.name.startswith(controls)
                  and p.name != 'manifest.json'
                  and not p.name.startswith('children.'))


def subtree(node, run):
    visited = set()
    def visit(directory):
        directory = Path(directory).resolve()
        if directory in visited:
            return
        visited.add(directory)
        yield directory
        try:
            children = resolve_children(directory, run)
        except (ValueError, OSError):
            children = []  # The invalid manifest itself is still hashed and reviewed as an error.
        for _, child in children:
            yield from visit(child)
    return list(visit(node))


def revision(node, run):
    digest = hashlib.sha256()
    for directory in sorted(subtree(node, run)):
        for path in program_files(directory):
            digest.update(str(path.relative_to(Path(run).resolve())).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def evidence_revision(run):
    """Changed observations/calibration reopen a stopped object, not changed briefs."""
    digest = hashlib.sha256()
    for name in ('measurement/evidence.json', 'capture.json', 'objects.json'):
        path = Path(run) / 'input' / name
        digest.update(name.encode())
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def can_delegate(node, run):
    """Object descent requires a measured residual, not a category or unused depth."""
    node, run = Path(node), Path(run)
    context = json.loads((node / 'node_context.json').read_text())
    if context['depth'] == 0:
        # Keep object identity fixed so a renamed object cannot acquire a fresh budget.
        objects = json.loads((run / 'input/objects.json').read_text())['objects']
        requested = json.loads((node / 'children.json').read_text())
        return set(requested) <= {obj['object_id'] for obj in objects}
    owner = node
    if context['depth'] >= 2:
        scope = node / 'node_scope.json'
        if not scope.exists():
            return False
        owner = run / 'fractal' / json.loads(scope.read_text())['owner_object_id']
    path = owner / 'adaptive_state.json'
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    return (state.get('decision') == 'REFINE'
            and state.get('evidence_revision') == evidence_revision(run))


def save_best(node, run):
    destination = Path(node) / '.adaptive_best'
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir()
    files = []
    for directory in subtree(node, run):
        for source in program_files(directory):
            relative = source.relative_to(Path(run).resolve())
            target = destination / 'files' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append(str(relative))
    (destination / 'manifest.json').write_text(json.dumps(files))


def restore_best(node, run):
    destination = Path(node) / '.adaptive_best'
    files = json.loads((destination / 'manifest.json').read_text())
    try:
        directories = set(subtree(node, run))
    except (ValueError, OSError):
        directories = {Path(node).resolve()}
    directories.update((Path(run) / relative).resolve().parent for relative in files)
    for directory in directories:
        for path in program_files(directory):
            path.unlink()
    for relative in files:
        target = Path(run) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination / 'files' / relative, target)


def decision(result, state, config, only_inherited=False):
    policy = config.get('refinement', {})
    score = result['score']
    previous = state.get('progress_best')
    if result['status'] != 'PROGRAM_ERROR':
        if previous is not None:
            required = max(policy.get('min_absolute_improvement', .005),
                           abs(previous) * policy.get('min_relative_improvement', .03))
            gain = previous - score
            state['stagnant_reviews'] = (0 if gain >= required else
                                         state.get('stagnant_reviews', 0) + 1)
        state['progress_best'] = score if previous is None else min(previous, score)
    if result['status'] == 'FIT_CHECKS_PASSED':
        return 'ACCEPTED'
    if only_inherited:
        return 'BLOCKED_BY_STOPPED_CHILD'
    if state['reviews'] >= config.get('max_reviews', 3):
        return 'REVIEW_BUDGET_EXHAUSTED'
    if state.get('stagnant_reviews', 0) >= policy.get('patience', 2):
        return 'NO_PROGRESS'
    return 'REFINE'


def dispatch(node, run, parent):
    """Reuse accepted/stopped programs; only a new measured regression reopens a pass."""
    node, run = Path(node), Path(run)
    path = node / 'adaptive_state.json'
    if not path.exists() or not (node / 'part.json').exists():
        return False
    state = json.loads(path.read_text())
    if state.get('decision') == 'REFINE' or state.get('evidence_revision') != evidence_revision(run):
        return False
    if state.get('reviewed_revision') != revision(node, run):
        if state['decision'] == 'ACCEPTED':
            return False
        # A parent cannot restart a stopped local budget by editing its child.
        restore_best(node, run)
        state['reviewed_revision'] = revision(node, run)
        path.write_text(json.dumps(state, indent=2))
    targets = run / 'fractal' / parent / 'adaptive_targets.json'
    if state['decision'] == 'ACCEPTED' and targets.exists():
        ticket = json.loads(targets.read_text()).get(node.name)
        if ticket and ticket.get('revision') == state['reviewed_revision']:
            identity = ticket['feedback']
            if identity != state.get('consumed_parent_feedback'):
                state['consumed_parent_feedback'] = identity
                path.write_text(json.dumps(state, indent=2))
                return False
    return True


def finish_review(node, run, config, result, feedback, only_inherited=False):
    """Review state remains cumulative across parent-driven revisits."""
    node, run = Path(node), Path(run)
    path = node / 'adaptive_state.json'
    state = json.loads(path.read_text()) if path.exists() else {'reviews': 0}
    current = revision(node, run)
    evidence = evidence_revision(run)
    if state.get('evidence_revision', evidence) != evidence:
        consumed = state.get('reviews', 0) if config.get('agent_correction', {}).get('enabled') else 0
        state = {'reviews': consumed}
    state['reviews'] += 1
    # A valid program always outranks a protocol error, even with a larger numeric score.
    valid = result['status'] != 'PROGRAM_ERROR'
    better = ('best_score' not in state or
              (valid and state.get('best_status') == 'PROGRAM_ERROR') or
              (valid == (state.get('best_status') != 'PROGRAM_ERROR') and
               result['score'] < state['best_score']))
    if better:
        state.update(best_score=result['score'], best_status=result['status'],
                     best_feedback=feedback, best_revision=current)
        save_best(node, run)
    state['decision'] = decision(result, state, config, only_inherited)
    state.update(evidence_revision=evidence, reviewed_revision=current,
                 latest_feedback=feedback)
    retry = state['decision'] == 'REFINE'
    if not retry and state['best_revision'] != current:
        restore_best(node, run)
        state['reviewed_revision'] = revision(node, run)
    path.write_text(json.dumps(state, indent=2))
    if not retry:
        (node / 'selection.json').write_text(json.dumps({
            'reviews': state['reviews'], 'selected_feedback': state['best_feedback'],
            'selected_score': state['best_score'], 'selected_status': state['best_status'],
            'restored_best': state['best_revision'] != current,
            'stop_reason': state['decision'],
            'budget_exhausted': state['best_status'] != 'FIT_CHECKS_PASSED',
            'independent_acceptance': False,
        }, indent=2))
    # A parent may request only residual owners with remaining local progress budget.
    targets = {}
    if retry:
        for obj in result.get('objects', []):
            if not obj.get('needs_revision'):
                continue
            child = run / 'fractal' / obj['object_id']
            if child == node or not (child / 'part.json').exists():
                continue
            stopped = child / 'adaptive_state.json'
            child_state = json.loads(stopped.read_text()) if stopped.exists() else {}
            if child_state.get('decision') not in (None, 'REFINE', 'ACCEPTED'):
                continue
            targets[child.name] = {'feedback': feedback, 'revision': revision(child, run)}
    (node / 'adaptive_targets.json').write_text(json.dumps(targets, indent=2))
    return retry, state
