"""Separate visual agreement, structure fidelity and measured geometric accuracy.

This module can audit historical runs without rewriting their receipts. A view
used to select/revise a generated asset is not an independent final test view.
"""
import argparse
import json
from pathlib import Path


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def view_leakage(root):
    root = Path(root)
    review = read(root/'view_review/result.json', {})
    held = {r['object_id']: set(r.get('holdout_view_ids', [])) for r in review.get('objects', [])}
    usage = {oid: {} for oid in held}
    for stage in ('completion', 'shape', 'texture', 'registration'):
        packet = read(root/stage/'result.json', {})
        for row in packet.get('objects', []):
            oid = row['object_id']
            if oid not in usage:
                continue
            views = set(row.get('reviewed_against_view_ids', [])) | set(row.get('input_view_ids', []))
            for key in ('review_uri', 'structural_review_uri'):
                if row.get(key):
                    report = read(root/row[key], {})
                    views.update(v['view_id'] for v in report.get('observations', []))
            for vid in views:
                usage[oid].setdefault(vid, []).append(stage)
    # Numerical fit and constraints are checked here as well as in stage gates.
    manifest = read(root/'registration/manifest.json', {})
    for obj in manifest.get('objects', []):
        oid = obj['object_id']
        if oid not in usage:
            continue
        for rule in obj.get('structural_constraints', []):
            for vid in rule.get('view_ids', []):
                usage[oid].setdefault(vid, []).append('geometry_constraint')
    for row in review.get('objects', []):
        for vid in row.get('fit_view_ids', []):
            usage[row['object_id']].setdefault(vid, []).append('fit')
    conflicts = [{'object_id': oid, 'view_id': vid, 'used_by': sorted(set(usage[oid][vid]))}
                 for oid, ids in held.items() for vid in sorted(ids) if vid in usage[oid]]
    return {'status': 'NEED_FRESH_VALIDATION_VIEW' if conflicts else 'NO_DECLARED_LEAKAGE',
            'conflicts': conflicts, 'usage': usage,
            'scope': 'declared provenance only; absence of a recorded use does not certify independence'}


def object_quality(node):
    fit = node.get('fit') or {}
    residual = fit.get('support_error_m')
    return {
        'appearance': {'status': 'CANDIDATE', 'authority': 'mesh UV/material; unseen texels may be generated'},
        'structure': {'status': 'UNVERIFIED', 'authority': 'separate structural review required'},
        'scale': {'status': 'ESTIMATED', 'source': fit.get('size_source', 'UNKNOWN')},
        'pose': {'status': 'ESTIMATED', 'fit_status': fit.get('status', 'UNKNOWN'),
                 'view_scores': fit.get('view_scores', {})},
        'contact_geometry': {'status': 'UNVERIFIED', 'support_residual_m': residual,
                             'note': 'model-to-fitted-plane residual, not ground-truth error'},
        'functional_geometry': {'status': 'UNVERIFIED'},
        'metric_error_bound_m': None,
        'metric_contact_ready': False,
    }


def compare_validation_views(before, after, roles, *, max_iou_regression=.01):
    """Accept/reject a candidate on withheld views, never optimize on them.

    The tolerance is a declared image-space engineering threshold, not a
    physical accuracy bound. Failure means keep the old version and obtain
    fresh evidence before selecting another candidate on the same test image.
    """
    import math
    if not math.isfinite(max_iou_regression) or not 0 <= max_iou_regression <= 1:
        raise ValueError('invalid heldout regression tolerance')
    held = [v for v, role in roles.items() if role == 'holdout']
    missing = [v for v in held if v not in before or v not in after]
    regressions = [v for v in held if v not in missing and
                   after[v]['triangle_iou']+max_iou_regression < before[v]['triangle_iou']]
    return {'status': 'NEED_FRESH_VALIDATION_VIEW' if missing or regressions else
            'VALIDATION_NOT_REGRESSED' if held else 'NO_INDEPENDENT_VIEW',
            'regressions': regressions, 'missing': missing, 'max_iou_regression': max_iou_regression,
            'next_action': 'retain prior package; inspect cause and acquire fresh validation' if regressions or missing else None}


def audit(root):
    root = Path(root)
    report = read(root/'fitted_world/build_report.json', {})
    quality = {oid: object_quality({'fit': fit}) for oid, fit in report.get('fit', {}).items()}
    return {'schema_version': 'reconstruction-quality-v1', 'scope': 'reconstruction only',
            'validation_views': view_leakage(root), 'objects': quality,
            'metric_contact_ready': False,
            'note': 'A workflow PASS is not a metric or contact accuracy certificate.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output')
    a = p.parse_args()
    text = json.dumps(audit(a.run), indent=2, ensure_ascii=False)+'\n'
    if a.output:
        Path(a.output).write_text(text)
    else:
        print(text)
