"""Observation-derived fixed hardware, shared by scene queries and collision checks.

Setup only: supplement an imported world without re-running reconstruction or
changing its task-object geometry. Components retain empty space between them.
"""
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import trimesh

from .world import atomic, rigid


def setup_fixed_obstacles(world, config, packet, robot):
    """Use the shared observed host fixture on matching calibrated X5 setups.

    Explicit per-scene geometry wins. Never transplant the profile to an
    unrelated camera/frame, and never rebuild an already modeled obstacle.
    """
    specs = {item['object_id']: item for item in config.get('fixed_obstacles', [])}
    profile = json.loads((Path(__file__).parent / 'resources/fixed_obstacles/robodojo_x5.json').read_text())
    view = next((v for v in packet.get('views', []) if v['view_id'] == 'cam_head'), None)
    matched = (robot in {'robodojo', 'robodojo_xpl', 'robodojo_xpl_rgb'} and
               packet.get('frame') == profile['frame'] and view is not None and
               np.allclose(view['T_W_C'], profile['reference_camera']['T_W_C'], atol=1e-6, rtol=0))
    if matched:
        for spec in profile['obstacles']:
            specs.setdefault(spec['object_id'], spec)
    existing = sorted(set(specs) & set(world.objects()))
    added = add_fixed_obstacles(world, [s for oid, s in specs.items() if oid not in existing])
    return {'profile': profile['profile'], 'matched_calibration': bool(matched),
            'added': added, 'retained_existing': existing,
            'reason': 'shared observed fixture' if matched else 'different frame/camera; explicit scene geometry only'}


def add_fixed_obstacles(world, specs):
    if not specs:
        return []
    state = world.state()
    manifest = copy.deepcopy(world.manifest)
    prepared = {}
    for spec in specs:
        oid = spec['object_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', oid) or oid in manifest['objects'] or oid in prepared:
            raise ValueError('Fixed obstacle must have a new unique object ID')
        if not spec.get('evidence') or not spec.get('basis'):
            raise ValueError('Describe the fixed obstacle geometry source')
        transform = rigid(spec.get('T_W_object', np.eye(4))).tolist()
        components = []
        for part in spec['boxes']:
            size = np.asarray(part['size_m'], float)
            if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
                raise ValueError('Positive finite box dimensions required')
            mesh = trimesh.creation.box(size)
            mesh.apply_transform(rigid(part['T_object_box']))
            components.append(mesh)
        if not components:
            raise ValueError('Fixed hardware requires geometry')
        prepared[oid] = (spec, transform, trimesh.util.concatenate(components))
    for oid, (spec, transform, mesh) in prepared.items():
        geometry = f'geometry/{oid}.glb'
        mesh.export(world.root / geometry)
        meta_path = f'objects/{oid}.json'
        meta = {'object_id': oid, 'label': spec.get('label', oid),
                'role': 'fixed_obstacle', 'movable': False,
                'geometry': {'local_mesh': geometry, 'local_bounds_m': mesh.bounds.tolist()},
                'initial_T_W_object': transform, 'source_nodes': [],
                'declared': {'movable': False, 'unknowns': spec.get('unknowns', []),
                             'components': copy.deepcopy(spec['boxes'])},
                'basis': spec['basis'], 'evidence': copy.deepcopy(spec['evidence'])}
        atomic(world.root / meta_path, meta)
        manifest['objects'][oid] = meta_path
        for rel in (geometry, meta_path):
            manifest['bound_files'][rel] = hashlib.sha256((world.root / rel).read_bytes()).hexdigest()
        state['objects'][oid] = {'T_W_object': transform, 'basis': spec['basis'],
                                 'evidence': copy.deepcopy(spec['evidence'])}
    revision = hashlib.sha256(json.dumps(manifest['bound_files'], sort_keys=True).encode()).hexdigest()
    manifest['geometry_revision'] = state['geometry_revision'] = revision
    state['revision'] += 1
    state['history'].append({'revision': state['revision'], 'kind': 'fixed_obstacles_added',
                             'objects': list(prepared), 'recorded_at_ns': time.time_ns()})
    atomic(world.root / 'world.json', manifest)
    atomic(world.state_path, state)
    world.manifest = manifest
    world.verify_binding()
    return list(prepared)
