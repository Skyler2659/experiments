"""Observation-bound functional regions on the same textured mesh.

Agents choose semantics, regions and relations; numeric code measures them.
Atomic objects use exactly the same interface as compound objects. Features
remain candidate geometry even after visual review, never certified contacts.
"""
import copy
import json
from pathlib import Path
import numpy as np
from .mesh_world import sha256
from .structure_program import select, write

FEATURE_COMPILER_VERSION = 3


def finite(value, name, positive=False):
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError('invalid '+name)
    return value


def direction(value):
    v = np.asarray(value, float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
        raise ValueError('invalid feature direction')
    return v/np.linalg.norm(v)


def compile_features(world, oid, proposal):
    from .structure_features import object_frame
    if proposal.get('object_id') != oid:
        raise ValueError('feature object mismatch')
    mesh = world.mesh(oid)
    observations = {v['view_id']: v for v in world.source.get('observations', [])}
    reference = object_frame(world, oid)
    output, ids = [], set()
    for spec in proposal.get('features', []):
        fid = spec['id']
        if not isinstance(fid, str) or not fid or fid in ids or not spec.get('purpose'):
            raise ValueError('feature needs unique ID and functional purpose')
        ids.add(fid)
        kind = spec['kind']
        if kind not in {'surface_patch', 'plane', 'axis', 'opening'}:
            raise ValueError('unknown functional feature kind')
        evidence = spec.get('evidence', [])
        if len({v['view_id'] for v in evidence}) < 2:
            raise ValueError('feature needs two observation views; request targeted capture')
        for item in evidence:
            view = observations[item['view_id']]
            if view.get('role') == 'holdout':
                raise ValueError('functional proposal may not consume held-out observations')
            if item['rgb_sha256'] != view['rgb_sha256'] or not item.get('reason'):
                raise ValueError('stale or unexplained feature evidence')
        selected, _ = select(mesh, spec['selector'], reference)
        face_ids = np.flatnonzero(selected)
        if not len(face_ids):
            raise ValueError('empty functional region')
        patch = mesh.submesh([face_ids], append=True, repair=False)
        points = np.asarray(patch.vertices)
        center = points.mean(0)
        _, singular, vt = np.linalg.svd(points-center, full_matrices=False)
        normal = vt[-1]
        if np.dot(normal, direction(spec.get('direction_hint_W', normal))) < 0:
            normal = -normal
        u = vt[0]; v = np.cross(normal, u)
        projected = (points-center) @ np.column_stack([u, v])
        geometry = {'center_W_m': center.tolist(), 'normal_W': normal.tolist(),
                    'axis_W': vt[0].tolist(), 'tangent_u_W': u.tolist(), 'tangent_v_W': v.tolist(),
                    'bounds_W_m': patch.bounds.tolist(), 'area_m2': float(patch.area),
                    'plane_rms_m': float(np.sqrt(np.mean(((points-center)@normal)**2))),
                    'plane_max_error_m': float(np.max(np.abs((points-center)@normal))),
                    'extent_in_plane_m': np.ptp(projected, axis=0).tolist()}
        geometry['axis_separation_ratio'] = float(singular[0]/max(singular[1], 1e-12))
        if kind == 'axis':
            hint = direction(spec.get('direction_hint_W', vt[0]))
            axis_index = int(np.argmax(np.abs(vt@hint)))
            axis = vt[axis_index]
            if np.dot(axis, hint) < 0:
                axis = -axis
            geometry['axis_W'] = axis.tolist()
            geometry['axis_principal_component'] = axis_index
            geometry['axis_hint_deviation_deg'] = float(np.degrees(np.arccos(np.clip(np.dot(axis,hint),-1,1))))
        if kind in {'plane', 'opening'}:
            tolerance = finite(spec['planarity_tolerance_m'], 'plane tolerance', True)
            if geometry['plane_max_error_m'] > tolerance:
                raise ValueError(f'{fid}: nonplanar selected region; revise selector or shape')
        if kind == 'opening':
            # Boundary cycles are measured from welded diagnostic geometry only;
            # the textured mesh and its UV seams remain unchanged.
            diagnostic = patch.copy(); diagnostic.merge_vertices()
            outline = diagnostic.outline()
            loops = [np.asarray(p) for p in outline.discrete if len(p) >= 4 and np.linalg.norm(p[0]-p[-1]) < 1e-6]
            loops.sort(key=lambda p: -float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
            index = spec.get('boundary_loop_index')
            if type(index) is not int or not 0 <= index < len(loops):
                raise ValueError(f'{fid}: choose an observed boundary cycle among {len(loops)}; no invented opening')
            geometry['boundary_W_m'] = loops[index].tolist()
            geometry['boundary_cycle_count'] = len(loops)
            geometry['free_space_verified'] = False
        output.append({'id': fid, 'requirement_id': spec.get('requirement_id',fid), 'kind': kind, 'purpose': spec['purpose'], 'selector': spec['selector'],
                       'source_face_ids': face_ids.tolist(), 'geometry': geometry, 'evidence': evidence,
                       'state': 'GEOMETRY_CANDIDATE', 'metric_error_bound_m': None})
    if not output and not proposal.get('not_applicable_reason') and not proposal.get('unresolved_requirements'):
        raise ValueError('empty features require a reason; uncertain objects request observation')
    by_id = {f['id']: f for f in output}
    relations = []
    for relation in proposal.get('relations', []):
        a, b = by_id[relation['a']]['geometry'], by_id[relation['b']]['geometry']
        kind = relation['kind']; tolerance = finite(relation['tolerance'], 'relation tolerance', True)
        if kind in {'parallel', 'perpendicular'}:
            dot = abs(float(np.dot(a['normal_W'], b['normal_W'])))
            angle = float(np.degrees(np.arccos(np.clip(dot, 0, 1))))
            residual = angle if kind == 'parallel' else abs(90-angle)
            unit = 'deg'
        elif kind == 'near':
            # Center distance is explicit, not a contact/connection certificate.
            residual = float(np.linalg.norm(np.array(a['center_W_m'])-b['center_W_m'])); unit = 'm'
        elif kind == 'surface_adjacent':
            from .mesh_world import nearest_mesh_surface
            a_ids = by_id[relation['a']]['source_face_ids']; b_ids = by_id[relation['b']]['source_face_ids']
            a_patch = mesh.submesh([a_ids], append=True, repair=False)
            b_patch = mesh.submesh([b_ids], append=True, repair=False)
            # Sampled separation is reported as an upper bound on minimum
            # distance; a small value establishes proximity, not attachment.
            points = np.asarray(a_patch.vertices)
            points = points[np.linspace(0,len(points)-1,min(len(points),128),dtype=int)]
            residual = min(nearest_mesh_surface(b_patch,p)[1] for p in points); unit = 'm'
        else:
            raise ValueError('unsupported functional relation')
        relations.append({**relation, 'residual': residual, 'unit': unit, 'candidate_satisfied': residual <= tolerance,
                          'physical_attachment_verified': False})
    if any(not r['candidate_satisfied'] for r in relations):
        raise ValueError('functional relation disagrees with source mesh; revise or request asset repair')
    requirements = proposal.get('requirements', [])
    unresolved = proposal.get('unresolved_requirements', [])
    covered = {f.get('requirement_id', f['id']) for f in proposal.get('features', [])}
    unknown = {r['id'] for r in unresolved if r.get('reason') and r.get('view_requirement')}
    expected = {r['id'] for r in requirements}
    if requirements and (not expected <= covered | unknown or not unknown <= expected):
        raise ValueError('functional coverage omitted required regions; do not drop failed features')
    return {'schema_version': 'functional-geometry-v1', 'compiler_version': FEATURE_COMPILER_VERSION, 'object_id': oid,
            'source_pose': world.node(oid)['pose_parent_object'],
            'source_observations': [v for v in world.source.get('observations',[]) if v['view_id'] in {e['view_id'] for f in output for e in f['evidence']}],
            'asset_sha256': world.node(oid)['asset']['sha256'], 'features': output,
            'requirements': requirements, 'unresolved_requirements': unresolved,
            'relations': relations, 'not_applicable_reason': proposal.get('not_applicable_reason'),
            'state': 'NEED_VISUAL_REVIEW', 'metric_contact_ready': False,
            'scope': 'semantic regions measured on reconstructed triangles; actual shape precision inherited'}


def feature_context(data):
    """Keep exact face membership in artifacts, not thousands of tokens in prompts."""
    result = copy.deepcopy(data)
    for feature in result.get('features', []):
        if 'source_face_ids' in feature:
            feature['source_face_count'] = len(feature.pop('source_face_ids'))
    return result


def transform_feature(feature, transform):
    result = copy.deepcopy(feature); g = result['geometry']; t = np.asarray(transform, float)
    for key in ('center_W_m',):
        g[key] = (t[:3, :3]@g[key]+t[:3, 3]).tolist()
    for key in ('normal_W', 'axis_W', 'tangent_u_W', 'tangent_v_W'):
        g[key] = (t[:3, :3]@g[key]).tolist()
    if 'boundary_W_m' in g:
        g['boundary_W_m'] = (np.array(g['boundary_W_m'])@t[:3, :3].T+t[:3, 3]).tolist()
    lo, hi = np.array(g['bounds_W_m'])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = corners@t[:3, :3].T+t[:3, 3]
    g['bounds_W_m'] = [corners.min(0).tolist(), corners.max(0).tolist()]
    g['bounds_kind'] = 'conservative transformed region bound'
    return result


def existing_structure(folder):
    """Expose already reviewed code structure instead of rediscovering it from RGB."""
    parts=Path(folder).parent/'parts_workflow'
    status_path=parts/'status.json'
    if not status_path.exists():
        return {'kind':'atomic','authority':'full original textured mesh; no invented parts'}
    status=json.loads(status_path.read_text())
    if status.get('status')!='STRUCTURALIZED':
        return {'kind':'unresolved','status':status.get('status')}
    path=parts/'rounds'/f"{status['round']:02d}"/'structure.json'
    if not path.exists():return {'kind':'unresolved','status':status.get('status')}
    program=json.loads(path.read_text())
    return {'kind':'reviewed_surface_partition','program_sha256':sha256(path),'nodes':program['nodes'],
            'note':'Reuse these semantic parts and measured bounds to narrow functional regions. Nested selectors refer to their parent region, not always the whole object. These are surface partitions, not closed detachable solids.'}


def run_object(world, oid, folder, *, max_rounds=3):
    """Resumable propose -> numeric extraction -> original-view review -> revise."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'accepted.json').exists():
        accepted = json.loads((folder/'accepted.json').read_text())
        if accepted.get('compiler_version') != FEATURE_COMPILER_VERSION:
            raise ValueError('accepted functional compiler changed: use a new stage-two package')
        node=world.node(oid)
        if accepted.get('asset_sha256')!=node['asset']['sha256'] or accepted.get('source_pose')!=node['pose_parent_object']:
            raise ValueError('accepted functional source changed: rebuild affected object in new package')
        return json.loads((folder/'status.json').read_text())
    requirements_path = folder/'requirements.json'
    if not requirements_path.exists():
        from .structure_features import profile
        write(folder/'inventory_request.json', {'object_id': oid, 'geometry_profile': profile(world,oid),
              'existing_structure':existing_structure(folder),
              'observations':[v for v in world.source.get('observations',[]) if v.get('role')!='holdout'],
              'instruction':'List functional geometry requirements before proposing selectors. Include uncertain requirements; later revisions may not silently drop them.'})
        result={'status':'NEED_FEATURE_INVENTORY','object_id':oid,'round':0}
        write(folder/'status.json',result);return result
    inventory=json.loads(requirements_path.read_text())
    requirements=inventory.get('requirements',[])
    if inventory.get('object_id')!=oid or not isinstance(requirements,list) or not requirements or any(not r.get('id') or not r.get('purpose') for r in requirements) or len({r['id'] for r in requirements})!=len(requirements):
        raise ValueError('functional inventory needs unique requirements and purposes')
    for round_id in range(max_rounds):
        case = folder/f'{round_id:02d}'; case.mkdir(exist_ok=True)
        proposal_path = case/'proposal.json'
        if not proposal_path.exists():
            from .structure_features import profile
            previous = folder/f'{round_id-1:02d}'
            feedback = {name: json.loads((previous/name).read_text()) for name in ('error.json', 'review.json') if (previous/name).exists()}
            write(case/'request.json', {'object_id': oid, 'geometry_profile': profile(world, oid),
                  'existing_structure':existing_structure(folder),
                  'requirements': requirements,
                  'observations': [v for v in world.source.get('observations', []) if v.get('role') != 'holdout'],
                  'feedback': feedback,
                  'contract': 'Choose functional regions from actual mesh and RGB. Uncertain geometry needs observation; do not invent contact accuracy.'})
            result = {'status': 'NEED_FEATURE_PROPOSAL', 'round': round_id, 'object_id': oid}
            write(folder/'status.json', result); return result
        try:
            proposal = json.loads(proposal_path.read_text())
            proposal['requirements']=requirements
            if proposal.get('observation_request'):
                result = {'status': 'NEED_FEATURE_OBSERVATION', 'round': round_id, 'object_id': oid,
                          'request': proposal['observation_request']}
                write(folder/'status.json', result); return result
            candidate = compile_features(world, oid, proposal)
            old = case/'features.json'
            if old.exists() and json.loads(old.read_text()) != candidate:
                old.rename(case/('features_previous_'+sha256(old)[:12]+'.json'))
                if (case/'review.json').exists():
                    review_path=case/'review.json';review_path.rename(case/('review_previous_'+sha256(review_path)[:12]+'.json'))
            write(case/'features.json', candidate)
            render_diagnostics(world, oid, candidate, case)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            write(case/'error.json', {'reason': str(exc), 'route': 'revise region or request source-asset repair'})
            continue
        review = case/'review.json'
        if not review.exists():
            result = {'status': 'NEED_FEATURE_REVIEW', 'round': round_id, 'object_id': oid}
            write(folder/'status.json', result); return result
        decision = json.loads(review.read_text())
        if decision.get('candidate_sha256') != sha256(case/'features.json'):
            raise ValueError('functional review refers to another candidate')
        if decision.get('accepted') and decision.get('reason'):
            result = {**candidate, 'state': 'VISUALLY_REVIEWED_CANDIDATE', 'review': decision}
            write(folder/'accepted.json', result)
            status = {'status': 'FEATURES_REVIEWED', 'round': round_id, 'object_id': oid}
            write(folder/'status.json', status); return status
    result = {'status': 'FEATURE_REPAIR_REQUIRED', 'object_id': oid, 'round': max_rounds-1}
    write(folder/'status.json', result); return result


def render_diagnostics(world, oid, candidate, folder):
    """Project actual selected triangles over original RGB; never generated RGB."""
    from PIL import Image, ImageDraw
    from .surface_visibility import visible_faces
    folder = Path(folder)
    mesh = world.mesh(oid)
    palette = [(230, 80, 80), (70, 180, 240), (90, 220, 110), (220, 120, 240)]
    used = {v['view_id'] for f in candidate['features'] for v in f['evidence']}
    for view in world.source.get('observations', []):
        if view['view_id'] not in used:
            continue
        rgb = world.root/view['rgb_ref']
        if sha256(rgb) != view['rgb_sha256']:
            raise ValueError('feature diagnostic observation changed')
        original = Image.open(rgb).convert('RGB'); width, height = original.size
        panels = [('Original RGB', original)]
        visible = None
        masks = view.get('mask_refs', {})
        if oid in masks:
            visible = np.asarray(Image.open(world.root/masks[oid]).convert('L')) > 127
        elif masks:
            visible = ~np.logical_or.reduce([np.asarray(Image.open(world.root/p).convert('L')) > 127 for p in masks.values()])
        nearest=visible_faces(mesh,view,width,height)
        for i, feature in enumerate(candidate['features']):
            image = original.copy()
            mask = Image.fromarray(np.isin(nearest,feature['source_face_ids']).astype(np.uint8)*100)
            if visible is not None:
                mask = Image.fromarray(np.asarray(mask)*visible.astype(np.uint8))
            image.paste(Image.new('RGB', image.size, palette[i % len(palette)]), (0, 0), mask)
            if feature['kind'] in {'axis','plane','opening'}:
                from .mesh_world import pose_matrix
                center=np.array(feature['geometry']['center_W_m'])
                vector=np.array(feature['geometry']['axis_W'] if feature['kind']=='axis' else feature['geometry']['normal_W'])
                camera=np.linalg.inv(pose_matrix(view['T_W_C']))
                pts=np.array([center,center+.04*vector])@camera[:3,:3].T+camera[:3,3]
                if np.all(pts[:,2]>.01):
                    uv=pts@np.array(view['K']).T;uv=uv[:,:2]/uv[:,2:]
                    line=[tuple(p) for p in uv]
                    draw=ImageDraw.Draw(image);draw.line(line,fill=(255,40,40),width=5)
                    x,y=line[-1];draw.ellipse((x-5,y-5,x+5,y+5),fill=(255,40,40))
            panels.append((feature['id'], image))
        columns=min(3,len(panels));rows=(len(panels)+columns-1)//columns
        canvas=Image.new('RGB',(512*columns,540*rows),'white');draw=ImageDraw.Draw(canvas)
        for i,(label,image) in enumerate(panels):
            x,y=(i%columns)*512,(i//columns)*540
            canvas.paste(image.resize((512,512)),(x,y+28));draw.text((x+6,y+6),label,fill='black')
        canvas.save(folder/(view['view_id']+'_features.png'))
