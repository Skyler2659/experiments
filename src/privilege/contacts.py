"""Task-selected local contact regions, independent of task names or world axes."""
import copy
import numpy as np


def validate_contacts(rules, world, robot):
    state = world.state()
    result = []
    for rule in rules or []:
        if set(rule) - {'first', 'second', 'reason'} or not {'first', 'second'} <= set(rule):
            raise ValueError('Contact requires first/second selectors and optional reason')
        item = copy.deepcopy(rule)
        for side in ('first', 'second'):
            selector = item[side]
            if not isinstance(selector, dict):
                raise ValueError('Contact selector must be a mapping, e.g. object + bounds')
            if 'local_bounds' in selector:
                if 'bounds' in selector:
                    raise ValueError('Specify bounds or its alias local_bounds, not both')
                selector['bounds'] = selector.pop('local_bounds')
            if 'object' in selector:
                if set(selector) - {'object', 'part', 'bounds'}:
                    raise ValueError('Scene contact selector uses object, part and/or bounds')
                if selector['object'] not in state['objects']:
                    raise ValueError(f"Unknown contact object {selector['object']!r}")
                row = state['objects'][selector['object']]
                if 'part' in selector:
                    parts = row.get('assembly', {}).get('parts', {})
                    if selector['part'] not in parts:
                        raise ValueError(f"Object {selector['object']!r} has no part {selector['part']!r}; "
                                         f"available parts: {list(parts)}. Use object + bounds for "
                                         "an unpartitioned mesh, or publish_parts first.")
                elif 'bounds' not in selector:
                    raise ValueError('Scene contact needs a named part or local bounds')
            else:
                if set(selector) - {'arm', 'link', 'bounds'} or not {'arm', 'link'} <= set(selector):
                    raise ValueError('Robot contact selector uses arm and link')
                if selector['arm'] not in robot.spec['bases'] or selector['link'] not in {k[0] for k in robot.meshes}:
                    raise ValueError('Unknown robot contact link')
            if 'bounds' in selector:
                bounds = np.asarray(selector['bounds'], dtype=float)
                if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                    raise ValueError('Contact bounds must be finite local min/max corners')
                selector['bounds'] = bounds.tolist()
        if 'object' not in item['first'] and 'object' not in item['second']:
            raise ValueError('Task contacts do not exempt robot self-collision')
        item['_geometry'] = [geometry_signature(item[s], state) for s in ('first', 'second')]
        result.append(item)
    return result


def geometry_signature(selector, state):
    if 'object' not in selector:
        return None
    row = state['objects'].get(selector['object'], {})
    parts = row.get('assembly', {}).get('parts', {})
    if 'part' in selector:
        parts = {selector['part']: parts.get(selector['part'])}
    return {pid: None if part is None else {k: part.get(k) for k in ('sha256', 'local_mesh')}
            for pid, part in parts.items()}


def matches(selector, body):
    if 'object' in selector:
        return (body['link'] is None and selector['object'] == body['name'] and
                ('part' not in selector or selector['part'] == body['part']))
    return selector['arm'] == body['arm'] and selector['link'] == body['link']


def permits(rules, a, b, fcl, cache):
    """Subtract the union of declared contact volumes from intersecting surfaces.

    A rule's two local regions are intersected in space. Rules for this exact
    body/part pair are unioned, so multiple intended patches can coexist without
    permitting crossed pairs or contact elsewhere on a coarse triangle.
    """
    relevant = [(rule[x], rule[y]) for rule in rules for x, y in
                (('first', 'second'), ('second', 'first'))
                if matches(rule[x], a) and matches(rule[y], b)]
    if not relevant:
        return False
    def intersects(first, second):
        return bool(fcl.collide(first, second, fcl.CollisionRequest(), fcl.CollisionResult()))
    if not intersects(a['body'], b['body']):
        return False  # Solid containment without surface contact is not exempt.
    volumes = []
    for sa, sb in relevant:
        planes = []
        for selector, body in ((sa, a), (sb, b)):
            if 'bounds' not in selector:
                continue
            frame = body['T'] if 'part' in selector or body['link'] else body['object_T']
            relative = np.linalg.inv(a['T']) @ frame
            bounds = np.asarray(selector['bounds'])
            for axis in range(3):
                for side in (0, 1):
                    normal = relative[:3, axis] * (1 if side == 0 else -1)
                    origin = relative[:3, 3] + relative[:3, axis] * bounds[side, axis]
                    planes.append((origin - normal * 1e-9, normal))
        if not planes:
            return True  # Both exact parts/links were explicitly selected.
        volumes.append(planes)
    key = (id(a['mesh']), tuple(tuple(tuple(np.round(np.r_[o, n], 10)) for o, n in v) for v in volumes))
    if key not in cache:
        import trimesh
        def slice_surface(mesh, origin, normal):
            vertices, faces, _ = trimesh.intersections.slice_faces_plane(
                mesh.vertices, mesh.faces, normal, origin)
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        remaining = a['mesh'].copy()
        for planes in volumes:
            inside, pieces = remaining, []
            for origin, normal in planes:
                if not len(inside.faces):
                    break
                outside = slice_surface(inside, origin, -normal)
                if len(outside.faces):
                    pieces.append(outside)
                inside = slice_surface(inside, origin, normal)
            if not pieces:
                remaining = None
                break
            remaining = trimesh.util.concatenate(pieces)
        if remaining is None:
            cache[key] = None
        else:
            model = fcl.BVHModel()
            model.beginModel(len(remaining.vertices), len(remaining.faces))
            model.addSubModel(np.asarray(remaining.vertices, float), np.asarray(remaining.faces, np.int32))
            model.endModel()
            cache[key] = model
    model = cache[key]
    if model is None:
        return True
    t = a['T']
    outside = fcl.CollisionObject(model, fcl.Transform(t[:3, :3], t[:3, 3]))
    return not intersects(outside, b['body'])
