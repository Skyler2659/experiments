"""Sampled URDF/scene/payload collision queries using public robot feedback only."""
import copy
import itertools
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .robot_assets import _origin
from .world import read, rigid


class RobotGeometry:
    """URDF collision meshes and FK; no IK or simulator access."""
    def __init__(self, spec):
        self.spec = spec
        path = Path(spec['urdf']).resolve()
        xml = ET.parse(path).getroot()
        self.joints = xml.findall('joint')
        roots = {x.get('name') for x in xml.findall('link')} - {
            j.find('child').get('link') for j in self.joints}
        if len(roots) != 1:
            raise ValueError('Collision URDF needs one root')
        self.root = roots.pop()
        self.neighbors = {frozenset((j.find('parent').get('link'), j.find('child').get('link')))
                          for j in self.joints}
        self.fingers = tuple(spec['finger_links'])
        self.neighbors.add(frozenset(self.fingers))
        # Fixed accessories share the collision body/adjacency of their parent.
        for _ in self.joints:
            for joint in self.joints:
                if joint.get('type') != 'fixed':
                    continue
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                for pair in list(self.neighbors):
                    if parent in pair:
                        self.neighbors.add(frozenset((child, *(pair - {parent}))))
        self.meshes = {}
        self.visual_fallback_links = []
        for link in xml.findall('link'):
            elements = link.findall('collision')
            if not elements and link.findall('visual'):
                elements = link.findall('visual')
                self.visual_fallback_links.append(link.get('name'))
            for i, element in enumerate(elements):
                g = element.find('geometry')
                source = g.find('mesh')
                if source is not None:
                    filename = source.get('filename')
                    if filename.startswith('package://'):
                        package, relative = filename[10:].split('/', 1)
                        filename = Path(spec['packages'][package]) / relative
                    else:
                        filename = path.parent / filename.removeprefix('file://')
                    mesh = trimesh.load(filename, force='scene', process=False).to_geometry()
                    mesh.apply_scale(np.fromstring(source.get('scale', '1 1 1'), sep=' '))
                elif g.find('box') is not None:
                    mesh = trimesh.creation.box(np.fromstring(g.find('box').get('size'), sep=' '))
                elif g.find('sphere') is not None:
                    mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(g.find('sphere').get('radius')))
                elif g.find('cylinder') is not None:
                    c = g.find('cylinder')
                    mesh = trimesh.creation.cylinder(float(c.get('radius')), float(c.get('length')))
                else:
                    raise ValueError('Unsupported collision geometry')
                mesh.apply_transform(_origin(element.find('origin')))
                self.meshes[(link.get('name'), i)] = mesh
        if not self.meshes:
            raise ValueError('Robot has no collision geometry')

    def frames(self, arm, q, opening):
        values = dict(zip(self.spec['joint_names'], q, strict=True))
        values.update({name: opening * self.spec['finger_scale_m'] + self.spec.get('finger_offset_m', 0.)
                       for name in self.spec['finger_joints']})
        transforms = {self.root: rigid(self.spec['bases'][arm])}
        pending = list(self.joints)
        while pending:
            ready = [j for j in pending if j.find('parent').get('link') in transforms and
                     (j.find('mimic') is None or j.find('mimic').get('joint') in values)]
            if not ready:
                raise ValueError('Disconnected/unbound URDF')
            for j in ready:
                name, kind = j.get('name'), j.get('type')
                mimic = j.find('mimic')
                if mimic is not None:
                    values[name] = values[mimic.get('joint')] * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
                delta = np.eye(4)
                if kind != 'fixed':
                    a = j.find('axis')
                    axis = np.fromstring(a.get('xyz', '1 0 0') if a is not None else '1 0 0', sep=' ')
                    axis /= np.linalg.norm(axis)
                    if kind in ('revolute', 'continuous'):
                        delta[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
                    elif kind == 'prismatic':
                        delta[:3, 3] = axis * values[name]
                    else:
                        raise ValueError('Unsupported joint ' + kind)
                transforms[j.find('child').get('link')] = (
                    transforms[j.find('parent').get('link')] @ _origin(j.find('origin')) @ delta)
                pending.remove(j)
        return transforms


def x5_spec(urdf):
    """Native IK bases and robot_config.yml gripper_scale [-.01, .044]."""
    from pyrualean.hosts import robodojo_kinematics as kin
    return {'urdf': str(urdf), 'bases': {a: kin._BASE_T[a].tolist() for a in kin.ARMS},
            'joint_names': [f'joint{i}' for i in range(1, 7)], 'ee_link': 'link6',
            'finger_joints': ['joint7', 'joint8'], 'finger_links': ['link7', 'link8'],
            'finger_scale_m': .054, 'finger_offset_m': -.01}


class SceneCollision:
    def __init__(self, world, robot_spec, *, max_joint_increment=.025, max_finger_increment_m=.002):
        import fcl
        self.fcl, self.world = fcl, world
        self.robot = RobotGeometry(robot_spec)
        self.max_joint_increment = float(max_joint_increment)
        self.max_finger_increment_m = float(max_finger_increment_m)
        if not all(np.isfinite(x) and x > 0 for x in (self.max_joint_increment, self.max_finger_increment_m)):
            raise ValueError('Positive finite collision sampling increments required')
        self._cache = {}
        self.contacts = {'grasp': {}, 'place': {}, 'contacts': []}
        self.last_report = None
        self._in_action = False
        self._released_poses = {}
        self._last_carried = {}

    def begin_action(self):
        self._in_action = True
        self._released_poses = {}
        self._last_carried = {}

    def end_action(self):
        self._in_action = False
        self._released_poses = {}
        self._last_carried = {}
        self.expect_contacts()

    def expect_contacts(self, *, grasp=None, place=None, contacts=None):
        from .contacts import validate_contacts
        intents = {'grasp': dict(grasp or {}), 'place': dict(place or {})}
        for targets in intents.values():
            for arm, oid in targets.items():
                if arm not in self.robot.spec['bases'] or oid not in self.world.objects():
                    raise ValueError('Contact must name an arm and current scene object')
        intents['contacts'] = validate_contacts(contacts, self.world, self.robot)
        self.contacts = intents

    def _geometry(self, key, mesh):
        if key not in self._cache:
            mesh = mesh.copy()
            mesh.merge_vertices()
            geom = self.fcl.BVHModel()
            geom.beginModel(len(mesh.vertices), len(mesh.faces))
            geom.addSubModel(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32))
            geom.endModel()
            components = list(mesh.split(only_watertight=False, repair=False))
            representatives = np.array([part.vertices[0] for part in components])
            solids = [part for part in components if part.is_watertight]
            self._cache[key] = (geom, mesh, trimesh.bounds.corners(mesh.bounds), representatives, solids)
        return self._cache[key]

    def check_layout(self, poses):
        """Optional assembly-site query against fixed hardware, using actual meshes.

        No AABB exclusion zone, padding, state mutation or robot action is added.
        Table/support contact and inter-piece stability are outside this query.
        """
        state = self.world.state()
        fixed = [oid for oid, path in self.world.manifest['objects'].items()
                 if read(self.world.root / path).get('role') == 'fixed_obstacle']
        proposed = {}
        for oid, transform in poses.items():
            if oid in fixed:
                raise ValueError('Cannot relocate fixed hardware in an assembly query')
            local = copy.deepcopy(state['objects'][oid])
            local['T_W_object'] = rigid(transform).tolist()
            proposed[oid] = self.world._mesh(oid, {'objects': {oid: local}})
        fixtures = {oid: self.world._mesh(oid, state) for oid in fixed}
        hits = []
        clearances = []
        for oid, mesh in proposed.items():
            for fixture, other in fixtures.items():
                geometries = []
                for m in (mesh, other):
                    m = m.copy(); m.merge_vertices()
                    model = self.fcl.BVHModel()
                    model.beginModel(len(m.vertices), len(m.faces))
                    model.addSubModel(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
                    model.endModel()
                    geometries.append(self.fcl.CollisionObject(model))
                intersects = bool(self.fcl.collide(*geometries, self.fcl.CollisionRequest(), self.fcl.CollisionResult()))
                # A closed fixture may fully enclose an object without surface crossing.
                if not intersects:
                    parts = []
                    for m in (mesh, other):
                        m = m.copy(); m.merge_vertices()
                        parts.append(list(m.split(only_watertight=False, repair=False)))
                    intersects = any(p.is_watertight and p.contains([q.vertices[0]]).any()
                                     for outer, inner in (parts, parts[::-1]) for p in outer for q in inner)
                if intersects:
                    hits.append({'first': oid, 'second': fixture})
                distance = 0.0 if intersects else max(0.0, float(self.fcl.distance(
                    *geometries, self.fcl.DistanceRequest(), self.fcl.DistanceResult())))
                clearances.append({'object_id': oid, 'obstacle': fixture, 'distance_m': distance})
        return {'status': 'COLLISION' if hits else 'CLEAR', 'hits': hits,
                'world_revision': state['revision'], 'fixed_obstacles': fixed,
                'clearances': clearances,
                'coverage': 'proposed_object_meshes_vs_fixed_hardware_only'}

    def check(self, raw, actions, *, phase='query'):
        """Actions use native joint/opening targets. Other arms remain at measured joints.

        CLEAR means sampled reconstructed-mesh clearance, not continuous or true-scene safety.
        No observations, robot actions, geometry updates or sleeps are performed here.
        """
        state = self.world.state()
        arms = tuple(self.robot.spec['bases'])
        current = {a: np.asarray(raw[f'{a}_arm_joint_state'], float) for a in arms}
        openings = {a: float(np.asarray(raw[f'{a}_ee_joint_state']).reshape(-1)[0]) for a in arms}
        if any(q.shape != (len(self.robot.spec['joint_names']),) or not np.isfinite(q).all() for q in current.values()):
            raise ValueError('Invalid measured joints')
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in openings.values()):
            raise ValueError('Invalid measured gripper openings')
        # A measured empty hand cannot transport geometry, even within a primitive.
        release_poses = copy.deepcopy(self._released_poses) if self._in_action else {}
        owners = state.get('attachments', {})
        for a, h in owners.items():
            if openings.get(a, 0) == 0 and h['object_id'] in self._last_carried:
                release_poses.setdefault(h['object_id'], self._last_carried[h['object_id']].copy())
        attached = {a: h for a, h in owners.items() if openings.get(a, 0) > 0}
        # Detaching a payload does not make an existing finger contact vanish.
        # Keep only that contact while opening in place in the immediately next
        # primitive. No arm motion, new contact pair, or other obstacle is exempt.
        step = state.get('tracking', {}).get('robot_state', {}).get('step')
        opening_in_place = {a for a in arms if actions and
            all(all(np.array_equal(action.get(f'{b}_arm_joint_state', current[b]), current[b]) for b in arms)
                for action in actions) and
            all(float(np.asarray(action.get(f'{a}_ee_joint_state', [openings[a]])).reshape(-1)[0]) >= openings[a]
                for action in actions) and
            any(float(np.asarray(action.get(f'{a}_ee_joint_state', [openings[a]])).reshape(-1)[0]) > openings[a]
                for action in actions)}
        released_contacts = {(event['arm'], oid) for oid, event in state.get('releases', {}).items()
                             if step is not None and event.get('host_step') == step and
                             event['arm'] in opening_in_place}
        release_pairs = set()
        from .contacts import geometry_signature, permits
        rules, invalidated = [], []
        contact_region_cache = {}
        for rule in self.contacts['contacts']:
            current_geometry = [geometry_signature(rule[side], state) for side in ('first', 'second')]
            (rules if current_geometry == rule['_geometry'] else invalidated).append(rule)
        object_geometry = {}
        for oid, row in state['objects'].items():
            meta = read(self.world.root / self.world.manifest['objects'][oid])
            if meta.get('role') == 'robot_snapshot':
                continue
            import json
            parts = row.get('assembly', {}).get('parts', {})
            for pid in (parts or {None: None}):
                local = copy.deepcopy(row); local['T_W_object'] = np.eye(4).tolist()
                if pid is None:
                    key = ('object', oid, 'null')
                    mesh = lambda: self.world._mesh(oid, {'objects': {oid: local}})
                else:
                    local['assembly']['parts'][pid]['T_object_part'] = np.eye(4).tolist()
                    key = ('part', oid, pid, json.dumps({k: parts[pid].get(k)
                           for k in ('sha256', 'local_mesh')}, sort_keys=True))
                    mesh = lambda: self.world._part_mesh(oid, pid, {'objects': {oid: local}})
                if key not in self._cache:
                    self._geometry(key, mesh())
                object_geometry[(oid, pid)] = key
        for key, mesh in self.robot.meshes.items():
            self._geometry(('robot', *key), mesh)
        contacts_seen = []
        escaping = {}
        escape_seen = []
        unmoved_initial = {}
        unchanged_seen = []
        samples = 0
        def body(name, key, t, arm=None, link=None, payload=False, part=None, object_t=None):
            geom, mesh, corners, representatives, solids = self._cache[key]
            points = corners @ t[:3, :3].T + t[:3, 3]
            return {'name': name, 'body': self.fcl.CollisionObject(geom, self.fcl.Transform(t[:3, :3], t[:3, 3])),
                    'bounds': np.array([points.min(0), points.max(0)]), 'mesh': mesh, 'T': t,
                    'arm': arm, 'link': link, 'payload': payload, 'part': part,
                    'object_T': t if object_t is None else object_t,
                    'representatives': representatives, 'solids': solids}
        def intersect(a, b):
            if np.any(a['bounds'][1] < b['bounds'][0]) or np.any(b['bounds'][1] < a['bounds'][0]):
                return False
            if self.fcl.collide(a['body'], b['body'], self.fcl.CollisionRequest(), self.fcl.CollisionResult()):
                return True
            # Surface intersection alone misses a solid wholly inside another.
            for outer, inner in ((a, b), (b, a)):
                if outer['solids']:
                    t = np.linalg.inv(outer['T']) @ inner['T']
                    p = inner['representatives'] @ t[:3, :3].T + t[:3, 3]
                    if any(part.contains(p).any() for part in outer['solids']):
                        return True
            return False
        def inspect(qs, gs, terminal, initial=False):
            nonlocal samples
            samples += 1
            robot_bodies, objects = [], []
            frames = {a: self.robot.frames(a, qs[a], gs[a]) for a in arms}
            for a in arms:
                for link, idx in self.robot.meshes:
                    robot_bodies.append(body(f'{a}/{link}', ('robot', link, idx), frames[a][link], a, link))
            held_by = {h['object_id']: a for a, h in attached.items() if h['object_id'] not in release_poses}
            for (oid, pid), key in object_geometry.items():
                arm = held_by.get(oid)
                t = rigid(state['objects'][oid]['T_W_object'])
                if arm:
                    t = frames[arm][self.robot.spec['ee_link']] @ rigid(attached[arm]['T_E_object'])
                elif oid in release_poses:
                    t = release_poses[oid]
                part_t = t if pid is None else t @ rigid(state['objects'][oid]['assembly']['parts'][pid]['T_object_part'])
                objects.append(body(oid, key, part_t, arm, payload=arm is not None, part=pid, object_t=t))
            hits = []
            pairs = itertools.chain(itertools.combinations(robot_bodies, 2),
                                    itertools.product(robot_bodies, objects),
                                    ((a, b) for a, b in itertools.combinations(objects, 2) if a['name'] != b['name'] and (a['payload'] or b['payload'])))
            for a, b in pairs:
                if a['link'] and b['link'] and a['arm'] == b['arm']:
                    if a['link'] == b['link'] or frozenset((a['link'], b['link'])) in self.robot.neighbors:
                        continue
                # Only fixed root/table mounting contacts are permanent exclusions.
                if a['link'] == self.robot.root and b['name'] == state.get('height_alignment', {}).get('table_object_id', 'tabletop'):
                    continue
                expected = False
                if a['link'] in self.robot.fingers:
                    expected = ((owners.get(a['arm'], {}).get('object_id') == b['name']) or
                                self.contacts['grasp'].get(a['arm']) == b['name'])
                if not intersect(a, b):
                    pair_key = (a['name'], a['part'], a['link'], b['name'], b['part'], b['link'])
                    escaping.pop(pair_key, None)
                    unmoved_initial.pop(pair_key, None)
                    continue
                if a['link'] in self.robot.fingers and (a['arm'], b['name']) in released_contacts:
                    pair_key = (a['name'], a['part'], a['link'], b['name'], b['part'], b['link'])
                    if initial:
                        release_pairs.add(pair_key)
                    expected = expected or pair_key in release_pairs
                # Placement contact only at the final query pose, shallow at support top.
                for payload, support in ((a, b), (b, a)):
                    if (terminal and payload['payload'] and self.contacts['place'].get(payload['arm']) == support['name']
                            and payload['bounds'][0, 2] >= support['bounds'][1, 2] - .002):
                        expected = True
                expected = expected or permits(rules, a, b, self.fcl, contact_region_cache)
                pair = {'first': a['name'], 'second': b['name']}
                if a['part'] is not None: pair['first_part'] = a['part']
                if b['part'] is not None: pair['second_part'] = b['part']
                if expected:
                    if pair not in contacts_seen:
                        contacts_seen.append(pair)
                else:
                    key = (a['name'], a['part'], a['link'], b['name'], b['part'], b['link'])
                    overlap = float(np.min(np.minimum(a['bounds'][1], b['bounds'][1]) -
                                           np.maximum(a['bounds'][0], b['bounds'][0])))
                    if initial and actions:
                        escaping[key] = (overlap, a['T'].copy(), b['T'].copy())
                        unmoved_initial[key] = (a['T'].copy(), b['T'].copy())
                        if pair not in escape_seen:
                            escape_seen.append(pair)
                        continue
                    baseline = unmoved_initial.get(key)
                    if baseline is not None:
                        if (np.allclose(a['T'], baseline[0], atol=1e-10, rtol=0) and
                                np.allclose(b['T'], baseline[1], atol=1e-10, rtol=0)):
                            if pair not in unchanged_seen:
                                unchanged_seen.append(pair)
                            continue  # An unchanged pair cannot invalidate another arm's endpoint.
                        unmoved_initial.pop(key)
                    previous = escaping.get(key)
                    if previous is not None and not terminal:
                        depth, ta, tb = previous
                        unchanged = np.allclose(a['T'], ta, atol=1e-10, rtol=0) and np.allclose(b['T'], tb, atol=1e-10, rtol=0)
                        translation_only = (np.allclose(a['T'][:3,:3], ta[:3,:3], atol=1e-8, rtol=0) and
                                            np.allclose(b['T'][:3,:3], tb[:3,:3], atol=1e-8, rtol=0))
                        if unchanged or (translation_only and overlap < depth - 1e-10):
                            escaping[key] = (overlap, a['T'].copy(), b['T'].copy())
                            continue
                    hits.append(pair)
            return hits
        initial_frames = {a: self.robot.frames(a, current[a], openings[a]) for a in arms}
        hits = inspect(current, openings, not actions, initial=True)
        for segment, action in enumerate(actions):
            if hits:
                break
            target = {a: np.asarray(action.get(f'{a}_arm_joint_state', current[a]), float) for a in arms}
            goal_opening = {a: float(np.asarray(action.get(f'{a}_ee_joint_state', [openings[a]])).reshape(-1)[0]) for a in arms}
            if any(q.shape != current[a].shape or not np.isfinite(q).all() for a, q in target.items()):
                raise ValueError('Invalid collision path joints')
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in goal_opening.values()):
                raise ValueError('Invalid collision path openings')
            for a in attached:
                if goal_opening[a] >= .9:
                    # A multi-segment query may carry first and open later.
                    # Anchor release at this segment, not the whole path's start.
                    release_frame = self.robot.frames(a, current[a], openings[a])[self.robot.spec['ee_link']]
                    release_poses.setdefault(attached[a]['object_id'],
                        release_frame @ rigid(attached[a]['T_E_object']))
                elif attached[a]['object_id'] not in release_poses:
                    # Closing against the held object does not make fingers pass through it.
                    goal_opening[a] = max(goal_opening[a], openings[a])
            count = max(1, math.ceil(max(float(np.max(abs(target[a]-current[a]))) for a in arms)/self.max_joint_increment),
                        math.ceil(max(abs(goal_opening[a]-openings[a]) for a in arms)*self.robot.spec['finger_scale_m']/self.max_finger_increment_m))
            for i in range(1, count+1):
                u = i/count
                qs = {a: current[a]+u*(target[a]-current[a]) for a in arms}
                gs = {a: openings[a]+u*(goal_opening[a]-openings[a]) for a in arms}
                hits = inspect(qs, gs, segment == len(actions)-1 and i == count)
                if hits:
                    break
            current, openings = target, goal_opening
        result = {'status': 'COLLISION' if hits else 'CLEAR', 'hits': hits[:12], 'samples_checked': samples,
                  'geometry_accuracy_verified': False,
                  'state_revision': state['revision'], 'phase': phase, 'expected_contacts': contacts_seen,
                  'initial_contacts_to_escape': escape_seen,
                  'unchanged_initial_contacts': unchanged_seen,
                  'invalidated_contact_intents': invalidated,
                  'coverage': 'sampled_URDF_and_reconstructed_meshes', 'max_joint_increment_rad': self.max_joint_increment,
                  'max_finger_increment_m': self.max_finger_increment_m,
                  'visual_fallback_links': self.robot.visual_fallback_links,
                  'unresolved_objects': [oid for oid, row in state['objects'].items()
                                         if row.get('review', {}).get('pose', {}).get('status') == 'unresolved']}
        self.last_report = result
        if self._in_action and phase == 'native_segment' and not hits:
            self._released_poses = release_poses
            self._last_carried.update({h['object_id']: initial_frames[a][self.robot.spec['ee_link']] @ rigid(h['T_E_object'])
                                       for a, h in attached.items() if h['object_id'] not in release_poses})
        return result
