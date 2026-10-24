"""URDF link-mesh collision queries against the CURRENT reconstructed CodeWorld.

No simulator task-object state is read. Robot/environment, non-adjacent self,
inter-arm and explicitly declared payload surfaces are tested with python-fcl.
Path coverage is sampled joint-space FK, not a continuous collision certificate.
"""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .motion import densify_joint_path
from .kinematics import URDFKinematics, mesh_path


class RobotOnlyScene:
    """No task geometry in the representation ablation; retain robot collisions."""
    def state(self):
        return {'revision': 0, 'objects': {}}

    def objects(self):
        return []


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


class RobotCollision:
    def __init__(self, urdf, bases, world, *, models=None, disabled_self_pairs=None, mounting_contacts=()):
        import fcl
        self.fcl = fcl
        self.world = world
        self.bases = {i: np.asarray(t, float) for i, t in bases.items()}
        path = Path(urdf)
        xml = ET.parse(path).getroot()
        self.models=models or {i:URDFKinematics(path,base=t) for i,t in self.bases.items()}
        if models is None:
            for model in self.models.values():
                if model.tcp_link=='link6':model.T_link_tcp[0,3]=.145
        if set(self.models)!=set(self.bases):raise ValueError('All arms need a robot model binding')
        # Backend-declared installation contacts are persistent, not policy waivers.
        root_links = {link.get('name') for link in xml.findall('link')} - {
            joint.find('child').get('link') for joint in xml.findall('joint')}
        fixed_bodies = {f'arm{arm}/{link}' for arm in self.bases for link in root_links}
        self.mounting_contacts = tuple(tuple(pair) for pair in mounting_contacts)
        for pair in self.mounting_contacts:
            if len(pair) != 2 or pair[0] not in fixed_bodies or pair[1] not in world.objects():
                raise ValueError('Mounting contacts require a fixed root body and a scene object')
        self.joints = []
        self.neighbors = set()
        for j in xml.findall('joint'):
            p, c = j.find('parent').get('link'), j.find('child').get('link')
            self.joints.append((j.get('name'), j.get('type'), p, c, origin(j.find('origin')),
                                np.fromstring(j.find('axis').get('xyz', '1 0 0'), sep=' ') if j.find('axis') is not None else np.array([1., 0, 0])))
            self.neighbors.add(frozenset((p, c)))
        finger_names=set(next(iter(self.models.values())).finger_names)
        finger_links=[j[3] for j in self.joints if j[0] in finger_names]
        if len(finger_links)==2:
            # Opposing fingers intentionally meet at full closure.
            self.neighbors.add(frozenset(finger_links))
        self.neighbors.update(frozenset(p) for p in (disabled_self_pairs or []))
        # A fixed accessory is part of its parent collision body; propagate adjacency.
        for _ in self.joints:
            for _, kind, p, c, _, _ in self.joints:
                if kind == 'fixed':
                    for pair in list(self.neighbors):
                        if p in pair: self.neighbors.add(frozenset((c, *(pair-{p}))))
                        if c in pair: self.neighbors.add(frozenset((p, *(pair-{c}))))
        self.meshes = {}
        for link in xml.findall('link'):
            for idx, col in enumerate(link.findall('collision')):
                g = col.find('geometry')
                if g.find('mesh') is not None:
                    spec = g.find('mesh'); filename = spec.get('filename')
                    mesh = trimesh.load(mesh_path(path,filename), force='mesh', process=False)
                    mesh.apply_scale(np.fromstring(spec.get('scale', '1 1 1'), sep=' '))
                elif g.find('box') is not None:
                    mesh = trimesh.creation.box(np.fromstring(g.find('box').get('size'), sep=' '))
                elif g.find('cylinder') is not None:
                    spec = g.find('cylinder'); mesh = trimesh.creation.cylinder(float(spec.get('radius')), float(spec.get('length')))
                elif g.find('sphere') is not None:
                    mesh = trimesh.creation.icosphere(radius=float(g.find('sphere').get('radius')))
                else:
                    raise ValueError('Unsupported collision geometry')
                mesh.apply_transform(origin(col.find('origin')))
                self.meshes[(link.get('name'), idx)] = mesh
        self.box_corners = {key: trimesh.bounds.corners(mesh.bounds) for key,mesh in self.meshes.items()}
        self.robot_bodies = {(arm,link,idx):self.body(mesh,np.eye(4)) for arm in self.bases for (link,idx),mesh in self.meshes.items()}
        self.scene_cache = None

    def fk(self, arm, q, fingers):
        model=self.models[arm]
        return model.transforms(q,np.asarray(fingers)[:len(model.finger_names)])

    def body(self, mesh, transform):
        f = self.fcl
        geom = f.BVHModel()
        geom.beginModel(len(mesh.vertices), len(mesh.faces))
        geom.addSubModel(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32))
        geom.endModel()
        t = np.asarray(transform)
        return f.CollisionObject(geom, f.Transform(t[:3, :3], t[:3, 3]))

    def check_path(self, starts, frames, fingers, *, allowed_contacts=None, attachments=None, finger_targets=None,
                   max_joint_increment=.025, max_hits=12):
        from codeworld_workspace.world import read
        allowed = {frozenset(pair) for pair in (*self.mounting_contacts, *(allowed_contacts or []))}
        if any(len(pair) != 2 for pair in allowed): raise ValueError('Contact exceptions must name two bodies')
        attachments = attachments or []
        if isinstance(self.world, RobotOnlyScene) and attachments:
            raise ValueError('Object payload geometry is unavailable without CodeWorld')
        state = self.world.state()
        held = {a['object_id'] for a in attachments if a.get('part_id') is None}
        held_parts = {(a['object_id'],a['part_id']) for a in attachments if a.get('part_id') is not None}
        def attachment_name(a):
            return a['object_id'] if a.get('part_id') is None else a['object_id']+'::'+a['part_id']
        cache_key=(state['revision'],frozenset(held),frozenset(held_parts))
        if self.scene_cache is None or self.scene_cache[0]!=cache_key:
            obstacles = {}
            for oid in self.world.objects():
                meta = read(self.world.root/self.world.manifest['objects'][oid])
                if meta.get('role') == 'robot_snapshot' or oid in held: continue
                parts=state['objects'][oid].get('assembly',{}).get('parts')
                if parts:
                    for pid in parts:
                        if (oid,pid) in held_parts:continue
                        mesh=self.world._part_mesh(oid,pid,state)
                        obstacles[oid+'::'+pid]=(mesh,self.body(mesh,np.eye(4)),mesh.bounds)
                else:
                    mesh = self.world._mesh(oid, state)
                    obstacles[oid] = (mesh, self.body(mesh, np.eye(4)), mesh.bounds)
            self.scene_cache=(cache_key,obstacles)
        obstacles=self.scene_cache[1]
        robots = self.robot_bodies
        attached_meshes = {}
        for item in attachments:
            oid = item['object_id']
            if item.get('part_id') is not None:
                pid=item['part_id'];part=state['objects'][oid]['assembly']['parts'][pid]
                mesh=self.world._part_mesh(oid,pid,state)
                frame=np.asarray(state['objects'][oid]['T_W_object'],float)@np.asarray(part['T_object_part'],float)
                if 'T_tcp_part' not in item:raise ValueError('Part attachment requires T_tcp_part')
            else:
                mesh=self.world._mesh(oid,state);frame=np.asarray(state['objects'][oid]['T_W_object'],float)
            mesh.apply_transform(np.linalg.inv(frame))
            attached_meshes[attachment_name(item)] = mesh
        payload_bodies={oid:self.body(mesh,np.eye(4)) for oid,mesh in attached_meshes.items()}
        checks = 0; samples = 0
        def overlap(name_a, body_a, bounds_a, name_b, body_b, bounds_b, hits):
            nonlocal checks
            # Retain explicit whole-object contact compatibility; part names permit
            # narrower exceptions without waiving contacts with the fixed body.
            if any(frozenset((a,b)) in allowed for a in {name_a,name_a.split('::')[0]} for b in {name_b,name_b.split('::')[0]}):return
            if np.any(bounds_a[1] < bounds_b[0]) or np.any(bounds_b[1] < bounds_a[0]): return
            checks += 1
            result = self.fcl.CollisionResult()
            if self.fcl.collide(body_a, body_b, self.fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), result):
                hits.append({'first': name_a, 'second': name_b,
                             'penetration_m': max((float(c.penetration_depth) for c in result.contacts), default=0.)})
        for sample, qs in enumerate(densify_joint_path(starts, frames, max_joint_increment)):
            samples += 1; bodies = []; hits = []
            finger_state = dict(fingers)
            if finger_targets:
                u = min(1., sample/max(1, len(frames)))
                for arm, target in finger_targets.items():
                    finger_state[arm] = np.asarray(fingers[arm]) + u*(np.asarray(target)-fingers[arm])
            for arm, q in qs.items():
                tf = self.fk(arm, q, finger_state[arm])
                for (link, idx), mesh in self.meshes.items():
                    t = tf[link]; body = robots[(arm, link, idx)]
                    body.setTransform(self.fcl.Transform(t[:3, :3], t[:3, 3]))
                    # Conservative transformed local AABB, followed by exact mesh narrow phase.
                    vertices = self.box_corners[(link,idx)]@t[:3, :3].T+t[:3, 3]
                    bounds = np.array([vertices.min(0), vertices.max(0)])
                    bodies.append((f'arm{arm}/{link}', body, bounds, arm, link))
            for a, b in itertools.combinations(bodies, 2):
                if a[3] == b[3] and (a[4] == b[4] or frozenset((a[4], b[4])) in self.neighbors): continue
                overlap(*a[:3], *b[:3], hits)
            for name, body, bounds, _, _ in bodies:
                for oid, (_, obstacle, box) in obstacles.items(): overlap(name, body, bounds, oid, obstacle, box, hits)
            payloads = []
            for item in attachments:
                oid, arm = attachment_name(item), int(item['arm'])
                model=self.models[arm]
                tool=model.T_link_tcp.copy()
                if 'tcp_offset_m' in item:tool[:3,3]=np.asarray(item['tcp_offset_m'],float)
                t_tcp = self.fk(arm, qs[arm], finger_state[arm])[model.tcp_link]@tool
                t = t_tcp@np.asarray(item['T_tcp_part'] if item.get('part_id') is not None else item['T_tcp_object'], float)
                mesh = attached_meshes[oid]; body = payload_bodies[oid]
                body.setTransform(self.fcl.Transform(t[:3,:3],t[:3,3]))
                v = mesh.vertices@t[:3, :3].T+t[:3, 3]; box = np.array([v.min(0), v.max(0)])
                payloads.append((oid, body, box))
                for other, (_, obstacle, obox) in obstacles.items(): overlap(oid, body, box, other, obstacle, obox, hits)
                for name, rb, rbox, _, _ in bodies: overlap(oid, body, box, name, rb, rbox, hits)
            for a, b in itertools.combinations(payloads, 2): overlap(*a, *b, hits)
            if hits:
                return {'status': 'COLLISION', 'sample': sample, 'samples_checked': samples,
                        'hits': hits[:max_hits], 'state_revision': state['revision'], 'narrow_phase_checks': checks,
                        'coverage': 'sampled_joint_fk', 'max_joint_increment_rad': max_joint_increment}
        return {'status': 'CLEAR', 'samples_checked': samples, 'state_revision': state['revision'],
                'narrow_phase_checks': checks, 'coverage': 'sampled_joint_fk',
                'max_joint_increment_rad': max_joint_increment}


def robodojo_collision(urdf, bases, world):
    """All RoboDojo tasks share fixed X5 base mounts on the table.

    Exclude only each fixed base/table pair. Moving links, fingers, payloads,
    other fixtures and robot self/inter-arm contacts still use normal checks.
    The representation without a scene has no table contacts to exempt.
    """
    mounts = [(f"arm{arm}/base_link", "tabletop") for arm in bases] if "tabletop" in world.objects() else []
    return RobotCollision(urdf, bases, world, mounting_contacts=mounts)
