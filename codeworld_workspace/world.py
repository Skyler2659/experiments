"""Read geometry, calculate in W, and maintain observed or hypothetical rigid state."""
import copy
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
import fcntl
import numpy as np
import trimesh
from .flow import FlowMixin,strict
from .completion import CompletionMixin
from .assembly import AssemblyMixin
from .review import ReviewMixin, record_pose_observation


# Immutable geometry is checked once per file version in each long-lived process.
_BOUND_FILE_DIGESTS={}

def bound_file_digest(path):
    path=Path(path);stat=path.stat()
    key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if key not in _BOUND_FILE_DIGESTS:
        # Retain one version per path; geometry revisions are relatively infrequent.
        for old in [k for k in _BOUND_FILE_DIGESTS if k[0]==key[0]]:del _BOUND_FILE_DIGESTS[old]
        _BOUND_FILE_DIGESTS[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    return _BOUND_FILE_DIGESTS[key]


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.state-')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6)):
        raise ValueError('Expected rigid row-major T_W_object (meters)')
    return t


class World(AssemblyMixin, ReviewMixin, CompletionMixin, FlowMixin):
    def __init__(self, root, branch=None):
        self.root = Path(root).resolve()
        self.manifest = read(self.root/'world.json')
        self.branch = branch
        if branch and not re.fullmatch(r'[A-Za-z0-9_-]+', branch):
            raise ValueError('Invalid branch name')
        self.state_path = self.root/('state/current.json' if not branch else f'scenarios/{branch}.json')

    def integrate_rgbd(self, capture, proposals, **options):
        from .sensor_update import integrate_rgbd
        return integrate_rgbd(self,capture,proposals,**options)

    def check_grasp_geometry(self, object_id, **candidate):
        """Read-only candidate jaw-width check; never asserts a physical grasp."""
        from .flow import DecisionContext
        if strict(self.root):
            self.sync_robot_feedback();DecisionContext(self,self.state())._object(object_id)
        from .grasp import check_grasp_geometry
        return check_grasp_geometry(self, object_id, **candidate)

    def planning_context(self):
        from .planning import PlanningContext
        return PlanningContext(self)

    def verify_binding(self):
        for rel, expected in self.manifest['bound_files'].items():
            if bound_file_digest(self.root/rel) != expected:
                raise ValueError('Published geometry/source changed; rebuild workspace: '+rel)
        for row in self.state()['objects'].values():
            for part in row.get('assembly',{}).get('parts',{}).values():
                if bound_file_digest(self.root/part['local_mesh'])!=part['sha256']:
                    raise ValueError('Published part geometry changed: '+part['local_mesh'])
        return {'status': 'BOUND', 'geometry_revision': self.manifest['geometry_revision']}

    def state(self):
        s = read(self.state_path)
        return s

    def objects(self):
        return list(self.manifest['objects'])

    def object(self, object_id):
        s = self.state()
        row = read(self.root/self.manifest['objects'][object_id])
        row['current'] = copy.deepcopy(s['objects'][object_id])
        row['state_revision'] = s['revision']
        return row

    def parts(self, object_id):
        """Source hierarchy with node transforms updated by the current rigid object pose."""
        s=self.state(); row=read(self.root/self.manifest['objects'][object_id])
        live=s['objects'][object_id]
        if 'assembly' in live:
            nodes=[]
            for pid,part in live['assembly']['parts'].items():
                node=copy.deepcopy(part);node['part_id']=pid
                node['current_matrix_W_node']=(rigid(live['T_W_object'])@rigid(part['T_object_part'])).tolist()
                nodes.append(node)
            return {'state_revision':s['revision'],'geometry_revision':live['assembly']['geometry_revision'],
                    'parameters':copy.deepcopy(live['assembly']['parameters']),'nodes':nodes}
        delta=rigid(s['objects'][object_id]['T_W_object']) @ np.linalg.inv(rigid(row['initial_T_W_object']))
        nodes=[]
        for node in row['source_nodes']:
            current=copy.deepcopy(node)
            initial=np.array(node['matrix_world_column_major']).reshape(4,4,order='F')
            current['current_matrix_W_node']=(delta @ initial).tolist()
            nodes.append(current)
        return {'state_revision':s['revision'],'nodes':nodes}

    def _part_mesh(self, object_id, part_id, state):
        live=state['objects'][object_id]
        part=live['assembly']['parts'][part_id]
        mesh=trimesh.load(self.root/part['local_mesh'],force='scene',process=False).to_geometry()
        mesh.apply_transform(rigid(live['T_W_object'])@rigid(part['T_object_part']))
        return mesh

    def _mesh(self, object_id, state):
        live=state['objects'][object_id]
        if 'assembly' in live:
            meshes=[]
            for part in live['assembly']['parts'].values():
                m=trimesh.load(self.root/part['local_mesh'],force='scene',process=False).to_geometry()
                m.apply_transform(rigid(live['T_W_object'])@rigid(part['T_object_part']));meshes.append(m)
            return trimesh.util.concatenate(meshes)
        row = read(self.root/self.manifest['objects'][object_id])
        mesh = trimesh.load(self.root/row['geometry']['local_mesh'], force='mesh', process=False)
        mesh.apply_transform(rigid(state['objects'][object_id]['T_W_object']))
        return mesh

    def mesh(self, object_id):
        """Complete generated triangle surface in current W. Not real-world truth."""
        return self._mesh(object_id, self.state())

    def bounds(self, object_id):
        s = self.state()
        return {'object_id': object_id, 'state_revision': s['revision'],
                'bounds_W_m': self._mesh(object_id, s).bounds.tolist(),
                'review': copy.deepcopy(s['objects'][object_id].get('review', {}))}

    def points(self, object_id, points_local):
        s = self.state(); points = np.asarray(points_local, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite Nx3 points')
        t = rigid(s['objects'][object_id]['T_W_object'])
        return {'state_revision': s['revision'], 'points_W_m': (points @ t[:3,:3].T+t[:3,3]).tolist()}

    def aabb_relation(self, first, second):
        """Conservative boxes only: zero separation does NOT prove mesh collision."""
        s = self.state(); a = self._mesh(first,s).bounds; b = self._mesh(second,s).bounds
        gap = np.maximum(np.maximum(a[0]-b[1], b[0]-a[1]), 0)
        return {'state_revision': s['revision'], 'aabb_separation_lower_bound_m': float(np.linalg.norm(gap)),
                'boxes_overlap': bool(np.all(gap == 0)), 'mesh_collision_checked': False}

    def nearest_surface(self, object_id, points_W):
        s = self.state(); p = np.asarray(points_W, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError('Expected finite Nx3 points')
        # Chunked exact point-to-triangle query; no rtree dependency, no signed-volume assumptions.
        mesh = self._mesh(object_id, s); results=[]
        for point in p:
            q,d,f = trimesh.proximity.closest_point_naive(mesh, point.reshape(1,3))
            results.append({'point_W_m':q[0].tolist(),'distance_m':float(d[0]),'face_id':int(f[0])})
        return {'state_revision':s['revision'],'results':results,'kind':'unsigned point-to-surface'}

    def fork(self, name):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name): raise ValueError('Invalid branch name')
        p=self.root/f'scenarios/{name}.json'
        s=self.state(); s['branch']=name; s['base_revision']=s['revision']
        with p.open('x') as f: json.dump(s,f,indent=2)
        return World(self.root,name)

    def _update(self, updates, expected_revision, evidence, kind):
        if strict(self.root) and kind=='observation_estimate':raise ValueError('Observed poses require integrate_rgbd surface validation; evidence paths alone are insufficient')
        with (self.root/'state/update.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            s=self.state()
            if expected_revision is not None and s['revision'] != expected_revision: raise ValueError('Stale state revision; reread and recompute')
            if not updates: raise ValueError('Empty update')
            normalized={}
            for oid,t in updates.items():
                row=read(self.root/self.manifest['objects'][oid])
                normalized[oid]=rigid(t).tolist()
                s['objects'][oid].update(T_W_object=normalized[oid],basis=kind,evidence=evidence)
                s['objects'][oid]['motion_epoch']=s['objects'][oid].get('motion_epoch',0)+1
                if strict(self.root):
                    s['objects'][oid].pop('sensor_validation',None)
                    for field in s['objects'][oid].get('observed_state',{}).get('fields',{}).values():field['status']='pending'
                record_pose_observation(s['objects'][oid],s['revision']+1,kind,evidence)
            s['revision']+=1
            s['history'].append({'revision':s['revision'],'kind':kind,'updates':normalized,'evidence':evidence,'actor':{'role':os.environ.get('CODEWORLD_ROLE','supervisor_manual'),'run_id':os.environ.get('CODEWORLD_EXECUTOR_ID')},'recorded_at_ns':time.time_ns()})
            atomic(self.state_path,s)
        return s['revision']

    def set_hypothesis(self, updates, expected_revision):
        if not self.branch: raise ValueError('Hypothetical poses require fork(name)')
        return self._update(updates,expected_revision,None,'hypothesis')

    def update(self, updates, *, basis='agent_estimate', note=None, evidence=None, expected_revision=None):
        """Codex maintains current poses; distinguish estimates from action predictions."""
        return self._update(updates,expected_revision,{'note':note,'references':evidence or []},basis)

    def update_observed(self, observation_file, expected_revision=None):
        """Optional observation import. No estimator-specific report or fit gate."""
        path=Path(observation_file).resolve();obs=read(path)
        return self.update(obs['poses_W'],basis='observation_estimate',evidence=[str(path)],expected_revision=expected_revision)

    def export_scene(self, output):
        """Export current geometry with current poses for inspection/downstream planning."""
        s=self.state(); scene=trimesh.Scene()
        for oid in self.objects(): scene.add_geometry(self._mesh(oid,s),node_name=oid,geom_name=oid)
        scene.export(str(output))
        return {'state_revision':s['revision'],'output':str(output)}
