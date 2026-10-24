"""Optional agent-authored part geometry and state; no perception or repair policy."""
import os
import copy,hashlib,uuid,time
from pathlib import Path
import fcntl
import numpy as np
import trimesh

class AssemblyMixin:
    def _edit_object(self, object_id, edit, expected_revision, basis, note, evidence):
        from .world import atomic
        from .flow import strict
        if strict(self.root) and basis=='observation_estimate':raise ValueError('Publish part hypotheses then validate candidate assembly with integrate_rgbd')
        with (self.root/'state/update.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.state()
            if expected_revision is not None and state['revision']!=expected_revision:
                raise ValueError('Stale state revision; reread and recompute')
            row=state['objects'][object_id]
            detail=edit(row)
            if strict(self.root):
                row.pop('sensor_validation',None)
                for field in row.get('observed_state',{}).get('fields',{}).values():field['status']='pending'
            state['revision']+=1
            state['history'].append({'revision':state['revision'],'kind':basis,'object_id':object_id,
                'edit':detail,'note':note,'evidence':evidence or [],'actor':{'role':os.environ.get('CODEWORLD_ROLE','supervisor_manual'),'run_id':os.environ.get('CODEWORLD_EXECUTOR_ID')},'recorded_at_ns':time.time_ns()})
            atomic(self.state_path,state)
        return state['revision']

    def publish_parts(self, object_id, parts, *, parameters=None, basis='agent_estimate',
                      note=None, evidence=None, expected_revision=None):
        """Replace this object's geometry with an explicit flat assembly in object frame.

        parts maps ID to {mesh: path, T_object_part: 4x4, ...free metadata}.
        Every mesh is in that part's local frame. This replaces the whole object's
        geometry, so include the fixed body too. Assets are copied once. Subsequent
        motion uses update_parts, without exports, fitting or reconstruction.
        Use fork() for hypothetical edits. Original reconstruction stays immutable.
        """
        from .world import rigid
        if not parts:raise ValueError('Assembly must have at least one part')
        prepared={};version=uuid.uuid4().hex
        for i,(pid,spec) in enumerate(parts.items()):
            if not isinstance(pid,str) or not pid:raise ValueError('Part IDs must be nonempty strings')
            spec=copy.deepcopy(spec);source=Path(spec.pop('mesh')).resolve()
            transform=rigid(spec.pop('T_object_part',np.eye(4))).tolist()
            # Validate only the provided changed geometry, no observation-based gate.
            mesh=trimesh.load(source,force='scene',process=False).to_geometry()
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise ValueError('Invalid part geometry')
            data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
            prepared[pid]=(spec,transform,data,digest,str(Path('geometry_edits')/version/(str(i)+source.suffix)))
        def edit(row):
            entries={}
            for pid,(meta,t,data,digest,rel) in prepared.items():
                dst=self.root/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
                entries[pid]={**meta,'local_mesh':rel,'sha256':digest,'T_object_part':t}
            row['assembly']={'geometry_revision':version,'parts':entries,'parameters':copy.deepcopy(parameters or {}),
                             'basis':basis,'evidence':evidence or []}
            return {'operation':'publish_parts','geometry_revision':version,'parts':list(entries)}
        return self._edit_object(object_id,edit,expected_revision,basis,note,evidence)

    def update_parts(self, object_id, transforms=None, *, parameters=None, basis='agent_estimate',
                     note=None, evidence=None, expected_revision=None):
        """Update arbitrary part transforms/parameters; Codex owns kinematics and estimation.

        Transforms are T_object_part, not world transforms. Parameters are descriptive
        state (e.g. joint values); they do not secretly drive transforms. Supply both
        when needed. No model call, mesh write, or perception check is performed.
        """
        from .world import rigid
        transforms={k:rigid(v).tolist() for k,v in (transforms or {}).items()}
        if not transforms and parameters is None:raise ValueError('Empty part update')
        def edit(row):
            assembly=row['assembly']
            for pid,t in transforms.items():assembly['parts'][pid]['T_object_part']=t
            if parameters is not None:assembly['parameters'].update(copy.deepcopy(parameters))
            assembly.update(basis=basis,evidence=evidence or [])
            return {'operation':'update_parts','transforms':transforms,'parameters':parameters}
        return self._edit_object(object_id,edit,expected_revision,basis,note,evidence)

    def part_mesh(self, object_id, part_id):
        from .world import rigid
        row=self.state()['objects'][object_id];part=row['assembly']['parts'][part_id]
        mesh=trimesh.load(self.root/part['local_mesh'],force='scene',process=False).to_geometry()
        mesh.apply_transform(rigid(row['T_W_object'])@rigid(part['T_object_part']))
        return mesh
