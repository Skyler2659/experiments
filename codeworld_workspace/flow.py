"""One observed world -> executable computation -> recorded decision boundary.

This is a protocol check, not a sandbox or proof of an LM's private reasoning.
"""
import copy,json,uuid,hashlib,ast,math,os
from pathlib import Path
from .completion import digest,path_in,state_ref,oracle_covers_feedback

def validate_program(source):
    """Keep direct external IO out of decisions while allowing normal Python helpers.

    This is a small protocol preflight, not a security sandbox or a proof of data
    dependence. Pass ctx to helpers and use loops/comprehensions normally.
    """
    tree=ast.parse(source)
    forbidden={'open','eval','exec','compile','__import__','input','globals','locals','vars','getattr','setattr','delattr','breakpoint'}
    io={'load','loadtxt','genfromtxt','save','savez','savetxt','fromfile','tofile','memmap','open_memmap','fromregex','DataSource','read','write','read_text','write_text','export','dump','to_csv','to_pickle'}
    context={'bounds','mesh','transform','object','uncertainty','proprioception','measurement','position','metadata','state'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Global,ast.Nonlocal)):
            raise ValueError('External mutable state is not a decision input')
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
            if any(n not in {'math','numpy','scipy.spatial.transform'} for n in names):
                raise ValueError('Only numerical imports are supported in world computations')
        if isinstance(node,ast.Name) and (node.id.startswith('__') or node.id in forbidden):
            raise ValueError('Direct external IO/execution is unavailable in a world computation')
        if isinstance(node,ast.Attribute):
            if node.attr.startswith('_') or node.attr in io:
                raise ValueError('No image/file IO in decision computation; integrate observations first')
            if isinstance(node.value,ast.Name) and node.value.id=='ctx' and node.attr not in context:
                raise ValueError('Use recorded CodeWorld queries')


def query_signature(world, state, queries):
    """Only inputs actually queried invalidate a computation; logs are not inputs."""
    from .world import read
    inputs=[]
    for q in queries:
        op=q['op'];oid=q.get('object');row=state['objects'].get(oid,{})
        if op in ('position','bounds','mesh','transform'):
            value={k:row.get(k) for k in ('T_W_object','assembly','motion_epoch')}
        elif op=='object':
            value={k:row.get(k) for k in ('T_W_object','basis','review','sensor_validation','motion_epoch')}
        elif op=='metadata':
            value=read(world.root/world.manifest['objects'][oid])
        elif op=='uncertainty':
            value={'review':row.get('review',{}),'sensor_validation':row.get('sensor_validation')}
        elif op=='state':
            field=q['field'];observed=row.get('observed_state',{})
            from .completion import field_supported
            value={'value':observed.get('values',{}).get(field),'evidence':observed.get('fields',{}).get(field), 'supported':field_supported(state,oid,field,state.get('runtime',{}).get('robot_state',{}))}
        elif op=='measurement':value=state.get('measurements',{}).get(q['id'])
        elif op=='proprioception':value=state.get('runtime',{}).get('robot_state',{})
        else:raise ValueError('Unknown CodeWorld query: '+op)
        inputs.append({'query':q,'value':value})
    return digest(inputs)


def registry_path(world, ident):
    if not isinstance(ident,str) or not ident or any(c not in '0123456789abcdef' for c in ident):
        raise ValueError('Invalid computation ID')
    return world.root/'decisions/registered'/f'{ident}.json'


def strict(root):
    p=Path(root)/'task/world_flow.json'
    return p.exists() and json.loads(p.read_text()).get('required') is True


class DecisionContext:
    def __init__(self,world,state):
        from .planning import PlanningContext
        self._world=world;self._snapshot=state;self._planning=PlanningContext(world);self._planning.snapshot=state
        self._queries=[];self._object_ids=set()
    def _object(self,oid,allow_unknown=False,orientation=True):
        # Planning reads estimates, including pending/predicted poses. Their
        # evidence remains in the same snapshot and is available via uncertainty.
        row=self._snapshot['objects'][oid];self._object_ids.add(oid)
        return row
    def metadata(self,oid):
        from .world import read
        self._object(oid,allow_unknown=True);self._queries.append({'op':'metadata','object':oid})
        row=read(self._world.root/self._world.manifest['objects'][oid])
        return {**{k:copy.deepcopy(row[k]) for k in ('object_id','name','label','role','category') if k in row},'unknowns':copy.deepcopy(row.get('declared',{}).get('unknowns',[]))}
    def position(self,oid):
        import numpy as np
        row=self._object(oid,orientation=False);self._queries.append({'op':'position','object':oid});return self._world._mesh(oid,self._snapshot).centroid.copy()
    def bounds(self,oid,part_id=None):
        self._object(oid);self._queries.append({'op':'bounds','object':oid,'part_id':part_id})
        return self._world._part_mesh(oid,part_id,self._snapshot).bounds.copy() if part_id is not None else self._planning.bounds(oid)
    def mesh(self,oid,part_id=None):
        self._object(oid);self._queries.append({'op':'mesh','object':oid,'part_id':part_id})
        return self._world._part_mesh(oid,part_id,self._snapshot) if part_id is not None else self._planning.mesh(oid)
    def transform(self,oid,part_id=None):
        import numpy as np
        row=self._object(oid);self._queries.append({'op':'transform','object':oid,'part_id':part_id})
        t=np.array(row['T_W_object'])
        return t@np.array(row['assembly']['parts'][part_id]['T_object_part']) if part_id is not None else t
    def object(self,oid):
        row=self._object(oid);self._queries.append({'op':'object','object':oid})
        # Free-form assembly parameters and old derived fields are not sensor measurements.
        return {k:copy.deepcopy(row[k]) for k in ('T_W_object','basis','review','sensor_validation') if k in row}
    def uncertainty(self,oid):
        row=self._object(oid,True);self._queries.append({'op':'uncertainty','object':oid});return {**copy.deepcopy(row.get('review',{})), 'basis':row.get('basis'), 'sensor_validation':copy.deepcopy(row.get('sensor_validation'))}
    def proprioception(self):
        self._queries.append({'op':'proprioception'});self._object_ids.update(i for i in self._snapshot['objects'] if i=='known_robot');return copy.deepcopy(self._snapshot.get('runtime',{}).get('robot_state',{}))
    def measurement(self,ident):
        d=self._snapshot['measurements'][ident]
        self._object_ids.update(d.get('object_ids',[d['object_id']]))
        self._queries.append({'op':'measurement','id':ident,'object':d['object_id']})
        # Historical controller outcomes remain useful for recovery. The payload
        # explicitly carries its original reference; completion checks freshness.
        return {**copy.deepcopy(d['value']), 'measurement_robot_state_ref':copy.deepcopy(d.get('robot_state_ref'))}
    def state(self,oid,field):
        from .completion import field_supported
        row=self._object(oid,True)
        if not field_supported(self._snapshot,oid,field,self._snapshot.get('runtime',{}).get('robot_state',{})):
            raise ValueError('State field is pending or unsupported: '+oid+'.'+field)
        self._queries.append({'op':'state','object':oid,'field':field})
        return copy.deepcopy(row['observed_state']['values'][field])



def run_computation(world,source,snapshot,object_ids,purpose):
    validate_program(source)
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .provenance import analyze_program
    ctx=DecisionContext(world,snapshot)
    import builtins
    safe={k:v for k,v in vars(builtins).items() if not k.startswith('_') and k not in {'open','eval','exec','compile','input','globals','locals','vars','getattr','setattr','delattr','breakpoint'}}
    safe['__import__']=builtins.__import__
    scope={'__builtins__':safe,'np':np,'math':math,'Rotation':Rotation}
    exec(compile(source,'<world-computation>','exec'),scope);result=scope['compute'](ctx)
    result=json.loads(json.dumps(result,default=lambda a:a.tolist() if hasattr(a,'tolist') else (_ for _ in ()).throw(TypeError('Unserializable result')),allow_nan=False))
    if not ctx._queries or not set(object_ids)<=ctx._object_ids:raise ValueError('Decision must query every declared relevant object in CodeWorld')
    if not isinstance(result,dict) or not (result.get('actions') or result.get('judgments')):raise ValueError('Return explicit actions or judgments')
    analysis=analyze_program(source,result);dependencies=analysis['judgment_dependencies']
    ctx.provenance_warnings=analysis['warnings']
    ctx.constant_judgments=analysis.get('constant_judgments',[])
    physical_objects={q['object'] for q in ctx._queries if q['op'] in ('position','bounds','mesh','transform','object','state')}
    for q in ctx._queries:
        if q['op']=='measurement':physical_objects.update(snapshot['measurements'][q['id']].get('object_ids',[q['object']]))
    # Resolve the controller's ambient arm selection into the recorded payload.
    # select_arm does not advance physical counters; leaving this default implicit
    # would let the same recorded action later operate a different arm.
    default_arm=snapshot.get('runtime',{}).get('robot_state',{}).get('active_arm',0)
    for action in result.get('actions',[]):
        if not isinstance(action.get('args'),dict) or not action.get('op'):raise ValueError('Malformed action')
        affected=action.get('affected_objects',[])
        if not isinstance(affected,list) or any(i not in snapshot['objects'] for i in affected):raise ValueError('Unknown affected object')
        if action['op'] in ('move_tcp','move_tcp_group','joints','joints_group','gripper','gripper_group','step'):
            if purpose!='sensing' and (not physical_objects or (affected and not physical_objects & set(affected))):
                raise ValueError('Manipulation must use object state relevant to affected objects')
        if action['op']=='select_arm' and 'arm' in action['args']:
            default_arm=action['args']['arm']
        elif action['op'] in ('move_tcp','joints','gripper','check_ik'):
            action['args'].setdefault('arm',default_arm)
        elif action['op'] in ('move_tcp_group','joints_group','gripper_group','check_motion'):
            for command in action['args'].get('commands',[]):
                if isinstance(command,dict):command.setdefault('arm',default_arm)
    return result,ctx,dependencies


def verify_decision(world,receipt,actor=None):
    """Bind dispatch to the registered computation without executing it twice.

    Registration prevents accidental artifact edits; it is not a security boundary
    against an executor with arbitrary filesystem access.
    """
    from .world import read
    if receipt.get('schema')!='codeworld-computation-v1':raise ValueError('Unknown decision receipt')
    if actor is not None and receipt.get('actor')!=actor:raise ValueError('Decision belongs to another executor')
    registered=registry_path(world,receipt.get('id'))
    if not registered.exists() or read(registered)!=receipt:raise ValueError('Decision differs from its registered computation')
    snapshot=world.state();world.verify_binding()
    if receipt.get('geometry_revision')!=world.manifest['geometry_revision']:raise ValueError('Changed world geometry')
    if receipt.get('dependency_signature')!=query_signature(world,snapshot,receipt['queries']):
        raise ValueError('Stale world decision: a queried input changed')
    live_ref=state_ref(world._robot_state())
    if not live_ref or receipt.get('robot_state_ref')!=live_ref:raise ValueError('Stale or missing public robot state')
    return receipt


class FlowMixin:
    def compute_decision(self, program, *, purpose, object_ids):
        """Run agent-authored compute(ctx) on one world snapshot, retaining code and I/O.

        Return {actions:[{op,args,affected_objects}], judgments:{...}}. Public task,
        tool dimensions and tolerances are legitimate declared constants. Coordinates
        and physical claims must be calculated from ctx; no image reads in this stage.
        """
        from .world import atomic,read
        if self.branch:raise ValueError('Publish observed world before deciding')
        self.sync_robot_feedback()
        if not purpose or not isinstance(object_ids,list):raise ValueError('Declare purpose and relevant object IDs')
        source=path_in(self.root,program).read_text();snapshot=self.state()
        self.verify_binding()
        result,ctx,dependencies=run_computation(self,source,snapshot,object_ids,purpose)
        if query_signature(self,self.state(),ctx._queries)!=query_signature(self,snapshot,ctx._queries):raise ValueError('Queried world inputs changed during computation')
        if strict(self.root):
            from .completion import evidence_current
            live=self._robot_state()
            if state_ref(live)!=state_ref(snapshot.get('runtime',{}).get('robot_state',{})):raise ValueError('Robot changed during computation')
        ident=uuid.uuid4().hex;receipt={'schema':'codeworld-computation-v1','id':ident,'purpose':purpose,'object_ids':object_ids,'queried_objects':sorted(ctx._object_ids),'queries':ctx._queries,'program':str(program),'source':source,'world_digest':digest(snapshot),'state_revision':snapshot['revision'],'geometry_revision':self.manifest['geometry_revision'],'actor':{'role':os.environ.get('CODEWORLD_ROLE','supervisor_manual'),'run_id':os.environ.get('CODEWORLD_EXECUTOR_ID')},'robot_state_ref':state_ref(snapshot.get('runtime',{}).get('robot_state',{})),'result':result,'judgment_dependencies':dependencies,'provenance_warnings':ctx.provenance_warnings,'constant_judgments':ctx.constant_judgments,'dependency_signature':query_signature(self,snapshot,ctx._queries)}
        (self.root/'decisions').mkdir(exist_ok=True)
        registry_path(self,ident).parent.mkdir(parents=True,exist_ok=True)
        atomic(registry_path(self,ident),receipt)
        atomic(self.root/'decisions'/f'{ident}.json',receipt)
        return {'decision':f'decisions/{ident}.json',**receipt}

    def _robot_state(self):
        from .world import read
        p=self.root/'task/robot_connection.json'
        if not p.exists():return {}
        q=Path(read(p)['io_directory'])/'robot_state.json'
        return read(q) if q.exists() else {}

    def sync_robot_feedback(self):
        """Ingest public receipts; malformed rejected requests cannot poison the world."""
        from .world import read,atomic
        import fcntl
        if not strict(self.root):return
        connection=self.root/'task/robot_connection.json'
        if not connection.exists():return
        io=Path(read(connection)['io_directory'])
        with (self.root/'state/update.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);s=self.state();runtime=s.setdefault('runtime',{});seen=set(runtime.get('processed_feedback',[]));changed=False
            ledger=self.root/'task/world_decision_uses.json';uses=read(ledger) if ledger.exists() else {}
            for f in sorted((io/'responses').glob('*.json'),key=lambda f:f.stat().st_mtime_ns):
                if f.name in seen:continue
                req=io/'requests'/f.name
                if not req.exists():continue
                packet=read(req);reply=read(f);ref=packet.get('args',{}).get('world_decision')
                decision=None;action=None
                if ref:
                    try:
                        matches=[(key,entry) for key,entry in uses.items() if isinstance(entry,dict) and entry.get('request_id')==packet['id']]
                        if len(matches)!=1:raise ValueError('Request was not authorized for dispatch')
                        key,entry=matches[0];decision=entry['receipt'];action=decision['result']['actions'][int(key.rsplit(':',1)[1])]
                    except (OSError,ValueError,KeyError,TypeError,IndexError):
                        # A missing/bad receipt rejected before dispatch is a transport diagnostic,
                        # never a physical measurement or a reason to fail future synchronization.
                        runtime.setdefault('rejected_feedback',[]).append(f.name);decision=None;action=None
                if action is not None:
                    if action['op'] in ('move_tcp','move_tcp_group','joints','joints_group','gripper','gripper_group','step'):
                        # Even partial/error actions may have changed objects. A rejected IK/path
                        # is exempt only if the public robot version did not advance.
                        same_robot=state_ref(reply.get('robot_state',{}))==decision['robot_state_ref']
                        no_motion=same_robot and (reply.get('status') in ('COLLISION','COLLISION_BLOCKED','IK_NOT_FOUND') or reply.get('steps')==0)
                        if not no_motion:
                            affected=set(action.get('affected_objects',[]))
                            for oid in affected:
                                row=s['objects'][oid]
                                if oracle_covers_feedback(s,oid,reply.get('robot_state',{})):continue
                                row.pop('sensor_validation',None);row.pop('oracle_validation',None);row['motion_epoch']=row.get('motion_epoch',0)+1
                                for r in row.get('observed_state',{}).get('fields',{}).values():r['status']='pending'
                                row.setdefault('review',{})['pose']={'status':'pending','reason':'Action may have changed this object; observe before relying on it','request':f.name}
                    objects=decision['queried_objects']
                    s.setdefault('measurements',{})[packet['id']]={'object_id':objects[0] if objects else 'known_robot','object_ids':objects,'kind':'robot_receipt','operation':action['op'],'value':reply,'robot_state_ref':state_ref(reply.get('robot_state',self._robot_state())),'decision':ref}
                seen.add(f.name);changed=True
            live=self._robot_state()
            if live!=runtime.get('robot_state'):runtime['robot_state']=live;changed=True
            runtime['processed_feedback']=sorted(seen)
            if changed:
                s['revision']+=1;s.setdefault('history',[]).append({'revision':s['revision'],'kind':'public_robot_feedback'});atomic(self.state_path,s)

    def derive_state(self, decision, object_id, fields):
        """Write state only from a recorded CodeWorld computation, not visual booleans."""
        from .world import read
        r=read(path_in(self.root,decision));self.sync_robot_feedback();s=self.state()
        verify_decision(self,r,actor={'role':os.environ.get('CODEWORLD_ROLE','supervisor_manual'),'run_id':os.environ.get('CODEWORLD_EXECUTOR_ID')})
        if object_id not in r['queried_objects']:raise ValueError('Unrelated state derivation')
        from .provenance import PHYSICAL
        relevant=[q for q in r['queries'] if q['op'] in PHYSICAL and (
            q.get('object')==object_id or
            (q['op']=='measurement' and object_id in s.get('measurements',{}).get(q['id'],{}).get('object_ids',[])) or
            (object_id=='known_robot' and q['op']=='proprioception'))]
        if not relevant:raise ValueError('Physical state requires recorded measurements of its target object')
        for field,key in fields.items():
            if key in r.get('constant_judgments',[]):
                raise ValueError('A declared constant is not a physical state measurement')
        values={}
        for field,key in fields.items():values[field]=r['result']['judgments'][key]
        self._flow_derivation={'decision':decision,'fields':fields,'field_queries':{field:[q for q in r['queries'] if not r['judgment_dependencies'].get(key) or q['op'] in r['judgment_dependencies'][key]] for field,key in fields.items()},'queries':copy.deepcopy(r['queries']),'judgment_dependencies':copy.deepcopy(r['judgment_dependencies']),'robot_state_ref':state_ref(self._robot_state()),'dependency_signature':query_signature(self,s,r['queries']),'state_dependencies':[{'object':q['object'],'field':q['field'],'digest':digest({'value':s['objects'][q['object']].get('observed_state',{}).get('values',{}).get(q['field']),'evidence':s['objects'][q['object']].get('observed_state',{}).get('fields',{}).get(q['field'])})} for q in r['queries'] if q['op']=='state'],'dependencies':{oid:{'pose':digest(s['objects'][oid]['T_W_object']),'validation':digest(s['objects'][oid].get('sensor_validation')),'review':digest(s['objects'][oid].get('review',{}).get('pose'))} for oid in r['queried_objects']}}
        try:return self.update_state({object_id:values},evidence=[decision],expected_revision=s['revision'])
        finally:del self._flow_derivation

    def ingest_grasp_feedback(self, **evidence):
        from .observation_evidence import assess_grasp, load_capture
        from .world import atomic
        import fcntl
        self.sync_robot_feedback()
        after_path,after=load_capture(self.root,evidence['after_capture'])
        measured_ref=state_ref(after.get('robot_state',{}))
        if not measured_ref or measured_ref!=state_ref(self._robot_state()):
            raise ValueError('Stale or unversioned grasp after-capture; acquire current RGB-D')
        value=assess_grasp(self.root,**evidence)
        # Preserve numerical observations in world; downstream interpretation uses ctx.measurement.
        ident=uuid.uuid4().hex
        with (self.root/'state/update.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if measured_ref!=state_ref(self._robot_state()):
                raise ValueError('Robot changed during grasp assessment; acquire current RGB-D')
            s=self.state()
            if evidence['object_id'] not in s['objects']:raise ValueError('Unknown grasp object')
            s.setdefault('measurements',{})[ident]={'object_id':evidence['object_id'],'kind':'rgbd_grasp',
                'value':value,'robot_state_ref':measured_ref,'after_capture':str(after_path),'evidence':copy.deepcopy(evidence)}
            s['revision']+=1;atomic(self.state_path,s)
        return {'status':'MEASUREMENT_IN_WORLD','measurement_id':ident,'state_revision':s['revision']}
