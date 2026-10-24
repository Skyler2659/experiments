"""Executable, observation-backed completion checks over the published CodeWorld.

These are agent-authored interpretations of public goals, not simulator predicates.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def path_in(root, value):
    p=Path(value)
    p=(root/p).resolve() if not p.is_absolute() else p.resolve()
    p.relative_to(root)
    return p


def state_ref(state):
    # Public backend counters, never task-object GT.
    return {k:state[k] for k in ('robot_state_ref','timestamp_ns','control_steps','official_actions') if k in state}


def oracle_covers_feedback(state, oid, robot_state):
    """A later oracle sample in the same episode also covers older receipts."""
    if not measured_pose_supported(state, oid) or state['objects'][oid].get('basis') != 'oracle_state':
        return False
    current = state['objects'][oid]['oracle_validation']['robot_state_ref']
    previous = state_ref(robot_state)
    if current == previous:
        return True
    a, b = current.get('robot_state_ref', ''), previous.get('robot_state_ref', '')
    return (':' in a and ':' in b and a.rsplit(':', 1)[0] == b.rsplit(':', 1)[0]
            and type(current.get('control_steps')) is int and type(previous.get('control_steps')) is int
            and current['control_steps'] >= previous['control_steps'])


def evidence_current(world, refs, robot_state):
    if not isinstance(refs,list) or not refs:return False
    wanted=state_ref(robot_state)
    if not wanted:return False
    for ref in refs:
        try:
            p=path_in(world.root,ref)
            p.relative_to(world.root/'robot_io/captures')
            d=json.loads(p.read_text())
            if d.get('views') and state_ref(d.get('robot_state',{}))==wanted:return True
        except (OSError,ValueError,TypeError):pass
    return False


def measured_pose_supported(state, oid, *, orientation=True, axes=None,
                            require_orientation_measurement=False):
    """Observed estimate support, scoped to the degrees needed by a predicate.

    Surface fitting need not prove a globally unique six-dimensional pose. Explicit
    ambiguity diagnostics apply only to relevant axes; geometric symmetries are
    equivalent for geometry predicates, not evidence of an absolute joint angle.
    """
    row=state['objects'][oid];stamp=row.get('sensor_validation',{})
    if row.get('basis')=='oracle_state':
        oracle=row.get('oracle_validation',{})
        return (state.get('oracle',{}).get('mode')=='live_state'
                and row.get('review',{}).get('pose',{}).get('status')=='oracle'
                and oracle.get('pose_digest')==digest(row['T_W_object'])
                and bool(oracle.get('robot_state_ref'))
                and oracle['robot_state_ref']==state_ref(state.get('runtime',{}).get('robot_state',{})))
    if row.get('basis')!='observation_estimate' or row.get('review',{}).get('pose',{}).get('status')!='observed':return False
    if stamp.get('pose_digest')!=digest(row['T_W_object']):return False
    if row.get('assembly') and stamp.get('assembly_digest')!=digest(row['assembly']):return False
    needed=range(3) if axes is None else axes
    observed=stamp.get('translation_observed_axes')
    if observed is not None:
        if any(observed[i] is False for i in needed):return False
    elif stamp.get('translation_observed') is False:return False
    if orientation:
        observed=stamp.get('orientation_observed_axes')
        equivalent=stamp.get('orientation_geometry_equivalent_axes') or [False]*3
        if require_orientation_measurement:
            # A fitted transform is an observed estimate even when uniqueness
            # diagnostics were not requested. Known orientation ambiguity cannot
            # establish an absolute angle; geometric equivalence does not fix it.
            if observed is not None:
                if any(v is False for v in observed):return False
            elif stamp.get('orientation_observed') is False:return False
        elif observed is not None:
            if any(v is False and not equivalent[i] for i,v in enumerate(observed)):return False
        elif stamp.get('orientation_observed') is False:return False
    return True


def measurement_supported(state, measurement, robot_state):
    # Historical outcomes remain queryable, but an old grasp/robot feasibility
    # measurement cannot establish the current robot-dependent physical state.
    return bool(measurement) and measurement.get('robot_state_ref')==state_ref(robot_state)


def field_supported(state, oid, field, robot_state, seen=None):
    seen=set(seen or []);key=(oid,field)
    if key in seen:return False
    seen.add(key)
    observed=state['objects'][oid].get('observed_state',{})
    evidence=observed.get('fields',{}).get(field,{})
    return (field in observed.get('values',{}) and evidence.get('status')=='observed'
            and derivation_supported(state,evidence.get('derivation'),robot_state,oid,seen))


def derivation_supported(state, derivation, robot_state, target_object, seen=None):
    """World-derived estimates survive unrelated motion, not changed inputs.

    Support means an observation-backed estimate, not proof of unique physical truth.
    """
    if not derivation:return False
    queries=derivation.get('queries',[]);physical=False;target_measured=False
    for query in queries:
        op=query.get('op');oid=query.get('object')
        if op in ('position','bounds','mesh','transform','object'):
            physical=True;target_measured |= oid==target_object
            if oid not in state['objects'] or not measured_pose_supported(
                    state,oid,orientation=op!='position',
                    require_orientation_measurement=op in ('transform','object')):return False
        elif op=='state':
            physical=True;target_measured |= oid==target_object
            if oid not in state['objects'] or not field_supported(state,oid,query['field'],robot_state,seen):return False
        elif op=='measurement':
            physical=True;measurement=state.get('measurements',{}).get(query.get('id'),{})
            if not measurement_supported(state,measurement,robot_state):return False
            target_measured |= target_object in measurement.get('object_ids',[measurement.get('object_id')])
        elif op=='proprioception':
            physical=True
            if derivation.get('robot_state_ref')!=state_ref(robot_state):return False
            target_measured |= target_object=='known_robot'
    if not physical or not target_measured:return False
    for dep in derivation.get('state_dependencies',[]):
        current=state['objects'].get(dep['object'],{}).get('observed_state',{})
        value={'value':current.get('values',{}).get(dep['field']),'evidence':current.get('fields',{}).get(dep['field'])}
        if digest(value)!=dep['digest']:return False
    for dep,guard in derivation.get('dependencies',{}).items():
        current=state['objects'].get(dep)
        if current is None:return False
        expected={'pose':digest(current['T_W_object']),'validation':digest(current.get('sensor_validation')),'review':digest(current.get('review',{}).get('pose'))}
        if any(guard.get(key)!=value for key,value in expected.items()):return False
    return True



def field_derivation_for_update(base, oid, field, before, written_fields):
    """Preserve old read versions when one atomic update replaces those fields.

    Expanding their already-validated physical dependencies avoids a new field
    depending on its own current value. The prior value/version is retained for
    audit; repeated filtering keeps a bounded set of physical input guards.
    """
    d=copy.deepcopy(base)
    queries=d.pop('field_queries',{}).get(field,d['queries'])
    relevant={q.get('object') for q in queries}
    guards={obj:guard for obj,guard in d.get('dependencies',{}).items() if obj in relevant}
    state_guards={(dep['object'],dep['field']):dep for dep in d.get('state_dependencies',[])}
    expanded=[];prior_inputs={}

    def depends_on_update(key, visited=()):
        if key in written_fields:return True
        if key in visited:return False
        entry=before['objects'][key[0]].get('observed_state',{}).get('fields',{}).get(key[1],{})
        return any(depends_on_update((q['object'],q['field']),visited+(key,))
                   for q in entry.get('derivation',{}).get('queries',[]) if q['op']=='state')

    def add_query(query, stack=()):
        key=(query.get('object'),query.get('field'))
        if query['op']!='state' or not depends_on_update(key):
            expanded.append(copy.deepcopy(query));return
        if key in stack:
            raise ValueError('Existing state evidence contains a cycle')
        observed=before['objects'][key[0]].get('observed_state',{})
        evidence=observed.get('fields',{}).get(key[1],{})
        prior=evidence.get('derivation',{})
        # A prior state read passed ctx.state and the dispatch dependency check.
        # Keep all its physical guards, including measurements of other objects.
        guards.update(copy.deepcopy(prior.get('dependencies',{})))
        state_guards.update({(dep['object'],dep['field']):copy.deepcopy(dep)
                             for dep in prior.get('state_dependencies',[])})
        prior_inputs[key]={'object':key[0],'field':key[1],
            'value':copy.deepcopy(observed.get('values',{}).get(key[1])),
            'evidence_revision':evidence.get('revision'),
            'decision':prior.get('decision')}
        for child in prior.get('queries',[]):add_query(child,stack+(key,))

    for query in queries:add_query(query)
    d['queries']=list({digest(q):q for q in expanded}.values())
    remaining={(q['object'],q['field']) for q in expanded if q['op']=='state'}
    d['state_dependencies']=[state_guards[key] for key in remaining if key in state_guards]
    d['dependencies']=guards
    d['prior_state_inputs']=list(prior_inputs.values())
    return d


def condition_axes(condition):
    if condition['kind']=='distance':return condition.get('axes',[0,1,2])
    if condition.get('frame')=='W':return [i for i,x in enumerate(condition['axis']) if abs(x)>1e-9]
    return [0,1,2]


def surface_error(row):
    stamp=row.get('sensor_validation',{})
    values=[stamp.get('surface_error_m',0.)]
    values.extend(v.get('p90_surface_error_m',0.) for v in stamp.get('report',{}).get('views',[]))
    return max(float(v) for v in values if v is not None)


def interval_decision(condition,value,radius):
    """Use actual fit residuals, not the caller's acceptance threshold, at success boundaries."""
    lo=condition.get('min',-np.inf);hi=condition.get('max',np.inf)
    if value-radius>=lo and value+radius<=hi:return 'SATISFIED'
    if value+radius<lo or value-radius>hi:return 'NOT_SATISFIED'
    return 'INCONCLUSIVE'


class CompletionMixin:
    def define_completion(self, goals):
        """Register all public task requirements before the first robot action.

        goals: {id: {description, conditions:[{kind, object, ...}]}}.
        Every condition is evaluated from world transforms/meshes or observed fields.
        Thresholds and coordinate conventions are agent-declared, never inferred from GT.
        """
        from .world import atomic,read
        if self.branch:raise ValueError('Completion requires the published world')
        if not isinstance(goals,dict) or not goals:raise ValueError('Nonempty task goals required')
        for name,g in goals.items():
            if not name or not g.get('description') or not g.get('conditions'):raise ValueError('Each goal needs a description and conditions')
            for c in g['conditions']:self._completion_measure(c,self.state())
        from .robot_home import required_goal, GOAL_ID
        required=required_goal(self.root)
        if required:
            goals=copy.deepcopy(goals)
            if GOAL_ID in goals and goals[GOAL_ID]!=required:raise ValueError('Required robot home goal cannot be overridden')
            goals[GOAL_ID]=required
        task=self.root/'task/task.json'
        spec={'schema':'codeworld-completion-contract-v1','task_digest':digest(read(task)), 'goals':copy.deepcopy(goals)}
        p=self.root/'task/completion.json'
        if p.exists():
            if read(p)!=spec:raise ValueError('Completion contract is immutable; start a new episode to revise task semantics')
            return spec
        if (self.root/'task/completion_started.json').exists():raise ValueError('Define task requirements before robot actions')
        atomic(p,spec);return spec

    def _completion_measure(self,c,state):
        if c['kind']=='robot_home':
            from .robot_home import measure_home
            result=measure_home(self.root,state.get('runtime',{}).get('robot_state',{}),c)
            return result,['known_robot'],'robot_home'
        oid=c['object'];row=state['objects'][oid];kind=c['kind']
        if kind=='state':
            value=row.get('observed_state',{}).get('values',{}).get(c['field'])
            if 'equals' not in c and 'min' not in c and 'max' not in c:raise ValueError('State condition needs equals or interval')
            return value,[oid],'state'
        a=self._mesh(oid,state).bounds
        other=c.get('reference');ids=[oid]
        if other:
            ids.append(other);b=self._mesh(other,state).bounds
        if kind in ('relative_center','bounds_gap'):
            if not other:raise ValueError('Explicit reference object required')
            axis=np.asarray(c['axis'],dtype=float)
            if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1):raise ValueError('Declare a unit axis')
            frame=c['frame']
            if frame=='reference':
                from .world import rigid
                axis=rigid(state['objects'][other]['T_W_object'])[:3,:3]@axis
            elif frame!='W':raise ValueError('Frame must be W or reference')
            if kind=='relative_center':value=float((a.mean(axis=0)-b.mean(axis=0))@axis)
            else:
                def extent(bounds,anchor):
                    corners=np.array(np.meshgrid(*zip(*bounds))).T.reshape(-1,3)@axis
                    return {'min':corners.min(),'max':corners.max(),'center':corners.mean()}[anchor]
                value=float(extent(a,c['object_anchor'])-extent(b,c['reference_anchor']))
        elif kind=='distance':
            if not other:raise ValueError('Explicit reference object required')
            axes=c.get('axes',[0,1,2])
            if not axes or len(set(axes))!=len(axes) or any(type(i)!=int or i not in (0,1,2) for i in axes):raise ValueError('Invalid distance axes')
            value=float(np.linalg.norm((a.mean(axis=0)-b.mean(axis=0))[axes]))
        else:raise ValueError('Unsupported completion condition: '+str(kind))
        if 'min' not in c and 'max' not in c:raise ValueError('An explicit acceptance interval is required')
        for k in ('min','max'):
            if k in c and (isinstance(c[k],bool) or not np.isfinite(c[k])):raise ValueError('Finite numeric thresholds required')
        if c.get('min',-np.inf)>c.get('max',np.inf):raise ValueError('Reversed interval')
        return value,ids,'pose'

    def verify_completion(self, goal_ids=None, *, robot_state):
        """Check a subtask subset or all goals using one consistent world snapshot.

        Returns VERIFIED / NOT_SATISFIED / INCONCLUSIVE. Images support world updates;
        no boolean or free-text visual-success input is accepted by this evaluator.
        """
        from .world import read
        if self.branch:raise ValueError('Hypothetical worlds cannot establish completion')
        self.verify_binding()
        contract=read(self.root/'task/completion.json')
        if contract['task_digest']!=digest(read(self.root/'task/task.json')):raise ValueError('Task changed after contract definition')
        ids=list(contract['goals']) if goal_ids is None else list(goal_ids)
        if not ids or len(ids)!=len(set(ids)) or any(i not in contract['goals'] for i in ids):raise ValueError('Invalid goal selection')
        from .flow import strict
        self.sync_robot_feedback()
        state=self.state();results={}
        for gid in ids:
            checks=[]
            for c in contract['goals'][gid]['conditions']:
                value,objects,aspect=self._completion_measure(c,state);supported=True
                if aspect=='robot_home':
                    # Robot feedback is ingested into CodeWorld, independent of scene pose fitting.
                    fresh=state_ref(state.get('runtime',{}).get('robot_state',{}))==state_ref(robot_state)
                    status=value['status'] if fresh else 'INCONCLUSIVE'
                    checks.append({'condition':c,'value':value,'objects':objects,'status':status,'robot_state_ref':state_ref(robot_state)})
                    continue
                for oid in objects:
                    row=state['objects'][oid]
                    review=(row.get('observed_state',{}).get('fields',{}).get(c['field'],{}) if aspect=='state' else row.get('review',{}).get(aspect,{}))
                    if strict(self.root):
                        # Object evidence is invalidated by object events, not by
                        # every subsequent movement of an unrelated robot joint.
                        supported &= review.get('status')=='observed' or (aspect=='pose' and review.get('status')=='oracle' and measured_pose_supported(state,oid))
                    else:supported &= review.get('status')=='observed' and evidence_current(self,review.get('evidence'),robot_state)
                    if aspect=='pose':
                        # A later commanded/predicted transform must not inherit an older observed stamp.
                        supported &= review.get('revision',-1)>=max([0]+[h['revision'] for h in state.get('history',[]) if oid in h.get('updates',{})])
                        if row.get('assembly') and not strict(self.root):
                            assembly=row['assembly'];supported &= assembly.get('basis')=='observation_estimate' and evidence_current(self,assembly.get('evidence'),robot_state)
                if strict(self.root):
                    for oid in objects:
                        row=state['objects'][oid]
                        if aspect=='pose':supported &= measured_pose_supported(state,oid,axes=condition_axes(c),orientation=c['kind']=='bounds_gap' or c.get('frame')=='reference')
                        else:
                            derivation=row.get('observed_state',{}).get('fields',{}).get(c['field'],{}).get('derivation',{})
                            supported &= derivation_supported(state,derivation,robot_state,oid)
                if aspect=='state':
                    if 'equals' in c:passed=value is not None and type(value)==type(c['equals']) and value==c['equals']
                    else:
                        if 'min' not in c and 'max' not in c:raise ValueError('State condition needs equals or an interval')
                        passed=isinstance(value,(int,float)) and not isinstance(value,bool) and c.get('min',-np.inf)<=value<=c.get('max',np.inf)
                else:passed=c.get('min',-np.inf)<=value<=c.get('max',np.inf)
                status='INCONCLUSIVE' if not supported else ('SATISFIED' if passed else 'NOT_SATISFIED')
                uncertainty=sum(surface_error(state['objects'][oid]) for oid in objects) if aspect=='pose' and strict(self.root) else 0.
                if supported and aspect=='pose':status=interval_decision(c,value,uncertainty)
                checks.append({'condition':c,'value':value,'objects':objects,'status':status,'measurement_error_radius_m':uncertainty})
            statuses=[r['status'] for r in checks]
            results[gid]={'status':'NOT_SATISFIED' if 'NOT_SATISFIED' in statuses else ('INCONCLUSIVE' if 'INCONCLUSIVE' in statuses else 'VERIFIED'),'checks':checks}
        statuses=[r['status'] for r in results.values()]
        return {'schema':'codeworld-completion-verification-v1','status':'NOT_SATISFIED' if 'NOT_SATISFIED' in statuses else ('INCONCLUSIVE' if 'INCONCLUSIVE' in statuses else 'VERIFIED'),
                'scope':'task' if set(ids)==set(contract['goals']) else 'subtask','goals':results,
                'state_revision':state['revision'],'world_digest':digest(state),'contract_digest':digest(contract),
                'geometry_revision':self.manifest['geometry_revision'],'robot_state_ref':state_ref(robot_state),
                'official_success':'NOT_EVALUATED'}

    def update_state(self, values, *, evidence, expected_revision=None):
        """Publish observation-estimated discrete/scalar object state, e.g. a joint angle.

        Values are perception estimates, not controller targets or evaluator feedback.
        """
        from .world import atomic
        from .flow import strict
        if strict(self.root) and not hasattr(self,'_flow_derivation'):raise ValueError('Physical state must be derived from CodeWorld computations, never set from a visual boolean')
        import fcntl
        from .review import stamp_review
        with (self.root/'state/update.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);s=self.state()
            if expected_revision is not None and s['revision']!=expected_revision:raise ValueError('Stale state revision')
            if not values or not isinstance(evidence,list) or not evidence:raise ValueError('Observed state needs values and capture evidence')
            digest(values)
            before=copy.deepcopy(s) if hasattr(self,'_flow_derivation') else None
            written_fields={(oid,field) for oid,fields in values.items() for field in fields}
            for oid,fields in values.items():
                if not isinstance(fields,dict) or not fields:raise ValueError('Nonempty state fields required')
                row=s['objects'][oid];row.setdefault('observed_state',{}).setdefault('values',{}).update(fields)
                stamp_review(row,'state','observed',s['revision']+1,evidence)
                for field in fields:
                    row['observed_state'].setdefault('fields',{})[field]=copy.deepcopy(row['review']['state'])
                    if hasattr(self,'_flow_derivation'):
                        derivation=field_derivation_for_update(self._flow_derivation,oid,field,before,written_fields)
                        entry=row['observed_state']['fields'][field]
                        entry['derivation']=derivation
                        if not derivation_supported(before,derivation,before.get('runtime',{}).get('robot_state',{}),oid):
                            entry['status']='pending'
                            entry['reason']='Derived estimate lacks current observation support; historical values remain available as evidence'
                if hasattr(self,'_flow_derivation') and any(row['observed_state']['fields'][f]['status']=='pending' for f in fields):
                    row['review']['state']['status']='pending'
            s['revision']+=1;s['history'].append({'revision':s['revision'],'kind':'observed_state','objects':list(values),'evidence':evidence})
            atomic(self.state_path,s)
        return s['revision']
