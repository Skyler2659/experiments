"""Macro/Micro grounding on the accepted textured mesh and part programs.

Geometry runs on the reconstruction host. Decisions run through planner_desktop
on the signed-in desktop. No task-asset catalog or simulator object state enters
this adapter. A grounded command remains a proposal until a controller validates
the full robot path against this same snapshot.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import trimesh
from .mesh_world import nearest_mesh_surface, sha256
from .scene_structuring import StructuredScene


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def vector(value, size=3):
    out=np.asarray(value, dtype=float)
    if out.shape!=(size,) or not np.isfinite(out).all():raise ValueError('invalid finite vector')
    return out


class MeshPlanner:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.world=StructuredScene(root)
        self.status=json.loads((self.root/'status.json').read_text())
        if self.status['status']!='SCENE_STRUCTURED':raise ValueError('structure batch incomplete')
        if self.world.source['frame']!='robot_base' or self.world.source['units']!='m':
            raise ValueError('planner requires robot-base metres')
        self.meshes={}
        self.binding={'scene':sha256(self.root/'frozen/scene.json'),'catalog':sha256(self.root/'agent_scene.json'),'programs':{},'assets':{}}
        for row in self.status['objects']:
            oid=row['object_id'];program=self.root/row['program']
            self.binding['programs'][oid]={'code':sha256(program)}
            if oid in self.world.structured:
                self.binding['programs'][oid]['structure']=sha256(program.with_name('structure.json'))
                self.binding['programs'][oid]['part_meshes']={pid:sha256(program.parent/part['mesh_ref'])
                    for pid,part in self.world.structured[oid].parts.items() if 'mesh_ref' in part}
            self.binding['assets'][oid]=sha256(self.world.root/self.world.node(oid)['asset']['mesh_ref'])
        self.snapshot_id=digest(self.binding)
        paths=[self.root/'frozen/scene.json',self.root/'agent_scene.json',self.root/'status.json']
        for row in self.status['objects']:
            program=self.root/row['program'];paths.append(program)
            paths.append(self.world.root/self.world.node(row['object_id'])['asset']['mesh_ref'])
            if row['object_id'] in self.world.structured:
                paths.append(program.with_name('structure.json'))
                paths.extend(program.parent/p['mesh_ref'] for p in self.world.structured[row['object_id']].parts.values() if 'mesh_ref' in p)
        self.bound_files={p:sha256(p) for p in paths}
        for oid in self.world.functional:
            path = self.root/'objects'/oid/'functional/accepted.json'
            self.binding['programs'][oid]['functional_geometry'] = sha256(path)
            self.bound_files[path] = sha256(path)
        self.snapshot_id = digest(self.binding)

    def assert_current(self):
        if any(not p.is_file() or sha256(p)!=expected for p,expected in self.bound_files.items()):
            raise ValueError('reconstruction changed; reprepare planner snapshot')

    def resolve(self, ref):
        if not isinstance(ref,dict) or set(ref)!={'object_id','part_id'}:raise ValueError('reference needs object_id and nullable part_id')
        oid=ref['object_id'];pid=ref['part_id'];self.world.node(oid)
        if pid is not None and (oid not in self.world.structured or pid not in self.world.structured[oid].parts):
            raise ValueError('unknown reconstructed part')
        return oid,pid

    def mesh(self, ref):
        key=self.resolve(ref)
        if key not in self.meshes:
            oid,pid=key
            self.meshes[key]=(self.world.mesh(oid) if pid is None else
                              trimesh.util.concatenate(self.world.structured[oid].part_pieces(pid)))
        return self.meshes[key]

    def query(self, request):
        """Finite exact-candidate queries; none assert real-world contact."""
        self.assert_current()
        if request['kind']=='view':
            view=next((v for v in self.world.source.get('observations',[]) if v['view_id']==request['view_id']),None)
            if view is None:raise ValueError('unknown calibrated observation')
            path=self.world.root/view['rgb_ref']
            if sha256(path)!=view['rgb_sha256']:raise ValueError('view evidence changed')
            return {'snapshot_id':self.snapshot_id,'request':request,'image':{'path':'scene/frozen/'+view['rgb_ref'],
                    'sha256':view['rgb_sha256'],'view_id':view['view_id'],'role':'observed_rgb'},
                    'calibration':view,'note':'historical reconstruction observation; not a new robot capture'}
        kind=request['kind'];ref=request['ref'];oid,pid=self.resolve(ref)
        mesh=self.mesh(ref)
        result={'snapshot_id':self.snapshot_id,'request':request,'accuracy':'reconstructed_candidate_only'}
        if kind=='bounds':result.update(bounds_W_m=mesh.bounds.tolist(),size_W_m=mesh.extents.tolist())
        elif kind=='surface':
            # A point in the selected part's W-axis bounds is projected onto its
            # actual triangles. This never treats a hollow object's box as solid.
            fraction=vector(request['fraction'])
            if np.any((fraction<0)|(fraction>1)):raise ValueError('fraction outside [0,1]')
            seed=mesh.bounds[0]+fraction*mesh.extents
            point,distance,normal=nearest_mesh_surface(mesh,seed,return_normal=True)
            result.update(point_W_m=point.tolist(),normal_W=normal.tolist(),normal_basis='triangle winding; outward direction unverified',projection_distance_m=distance,
                          meaning='surface anchor, not a certified grasp or interior point')
        elif kind=='nearest':
            point,distance,normal=nearest_mesh_surface(mesh,vector(request['point_W_m']),return_normal=True)
            result.update(point_W_m=point.tolist(),normal_W=normal.tolist(),normal_basis='triangle winding; outward direction unverified',distance_m=distance,
                          meaning='surface anchor, not a certified grasp or interior point')
        elif kind=='part':
            if pid is None:raise ValueError('part query needs part_id')
            result['part']=self.world.part_info(oid,pid)
        elif kind=='feature':
            if pid is not None:raise ValueError('functional features are object-scoped')
            result['feature']=self.world.feature_info(oid,request['feature_id'])
        else:raise ValueError('unsupported query kind')
        result['query_id']=digest(result)
        return result

    def context(self):
        objects=[]
        for row in self.status['objects']:
            oid=row['object_id'];info=self.world.object_info(oid)
            info['parts']=[];info['relations']=[]
            if oid in self.world.structured:
                world=self.world.structured[oid]
                info['parts']=[self.world.part_info(oid,pid) for pid in world.parts]
                info['relations']=world.program['relations']
                info['structure_limitations']=world.program['limitations']
                report=json.loads((self.root/row['program']).with_name('verification.json').read_text())
                info['unresolved_area_fraction']=report['unresolved_area_fraction']
            info['program_ref']=row['program']
            current=self.mesh({'object_id':oid,'part_id':None})
            info['current_geometry']={'bounds_W_m':current.bounds.tolist(),'size_W_m':current.extents.tolist(),
                'snapshot_id':self.snapshot_id,'authority':'current reconstructed candidate, not certified real geometry'}
            objects.append(info)
        return {'schema_version':'mesh-planner-context-v1','snapshot_id':self.snapshot_id,
                'world_id':self.world.source['world_id'],'world_version':self.world.source['world_version'],
                'frame':'robot_base','units':'m','objects':objects,
                'robot_observation':self.world.source.get('robot_state'),
                'observations':self.world.source.get('observations',[]),
                'calibration':self.world.source.get('camera_registration'),
                'query_capabilities':['bounds','part','feature','surface','nearest'],
                'state_contract':{'geometry_queries':'current snapshot; frozen geometry only for explicit reference queries',
                    'program_edits':'counterfactual hypotheses until independently observed',
                    'completion':'independent evidence tied to the current snapshot and physical observation'},
                'limits':['Scene is an observation snapshot; historical wrist views are not current robot state.',
                          'Generated or completed hidden surfaces are hypotheses, not measured free space.',
                          'Bounds do not establish a cavity or containment. Surface regions are not detachable solids.',
                          'Texture identifies appearance; it does not establish friction or material mechanics.',
                          'No validated metric error bound is supplied by default. Request observation or refinement when needed.']}

    def export(self, output, instruction):
        self.assert_current();out=Path(output).resolve()
        if out==self.root or self.root in out.parents:raise ValueError('export must be outside scene package')
        out.mkdir(parents=True,exist_ok=True)
        # A complete portable package makes all texture, code, evidence and part
        # data available, while the prompt uses a hierarchy-sized index.
        shutil.copytree(self.root,out/'scene',dirs_exist_ok=True,symlinks=False)
        context=self.context();context['instruction']=instruction;context['images']=[]
        for view in context['observations']:
            source=self.world.root/view['rgb_ref']
            if sha256(source)!=view['rgb_sha256']:raise ValueError('observation changed')
            context['images'].append({'path':'scene/frozen/'+view['rgb_ref'],
                                      'sha256':view['rgb_sha256'],'role':'observed_rgb','view_id':view['view_id']})
        for name in ('canonical_from_code.png',):
            if (self.root/name).exists():context['images'].append({'path':'scene/'+name,'sha256':sha256(self.root/name),'role':'candidate_render'})
        for row in self.status['objects']:
            program=Path(row['program'])
            image=program.parent/'renders/view_0.png' if row['status']=='STRUCTURALIZED' else program.parent/'asset.png'
            if (self.root/image).exists():context['images'].append({'path':'scene/'+str(image),'sha256':sha256(self.root/image),'role':'part_or_asset_preview','object_id':row['object_id']})
        write(out/'context.json',context)
        write(out/'binding.json',self.binding)
        return context

    def validate_condition(self, item):
        if not isinstance(item,dict):raise ValueError('condition must be an object')
        kind=item.get('kind')
        if kind=='fact':
            if item.get('name') not in {'holding','lifted_from','supported_by','inside','stable','observed','gripper_open','gripper_fully_open','separated_from','flat','upright','released'}:raise ValueError('unknown observed fact')
            if not isinstance(item.get('refs'),list) or not item['refs']:raise ValueError('fact needs refs')
            arity=2 if item['name'] in {'inside','supported_by','separated_from','lifted_from'} else 1
            if len(item['refs'])!=arity:raise ValueError('fact has incorrect subject/target arity')
            if len({digest(r) for r in item['refs']})!=arity:raise ValueError('fact repeats the same subject and target')
            if 'after_goal_id' in item and (not isinstance(item['after_goal_id'],str) or not item['after_goal_id']):raise ValueError('invalid after_goal_id')
            for ref in item['refs']:self.resolve(ref)
        elif kind=='distance_le':
            for key in ('a','b'):
                q=item[key]
                if q.get('kind')!='surface':raise ValueError('distance condition needs surface anchors')
                self.resolve(q['ref']);fraction=vector(q['fraction'])
                if np.any((fraction<0)|(fraction>1)):raise ValueError('invalid anchor fraction')
            if isinstance(item.get('threshold_m'),bool) or not isinstance(item.get('threshold_m'),(float,int)) or not np.isfinite(item['threshold_m']) or item['threshold_m']<0:raise ValueError('invalid distance threshold')
        elif kind=='joint_range':
            self.resolve(item['ref']);lo,hi=vector(item['range'],2)
            if lo>hi:raise ValueError('reversed joint range')
        else:raise ValueError('unsupported completion condition')
        return item

    def validate_plan(self, plan):
        if plan.get('snapshot_id')!=self.snapshot_id:raise ValueError('stale plan')
        goals=plan.get('goals',[])
        if not goals or not plan.get('reason'):raise ValueError('invalid goal list')
        seen=set()
        for goal in goals:
            gid=goal['goal_id']
            if not isinstance(gid,str) or not gid or gid in seen:raise ValueError('duplicate/invalid goal')
            earlier=set(seen)
            seen.add(gid)
            if not goal.get('intent') or not isinstance(goal.get('refs'),list) or not goal['refs']:raise ValueError('goal needs intent and concrete references')
            for ref in goal['refs']:self.resolve(ref)
            if not goal.get('completion'):raise ValueError('goal needs executable completion conditions')
            for field in ('entry_conditions','preconditions','completion'):
                if not isinstance(goal.get(field,[]),list):raise ValueError('goal conditions must be lists')
            for cond in goal.get('entry_conditions',[])+goal.get('preconditions',[])+goal['completion']:
                self.validate_condition(cond)
                if cond.get('after_goal_id') and cond['after_goal_id'] not in earlier:raise ValueError('after_goal_id must reference an earlier ordered goal')
                for ref in self.condition_refs(cond):
                    if not any(self.ref_covers(declared,ref) for declared in goal['refs']):
                        raise ValueError('goal '+gid+' refs omit reference '+json.dumps(ref)+' used by condition '+json.dumps(cond))
            if goal.get('on_unknown') not in {'OBSERVE','REFINE'} or goal.get('on_failure') not in {'REPLAN','RETRY'}:raise ValueError('missing recovery branch')
        if not plan.get('task_completion'):raise ValueError('task needs independent final conditions')
        for cond in plan['task_completion']:
            self.validate_condition(cond)
            if cond.get('after_goal_id') and cond['after_goal_id'] not in seen:raise ValueError('after_goal_id must reference an ordered goal')
        return plan

    @staticmethod
    def condition_refs(condition):
        if condition['kind']=='fact':return condition['refs']
        if condition['kind']=='distance_le':return [condition[k]['ref'] for k in ('a','b')]
        return [condition['ref']]

    @staticmethod
    def ref_covers(declared, referenced):
        return declared['object_id']==referenced['object_id'] and (declared['part_id'] is None or declared['part_id']==referenced['part_id'])

    def evaluate(self, condition, evidence):
        """Evidence is supplied by an observation verifier, never by the planner."""
        self.validate_condition(condition)
        unknown={'value':'UNKNOWN','condition':condition}
        if evidence.get('snapshot_id')!=self.snapshot_id:return {**unknown,'reason':'missing/current snapshot evidence required'}
        kind=condition['kind']
        if kind=='fact':
            for fact in evidence.get('facts',[]):
                if fact.get('condition')==condition and fact.get('verifier') and fact.get('evidence_refs') and fact.get('value') in {'TRUE','FALSE','UNKNOWN'}:
                    return {'value':fact['value'],'condition':condition,'evidence_refs':fact['evidence_refs'],'verifier':fact['verifier']}
            return {**unknown,'reason':'independent observation verifier has not established this relation'}
        if kind=='distance_le':
            a=self.query(condition['a']);b=self.query(condition['b'])
            distance=float(np.linalg.norm(np.array(a['point_W_m'])-b['point_W_m']))
            errors=evidence.get('position_error_bounds',{})
            records=[errors.get(condition[k]['ref']['object_id'],{}) for k in ('a','b')]
            if not all(r.get('evidence_refs') and isinstance(r.get('bound_m'),(int,float)) and np.isfinite(r['bound_m']) and r['bound_m']>=0 for r in records):
                return {**unknown,'candidate_distance_m':distance,'reason':'metric error bounds not validated'}
            error=sum(r['bound_m'] for r in records);threshold=condition['threshold_m']
            value='TRUE' if distance+error<=threshold else 'FALSE' if distance-error>threshold else 'UNKNOWN'
            return {'value':value,'condition':condition,'distance_interval_m':[max(0,distance-error),distance+error]}
        oid,_=self.resolve(condition['ref']);joint=self.world.joint_state(oid)
        record=evidence.get('joint_error_bounds',{}).get(oid,{})
        if not joint or not record.get('evidence_refs') or not isinstance(record.get('bound'),(int,float)) or not np.isfinite(record['bound']) or record['bound']<0:
            return {**unknown,'reason':'observed joint value and validated uncertainty required'}
        value=joint['value'];err=record['bound'];lo,hi=condition['range']
        result='TRUE' if lo<=value-err and value+err<=hi else 'FALSE' if value+err<lo or value-err>hi else 'UNKNOWN'
        return {'value':result,'condition':condition,'interval':[value-err,value+err]}

    def ground(self, plan, decision, *, evidence=None):
        """Compile a current Micro proposal or refuse a completion claim."""
        self.assert_current();self.validate_plan(plan);evidence=evidence or {}
        if decision.get('snapshot_id')!=self.snapshot_id:raise ValueError('stale Micro decision')
        goal=next((g for g in plan['goals'] if g['goal_id']==decision.get('goal_id')),None)
        if goal is None:raise ValueError('Micro goal absent from plan')
        kind=decision['kind'];result={'snapshot_id':self.snapshot_id,'decision':decision}
        if kind in {'GOAL_DONE','TASK_DONE'}:
            conditions=goal['completion'] if kind=='GOAL_DONE' else plan['task_completion']
            checks=[self.evaluate(c,evidence) for c in conditions]
            return {**result,'status':'VERIFIED' if all(c['value']=='TRUE' for c in checks) else 'NEED_VERIFICATION','checks':checks}
        if kind in {'OBSERVE','REFINE','REPLAN'}:
            refs=decision.get('refs',[])
            if kind!='REPLAN' and not refs:raise ValueError('observation/refinement needs targets')
            for ref in refs:self.resolve(ref)
            if not decision.get('reason'):raise ValueError('request needs reason')
            return {**result,'status':kind+'_REQUESTED','request':{'refs':refs,'reason':decision['reason']}}
        checks=[self.evaluate(c,evidence) for c in goal.get('entry_conditions',[])+goal.get('preconditions',[])]
        if any(c['value']!='TRUE' for c in checks):return {**result,'status':'PRECONDITION_UNVERIFIED','checks':checks}
        ref=decision.get('contact_ref')
        if ref is not None:self.resolve(ref)
        if kind=='MOVE_TCP':
            target=decision['target']
            if target['anchor']['kind'] not in {'surface','nearest'}:raise ValueError('movement anchor must be a current surface or nearest query')
            anchor=self.query(target['anchor'])
            point=np.asarray(anchor['point_W_m'])+vector(target['offset_W_m'])
            quat=vector(target['quaternion_xyzw'],4)
            if abs(np.linalg.norm(quat)-1)>1e-4:raise ValueError('target quaternion not normalized')
            action={'kind':kind,'target_pose_W':{'translation_m':point.tolist(),'quaternion_xyzw':quat.tolist()},'gripper':None}
            result['grounding']=anchor
        elif kind=='SET_GRIPPER':
            if decision.get('gripper') not in {'OPEN','CLOSE'}:raise ValueError('invalid gripper command')
            action={'kind':kind,'target_pose_W':None,'gripper':decision['gripper']}
        else:raise ValueError('unsupported Micro kind')
        action.update(reason=decision['reason'],expected_observation=decision['expected_observation'],object_id=ref['object_id'] if ref else None)
        from .action_contract import validate_open_action
        validate_open_action(action)
        return {**result,'status':'GROUNDED_PROPOSAL','action':action,
                'execution_requirements':['fresh robot checkpoint in W','IK and full robot plus held-object path check on this mesh snapshot','observed post-action update'],
                'note':'no motion authorized by grounding; original proxy-only collision world is not interchangeable'}


class MeshPlanningSession:
    """Versioned decision state; only an independent verifier advances goals.

    The execution host supplies a newly reconstructed package after motion.
    Merely receiving a controller return does not update or complete a goal.
    """
    def __init__(self, planner, instruction):
        self.planner=planner;self.instruction=instruction;self.plan=None
        self.active=0;self.history=[];self.awaiting_observation=False;self.pending=None;self.completed=False

    def set_plan(self, plan):
        if self.completed:raise ValueError('task already independently verified')
        if self.awaiting_observation:raise ValueError('post-action observation required before replanning')
        self.plan=copy.deepcopy(self.planner.validate_plan(plan));self.active=0

    def decide(self, decision, evidence=None):
        if self.completed:raise ValueError('task already independently verified')
        if self.awaiting_observation:raise ValueError('cannot issue another command before world update')
        if self.plan is None:raise ValueError('Macro plan required')
        if decision['goal_id']!=self.plan['goals'][self.active]['goal_id']:raise ValueError('inactive goal')
        result=self.planner.ground(self.plan,decision,evidence=evidence)
        if result['status']=='VERIFIED' and decision['kind']=='GOAL_DONE':
            self.active=min(self.active+1,len(self.plan['goals'])-1)
        if result['status']=='VERIFIED' and decision['kind']=='TASK_DONE':self.completed=True
        if decision['kind']=='REPLAN':self.plan=None
        self.pending=result if result['status']=='GROUNDED_PROPOSAL' else None
        self.history.append(result)
        return result

    def record_execution(self, proposal, receipt):
        if self.pending!=proposal or proposal.get('status')!='GROUNDED_PROPOSAL' or proposal['snapshot_id']!=self.planner.snapshot_id:
            raise ValueError('execution does not refer to this grounded snapshot')
        if receipt.get('snapshot_id')!=self.planner.snapshot_id or receipt.get('action_digest')!=digest(proposal['action']):
            raise ValueError('execution receipt mismatched')
        if type(receipt.get('steps_executed')) is not int or receipt['steps_executed']<0:raise ValueError('invalid execution receipt')
        self.history.append({'execution':receipt})
        self.pending=None
        self.awaiting_observation=receipt['steps_executed']>0 or receipt.get('outcome')=='EXECUTION_UNKNOWN'

    def update(self, planner):
        if planner.snapshot_id==self.planner.snapshot_id:raise ValueError('new observation snapshot required')
        if planner.world.source['world_id']!=self.planner.world.source['world_id'] or planner.world.source['world_version']<=self.planner.world.source['world_version']:
            raise ValueError('update needs a later version of the same world')
        self.history.append({'world_update':{'before':self.planner.snapshot_id,'after':planner.snapshot_id}})
        # Geometry/part identity may have changed. Require Macro to reconsider
        # with history rather than silently rebinding old targets to new meshes.
        self.planner=planner;self.plan=None;self.active=0;self.awaiting_observation=False
        self.pending=None

    def save(self, path):
        state={'snapshot_id':self.planner.snapshot_id,'instruction':self.instruction,'plan':self.plan,
               'active':self.active,'history':self.history,'awaiting_observation':self.awaiting_observation,'pending':self.pending,'completed':self.completed}
        path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');write(temporary,state);temporary.replace(path)

    @classmethod
    def load(cls, planner, path):
        state=json.loads(Path(path).read_text())
        if state['snapshot_id']!=planner.snapshot_id:raise ValueError('resume with original snapshot then explicitly update')
        session=cls(planner,state['instruction'])
        for key in ('plan','active','history','awaiting_observation','pending','completed'):setattr(session,key,state[key])
        return session

    def dispatch(self, proposal, executor, journal):
        """Adapter hook; executor must perform same-mesh full-path validation.

        Persist intent before crossing the robot boundary. A timeout cannot be
        interpreted as zero motion or retried on the old world.
        """
        self.planner.assert_current()
        if self.awaiting_observation or self.pending!=proposal:raise ValueError('command not currently dispatchable')
        self.awaiting_observation=True
        self.history.append({'event':'EXECUTION_INTENT','snapshot_id':self.planner.snapshot_id,'action_digest':digest(proposal['action'])})
        self.save(journal)
        try:
            receipt=executor(proposal)
            self.record_execution(proposal,receipt)
        except Exception:
            self.awaiting_observation=True
            self.history.append({'event':'EXECUTION_UNKNOWN','snapshot_id':self.planner.snapshot_id})
            self.save(journal)
            raise
        self.save(journal)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['export','query','validate-plan','ground'])
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--instruction')
    p.add_argument('--input');p.add_argument('--plan');p.add_argument('--evidence');a=p.parse_args();planner=MeshPlanner(a.scene)
    if a.command=='export':result=planner.export(a.output,a.instruction)
    elif a.command=='query':result=[planner.query(q) for q in json.loads(Path(a.input).read_text())]
    elif a.command=='validate-plan':result=planner.validate_plan(json.loads(Path(a.input).read_text()))
    else:result=planner.ground(json.loads(Path(a.plan).read_text()),json.loads(Path(a.input).read_text()),evidence=json.loads(Path(a.evidence).read_text()) if a.evidence else None)
    if a.command!='export':write(a.output,result)
    print(json.dumps({'command':a.command,'snapshot_id':planner.snapshot_id,'status':result.get('status') if isinstance(result,dict) else 'QUERIED'}))


if __name__=='__main__':main()
