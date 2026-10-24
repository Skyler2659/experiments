"""Direct file transport; no task, geometry, perception or attachment policy."""
import time,uuid
from pathlib import Path
from .io import atomic,read
from .ownership import actor

class Robot:
    def __init__(self,workspace):
        self.root=Path(workspace).resolve();self.io=Path(read(self.root/'task/robot_connection.json')['io_directory'])
    def call(self,op,**args):
        ident=uuid.uuid4().hex;atomic(self.io/'requests'/f'{ident}.json',{'id':ident,'op':op,'args':args,'actor':actor()})
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            p=self.io/'responses'/f'{ident}.json'
            if p.exists():
                result=read(p)
                from codeworld_workspace.flow import strict
                from .flow import OPERATIONS
                if strict(self.root) and op in OPERATIONS:
                    # Return the value only after it is part of CodeWorld. Callers
                    # can inspect/log this published measurement without a second
                    # CLI round trip; later decisions use the same measurement ID.
                    from codeworld_workspace import World
                    world=World(self.root);world.sync_robot_feedback()
                    state=world.state()
                    if ident not in state.get('measurements',{}):
                        if result.get('status')=='ERROR':raise RuntimeError(result['error'])
                        raise RuntimeError('Controller response was not integrated into CodeWorld: '+ident)
                    return {'status':'MEASUREMENT_IN_WORLD','measurement_id':ident,
                            'request_id':ident,'state_revision':state['revision'],
                            'value':state['measurements'][ident]['value']}
                if result.get('status')=='ERROR':raise RuntimeError(result['error'])
                return result
            if (self.io/'closed.json').exists():
                raise RuntimeError('Robot episode is closed; no further actions can be submitted')
            time.sleep(.1)
        raise TimeoutError('No receipt yet; inspect request '+ident+' before resubmitting')
    def execute_decision(self, decision, index=0):
        """Submit only the exact command produced by a CodeWorld computation."""
        from codeworld_workspace.completion import path_in
        receipt=read(path_in(self.root,decision));a=receipt['result']['actions'][index]
        return self.call(a['op'],**a['args'],world_decision={'file':str(decision),'index':index})

    def observe(self):return self.call('observe')
    def capabilities(self):return read(self.root/'task/robot_connection.json').get('capabilities', {})
    def state(self, arm=None):
        from codeworld_workspace.flow import strict
        if strict(self.root):
            raise ValueError('Read robot proprioception through World.compute_decision and ctx.proprioception(); Robot.state() is unavailable in strict flow')
        state=read(self.io/'robot_state.json')
        if arm is None:return state
        from .multiarm import arm_index
        rows=state.get('arms',[state])
        return rows[arm_index(arm,[r.get('arm_name','arm0') for r in rows])]
    def select_arm(self,arm):return self.call('select_arm',arm=arm)
    def move_tcp_group(self,commands,**options):return self.call('move_tcp_group',commands=commands,**options)
    def joints(self,joint_positions,arm=None,**options):
        if arm is not None:options['arm']=arm
        return self.call('joints',joint_positions=joint_positions,**options)
    def joints_group(self,commands,**options):return self.call('joints_group',commands=commands,**options)
    def gripper_group(self,commands,steps=80,**options):return self.call('gripper_group',commands=commands,steps=steps,**options)
    def check_ik(self,pose_W,arm=None):
        return self.call('check_ik',pose_W=pose_W,**({} if arm is None else {'arm':arm}))
    def check_motion(self,commands,**options):
        """Evaluate candidate joint/TCP paths against current CodeWorld, without moving."""
        return self.call('check_motion',commands=commands,**options)
    def move_tcp(self,pose_W,max_steps=150,**servo_options):return self.call('move_tcp',pose_W=pose_W,max_steps=max_steps,**servo_options)
    def gripper(self,mode,steps=12,arm=None,**options):return self.call('gripper',mode=mode,steps=steps,**({} if arm is None else {'arm':arm}),**options)
    def step(self,actions):return self.call('step',actions=actions)
    def move_from_world(self, decision_file, waypoint_index, quaternion_xyzw, max_steps=150, arm=None):
        """Execute a waypoint from an explicit world-derived candidate decision.

        Orientation and tool model remain the agent's responsibility. This binds
        action provenance, not a full robot feasibility certificate.
        """
        from codeworld_workspace import World
        import hashlib
        path=Path(decision_file).resolve();data=path.read_bytes()
        import json
        decision=json.loads(data)
        if decision.get('schema')!='codeworld-route-decision-v1':raise ValueError('Unknown decision schema')
        if decision.get('branch') is not None:raise ValueError('Publish observed state before executing a hypothesis')
        state=World(self.root).state()
        if (state['revision']!=decision['state_revision'] or
                state['geometry_revision']!=decision['geometry_revision']):raise ValueError('Stale world decision; replan')
        chosen=next((c for c in decision['candidates'] if c['name']==decision['selected']),None)
        if chosen is None:raise ValueError('No selected route')
        if not isinstance(waypoint_index,int) or not 1<=waypoint_index<len(chosen['points_W_m']):raise ValueError('Invalid waypoint index')
        return self.call('move_tcp',pose_W={'translation_m':chosen['points_W_m'][waypoint_index],
            'quaternion_xyzw':quaternion_xyzw},max_steps=max_steps,decision_ref={
            'file':str(path),'sha256':hashlib.sha256(data).hexdigest(),'state_revision':state['revision'],
            'selected':decision['selected'],'waypoint_index':waypoint_index},**({} if arm is None else {'arm':arm}))

    def check_grasp_geometry(self, object_id, **candidate):
        """Read reconstructed jaw engagement geometry without moving the robot."""
        from codeworld_workspace import World
        return World(self.root).check_grasp_geometry(object_id, **candidate)

    def assess_grasp(self, **evidence):
        """Assess stored RGB-D/TCP lift evidence; no actuation or world update."""
        from .grasp import assess_grasp
        from codeworld_workspace import World
        world=World(self.root)
        if evidence['object_id'] not in world.objects():raise KeyError(evidence['object_id'])
        from codeworld_workspace.flow import strict
        if strict(self.root):
            result=world.ingest_grasp_feedback(**evidence)
            return {**result,'value':world.state()['measurements'][result['measurement_id']]['value']}
        result=assess_grasp(self.root, **evidence)
        result['state_revision']=world.state()['revision']
        result['actor']=actor()
        return result

    def verify_completion(self, goal_ids=None):
        """Evaluate and save task/subtask conditions from current observed CodeWorld."""
        from codeworld_workspace import World
        result=World(self.root).verify_completion(goal_ids,robot_state=read(self.io/'robot_state.json'))
        path=self.root/'work/completion'/f'{uuid.uuid4().hex}.json'
        atomic(path,result)
        return {**result,'verification':str(path.relative_to(self.root))}

    def finish(self, *, verification=None, outcome='success', reason=None, judgment=None, system_reason=None):
        """Physical outcomes use World; typed runtime termination is not a task claim.

        system_reason: {kind: budget_exhausted|timeout|tool_error|process_error|
        interrupted, detail: nonempty str}. It only applies to failed/aborted exits.
        """
        args=dict(verification=verification,outcome=outcome,reason=reason,judgment=judgment)
        if system_reason is not None:args['system_reason']=system_reason
        return self.call('finish',**args)
