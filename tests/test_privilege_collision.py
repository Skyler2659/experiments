"""Geometry and native dispatch regressions; no simulator, RGB fitting or GPU."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip('fcl')
pytest.importorskip('rtree')
from privilege.collision import SceneCollision
from privilege.storage import from_meshes
from privilege.world import atomic


@pytest.fixture
def setup(tmp_path):
    urdf = tmp_path/'robot.urdf'
    urdf.write_text('''<robot name="test"><link name="base"/>
    <link name="hand"><collision><origin xyz=".8 0 0"/><geometry><box size=".12 .06 .06"/></geometry></collision></link>
    <joint name="j" type="revolute"><parent link="base"/><child link="hand"/><axis xyz="0 0 1"/></joint>
    <link name="f1"><collision><geometry><box size=".06 .02 .04"/></geometry></collision></link>
    <joint name="g1" type="prismatic"><parent link="hand"/><child link="f1"/><origin xyz="1 .05 0"/><axis xyz="0 1 0"/></joint>
    <link name="f2"><collision><geometry><box size=".06 .02 .04"/></geometry></collision></link>
    <joint name="g2" type="prismatic"><parent link="hand"/><child link="f2"/><origin xyz="1 -.05 0"/><axis xyz="0 -1 0"/></joint></robot>''')
    left=np.eye(4);left[0,3]=-5
    spec={'urdf':str(urdf),'bases':{'right':np.eye(4).tolist(),'left':left.tolist()},
          'joint_names':['j'],'finger_joints':['g1','g2'],'finger_links':['f1','f2'],
          'finger_scale_m':.1,'ee_link':'hand'}
    raw={f'{a}_arm_joint_state':[0.] for a in ('left','right')}
    raw.update({f'{a}_ee_joint_state':[.5] for a in ('left','right')})
    def build(objects, held=None):
        rows={}
        for oid,(size,pos) in objects.items():
            mesh=tmp_path/f'{oid}.glb';trimesh.creation.box(size).export(mesh)
            t=np.eye(4);t[:3,3]=pos
            rows[oid]={'mesh':mesh,'T_W_object':t,'movable':True}
        world=from_meshes(rows,tmp_path/f'world{len(list(tmp_path.glob("world*")))}')
        if held:
            s=world.state();s['attachments']={a:{'object_id':o,'T_E_object':s['objects'][o]['T_W_object']} for a,o in held.items()}
            atomic(world.state_path,s)
        return world,SceneCollision(world,spec)
    return raw,spec,build


def test_swept_arm_collision_with_clear_endpoints(setup):
    raw,spec,build=setup
    world,c=build({'obstacle':([.12]*3,[0,.8,0])})
    assert c.check(raw,[])['status']=='CLEAR'
    end=copy.deepcopy(raw);end['right_arm_joint_state']=[np.pi]
    assert c.check(end,[])['status']=='CLEAR'
    r=c.check(raw,[{'right_arm_joint_state':[np.pi]}])
    assert r['status']=='COLLISION'
    assert any(x['first']=='right/hand' and x['second']=='obstacle' for x in r['hits'])


def test_payload_sweep_rotation_and_current_correction(setup):
    raw,spec,build=setup
    world,c=build({'payload':([.3,.04,.04],[2,0,0]),'obstacle':([.1]*3,[0,2,0])}, {'right':'payload'})
    path=[{'right_arm_joint_state':[np.pi]}]
    r=c.check(raw,path)
    assert r['status']=='COLLISION' and any({x['first'],x['second']}=={'payload','obstacle'} for x in r['hits'])
    # Corrected attachment must be used immediately, with no stale geometry cache.
    s=world.state();s['attachments']['right']['T_E_object'][0][3]=3
    s['objects']['payload']['T_W_object'][0][3]=3;s['revision']+=1;atomic(world.state_path,s)
    assert c.check(raw,path)['status']=='CLEAR'


def test_measured_empty_gripper_does_not_sweep_payload(setup):
    raw,spec,build=setup
    _,c=build({'payload':([.3,.04,.04],[2,0,0]),'obstacle':([.1]*3,[0,2,0])},{'right':'payload'})
    raw['right_ee_joint_state']=[0.]
    assert c.check(raw,[{'right_arm_joint_state':[np.pi]}])['status']=='CLEAR'


def test_finger_sweep_and_narrow_contact_intent(setup):
    raw,spec,build=setup
    raw['right_ee_joint_state']=[0.]
    _,c=build({'block':([.02]*3,[1,.10,0])})
    path=[{'right_ee_joint_state':[1.]}]
    assert c.check(raw,path)['status']=='COLLISION'
    c.expect_contacts(grasp={'right':'block'})
    r=c.check(raw,path)
    assert r['status']=='CLEAR' and r['expected_contacts']
    _,other=build({'block':([.1]*3,[0,.8,0])})
    other.expect_contacts(grasp={'right':'block'})
    assert other.check(raw,[{'right_arm_joint_state':[np.pi]}])['status']=='COLLISION'


def test_stationary_other_arm_is_an_obstacle(setup):
    raw,spec,build=setup
    _,c=build({'distant':([.1]*3,[10,10,10])})
    # Left hand at y=.8 intercepts the right hand's quarter turn.
    base=np.eye(4);base[:3,3]=[-.8,.8,0]
    c.robot.spec['bases']['left']=base.tolist()
    r=c.check(raw,[{'right_arm_joint_state':[np.pi]}])
    assert r['status']=='COLLISION'
    assert any(x['first'].startswith('right/') and x['second'].startswith('left/') for x in r['hits'])


def test_unchanged_other_arm_contact_does_not_block_endpoint(setup):
    raw, spec, build = setup
    _, c = build({'existing': ([.04]*3, [-4.2, 0, 0])})
    assert c.check(raw, [])['status'] == 'COLLISION'  # Not a claim of global clearance.
    result = c.check(raw, [{'right_arm_joint_state': [.2]}])
    assert result['status'] == 'CLEAR'
    assert {'first': 'left/hand', 'second': 'existing'} in result['unchanged_initial_contacts']
    # Moving the touching arm itself does not receive the unrelated-arm exemption.
    assert c.check(raw, [{'left_arm_joint_state': [.01]}])['status'] == 'COLLISION'


def test_unrelated_contact_does_not_hide_new_active_arm_collision(setup):
    raw, spec, build = setup
    _, c = build({'existing': ([.04]*3, [-4.2, 0, 0]),
                  'new': ([.12]*3, [0, .8, 0])})
    result = c.check(raw, [{'right_arm_joint_state': [np.pi]}])
    assert result['status'] == 'COLLISION'
    assert any(h['first'] == 'right/hand' and h['second'] == 'new' for h in result['hits'])


def test_mesh_containment_is_not_clear(setup):
    raw,spec,build=setup
    _,c=build({'solid':([.4]*3,[.8,0,0])})
    assert c.check(raw,[])['status']=='COLLISION'
    # Staying inside a pre-existing collision is not an escape.
    assert c.check(raw,[{'right_arm_joint_state':[.01]}])['status']=='COLLISION'


def test_nonadjacent_self_collision(setup):
    raw,spec,build=setup
    urdf=Path(spec['urdf']);text=urdf.read_text()
    text=text.replace('<link name="base"/>', '<link name="base"><collision><origin xyz="0 .8 0"/><geometry><box size=".1 .1 .1"/></geometry></collision></link><link name="pivot"/>')
    text=text.replace('<parent link="base"/><child link="hand"/>','<parent link="pivot"/><child link="hand"/>')
    text=text.replace('</robot>','<joint name="anchor" type="revolute"><parent link="base"/><child link="pivot"/><axis xyz="0 0 1"/><mimic joint="j" multiplier="0"/></joint></robot>')
    urdf.write_text(text)
    _,c=build({'distant':([.1]*3,[10,10,10])})
    assert c.check(raw,[])['status']=='CLEAR'
    r=c.check(raw,[{'right_arm_joint_state':[np.pi]}])
    assert r['status']=='COLLISION'
    assert any({x['first'],x['second']}=={'right/base','right/hand'} for x in r['hits'])


def test_open_and_retreat_leaves_payload_at_release_pose(setup):
    raw,spec,build=setup
    _,c=build({'payload':([.3,.04,.04],[2,0,0]),'obstacle':([.1]*3,[0,2,0])},{'right':'payload'})
    assert c.check(raw,[{'right_arm_joint_state':[np.pi],'right_ee_joint_state':[1.]}])['status']=='CLEAR'
    # Native actions within the same primitive must retain the ORIGINAL release
    # anchor instead of recomputing it at the already-retreated hand.
    c.begin_action()
    before=c.world.state()
    assert c.check(raw,[{'right_arm_joint_state':[np.pi/2],'right_ee_joint_state':[1.]}],phase='native_segment')['status']=='CLEAR'
    raw['right_arm_joint_state']=[np.pi/2];raw['right_ee_joint_state']=[1.]
    assert c.check(raw,[{'right_arm_joint_state':[np.pi],'right_ee_joint_state':[1.]}],phase='native_segment')['status']=='CLEAR'
    assert c.world.state()==before
    c.end_action()


def test_native_rejection_no_motion_no_pending_update(tmp_path):
    from tests.test_robodojo_colour import Scene, Segmenter
    from pyrualean.hosts.robodojo_native import NativeToolkit,NativeBackend
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    from pyrualean._backend import ToolError
    from privilege.pyrualean import AgentWorld
    env=Scene([]);backend=NativeBackend(NativeToolkit(env,Segmenter(env),tmp_path/'episode'))
    robo=RobodojoOfficialColourRobot(backend)
    mesh=tmp_path/'cube.glb';trimesh.creation.box([.01]*3).export(mesh)
    world=from_meshes({'block':{'mesh':mesh}},tmp_path/'world')
    api=AgentWorld(world,robo,tmp_path);robo._action_observer=api
    seen=[]
    def refuse(state,actions,phase):
        seen.append((state,actions,phase))
        return {'status':'COLLISION','hits':[{'first':'right/hand','second':'block'}]}
    backend.set_motion_checker(refuse)
    api.read_context();before=world.state()
    with pytest.raises(ToolError,match='Scene collision'):
        robo.set_gripper(arm='right',gripper=robo.CLOSE)
    assert seen and env.step_count==0
    assert world.state()==before
    assert api._loop.pending is None and api._loop.action_count==0
    assert len(backend.toolkit.state.records)==1
    assert not hasattr(backend.toolkit._primitives,'_grip_commands')
    # No new observation/update is required to execute the next clear action.
    backend.set_motion_checker(lambda *a,**kw: {'status':'CLEAR','hits':[]})
    robo.set_gripper(arm='right',gripper=robo.CLOSE)
    assert env.step_count>0 and api._loop.pending is not None


@pytest.mark.parametrize('tool', ['move_pose','home','grip'])
def test_native_motion_routes_all_call_checker(tmp_path,tool):
    from tests.test_robodojo_colour import Scene,Segmenter
    from pyrualean.hosts.robodojo_native import NativeToolkit,NativeBackend
    env=Scene([])
    if tool=='home':env.joints['right']=[.2,0,0,0,0,0]
    native=NativeToolkit(env,Segmenter(env),tmp_path/'episode');backend=NativeBackend(native)
    if tool=='home':env.joints['right']=[.4,0,0,0,0,0]
    seen=[]
    def check(state,actions,phase):
        seen.append(phase)
        return {'status':'COLLISION','hits':[{'first':'right/hand','second':'obstacle'}]}
    backend.set_motion_checker(check)
    args={'arm':'right'}
    if tool=='move_pose':
        from pyrualean.hosts import robodojo_kinematics as kin
        args['xyz']=kin.tcp(kin.fk('right',[.1,0,0,0,0,0])).tolist()
    result=native.execute_tool(tool,args)
    assert seen and env.step_count==0
    assert result['log']['result']['code']=='scene_collision'


def test_initial_contact_allows_separating_lift_not_deeper_motion(setup):
    raw,spec,build=setup
    urdf=Path(spec['urdf'])
    urdf.write_text(urdf.read_text().replace('name="j" type="revolute"','name="j" type="prismatic"').replace('axis xyz="0 0 1"','axis xyz="0 1 0"'))
    _,c=build({'block':([.06]*3,[.8,.05,0])})
    assert c.check(raw,[])['status']=='COLLISION'
    assert c.check(raw,[{'right_arm_joint_state':[-.1]}])['status']=='CLEAR'
    assert c.check(raw,[{'right_arm_joint_state':[.1]}])['status']=='COLLISION'
    assert c.check(raw,[{'right_arm_joint_state':[-.001]}])['status']=='COLLISION'


def test_place_contact_only_shallow_and_at_endpoint(setup):
    raw,spec,build=setup
    urdf=Path(spec['urdf']);urdf.write_text(urdf.read_text().replace('name="j" type="revolute"','name="j" type="prismatic"'))
    _,c=build({'payload':([.1]*3,[2,0,.2]),'support':([.3,.3,.1],[2,0,0])},{'right':'payload'})
    path=[{'right_arm_joint_state':[-.101]}]
    assert c.check(raw,path)['status']=='COLLISION'
    c.expect_contacts(place={'right':'support'})
    assert c.check(raw,path)['status']=='CLEAR'
    assert c.check(raw,[{'right_arm_joint_state':[-.15]}])['status']=='COLLISION'


def test_fixed_hardware_shared_by_layout_and_payload_sweep(setup):
    from privilege.fixtures import add_fixed_obstacles
    from privilege.context import SceneContext
    raw, spec, build = setup
    world, c = build({'payload': ([.3, .04, .04], [2, 0, 0])}, {'right': 'payload'})
    before = world.state()['objects']['payload']
    t = np.eye(4); t[:3, 3] = [0, 2, 0]
    add_fixed_obstacles(world, [{'object_id': 'stand', 'basis': 'rgb_fixture_estimate',
        'evidence': ['saved_rgb'], 'boxes': [{'size_m': [.1]*3, 'T_object_box': t.tolist()}]}])
    assert world.state()['objects']['payload'] == before
    assert world.verify_binding()['status'] == 'BOUND'
    assert any(o['object_id'] == 'stand' and o['role'] == 'fixed_obstacle'
               for o in SceneContext(world).packet()['objects'])
    state = world.state()
    pose = np.eye(4); pose[:3, 3] = [0, 2, 0]
    assert c.check_layout({'payload': pose})['status'] == 'COLLISION'
    assert c.check(raw, [{'right_arm_joint_state': [np.pi]}])['status'] == 'COLLISION'
    pose[1, 3] = 1.8  # near the fixture, but genuinely clear
    assert c.check_layout({'payload': pose})['status'] == 'CLEAR'
    assert world.state() == state  # queries do not alter poses or consume actions


def test_fixed_hardware_gap_is_not_filled_by_bounding_box(setup):
    from privilege.fixtures import add_fixed_obstacles
    raw, spec, build = setup
    world, c = build({'payload': ([.04]*3, [2, 0, 0])})
    boxes = []
    for x in [-.2, .2]:
        t = np.eye(4); t[:3, 3] = [x, 2, 0]
        boxes.append({'size_m': [.05, .1, .5], 'T_object_box': t.tolist()})
    add_fixed_obstacles(world, [{'object_id': 'stand', 'basis': 'rgb_fixture_estimate',
                                'evidence': ['saved_rgb'], 'boxes': boxes}])
    pose = np.eye(4); pose[:3, 3] = [0, 2, 0]
    assert c.check_layout({'payload': pose})['status'] == 'CLEAR'
    pose[0, 3] = .2
    assert c.check_layout({'payload': pose})['status'] == 'COLLISION'


def test_partial_motion_refusal_keeps_actual_update(tmp_path):
    from tests.test_robodojo_colour import Scene,Segmenter
    from pyrualean.hosts.robodojo_native import NativeToolkit,NativeBackend
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    from pyrualean._backend import ToolError
    from privilege.pyrualean import AgentWorld
    env=Scene([]);backend=NativeBackend(NativeToolkit(env,Segmenter(env),tmp_path/'episode'))
    robo=RobodojoOfficialColourRobot(backend)
    mesh=tmp_path/'cube.glb';trimesh.creation.box([.01]*3).export(mesh)
    world=from_meshes({'block':{'mesh':mesh}},tmp_path/'world')
    api=AgentWorld(world,robo,tmp_path);robo._action_observer=api
    count=0
    def check(*args,**kwargs):
        nonlocal count
        count+=1
        return {'status':'CLEAR' if count==1 else 'COLLISION','hits':[{'first':'right/f1','second':'block'}]}
    backend.set_motion_checker(check)
    api.read_context();rev=world.state()['revision']
    with pytest.raises(ToolError):robo.set_gripper(arm='right',gripper=robo.CLOSE)
    assert env.step_count==1 and world.state()['revision']==rev+1
    assert api._loop.pending is not None and api._loop.action_count==1


def test_release_in_later_path_segment_uses_that_segments_start(setup):
    raw, spec, build = setup
    _, c = build({'payload': ([.3, .04, .04], [2, 0, 0]),
                  'distant': ([.1]*3, [10, 10, 10])}, {'right': 'payload'})
    c.begin_action()
    result = c.check(raw, [
        {'right_arm_joint_state': [np.pi/2]},
        {'right_arm_joint_state': [np.pi], 'right_ee_joint_state': [1.]},
    ], phase='native_segment')
    assert result['status'] == 'CLEAR'
    assert np.allclose(c._released_poses['payload'][:3, 3], [0, 2, 0])


def test_published_and_moved_part_enters_live_collision_without_reinstall(setup, tmp_path):
    raw, spec, build = setup
    world, c = build({'obstacle': ([.1]*3, [3, 0, 0])})
    path = [{'right_arm_joint_state': [np.pi]}]
    assert c.check(raw, path)['status'] == 'CLEAR'
    mesh = tmp_path/'part.glb'; trimesh.creation.box([.1]*3).export(mesh)
    t = np.eye(4); t[:3, 3] = [-3, .8, 0]
    world.publish_parts('obstacle', {'body': {'mesh': mesh, 'T_object_part': t}})
    assert c.check(raw, path)['status'] == 'COLLISION'
    t[:3, 3] = [0, 0, 0]
    world.update_parts('obstacle', {'body': t})
    assert c.check(raw, path)['status'] == 'CLEAR'


def test_release_contact_survives_detach_only_for_existing_finger_opening(setup):
    raw, spec, build = setup
    raw['right_ee_joint_state'] = [0.]
    world, c = build({'payload': ([.08, .08, .06], [1, .06, 0])})
    s = world.state(); s['tracking'] = {'robot_state': {'step': 7}}
    s['releases'] = {'payload': {'arm': 'right', 'host_step': 7}}
    atomic(world.state_path, s)
    # Fingers still touch the released body while opening a little in place.
    path = [{'right_ee_joint_state': [.1]}]
    assert c.check(raw, path)['status'] == 'CLEAR'
    # Permission expires after another primitive; it is not a permanent ignore.
    s['tracking']['robot_state']['step'] = 8; atomic(world.state_path, s)
    assert c.check(raw, path)['status'] == 'COLLISION'
    s['tracking']['robot_state']['step'] = 7; atomic(world.state_path, s)
    assert c.check(raw, [{'right_arm_joint_state': [.01], 'right_ee_joint_state': [.1]}])['status'] == 'COLLISION'


def test_release_permission_does_not_allow_new_finger_object_collision(setup):
    raw, spec, build = setup
    raw['right_ee_joint_state'] = [0.]
    world, c = build({'payload': ([.02]*3, [1, .10, 0])})
    s = world.state(); s['tracking'] = {'robot_state': {'step': 7}}
    s['releases'] = {'payload': {'arm': 'right', 'host_step': 7}}; atomic(world.state_path, s)
    assert c.check(raw, [{'right_ee_joint_state': [1.]}])['status'] == 'COLLISION'
