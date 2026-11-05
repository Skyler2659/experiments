"""Task-independent interaction semantics; real mesh checker, no robot episodes."""
import copy
import numpy as np
import pytest
import trimesh
from tests.test_privilege_collision import setup


def parts(world, oid, tmp_path, extra=False):
    path = tmp_path / (oid + '_part.glb')
    trimesh.creation.box([.1]*3).export(path)
    data = {'interface': {'mesh': path}}
    if extra:
        t = np.eye(4); t[0, 3] = .2
        data['other'] = {'mesh': path, 'T_object_part': t}
    world.publish_parts(oid, data)


def contact():
    return {'first': {'object': 'payload', 'part': 'interface'},
            'second': {'object': 'fixture', 'part': 'interface'},
            'reason': 'Agent-selected interaction surface'}


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_contact_has_no_task_or_vertical_axis_assumption(setup, tmp_path, axis):
    raw, spec, build = setup
    p = tmp_path/'robot.urdf'
    axis_s = ['0', '0', '0']; axis_s[axis] = '1'
    p.write_text(p.read_text().replace('name="j" type="revolute"', 'name="j" type="prismatic"')
                 .replace('axis xyz="0 0 1"', 'axis xyz="'+' '.join(axis_s)+'"'))
    pos = np.array([2., 0, 0]); obstacle = pos.copy(); obstacle[axis] += .14
    world, checker = build({'payload': ([.1]*3, pos), 'fixture': ([.1]*3, obstacle)}, {'right':'payload'})
    parts(world, 'payload', tmp_path); parts(world, 'fixture', tmp_path)
    path = [{'right_arm_joint_state': [.06]}]
    assert checker.check(raw, path)['status'] == 'COLLISION'
    checker.expect_contacts(contacts=[contact()])
    result = checker.check(raw, path)
    assert result['status'] == 'CLEAR' and result['expected_contacts']
    # Native segments receive the same rule, which then expires even on refusal.
    checker.begin_action()
    assert checker.check(raw, path, phase='native_segment')['status'] == 'CLEAR'
    checker.end_action()
    assert checker.check(raw, path)['status'] == 'COLLISION'


def test_permission_does_not_hide_other_part_or_arm(setup, tmp_path):
    raw, spec, build = setup
    p=tmp_path/'robot.urdf'
    p.write_text(p.read_text().replace('name="j" type="revolute"', 'name="j" type="prismatic"')
                 .replace('axis xyz="0 0 1"', 'axis xyz="1 0 0"'))
    world,c=build({'payload':([.1]*3,[2,0,0]),'fixture':([.1]*3,[2.14,0,0])},{'right':'payload'})
    parts(world,'payload',tmp_path);parts(world,'fixture',tmp_path,extra=True)
    c.expect_contacts(contacts=[contact()])
    assert c.check(raw,[{'right_arm_joint_state':[.06]}])['status']=='CLEAR'
    result=c.check(raw,[{'right_arm_joint_state':[.3]}])
    assert result['status']=='COLLISION'
    assert any(h.get('second_part')=='other' for h in result['hits'])
    t=np.eye(4);t[:3,3]=[.8,0,0];world.update({'fixture':t})
    result=c.check(raw,[])
    assert result['status']=='COLLISION'
    assert any(h['first']=='right/hand' for h in result['hits'])


def test_local_region_clips_large_triangles_and_follows_pose(setup):
    raw,spec,build=setup
    # Fingertip intersects a small patch of a large face; whole-triangle tests
    # would incorrectly refuse this valid local declaration.
    world,c=build({'surface':([.02,.12,.4],[1,.10,0])})
    rule={'first':{'arm':'right','link':'f1'},
          'second':{'object':'surface','bounds':[[-.02,-.04,-.04],[.02,.04,.04]]}}
    assert c.check(raw,[])['status']=='COLLISION'
    c.expect_contacts(contacts=[rule])
    assert c.check(raw,[])['status']=='CLEAR'
    # Narrower patch does not cover the same actual intersection.
    bad=copy.deepcopy(rule);bad['second']['bounds'][1][2]=.001
    c.expect_contacts(contacts=[bad])
    assert c.check(raw,[])['status']=='COLLISION'
    c.expect_contacts(contacts=[rule])
    # Object local region follows a translated object without being frozen in W.
    t=np.eye(4);t[:3,3]=[1,.15,0];world.update({'surface':t})
    raw['right_ee_joint_state']=[1.]
    assert c.check(raw,[])['status']=='CLEAR'


def test_geometry_replacement_invalidates_intent_not_pose_update(setup,tmp_path):
    raw,spec,build=setup
    world,c=build({'payload':([.1]*3,[2,0,0]),'fixture':([.1]*3,[2.08,0,0])},{'right':'payload'})
    parts(world,'payload',tmp_path);parts(world,'fixture',tmp_path)
    c.expect_contacts(contacts=[contact()])
    assert c.check(raw,[])['status']=='CLEAR'
    path=tmp_path/'replacement.glb';trimesh.creation.box([.12]*3).export(path)
    world.publish_parts('fixture',{'interface':{'mesh':path}})
    result=c.check(raw,[])
    assert result['status']=='COLLISION' and len(result['invalidated_contact_intents'])==1


def test_whole_solid_containment_is_not_exempt(setup,tmp_path):
    raw,spec,build=setup
    world,c=build({'payload':([.1]*3,[2,0,0]),'fixture':([.5]*3,[2,0,0])},{'right':'payload'})
    c.expect_contacts(contacts=[{'first':{'object':'payload','bounds':[[-1]*3,[1]*3]},
                                'second':{'object':'fixture','bounds':[[-1]*3,[1]*3]}}])
    assert c.check(raw,[])['status']=='COLLISION'


def test_invalid_contact_does_not_replace_existing_intent(setup):
    raw,spec,build=setup
    world,c=build({'block':([.1]*3,[2,0,0])})
    c.expect_contacts(grasp={'right':'block'})
    before=copy.deepcopy(c.contacts)
    with pytest.raises(ValueError):
        c.expect_contacts(contacts=[{'first':{'object':'block'},'second':{'arm':'right','link':'hand'}}])
    assert c.contacts==before


def test_unpartitioned_contact_error_and_local_bounds_alias(setup):
    raw, spec, build = setup
    _, c = build({'block': ([.02,.12,.4], [1,.10,0])})
    c.expect_contacts(grasp={'right': 'block'})
    before = copy.deepcopy(c.contacts)
    with pytest.raises(ValueError, match='available parts.*bounds'):
        c.expect_contacts(contacts=[{'first': {'object':'block','part':'block'},
                                    'second': {'arm':'right','link':'f1'}}])
    assert c.contacts == before
    rule = {'first': {'object':'block','local_bounds':[[-.02,-.04,-.04],[.02,.04,.04]]},
            'second': {'arm':'right','link':'f1'}}
    c.expect_contacts(contacts=[rule])
    assert c.check(raw, [])['status'] == 'CLEAR'
    assert 'local_bounds' in rule['first']  # No caller dictionary mutation.
    assert 'bounds' in c.contacts['contacts'][0]['first']
    rule['first']['bounds'] = rule['first']['local_bounds']
    with pytest.raises(ValueError, match='not both'):
        c.expect_contacts(contacts=[rule])


def test_multiple_contact_patches_union_without_whole_object_exemption(setup):
    raw,spec,build=setup
    world,c=build({'surface':([.02,.12,.4],[1,.10,0])})
    def patch(lo,hi):
        return {'first':{'arm':'right','link':'f1'},
                'second':{'object':'surface','bounds':[[lo,-.04,-.04],[hi,.04,.04]]}}
    left,right=patch(-.02,-.005),patch(.005,.02)
    c.expect_contacts(contacts=[left])
    assert c.check(raw,[])['status']=='COLLISION'
    c.expect_contacts(contacts=[right])
    assert c.check(raw,[])['status']=='COLLISION'
    c.expect_contacts(contacts=[left,right])
    assert c.check(raw,[])['status']=='CLEAR'


def test_part_local_contact_region_moves_with_articulation(setup,tmp_path):
    raw,spec,build=setup
    world,c=build({'surface':([.02,.12,.4],[1,.10,0])})
    mesh=tmp_path/'panel.glb';trimesh.creation.box([.02,.12,.4]).export(mesh)
    world.publish_parts('surface',{'panel':{'mesh':mesh}})
    c.expect_contacts(contacts=[{'first':{'arm':'right','link':'f1'},
        'second':{'object':'surface','part':'panel','bounds':[[-.02,-.04,-.04],[.02,.04,.04]]}}])
    assert c.check(raw,[])['status']=='CLEAR'
    t=np.eye(4);t[1,3]=.05
    world.update_parts('surface',{'panel':t})
    raw['right_ee_joint_state']=[1.]
    assert c.check(raw,[])['status']=='CLEAR'
