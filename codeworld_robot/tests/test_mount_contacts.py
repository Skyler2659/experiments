import json
import numpy as np
import pytest
import trimesh
from codeworld_robot.collision import RobotCollision, RobotOnlyScene, robodojo_collision


class Scene:
    def __init__(self, root, name, center):
        self.root = root
        mesh = trimesh.creation.box([.12, .08, .08]); mesh.apply_translation(center)
        mesh.export(root / 'scene.glb')
        (root / 'object.json').write_text(json.dumps({'role': 'fixture'}))
        self.manifest = {'objects': {name: 'object.json'}}
    def objects(self):return list(self.manifest['objects'])
    def state(self):return {'revision': 0, 'objects': {oid: {} for oid in self.objects()}}
    def _mesh(self, *args):return trimesh.load(self.root / 'scene.glb', force='mesh')


def fixture(tmp_path):
    p = tmp_path / 'robot.urdf'
    p.write_text('''<robot name="mount-test">
    <link name="base_link"><collision><geometry><box size=".1 .1 .1"/></geometry></collision></link>
    <link name="link1"><collision><origin xyz=".7 0 0"/><geometry><box size=".1 .1 .1"/></geometry></collision></link>
    <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/><axis xyz="0 0 1"/></joint>
    </robot>''')
    return p


def check(c):
    q = {0: np.zeros(1)}
    return c.check_path(q, [q], {0: [0, 0]}, allowed_contacts=[])


def test_fixed_mount_is_excluded_without_policy_exception(tmp_path):
    p=fixture(tmp_path); scene=Scene(tmp_path, 'tabletop', [0, 0, 0]); bases={0: np.eye(4)}
    assert check(RobotCollision(p, bases, scene))['status']=='COLLISION'
    c=robodojo_collision(p, bases, scene)
    assert check(c)['status']=='CLEAR'
    assert c.mounting_contacts == (('arm0/base_link', 'tabletop'),)
    # Both arms are bound automatically, with no task-name dependency.
    two=robodojo_collision(p, {0:np.eye(4), 1:np.eye(4)}, scene)
    assert len(two.mounting_contacts)==2
    assert check(robodojo_collision(p,bases,RobotOnlyScene()))['status']=='CLEAR'


@pytest.mark.parametrize('name,center,body', [('tabletop',[.7,0,0],'arm0/link1'), ('obstacle',[0,0,0],'arm0/base_link')])
def test_moving_link_and_non_mount_collisions_remain(tmp_path,name,center,body):
    c=robodojo_collision(fixture(tmp_path),{0:np.eye(4)},Scene(tmp_path,name,center))
    result=check(c)
    assert result['status']=='COLLISION'
    assert any(hit['first']==body and hit['second']==name for hit in result['hits'])


def test_moving_links_cannot_be_declared_fixed_mounts(tmp_path):
    p=fixture(tmp_path);scene=Scene(tmp_path,'tabletop',[.7,0,0])
    with pytest.raises(ValueError):
        RobotCollision(p,{0:np.eye(4)},scene,mounting_contacts=[('arm0/link1','tabletop')])
