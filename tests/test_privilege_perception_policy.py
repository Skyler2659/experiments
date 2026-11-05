"""CodeWorld runs expose RGB only; original PyRUA keeps its perception surface."""

import pytest

from tests.test_privilege_multiview import scene
from privilege.pyrualean import AgentWorld
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from pyrualean.play import build_prompt
from pyrualean.prompt import TaskCard, api_reference, knowledge_text
from privilege.policy import codeworld_knowledge


class Backend:
    def __init__(self):
        self.calls = []

    def execute_tool(self, name, kwargs):
        self.calls.append(name)
        raise RuntimeError('backend reached')


@pytest.mark.parametrize('tool,args', [('locate', ('board',)), ('back_project', (10, 10)),
                                       ('segment', ('board',))])
def test_blocked_before_backend_including_unbound_class_method(scene, tool, args):
    world, _, _ = scene
    backend = Backend()
    robot = RobodojoOfficialColourRobot(backend, primitives='no-vla')
    api = AgentWorld(world, robot, world.root)
    saved_method = getattr(robot, tool)
    api._configure_perception('codeworld')
    for invoke in (saved_method, lambda *a: getattr(type(robot), tool)(robot, *a)):
        with pytest.raises(RuntimeError, match='disabled in this CodeWorld run'):
            invoke(*args)
    assert backend.calls == []
    baseline = RobodojoOfficialColourRobot(backend, primitives='no-vla')
    with pytest.raises(RuntimeError, match='backend reached'):
        getattr(baseline, tool)(*args)
    assert backend.calls == [tool]


def test_location_import_and_branch_cannot_bypass_policy(scene):
    _, _, api = scene
    api._configure_perception('codeworld')
    for target in (api, api.fork('scenario')):
        with pytest.raises(RuntimeError, match='disabled'):
            target.integrate_location('unused', 'block', {}, expected_revision=0, note='unused')
        with pytest.raises(RuntimeError, match='disabled'):
            target.reconcile_location('block', {})


@pytest.mark.parametrize('primitives', ['full', 'no-vla'])
def test_prompt_surface_and_motor_knowledge_match_policy(primitives):
    card = TaskCard(suite='test', task=0, seed=0, task_language='build', object_names=[],
                    robot='robodojo_xpl_rgb')
    baseline = build_prompt('cells', card, budget_s=100, max_programs=10, primitives=primitives)
    ours = build_prompt('cells', card, budget_s=100, max_programs=10, primitives=primitives,
                        excluded_members=frozenset({'locate', 'back_project', 'segment'}))
    assert 'robo.locate(' in baseline
    for name in ('locate', 'back_project', 'segment'):
        assert f'robo.{name}(' not in ours
    assert 'SAM3 segmentation service' not in ours
    assert 'class ObjectSight(' not in ours and 'class SeenPoint(' not in ours
    assert 'robo.move_to(' in ours and 'robo.set_gripper(' in ours and 'robo.home(' in ours
    original = knowledge_text('robodojo_xpl_rgb', primitives)
    modified = codeworld_knowledge(original)
    # Exact preservation of the complete native motion section, not just API names.
    motion = original.split('## Motion\n', 1)[1].split('\n## ', 1)[0]
    assert motion in modified
    assert '`locate`' not in modified
