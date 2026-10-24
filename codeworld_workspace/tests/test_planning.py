import unittest
import numpy as np
from codeworld_workspace.tests import test_world as fixtures
from codeworld_robot.client import Robot

class PlanningTests(unittest.TestCase):
    setUp=fixtures.WorldTests.setUp
    tearDown=fixtures.WorldTests.tearDown

    def test_geometry_changes_selection_not_just_posthoc_rejection(self):
        w=self.w;routes=[{'name':'direct','points_W_m':[[-.4,0,0],[.4,0,0]]},
                        {'name':'above','points_W_m':[[-.4,0,0],[-.4,0,.4],[.4,0,.4],[.4,0,0]]}]
        kwargs={'tool_half_extents_W':[.01,.01,.01],'exclude':{'table':'fixture duplicate'}}
        ctx=w.planning_context();decision=ctx.compare_routes(routes,**kwargs)
        self.assertEqual(decision['selected'],'above')
        t=np.eye(4);t[1,3]=.8;w.update({'box':t},basis='observation_estimate',evidence=['capture'])
        updated=w.planning_context().compare_routes(routes,**kwargs)
        self.assertEqual(updated['selected'],'direct')
        with self.assertRaisesRegex(ValueError,'Stale'):ctx.save(decision,self.root/'old.json')

    def test_current_transform_changes_anchor_and_preserves_snapshot(self):
        w=self.w;ctx=w.planning_context();initial=ctx.point('box',[.5,.5,1])
        t=np.eye(4);t[0,3]=.3;w.update({'box':t})
        np.testing.assert_allclose(ctx.point('box',[.5,.5,1]),initial)
        np.testing.assert_allclose(w.planning_context().point('box',[.5,.5,1]),initial+[.3,0,0])
        np.testing.assert_allclose(w.planning_context().local_point('box',[0,0,0]),[.3,0,0])

    def test_execution_uses_selected_waypoint_and_rejects_stale(self):
        w=self.w;ctx=w.planning_context()
        decision=ctx.compare_routes([{'name':'above','points_W_m':[[0,0,.5],[.4,0,.5]]}],tool_half_extents_W=[.01]*3)
        path=ctx.save(decision,self.root/'route.json');robot=Robot.__new__(Robot);robot.root=self.root
        robot.call=lambda op,**args: {'op':op,**args}
        result=robot.move_from_world(path,1,[1,0,0,0])
        self.assertEqual(result['pose_W']['translation_m'],[.4,0,.5]);self.assertEqual(result['decision_ref']['selected'],'above')
        w.update({'box':np.eye(4)})
        with self.assertRaisesRegex(ValueError,'Stale'):robot.move_from_world(path,1,[1,0,0,0])

if __name__=='__main__':unittest.main()
