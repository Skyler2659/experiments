import numpy as np
from codeworld_workspace.tests.test_assembly import AssemblyTests
from codeworld_workspace.flow import DecisionContext,query_signature
class PartDecisionTests(AssemblyTests):
 def test_part_queries_use_recorded_snapshot(self):
  w=self.w;t=np.eye(4);t[0,3]=.4
  w.publish_parts('box',{'body':{'mesh':self.root/'geometry/box.glb'},'door':{'mesh':self.root/'geometry/box.glb','T_object_part':t.tolist()}})
  snapshot=w.state();ctx=DecisionContext(w,snapshot)
  np.testing.assert_allclose(ctx.mesh('box',part_id='door').centroid,[.4,0,0],atol=1e-7)
  np.testing.assert_allclose(ctx.transform('box',part_id='door'),t)
  np.testing.assert_allclose(ctx.bounds('box',part_id='door'),w.part_mesh('box','door').bounds)
  sig=query_signature(w,snapshot,ctx._queries);t[1,3]=.2;w.update_parts('box',{'door':t})
  self.assertNotEqual(sig,query_signature(w,w.state(),ctx._queries))
  np.testing.assert_allclose(ctx.mesh('box',part_id='door').centroid,[.4,0,0],atol=1e-7)
