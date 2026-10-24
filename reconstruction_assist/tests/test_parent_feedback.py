import json,tempfile,unittest
from pathlib import Path
from reconstruction_assist.feedback_hook import inherited_residuals
class ParentBudgetTest(unittest.TestCase):
 def test_unchanged_child_residual_does_not_restart_its_budget(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);nd=r/'fractal/scene';child=r/'fractal/child';feedback=child/'feedback/one';nd.mkdir(parents=True);feedback.mkdir(parents=True)
   (nd/'part.json').write_text(json.dumps({'children':['child']}));(child/'selection.json').write_text(json.dumps({'budget_exhausted':True,'selected_feedback':'fractal/child/feedback/one'}));(feedback/'feedback.json').write_text(json.dumps({'objects':[{'object_id':'object','score':1.}]}))
   result={'status':'FIT_REVIEW_REQUIRED','objects':[{'object_id':'object','score':1.,'needs_revision':True}],'missing_object_ids':[]}
   self.assertEqual(inherited_residuals(result,nd,r),['object'])
   result['objects'][0]['score']=2.
   self.assertEqual(inherited_residuals(result,nd,r),[])
   result['objects'][0]['score']=1.;result['missing_object_ids']=['missing']
   self.assertEqual(inherited_residuals(result,nd,r),[])
if __name__=='__main__':unittest.main()
