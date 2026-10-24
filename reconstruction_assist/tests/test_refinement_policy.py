import json
from pathlib import Path
import tempfile
import unittest
from reconstruction_assist.refinement_policy import subpart_scope,decomposition_required,should_retry

class DeepRefinementTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name);self.node=self.run/'fractal/cabinet';self.node.mkdir(parents=True)
        self.config={'max_reviews':5,'refinement':{'enabled':True,'structural_objects':['cabinet'],'minimum_parts':2}}
        self.write(self.node/'node_context.json',{'parent_id':'scene','depth':1,'max_depth':4})
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def test_primitive_delivery_cannot_bypass_structural_descent(self):
        self.write(self.node/'part.json',{'children':[]})
        self.assertTrue(decomposition_required(self.node,self.config))
        self.write(self.node/'part.json',{'children':['cabinet_shell','cabinet_fronts']})
        self.assertFalse(decomposition_required(self.node,self.config))
    def test_inherited_error_retries_but_does_not_reset_review_budget(self):
        failure={'status':'FIT_REVIEW_REQUIRED'}
        self.assertTrue(should_retry(failure,{'reviews':1},self.config,True))
        self.assertFalse(should_retry(failure,{'reviews':5},self.config,True))
        self.assertFalse(should_retry(failure,{'reviews':1},{'max_reviews':3},True))
    def test_subpart_must_belong_to_its_actual_parent_owner(self):
        child=self.run/'fractal/cabinet_fronts'
        self.write(child/'node_context.json',{'parent_id':'cabinet','depth':2,'max_depth':4})
        self.write(child/'node_scope.json',{'kind':'subpart','owner_object_id':'cabinet','parent_node':'cabinet'})
        self.assertEqual(subpart_scope(child,self.run)['owner_object_id'],'cabinet')
        self.write(child/'node_scope.json',{'kind':'subpart','owner_object_id':'other_object','parent_node':'scene'})
        with self.assertRaises(ValueError):subpart_scope(child,self.run)
    def test_scene_child_cannot_defer_its_whole_object_review(self):
        self.write(self.node/'node_scope.json',{'kind':'subpart','owner_object_id':'cabinet','parent_node':'scene'})
        with self.assertRaises(ValueError):subpart_scope(self.node,self.run)
    def test_depth_limit_does_not_demand_impossible_new_children(self):
        self.write(self.node/'part.json',{'children':[]})
        self.write(self.node/'node_context.json',{'parent_id':'parent','depth':4,'max_depth':4})
        self.assertFalse(decomposition_required(self.node,self.config))

class AssemblyRevisionTest(unittest.TestCase):
    def test_child_change_invalidates_parent_best_without_granting_new_reviews(self):
        from reconstruction_assist.refinement_policy import update_child_revision
        state={'reviews':3,'best_score':.1,'best_feedback':'old','best_status':'FIT_CHECKS_PASSED','child_source_revision':'a'}
        self.assertTrue(update_child_revision(state,'b'))
        self.assertEqual(state['reviews'],3)
        self.assertEqual(state['best_score'],float('inf'))
        self.assertNotIn('best_feedback',state)
    def test_same_children_preserve_best_candidate(self):
        from reconstruction_assist.refinement_policy import update_child_revision
        state={'reviews':2,'best_score':.1,'child_source_revision':'a'}
        self.assertFalse(update_child_revision(state,'a'))
        self.assertEqual(state['best_score'],.1)
    def test_missing_scope_for_deep_node_is_model_feedback_error(self):
        from reconstruction_assist.refinement_policy import subpart_scope
        with tempfile.TemporaryDirectory() as d:
            node=Path(d)/'fractal/part';node.mkdir(parents=True)
            (node/'node_context.json').write_text(json.dumps({'depth':2,'parent_id':'owner'}))
            with self.assertRaisesRegex(ValueError,'missing its own node_scope'):
                subpart_scope(node,Path(d))


class ChildReferenceTest(unittest.TestCase):
    def test_mixed_flat_and_path_references_resolve_and_track_child_changes(self):
        from reconstruction_assist.refinement_policy import resolve_children,child_source_revision
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);node=run/'fractal/rack';a=run/'fractal/base';b=node/'upper'
            for p in [node,a,b]:p.mkdir(parents=True,exist_ok=True)
            for p in [a,b]:(p/'part.json').write_text('{"children":[]}');(p/'component.js').write_text('old')
            (node/'part.json').write_text(json.dumps({'children':['base',{'id':'upper','part':'./upper/part.json'}]}))
            self.assertEqual([name for name,_ in resolve_children(node,run)],['base','upper'])
            sources=lambda p:[f for f in p.iterdir() if f.suffix in ('.js','.json')]
            old=child_source_revision(node,run,sources);(b/'component.js').write_text('new')
            self.assertNotEqual(old,child_source_revision(node,run,sources))
    def test_missing_child_is_an_actionable_reference_error(self):
        from reconstruction_assist.refinement_policy import resolve_children
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);node=run/'fractal/rack';node.mkdir(parents=True)
            (node/'part.json').write_text('{"children":[{"id":"missing","part":"./missing/part.json"}]}')
            with self.assertRaisesRegex(ValueError,'Missing child delivery'):resolve_children(node,run)

if __name__=='__main__':unittest.main()
