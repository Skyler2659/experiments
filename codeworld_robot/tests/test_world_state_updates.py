"""Ordinary helpers and incremental field updates preserve recorded world evidence."""
import copy
import unittest
from codeworld_robot.io import atomic,read
from codeworld_robot.tests import test_world_flow
from codeworld_workspace.completion import digest,field_supported

class WorldStateUpdateTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_world_flow.WorldFlowTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.world=self.fixture.world

    def compute(self,source,objects=None):
        return self.world.compute_decision(self.fixture.program(source),purpose='judgment',object_ids=objects or ['box'])

    def read_field(self,field):
        return self.compute("def read(w):\n return w.state('box',"+repr(field)+")\ndef compute(ctx):\n return {'judgments':{'value':read(ctx)}}")['result']['judgments']['value']

    def seed(self):
        r=self.compute("def height(w):\n return float(w.position('box')[2])\ndef compute(ctx):\n h=height(ctx)\n return {'judgments':{'height':h,'other':h+2.0}}")
        self.world.derive_state(r['decision'],'box',{'height':'height','other':'other'})

    def test_helper_can_only_use_recorded_context_methods(self):
        for attr in ('world','snapshot','ctx','queries','object_ids'):
            source="def helper(w):\n return w."+attr+"\ndef compute(ctx):\n p=ctx.position('box')\n return {'judgments':{'value':helper(ctx)}}"
            with self.subTest(attr=attr), self.assertRaises(AttributeError):self.compute(source)
        self.seed();self.assertAlmostEqual(self.read_field('height'),0.)

    def test_same_field_filter_preserves_prior_value_and_bounded_physical_dependencies(self):
        self.seed()
        for i in range(12):
            r=self.compute("def compute(ctx):\n old=ctx.state('box','height')\n p=ctx.position('box')\n return {'judgments':{'height':(old+float(p[2]))/2}}")
            revision=self.world.derive_state(r['decision'],'box',{'height':'height'})
            self.assertIsInstance(revision,int)
            self.assertAlmostEqual(self.read_field('height'),0.)
            d=self.world.state()['objects']['box']['observed_state']['fields']['height']['derivation']
            self.assertLessEqual(len(d['queries']),2)
            self.assertEqual(d['state_dependencies'],[])
            self.assertEqual(d['prior_state_inputs'][0]['value'],0.)
            self.assertEqual(len(d['prior_state_inputs']),1)
            self.assertTrue(all(q['op']!='state' for q in d['queries']))

    def test_same_batch_mutual_updates_use_pre_update_versions(self):
        self.seed()
        r=self.compute("def compute(ctx):\n a=ctx.state('box','height')\n b=ctx.state('box','other')\n return {'judgments':{'height':b+.5,'other':a+.5}}")
        self.world.derive_state(r['decision'],'box',{'height':'height','other':'other'})
        self.assertAlmostEqual(self.read_field('height'),2.5)
        self.assertAlmostEqual(self.read_field('other'),.5)
        for field in ('height','other'):
            d=self.world.state()['objects']['box']['observed_state']['fields'][field]['derivation']
            self.assertEqual(d['state_dependencies'],[])
            self.assertTrue(all(q['op']!='state' for q in d['queries']))

    def test_sequential_cross_field_updates_do_not_create_a_cycle(self):
        self.seed()
        r=self.compute("def compute(ctx):\n a=ctx.state('box','height')\n return {'judgments':{'other':a+1}}")
        self.world.derive_state(r['decision'],'box',{'other':'other'})
        r=self.compute("def compute(ctx):\n b=ctx.state('box','other')\n return {'judgments':{'height':b+1}}")
        self.world.derive_state(r['decision'],'box',{'height':'height'})
        self.assertAlmostEqual(self.read_field('height'),2.)
        d=self.world.state()['objects']['box']['observed_state']['fields']['height']['derivation']
        self.assertEqual(d['state_dependencies'],[])
        self.assertTrue(all(q['op']!='state' for q in d['queries']))
        self.assertEqual({i['field'] for i in d['prior_state_inputs']},{'height','other'})

    def test_recurrent_field_retains_other_object_geometry_dependency(self):
        s=self.world.state();s['objects']['reference']=copy.deepcopy(s['objects']['box'])
        atomic(self.world.state_path,s)
        manifest=read(self.root/'world.json');manifest['objects']['reference']='objects/box.json'
        atomic(self.root/'world.json',manifest);self.world.manifest=manifest
        r=self.compute("def compute(ctx):\n p=ctx.position('box')\n q=ctx.position('reference')\n return {'judgments':{'height':float(p[2]-q[2])}}",objects=['box','reference'])
        self.world.derive_state(r['decision'],'box',{'height':'height'})
        r=self.compute("def compute(ctx):\n old=ctx.state('box','height')\n return {'judgments':{'height':old*.5}}")
        self.world.derive_state(r['decision'],'box',{'height':'height'})
        self.assertAlmostEqual(self.read_field('height'),0.)
        d=self.world.state()['objects']['box']['observed_state']['fields']['height']['derivation']
        self.assertIn('reference',d['dependencies'])
        T=self.world.state()['objects']['reference']['T_W_object'];T[2][3]=.2
        self.world.update({'reference':T},basis='action_prediction')
        with self.assertRaisesRegex(ValueError,'unsupported'):self.read_field('height')

    def test_old_measurement_derivation_is_pending_without_changing_return_type(self):
        self.world.sync_robot_feedback();s=self.world.state()
        s['measurements']={'lift':{'object_id':'box','robot_state_ref':{'robot_state_ref':'old'},'value':{'status':'LIFT_EVIDENCE'}}}
        atomic(self.world.state_path,s)
        r=self.compute("def compute(ctx):\n m=ctx.measurement('lift')\n return {'judgments':{'holding':m['status']=='LIFT_EVIDENCE'}}")
        revision=self.world.derive_state(r['decision'],'box',{'holding':'holding'})
        self.assertIsInstance(revision,int)
        entry=self.world.state()['objects']['box']['observed_state']['fields']['holding']
        self.assertEqual(entry['status'],'pending')
        with self.assertRaisesRegex(ValueError,'pending'):self.read_field('holding')

    def test_recurrent_field_retains_measurement_freshness(self):
        self.world.sync_robot_feedback();s=self.world.state()
        s['measurements']={'lift':{'object_id':'box','robot_state_ref':{'robot_state_ref':'fixture:0','timestamp_ns':0},'value':{'status':'LIFT_EVIDENCE'}}}
        atomic(self.world.state_path,s)
        r=self.compute("def compute(ctx):\n m=ctx.measurement('lift')\n return {'judgments':{'holding':m['status']=='LIFT_EVIDENCE'}}")
        self.world.derive_state(r['decision'],'box',{'holding':'holding'})
        r=self.compute("def compute(ctx):\n h=ctx.state('box','holding')\n return {'judgments':{'holding':bool(h)}}")
        self.world.derive_state(r['decision'],'box',{'holding':'holding'})
        self.assertTrue(self.read_field('holding'))
        d=self.world.state()['objects']['box']['observed_state']['fields']['holding']['derivation']
        self.assertIn({'op':'measurement','id':'lift','object':'box'},d['queries'])
        atomic(self.root/'robot_io/robot_state.json',{**self.fixture.robot,'robot_state_ref':'new','timestamp_ns':1})
        with self.assertRaisesRegex(ValueError,'unsupported'):self.read_field('holding')

if __name__=='__main__':unittest.main()
