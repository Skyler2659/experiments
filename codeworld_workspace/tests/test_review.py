import unittest
import numpy as np
from codeworld_workspace.tests import test_world as fixtures

class ReviewTests(unittest.TestCase):
    setUp=fixtures.WorldTests.setUp
    tearDown=fixtures.WorldTests.tearDown

    def test_bystander_pending_survives_target_and_prediction(self):
        w=self.w
        w.assess({'box':{'pose':'pending','geometry:interior':'unreviewed'},
                  'table':{'pose':'pending'}},evidence=['capture1'],note='possible disturbance')
        w.update({'box':np.eye(4)},basis='action_prediction')
        self.assertEqual(w.review_status()['objects']['box']['aspects']['pose']['status'],'pending')
        w.update({'box':np.eye(4)},basis='observation_estimate',evidence=['capture2'])
        statuses=w.review_status()['objects']
        self.assertEqual(statuses['box']['aspects']['pose']['status'],'observed')
        self.assertEqual(statuses['box']['aspects']['geometry:interior']['status'],'unreviewed')
        self.assertEqual(statuses['table']['aspects']['pose']['status'],'pending')
        self.assertEqual(w.bounds('table')['review']['pose']['status'],'pending')
        self.assertTrue(all(x['recorded_at_ns']>0 for x in w.state()['history']))

    def test_occlusion_confirmation_atomicity_and_branch(self):
        w=self.w;before=w.state()
        with self.assertRaises(ValueError):w.assess({'box':{'pose':'observed'}})
        with self.assertRaises(KeyError):w.assess({'box':{'pose':'pending'},'missing':{'pose':'pending'}})
        self.assertEqual(w.state(),before)
        w.assess({'box':{'pose':'occluded'}},evidence=['capture1'])
        w.update({'box':np.eye(4)},basis='observation_estimate')
        self.assertEqual(w.review_status(['box'])['objects']['box']['aspects']['pose']['status'],'occluded')
        branch=w.fork('review');branch.assess({'box':{'pose':'observed'}},evidence=['capture2'])
        self.assertEqual(w.review_status(['box'])['objects']['box']['aspects']['pose']['status'],'occluded')
        with self.assertRaisesRegex(ValueError,'Stale'):w.assess({'box':{'pose':'pending'}},expected_revision=0)

    def test_geometry_publication_does_not_certify_unknown_surfaces(self):
        w=self.w;w.assess({'box':{'geometry:interior':'unreviewed','pose':'pending'}})
        w.publish_parts('box',{'front':{'mesh':self.root/'geometry/box.glb'}})
        w.update_parts('box',{'front':np.eye(4)},basis='observation_estimate',evidence=['capture'])
        self.assertEqual(w.review_status()['objects']['box']['aspects']['pose']['status'],'pending')
        self.assertEqual(w.review_status()['objects']['box']['aspects']['geometry:interior']['status'],'unreviewed')
        w.assess({'box':{'geometry:floor':'observed','geometry:hidden_walls':'hypothesized'}},evidence=['depth'])
        self.assertEqual(w.review_status()['objects']['box']['aspects']['geometry:hidden_walls']['status'],'hypothesized')

if __name__=='__main__':unittest.main()
