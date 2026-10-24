import unittest
import numpy as np
from reconstruction_assist.evaluate import compare_depth
class DepthFeedbackTest(unittest.TestCase):
 def test_foreground_occlusion_is_not_false_occupied(self):
  depth=np.ones((20,20));mask=np.zeros((20,20),bool);hit=np.full((20,20),1.5)
  r,*_=compare_depth(hit,depth,mask)
  self.assertEqual(r['false_occupied_fraction'],0)
 def test_missing_and_false_occupied_are_distinct(self):
  depth=np.ones((30,30));mask=np.zeros((30,30),bool);mask[10:20,10:20]=True
  hit=np.full((30,30),np.inf);hit[2:7,2:7]=.8
  r,*_=compare_depth(hit,depth,mask)
  self.assertEqual(r['missing_observed_fraction'],1);self.assertEqual(r['false_occupied_fraction'],1)
 def test_correct_surface_matches(self):
  depth=np.ones((20,20));mask=np.ones((20,20),bool)
  r,*_=compare_depth(depth,depth,mask)
  self.assertEqual(r['matched_observed_fraction'],1);self.assertEqual(r['missing_observed_fraction'],0)
if __name__=='__main__':unittest.main()
