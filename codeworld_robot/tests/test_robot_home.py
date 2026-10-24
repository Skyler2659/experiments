import copy,json,unittest
from codeworld_robot.tests.test_completion import CompletionTests
from codeworld_robot.io import atomic,read
from codeworld_workspace.robot_home import capture_home,required_goal,measure_home,GOAL_ID

class RobotHomeTests(unittest.TestCase):
 def setUp(self):
  self.fixture=CompletionTests('test_l07_direction_alone_does_not_prove_adjacency');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.w=self.fixture.w;self.root=self.fixture.root;self.fixture.satisfied()
  self.robot={'robot_state_ref':'episode:10','arms':[{'arm':0,'joint_positions':[.2,-.3],'gripper_qpos':[.044,.044]},{'arm':1,'joint_positions':[-.2,.4],'gripper_qpos':[.044,.044]}]}
  atomic(self.root/'task/robot_home.json',capture_home(self.robot))
  (self.root/'task/completion.json').unlink();self.contract=self.w.define_completion(self.fixture.goals)
 def verify(self,robot=None,live=None):
  robot=copy.deepcopy(robot or self.robot);s=self.w.state();s['runtime']={'robot_state':robot};atomic(self.w.state_path,s)
  return self.w.verify_completion(robot_state=live or robot)
 def test_automatically_required_and_measured_success(self):
  self.assertIn(GOAL_ID,self.contract['goals']);self.assertEqual(self.verify()['status'],'VERIFIED')
 def test_both_arms_and_grippers_required(self):
  for index,field in [(0,'joint_positions'),(1,'joint_positions'),(0,'gripper_qpos'),(1,'gripper_qpos')]:
   robot=copy.deepcopy(self.robot);robot['arms'][index][field][0]=0
   self.assertEqual(self.verify(robot)['goals'][GOAL_ID]['status'],'NOT_SATISFIED')
 def test_missing_or_nonfinite_feedback_not_success(self):
  robot=copy.deepcopy(self.robot);robot['arms'].pop();self.assertEqual(self.verify(robot)['goals'][GOAL_ID]['status'],'INCONCLUSIVE')
  condition=self.contract['goals'][GOAL_ID]['conditions'][0];robot=copy.deepcopy(self.robot);robot['arms'][0]['joint_positions'][0]=float('nan')
  self.assertEqual(measure_home(self.root,robot,condition)['status'],'INCONCLUSIVE')
 def test_stale_feedback_cannot_certify_home(self):
  live=copy.deepcopy(self.robot);live['robot_state_ref']='episode:11'
  self.assertEqual(self.verify(live=live)['goals'][GOAL_ID]['status'],'INCONCLUSIVE')
 def test_initial_target_is_not_zero_and_cannot_be_replaced(self):
  self.assertEqual(read(self.root/'task/robot_home.json')['arms'][0]['joint_positions'],[.2,-.3])
  home=read(self.root/'task/robot_home.json');home['arms'][0]['joint_positions']=[0,0];atomic(self.root/'task/robot_home.json',home)
  with self.assertRaisesRegex(ValueError,'baseline changed'):self.verify()
 def test_required_goal_cannot_be_overridden(self):
  goals={**self.fixture.goals,GOAL_ID:{'description':'ignore','conditions':self.fixture.goals['cup']['conditions']}}
  with self.assertRaisesRegex(ValueError,'cannot be overridden'):self.w.define_completion(goals)

if __name__=='__main__':unittest.main()
