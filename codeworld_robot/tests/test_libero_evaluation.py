"""Private benchmark accounting: transient success and hard action budgets."""
import tempfile,time,unittest
from pathlib import Path
import numpy as np
from codeworld_robot.server import Server
from codeworld_robot.io import read

class EvaluationTest(unittest.TestCase):
    def test_transient_success_is_not_terminal_success_and_budget_prevents_step(self):
        class Env:
            count=0
            def check_success(self):return self.count==1
            def step(self,action):self.count+=1;return {},0,False,{}
            @property
            def sim(self):return self
            def render(self,**kw):return np.zeros((2,2,3),dtype=np.uint8)
        class Video:
            def append_data(self,frame):pass
        with tempfile.TemporaryDirectory() as directory:
            server=Server.__new__(Server);server.env=Env();server.video=Video();server.steps=0
            server.cfg={'evaluation_output':str(Path(directory)/'private/result.json'),'max_action_steps':2,
                        'suite':'fixture','scene_task':'fixture','seed':0,'initial_state':0}
            server.evaluation_started_at=time.time();server.official_first_success_step=None;server.initial_official_success=False
            server.step([0]*7);server.step([0]*7)
            with self.assertRaises(RuntimeError):server.step([0]*7)
            server.write_evaluation('CLOSED');result=read(server.cfg['evaluation_output'])
            self.assertEqual(server.env.count,2)
            self.assertTrue(result['official_success_ever'])
            self.assertFalse(result['official_success_final'])
            self.assertEqual(result['first_success_step'],1)

if __name__=='__main__':unittest.main()
