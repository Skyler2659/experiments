import unittest
from pathlib import Path
from codeworld_robot.backend_contract import execution_instructions

class BackendContractTests(unittest.TestCase):
    def test_libero_excludes_x5_operation_sections(self):
        text=execution_instructions('libero',(Path(__file__).parents[1]/'README.md').read_text())
        self.assertIn('orientation is the native end-effector BODY orientation',text)
        self.assertNotIn('RoboDojo uses public X5 URDF',text)
        self.assertNotIn('## RoboDojo official action budget',text)
        self.assertNotIn('Fresh RoboDojo episodes',text)
        self.assertIn('## CodeWorld completion contract',text)
    def test_robodojo_excludes_panda_adapter(self):
        text=execution_instructions('robodojo',(Path(__file__).parents[1]/'README.md').read_text())
        self.assertIn('TCP local +X is the approach direction for this backend',text)
        self.assertNotIn('## Panda and FR3 joint adapters',text)
        self.assertNotIn('For the LIBERO joint-path adapter',text)
        self.assertIn('## CodeWorld completion contract',text)
    def test_unknown_backend_rejected(self):
        with self.assertRaises(KeyError):execution_instructions('unknown','')
