import os
import unittest
from lib.system import System
from lib.config import MODELS_FOLDER

class TestSystem(unittest.TestCase):
    def test_check_system_requirements(self):
        system = System()
        self.assertTrue(system.check_system_requirements())
    
    def test_check_cuda_availability(self):
        system = System()
        self.assertTrue(system.check_cuda_availability())

    def test_setup_environment_variables(self):
        system = System()
        system.setup_environment_variables()
        self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(os.environ["TRANSFORMERS_CACHE"], MODELS_FOLDER)
        self.assertEqual(os.environ["HF_HOME"], MODELS_FOLDER)

if __name__ == '__main__':
    unittest.main()