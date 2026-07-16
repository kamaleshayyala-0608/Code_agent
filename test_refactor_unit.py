import unittest
from unittest.mock import patch, MagicMock
from agent_core import LocalCodeAgentEngine

class TestRefactoringPipeline(unittest.TestCase):
    @patch('ollama.list')
    def setUp(self, mock_list):
        # Mock ollama list so we don't try to connect during initialization
        mock_list.return_value = []
        self.engine = LocalCodeAgentEngine()

    def test_spec_loaded(self):
        # Check that spec_rules is loaded and contains our standards
        self.assertTrue(len(self.engine.spec_rules) > 0)
        self.assertIn("Single Responsibility Principle", self.engine.spec_rules)
        self.assertIn("Dependency Injection", self.engine.spec_rules)

    @patch('agent_core.LocalCodeAgentEngine._generate_local_response')
    def test_transform_file(self, mock_response):
        # Mock responses: returns the refactored code
        mock_response.return_value = "print('refactored')"
        
        # Run transform_file
        res = self.engine.transform_file("test.py", "print('hello')")
        
        self.assertEqual(res, "print('refactored')")
        self.assertEqual(mock_response.call_count, 1)
        
        args1, _ = mock_response.call_args_list[0]
        prompt1 = args1[0]
        self.assertIn("Enterprise Code Transformation Engine", prompt1)
        self.assertIn("Single Responsibility Principle", prompt1)

if __name__ == '__main__':
    unittest.main()

