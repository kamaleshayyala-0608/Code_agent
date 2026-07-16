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
    def test_refactor_file_stage1_prompt(self, mock_response):
        mock_response.return_value = "No refactoring required"
        
        # Run refactor_file_two_stage
        self.engine.refactor_file_two_stage("test.py", "print('hello')", self.engine.spec_rules)
        
        # Check that _generate_local_response was called with the spec rules inside the prompt
        args, kwargs = mock_response.call_args_list[0]
        prompt = args[0]
        self.assertIn("Apply ALL rules below.", prompt)
        self.assertIn("Single Responsibility Principle", prompt)
        self.assertIn("Dependency Injection", prompt)

if __name__ == '__main__':
    unittest.main()
