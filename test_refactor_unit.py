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

    def test_ast_parser(self):
        from utils.ast_parser import ASTParser
        test_code = """
import os
import sys
from collections import defaultdict

class UserService:
    def __init__(self, db_client):
        self.db = db_client

    def fetch_user(self, user_id: str) -> dict:
        return self.db.find(user_id)

def get_status():
    return "OK"
"""
        meta = ASTParser.parse_file("user_service.py", test_code)
        
        # Verify imports
        self.assertIn("os", meta["imports"])
        self.assertIn("sys", meta["imports"])
        self.assertIn("collections", meta["imports"])
        
        # Verify classes
        self.assertEqual(len(meta["classes"]), 1)
        self.assertEqual(meta["classes"][0]["name"], "UserService")
        
        # Verify methods
        methods = meta["classes"][0]["methods"]
        method_names = [m["name"] for m in methods]
        self.assertIn("__init__", method_names)
        self.assertIn("fetch_user", method_names)
        
        # Verify standalone functions
        func_names = [f["name"] for f in meta["functions"]]
        self.assertIn("get_status", func_names)
        
        # Verify complexity estimate is calculated
        self.assertTrue(meta["complexity_estimate"] >= 1)

    def test_dependency_analyzer(self):
        from utils.dependency_analyzer import DependencyAnalyzer
        
        files = {
            "app.py": "from db import Database\nfrom models.user import User\n",
            "db.py": "import sqlite3\n",
            "models/user.py": "class User:\n    pass\n"
        }
        
        parsed_metadata = {
            "app.py": {"imports": ["db", "models.user"]},
            "db.py": {"imports": ["sqlite3"]},
            "models/user.py": {"imports": []}
        }
        
        dep_graph = DependencyAnalyzer.analyze_workspace_dependencies(files, parsed_metadata)
        
        # Verify app.py depends on db.py and models/user.py
        self.assertIn("db.py", dep_graph["app.py"]["depends_on"])
        self.assertIn("models/user.py", dep_graph["app.py"]["depends_on"])
        
        # Verify db.py depended_on_by contains app.py
        self.assertIn("app.py", dep_graph["db.py"]["depended_on_by"])
        
        # Verify models/user.py depended_on_by contains app.py
        self.assertIn("app.py", dep_graph["models/user.py"]["depended_on_by"])

    @patch('utils.vector_db.LocalVectorDB._get_embedding')
    def test_vector_db_retrieval(self, mock_embed):
        from utils.vector_db import LocalVectorDB
        
        # Mock embedding return: vector of length 3
        mock_embed.return_value = [0.1, 0.2, 0.3]
        
        # Create DB instance
        db = LocalVectorDB()
        
        # Mock manual insertion of a few rules with embeddings
        db.rules = [
            {"id": 1, "title": "Single Responsibility Principle", "text": "Details about SRP", "embedding": [1.0, 0.0, 0.0]},
            {"id": 2, "title": "Dependency Injection", "text": "Details about DI", "embedding": [0.0, 1.0, 0.0]},
        ]
        
        # Query rules
        query = "Need dependency injection info"
        retrieved = db.retrieve_relevant_rules(query, top_k=1)
        
        # Verify it returns rules
        self.assertEqual(len(retrieved), 1)
        self.assertIn(retrieved[0][0]["id"], [1, 2])

    def test_export_agent(self):
        from agents.export_agent import ExportAgent

        original_files = {
            "app.py": "print('original')"
        }
        refactored_files = {
            "app.py": "print('refactored')"
        }
        reports = {
            "app.py": {
                "rules": "### Rule 1: Use Type Hints",
                "patterns_list": ["Magic Numbers"],
                "patterns_report": "Found magic numbers.",
                "plan": {"priority": "High", "confidence": 95, "steps_md": "- Add hints"},
                "validation": {"success": True, "syntax_msg": "Syntax passed", "behavior_msg": "Behavior passed"},
                "retries": [],
                "quality": {
                    "score_before": 70,
                    "score_after": 95,
                    "orig_readability": 70,
                    "ref_readability": 95,
                    "orig_lines": 10,
                    "ref_lines": 8
                }
            }
        }

        exporter = ExportAgent()
        packaged = exporter.package_refactored_project(original_files, refactored_files, reports, spec_rules="# Refactoring Spec Rules")

        # Verify structured ZIP layout
        self.assertIn("refactored/app.py", packaged)
        self.assertEqual(packaged["refactored/app.py"], "print('refactored')")
        self.assertIn("original/app.py", packaged)
        self.assertEqual(packaged["original/app.py"], "print('original')")
        self.assertIn("summary.json", packaged)
        self.assertIn("README.md", packaged)
        self.assertIn("reports/app.py.md", packaged)
        self.assertIn("metrics/quality_metrics.json", packaged)
        self.assertIn("suggestions/code_smells.json", packaged)

if __name__ == '__main__':
    unittest.main()


