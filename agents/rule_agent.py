from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.vector_db import LocalVectorDB

class RuleExtractionAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)
        self.vector_db = LocalVectorDB()

    def extract_relevant_rules(self, file_name: str, code: str, metadata: Dict[str, Any]) -> str:
        """
        Retrieves relevant refactoring rules locally via Vector DB embeddings/keywords.
        Bypasses LLM calls to prevent latency and parser instability.
        """
        # Formulate search query from file details
        classes_str = ", ".join([c.get("name", "") for c in metadata.get("classes", [])])
        funcs_str = ", ".join([f.get("name", "") for f in metadata.get("functions", [])])
        hooks_str = ", ".join(metadata.get("hooks", []))
        
        query = f"File: {file_name}\n"
        if classes_str:
            query += f"Classes: {classes_str}\n"
        if funcs_str:
            query += f"Functions: {funcs_str}\n"
        if hooks_str:
            query += f"React Hooks: {hooks_str}\n"
            
        # Append some context lines
        lines = code.split("\n")
        query += "\n".join(lines[:20])

        # Retrieve top 5 relevant rules
        retrieved_rules = self.vector_db.retrieve_relevant_rules(query, top_k=5)

        if not retrieved_rules:
            return "## Applied Rules\n\n- No coding standards matched."

        # Format cleanly as an applied rules checklist
        markdown = "## Applied Rules\n\n"
        for rule, score in retrieved_rules:
            title = rule.get("title", f"Rule {rule.get('id')}")
            # Format title (e.g. "1. Single Responsibility Principle (SRP)")
            markdown += f"✓ **{title}**\n"
            
        markdown += "\n### Selected Rules Details\n"
        for rule, score in retrieved_rules:
            markdown += f"\n#### {rule.get('title')}\n{rule.get('text')}\n"
            
        return markdown
