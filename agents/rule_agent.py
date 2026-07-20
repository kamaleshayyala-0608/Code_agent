from typing import List, Dict, Any
from agents.base_agent import BaseAgent
from utils.vector_db import LocalVectorDB

class RuleExtractionAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)
        self.vector_db = LocalVectorDB()

    def extract_relevant_rules(self, file_name: str, code: str, metadata: Dict[str, Any]) -> str:
        """
        Retrieves relevant refactoring rules using local Vector DB embeddings
        and refines them using LLM reasoning.
        """
        # Formulate query from file context
        query = f"File: {file_name}\n"
        classes_str = ", ".join([c.get("name", "") for c in metadata.get("classes", [])])
        funcs_str = ", ".join([f.get("name", "") for f in metadata.get("functions", [])])
        
        if classes_str:
            query += f"Classes: {classes_str}\n"
        if funcs_str:
            query += f"Functions: {funcs_str}\n"
            
        # Get head of the code to understand context
        lines = code.split("\n")
        query += "\n".join(lines[:30])

        # Retrieve top 6 relevant rules from the vector DB
        retrieved_rules_scores = self.vector_db.retrieve_relevant_rules(query, top_k=6)
        
        if not retrieved_rules_scores:
            return "No relevant specification rules found."

        rules_context = ""
        for rule, score in retrieved_rules_scores:
            rules_context += f"Rule ID: {rule.get('id')}\nTitle: {rule.get('title')}\nDetails:\n{rule.get('text')}\n\n"

        # Ask Gemma to review the rules and extract only the ones that directly apply to this code snippet
        system_prompt = """You are a Code Standards Auditor.
Given the target source code file context and a set of candidate coding standards, extract and refine ONLY the guidelines that directly apply to refactoring this specific file.
Summarize the selected rules and provide a checklist. Do not include rules that are irrelevant to the target file's content or language.
Return your response in clear, concise Markdown."""

        user_prompt = f"""Target File: {file_name}

Candidate Coding Standards:
{rules_context}

Source Code (Excerpt):
```
{code[:8000]}
```

Provide the filtered, highly-relevant rule specification subset in Markdown."""

        try:
            refined_rules = self.run_prompt(system_prompt, user_prompt, num_predict=1500)
            return refined_rules
        except Exception:
            # Fallback to returning raw retrieved rules if LLM fails
            fallback_md = "## Relevant Rules (Vector DB Match)\n\n"
            for rule, score in retrieved_rules_scores:
                fallback_md += f"### {rule.get('title')} (Match score: {score:.2f})\n{rule.get('text')}\n\n"
            return fallback_md
