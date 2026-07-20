from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.vector_db import LocalVectorDB

class RuleExtractionAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)
        self.vector_db = LocalVectorDB()
        self._rules_cache: Dict[str, str] = {} # In-memory cache

    def extract_relevant_rules(self, file_name: str, code: str, metadata: Dict[str, Any]) -> str:
        """
        Retrieves relevant refactoring rules locally. Filters matches by file language,
        detected frameworks, and severity tags, caching results in-memory.
        """
        cache_key = f"{file_name}_{hash(code[:2000])}"
        if cache_key in self._rules_cache:
            return self._rules_cache[cache_key]

        # 1. Determine Language and Framework metadata tags
        is_python = file_name.endswith(".py")
        is_js_ts = file_name.endswith((".js", ".jsx", ".ts", ".tsx"))
        is_react = "react" in code.lower() or metadata.get("hooks") or file_name.endswith((".jsx", ".tsx"))
        
        # Check frameworks in code
        detected_frameworks = []
        if is_react:
            detected_frameworks.append("react")
        if "streamlit" in code.lower():
            detected_frameworks.append("streamlit")
        if "django" in code.lower():
            detected_frameworks.append("django")
        if "fastapi" in code.lower():
            detected_frameworks.append("fastapi")
            
        # 2. Formulate query from file context
        classes_str = ", ".join([c.get("name", "") for c in metadata.get("classes", [])])
        funcs_str = ", ".join([f.get("name", "") for f in metadata.get("functions", [])])
        
        query = f"File: {file_name}\n"
        if is_python:
            query += "Language: Python\n"
        if is_js_ts:
            query += "Language: JavaScript/TypeScript\n"
        if detected_frameworks:
            query += f"Frameworks: {', '.join(detected_frameworks)}\n"
        if classes_str:
            query += f"Classes: {classes_str}\n"
        if funcs_str:
            query += f"Functions: {funcs_str}\n"

        # 3. Retrieve relevant rules locally
        retrieved_rules = self.vector_db.retrieve_relevant_rules(query, top_k=6)

        # 4. Filter matched rules based on metadata tags
        filtered_rules = []
        for rule, score in retrieved_rules:
            rule_text = rule.get("text", "").lower()
            
            # Exclude rules that mismatch core language constraints
            if is_python and ("react" in rule_text or "hooks" in rule_text or "typescript" in rule_text):
                continue
            if is_js_ts and "python" in rule_text and "type hints" in rule_text:
                # Rule 7 is "Use Type Hints in Python". Exclude for TS/JS.
                continue
                
            filtered_rules.append(rule)

        # Fallback rules if no matches pass filtering
        if not filtered_rules:
            # Revert to top 2 rules from vector db as safe fallback
            filtered_rules = [r[0] for r in retrieved_rules[:2]]

        # 5. Format rules checklist
        markdown = "## Applied Rules\n\n"
        for rule in filtered_rules:
            title = rule.get("title", f"Rule {rule.get('id')}")
            markdown += f"✓ **{title}**\n"
            
        markdown += "\n### Selected Rules Details\n"
        for rule in filtered_rules:
            markdown += f"\n#### {rule.get('title')}\n{rule.get('text')}\n"

        self._rules_cache[cache_key] = markdown
        return markdown
