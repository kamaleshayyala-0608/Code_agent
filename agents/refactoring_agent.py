import os
from typing import Dict, Any
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code, LocalCodeAgentEngine, compute_dynamic_token_budget
from core.tool_registry import ToolRegistry
from utils.completeness_validator import CompletenessValidator

class RefactoringAgent(BaseAgent):
    """
    Refactoring Agent: Executes full-file code refactoring in ONE SINGLE LLM CALL
    with strict anti-truncation rules, dynamic token budgeting, and completeness validation.
    """

    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)
        self.spec_rules = ""
        spec_path = "memory/spec.md"
        if not os.path.exists(spec_path):
            spec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory", "spec.md")
        if not os.path.exists(spec_path):
            spec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules", "refactoring_spec.md")
        if os.path.exists(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    self.spec_rules = f.read()
            except Exception:
                pass

    def execute_refactor(self, context: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Executes full-file code refactoring via a SINGLE high-integrity LLM call.
        Returns the COMPLETE refactored source code string.
        """
        result = self.transform_full_file(context, plan)
        return result["refactored_content"]

    def transform_full_file(self, context: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        original_code = context['original_code']
        file_name = context['file_name']
        steps_md = plan.get("steps_md", "Apply standard refactoring rules.")

        system_instruction = LocalCodeAgentEngine.FULL_FILE_REFACTOR_PROMPT

        user_prompt = f"""File Path: {file_name}

Refactoring Specification (spec.md):
{self.spec_rules}

Planner Execution Tasks:
{steps_md}

Project Context & Dependencies:
{context.get('dependency_narrative', '')}

Original Source Code:
```
{original_code}
```

Rewrite the COMPLETE source file from line 1 to the end:"""

        # Dynamic Token Budgeting based on line count (Item 9)
        token_budget = compute_dynamic_token_budget(original_code)

        raw_response = self.run_prompt_complete(system_instruction, user_prompt, num_predict=token_budget)

        # Extract code & check for banned placeholders
        candidate_code = clean_refactored_code(raw_response)

        # Completeness Check (Item 3 & 5)
        completeness_ok, completeness_msg = CompletenessValidator.validate(file_name, original_code, candidate_code)
        if not completeness_ok:
            raise ValueError(f"Completeness validation failed: {completeness_msg}")

        # Deterministic Formatting (Item 4)
        final_code = ToolRegistry.format_code(file_name, candidate_code)

        return {
            "file_path": file_name,
            "original_content": original_code,
            "refactored_content": final_code
        }
