import os
from typing import Dict, Any
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code, LocalCodeAgentEngine, compute_dynamic_token_budget, assess_single_pass_feasibility
from core.tool_registry import ToolRegistry
from utils.completeness_validator import CompletenessValidator

class TooLargeForSinglePassError(Exception):
    """Raised when a file is too large to fit input+output in the model's context window."""
    pass


class RefactoringAgent(BaseAgent):
    """
    Refactoring Agent: Executes full-file code refactoring in ONE SINGLE LLM CALL
    with strict anti-truncation rules, dynamic token budgeting, and completeness validation.
    """

    def __init__(self, model_name: str = "qwen3:8b"):
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
        return result["refactored_code"]

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

        # Pre-flight feasibility check (Item: prevent guaranteed truncation on large files) —
        # the single-pass strategy needs the full original file AND a full rewritten output
        # to both fit in context. If the input alone already eats the budget, refuse up front
        # instead of firing the LLM call and getting back silently truncated/corrupted output.
        extra_context = f"{self.spec_rules}\n{steps_md}\n{context.get('dependency_narrative', '')}"
        feasible, feasibility_msg = assess_single_pass_feasibility(
            original_code, system_instruction, extra_context, self.num_ctx
        )
        if not feasible:
            raise TooLargeForSinglePassError(feasibility_msg)

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
            "refactored_code": final_code
        }
