import os
from typing import Dict, Any
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code, LocalCodeAgentEngine
from core.tool_registry import ToolRegistry

class RefactoringAgent(BaseAgent):
    """
    Executes multi-pass refactoring across 5 distinct passes:
    Pass 1: Modernization & Type Safety
    Pass 2: Performance & Caching
    Pass 3: Naming & Readability
    Pass 4: Security & Error Handling
    Pass 5: Formatting & SRP Structuring
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

    def _predict_budget(self, prompt_text: str) -> int:
        est_input_tokens = len(prompt_text) // 4
        return max(1024, min(4096, self.num_ctx - est_input_tokens - 256))

    def execute_refactor(self, context: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Executes full-file code refactoring in 5 logical passes.
        Returns the COMPLETE refactored source code string.
        """
        result = self.transform_full_file(context, plan)
        return result["refactored_content"]

    def transform_full_file(self, context: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        original_code = context['original_code']
        file_name = context['file_name']
        plan_steps = plan.get("steps", {})

        current_code = original_code

        # Pass Definitions
        passes = [
            ("Pass 1: Modernization & Types", LocalCodeAgentEngine.REFACTOR_PASS_1_MODERNIZATION_PROMPT, plan_steps.get("pass1", [])),
            ("Pass 2: Performance & Caching", LocalCodeAgentEngine.REFACTOR_PASS_2_PERFORMANCE_PROMPT, plan_steps.get("pass2", [])),
            ("Pass 3: Naming & Readability", LocalCodeAgentEngine.REFACTOR_PASS_3_NAMING_PROMPT, plan_steps.get("pass3", [])),
            ("Pass 4: Security & Errors", LocalCodeAgentEngine.REFACTOR_PASS_4_SECURITY_PROMPT, plan_steps.get("pass4", [])),
            ("Pass 5: Formatting & Structure", LocalCodeAgentEngine.REFACTOR_PASS_5_FORMATTING_PROMPT, plan_steps.get("pass5", []))
        ]

        for pass_title, system_instruction, task_list in passes:
            instructions_text = "\n".join(task_list) if task_list else f"- Execute {pass_title} based on spec.md rules."

            user_prompt = f"""File Path: {file_name}

{pass_title} Guidelines:
{instructions_text}

Project Context & Dependencies:
{context.get('dependency_narrative', '')}

Input Source Code:
```
{current_code}
```

Provide the COMPLETE updated source code file with all {pass_title} improvements applied:"""

            try:
                raw_response = self.run_prompt_complete(system_instruction, user_prompt, num_predict=self._predict_budget(user_prompt))
                candidate_code = clean_refactored_code(raw_response)
                if candidate_code and candidate_code.strip():
                    current_code = candidate_code
            except Exception:
                # If a pass fails, retain current code and proceed to next pass
                pass

        # Final tool-based deterministic formatting
        final_code = ToolRegistry.format_code(file_name, current_code)

        return {
            "file_path": file_name,
            "original_content": original_code,
            "refactored_content": final_code
        }
