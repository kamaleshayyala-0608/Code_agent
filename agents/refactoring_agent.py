from typing import Dict, Any
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code

class RefactoringAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def execute_refactor(self, context: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Executes code refactoring by applying rules, fixing patterns, following the plan,
        and outputting ONLY the complete refactored code.
        """
        system_prompt = """You are an Expert Refactoring Engine.
You rewrite source code files to optimize their structure, performance, readability, and compliance with standards.
You MUST strictly follow the provided Refactoring Plan and Rules.
Preserve ALL behaviors, interfaces, signatures, public functions, and logic.

CRITICAL INSTRUCTIONS:
- Do NOT output recommendations.
- Do NOT output explanations.
- Do NOT output markdown guides.
- Do NOT skip any code or write "...existing code...".
- Return ONLY the COMPLETE refactored source code.
- Return the code wrapped inside standard markdown code fences matching the file language (e.g. ```python ... ```)."""

        user_prompt = f"""File name: {context['file_name']}

Dependency Constraints:
{context['dependency_narrative']}

Rules to Apply:
{context['rules_applied']}

Patterns to Resolve:
{context['patterns_identified']}

Refactoring Plan Steps:
{plan['steps_md']}

Original Code:
```
{context['original_code']}
```

Provide the COMPLETE refactored source file code below."""

        try:
            refactored_raw = self.run_prompt(system_prompt, user_prompt, num_predict=4096)
            cleaned = clean_refactored_code(refactored_raw)
            return cleaned
        except Exception as e:
            raise RuntimeError(f"Refactoring agent failed to rewrite code: {str(e)}")
