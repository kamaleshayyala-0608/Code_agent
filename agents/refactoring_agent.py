from typing import Dict, Any
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code

class RefactoringAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def execute_refactor(self, context: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Executes code refactoring in logical Passes (Pass 1: Types & Smells;
        Pass 2: Performance & Architecture) using LLM reasoning to ensure code quality.
        """
        original_code = context['original_code']
        file_name = context['file_name']
        plan_steps = plan.get("steps", {"pass1": [], "pass2": [], "pass3": [], "pass4": []})
        
        # 1. Pass 1: Types, Safety, and Structural Code Smells
        pass1_instructions = "\n".join(plan_steps.get("pass1", [])) + "\n" + "\n".join(plan_steps.get("pass2", []))
        if not pass1_instructions.strip():
            pass1_instructions = "- General code formatting and readability cleanup."

        system_prompt_pass1 = """You are an Expert Refactoring Engine (Pass 1: Safety & Structural Smells).
Your task is to apply Pass 1 changes to the source code.
Focus on:
- Adding type safety parameter signatures.
- Flattening logical indentation structures (guard clauses).
- Extracting long helper blocks.

CRITICAL INSTRUCTIONS:
- Do NOT change public signatures unless strictly specified.
- Do NOT output guides, explanations, or conversations.
- Return ONLY the COMPLETE refactored code wrapped in code fences."""

        user_prompt_pass1 = f"""File: {file_name}

Pass 1 Guidelines:
{pass1_instructions}

Original Code:
```
{original_code}
```

Provide the COMPLETE refactored code after executing Pass 1:"""

        try:
            # Execute Pass 1
            pass1_raw = self.run_prompt(system_prompt_pass1, user_prompt_pass1, num_predict=4096)
            pass1_code = clean_refactored_code(pass1_raw)
        except Exception as e:
            # Fallback to original code if Pass 1 fails
            pass1_code = original_code

        # 2. Pass 2: Performance, Caching, and Architecture (SRP)
        pass2_instructions = "\n".join(plan_steps.get("pass3", [])) + "\n" + "\n".join(plan_steps.get("pass4", []))
        if not pass2_instructions.strip():
            # If no performance or architecture steps are scheduled, Pass 1 output is final
            return pass1_code

        system_prompt_pass2 = """You are an Expert Refactoring Engine (Pass 2: Performance & Architecture).
Your task is to refine the Pass 1 code by applying Pass 2 changes.
Focus on:
- Optimizing loop computations.
- Adding memoization helpers (e.g. React.useMemo / React.useCallback).
- Aligning structures with Single Responsibility Principles (SRP).

CRITICAL INSTRUCTIONS:
- Do NOT skip any code block or write "...existing code...".
- Do NOT output guides, explanations, or conversations.
- Return ONLY the COMPLETE refactored code wrapped in code fences."""

        user_prompt_pass2 = f"""File: {file_name}

Pass 2 Guidelines:
{pass2_instructions}

Code from Pass 1:
```
{pass1_code}
```

Provide the COMPLETE final refactored code after executing Pass 2:"""

        try:
            # Execute Pass 2
            pass2_raw = self.run_prompt(system_prompt_pass2, user_prompt_pass2, num_predict=4096)
            final_code = clean_refactored_code(pass2_raw)
            return final_code
        except Exception:
            return pass1_code
