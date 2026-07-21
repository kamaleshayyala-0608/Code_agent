from typing import Dict, Any
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code

class RefactoringAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def _predict_budget(self, prompt_text: str) -> int:
        # Rough token estimate: ~4 chars/token. Leave headroom in num_ctx (32768) for the output.
        est_input_tokens = len(prompt_text) // 4
        return max(1024, min(4096, self.num_ctx - est_input_tokens - 256))

    def execute_refactor(self, context: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Executes full-file code refactoring in logical Passes (Pass 1: Types & Smells;
        Pass 2: Performance & Architecture) using LLM reasoning to ensure code quality.
        Returns the COMPLETE refactored source code string.
        """
        result = self.transform_full_file(context, plan)
        return result["refactored_content"]

    def transform_full_file(self, context: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full-file transformation engine:
        - Reads the entire file.
        - Applies all refactorings across the file.
        - Validates and returns full file content object.
        """
        original_code = context['original_code']
        file_name = context['file_name']
        plan_steps = plan.get("steps", {"pass1": [], "pass2": [], "pass3": [], "pass4": []})
        
        # 1. Pass 1: Types, Safety, and Structural Code Smells
        pass1_instructions = "\n".join(plan_steps.get("pass1", [])) + "\n" + "\n".join(plan_steps.get("pass2", []))
        if not pass1_instructions.strip():
            pass1_instructions = "- General code formatting and readability cleanup."

        system_prompt_pass1 = """You are an Expert Full-File Refactoring Engine (Pass 1: Safety & Structural Smells).
Your task is to refactor the ENTIRE source file. Do not output code snippets, patches, or diffs.

Focus on:
- Adding type safety parameter signatures.
- Flattening logical indentation structures (guard clauses).
- Extracting long helper blocks.

CRITICAL INSTRUCTIONS:
- Do NOT change public signatures unless strictly specified.
- Do NOT output guides, explanations, snippet blocks, or conversations.
- Do NOT write "...existing code..." or omit any part of the file.
- Return ONLY the COMPLETE refactored source code wrapped in a code fence."""

        user_prompt_pass1 = f"""File Path: {file_name}

Pass 1 Refactoring Guidelines:
{pass1_instructions}

Original Full File Content:
```
{original_code}
```

Provide the COMPLETE refactored file content after Pass 1:"""

        try:
            # Execute Pass 1
            pass1_raw = self.run_prompt_complete(system_prompt_pass1, user_prompt_pass1, num_predict=self._predict_budget(user_prompt_pass1))
            pass1_code = clean_refactored_code(pass1_raw)
            if not pass1_code.strip():
                pass1_code = original_code
        except Exception:
            # Fallback to original code if Pass 1 fails
            pass1_code = original_code

        # 2. Pass 2: Performance, Caching, and Architecture (SRP)
        pass2_instructions = "\n".join(plan_steps.get("pass3", [])) + "\n" + "\n".join(plan_steps.get("pass4", []))
        if not pass2_instructions.strip():
            return {
                "file_path": file_name,
                "original_content": original_code,
                "refactored_content": pass1_code
            }

        system_prompt_pass2 = """You are an Expert Full-File Refactoring Engine (Pass 2: Performance & Architecture).
Your task is to refine the full source code from Pass 1.

Focus on:
- Optimizing loop computations.
- Adding memoization helpers (e.g. React.useMemo / React.useCallback).
- Aligning structures with Single Responsibility Principles (SRP).

CRITICAL INSTRUCTIONS:
- Do NOT skip any code block, write snippet fragments, or write "...existing code...".
- Do NOT output guides, explanations, or conversations.
- Return ONLY the COMPLETE refactored source code wrapped in a code fence."""

        user_prompt_pass2 = f"""File Path: {file_name}

Pass 2 Refactoring Guidelines:
{pass2_instructions}

Full File Content from Pass 1:
```
{pass1_code}
```

Provide the COMPLETE final refactored file content:"""

        try:
            # Execute Pass 2
            pass2_raw = self.run_prompt_complete(system_prompt_pass2, user_prompt_pass2, num_predict=self._predict_budget(user_prompt_pass2))
            final_code = clean_refactored_code(pass2_raw)
            if not final_code.strip():
                final_code = pass1_code
        except Exception:
            final_code = pass1_code

        return {
            "file_path": file_name,
            "original_content": original_code,
            "refactored_content": final_code
        }
