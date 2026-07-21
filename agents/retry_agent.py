from typing import Dict, Any, Callable, Tuple, List
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code, compute_dynamic_token_budget
from utils.completeness_validator import CompletenessValidator

class RetryAgent(BaseAgent):
    """
    Retry Agent: Autonomous repair loop that handles truncation, placeholders,
    AST mismatches, line drops, and syntax validation failures.
    Extends repair loop across 3 strategies.
    """

    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def attempt_auto_fix(
        self,
        file_name: str,
        original_code: str,
        broken_code: str,
        error_log: str,
        validation_fn: Callable[[str, str], Tuple[bool, str]],
        max_retries: int = 3
    ) -> Tuple[bool, str, List[str]]:
        current_code = broken_code
        retry_logs = []

        system_prompts = [
            # Attempt 1: Direct Error & Completeness Fix
            """You are an Autonomous Code Repair Agent (Attempt 1: Anti-Truncation & Fix).
Analyze the error log, compare original vs broken code. Rewrite the ENTIRE file from line 1 to end.
CRITICAL: Never write '...', 'existing code', 'same as before', 'omitted', or summaries.
Return ONLY the complete corrected source code in code fence.""",

            # Attempt 2: Conservative Interface Preservation
            """You are an Autonomous Code Repair Agent (Attempt 2: Conservative Interface Fix).
The previous attempt failed. Rewrite the FULL source file, preserving all classes, functions, and interfaces.
CRITICAL: Output must contain the complete source code from line 1 to end without omissions.
Return ONLY the complete corrected source code in code fence.""",

            # Attempt 3: Full-File Preservation Repair
            """You are an Autonomous Code Repair Agent (Attempt 3: Full-File Restoration).
Return the COMPLETE working file, preserving original logic while resolving reported compilation/truncation errors.
Return ONLY the complete corrected source code in code fence."""
        ]

        token_budget = compute_dynamic_token_budget(original_code)

        for attempt in range(1, max_retries + 1):
            prompt_idx = min(attempt - 1, len(system_prompts) - 1)
            system_prompt = system_prompts[prompt_idx]

            log_entry = f"Auto-Fix Attempt {attempt}/{max_retries} using Strategy {attempt}..."
            retry_logs.append(log_entry)

            user_prompt = f"""File Name: {file_name}

[Original Working Code]
```
{original_code}
```

[Broken Refactored Code (Attempt {attempt})]
```
{current_code}
```

[Validation Error Feedback]
{error_log}

Rewrite the COMPLETE corrected source file from line 1 to end:"""

            try:
                fixed_raw = self.run_prompt_complete(system_prompt, user_prompt, num_predict=token_budget)
                candidate_code = clean_refactored_code(fixed_raw)

                # Check completeness first (Item 7)
                comp_ok, comp_msg = CompletenessValidator.validate(file_name, original_code, candidate_code)
                if not comp_ok:
                    error_log = f"Completeness Check Failed: {comp_msg}"
                    retry_logs.append(f"✗ Attempt {attempt} completeness failed: {comp_msg}")
                    continue

                current_code = candidate_code

                # Re-validate using validation callback
                success, validation_msg = validation_fn(file_name, current_code)
                if success:
                    retry_logs.append(f"✓ Fix successful on attempt {attempt}: {validation_msg}")
                    return True, current_code, retry_logs
                else:
                    error_log = validation_msg
                    retry_logs.append(f"✗ Attempt {attempt} failed: {validation_msg}")

            except Exception as e:
                error_log = f"Exception: {str(e)}"
                retry_logs.append(f"✗ Exception during auto-fix attempt {attempt}: {str(e)}")

        return False, current_code, retry_logs
