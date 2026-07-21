from typing import Dict, Any, Callable, Tuple, List
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code

class RetryAgent(BaseAgent):
    """
    Retry Agent: Autonomous repair loop that switches prompts on each retry attempt.
    Attempt 1: Direct syntax/error fix.
    Attempt 2: Conservative interface-preserving rewrite.
    Attempt 3: Minimal surgical patch.
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
            # Attempt 1: Direct Error Fix
            """You are an Autonomous Code Repair Agent (Attempt 1: Direct Fix).
Analyze the error log, compare original vs broken code, and fix the specific syntax/behavior error.
Return ONLY the complete corrected source code in code fence.""",

            # Attempt 2: Conservative Interface Preservation
            """You are an Autonomous Code Repair Agent (Attempt 2: Conservative Interface Fix).
The previous attempt failed validation. Ensure ALL original class names, function signatures, imports, and component exports remain strictly untouched while fixing the bug.
Return ONLY the complete corrected source code in code fence.""",

            # Attempt 3: Minimal Surgical Patch
            """You are an Autonomous Code Repair Agent (Attempt 3: Surgical Patch).
Focus exclusively on fixing the compiler error or broken reference with minimal alterations to the original structure.
Return ONLY the complete corrected source code in code fence."""
        ]

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

Please fix the file and return the complete corrected code:"""

            try:
                fixed_raw = self.run_prompt_complete(system_prompt, user_prompt, num_predict=4096)
                candidate_code = clean_refactored_code(fixed_raw)

                if candidate_code and candidate_code.strip():
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
                retry_logs.append(f"✗ Exception during auto-fix attempt {attempt}: {str(e)}")

        return False, current_code, retry_logs
