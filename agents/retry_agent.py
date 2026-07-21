from typing import Dict, Any, Callable, Tuple
from agents.base_agent import BaseAgent
from agent_core import clean_refactored_code

class RetryAgent(BaseAgent):
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
    ) -> Tuple[bool, str, list]:
        """
        Orchestrates autonomous repair loop.
        Sends compilation errors or validation failures back to the LLM to self-correct.
        Returns (success, fixed_code, retry_logs).
        """
        current_code = broken_code
        retry_logs = []
        
        system_prompt = """You are an Autonomous Code Repair Agent.
Your task is to fix a refactored file that has failed validation checks (compilation error or behavior mismatch).
You will be given the original code, the broken refactored code, and the error logs from the checker.
Analyze the error carefully and output the corrected COMPLETE source code file.

CRITICAL INSTRUCTIONS:
- Preserve all external interfaces and behaviors.
- Fix ONLY the syntax error or logical bug reported.
- Do NOT output explanations or conversations.
- Return ONLY the COMPLETE corrected source code inside markdown code fences."""

        for attempt in range(1, max_retries + 1):
            log_entry = f"Auto-Fix Attempt {attempt}/{max_retries}..."
            retry_logs.append(log_entry)
            
            user_prompt = f"""File Name: {file_name}

[Original Code]
```
{original_code}
```

[Broken Refactored Code]
```
{current_code}
```

[Validation Error Log]
{error_log}

Please fix the file and return the complete corrected code."""

            try:
                fixed_raw = self.run_prompt_complete(system_prompt, user_prompt, num_predict=4096)
                current_code = clean_refactored_code(fixed_raw)
                
                # Re-validate
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
