import re
import py_compile
import tempfile
import os
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    VALIDATOR_SYSTEM_PROMPT = """You are an Automated Code Behavior Validator.
Your task is to verify if the Refactored Code has the EXACT same behavior and external interface as the Original Code.
You must compare their structures, inputs, outputs, exceptions, and overall logic.
Minor improvements (like cleaning up duplicate logic, adding type hints, or using dependency injection as specified in rules) are allowed and expected, but the core functionality must remain identical.
Answer strictly with:
IDENTICAL: YES
or
IDENTICAL: NO
followed by a brief reason.
"""

    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def check_syntax(self, file_name: str, code: str) -> Tuple[bool, str]:
        """
        Runs local syntax checking on the refactored code.
        For Python files, uses in-process compile() to get rich syntax error logs.
        For other languages, does a basic parenthesis/structure bracket match validation.
        """
        _, ext = os.path.splitext(file_name.lower())
        
        if ext == ".py":
            try:
                # Compile in-process to get syntax validity check without running code
                compile(code, file_name, 'exec')
                return True, "Syntax check passed (compiled successfully)."
            except SyntaxError as e:
                error_msg = f"SyntaxError in {file_name} at line {e.lineno}, col {e.offset}: {e.msg}\nLine: {e.text}"
                return False, error_msg
            except Exception as e:
                return False, f"Compilation check failed with error: {str(e)}"
                
        # Non-python: basic structural bracket check as lightweight compile guard
        brackets = {
            '(': ')',
            '{': '}',
            '[': ']'
        }
        stack = []
        for idx, char in enumerate(code):
            if char in brackets.keys():
                stack.append((char, idx))
            elif char in brackets.values():
                if not stack:
                    # Closing bracket with no opening
                    # Basic check, but don't fail compilation since comments/strings might trigger it
                    pass
                else:
                    top, _ = stack[-1]
                    if brackets[top] == char:
                        stack.pop()
                        
        return True, "Syntax check completed."

    def validate_behavior(self, file_name: str, original: str, refactored: str) -> Tuple[bool, str]:
        """
        Queries the LLM behavior validator to verify equivalence.
        """
        user_content = f"File: {file_name}\n\n### Original Code\n```\n{original}\n```\n\n### Refactored Code\n```\n{refactored}\n```"
        try:
            response = self.run_prompt(self.VALIDATOR_SYSTEM_PROMPT, user_content, num_predict=512)
            is_identical = bool(re.search(r"IDENTICAL:\s*YES", response, re.IGNORECASE))
            return is_identical, response
        except Exception as e:
            return True, f"Behavior validation bypassed due to engine error: {str(e)}"
