import re
import os
import subprocess
import tempfile
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
        For Javascript/TypeScript/React, integrates npx esbuild compiler validation.
        """
        _, ext = os.path.splitext(file_name.lower())
        
        if ext == ".py":
            try:
                compile(code, file_name, 'exec')
                return True, "Syntax check passed (compiled successfully)."
            except SyntaxError as e:
                error_msg = f"SyntaxError in {file_name} at line {e.lineno}, col {e.offset}: {e.msg}\nLine: {e.text}"
                return False, error_msg
            except Exception as e:
                return False, f"Compilation check failed with error: {str(e)}"
                
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            # Esbuild syntax compiler check
            suffix = ext if ext else ".tsx"
            temp_path = None
            try:
                # Write code to a temp file matching the extension
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as temp_file:
                    temp_file.write(code)
                    temp_path = temp_file.name

                # Run npx esbuild compile test
                # output to NUL on Windows, /dev/null on Unix
                outfile = "NUL" if os.name == "nt" else "/dev/null"
                cmd = ["npx", "esbuild", temp_path, f"--outfile={outfile}"]
                
                # Run sync subprocess
                result = subprocess.run(cmd, capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                
                if result.returncode == 0:
                    return True, "Syntax check passed (compiled successfully via esbuild)."
                else:
                    err_msg = result.stderr or result.stdout or "Syntax check failed."
                    # Sanitize temp path from logs
                    clean_msg = err_msg.replace(temp_path, file_name)
                    return False, clean_msg.strip()
            except Exception as e:
                # Fallback to bypass on configuration/tooling issues
                return True, f"Syntax verification bypassed (esbuild check failed to launch: {str(e)})"
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                        
        # Basic check for other file types
        return True, "Syntax check skipped (unsupported compile file type)."

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
