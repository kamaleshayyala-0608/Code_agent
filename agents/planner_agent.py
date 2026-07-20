import re
import json
from typing import Dict, Any
from agents.base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a structured refactoring plan based on codebase dependencies,
        extracted rules, and identified patterns.
        """
        system_prompt = """You are a Principal Software Architect.
Analyze the target file's code, dependency constraints, coding standards, and identified anti-patterns.
Generate a structured Refactoring Plan.

You MUST decide:
1. should_refactor (YES or NO) - decide NO if code is already clean, has low complexity, or if refactoring has too high risk of breaking critical imports.
2. priority (Critical, High, Medium, Low)
3. confidence (0-100%) - estimate how safe the refactoring is based on dependency constraints.
4. steps (a numbered list of step-by-step refactoring actions).

Your output MUST start with a JSON block in the format:
```json
{
  "should_refactor": true/false,
  "priority": "Critical/High/Medium/Low",
  "confidence": 95,
  "reason": "explanation of decision"
}
```
followed by a detailed explanation under '# Refactoring Implementation Plan'."""

        user_prompt = f"""File to analyze: {context['file_name']}

Dependency Constraints:
{context['dependency_narrative']}

Extracted Rules Applied:
{context['rules_applied']}

Patterns / Smells Found:
{context['patterns_identified']}

Target Source Code:
```
{context['original_code'][:8000]}
```

Generate the JSON configuration and the detailed refactoring plan."""

        plan_result = {
            "should_refactor": True,
            "priority": "Medium",
            "confidence": 80,
            "steps_md": "",
            "raw_output": ""
        }

        try:
            raw_output = self.run_prompt(system_prompt, user_prompt, num_predict=1500)
            plan_result["raw_output"] = raw_output
            
            # Parse JSON block
            json_match = re.search(r"```json\s*([\s\S]*?)```", raw_output, re.IGNORECASE)
            if json_match:
                try:
                    meta = json.loads(json_match.group(1).strip())
                    plan_result["should_refactor"] = bool(meta.get("should_refactor", True))
                    plan_result["priority"] = str(meta.get("priority", "Medium"))
                    plan_result["confidence"] = int(meta.get("confidence", 80))
                except Exception:
                    # Fallback matches if JSON parser fails
                    if "should_refactor\": false" in raw_output.lower():
                        plan_result["should_refactor"] = False
            else:
                # Fallback if no json fence found
                if "should_refactor\": false" in raw_output.lower():
                    plan_result["should_refactor"] = False

            # Extract markdown plan section
            plan_header_idx = raw_output.find("# ")
            if plan_header_idx != -1:
                plan_result["steps_md"] = raw_output[plan_header_idx:].strip()
            else:
                # Strip json code block if present
                clean_md = re.sub(r"```json[\s\S]*?```", "", raw_output).strip()
                plan_result["steps_md"] = f"# Refactoring Plan\n\n{clean_md}"
                
            return plan_result
            
        except Exception as e:
            plan_result["steps_md"] = f"# Refactoring Plan\n\nFailed to generate plan: {str(e)}"
            return plan_result
