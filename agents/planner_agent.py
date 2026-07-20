from typing import Dict, Any

class PlannerAgent:
    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name

    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a structured refactoring plan locally based on the detected
        patterns and dependency constraints, bypassing the LLM.
        """
        patterns = context.get("patterns_list", [])
        file_name = context.get("file_name", "unknown")
        
        # Decide if refactoring is required
        # If the only pattern is the default fallback, skip refactoring
        clean_code = len(patterns) == 0 or (len(patterns) == 1 and patterns[0] == "Code Quality Opportunity")
        should_refactor = not clean_code
        
        # Determine priority
        if "God Class / Overly Complex Component" in patterns:
            priority = "Critical"
        elif len(patterns) >= 3:
            priority = "High"
        elif len(patterns) >= 1:
            priority = "Medium"
        else:
            priority = "Low"

        # Determine confidence based on dependency graph
        # More files importing this file -> higher risk, lower confidence score
        dep_narrative = context.get("dependency_narrative", "")
        # Extract imports counting by parsing narrative or metadata
        imported_count = dep_narrative.count("Imported by local workspace files")
        confidence = max(50, 95 - (imported_count * 8))

        # Build step-by-step instructions based on detected patterns
        steps = []
        step_num = 1
        
        if "God Class / Overly Complex Component" in patterns:
            steps.append(f"{step_num}. **Split Complex Structure**: Break down the large component or class into modular, single-responsibility files or classes.")
            step_num += 1
            
        if "Long Method / Function" in patterns:
            steps.append(f"{step_num}. **Extract Functions**: Identify active code blocks inside methods exceeding 40 lines and extract them into separate standalone helper functions.")
            step_num += 1
            
        if "Nested condition blocks (Deep Nesting)" in patterns:
            steps.append(f"{step_num}. **Simplify Conditionals**: Use guard clauses (early returns) to flatten deeply nested `if/else` condition blocks.")
            step_num += 1
            
        if "Magic Numbers / Hardcoded constants" in patterns:
            steps.append(f"{step_num}. **Extract Constants**: Gather all inline raw numbers/strings and declare them as named UPPER_CASE constants at the top of the file.")
            step_num += 1
            
        if "Lack of Type Annotations / Type Safety" in patterns:
            steps.append(f"{step_num}. **Add Type System Definitions**: Declare explicit type annotations for parameter arguments, variables, and function return values.")
            step_num += 1
            
        if "Improper Exception Handling (Silent Exceptions)" in patterns:
            steps.append(f"{step_num}. **Enhance Exception Handling**: Stop catching generic exceptions silently. Add logging or throw errors back to callers.")
            step_num += 1
            
        if "Missing Memoization / Performance Optimization" in patterns:
            steps.append(f"{step_num}. **Memoize React Mappings**: Wrap array mappings, operations, and child callbacks inside React's `useMemo` and `useCallback` hooks.")
            step_num += 1

        if not steps:
            steps.append("1. **Minor formatting**: Apply minor import sorting and code spacing improvements.")

        # Build steps Markdown
        steps_md = "# Refactoring Implementation Plan\n\n"
        steps_md += f"We have formulated a customized refactoring strategy for `{file_name}` targeting the resolved code smells.\n\n"
        steps_md += "### Implementation Steps:\n"
        for step in steps:
            steps_md += f"- {step}\n"
            
        steps_md += f"\n### Verification Constraints:\n"
        steps_md += f"- **Target Interface Preservation**: {dep_narrative.strip()}\n"

        return {
            "should_refactor": should_refactor,
            "priority": priority,
            "confidence": confidence,
            "steps_md": steps_md,
            "steps": [s.split("**")[1].split("**")[0] for s in steps if "**" in s], # String list of step names
            "raw_output": steps_md
        }
