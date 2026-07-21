from typing import Dict, Any, List

class PlannerAgent:
    """
    Planner Agent: Formulates structured refactoring task lists (Task 1, Task 2, Task 3...)
    deterministically from AST scanning, dependency graphs, and code smell analysis
    before invoking the LLM transformation passes.
    """

    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name

    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formulates a 5-pass refactoring plan containing explicit numbered tasks.
        """
        patterns = context.get("patterns_list", [])
        file_name = context.get("file_name", "unknown")

        should_refactor = True

        # Determine priority
        if "Circular Imports" in patterns or "Large Component" in patterns:
            priority = "Critical"
        elif len(patterns) >= 3:
            priority = "High"
        elif len(patterns) >= 1:
            priority = "Medium"
        else:
            priority = "Low"

        # Determine confidence score
        dep_narrative = context.get("dependency_narrative", "")
        imported_count = dep_narrative.count("Imported by local workspace")

        confidence = 95
        if "Circular Imports" in patterns:
            confidence -= 15
        confidence -= min(30, imported_count * 5)
        confidence = max(50, confidence)

        # 5-Pass task lists
        pass1_tasks = ["Task 1: Modernize syntax standards and inject explicit type annotations/signatures."]
        pass2_tasks = ["Task 2: Optimize loops, array operations, and React rendering memoization."]
        pass3_tasks = ["Task 3: Refactor variable/function names for readability and extract magic numbers to UPPERCASE constants."]
        pass4_tasks = ["Task 4: Sanitize error handling, avoid bare excepts/catch-alls, and validate input parameters."]
        pass5_tasks = ["Task 5: Format line endings, enforce Single Responsibility Principle, and standardize indents."]

        if "Circular Imports" in patterns:
            pass1_tasks.append("Task 1b: Decouple circular dependencies by isolating shared symbols.")
        if "Long Function" in patterns or "Deep Nesting" in patterns:
            pass5_tasks.append("Task 5b: Collapse deep nesting using early guard clauses and extract helper subroutines.")
        if "Missing Memoization / Performance Optimization" in patterns:
            pass2_tasks.append("Task 2b: Wrap expensive array mappings inside useMemo/useCallback hooks.")
        if "Magic Numbers" in patterns:
            pass3_tasks.append("Task 3b: Move inline numeric literals to module-level named constants.")

        all_tasks = pass1_tasks + pass2_tasks + pass3_tasks + pass4_tasks + pass5_tasks

        # Build steps Markdown
        steps_md = f"# Multi-Pass Refactoring Execution Plan for `{file_name}`\n\n"
        steps_md += "### Task Execution List\n"
        for t in all_tasks:
            steps_md += f"- {t}\n"

        steps_md += "\n### Pass Breakdown\n"
        steps_md += "**Pass 1 (Modernization):** " + " ".join(pass1_tasks) + "\n"
        steps_md += "**Pass 2 (Performance):** " + " ".join(pass2_tasks) + "\n"
        steps_md += "**Pass 3 (Naming):** " + " ".join(pass3_tasks) + "\n"
        steps_md += "**Pass 4 (Security):** " + " ".join(pass4_tasks) + "\n"
        steps_md += "**Pass 5 (Formatting):** " + " ".join(pass5_tasks) + "\n"

        return {
            "should_refactor": should_refactor,
            "priority": priority,
            "confidence": confidence,
            "task_list": all_tasks,
            "steps_md": steps_md,
            "steps": {
                "pass1": pass1_tasks,
                "pass2": pass2_tasks,
                "pass3": pass3_tasks,
                "pass4": pass4_tasks,
                "pass5": pass5_tasks
            },
            "raw_output": steps_md
        }
