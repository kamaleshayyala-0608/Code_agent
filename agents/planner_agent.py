from typing import Dict, Any

class PlannerAgent:
    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name

    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formulates a multi-pass refactoring plan locally based on statically
        scanned code smells and dependency graphs, bypassing the LLM.
        """
        patterns = context.get("patterns_list", [])
        file_name = context.get("file_name", "unknown")
        
        # Determine if refactoring is required
        clean_code = len(patterns) == 0 or (len(patterns) == 1 and patterns[0] == "Code Quality Opportunity")
        should_refactor = not clean_code
        
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
        # Base is 95, penalize for circular dependencies and high caller coupling
        dep_narrative = context.get("dependency_narrative", "")
        imported_count = dep_narrative.count("Imported by local workspace")
        
        confidence = 95
        if "Circular Imports" in patterns:
            confidence -= 15
        confidence -= min(30, imported_count * 5)
        confidence = max(50, confidence)

        # Multi-Pass steps mapping
        pass1_steps = [] # Safety & Types / Imports
        pass2_steps = [] # Code Smells
        pass3_steps = [] # Performance / Memoization
        pass4_steps = [] # Architecture / SRP

        if "Lack of Type Annotations / Type Safety" in patterns:
            pass1_steps.append("- Inject proper type definitions (e.g. PEP 484 type annotations or TypeScript interfaces) for function arguments and return blocks.")
        if "Circular Imports" in patterns:
            pass1_steps.append("- Decouple imported circular dependencies by moving shared symbols into a separate utility module.")
            
        if "Long Function" in patterns:
            pass2_steps.append("- Extract nested block statements inside long functions into standalone module-level helper routines.")
        if "Deep Nesting" in patterns:
            pass2_steps.append("- Collapsed deeply nested logic branches using guard clauses (early exit checks).")
        if "Magic Numbers" in patterns:
            pass2_steps.append("- Gather and extract unmapped numeric literals, replacing them with named constants at the module header.")

        if "Missing Memoization / Performance Optimization" in patterns:
            pass3_steps.append("- Wrap inline list `.map` / `.filter` structures inside React's `useMemo` hooks to prevent redundant rendering.")
        if "Duplicate Code" in patterns:
            pass3_steps.append("- Consolidate duplicated/repeated logic nodes into helper functions to enforce DRY principles.")

        if "Large Component" in patterns:
            pass4_steps.append("- Reorganize the large monolithic file internally: group related logic into clearly separated, single-responsibility sections/functions/sub-components WITHIN this same file. Do not attempt to output separate files — this pipeline only accepts one returned file per input file.")

        # If clean, add default pass formatting
        if not pass1_steps and not pass2_steps and not pass3_steps and not pass4_steps:
            pass1_steps.append("- Standardize import formatting and spacing.")

        # Build steps Markdown
        steps_md = "# Multi-Pass Refactoring Implementation Plan\n\n"
        steps_md += f"We have structured a 4-pass refactoring pipeline for `{file_name}` to safely isolate changes.\n\n"
        
        if pass1_steps:
            steps_md += "### Pass 1: Safety & Imports\n"
            for step in pass1_steps:
                steps_md += f"{step}\n"
        if pass2_steps:
            steps_md += "\n### Pass 2: Structural Code Smells\n"
            for step in pass2_steps:
                steps_md += f"{step}\n"
        if pass3_steps:
            steps_md += "\n### Pass 3: Performance & Optimization\n"
            for step in pass3_steps:
                steps_md += f"{step}\n"
        if pass4_steps:
            steps_md += "\n### Pass 4: Clean Architecture (SRP)\n"
            for step in pass4_steps:
                steps_md += f"{step}\n"

        return {
            "should_refactor": should_refactor,
            "priority": priority,
            "confidence": confidence,
            "steps_md": steps_md,
            "steps": {
                "pass1": pass1_steps,
                "pass2": pass2_steps,
                "pass3": pass3_steps,
                "pass4": pass4_steps
            },
            "raw_output": steps_md
        }
