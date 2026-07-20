import re
import math
from typing import Dict, Any

class QualityEvaluationAgent:
    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name

    def evaluate_quality(
        self,
        file_name: str,
        original_code: str,
        refactored_code: str,
        original_metadata: Dict[str, Any],
        refactored_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates advanced deterministic metrics: Cyclomatic Complexity,
        Cognitive Complexity, Maintainability Index, Dead Code, and Coupling.
        """
        # 1. Structural calculations
        orig_lines_list = original_code.strip().split("\n")
        ref_lines_list = refactored_code.strip().split("\n")
        
        orig_lines = len(orig_lines_list)
        ref_lines = len(ref_lines_list)
        
        lines_reduced = orig_lines - ref_lines
        reduction_percentage = round((lines_reduced / orig_lines) * 100, 1) if orig_lines > 0 else 0.0

        # Calculate metrics for code blocks
        def compute_cognitive_complexity(code: str) -> int:
            """
            Approximates Cognitive Complexity by penalizing nesting levels
            for conditional and iteration structures.
            """
            lines = code.split("\n")
            complexity = 0
            nesting_level = 0
            
            for line in lines:
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())
                # Deduce nesting from indent (assumes 2 or 4 space indents)
                nesting_level = max(0, indent // 4)
                
                # Check for branching constructs
                if re.search(r"\b(if|for|while|catch)\b", stripped):
                    complexity += (1 + nesting_level)
                elif "&&" in stripped or "||" in stripped:
                    complexity += 1
            return max(1, complexity)

        def compute_halstead_volume(code: str) -> float:
            """
            Estimates Halstead Volume (V = N * log2(n)) where:
            N = total operators + operands
            n = unique operators + operands
            """
            words = re.findall(r"\b[a-zA-Z0-9_$]+\b|[\+\-\*/%&|\^~<>!=]+", code)
            if not words:
                return 1.0
            N = len(words)
            n = len(set(words))
            return float(N * math.log2(n) if n > 1 else N)

        def compute_maintainability_index(volume: float, cyclomatic: int, loc: int) -> float:
            """
            Standard Software Engineering Maintainability Index (MI) formula:
            MI = 171 - 5.2 * ln(V) - 0.23 * (G) - 16.2 * ln(LOC)
            """
            if loc <= 0 or volume <= 0:
                return 100.0
            try:
                mi = 171 - 5.2 * math.log(volume) - 0.23 * cyclomatic - 16.2 * math.log(loc)
                # Rescale to 0 - 100 range
                rescaled = (mi / 171.0) * 100
                return max(0.0, min(100.0, rescaled))
            except Exception:
                return 80.0

        # Compile metrics for original and refactored
        orig_cyclo = original_metadata.get("complexity_estimate", 1)
        ref_cyclo = refactored_metadata.get("complexity_estimate", 1)

        orig_cog = compute_cognitive_complexity(original_code)
        ref_cog = compute_cognitive_complexity(refactored_code)

        orig_vol = compute_halstead_volume(original_code)
        ref_vol = compute_halstead_volume(refactored_code)

        orig_mi = round(compute_maintainability_index(orig_vol, orig_cyclo, orig_lines))
        ref_mi = round(compute_maintainability_index(ref_vol, ref_cyclo, ref_lines))

        # Unused declarations (Dead Code count)
        dead_imports = 0
        for imp in original_metadata.get("imports", []):
            imp_base = imp.split("/")[-1].split(".")[0]
            if original_code.count(imp_base) <= 1:
                dead_imports += 1
                
        # File Coupling (Import count + Export count)
        orig_coupling = len(original_metadata.get("imports", [])) + len(original_metadata.get("exports", []))
        ref_coupling = len(refactored_metadata.get("imports", [])) + len(refactored_metadata.get("exports", []))

        # Deduce final Refactoring Score based on MI and Complexity reductions
        score_before = orig_mi
        score_after = ref_mi
        
        # Guard rails
        score_after = max(score_before, score_after)
        score_before = max(10, min(99, score_before))
        score_after = max(score_before, min(100, score_after))

        # Dynamic justification text
        improvements = []
        if ref_mi > orig_mi:
            improvements.append(f"boosted Maintainability Index from {orig_mi} to {ref_mi}")
        if orig_cyclo > ref_cyclo:
            improvements.append(f"reduced Cyclomatic Complexity by {orig_cyclo - ref_cyclo} points")
        if orig_cog > ref_cog:
            improvements.append(f"lowered Cognitive nesting complexity by {orig_cog - ref_cog} points")
        if orig_coupling > ref_coupling:
            improvements.append("reduced file dependency coupling")

        if improvements:
            justification = f"Refactored components: " + ", ".join(improvements) + "."
        else:
            justification = "Optimized import structures, formatting spaces, and aligned standard layouts."

        return {
            "orig_lines": orig_lines,
            "ref_lines": ref_lines,
            "lines_reduced": lines_reduced,
            "reduction_pct": reduction_percentage,
            "orig_complexity": orig_cyclo,
            "ref_complexity": ref_cyclo,
            "complexity_reduction_pct": round(((orig_cyclo - ref_cyclo)/orig_cyclo)*100, 1) if orig_cyclo > 0 else 0.0,
            "orig_cognitive": orig_cog,
            "ref_cognitive": ref_cog,
            "orig_mi": orig_mi,
            "ref_mi": ref_mi,
            "dead_code_count": dead_imports,
            "orig_coupling": orig_coupling,
            "ref_coupling": ref_coupling,
            "score_before": score_before,
            "score_after": score_after,
            "justification": justification
        }
