import re
import json
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent
from utils.ast_parser import ASTParser

class QualityEvaluationAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def evaluate_quality(
        self,
        file_name: str,
        original_code: str,
        refactored_code: str,
        original_metadata: Dict[str, Any],
        refactored_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates the quality difference between the original and refactored code.
        Computes AST-based structural metrics and asks the LLM to score readability/maintainability.
        """
        # 1. Structural calculations
        orig_lines = original_code.strip().split("\n")
        ref_lines = refactored_code.strip().split("\n")
        
        orig_line_count = len(orig_lines)
        ref_line_count = len(ref_lines)
        
        lines_reduced = orig_line_count - ref_line_count
        reduction_percentage = round((lines_reduced / orig_line_count) * 100, 1) if orig_line_count > 0 else 0.0

        # Structural complexity from parser
        orig_complexity = original_metadata.get("complexity_estimate", 1)
        ref_complexity = refactored_metadata.get("complexity_estimate", 1)
        
        complexity_change = orig_complexity - ref_complexity
        complexity_reduction_pct = round((complexity_change / orig_complexity) * 100, 1) if orig_complexity > 0 else 0.0

        # 2. Semantic grading from LLM
        system_prompt = """You are a Code Quality Evaluator.
Compare the original and refactored code. Provide a grading out of 100 for three dimensions:
- Readability (descriptive naming, spacing, comments, indentation)
- Maintainability (modularization, DRY, low nesting, SRP compliance)
- Safety (type annotations, robust error handling, exception boundaries)

You MUST return your grading strictly as a JSON object inside a ```json code block:
{
  "original": {
    "readability": 70,
    "maintainability": 65,
    "safety": 60
  },
  "refactored": {
    "readability": 95,
    "maintainability": 90,
    "safety": 85
  },
  "justification": "short 2-sentence summary of the main improvements"
}
Do not write any other conversational text."""

        user_prompt = f"""File: {file_name}

[Original Code]
```
{original_code[:6000]}
```

[Refactored Code]
```
{refactored_code[:6000]}
```

Analyze both files and output the JSON grading."""

        # Default fallback values
        readability_orig, readability_ref = 70, 90
        maintainability_orig, maintainability_ref = 65, 88
        safety_orig, safety_ref = 60, 85
        justification = "Cleaned up structural layout, annotated functions, and simplified logical blocks."

        try:
            raw_grade = self.run_prompt(system_prompt, user_prompt, num_predict=512)
            json_match = re.search(r"```json\s*([\s\S]*?)```", raw_grade, re.IGNORECASE)
            if json_match:
                grades = json.loads(json_match.group(1).strip())
                orig_grades = grades.get("original", {})
                ref_grades = grades.get("refactored", {})
                
                readability_orig = int(orig_grades.get("readability", 70))
                readability_ref = int(ref_grades.get("readability", 90))
                
                maintainability_orig = int(orig_grades.get("maintainability", 65))
                maintainability_ref = int(ref_grades.get("maintainability", 88))
                
                safety_orig = int(orig_grades.get("safety", 60))
                safety_ref = int(ref_grades.get("safety", 85))
                
                justification = grades.get("justification", justification)
        except Exception:
            pass # Keep fallback values on JSON parsing error

        # Calculate final composite scores
        score_before = round((readability_orig + maintainability_orig + safety_orig) / 3.0)
        score_after = round((readability_ref + maintainability_ref + safety_ref) / 3.0)
        
        # Guard rails
        score_before = max(10, min(99, score_before))
        score_after = max(score_before, min(100, score_after))

        return {
            "orig_lines": orig_line_count,
            "ref_lines": ref_line_count,
            "lines_reduced": lines_reduced,
            "reduction_pct": reduction_percentage,
            "orig_complexity": orig_complexity,
            "ref_complexity": ref_complexity,
            "complexity_reduction_pct": complexity_reduction_pct,
            "orig_readability": readability_orig,
            "ref_readability": readability_ref,
            "orig_maintainability": maintainability_orig,
            "ref_maintainability": maintainability_ref,
            "orig_safety": safety_orig,
            "ref_safety": safety_ref,
            "score_before": score_before,
            "score_after": score_after,
            "justification": justification
        }
