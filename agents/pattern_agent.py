import re
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class PatternRetrievalAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)
        # Predefined common smells we want to match
        self.common_smells = [
            "Long Method / Function",
            "Duplicate Logic",
            "Nested condition blocks (Deep Nesting)",
            "Magic Numbers / Hardcoded constants",
            "God Class / Overly Complex Component",
            "Long Parameter List",
            "Unused Imports / Variables",
            "Repeated API Calls / Missing Cache",
            "Lack of Type Annotations / Type Safety",
            "Improper Exception Handling (Silent Exceptions)"
        ]

    def identify_patterns(self, file_name: str, code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs analysis to identify anti-patterns and code smells in the target file.
        Returns a dict containing a list of matched pattern strings and a markdown description.
        """
        # Static checks from AST/Regex
        static_patterns = []
        
        # Check function length
        for func in metadata.get("functions", []):
            # Fallback estimation if lines of code isn't directly in AST
            pass
            
        # Hardcoded constants check (simple regex for magic numbers)
        magic_numbers = re.findall(r"(?<![a-zA-Z0-9_])(?<!Line )(?<!\d\.)[2-9]\d*(?![a-zA-Z0-9_])", code)
        if len(magic_numbers) > 5:
            static_patterns.append("Magic Numbers / Hardcoded constants")
            
        # Nested conditions check (regex for deep indentation)
        deep_nesting = re.findall(r"^[ \t]{12,}(?:if|for|while)\b", code, re.MULTILINE)
        if deep_nesting:
            static_patterns.append("Nested condition blocks (Deep Nesting)")

        # Formulate query to LLM to run semantic pattern check
        system_prompt = f"""You are a Code Quality Reviewer.
Analyze the target code and identify code smells, structural issues, or anti-patterns from this catalog:
{self.common_smells}

List only the patterns that are actually present in the file, with a very brief explanation (1-2 sentences each) of where they occur and why they are smells.
Return your response in Markdown with a clear header '# Identified Patterns'."""

        user_prompt = f"""File: {file_name}

Code:
```
{code[:8000]}
```

Provide the analysis report."""

        try:
            raw_report = self.run_prompt(system_prompt, user_prompt, num_predict=1000)
            
            # Extract bullet points/patterns from LLM response
            matched_patterns = list(static_patterns)
            for smell in self.common_smells:
                # If LLM mentions the smell name in lowercase or matches parts of it
                short_name = smell.split(" / ")[0].split(" (")[0].lower()
                if short_name in raw_report.lower() and smell not in matched_patterns:
                    matched_patterns.append(smell)
                    
            return {
                "patterns": matched_patterns,
                "report_md": raw_report
            }
        except Exception as e:
            return {
                "patterns": static_patterns or ["Code Quality Opportunity"],
                "report_md": f"# Identified Patterns\n\nFailed to query LLM for patterns: {str(e)}\n\n- Detected statically: {static_patterns}"
            }
