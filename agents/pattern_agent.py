import re
from typing import Dict, Any, List

class PatternRetrievalAgent:
    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name
        self.common_smells = {
            "God Class / Overly Complex Component": "The module is overly large and complex, violating SRP.",
            "Long Method / Function": "Contains methods or functions exceeding 40 lines of logic.",
            "Nested condition blocks (Deep Nesting)": "Indentation levels go deeper than 12 spaces, harming readability.",
            "Magic Numbers / Hardcoded constants": "Undeclared numeric literals found inside business logic.",
            "Lack of Type Annotations / Type Safety": "Function parameters or return signatures are missing explicit types.",
            "Improper Exception Handling (Silent Exceptions)": "Empty catch blocks or silent exception catching detected.",
            "Missing Memoization / Performance Optimization": "Heavy loop operations or array mapping found in React component without useMemo/useCallback."
        }

    def identify_patterns(self, file_name: str, code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scans code programmatically using AST metadata and regex metrics to detect
        anti-patterns and code smells without querying the LLM.
        """
        matched_patterns = []
        details = []

        # 1. God Class / Complex Component Check
        lines = code.split("\n")
        total_lines = len(lines)
        complexity = metadata.get("complexity_estimate", 1)
        
        is_react = ".jsx" in file_name.lower() or ".tsx" in file_name.lower() or "react" in code.lower()
        
        if total_lines > 200 or complexity > 15:
            matched_patterns.append("God Class / Overly Complex Component")
            details.append(
                f"- **God Class / Overly Complex Component**: File is {total_lines} lines long with a complexity rating of `{complexity}`. "
                "Recommend breaking it down along functional boundaries."
            )

        # 2. Long Method Check
        func_found_long = False
        for func in metadata.get("functions", []):
            # If function name is known, try to isolate its body lines
            fname = func.get("name", "")
            func_lines = 0
            in_func = False
            for line in lines:
                if re.search(r"\b" + re.escape(fname) + r"\b", line):
                    in_func = True
                if in_func:
                    func_lines += 1
                    if line.strip() == "" and func_lines > 10: # Rough approximation
                        pass
            # If standalone function is long
            # Or programmatically scan code for def/function block line length
            pass
            
        # Programmatic count of function blocks line count
        long_blocks = 0
        indent_level = -1
        current_block_lines = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if stripped.startswith(("def ", "function ", "const ", "class ")):
                if current_block_lines > 40:
                    long_blocks += 1
                current_block_lines = 1
            elif indent > 0:
                current_block_lines += 1
        if current_block_lines > 40:
            long_blocks += 1
            
        if long_blocks > 0:
            matched_patterns.append("Long Method / Function")
            details.append(f"- **Long Method / Function**: Detected {long_blocks} method(s)/block(s) exceeding 40 active logic lines.")

        # 3. Deep Nesting Check
        deep_lines = [i + 1 for i, line in enumerate(lines) if len(line) - len(line.lstrip()) >= 12 and line.strip().startswith(("if", "for", "while", "switch", "catch"))]
        if deep_lines:
            matched_patterns.append("Nested condition blocks (Deep Nesting)")
            details.append(f"- **Nested condition blocks (Deep Nesting)**: Indentation goes beyond 12 spaces at line(s): {deep_lines[:4]}.")

        # 4. Magic Numbers Check
        # Exclude common index 0, 1, and version/ports patterns
        magic_matches = re.finditer(r"(?<![a-zA-Z0-9_])(?<!Line )(?<!\d\.)[2-9]\d*(?![a-zA-Z0-9_])", code)
        magic_numbers = []
        for m in magic_matches:
            num = m.group(0)
            # Filter line index numbers or common formats
            if int(num) not in (80, 443, 8080, 3000): # Allow ports
                magic_numbers.append((num, code.count(f"\n") - code[m.start():].count(f"\n") + 1))
                
        if len(magic_numbers) > 2:
            matched_patterns.append("Magic Numbers / Hardcoded constants")
            details.append(f"- **Magic Numbers**: Hardcoded values like {list(set([n[0] for n in magic_numbers[:4]]))} found directly in logic.")

        # 5. Type Annotations Check
        has_any_types = ":" in code or "as " in code or "type " in code or "interface " in code
        is_ts = ".ts" in file_name.lower() or ".tsx" in file_name.lower()
        if (is_ts or ".py" in file_name.lower()) and not has_any_types:
            matched_patterns.append("Lack of Type Annotations / Type Safety")
            details.append("- **Lack of Type Annotations / Type Safety**: Function arguments and return types lack annotations.")

        # 6. Silent Exceptions Check
        silent_excepts = re.findall(r"except\s*:\s*pass|except\s+Exception\s*:\s*pass|catch\s*\([^)]*\)\s*\{\s*\}", code.replace(" ", ""))
        if silent_excepts:
            matched_patterns.append("Improper Exception Handling (Silent Exceptions)")
            details.append("- **Improper Exception Handling**: Empty or silent exception/catch clauses detected.")

        # 7. React Memoization Check
        if is_react and (".map" in code or ".filter" in code) and not ("useMemo" in code or "useCallback" in code):
            matched_patterns.append("Missing Memoization / Performance Optimization")
            details.append("- **Missing Memoization**: Heavy list mappings (.map) in React component without useMemo hook optimization.")

        # Construct Report Markdown
        if not matched_patterns:
            report_md = "# Identified Patterns\n\n✓ **Clean Codebase**: No structural code smells or anti-patterns detected statically."
        else:
            report_md = "# Identified Patterns\n\n" + "\n".join(details)

        return {
            "patterns": matched_patterns if matched_patterns else ["Code Quality Opportunity"],
            "report_md": report_md
        }
