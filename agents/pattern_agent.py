import os
import re
from typing import Dict, Any, List

class PatternRetrievalAgent:
    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name

    def identify_patterns(self, file_name: str, code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statically scans AST metadata and file contents to programmatically detect
        anti-patterns (Dead Code, Circular Imports, God Class, Nesting, etc.),
        bypassing the LLM.
        """
        matched_patterns = []
        details = []

        lines = code.split("\n")
        total_lines = len(lines)
        complexity = metadata.get("complexity_estimate", 1)
        dep_ctx = metadata.get("dependencies_context", {"depends_on": [], "depended_on_by": [], "is_in_circular_loop": False, "circular_loops": []})

        # 1. Large Component / God Class
        if total_lines > 150:
            matched_patterns.append("Large Component")
            details.append(f"- **Large Component**: File is {total_lines} lines long. Propose splitting it into smaller utility segments.")
        
        # 2. High Complexity
        if complexity > 10:
            matched_patterns.append("High Complexity")
            details.append(f"- **High Complexity**: Cyclomatic complexity count of `{complexity}` detected in branching structures.")

        # 3. Long Function
        long_functions_list = []
        # Approximate function bodies by matching line scopes
        current_fn = None
        fn_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "function ", "const ", "let ")) and ("=>" in line or "function" in line or "def" in line):
                if current_fn and fn_lines > 45:
                    long_functions_list.append(current_fn)
                fn_match = re.search(r"\b([a-zA-Z0-9_$]+)\b", stripped.replace("def", "").replace("function", ""))
                current_fn = fn_match.group(1) if fn_match else "unknown"
                fn_lines = 1
            elif current_fn:
                fn_lines += 1
        if current_fn and fn_lines > 45:
            long_functions_list.append(current_fn)

        if long_functions_list:
            matched_patterns.append("Long Function")
            details.append(f"- **Long Function**: Standalone function(s) {list(set(long_functions_list))} exceed 40-50 lines of active logic.")

        # 4. Deep Nesting
        deep_nesting_lines = []
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent >= 12 and stripped.startswith(("if", "for", "while", "switch", "catch")):
                deep_nesting_lines.append(idx + 1)
        if deep_nesting_lines:
            matched_patterns.append("Deep Nesting")
            details.append(f"- **Deep Nesting**: Nested statements exceed 3 logic blocks at line(s): {deep_nesting_lines[:3]}.")

        # 5. Magic Numbers
        magic_numbers = []
        # Find raw integers in assignments/operations, exclude version/index values
        num_matches = re.finditer(r"(?<![a-zA-Z0-9_])[2-9]\d*(?![a-zA-Z0-9_])", code)
        for m in num_matches:
            num = m.group(0)
            if int(num) not in (80, 443, 8080, 3000):
                magic_numbers.append(num)
        if len(magic_numbers) > 2:
            matched_patterns.append("Magic Numbers")
            details.append(f"- **Magic Numbers**: Unmapped numerical literals like {list(set(magic_numbers[:3]))} found in logic.")

        # 6. Dead Code (Unused imports or variables)
        dead_imports = []
        for imp in metadata.get("imports", []):
            imp_base = imp.split("/")[-1].split(".")[0]
            # Verify if import symbol is referenced anywhere else in the code
            if code.count(imp_base) <= 1:
                dead_imports.append(imp)
        if dead_imports:
            matched_patterns.append("Dead Code")
            details.append(f"- **Dead Code**: Detected unused import declaration modules: {dead_imports}.")

        # 7. Circular Imports
        if dep_ctx.get("is_in_circular_loop", False):
            matched_patterns.append("Circular Imports")
            loops_str = " -> ".join([os.path.basename(f) for f in dep_ctx.get("circular_loops", [[]])[0]])
            details.append(f"- **Circular Imports**: File is bound inside a circular dependency cycle: `{loops_str}`.")

        # 8. Duplicate Code
        # Search for repeating non-trivial lines
        active_lines = [l.strip() for l in lines if len(l.strip()) > 15 and not l.strip().startswith(("#", "//", "import", "from"))]
        duplicates = set([x for x in active_lines if active_lines.count(x) > 2])
        if duplicates:
            matched_patterns.append("Duplicate Code")
            details.append(f"- **Duplicate Code**: Detected {len(duplicates)} duplicate/repeated code line logic blocks.")

        # Final layout
        if not matched_patterns:
            report_md = "# Identified Patterns\n\n✓ **Clean Codebase**: No anti-patterns detected statically."
        else:
            report_md = "# Identified Patterns\n\n" + "\n".join(details)

        return {
            "patterns": matched_patterns if matched_patterns else ["Code Quality Opportunity"],
            "report_md": report_md
        }
