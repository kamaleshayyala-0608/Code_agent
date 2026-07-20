from typing import Dict, Any, List

class ContextBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_refactoring_context(
        file_name: str, 
        code: str, 
        metadata: Dict[str, Any], 
        rules_md: str, 
        patterns_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes AST structures, dependencies, matched rules, and code smells
        into a structured context context.
        """
        dep_ctx = metadata.get("dependencies_context", {"depends_on": [], "depended_on_by": []})
        depends_on = dep_ctx.get("depends_on", [])
        depended_on_by = dep_ctx.get("depended_on_by", [])
        
        # Build dependency narrative
        dep_narrative = ""
        if depends_on:
            dep_narrative += f"- Depends on local workspace files: {', '.join(depends_on)}\n"
        if depended_on_by:
            dep_narrative += f"- Imported by local workspace files: {', '.join(depended_on_by)}\n"
            dep_narrative += "  CRITICAL: Do not change class names, public function signatures, or import paths to avoid breaking these consumers.\n"
        if not depends_on and not depended_on_by:
            dep_narrative += "- Isolated file (no workspace dependencies detected).\n"

        context = {
            "file_name": file_name,
            "original_code": code,
            "dependency_narrative": dep_narrative,
            "rules_applied": rules_md,
            "patterns_identified": patterns_data.get("report_md", ""),
            "patterns_list": patterns_data.get("patterns", [])
        }
        
        return context
