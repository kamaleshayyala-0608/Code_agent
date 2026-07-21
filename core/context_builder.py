import os
from typing import Dict, Any, List

class ContextBuilder:
    """
    Builds rich project-aware context for refactoring operations.
    Combines AST metadata, workspace dependency graphs, nearby components,
    shared hooks, and persistent project memory into a unified context object.
    """

    @staticmethod
    def build_project_context(
        files: Dict[str, str],
        parsed_metadata: Dict[str, Dict[str, Any]],
        dependency_graph: Dict[str, Any],
        symbol_index: Dict[str, Any],
        project_memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds overall workspace-level context before individual file refactoring starts.
        """
        shared_hooks = []
        shared_utilities = []

        for fname, meta in parsed_metadata.items():
            hooks = meta.get("hooks", [])
            for h in hooks:
                if h not in shared_hooks:
                    shared_hooks.append(h)
            for func in meta.get("functions", []):
                name = func.get("name", "")
                if name.startswith("use") and name not in shared_hooks:
                    shared_hooks.append(name)
                elif name and not name.startswith("_") and name not in shared_utilities:
                    shared_utilities.append(name)

        return {
            "total_files": len(files),
            "file_list": list(files.keys()),
            "shared_hooks": shared_hooks,
            "shared_utilities": shared_utilities,
            "dependency_graph": dependency_graph,
            "symbol_index": symbol_index,
            "project_memory": project_memory
        }

    @staticmethod
    def build_file_context(
        file_name: str,
        code: str,
        project_context: Dict[str, Any],
        parsed_metadata: Dict[str, Dict[str, Any]],
        rules_md: str = "",
        patterns_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Builds detailed context for a specific target file:
        File -> Imports -> Local Dependencies -> Nearby Components -> Shared Hooks -> Memory Rules
        """
        if patterns_data is None:
            patterns_data = {}

        file_meta = parsed_metadata.get(file_name, {})
        dep_graph = project_context.get("dependency_graph", {}).get(file_name, {})

        depends_on = dep_graph.get("depends_on", [])
        depended_on_by = dep_graph.get("depended_on_by", [])
        symbol_deps = dep_graph.get("symbol_dependencies", [])

        # Identify nearby components in same directory
        file_dir = os.path.dirname(file_name.replace("\\", "/"))
        nearby_components = [
            f for f in project_context.get("file_list", [])
            if f != file_name and os.path.dirname(f.replace("\\", "/")) == file_dir
        ]

        # Detect local hooks used in this file
        file_hooks = file_meta.get("hooks", [])
        shared_hooks_used = [h for h in project_context.get("shared_hooks", []) if h in file_hooks or h in code]

        # Build narrative
        dep_narrative = ""
        if depends_on:
            dep_narrative += f"- Depends on local workspace files: {', '.join(depends_on)}\n"
        if depended_on_by:
            dep_narrative += f"- Imported by local workspace files: {', '.join(depended_on_by)}\n"
            dep_narrative += "  CRITICAL: Preserve class names, public functions, exported interfaces, and module signatures to avoid breaking consumers.\n"
        if nearby_components:
            dep_narrative += f"- Nearby components in same directory: {', '.join(nearby_components[:5])}\n"
        if shared_hooks_used:
            dep_narrative += f"- Shared hooks detected: {', '.join(shared_hooks_used)}\n"
        if not depends_on and not depended_on_by:
            dep_narrative += "- Isolated file (no workspace dependencies detected).\n"

        return {
            "file_name": file_name,
            "original_code": code,
            "file_metadata": file_meta,
            "depends_on": depends_on,
            "depended_on_by": depended_on_by,
            "symbol_dependencies": symbol_deps,
            "nearby_components": nearby_components,
            "shared_hooks": shared_hooks_used,
            "dependency_narrative": dep_narrative,
            "rules_applied": rules_md or project_context.get("project_memory", {}).get("spec_md", ""),
            "patterns_identified": patterns_data.get("report_md", ""),
            "patterns_list": patterns_data.get("patterns", []),
            "project_context": project_context
        }
