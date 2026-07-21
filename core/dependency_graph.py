import os
from typing import Dict, Any, List, Set
from utils.dependency_analyzer import DependencyAnalyzer

class DependencyGraph:
    """
    Constructs a structural project graph mapping:
    Component -> Imports -> Uses -> Exports -> References -> Callers
    Detects circular dependencies and symbol-level connections across the workspace.
    """

    def __init__(self):
        self.graph: Dict[str, Any] = {}

    def generate_graph(self, files: Dict[str, str], parsed_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a comprehensive dependency graph mapping files, components, imports,
        exports, symbol usages, and circular references.
        """
        raw_deps = DependencyAnalyzer.analyze_workspace_dependencies(files, parsed_metadata)

        for fname, meta in parsed_metadata.items():
            file_deps = raw_deps.get(fname, {})

            # Extract exports
            exports = []
            for cls in meta.get("classes", []):
                if cls.get("name"):
                    exports.append(f"class:{cls['name']}")
            for fn in meta.get("functions", []):
                if fn.get("name") and not fn["name"].startswith("_"):
                    exports.append(f"function:{fn['name']}")
            for interf in meta.get("interfaces", []):
                if interf.get("name"):
                    exports.append(f"interface:{interf['name']}")

            # Extract uses (symbols referenced defined elsewhere)
            uses = []
            for sym_conn in file_deps.get("symbol_dependencies", []):
                uses.append(f"{sym_conn['symbol']} (from {sym_conn['defined_in']})")

            self.graph[fname] = {
                "file_name": fname,
                "imports": meta.get("imports", []),
                "exports": exports,
                "uses": uses,
                "references": meta.get("symbol_references", []),
                "depends_on": file_deps.get("depends_on", []),
                "depended_on_by": file_deps.get("depended_on_by", []),
                "symbol_dependencies": file_deps.get("symbol_dependencies", []),
                "circular_loops": file_deps.get("circular_loops", []),
                "is_in_circular_loop": file_deps.get("is_in_circular_loop", False)
            }

        return self.graph

    def get_file_dependencies(self, fname: str) -> Dict[str, Any]:
        return self.graph.get(fname, {
            "file_name": fname,
            "imports": [],
            "exports": [],
            "uses": [],
            "references": [],
            "depends_on": [],
            "depended_on_by": [],
            "symbol_dependencies": [],
            "circular_loops": [],
            "is_in_circular_loop": False
        })
