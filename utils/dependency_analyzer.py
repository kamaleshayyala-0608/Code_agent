import os
from typing import Dict, List, Any, Set

class DependencyAnalyzer:
    @staticmethod
    def analyze_workspace_dependencies(files: Dict[str, str], parsed_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes dependencies between files in the workspace.
        Returns a dictionary mapping each file to its local dependencies and dependent files.
        """
        dependency_graph = {}
        depended_on_by = {fname: set() for fname in files.keys()}
        depends_on = {fname: set() for fname in files.keys()}
        
        # Helper to try to match an import string to a workspace file
        def resolve_import_to_file(source_file: str, import_str: str) -> str | None:
            if not import_str:
                return None
                
            # Normalize path separators
            source_dir = os.path.dirname(source_file.replace("\\", "/"))
            import_norm = import_str.replace("\\", "/").strip()
            
            # Case 1: Relative import (e.g. `./utils`, `../services/auth`)
            if import_norm.startswith("."):
                # Construct possible relative paths
                possible_paths = []
                # Remove leading ./ or ../ and compute relative directory
                parts = import_norm.split("/")
                rel_dir = source_dir
                
                start_idx = 0
                for part in parts:
                    if part == ".":
                        start_idx += 1
                        continue
                    elif part == "..":
                        rel_dir = os.path.dirname(rel_dir)
                        start_idx += 1
                        continue
                    break
                    
                sub_path = "/".join(parts[start_idx:])
                target_base = os.path.join(rel_dir, sub_path).replace("\\", "/").lstrip("/")
                
                # Check workspace files that match this target base path
                for fname in files.keys():
                    fname_norm = fname.replace("\\", "/").lstrip("/")
                    # Match exact, or with extension (e.g. target_base + '.ts' matches fname_norm)
                    fname_no_ext, _ = os.path.splitext(fname_norm)
                    if fname_norm == target_base or fname_no_ext == target_base:
                        return fname
                    if fname_norm.startswith(target_base + "/"):
                        # E.g. target_base is folder containing index file
                        if fname_norm.endswith("index.js") or fname_norm.endswith("index.ts") or fname_norm.endswith("index.tsx"):
                            return fname
                            
            # Case 2: Module/absolute style local import (e.g. `generators.cicd_generator` or `utils/vector_db`)
            # Try to match imports by checking if any workspace file name contains or matches the import path
            import_as_path = import_norm.replace(".", "/")
            for fname in files.keys():
                fname_norm = fname.replace("\\", "/").lstrip("/")
                fname_no_ext, _ = os.path.splitext(fname_norm)
                
                if (fname_no_ext == import_norm or 
                    fname_no_ext == import_as_path or 
                    fname_norm.endswith("/" + import_norm) or 
                    fname_norm.endswith("/" + import_as_path)):
                    return fname
                    
            return None

        # Build graphs
        for fname, metadata in parsed_metadata.items():
            imports = metadata.get("imports", [])
            for imp in imports:
                resolved = resolve_import_to_file(fname, imp)
                if resolved and resolved != fname:
                    depends_on[fname].add(resolved)
                    depended_on_by[resolved].add(fname)
                    
        # Construct output dict with serialized lists
        result = {}
        for fname in files.keys():
            result[fname] = {
                "depends_on": sorted(list(depends_on[fname])),
                "depended_on_by": sorted(list(depended_on_by[fname]))
            }
            
        return result
