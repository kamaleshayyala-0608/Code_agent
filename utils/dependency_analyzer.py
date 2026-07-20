import os
from typing import Dict, List, Any, Set, Tuple

class DependencyAnalyzer:
    @staticmethod
    def analyze_workspace_dependencies(files: Dict[str, str], parsed_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes dependencies between files, building a symbol-level dependency map,
        a symbol definition registry, and detecting circular references.
        """
        depends_on = {fname: set() for fname in files.keys()}
        depended_on_by = {fname: set() for fname in files.keys()}
        
        # 1. Symbol definitions registry
        # Maps symbol_name -> defining_file
        symbol_registry: Dict[str, str] = {}
        for fname, meta in parsed_metadata.items():
            # Register classes
            for cls in meta.get("classes", []):
                name = cls.get("name")
                if name:
                    symbol_registry[name] = fname
            # Register functions
            for func in meta.get("functions", []):
                name = func.get("name")
                if name:
                    symbol_registry[name] = fname
            # Register interfaces
            for interf in meta.get("interfaces", []):
                name = interf.get("name")
                if name:
                    symbol_registry[name] = fname

        # Helper to resolve imports to files
        def resolve_import_to_file(source_file: str, import_str: str) -> str | None:
            if not import_str:
                return None
            source_dir = os.path.dirname(source_file.replace("\\", "/"))
            import_norm = import_str.replace("\\", "/").strip()
            
            if import_norm.startswith("."):
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
                
                for fname in files.keys():
                    fname_norm = fname.replace("\\", "/").lstrip("/")
                    fname_no_ext, _ = os.path.splitext(fname_norm)
                    if fname_norm == target_base or fname_no_ext == target_base:
                        return fname
                    if fname_norm.startswith(target_base + "/"):
                        if fname_norm.endswith("index.js") or fname_norm.endswith("index.ts") or fname_norm.endswith("index.tsx"):
                            return fname
                            
            # Check absolute local modules
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

        # Build file-level import graph
        for fname, metadata in parsed_metadata.items():
            imports = metadata.get("imports", [])
            for imp in imports:
                resolved = resolve_import_to_file(fname, imp)
                if resolved and resolved != fname:
                    depends_on[fname].add(resolved)
                    depended_on_by[resolved].add(fname)

        # 2. Build Symbol-level Graph
        symbol_connections = {fname: [] for fname in files.keys()} # file -> list of (symbol, defining_file)
        
        for fname, metadata in parsed_metadata.items():
            references = metadata.get("symbol_references", [])
            for ref in references:
                if ref in symbol_registry:
                    defining_file = symbol_registry[ref]
                    if defining_file != fname:
                        # This file references a symbol defined in defining_file
                        symbol_connections[fname].append({
                            "symbol": ref,
                            "defined_in": defining_file
                        })
                        # Also add to file dependencies if not already caught
                        depends_on[fname].add(defining_file)
                        depended_on_by[defining_file].add(fname)

        # 3. Detect Circular Dependencies (DFS)
        circular_loops = []
        
        def find_cycles():
            visited = {} # file -> 0=unvisited, 1=visiting, 2=visited
            path = []
            
            for f in files.keys():
                visited[f] = 0

            def dfs(node):
                visited[node] = 1
                path.append(node)
                
                for neighbor in depends_on[node]:
                    if visited[neighbor] == 1:
                        # Found a circular cycle back-edge!
                        cycle_start_idx = path.index(neighbor)
                        cycle = path[cycle_start_idx:] + [neighbor]
                        circular_loops.append(cycle)
                    elif visited[neighbor] == 0:
                        dfs(neighbor)
                        
                path.pop()
                visited[node] = 2

            for f in files.keys():
                if visited[f] == 0:
                    dfs(f)

        find_cycles()

        # Compile final results
        result = {}
        for fname in files.keys():
            result[fname] = {
                "depends_on": sorted(list(depends_on[fname])),
                "depended_on_by": sorted(list(depended_on_by[fname])),
                "symbol_dependencies": symbol_connections[fname],
                "circular_loops": [loop for loop in circular_loops if fname in loop],
                "is_in_circular_loop": any(fname in loop for loop in circular_loops)
            }
            
        return result
