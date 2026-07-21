import os
import json
from typing import Dict, Any, List

class SymbolIndexBuilder:
    """
    Builds a symbol-level index across all project files:
    Component/File -> Functions -> Variables -> Classes -> Exports
    Outputs symbol_index.json payload.
    """

    @staticmethod
    def build_symbol_index(files: Dict[str, str], parsed_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        index = {
            "files": {},
            "all_classes": {},
            "all_functions": {},
            "all_exports": {}
        }

        for fname, meta in parsed_metadata.items():
            classes_list = [c.get("name") for c in meta.get("classes", []) if c.get("name")]
            functions_list = [f.get("name") for f in meta.get("functions", []) if f.get("name")]
            variables_list = meta.get("symbol_references", [])

            exports_list = []
            for c in classes_list:
                exports_list.append({"type": "class", "name": c})
                index["all_classes"][c] = fname
                index["all_exports"][c] = fname

            for f_item in meta.get("functions", []):
                fname_str = f_item.get("name")
                if fname_str and not fname_str.startswith("_"):
                    exports_list.append({"type": "function", "name": fname_str})
                    index["all_functions"][fname_str] = fname
                    index["all_exports"][fname_str] = fname

            index["files"][fname] = {
                "classes": classes_list,
                "functions": functions_list,
                "variables": variables_list[:30], # Top variables
                "exports": exports_list,
                "hooks": meta.get("hooks", []),
                "imports": meta.get("imports", [])
            }

        return index

    @staticmethod
    def save_symbol_index_json(symbol_index: Dict[str, Any], filepath: str = "symbol_index.json") -> str:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(symbol_index, f, indent=2)
            return json.dumps(symbol_index, indent=2)
        except Exception as e:
            return f"{{\"error\": \"Failed to save symbol index: {str(e)}\"}}"
