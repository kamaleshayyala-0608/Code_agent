import ast
import re
import os
import subprocess
import json
from typing import Dict, Any, List

class ASTParser:
    @staticmethod
    def parse_python_file(content: str) -> Dict[str, Any]:
        """
        Parses Python code using the native ast module.
        """
        metadata = {
            "classes": [],
            "functions": [],
            "interfaces": [],
            "imports": [],
            "exports": [],
            "dependencies": [],
            "hooks": [],
            "symbol_references": [],
            "complexity_estimate": 1
        }
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to regex if syntax error
            return ASTParser.parse_with_regex(content, "py")

        complexity = 1
        
        # Traverse AST nodes
        for node in ast.walk(tree):
            # Estimate complexity (branching points)
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
            # Extract imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    metadata["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                metadata["imports"].append(module)
                for alias in node.names:
                    metadata["dependencies"].append(f"{module}.{alias.name}")

            # Extract classes
            elif isinstance(node, ast.ClassDef):
                cls_info = {
                    "name": node.name,
                    "methods": [],
                    "docstring": ast.get_docstring(node) or "",
                    "bases": [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else []
                }
                
                # Extract methods of this class
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        args = [arg.arg for arg in child.args.args]
                        ret = ""
                        if child.returns:
                            ret = ast.unparse(child.returns) if hasattr(ast, "unparse") else ""
                        cls_info["methods"].append({
                            "name": child.name,
                            "arguments": args,
                            "returns": ret,
                            "docstring": ast.get_docstring(child) or ""
                        })
                metadata["classes"].append(cls_info)

            # Track variable names / function names as symbol references
            elif isinstance(node, ast.Name):
                name = node.id
                if name not in metadata["symbol_references"] and len(name) > 1:
                    metadata["symbol_references"].append(name)

        # Standalone functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                ret = ast.unparse(node.returns) if node.returns and hasattr(ast, "unparse") else ""
                metadata["functions"].append({
                    "name": node.name,
                    "arguments": args,
                    "returns": ret,
                    "docstring": ast.get_docstring(node) or ""
                })

        metadata["complexity_estimate"] = complexity
        return metadata

    @staticmethod
    def parse_with_regex(content: str, ext: str) -> Dict[str, Any]:
        """
        Regex-based parsing fallback for TS/JS files.
        """
        metadata = {
            "classes": [],
            "functions": [],
            "interfaces": [],
            "imports": [],
            "exports": [],
            "dependencies": [],
            "hooks": [],
            "symbol_references": [],
            "complexity_estimate": 1
        }
        
        branching_keywords = [
            r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bcatch\b", r"\btry\b", 
            r"\bcase\b", r"&&", r"\|\|", r"\?\."
        ]
        complexity = 1
        for pattern in branching_keywords:
            complexity += len(re.findall(pattern, content))
        metadata["complexity_estimate"] = complexity

        # 1. Classes regex
        class_pattern = r"\bclass\s+([a-zA-Z0-9_$]+)(?:\s+extends\s+([a-zA-Z0-9_$.]+))?"
        classes = re.findall(class_pattern, content)
        for cls, base in classes:
            metadata["classes"].append({
                "name": cls,
                "methods": [],
                "docstring": "",
                "bases": [base] if base else []
            })

        # 2. React Hooks extraction
        hook_patterns = [
            r"\buseState\s*\(",
            r"\buseEffect\s*\(",
            r"\buseMemo\s*\(",
            r"\buseCallback\s*\(",
            r"\buseRef\s*\(",
            r"\buseContext\s*\(",
            r"\buseReducer\s*\("
        ]
        custom_hook_matches = re.findall(r"\b(use[A-Z][a-zA-Z0-9_$]+)\s*\(", content)
        for hook in custom_hook_matches:
            if hook not in metadata["hooks"]:
                metadata["hooks"].append(hook)
                
        for pattern in hook_patterns:
            name = pattern.replace(r"\b", "").replace(r"\s*\(", "")
            if re.search(pattern, content):
                if name not in metadata["hooks"]:
                    metadata["hooks"].append(name)

        # 3. Functions / React Component signatures
        func_signatures = []
        fc_pattern = r"\b(?:const|let|var)\s+([A-Z][a-zA-Z0-9_$]*)\s*(?::\s*[^=]+)?\s*=\s*(?:\([^)]*\)|[a-zA-Z0-9_$]+)\s*=>"
        fc_components = re.findall(fc_pattern, content)
        for comp in fc_components:
            func_signatures.append({
                "name": comp,
                "arguments": [],
                "returns": "JSX.Element",
                "docstring": "React Functional Component"
            })
            
        traditional_fn_pattern = r"\bfunction\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)"
        traditional_fns = re.findall(traditional_fn_pattern, content)
        for name, args in traditional_fns:
            arg_list = [a.strip() for a in args.split(",") if a.strip()]
            func_signatures.append({
                "name": name,
                "arguments": arg_list,
                "returns": "unknown",
                "docstring": ""
            })
                    
        metadata["functions"] = func_signatures

        # 4. Imports & local dependencies
        import_patterns = [
            r"\bimport\s+[\s\S]*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"\brequire\(['\"]([^'\"]+)['\"]\)"
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for imp in matches:
                metadata["imports"].append(imp)
                if imp.startswith("."):
                    metadata["dependencies"].append(imp)

        # 5. Simple symbol reference matching
        words = re.findall(r"\b[a-zA-Z0-9_$]+\b", content)
        for w in words:
            if w not in metadata["symbol_references"] and len(w) > 1:
                metadata["symbol_references"].append(w)

        return metadata

    @classmethod
    def parse_file(cls, filename: str, content: str) -> Dict[str, Any]:
        _, ext = os.path.splitext(filename.lower())
        
        if ext == ".py":
            return cls.parse_python_file(content)
            
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            # Execute node sidecar for TS AST Compiler API parsing
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parser_js_path = os.path.join(base_dir, "utils", "ts_parser.js")
            
            # Formulate temp file to feed into Node
            import tempfile
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w", encoding="utf-8") as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name

                cmd = ["node", parser_js_path, temp_path]
                result = subprocess.run(cmd, capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                
                if result.returncode == 0 and result.stdout.strip():
                    parsed_meta = json.loads(result.stdout)
                    
                    # Compute local dependencies relative to this folder
                    parsed_meta["dependencies"] = []
                    for imp in parsed_meta.get("imports", []):
                        if imp.startswith("."):
                            parsed_meta["dependencies"].append(imp)
                    return parsed_meta
            except Exception:
                pass # Fallback to regex parsing on error
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        # Regex fallback on non-TS/non-py
        return cls.parse_with_regex(content, ext[1:] if ext else "")
