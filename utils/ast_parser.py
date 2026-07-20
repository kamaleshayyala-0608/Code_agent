import ast
import re
import os
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
            "imports": [],
            "dependencies": [],
            "complexity_estimate": 1
        }
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to regex if syntax error (e.g. incomplete code)
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

            # Extract standalone functions
            elif isinstance(node, ast.FunctionDef):
                # Check if it's not nested inside a class (we only want module-level here)
                # Walk does not tell us nesting, so let's verify parent scope by traversing tree hierarchically
                pass

        # To get proper scoping, traverse node bodies hierarchically
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
        Regex-based parsing fallback for non-Python or broken files.
        """
        metadata = {
            "classes": [],
            "functions": [],
            "imports": [],
            "dependencies": [],
            "complexity_estimate": 1
        }
        
        # Approximate complexity by counting branching keywords
        branching_keywords = ["if", "for", "while", "catch", "try", "switch", "&&", "||"]
        complexity = 1
        for kw in branching_keywords:
            complexity += content.count(kw)
        metadata["complexity_estimate"] = complexity

        # 1. Classes regex
        class_pattern = r"(?:class|interface)\s+([a-zA-Z0-9_$]+)"
        classes = re.findall(class_pattern, content)
        for cls in classes:
            metadata["classes"].append({
                "name": cls,
                "methods": [],
                "docstring": "",
                "bases": []
            })

        # 2. Functions regex
        # JavaScript/TypeScript/Python/C++/Java function signatures
        func_patterns = [
            r"function\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)",  # JS/TS/PHP function
            r"(?:public|private|protected|static|\s)+\s+([a-zA-Z0-9_$<>]+)\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)\s*\{",  # Java/C++ method
            r"(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*\(([^)]*)\)\s*=>",  # JS/TS arrow function
            r"def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\):"  # Python def
        ]
        
        for pattern in func_patterns:
            matches = re.finditer(pattern, content)
            for m in matches:
                groups = m.groups()
                if len(groups) == 2:
                    name, args = groups
                elif len(groups) == 3:
                    _, name, args = groups
                else:
                    continue
                    
                arg_list = [a.strip().split(":")[-1].strip() for a in args.split(",") if a.strip()]
                metadata["functions"].append({
                    "name": name,
                    "arguments": arg_list,
                    "returns": "unknown",
                    "docstring": ""
                })

        # 3. Imports regex
        import_patterns = [
            r"import\s+[\s\S]*?\s+from\s+['\"]([^'\"]+)['\"]",  # ES6 imports
            r"require\(['\"]([^'\"]+)['\"]\)",  # CommonJS require
            r"import\s+([a-zA-Z0-9_.*]+);",  # Java imports
            r"#include\s+<([^>]+)>",  # C++ header include
            r"#include\s+\"([^\"]+)\"",  # C++ local include
            r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"  # Dynamic imports
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for imp in matches:
                metadata["imports"].append(imp)
                metadata["dependencies"].append(imp)

        return metadata

    @classmethod
    def parse_file(cls, filename: str, content: str) -> Dict[str, Any]:
        _, ext = os.path.splitext(filename.lower())
        if ext == ".py":
            return cls.parse_python_file(content)
        return cls.parse_with_regex(content, ext[1:] if ext else "")
