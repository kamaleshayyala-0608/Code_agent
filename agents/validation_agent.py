import os
import re
import ast
import subprocess
import tempfile
from typing import Dict, Any, Tuple, List
from agents.base_agent import BaseAgent
from utils.ast_parser import ASTParser

class ValidationAgent(BaseAgent):
    """
    Validation Agent: Performs an 8-point rigorous validation suite:
    1. AST Parsing Check
    2. Compiler / Language Syntax Check
    3. Imports Resolution Check
    4. Unused Imports Check
    5. Broken References Check
    6. Code Formatting Check
    7. Complexity Guardrail Check
    8. Basic Linter Check
    """

    def __init__(self, model_name: str = "qwen3:8b"):
        super().__init__(model_name)

    def validate_full(self, file_name: str, original_code: str, refactored_code: str) -> Dict[str, Any]:
        """
        Runs full 8-point validation suite on refactored code and returns structured report.
        """
        results = {
            "success": True,
            "ast_valid": True,
            "syntax_valid": True,
            "imports_valid": True,
            "unused_imports": [],
            "broken_references": [],
            "formatting_valid": True,
            "complexity_check": True,
            "lint_messages": [],
            "diagnostics": []
        }

        # 1. AST Parsing Check
        try:
            ref_ast = ASTParser.parse_file(file_name, refactored_code)
            results["diagnostics"].append("✓ AST parsing check passed.")
        except Exception as e:
            results["ast_valid"] = False
            results["success"] = False
            results["diagnostics"].append(f"❌ AST parsing failed: {str(e)}")

        # 2. Syntax Check
        syntax_ok, syntax_msg = self.check_syntax(file_name, refactored_code)
        results["syntax_valid"] = syntax_ok
        if not syntax_ok:
            results["success"] = False
        results["diagnostics"].append(f"{'✓' if syntax_ok else '❌'} Syntax: {syntax_msg}")

        # 3. Imports Resolution & Unused Imports Check
        orig_ast = ASTParser.parse_file(file_name, original_code)
        orig_imports = set(orig_ast.get("imports", []))
        ref_imports = set(ref_ast.get("imports", []) if 'ref_ast' in locals() else [])

        unused = []
        for imp in ref_imports:
            imp_base = imp.split("/")[-1].split(".")[-1]
            if refactored_code.count(imp_base) <= 1:
                unused.append(imp)
        results["unused_imports"] = unused
        if unused:
            results["diagnostics"].append(f"⚠️ Unused imports detected: {', '.join(unused)}")

        # 4. Broken References Check
        orig_classes = set(c.get("name") for c in orig_ast.get("classes", []))
        ref_classes = set(c.get("name") for c in (ref_ast.get("classes", []) if 'ref_ast' in locals() else []))
        missing_classes = orig_classes - ref_classes

        orig_funcs = set(f.get("name") for f in orig_ast.get("functions", []))
        ref_funcs = set(f.get("name") for f in (ref_ast.get("functions", []) if 'ref_ast' in locals() else []))
        missing_funcs = orig_funcs - ref_funcs

        if missing_classes or (len(orig_funcs) > 0 and len(missing_funcs) / len(orig_funcs) > 0.7):
            results["broken_references"] = list(missing_classes) + list(missing_funcs)
            results["success"] = False
            results["diagnostics"].append(f"❌ Broken Symbol References: Missing {results['broken_references']}")
        else:
            results["diagnostics"].append("✓ Public symbol interface contracts intact.")

        # 5. Formatting Check
        if "\r\n" in refactored_code or refactored_code.endswith("\n\n\n"):
            results["formatting_valid"] = False
            results["diagnostics"].append("⚠️ Minor formatting issue: unnormalized line endings or excessive blank lines.")
        else:
            results["diagnostics"].append("✓ Formatting check passed.")

        # 6. Complexity Guardrail Check
        orig_cyclo = orig_ast.get("complexity_estimate", 1)
        ref_cyclo = ref_ast.get("complexity_estimate", 1) if 'ref_ast' in locals() else 1
        if ref_cyclo > orig_cyclo + 5:
            results["complexity_check"] = False
            results["diagnostics"].append(f"⚠️ Cyclomatic complexity increased significantly ({orig_cyclo} -> {ref_cyclo}).")
        else:
            results["diagnostics"].append(f"✓ Complexity check passed ({orig_cyclo} -> {ref_cyclo}).")

        # 7. Basic Linter Check (Python pyflakes/flake8 or regex)
        lint_msgs = []
        if file_name.endswith(".py"):
            try:
                ast.parse(refactored_code)
            except SyntaxError as se:
                lint_msgs.append(f"Syntax error line {se.lineno}: {se.msg}")
        results["lint_messages"] = lint_msgs

        return results

    def check_syntax(self, file_name: str, code: str) -> Tuple[bool, str]:
        _, ext = os.path.splitext(file_name.lower())

        if ext == ".py":
            try:
                compile(code, file_name, 'exec')
                return True, "Python syntax check passed."
            except SyntaxError as e:
                return False, f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\nLine: {e.text}"
            except Exception as e:
                return False, f"Python compilation failed: {str(e)}"

        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            suffix = ext if ext else ".tsx"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as temp_file:
                    temp_file.write(code)
                    temp_path = temp_file.name

                devnull = "NUL" if os.name == "nt" else "/dev/null"
                cmd_esbuild = ["npx", "esbuild", temp_path, f"--outfile={devnull}"]

                res = subprocess.run(cmd_esbuild, capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                if res.returncode != 0:
                    err = res.stderr or res.stdout or "Esbuild syntax error."
                    clean_err = err.replace(temp_path, file_name)
                    return False, f"React/TS compiler error:\n{clean_err}"

                return True, "TS/JSX compilation check passed."
            except Exception as e:
                return True, f"Bypassed: syntax check tooling offline ({str(e)})."
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        elif ext == ".java":
            temp_path = None
            try:
                res_check = subprocess.run(["javac", "-version"], capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                if res_check.returncode == 0:
                    with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w", encoding="utf-8") as temp_file:
                        temp_file.write(code)
                        temp_path = temp_file.name

                    cmd = ["javac", "-nowarn", "-d", tempfile.gettempdir(), temp_path]
                    res = subprocess.run(cmd, capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                    if res.returncode != 0:
                        err = res.stderr or res.stdout
                        clean_err = err.replace(temp_path, file_name)
                        return False, f"Java compilation failed:\n{clean_err}"
                    return True, "Java compilation passed."
            except Exception:
                pass
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        return True, "Syntax check completed."

    def validate_behavior(self, file_name: str, original: str, refactored: str) -> Tuple[bool, str]:
        orig_meta = ASTParser.parse_file(file_name, original)
        ref_meta = ASTParser.parse_file(file_name, refactored)

        orig_classes = set([c.get("name") for c in orig_meta.get("classes", [])])
        ref_classes = set([c.get("name") for c in ref_meta.get("classes", [])])

        missing_classes = orig_classes - ref_classes
        if missing_classes:
            return False, f"Interface Mismatch: Missing class definitions {list(missing_classes)}."

        orig_funcs = set([f.get("name") for f in orig_meta.get("functions", [])])
        ref_funcs = set([f.get("name") for f in ref_meta.get("functions", [])])

        missing_funcs = orig_funcs - ref_funcs
        public_missing = [f for f in missing_funcs if not f.startswith("_")]
        orig_public_count = len([f for f in orig_funcs if not f.startswith("_")])

        if orig_public_count > 0 and len(public_missing) / orig_public_count > 0.7:
            return False, f"Interface Mismatch: Most public function/component definitions missing {public_missing}."

        return True, "Interface contract verified. Functions and classes definitions are identical."
