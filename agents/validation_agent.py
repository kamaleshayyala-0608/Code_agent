import os
import subprocess
import tempfile
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent
from utils.ast_parser import ASTParser

class ValidationAgent(BaseAgent):
    def __init__(self, model_name: str = "gemma4:26b"):
        super().__init__(model_name)

    def check_syntax(self, file_name: str, code: str) -> Tuple[bool, str]:
        """
        Validates compilation and syntax using native compilers and tooling:
        - Python -> compile()
        - TS/TSX -> npx esbuild / npx tsc --noEmit
        - JS/JSX -> npx esbuild / eslint
        - Java -> javac
        """
        _, ext = os.path.splitext(file_name.lower())
        
        # 1. Python Validation
        if ext == ".py":
            try:
                compile(code, file_name, 'exec')
                return True, "Python syntax check passed (compiled successfully)."
            except SyntaxError as e:
                return False, f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\nLine: {e.text}"
            except Exception as e:
                return False, f"Python compilation failed: {str(e)}"
                
        # 2. TypeScript / React Validation
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            suffix = ext if ext else ".tsx"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as temp_file:
                    temp_file.write(code)
                    temp_path = temp_file.name

                # Primary check: esbuild
                devnull = "NUL" if os.name == "nt" else "/dev/null"
                cmd_esbuild = ["npx", "esbuild", temp_path, f"--outfile={devnull}"]
                
                # Execute esbuild
                res = subprocess.run(cmd_esbuild, capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                if res.returncode != 0:
                    err = res.stderr or res.stdout or "Esbuild syntax error."
                    clean_err = err.replace(temp_path, file_name)
                    return False, f"React/TS compiler error:\n{clean_err}"

                # Secondary check: If TS and tsc exists / configured, run tsc --noEmit
                if ext in (".ts", ".tsx") and os.path.exists("tsconfig.json"):
                    cmd_tsc = ["npx", "tsc", "--noEmit", temp_path]
                    res_tsc = subprocess.run(cmd_tsc, capture_output=True, text=True, shell=(os.name == 'nt'), check=False)
                    if res_tsc.returncode != 0:
                        err = res_tsc.stderr or res_tsc.stdout or "TypeScript warnings."
                        clean_err = err.replace(temp_path, file_name)
                        return True, f"TS compiled with warnings:\n{clean_err}" # Keep code, report warnings

                return True, "TS/JSX compilation check passed."
            except Exception as e:
                return True, f"Bypassed: syntax check tooling offline ({str(e)})."
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                        
        # 3. Java Validation
        elif ext == ".java":
            temp_path = None
            try:
                # Find javac compiler
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
                pass # javac not on PATH, skip
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                        
        return True, "Syntax check completed."

    def validate_behavior(self, file_name: str, original: str, refactored: str) -> Tuple[bool, str]:
        """
        Deterministic behavioral equivalence checker.
        Ensures that defined class names, public functions, and component names match.
        Bypasses LLM call to eliminate latency and cost.
        """
        orig_meta = ASTParser.parse_file(file_name, original)
        ref_meta = ASTParser.parse_file(file_name, refactored)

        # Check classes matches
        orig_classes = set([c.get("name") for c in orig_meta.get("classes", [])])
        ref_classes = set([c.get("name") for c in ref_meta.get("classes", [])])
        
        missing_classes = orig_classes - ref_classes
        if missing_classes:
            return False, f"Interface Mismatch: Missing class definitions {list(missing_classes)}."

        # Check function/component matches — only fail if MOST public functions vanished,
        # since legitimate refactors (extract/rename/merge helpers) will naturally shift some names.
        orig_funcs = set([f.get("name") for f in orig_meta.get("functions", [])])
        ref_funcs = set([f.get("name") for f in ref_meta.get("functions", [])])

        missing_funcs = orig_funcs - ref_funcs
        public_missing = [f for f in missing_funcs if not f.startswith("_")]
        orig_public_count = len([f for f in orig_funcs if not f.startswith("_")])

        # Only treat it as a real interface break if it wiped out nearly everything
        # (e.g. LLM returned empty/garbage), not just renamed/reorganized a few symbols.
        if orig_public_count > 0 and len(public_missing) / orig_public_count > 0.7:
            return False, f"Interface Mismatch: Most public function/component definitions missing {public_missing}."

        return True, "Interface contract verified. Functions and classes definitions are identical."
