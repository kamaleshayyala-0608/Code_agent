import os
import re
from typing import Dict, Any, Tuple, Callable
from utils.ast_parser import ASTParser

class ToolRegistry:
    """
    Tool-based refactoring helper suite.
    Replaces prompt-only direct LLM output generation with structured tool calls:
    Read File -> AST Analysis -> Dependency Graph Lookup -> LLM Refactor -> Formatter -> Validator
    """

    @staticmethod
    def read_file(file_name: str, files: Dict[str, str]) -> str:
        """Read file contents safely from workspace files dict."""
        return files.get(file_name, "")

    @staticmethod
    def get_ast(file_name: str, code: str) -> Dict[str, Any]:
        """Runs AST parser on file content."""
        return ASTParser.parse_file(file_name, code)

    @staticmethod
    def format_code(file_name: str, code: str) -> str:
        """
        Applies deterministic code formatting rules (trim trailing whitespace,
        ensure trailing newline, normalize line endings to LF, format indentation).
        """
        if not code:
            return ""

        # Normalize line endings
        code = code.replace("\r\n", "\n").replace("\r", "\n")

        # Trim trailing whitespace on lines
        lines = [line.rstrip() for line in code.split("\n")]

        # Remove redundant multi-blank lines (max 2 consecutive blank lines)
        cleaned_lines = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)

        formatted = "\n".join(cleaned_lines)
        if not formatted.endswith("\n"):
            formatted += "\n"

        return formatted

    @staticmethod
    def validate_code(
        file_name: str,
        original_code: str,
        refactored_code: str,
        validator_agent: Any
    ) -> Tuple[bool, str]:
        """
        Executes AST, syntax, interface, and behavioral checks via ValidationAgent.
        """
        syntax_ok, syntax_msg = validator_agent.check_syntax(file_name, refactored_code)
        if not syntax_ok:
            return False, f"Syntax Error: {syntax_msg}"

        behavior_ok, behavior_msg = validator_agent.validate_behavior(file_name, original_code, refactored_code)
        if not behavior_ok:
            return False, f"Behavior Mismatch: {behavior_msg}"

        return True, "Code passed tool-based formatting and validation checks."
