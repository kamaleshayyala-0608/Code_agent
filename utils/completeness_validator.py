import os
from typing import Tuple, List, Dict, Any
from utils.ast_parser import ASTParser

BANNED_PLACEHOLDERS = [
    "...",
    "existing code",
    "unchanged code",
    "omitted",
    "same as above",
    "same as before",
    "rest of file",
    "code unchanged",
    "keep existing",
    "remaining code",
    "as defined above",
    "code remains the same"
]

class CompletenessValidator:
    """
    Completeness Validator: Verifies that refactored source code is complete,
    uncut, fully-formed, and contains no truncation placeholders or summaries.
    """

    @staticmethod
    def validate(file_name: str, original_code: str, refactored_code: str) -> Tuple[bool, str]:
        if not refactored_code or not refactored_code.strip():
            return False, "Refactored code is completely empty."

        orig_lines = len(original_code.strip().split("\n"))
        ref_lines = len(refactored_code.strip().split("\n"))

        # 1. Length ratio check (only for non-trivial original files > 10 lines)
        if orig_lines > 10:
            if len(refactored_code) < len(original_code) * 0.85 or ref_lines < orig_lines * 0.70:
                return False, f"Drastic file truncation detected: Original had {orig_lines} lines ({len(original_code)} chars), refactored has {ref_lines} lines ({len(refactored_code)} chars)."

        # 2. Check for banned truncation placeholder tokens
        refactored_lower = refactored_code.lower()
        for placeholder in BANNED_PLACEHOLDERS:
            if placeholder in refactored_lower:
                return False, f"Banned truncation placeholder detected: '{placeholder}' in output."

        # 3. Check AST parseability
        try:
            parsed_meta = ASTParser.parse_file(file_name, refactored_code)
            if not parsed_meta:
                return False, "AST parser failed to generate valid symbol metadata."
        except Exception as e:
            return False, f"AST parsing error on refactored code: {str(e)}"

        return True, "Completeness check passed successfully (file is complete, un-truncated, and valid AST)."
