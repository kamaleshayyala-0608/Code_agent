import os
import io
import zipfile
from typing import Dict, Any

class ExportAgent:
    def __init__(self):
        pass

    def package_refactored_project(
        self,
        original_files: Dict[str, str],
        refactored_files: Dict[str, str],
        reports: Dict[str, Any],
        spec_rules: str = ""
    ) -> Dict[str, str]:
        """
        Packages ONLY the fully refactored source files and spec.md.
        No reports, no planning files, no rule files, no summary.
        """
        packaged = {}

        # 1. Include spec.md at root of the ZIP
        if spec_rules and spec_rules.strip():
            packaged["spec.md"] = spec_rules.strip()
        else:
            spec_path = "rules/refactoring_spec.md"
            if not os.path.exists(spec_path):
                spec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules", "refactoring_spec.md")
            if os.path.exists(spec_path):
                try:
                    with open(spec_path, "r", encoding="utf-8") as f:
                        packaged["spec.md"] = f.read()
                except Exception:
                    pass

        # 2. Add complete refactored source code for every file
        for fname, ref_code in refactored_files.items():
            fname_norm = fname.replace("\\", "/")
            if fname_norm.startswith("Refactored_Project/"):
                fname_norm = fname_norm[len("Refactored_Project/"):]
            packaged[fname_norm] = ref_code

        return packaged

    def build_zip_archive(self, packaged_files: Dict[str, str]) -> bytes:
        """
        Generates a ZIP archive in-memory for download.
        Every file in the ZIP is written directly to its relative path.
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filepath, content in packaged_files.items():
                clean_path = filepath.replace("\\", "/").lstrip("/")
                zip_file.writestr(clean_path, content)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
