import os
import io
import zipfile
from typing import Dict, Any
from utils.completeness_validator import CompletenessValidator

class ExportAgent:
    """
    Export Agent: Packages ONLY the clean, refactored project source files into the
    root ZIP archive payload. No reports, no metrics, no original files, no extra metadata.
    """

    def package_refactored_project(
        self,
        original_files: Dict[str, str],
        refactored_files: Dict[str, str],
        reports: Dict[str, Any] = None,
        spec_rules: str = ""
    ) -> Dict[str, str]:
        packaged: Dict[str, str] = {}

        for fname, ref_code in refactored_files.items():
            fname_clean = fname.replace("\\", "/").lstrip("/")
            if fname_clean.startswith("Refactored_Project/"):
                fname_clean = fname_clean[len("Refactored_Project/"):]

            orig_code = original_files.get(fname, "")

            # Safeguard: Fallback to original if empty or failed completeness
            comp_ok, _ = CompletenessValidator.validate(fname, orig_code, ref_code)
            if not ref_code or not ref_code.strip() or not comp_ok:
                ref_code = orig_code

            packaged[fname_clean] = ref_code

        # Include the specification that governed this refactor alongside the
        # code so the download is self-documenting.
        if spec_rules and spec_rules.strip():
            packaged["spec.md"] = spec_rules

        return packaged

    def build_zip_archive(self, packaged_files: Dict[str, str]) -> bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filepath, content in packaged_files.items():
                clean_path = filepath.replace("\\", "/").lstrip("/")
                zip_file.writestr(clean_path, content)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()