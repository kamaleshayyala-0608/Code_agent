import os
import io
import json
import zipfile
from typing import Dict, Any

class ExportAgent:
    """
    Export Agent: Packages refactored code and comprehensive diagnostic reports into
    the standard structured ZIP layout:
    Refactored_Project.zip
    │
    ├── refactored/
    ├── original/
    ├── reports/
    ├── metrics/
    ├── suggestions/
    ├── summary.json
    └── README.md
    """

    def package_refactored_project(
        self,
        original_files: Dict[str, str],
        refactored_files: Dict[str, str],
        reports: Dict[str, Any],
        spec_rules: str = ""
    ) -> Dict[str, str]:
        packaged: Dict[str, str] = {}

        total_files = len(original_files)
        changed_files = 0
        overall_quality_scores = []
        file_metrics = {}
        suggestions = {}
        file_reports_md = {}

        for fname in refactored_files.keys():
            fname_clean = fname.replace("\\", "/").lstrip("/")

            orig_code = original_files.get(fname, "")
            ref_code = refactored_files.get(fname, "")

            is_changed = orig_code.strip() != ref_code.strip()
            if is_changed:
                changed_files += 1

            # Populate refactored/ and original/
            packaged[f"refactored/{fname_clean}"] = ref_code
            packaged[f"original/{fname_clean}"] = orig_code

            # Extract report details
            f_report = reports.get(fname, {})
            qual = f_report.get("quality", {})
            score = qual.get("score_after", 80)
            overall_quality_scores.append(score)

            file_metrics[fname_clean] = qual
            suggestions[fname_clean] = f_report.get("patterns_report", "No code smells identified.")
            file_reports_md[fname_clean] = f"# Diagnostic Report for {fname_clean}\n\n" \
                                          f"## Planning\n{f_report.get('plan', {}).get('steps_md', 'N/A')}\n\n" \
                                          f"## Rules Applied\n{f_report.get('rules', 'N/A')}\n"

            packaged[f"reports/{fname_clean}.md"] = file_reports_md[fname_clean]

        avg_score = round(sum(overall_quality_scores) / len(overall_quality_scores)) if overall_quality_scores else 85
        risk_level = "Low" if avg_score >= 80 else "Medium" if avg_score >= 60 else "High"

        summary_json = {
            "total_files": total_files,
            "changed_files": changed_files,
            "unchanged_files": total_files - changed_files,
            "overall_quality_score": avg_score,
            "risk_level": risk_level,
            "refactored_files_list": list(refactored_files.keys())
        }

        readme_md = f"""# Refactored Project Export Package

## Executive Summary
- **Total Workspace Files:** {total_files}
- **Files Refactored:** {changed_files}
- **Average Quality Score:** {avg_score}/100
- **Project Risk Level:** {risk_level}

## Package Structure
- `refactored/` : Updated production-ready source code
- `original/`   : Original untouched baseline code
- `reports/`    : Detailed per-file agent execution diagnostics
- `metrics/`    : Quantitative complexity and maintainability metrics
- `suggestions/`: Identified code smells and recommended design patterns
- `summary.json`: Structured JSON payload of project transformation metrics
"""

        packaged["summary.json"] = json.dumps(summary_json, indent=2)
        packaged["metrics/quality_metrics.json"] = json.dumps(file_metrics, indent=2)
        packaged["suggestions/code_smells.json"] = json.dumps(suggestions, indent=2)
        packaged["README.md"] = readme_md

        return packaged

    def build_zip_archive(self, packaged_files: Dict[str, str]) -> bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filepath, content in packaged_files.items():
                clean_path = filepath.replace("\\", "/").lstrip("/")
                zip_file.writestr(clean_path, content)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
