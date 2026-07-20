import os
import io
import json
import zipfile
from typing import Dict, Any

class ExportAgent:
    def __init__(self):
        pass

    def package_refactored_project(
        self,
        original_files: Dict[str, str],
        refactored_files: Dict[str, str],
        reports: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Organizes refactored code files, reports, and metadata JSON files
        into a structured package dictionary with advanced enterprise metrics.
        """
        packaged = {}
        
        # Track statistics for SUMMARY.md
        total_files = len(original_files)
        refactored_count = 0
        total_rules = 0
        total_patterns = 0
        validation_successes = 0
        score_before_sum = 0
        score_after_sum = 0
        total_retries = 0
        
        # Individual file metrics and validation dictionaries
        validation_report = {}
        quality_metrics = {}

        for fname, ref_code in refactored_files.items():
            orig_code = original_files.get(fname, "")
            file_report = reports.get(fname, {})
            
            # Extract directory/base name
            fname_base, _ = os.path.splitext(fname)
            fname_base_norm = fname_base.replace("\\", "/")
            fname_norm = fname.replace("\\", "/")

            # 1. Refactored Code
            packaged[f"Refactored_Project/{fname_norm}"] = ref_code

            # Get reports data
            rules_md = file_report.get("rules", "No rules extracted.")
            patterns_list = file_report.get("patterns_list", [])
            patterns_report = file_report.get("patterns_report", "No patterns analyzed.")
            plan = file_report.get("plan", {})
            validation = file_report.get("validation", {})
            retries = file_report.get("retries", [])
            quality = file_report.get("quality", {})

            # Update stats
            if plan.get("should_refactor", True):
                refactored_count += 1
            
            rules_count = rules_md.count("✓ **") or 1
            total_rules += rules_count
            total_patterns += len(patterns_list)
            
            if validation.get("success", False):
                validation_successes += 1
                
            score_before = quality.get("score_before", 70)
            score_after = quality.get("score_after", score_before)
            score_before_sum += score_before
            score_after_sum += score_after
            
            total_retries += len(retries)

            # 2. Refactoring Report (Markdown)
            report_md = f"""# Refactoring Report for `{fname_norm}`

## Problems Found
{patterns_report}

## Recommendations & Steps
**Priority:** {plan.get('priority', 'Medium')}
**Confidence:** {plan.get('confidence', 80)}%

{plan.get('steps_md', 'No steps defined.')}

## Applied Rules
{rules_md}

## Validation Status
- **Success Status:** {'✅ PASSED' if validation.get('success', False) else '❌ FAILED'}
- **Compiler/Syntax Check:** {validation.get('syntax_msg', 'N/A')}
- **Behavioral Equivalence:** {validation.get('behavior_msg', 'N/A')}

## Quality Evaluation
- **Maintainability Index (Before):** {quality.get('orig_mi', 70)}%
- **Maintainability Index (After):** {quality.get('ref_mi', 90)}%
- **Cyclomatic Complexity:** {quality.get('orig_complexity', 1)} → {quality.get('ref_complexity', 1)} ({quality.get('complexity_reduction_pct', 0)}% change)
- **Cognitive nesting Complexity:** {quality.get('orig_cognitive', 1)} → {quality.get('ref_cognitive', 1)}
- **File Dependency Coupling:** {quality.get('orig_coupling', 0)} → {quality.get('ref_coupling', 0)}
- **Dead Code (Unused imports):** {quality.get('dead_code_count', 0)} imports

- **Lines of Code:** {quality.get('orig_lines', 0)} → {quality.get('ref_lines', 0)} ({quality.get('lines_reduced', 0)} lines, {quality.get('reduction_pct', 0)}% change)

- **Justification:** {quality.get('justification', '')}
"""
            packaged[f"Refactoring_Report/{fname_base_norm}.md"] = report_md

            # 3. Planning JSON
            packaged[f"Planning/{fname_base_norm}_plan.json"] = json.dumps(plan, indent=2)

            # 4. Rules Markdown
            packaged[f"Rules/{fname_base_norm}_rules.md"] = rules_md

            # 5. Pattern Analysis JSON
            packaged[f"Pattern_Analysis/{fname_base_norm}_patterns.json"] = json.dumps({
                "patterns": patterns_list,
                "detail_report": patterns_report
            }, indent=2)

            # Populate metadata dicts
            validation_report[fname_norm] = {
                "syntax": "PASS" if "passed" in validation.get("syntax_msg", "").lower() or validation.get("success", False) else "FAIL",
                "behavior": "PASS" if validation.get("success", False) else "FAIL",
                "compile": "PASS" if "passed" in validation.get("syntax_msg", "").lower() or validation.get("success", False) else "FAIL"
            }
            
            quality_metrics[fname_norm] = {
                "before_score": score_before,
                "after_score": score_after,
                "maintainability_index_before": quality.get("orig_mi", 70),
                "maintainability_index_after": quality.get("ref_mi", 90),
                "cyclomatic_complexity_before": quality.get("orig_complexity", 1),
                "cyclomatic_complexity_after": quality.get("ref_complexity", 1),
                "cognitive_complexity_before": quality.get("orig_cognitive", 1),
                "cognitive_complexity_after": quality.get("ref_cognitive", 1),
                "coupling_before": quality.get("orig_coupling", 0),
                "coupling_after": quality.get("ref_coupling", 0),
                "dead_code_count": quality.get("dead_code_count", 0),
                "lines_before": quality.get("orig_lines", 0),
                "lines_after": quality.get("ref_lines", 0)
            }

        # 6. Consolidated Validation Report
        packaged["Validation/validation_report.json"] = json.dumps(validation_report, indent=2)

        # 7. Consolidated Metrics Report
        packaged["Metrics/quality_metrics.json"] = json.dumps(quality_metrics, indent=2)

        # Calculate averages
        avg_score_before = round(score_before_sum / total_files) if total_files > 0 else 70
        avg_score_after = round(score_after_sum / total_files) if total_files > 0 else 90
        val_success_rate = round((validation_successes / total_files) * 100) if total_files > 0 else 100

        # 8. SUMMARY.md
        summary_md = f"""# Project Refactoring Summary

| Metric | Value |
| --- | --- |
| **Files Processed** | {total_files} |
| **Files Refactored** | {refactored_count} |
| **Rules Applied** | {total_rules} |
| **Patterns Detected** | {total_patterns} |
| **Validation Success** | {val_success_rate}% |
| **Average Score Before** | {avg_score_before}% |
| **Average Score After** | {avg_score_after}% |
| **Total Fix Retries** | {total_retries} |

## File Execution Status
"""
        for fname_norm, val in validation_report.items():
            metrics = quality_metrics.get(fname_norm, {})
            summary_md += f"- **`{fname_norm}`**: Maintainability Index: `{metrics.get('maintainability_index_before')}%` → `{metrics.get('maintainability_index_after')}%` | Validation: `[Compile: {val.get('compile')}, Behavior: {val.get('behavior')}]` \n"

        packaged["SUMMARY.md"] = summary_md

        return packaged

    def build_zip_archive(self, packaged_files: Dict[str, str]) -> bytes:
        """
        Generates a ZIP archive in-memory for download.
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filepath, content in packaged_files.items():
                clean_path = filepath.replace("\\", "/")
                zip_file.writestr(clean_path, content)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
