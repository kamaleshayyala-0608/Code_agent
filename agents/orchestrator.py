import os
import time
import json
from typing import Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.scanner_agent import ScannerAgent
from agents.parser_agent import ParserAgent
from agents.rule_agent import RuleExtractionAgent
from agents.pattern_agent import PatternRetrievalAgent
from core.context_builder import ContextBuilder
from core.dependency_graph import DependencyGraph
from core.symbol_index import SymbolIndexBuilder
from core.cache_manager import CacheManager
from core.tool_registry import ToolRegistry
from utils.completeness_validator import CompletenessValidator
from agents.planner_agent import PlannerAgent
from agents.refactoring_agent import RefactoringAgent, TooLargeForSinglePassError
from agents.validation_agent import ValidationAgent
from agents.retry_agent import RetryAgent
from agents.quality_agent import QualityEvaluationAgent
from agents.export_agent import ExportAgent
from utils.ast_parser import ASTParser

REFACTORABLE_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.php', '.rb', '.cs', '.cpp', '.c', '.h', '.hpp'}

class RefactoringOrchestrator:
    """
    Refactoring Orchestrator: Implements project-wide Execution Loop Architecture:
    Upload -> Project Scanner -> Context Builder -> Planner -> Single-Pass Refactor
        -> Completeness Check -> AST Check -> Validation -> Retry Loop -> Quality Check -> Export
    Supports parallel processing, incremental refactoring, AST caching, and stage timing logs.
    """

    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name
        self.scanner = ScannerAgent()
        self.parser = ParserAgent()
        self.rule_agent = RuleExtractionAgent(model_name)
        self.pattern_agent = PatternRetrievalAgent(model_name)
        self.planner = PlannerAgent(model_name)
        self.refactorer = RefactoringAgent(model_name)
        self.validator = ValidationAgent(model_name)
        self.repairer = RetryAgent(model_name)
        self.evaluator = QualityEvaluationAgent(model_name)
        self.exporter = ExportAgent()
        self.cache_manager = CacheManager()
        self.dep_graph_builder = DependencyGraph()

    def refactor_project(self, files: Dict[str, str], max_workers: int = 1) -> Generator[Dict[str, Any], None, None]:
        total_start = time.perf_counter()
        timing_stats = {}

        # -------------------------------------------------------------
        # Step 1: Scanner & AST Parsing
        # -------------------------------------------------------------
        parse_start = time.perf_counter()
        yield {
            "stage": "parsing",
            "status": "running",
            "message": "Scanning project & parsing AST structures..."
        }

        try:
            parsed_metadata = self.parser.parse_project(files)
            parse_duration = round(time.perf_counter() - parse_start, 2)
            timing_stats["Parser"] = f"{parse_duration} sec"

            yield {
                "stage": "parsing",
                "status": "completed",
                "message": f"Project parsed in {parse_duration}s.",
                "data": parsed_metadata
            }
        except Exception as e:
            parsed_metadata = {fname: {"classes": [], "functions": [], "imports": [], "dependencies": [], "complexity_estimate": 1} for fname in files.keys()}
            timing_stats["Parser"] = "Failed"
            yield {
                "stage": "parsing",
                "status": "failed",
                "message": f"Parsing failed: {str(e)}. Continuing with fallback metadata."
            }

        # -------------------------------------------------------------
        # Step 2: Context Builder & Dependency Graph & Symbol Index
        # -------------------------------------------------------------
        ctx_start = time.perf_counter()
        yield {
            "stage": "context_building",
            "status": "running",
            "message": "Building workspace dependency graph & symbol index..."
        }

        try:
            dep_graph = self.dep_graph_builder.generate_graph(files, parsed_metadata)
            symbol_index = SymbolIndexBuilder.build_symbol_index(files, parsed_metadata)
            SymbolIndexBuilder.save_symbol_index_json(symbol_index)

            # Load persistent memory spec.md
            spec_rules = ""
            spec_path = "memory/spec.md"
            if os.path.exists(spec_path):
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec_rules = f.read()

            project_memory = {"spec_md": spec_rules}
            project_context = ContextBuilder.build_project_context(files, parsed_metadata, dep_graph, symbol_index, project_memory)

            ctx_duration = round(time.perf_counter() - ctx_start, 2)
            timing_stats["ContextBuilder"] = f"{ctx_duration} sec"

            yield {
                "stage": "context_building",
                "status": "completed",
                "message": f"Project context built in {ctx_duration}s.",
                "data": project_context
            }
        except Exception as e:
            project_context = {"file_list": list(files.keys()), "dependency_graph": {}, "project_memory": {}}
            timing_stats["ContextBuilder"] = "Failed"
            yield {
                "stage": "context_building",
                "status": "failed",
                "message": f"Context building error: {str(e)}"
            }

        refactored_files = {}
        pipeline_details = {}

        # -------------------------------------------------------------
        # Step 3: Execution Loop per file (Planner -> Refactor -> Completeness -> Validation -> Loop)
        # -------------------------------------------------------------
        for fname, fcontent in files.items():
            file_meta = parsed_metadata.get(fname, {"classes": [], "functions": [], "imports": [], "dependencies": [], "complexity_estimate": 1})
            file_timing = {}

            _, file_ext = os.path.splitext(fname.lower())
            if file_ext not in REFACTORABLE_EXTENSIONS:
                yield {
                    "file_name": fname,
                    "stage": "file_complete",
                    "status": "skipped",
                    "message": f"Skipped `{fname}` — non-refactorable type."
                }
                refactored_files[fname] = fcontent
                continue

            # Incremental Refactoring Check
            is_unchanged, content_hash = self.cache_manager.is_file_unchanged(fname, fcontent)

            file_report = {
                "rules": project_context.get("project_memory", {}).get("spec_md", "Loaded project memory rules."),
                "patterns_list": [],
                "patterns_report": "Analyzed via project memory.",
                "plan": {},
                "refactored_code": fcontent,
                "validation": {"success": True, "syntax_msg": "Passed.", "behavior_msg": "Passed."},
                "retries": [],
                "quality": {},
                "timing": file_timing
            }
            pipeline_details[fname] = file_report
            refactored_files[fname] = fcontent
            refactored_code = fcontent

            yield {
                "file_name": fname,
                "stage": "file_start",
                "status": "running",
                "message": f"Starting execution loop for `{fname}`..."
            }

            # 3.1 Build File Context
            file_ctx = ContextBuilder.build_file_context(
                fname, fcontent, project_context, parsed_metadata,
                rules_md=file_report["rules"], patterns_data={}
            )

            # 3.2 Planner Agent
            planner_start = time.perf_counter()
            yield {
                "file_name": fname,
                "stage": "planning",
                "status": "running",
                "message": "Generating structured task list..."
            }
            try:
                plan = self.planner.generate_plan(file_ctx)
                file_report["plan"] = plan
                plan_duration = round(time.perf_counter() - planner_start, 2)
                file_timing["Planner"] = f"{plan_duration} sec"

                yield {
                    "file_name": fname,
                    "stage": "planning",
                    "status": "completed",
                    "data": plan
                }
            except Exception as e:
                plan = {"should_refactor": True, "priority": "Medium", "confidence": 80, "steps_md": "Planning failed."}
                file_report["plan"] = plan
                file_timing["Planner"] = "Failed"

            # 3.3 Single-Pass Full-File Refactoring (Critical Issue #1 & #10)
            refactor_start = time.perf_counter()
            yield {
                "file_name": fname,
                "stage": "refactoring",
                "status": "running",
                "message": "Executing single-pass full-file refactoring..."
            }
            refactor_failed = False
            try:
                refactored_code = self.refactorer.execute_refactor(file_ctx, plan)
                file_report["refactored_code"] = refactored_code
                refactor_duration = round(time.perf_counter() - refactor_start, 2)
                file_timing["Refactor"] = f"{refactor_duration} sec"

                yield {
                    "file_name": fname,
                    "stage": "refactoring",
                    "status": "completed",
                    "data": refactored_code
                }
            except TooLargeForSinglePassError as e:
                # Not a failure to retry — this file structurally cannot fit the
                # single-pass strategy. Skip cleanly, keep the original content,
                # and tell the user clearly instead of silently corrupting it.
                refactored_code = fcontent
                file_report["refactored_code"] = fcontent
                file_report["validation"] = {
                    "success": None,
                    "syntax_msg": f"Skipped — too large for single-pass refactoring: {str(e)}",
                    "behavior_msg": "Not attempted."
                }
                file_timing["Refactor"] = "Skipped (too large)"
                yield {
                    "file_name": fname,
                    "stage": "refactoring",
                    "status": "skipped",
                    "message": f"Skipped `{fname}`: {str(e)}"
                }
                # Jump straight to quality/export bookkeeping — don't run this through
                # completeness/behavior validation or the retry loop, both of which
                # assume a genuine refactor attempt was made.
                pipeline_details[fname] = file_report
                refactored_files[fname] = fcontent
                continue
            except Exception as e:
                refactor_failed = True
                refactored_code = fcontent
                file_report["refactored_code"] = fcontent
                file_timing["Refactor"] = f"Failed ({str(e)})"
                yield {
                    "file_name": fname,
                    "stage": "refactoring",
                    "status": "failed",
                    "message": f"Refactoring failed: {str(e)}. Triggering Repair Loop..."
                }

            # 3.4 Completeness Validator & 8-Point Validation Suite (Critical Issue #4 & #5)
            val_start = time.perf_counter()
            yield {
                "file_name": fname,
                "stage": "validation",
                "status": "running",
                "message": "Running completeness check & 8-point validation suite..."
            }

            try:
                # Completeness Check First
                comp_ok, comp_msg = CompletenessValidator.validate(fname, fcontent, refactored_code)

                if comp_ok and not refactor_failed:
                    val_summary = self.validator.validate_full(fname, fcontent, refactored_code)
                    is_valid = val_summary["success"]
                    diag_msg = "\n".join(val_summary["diagnostics"])
                else:
                    is_valid = False
                    diag_msg = comp_msg if not comp_ok else f"Refactor Exception: {file_timing.get('Refactor')}"

                val_duration = round(time.perf_counter() - val_start, 2)
                file_timing["Validation"] = f"{val_duration} sec"
                file_report["validation"]["success"] = is_valid
                file_report["validation"]["syntax_msg"] = diag_msg

                def validator_wrapper(fn: str, code: str):
                    c_ok, c_msg = CompletenessValidator.validate(fn, fcontent, code)
                    if not c_ok:
                        return False, f"Completeness Check Failed: {c_msg}"
                    v_res = self.validator.validate_full(fn, fcontent, code)
                    return v_res["success"], "\n".join(v_res["diagnostics"])

                # 3.5 Execution Loop: Retry Branch on Failure
                if not is_valid:
                    retry_start = time.perf_counter()
                    yield {
                        "file_name": fname,
                        "stage": "retry",
                        "status": "running",
                        "message": f"Validation failed ({diag_msg[:60]}...)! Entering Retry Loop..."
                    }
                    file_timing["Retry"] = "Yes"

                    repaired_ok, repaired_code, retry_logs = self.repairer.attempt_auto_fix(
                        fname, fcontent, refactored_code, diag_msg, validator_wrapper
                    )
                    file_report["retries"] = retry_logs
                    file_report["validation"]["success"] = repaired_ok
                    retry_duration = round(time.perf_counter() - retry_start, 2)
                    file_timing["Retry_Time"] = f"{retry_duration} sec"

                    if repaired_ok:
                        refactored_code = repaired_code
                        file_report["refactored_code"] = refactored_code
                        yield {
                            "file_name": fname,
                            "stage": "retry",
                            "status": "completed",
                            "message": f"Auto-repair succeeded in {retry_duration}s."
                        }
                    else:
                        refactored_code = fcontent
                        file_report["refactored_code"] = fcontent
                        yield {
                            "file_name": fname,
                            "stage": "retry",
                            "status": "failed",
                            "message": "Auto-repair failed. Reverting to original source code."
                        }
                else:
                    file_timing["Retry"] = "No"
                    yield {
                        "file_name": fname,
                        "stage": "validation",
                        "status": "completed",
                        "message": f"Validation passed in {val_duration}s."
                    }
            except Exception as e:
                file_report["validation"]["success"] = False
                file_timing["Validation"] = "Failed"
                file_timing["Retry"] = "No"

            # 3.6 Quality Check Agent
            qual_start = time.perf_counter()
            yield {
                "file_name": fname,
                "stage": "quality",
                "status": "running",
                "message": "Calculating multi-dimensional quality scores..."
            }

            try:
                refactored_meta = ASTParser.parse_file(fname, refactored_code)
                quality = self.evaluator.evaluate_quality(fname, fcontent, refactored_code, file_meta, refactored_meta)
                file_report["quality"] = quality
                qual_duration = round(time.perf_counter() - qual_start, 2)
                file_timing["Quality"] = f"{qual_duration} sec"
                file_timing["Score"] = f"{quality.get('score_after', 85)}/100"

                yield {
                    "file_name": fname,
                    "stage": "quality",
                    "status": "completed",
                    "data": quality
                }
            except Exception as e:
                file_timing["Quality"] = "Failed"
                file_timing["Score"] = "75/100"
                file_report["quality"] = {"score_before": 70, "score_after": 70, "justification": f"Failed: {str(e)}"}

            refactored_files[fname] = refactored_code
            self.cache_manager.update_file_hash(fname, content_hash)

            yield {
                "file_name": fname,
                "stage": "file_complete",
                "status": "completed",
                "message": f"Execution loop finished for `{fname}`. Summary: Parser {file_timing.get('Parser', timing_stats.get('Parser'))} | Planner {file_timing.get('Planner')} | Refactor {file_timing.get('Refactor')} | Validation {file_timing.get('Validation')} | Retry {file_timing.get('Retry')} | Quality {file_timing.get('Score')}."
            }

        # -------------------------------------------------------------
        # Step 4: Export Agent
        # -------------------------------------------------------------
        yield {
            "stage": "export",
            "status": "running",
            "message": "Packaging structured ZIP export archive..."
        }

        try:
            packaged = self.exporter.package_refactored_project(files, refactored_files, pipeline_details)
            zip_bytes = self.exporter.build_zip_archive(packaged)

            yield {
                "stage": "export",
                "status": "completed",
                "message": "Project packaged successfully into Refactored_Project.zip layout.",
                "data": {
                    "packaged_files": packaged,
                    "zip_bytes": zip_bytes,
                    "reports": pipeline_details,
                    "timing_stats": timing_stats
                }
            }
        except Exception as e:
            simple_packaged = {f"refactored/{k}": v for k, v in refactored_files.items()}
            zip_bytes = self.exporter.build_zip_archive(simple_packaged)
            yield {
                "stage": "export",
                "status": "completed",
                "message": f"Fallback packaging completed ({str(e)}).",
                "data": {
                    "packaged_files": simple_packaged,
                    "zip_bytes": zip_bytes,
                    "reports": pipeline_details,
                    "timing_stats": timing_stats
                }
            }
