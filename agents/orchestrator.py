import os
import traceback
from typing import Dict, Any, Generator
from agents.scanner_agent import ScannerAgent
from agents.parser_agent import ParserAgent
from agents.rule_agent import RuleExtractionAgent
from agents.pattern_agent import PatternRetrievalAgent
from agents.context_builder import ContextBuilder
from agents.planner_agent import PlannerAgent
from agents.refactoring_agent import RefactoringAgent
from agents.validation_agent import ValidationAgent
from agents.retry_agent import RetryAgent
from agents.quality_agent import QualityEvaluationAgent
from agents.export_agent import ExportAgent
from utils.ast_parser import ASTParser

class RefactoringOrchestrator:
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

    def refactor_project(self, files: Dict[str, str]) -> Generator[Dict[str, Any], None, None]:
        """
        Orchestrates the multi-agent refactoring pipeline file-by-file.
        Wraps every stage in defensive try/except to ensure fault tolerance.
        Every file is guaranteed to be saved and exported, falling back to original code on failure.
        """
        # Step 1: Parsing
        yield {
            "stage": "parsing",
            "status": "running",
            "message": "Parsing codebase structures and analyzing imports..."
        }
        
        try:
            project_metadata = self.parser.parse_project(files)
            yield {
                "stage": "parsing",
                "status": "completed",
                "message": "Project parsed successfully.",
                "data": project_metadata
            }
        except Exception as e:
            project_metadata = {fname: {"classes": [], "functions": [], "imports": [], "dependencies": [], "complexity_estimate": 1} for fname in files.keys()}
            yield {
                "stage": "parsing",
                "status": "failed",
                "message": f"Parsing failed: {str(e)}. Continuing with empty metadata."
            }

        refactored_files = {}
        pipeline_details = {}

        # Step 2: Process files
        for fname, fcontent in files.items():
            file_meta = project_metadata.get(fname, {"classes": [], "functions": [], "imports": [], "dependencies": [], "complexity_estimate": 1, "dependencies_context": {"depends_on": [], "depended_on_by": []}})
            
            # Pre-populate fallback reports and codes
            file_report = {
                "rules": "No rules applied.",
                "patterns_list": [],
                "patterns_report": "No patterns matched.",
                "plan": {"should_refactor": False, "priority": "Low", "confidence": 100, "steps_md": "No plan generated."},
                "refactored_code": fcontent,
                "validation": {"success": True, "syntax_msg": "Syntax check bypassed.", "behavior_msg": "Behavior equivalence check bypassed."},
                "retries": [],
                "quality": {
                    "score_before": 70,
                    "score_after": 70,
                    "orig_lines": len(fcontent.split("\n")),
                    "ref_lines": len(fcontent.split("\n")),
                    "lines_reduced": 0,
                    "orig_complexity": file_meta.get("complexity_estimate", 1),
                    "ref_complexity": file_meta.get("complexity_estimate", 1),
                    "justification": "Retained original code due to check bypass."
                }
            }
            
            pipeline_details[fname] = file_report
            refactored_files[fname] = fcontent # Default fallback
            refactored_code = fcontent

            yield {
                "file_name": fname,
                "stage": "file_start",
                "status": "running",
                "message": f"Starting multi-agent refactoring pipeline for `{fname}`..."
            }

            # 2.1 Rule Extraction (100% Local / Fault-tolerant)
            yield {
                "file_name": fname,
                "stage": "rule_extraction",
                "status": "running",
                "message": "Extracting context-relevant rules..."
            }
            try:
                rules = self.rule_agent.extract_relevant_rules(fname, fcontent, file_meta)
                file_report["rules"] = rules
                yield {
                    "file_name": fname,
                    "stage": "rule_extraction",
                    "status": "completed",
                    "data": rules
                }
            except Exception as e:
                rules = "## Applied Rules\n\nRule extraction failed."
                file_report["rules"] = rules
                yield {
                    "file_name": fname,
                    "stage": "rule_extraction",
                    "status": "failed",
                    "message": f"Rule extraction failed: {str(e)}. Continuing..."
                }

            # 2.2 Pattern Retrieval (100% Local / Fault-tolerant)
            yield {
                "file_name": fname,
                "stage": "pattern_retrieval",
                "status": "running",
                "message": "Analyzing for code smells and anti-patterns..."
            }
            try:
                patterns_data = self.pattern_agent.identify_patterns(fname, fcontent, file_meta)
                file_report["patterns_list"] = patterns_data["patterns"]
                file_report["patterns_report"] = patterns_data["report_md"]
                yield {
                    "file_name": fname,
                    "stage": "pattern_retrieval",
                    "status": "completed",
                    "data": patterns_data
                }
            except Exception as e:
                patterns_data = {"patterns": [], "report_md": "Pattern check failed."}
                file_report["patterns_list"] = []
                file_report["patterns_report"] = "Pattern check failed."
                yield {
                    "file_name": fname,
                    "stage": "pattern_retrieval",
                    "status": "failed",
                    "message": f"Pattern retrieval failed: {str(e)}. Continuing..."
                }

            # 2.3 Context Construction
            context = ContextBuilder.build_refactoring_context(fname, fcontent, file_meta, rules, patterns_data)
            context["patterns_list"] = file_report["patterns_list"]

            # 2.4 Planning (100% Local / Fault-tolerant)
            yield {
                "file_name": fname,
                "stage": "planning",
                "status": "running",
                "message": "Creating refactoring execution plan..."
            }
            try:
                plan = self.planner.generate_plan(context)
                file_report["plan"] = plan
                yield {
                    "file_name": fname,
                    "stage": "planning",
                    "status": "completed",
                    "data": plan
                }
            except Exception as e:
                plan = {"should_refactor": False, "priority": "Medium", "confidence": 80, "steps_md": "Planning failed."}
                file_report["plan"] = plan
                yield {
                    "file_name": fname,
                    "stage": "planning",
                    "status": "failed",
                    "message": f"Planning failed: {str(e)}. Continuing..."
                }

            # If the planner decides no refactoring is needed, we skip LLM call but run remaining stages
            if not plan.get("should_refactor", True):
                yield {
                    "file_name": fname,
                    "stage": "file_complete",
                    "status": "skipped",
                    "message": f"Skipped refactoring `{fname}` (no smells detected)."
                }
                # Keep file_report quality & refactored code matching original
                refactored_files[fname] = fcontent
                continue

            # 2.5 Refactoring Code (LLM Call)
            yield {
                "file_name": fname,
                "stage": "refactoring",
                "status": "running",
                "message": "Executing code refactoring via LLM..."
            }
            try:
                refactored_code = self.refactorer.execute_refactor(context, plan)
                file_report["refactored_code"] = refactored_code
                yield {
                    "file_name": fname,
                    "stage": "refactoring",
                    "status": "completed",
                    "data": refactored_code
                }
            except Exception as e:
                refactored_code = fcontent
                file_report["refactored_code"] = fcontent
                yield {
                    "file_name": fname,
                    "stage": "refactoring",
                    "status": "failed",
                    "message": f"Refactoring failed: {str(e)}. Reverting to original source."
                }
                # Even if refactoring fails, we continue with original file so export works
                refactored_files[fname] = fcontent
                continue

            # 2.6 Validation & Compiler Check
            yield {
                "file_name": fname,
                "stage": "validation",
                "status": "running",
                "message": "Running compiler syntax validation check..."
            }
            
            try:
                syntax_ok, syntax_msg = self.validator.check_syntax(fname, refactored_code)
                file_report["validation"]["syntax_msg"] = syntax_msg
                
                # Wrapper logic for the Retry Agent repair loops
                def validator_wrapper(fn: str, code: str):
                    s_ok, s_msg = self.validator.check_syntax(fn, code)
                    if not s_ok:
                        return False, s_msg
                    b_ok, b_msg = self.validator.validate_behavior(fn, fcontent, code)
                    if not b_ok:
                        return False, f"Behavioral Mismatch: {b_msg}"
                    return True, "Passed."

                # 2.7 Autonomous Retry Repair Loop
                if not syntax_ok:
                    yield {
                        "file_name": fname,
                        "stage": "retry",
                        "status": "running",
                        "message": "Syntax compilation failed! Initiating Retry Agent..."
                    }
                    repaired_ok, repaired_code, retry_logs = self.repairer.attempt_auto_fix(
                        fname, fcontent, refactored_code, syntax_msg, validator_wrapper
                    )
                    file_report["retries"] = retry_logs
                    file_report["validation"]["success"] = repaired_ok
                    
                    if repaired_ok:
                        refactored_code = repaired_code
                        file_report["refactored_code"] = refactored_code
                        file_report["validation"]["syntax_msg"] = "Auto-repaired successfully."
                        yield {
                            "file_name": fname,
                            "stage": "retry",
                            "status": "completed",
                            "message": "Auto-repair succeeded.",
                            "data": repaired_code
                        }
                    else:
                        file_report["validation"]["syntax_msg"] = f"Auto-repair failed: {syntax_msg}"
                        yield {
                            "file_name": fname,
                            "stage": "retry",
                            "status": "failed",
                            "message": "Auto-repair failed. Reverting to original source."
                        }
                        refactored_code = fcontent
                        file_report["refactored_code"] = fcontent
                else:
                    # Run behavioral equivalence validation
                    yield {
                        "file_name": fname,
                        "stage": "validation",
                        "status": "running",
                        "message": "Running behavioral equivalence verification..."
                    }
                    behavior_ok, behavior_msg = self.validator.validate_behavior(fname, fcontent, refactored_code)
                    file_report["validation"]["success"] = behavior_ok
                    file_report["validation"]["behavior_msg"] = behavior_msg
                    
                    if not behavior_ok:
                        yield {
                            "file_name": fname,
                            "stage": "retry",
                            "status": "running",
                            "message": "Behavior verification failed! Initiating Retry Agent..."
                        }
                        repaired_ok, repaired_code, retry_logs = self.repairer.attempt_auto_fix(
                            fname, fcontent, refactored_code, f"Behavior Mismatch: {behavior_msg}", validator_wrapper
                        )
                        file_report["retries"] = retry_logs
                        file_report["validation"]["success"] = repaired_ok
                        
                        if repaired_ok:
                            refactored_code = repaired_code
                            file_report["refactored_code"] = refactored_code
                            yield {
                                "file_name": fname,
                                "stage": "retry",
                                "status": "completed",
                                "message": "Auto-repair succeeded."
                            }
                        else:
                            yield {
                                "file_name": fname,
                                "stage": "retry",
                                "status": "failed",
                                "message": "Auto-repair failed. Reverting to original source."
                            }
                            refactored_code = fcontent
                            file_report["refactored_code"] = fcontent
                    else:
                        yield {
                            "file_name": fname,
                            "stage": "validation",
                            "status": "completed",
                            "message": "Equivalence verification completed."
                        }
            except Exception as e:
                file_report["validation"]["success"] = False
                file_report["validation"]["syntax_msg"] = f"Validation failed with exception: {str(e)}"
                yield {
                    "file_name": fname,
                    "stage": "validation",
                    "status": "failed",
                    "message": f"Validation step encountered an error: {str(e)}. Continuing..."
                }

            # 2.8 Quality Evaluation (100% Local / Fault-tolerant)
            yield {
                "file_name": fname,
                "stage": "quality",
                "status": "running",
                "message": "Running code quality metrics evaluation..."
            }
            try:
                refactored_meta = ASTParser.parse_file(fname, refactored_code)
                quality = self.evaluator.evaluate_quality(fname, fcontent, refactored_code, file_meta, refactored_meta)
                file_report["quality"] = quality
                yield {
                    "file_name": fname,
                    "stage": "quality",
                    "status": "completed",
                    "data": quality
                }
            except Exception as e:
                # Default fallback metrics
                file_report["quality"] = {
                    "score_before": 70,
                    "score_after": 70,
                    "orig_lines": len(fcontent.split("\n")),
                    "ref_lines": len(refactored_code.split("\n")),
                    "lines_reduced": len(fcontent.split("\n")) - len(refactored_code.split("\n")),
                    "reduction_pct": 0.0,
                    "orig_complexity": file_meta.get("complexity_estimate", 1),
                    "ref_complexity": file_meta.get("complexity_estimate", 1),
                    "justification": f"Quality analysis failed: {str(e)}"
                }
                yield {
                    "file_name": fname,
                    "stage": "quality",
                    "status": "failed",
                    "message": f"Quality evaluation failed: {str(e)}. Continuing..."
                }

            refactored_files[fname] = refactored_code
            
            yield {
                "file_name": fname,
                "stage": "file_complete",
                "status": "completed",
                "message": f"Pipeline finished for `{fname}`."
            }

        # Step 3: Exporting
        yield {
            "stage": "export",
            "status": "running",
            "message": "Packaging refactored files..."
        }
        
        try:
            packaged = self.exporter.package_refactored_project(files, refactored_files, pipeline_details)
            zip_bytes = self.exporter.build_zip_archive(packaged)
            
            yield {
                "stage": "export",
                "status": "completed",
                "message": "Project packaged successfully.",
                "data": {
                    "packaged_files": packaged,
                    "zip_bytes": zip_bytes,
                    "reports": pipeline_details
                }
            }
        except Exception as e:
            # Re-attempt basic packaging on export failure
            try:
                simple_packaged = {f"Refactored_Project/{k}": v for k, v in refactored_files.items()}
                zip_bytes = self.exporter.build_zip_archive(simple_packaged)
                yield {
                    "stage": "export",
                    "status": "completed",
                    "message": f"Fallback packaging succeeded. (Full report export failed: {str(e)})",
                    "data": {
                        "packaged_files": simple_packaged,
                        "zip_bytes": zip_bytes,
                        "reports": pipeline_details
                    }
                }
            except Exception as e2:
                yield {
                    "stage": "export",
                    "status": "failed",
                    "message": f"Export packaging failed: {str(e)} -> {e2}"
                }
