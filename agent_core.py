import os
import re
import ollama
from typing import Dict, Any, Generator
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

REQUIRED_MODEL_NAME = "gemma4:26b"

def parse_stage1_output(text: str) -> list:
    if "No refactoring required" in text or not text.strip():
        return []
    
    findings_blocks = re.split(r"###\s*Finding\s*\d+", text, flags=re.IGNORECASE)
    findings = []
    
    for block in findings_blocks[1:]:
        block = block.strip()
        if not block:
            continue
        
        fields = {
            "Category": "",
            "Problem": "",
            "Evidence": "",
            "Current Code": "",
            "Recommendation": "",
            "Improved Code": "",
            "Implementation Notes": "",
            "Expected Benefit": "",
            "Estimated Effort": "",
            "Priority": ""
        }
        
        headers = ["Category", "Problem", "Evidence", "Current Code", "Recommendation", "Improved Code", "Implementation Notes", "Expected Benefit", "Estimated Effort", "Priority"]
        positions = {}
        for h in headers:
            match = re.search(r"(?:^|\n)\s*" + re.escape(h) + r"\s*(?:\r?\n|$)", block, re.IGNORECASE)
            if match:
                positions[h] = (match.start(), match.end())
        
        sorted_headers = sorted(positions.keys(), key=lambda x: positions[x][0])
        
        for idx, h in enumerate(sorted_headers):
            start_idx = positions[h][1]
            end_idx = positions[sorted_headers[idx+1]][0] if idx + 1 < len(sorted_headers) else len(block)
            content = block[start_idx:end_idx].strip()
            content = re.sub(r"^[-=\s]+", "", content)
            content = re.sub(r"(?:^|\n)\s*-{3,}\s*$", "", content)
            fields[h] = content.strip()
            
        findings.append(fields)
    return findings

def parse_stage2_output(text: str) -> dict:
    blocks = re.split(r"###\s*Finding\s*\d+", text, flags=re.IGNORECASE)
    findings_improvements = {}
    
    for idx, block in enumerate(blocks[1:]):
        block = block.strip()
        if not block:
            continue
        
        improved_code = ""
        impl_notes = ""
        
        imp_match = re.search(r"Improved\s*Code\s*([\s\S]*?)(?:Implementation\s*Notes|$)", block, re.IGNORECASE)
        notes_match = re.search(r"Implementation\s*Notes\s*([\s\S]*)", block, re.IGNORECASE)
        
        if imp_match:
            improved_code_raw = imp_match.group(1).strip()
            code_block_match = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", improved_code_raw)
            if code_block_match:
                improved_code = code_block_match.group(1).strip()
            else:
                improved_code = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\n", "", improved_code_raw)
                improved_code = re.sub(r"\n```$", "", improved_code).strip()
        if notes_match:
            impl_notes = notes_match.group(1).strip()
            impl_notes = re.sub(r"^[-\s]+", "", impl_notes).strip()
            divider_match = re.search(r"^(?:--------------------------------|###\s*\w+)", impl_notes, re.MULTILINE)
            if divider_match:
                impl_notes = impl_notes[:divider_match.start()].strip()
        
        findings_improvements[idx + 1] = {
            "Improved Code": improved_code,
            "Implementation Notes": impl_notes
        }
    return findings_improvements


class LocalCodeAgentEngine:
    """
    Core engine for interacting with the local Ollama daemon.
    Strictly separates agent orchestration from presentation logic.
    """
    
    # Expose prompts as class attributes so app.py can access them for streaming
    REVIEW_PROMPT = """You are a Principal Software Engineer.
Analyze the COMPLETE project. Do not analyze files independently. Understand relationships between files.
Return the report in Markdown.
Include:
# Executive Summary
# Critical Issues
# High Issues
# Medium Issues
# Low Issues
# Performance Problems
# Security Vulnerabilities
# Recommended Fixes

For every issue provide:
- File Name
- Description
- Impact
- Fix"""

    ANALYSIS_PROMPT = """You are a Software Architect.
Analyze the complete project.
Explain:
- Project Architecture
- Module Dependencies
- Complexity
- Design Patterns
- Performance Bottlenecks
- Memory Usage
- Scalability
Return a professional report."""

    DOCUMENT_FILE_PROMPT = """You are a Technical Writer and Software Engineer.
Generate detailed technical documentation for ONLY the provided file. Do NOT analyze the entire project.

File Name:
{file_name}

You MUST return your documentation strictly as a JSON object matching the schema below. Do not wrap the JSON output in markdown formatting except standard json fenced code blocks, and output no additional conversational text.

JSON Schema:
{{
  "file_name": "string",
  "purpose": "string",
  "responsibilities": ["string"],
  "imports": ["string"],
  "dependencies": ["string"],
  "classes": [
    {{
      "name": "string",
      "purpose": "string",
      "methods": [
        {{"name": "string", "purpose": "string", "arguments": ["string"], "returns": "string"}}
      ]
    }}
  ],
  "functions": [
    {{"name": "string", "purpose": "string", "arguments": ["string"], "returns": "string"}}
  ],
  "flow": "string describing execution call flow",
  "inputs": ["string"],
  "outputs": ["string"],
  "exceptions": [
    {{"type": "string", "description": "string"}}
  ],
  "configuration": ["string detailing configuration parameters/settings/secrets if any"],
  "example_usage": "string with snippet or guide",
  "future_improvements": ["string"]
}}"""

    GLOBAL_DOCS_PROMPT = """You are a Technical Writer and Software Architect.
Analyze the provided summaries of all modules in the project and generate a consulting-quality documentation package.

You MUST format your output as a collection of separate file blocks.
For each file you generate, start with a header exactly in the format:
### File: <filename>
followed by the markdown content.

You MUST generate the following files:
1. README.md
2. Architecture.md
3. ExecutionFlow.md
4. DependencyGraph.md
5. Configuration.md
6. API.md
7. FolderStructure.md (include a Current vs Recommended folder tree comparison)
8. DeploymentGuide.md
9. Troubleshooting.md
10. Glossary.md
11. UserGuide.md
12. DeveloperGuide.md

For FolderStructure.md, explicitly present:
- Current folder structure (derived from the metadata)
- Recommended folder structure (propose a clean, standard layout)
- Rationale for each change

Use Markdown prose directly after each file header; do not use code fences. Include an ASCII folder tree, dependency and execution-flow descriptions where relevant. Only state facts supported by the supplied metadata:
{metadata_summary}"""

    GLOBAL_REFACTOR_PROMPT = """You are a Principal Software Architect and Technical Director.

You have received detailed refactoring findings from multiple files across the project. Your job is to perform a cross-module, architecture-aware synthesis of these findings and produce a highly structured, project-level refactoring package.
This package must guide the engineering team on consolidating duplicate logic, introducing robust abstractions, and refactoring codebases systematically.

You MUST generate the following files:

1. ### File: REFACTORING_GUIDE.md
Provide an executive architectural summary. Outline the major technical debt trends discovered across the codebase (e.g., coupling of UI and logic, lack of caching, rendering bottlenecks, loose typing). Detail the high-level strategy for addressing these issues. Include a prioritized refactoring roadmap categorizing efforts into Phase 1 (Quick Wins / Critical), Phase 2 (Medium Effort / High Value), and Phase 3 (Structural / Architectural).

2. ### File: COMMON_FUNCTIONS.md
Systematically identify all duplicate helper functions, utility methods, validation logic, formatting code, and calculations across different files. Recommend specific target locations for extraction (e.g., utils/formatters.ts, services/api.ts). For each recommended utility:
- Define the input/output signature.
- Provide a brief specification.
- List all source files that should be updated to use this shared utility.

3. ### File: CUSTOM_HOOKS.md
Identify opportunities to extract stateful or side-effect-heavy logic into reusable custom hooks (e.g., useApiFetch, useFormValidation, useDebounce). For each proposed hook, explain:
- The problem it solves and why standard components should delegate to it.
- Its internal state and side effects.
- The precise API surface (inputs, outputs).
- List the components that will consume this hook.

4. ### File: COMMON_PATTERNS.md
Outline architectural pattern changes and software engineering best practices that should be applied globally across the codebase (e.g., transition to repository pattern for data access, structured error-boundary strategies, standardized loading/error state schemas for all views). Contrast the current sub-optimal patterns with the recommended clean patterns using conceptual code structures.

5. ### File: MIGRATION_PLAN.md
Provide a concrete, step-by-step technical plan for migrating the codebase from its current state to the refactored architecture. Address risk mitigation, dependency ordering (which modules to refactor first to prevent breaking down-stream dependencies), testing strategies (how to verify behavior remains identical), and roll-out recommendation.

6. ### File: REFACTORED_FILES.md
For every source file
Generate:
File Name
Reason for Refactoring
Complete Refactored Code
Migration Notes

The generated code must be production-ready.
The generated code must preserve behaviour.
Return the COMPLETE code.

You MUST format the output as separate Markdown file blocks. Every block starts exactly with:
### File: <filename>
then its Markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    REFACTOR_FILE_PROMPT = """CRITICAL INSTRUCTION

Your response is INVALID unless every finding contains ALL of the following sections:

Current Code
Recommendation
Improved Code
Implementation Notes
Expected Benefit
Estimated Effort

Never skip Improved Code.
Never summarize code.
Never write "...existing code..."
Generate COMPLETE replacement code.

If no improvements exist, explicitly state:
"No refactoring required."

Always finish with
### Complete Refactored File
Return ONLY the complete code.

You are a Principal Software Engineer and Enterprise Architect.
Analyze ONLY the provided file: {file_name}

Your goal is to identify concrete, architecture-aware, high-impact refactoring opportunities that improve maintainability, readability, scalability, and performance while strictly preserving behavior.
You must perform a deep layer-by-layer engineering analysis across the following dimensions:
1. Component Structure & Modularization:
   - Single Responsibility Principle (SRP) violations.
   - Candidates for splitting large files/components (e.g., components over 200 lines, nested markup).
   - Component rendering complexity (too many inline calculations, excessive prop drilling).
2. State & Hook Management (specifically for React/TS, or corresponding state patterns):
   - Redundant or derived state stored in useState instead of being calculated on the fly.
   - Duplicate or overlapping useEffect hooks.
   - Incorrect or missing dependency arrays in useEffect, useMemo, or useCallback.
   - Lack of cleanup functions in subscription/listener hooks.
   - Overuse of state triggering infinite render loops.
3. API, Data Ingestion, and Caching:
   - Duplicate, sequential, or un-batched API requests.
   - Missing caching layers, debouncing, or throttling for user interactions/search input.
   - Lack of robust error handling and loading indicators.
4. Logic Duplication and Extraction:
   - Business logic coupled to the presentation layer.
   - Utility or helper functions inside components that could be extracted to pure functions.
   - Shared algorithms or calculations that are candidates for domain services or shared utils.
5. Rendering Performance & Memoization:
   - Expensive calculations running on every render (missing useMemo).
   - Unnecessary re-renders caused by inline object/array references or inline callbacks (missing useCallback).
   - Missing virtualization for long lists/tables.
6. Code Quality, Typing, and Robustness:
   - Loose types (any, unknown) where strict types or interfaces are possible.
   - Code readability, naming conventions, magic numbers, or lack of self-documenting code.
   - Security vulnerabilities (e.g., unsafe input handling, sensitive information exposure).

Return your response in Markdown using the EXACT structure below. Ensure you do not omit any of the headers or the horizontal dividers (---), as downstream parsers rely on this specific syntax.

## File
{file_name}

### Finding 1

Category
[Category Name - e.g., Component Structure, State Management, API Layer, Logic Duplication, Performance, Security]

Problem
[Provide a rigorous, architecture-aware explanation of the code smell, anti-pattern, or performance bottleneck. Explain exactly why this is a concern in an enterprise application.]

Evidence
```[language]
[Paste the specific code snippet(s) from the file demonstrating the problem]
```

--------------------------------

Current Code
[Copy ONLY the exact code snippet from the uploaded file that needs refactoring. Do NOT modify it.]

--------------------------------

Recommendation
[Explain what should be improved.]

--------------------------------

Improved Code
[Generate the COMPLETE improved implementation.]

--------------------------------

Implementation Notes
[Explain why this implementation is better.]

--------------------------------

Expected Benefit
[Describe the expected benefit]

--------------------------------

Estimated Effort
[Provide an estimate of effort, e.g., Low (1-2 hours), Medium (half day), High (1-2 days)]

--------------------------------

Priority
[Critical, High, Medium, or Low]

--------------------------------

[Repeat the Findings block above for each additional finding, separating them with the 32-hyphen divider: --------------------------------]

--------------------------------

### Complete Refactored File

[Generate the COMPLETE updated file. Return ONLY code. Do NOT explain anything.]
"""

    REFACTOR_FILE_STAGE1_PROMPT = """You are a Principal Software Engineer and Enterprise Architect.
Analyze ONLY the provided file: {file_name}

Your goal is to identify concrete, architecture-aware, high-impact refactoring opportunities that improve maintainability, readability, scalability, and performance while strictly preserving behavior.
You must perform a deep layer-by-layer engineering analysis across the following dimensions:
1. Component Structure & Modularization
2. State & Hook Management
3. API, Data Ingestion, and Caching
4. Logic Duplication and Extraction
5. Rendering Performance & Memoization
6. Code Quality, Typing, and Robustness

If no improvements exist, explicitly state:
"No refactoring required."

Otherwise, return your response in Markdown using the EXACT structure below. Ensure you do not omit any of the headers or the horizontal dividers (---), as downstream parsers rely on this specific syntax.
Do NOT generate Improved Code, Implementation Notes, or the Complete Refactored File. Only identify the findings and recommendations.

## File
{file_name}

### Finding 1

Category
[Category Name]

Problem
[Provide a rigorous, architecture-aware explanation of the code smell, anti-pattern, or performance bottleneck.]

Evidence
```[language]
[Paste the specific code snippet(s) from the file demonstrating the problem]
```

--------------------------------

Current Code
[Copy ONLY the exact code snippet from the uploaded file that needs refactoring. Do NOT modify it.]

--------------------------------

Recommendation
[Explain what should be improved.]

--------------------------------

Expected Benefit
[Describe the expected benefit]

--------------------------------

Estimated Effort
[Low/Medium/High]

--------------------------------

Priority
[Critical/High/Medium/Low]

--------------------------------

[Repeat the Findings block above for each additional finding, separating them with the 32-hyphen divider: --------------------------------]
"""

    REFACTOR_FINDINGS_STAGE2_PROMPT = """You are a Principal Software Engineer and Enterprise Architect.
You are refactoring the file: {file_name}

Here is the list of findings/recommendations identified for this file:
{findings_summary}

For each finding, generate the following two sections:
1. Improved Code: Generate the COMPLETE improved implementation of the Current Code.
Rules:
- Preserve functionality.
- Do not use pseudo code.
- Do not use comments like "...existing code..."
- Generate production-ready code.
- Return the ENTIRE function/class.
- Include imports if required.
- The code should be directly replaceable.

2. Implementation Notes: Explain why this implementation is better.

Format your response EXACTLY as follows for each finding:

### Finding <N>
Improved Code
```[language]
[Complete improved code]
```

--------------------------------

Implementation Notes
[Explain why this implementation is better]
"""

    REFACTOR_FILE_ASSEMBLE_PROMPT = """You are a Principal Software Engineer.
Below is the original file and the list of refactoring improvements that need to be merged.
Generate the COMPLETE refactored file.

Original File:
{original_content}

Improvements:
{improvements}

CRITICAL INSTRUCTION:
Generate the COMPLETE replacement code.
Never skip code.
Never summarize code.
Never write "...existing code..."
Return ONLY the complete code.

Always finish with:
### Complete Refactored File
Return ONLY the complete code.
"""

    GLOBAL_REFACTORING_GUIDE_PROMPT = """You are a Principal Software Architect and Technical Director.
You have received detailed refactoring findings from multiple files across the project.
Generate the REFACTORING_GUIDE.md file.

Provide an executive architectural summary. Outline the major technical debt trends discovered across the codebase (e.g., coupling of UI and logic, lack of caching, rendering bottlenecks, loose typing). Detail the high-level strategy for addressing these issues. Include a prioritized refactoring roadmap categorizing efforts into Phase 1 (Quick Wins / Critical), Phase 2 (Medium Effort / High Value), and Phase 3 (Structural / Architectural).

You MUST format the output as a Markdown file block. Start exactly with:
### File: REFACTORING_GUIDE.md
followed by the markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    GLOBAL_COMMON_FUNCTIONS_PROMPT = """You are a Principal Software Architect and Technical Director.
You have received detailed refactoring findings from multiple files across the project.
Generate the COMMON_FUNCTIONS.md file.

Systematically identify all duplicate helper functions, utility methods, validation logic, formatting code, and calculations across different files. Recommend specific target locations for extraction (e.g., utils/formatters.ts, services/api.ts). For each recommended utility:
- Define the input/output signature.
- Provide a brief specification.
- List all source files that should be updated to use this shared utility.

You MUST format the output as a Markdown file block. Start exactly with:
### File: COMMON_FUNCTIONS.md
followed by the markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    GLOBAL_CUSTOM_HOOKS_PROMPT = """You are a Principal Software Architect and Technical Director.
You have received detailed refactoring findings from multiple files across the project.
Generate the CUSTOM_HOOKS.md file.

Identify opportunities to extract stateful or side-effect-heavy logic into reusable custom hooks (e.g., useApiFetch, useFormValidation, useDebounce). For each proposed hook, explain:
- The problem it solves and why standard components should delegate to it.
- Its internal state and side effects.
- The precise API surface (inputs, outputs).
- List the components that will consume this hook.

You MUST format the output as a Markdown file block. Start exactly with:
### File: CUSTOM_HOOKS.md
followed by the markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    GLOBAL_COMMON_PATTERNS_PROMPT = """You are a Principal Software Architect and Technical Director.
You have received detailed refactoring findings from multiple files across the project.
Generate the COMMON_PATTERNS.md file.

Outline architectural pattern changes and software engineering best practices that should be applied globally across the codebase (e.g., transition to repository pattern for data access, structured error-boundary strategies, standardized loading/error state schemas for all views). Contrast the current sub-optimal patterns with the recommended clean patterns using conceptual code structures.

You MUST format the output as a Markdown file block. Start exactly with:
### File: COMMON_PATTERNS.md
followed by the markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    GLOBAL_MIGRATION_PLAN_PROMPT = """You are a Principal Software Architect and Technical Director.
You have received detailed refactoring findings from multiple files across the project.
Generate the MIGRATION_PLAN.md file.

Provide a concrete, step-by-step technical plan for migrating the codebase from its current state to the refactored architecture. Address risk mitigation, dependency ordering (which modules to refactor first to prevent breaking down-stream dependencies), testing strategies (how to verify behavior remains identical), and roll-out recommendation.

You MUST format the output as a Markdown file block. Start exactly with:
### File: MIGRATION_PLAN.md
followed by the markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    GLOBAL_REFACTORED_FILES_PROMPT = """You are a Principal Software Architect and Technical Director.
You have received detailed refactoring findings from multiple files across the project.
Generate the REFACTORED_FILES.md file.

For every source file, generate:
- File Name
- Reason for Refactoring
- Complete Refactored Code
- Migration Notes

The generated code must be production-ready and preserve behaviour. Return the COMPLETE code.

You MUST format the output as a Markdown file block. Start exactly with:
### File: REFACTORED_FILES.md
followed by the markdown content, with no surrounding code fences or blocks. Do not wrap the file names in backticks, asterisks, or any other formatting characters.

Per-file findings:
{metadata_summary}"""

    REFACTORING_SPEC_PROMPT = """You are a Principal Software Architect and Lead Quality Engineer.
Generate a project-wide refactoring specification named REFACTORING_SPEC.md based on the provided per-file findings. This specification will act as the engineering standards document for all code modifications, ensuring all developers on the team adhere to identical conventions.

Your document must cover the following sections in detail:
1. React & Component Conventions (e.g., functional components, proper hooks placement, prop typing).
2. Hook & Side-Effect Conventions (e.g., custom hooks vs inline effects, dependency array strictness, cleanup guarantees).
3. State Management Rules (e.g., local state vs global context, avoiding derived state in state variables, state normalization).
4. Runtime & Rendering Performance Rules (e.g., when to use useMemo/useCallback/React.memo, virtualization criteria, avoiding inline objects).
5. Folder & Module Structure Rules (e.g., feature-based vs layer-based layout, strict import boundaries, forbidden circular dependencies).
6. Naming & Case Conventions (e.g., components, files, hooks, utils, variables, types/interfaces).
7. API & Integration Patterns (e.g., unified fetch wrappers, request/response interceptors, strict type definitions for all payloads).
8. Data Transformation & Validation Rules (e.g., parsing/validation at system boundaries, decoupling raw API models from UI models).

You MUST start your response exactly with the header:
### File: REFACTORING_SPEC.md
followed by the markdown content, with no surrounding code fences or formatting characters on the header line.

Per-file findings:
{metadata_summary}"""


    @staticmethod
    def list_available_models() -> list:
        res = ollama.list()
        if hasattr(res, "models"):
            return [m.model for m in res.models]
        elif isinstance(res, dict) and "models" in res:
            return [m.get("model", m.get("name", "")) for m in res["models"]]
        return []

    def __init__(self, model_name: str = REQUIRED_MODEL_NAME):
        self.model = model_name
        self.temperature = 0
        # The UI sends at most 8k characters per request (~2k tokens).  A
        # smaller context and concise response budget greatly reduce latency on
        # a local 26B model without truncating the input.
        self.num_ctx = 8192
        self.num_predict = 2048
        self.keep_alive = "15m"
        self._response_cache: Dict[tuple[str, str], str] = {}
        
        try:
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to local Ollama daemon: {str(e)}")

    def generate_stream_response(self, system_instruction: str, user_code: str, num_predict: int | None = None) -> Generator[str, None, None]:
        """
        NEW: Yields tokens chunk-by-chunk to keep the UI perfectly responsive.
        """
        output_limit = num_predict or self.num_predict
        cache_key = (system_instruction, user_code, str(output_limit))
        cached_response = self._response_cache.get(cache_key)
        if cached_response is not None:
            yield cached_response
            return

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_code}
        ]
        try:
            response_stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,  # Crucial for 26B model responsiveness
                think=False,  # Reports need answers, not a visible reasoning trace.
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": output_limit
                },
                keep_alive=self.keep_alive,
            )
            response_parts = []
            for chunk in response_stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    response_parts.append(content)
                    yield content
            self._response_cache[cache_key] = "".join(response_parts)
        except Exception as e:
            yield f"\nRuntime Error during streaming: {str(e)}"

    def _generate_local_response(self, system_instruction: str, user_code: str, num_predict: int | None = None) -> str:
        # Fallback method kept for synchronous operations/summary steps
        output_limit = num_predict or self.num_predict
        cache_key = (system_instruction, user_code, str(output_limit))
        cached_response = self._response_cache.get(cache_key)
        if cached_response is not None:
            return cached_response

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_code}
        ]
        try:
            response = ollama.chat(
                model=self.model, messages=messages, stream=False,
                think=False,
                options={"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": output_limit},
                keep_alive=self.keep_alive,
            )
            if hasattr(response, "message") and hasattr(response.message, "content"):
                content = response.message.content or ""
                self._response_cache[cache_key] = content
                return content
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
                self._response_cache[cache_key] = content
                return content
            content = str(response)
            self._response_cache[cache_key] = content
            return content
        except Exception as e:
            raise RuntimeError(f"Error during model generation: {str(e)}")

    def review_code(self, code: str) -> str:
        return self._generate_local_response(self.REVIEW_PROMPT, code)

    def analyze_code(self, code: str) -> str:
        return self._generate_local_response(self.ANALYSIS_PROMPT, code)

    def document_file(self, file_name: str, file_content: str) -> str:
        prompt = self.DOCUMENT_FILE_PROMPT.format(file_name=file_name)
        return self._generate_local_response(prompt, file_content)

    def generate_global_docs(self, metadata_summary: str) -> str:
        prompt = self.GLOBAL_DOCS_PROMPT.format(metadata_summary=metadata_summary)
        return self._generate_local_response(prompt, metadata_summary, num_predict=1200)

    def generate_global_refactor(self, metadata_summary: str) -> str:
        prompt = self.GLOBAL_REFACTOR_PROMPT.format(metadata_summary=metadata_summary)
        return self._generate_local_response(prompt, metadata_summary, num_predict=1200)


    def refactor_file(self, file_name: str, file_content: str) -> str:
        return self.refactor_file_two_stage(file_name, file_content, generate_complete_file=False)

    def refactor_file_two_stage(self, file_name: str, file_content: str, prev_spec: str = None, generate_complete_file: bool = False) -> str:
        # Import dynamically to avoid circular import issues
        from generators.cicd_generator import extract_complete_refactored_file
        
        # 1. Run Stage 1 to find recommendations
        stage1_prompt = ""
        if prev_spec:
            stage1_prompt = f"""You are a Principal Software Engineer.
Follow the rules in this REFACTORING_SPEC.md:
{prev_spec}

"""
        stage1_prompt += self.REFACTOR_FILE_STAGE1_PROMPT.format(file_name=file_name)
        
        stage1_output = self._generate_local_response(stage1_prompt, file_content, num_predict=2048)
        
        if "No refactoring required" in stage1_output or not stage1_output.strip():
            report = f"## File\n{file_name}\n\nNo refactoring required."
            if generate_complete_file:
                report += f"\n\n### Complete Refactored File\n\n```\n{file_content}\n```"
            return report

        # Parse findings from Stage 1
        findings = parse_stage1_output(stage1_output)
        if not findings:
            return stage1_output

        # 2. Run Stage 2 to get Improved Code and Implementation Notes for all findings in one call
        findings_summary_list = []
        for idx, finding in enumerate(findings):
            findings_summary_list.append(
                f"### Finding {idx+1}\n"
                f"Category: {finding.get('Category')}\n"
                f"Problem: {finding.get('Problem')}\n"
                f"Current Code:\n{finding.get('Current Code')}\n"
                f"Recommendation: {finding.get('Recommendation')}\n"
            )
        findings_summary = "\n\n".join(findings_summary_list)
        
        stage2_prompt = ""
        if prev_spec:
            stage2_prompt = f"Follow the rules in this REFACTORING_SPEC.md:\n{prev_spec}\n\n"
        stage2_prompt += self.REFACTOR_FINDINGS_STAGE2_PROMPT.format(
            file_name=file_name,
            findings_summary=findings_summary
        )
        
        stage2_output = self._generate_local_response(stage2_prompt, file_content, num_predict=4096)
        improvements_map = parse_stage2_output(stage2_output)
        
        assembled_findings = []
        improvements_summary = []
        
        for idx, finding in enumerate(findings):
            category = finding.get("Category", "Refactoring Opportunity")
            problem = finding.get("Problem", "")
            evidence = finding.get("Evidence", "")
            current_code = finding.get("Current Code", "")
            recommendation = finding.get("Recommendation", "")
            expected_benefit = finding.get("Expected Benefit", "")
            estimated_effort = finding.get("Estimated Effort", "")
            priority = finding.get("Priority", "")
            
            clean_current_code = current_code
            if clean_current_code.startswith("```"):
                code_match = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", clean_current_code)
                if code_match:
                    clean_current_code = code_match.group(1).strip()
            
            improvements = improvements_map.get(idx + 1, {})
            improved_code = improvements.get("Improved Code", "")
            impl_notes = improvements.get("Implementation Notes", "")
            
            # Assemble findings
            finding_md = f"""### Finding {idx+1}

Category
{category}

Problem
{problem}

Evidence
{evidence}

--------------------------------

Current Code
```
{clean_current_code}
```

--------------------------------

Recommendation
{recommendation}

--------------------------------

Improved Code
```
{improved_code}
```

--------------------------------

Implementation Notes
{impl_notes}

--------------------------------

Expected Benefit
{expected_benefit}

--------------------------------

Estimated Effort
{estimated_effort}

--------------------------------

Priority
{priority}"""
            assembled_findings.append(finding_md)
            improvements_summary.append(f"Finding {idx+1}:\n- Current Code:\n{clean_current_code}\n- Improved Code:\n{improved_code}\n")

        # Build final markdown report
        final_report = []
        final_report.append(f"## File\n{file_name}\n")
        final_report.append("\n\n--------------------------------\n\n".join(assembled_findings))
        
        # 3. Assemble the final complete refactored file only if requested
        if generate_complete_file:
            improvements_summary_str = "\n".join(improvements_summary)
            refactored_code = self._run_assembly(file_name, file_content, improvements_summary_str, prev_spec)
            final_report.append("\n\n--------------------------------\n\n### Complete Refactored File\n")
            final_report.append(f"```\n{refactored_code}\n```")
        
        return "\n".join(final_report)

    def _run_assembly(self, file_name: str, file_content: str, improvements_summary_str: str, prev_spec: str = None) -> str:
        from generators.cicd_generator import extract_complete_refactored_file
        prefix = ""
        if prev_spec:
            prefix = f"Follow the rules in this REFACTORING_SPEC.md:\n{prev_spec}\n\n"
        assembly_prompt = prefix + self.REFACTOR_FILE_ASSEMBLE_PROMPT.format(
            original_content=file_content,
            improvements=improvements_summary_str
        )
        raw_out = self._generate_local_response(assembly_prompt, file_content, num_predict=4096)
        
        has_header = False
        for header in ["Complete Refactored File", "Final Refactored Code", "Refactored File", "Merged Code"]:
            if header.lower() in raw_out.lower():
                has_header = True
                break
        
        if not has_header:
            retry_prompt = f"""You previously generated a response but did not include the '### Complete Refactored File' section.
Please generate the COMPLETE refactored file now. Merge all the improvements into the original file.
Return ONLY the code.

Original File:
{file_content}

Improvements:
{improvements_summary_str}

CRITICAL INSTRUCTION:
Return your response under the header '### Complete Refactored File' and include the complete code.
"""
            raw_out = self._generate_local_response(retry_prompt, file_content, num_predict=4096)
            
        refactored_code = extract_complete_refactored_file(raw_out)
        if not refactored_code:
            code_match = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", raw_out)
            if code_match:
                refactored_code = code_match.group(1).strip()
            else:
                refactored_code = raw_out.strip()
        return refactored_code

    def assemble_refactored_file(self, file_name: str, file_content: str, findings_text: str, prev_spec: str = None) -> str:
        findings = parse_stage1_output(findings_text)
        improvements_summary = []
        for idx, finding in enumerate(findings):
            current_code = finding.get("Current Code", "")
            improved_code = finding.get("Improved Code", "")
            
            clean_current_code = current_code
            if clean_current_code.startswith("```"):
                code_match = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", clean_current_code)
                if code_match:
                    clean_current_code = code_match.group(1).strip()
            
            clean_improved_code = improved_code
            if clean_improved_code.startswith("```"):
                code_match = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", clean_improved_code)
                if code_match:
                    clean_improved_code = code_match.group(1).strip()
            
            improvements_summary.append(f"Finding {idx+1}:\n- Current Code:\n{clean_current_code}\n- Improved Code:\n{clean_improved_code}\n")
            
        improvements_summary_str = "\n".join(improvements_summary)
        return self._run_assembly(file_name, file_content, improvements_summary_str, prev_spec)

    @lru_cache(maxsize=10)
    def scan_local_folder(self, folder_path: str) -> str:
        """
        Traverses a local folder path and aggregates supported code files into a
        structured payload, skipping dependency/build directories and binaries.
        """
        folder_path = folder_path.strip().strip('"').strip("'")
        folder_path = os.path.abspath(folder_path)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The path '{folder_path}' does not exist.")

        MAX_FILE_SIZE = 500 * 1024  # 500 KB

        ignored_dirs = {
            '.git',
            '.idea',
            '.vscode',
            '__pycache__',
            'node_modules',
            'env',
            'venv',
            '.venv',
            '.pytest_cache',
            'build',
            'dist',
            'target',
            '.next',
            '.cache',
            'coverage',
            'vendor'
        }
        valid_extensions = {
            # Python
            '.py',

            # JavaScript / TypeScript
            '.js', '.jsx', '.ts', '.tsx',

            # Java
            '.java',

            # C / C++
            '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',

            # C#
            '.cs',

            # Go
            '.go',

            # Rust
            '.rs',

            # PHP
            '.php',

            # Ruby
            '.rb',

            # Swift
            '.swift',

            # Kotlin
            '.kt',

            # SQL
            '.sql',

            # Web
            '.html', '.css', '.scss', '.sass',

            # Config files
            '.json', '.yaml', '.yml',
            '.xml', '.toml',
            '.ini', '.cfg',

            # Documentation
            '.md', '.txt',

            # Scripts
            '.sh', '.bat', '.ps1'
        }

        binary_extensions = {
            '.png',
            '.jpg',
            '.jpeg',
            '.gif',
            '.pdf',
            '.zip',
            '.rar',
            '.exe',
            '.dll',
            '.mp4',
            '.mp3',
            '.wav'
        }

        special_files = {
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "requirements.txt",
            "package.json",
            "package-lock.json",
            "pyproject.toml",
            ".gitignore",
            ".env",
            ".env.example",
            "README.md",
            "LICENSE"
        }

        # Collect candidate file paths first so they can be read in parallel
        file_tasks = []
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            for file in files:
                if file.endswith(".min.js"):
                    continue
                if file.endswith(".bundle.js"):
                    continue
                if file.endswith(".map"):
                    continue

                _, ext = os.path.splitext(file)
                if ext.lower() in binary_extensions:
                    continue

                if (
                    ext.lower() in valid_extensions
                    or file in special_files
                ):
                    full_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        continue
                    if size == 0 or size > MAX_FILE_SIZE:
                        continue
                    relative_path = os.path.relpath(full_path, folder_path)
                    file_tasks.append((full_path, relative_path))

        def read_file(task):
            full_path, relative_path = task
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(500000)
                return f"--- FILE: {relative_path} ---\n{content}\n"
            except Exception:
                return None

        aggregated_code = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for result in executor.map(read_file, file_tasks):
                if result:
                    aggregated_code.append(result)

        # Cap the number of files sent to the model
        aggregated_code = aggregated_code[:100]

        if not aggregated_code:
            return ""
        return "\n".join(aggregated_code)
