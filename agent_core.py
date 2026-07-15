import os
import ollama
from typing import Dict, Any, Generator
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

REQUIRED_MODEL_NAME = "gemma4:26b"

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

    REFACTOR_FILE_PROMPT = """You are a Principal Software Engineer and Enterprise Architect.
Analyze ONLY the provided file: {file_name}

IMPORTANT
For EVERY finding you MUST generate BOTH
1. Current Code
2. Improved Code

Never recommend a change without generating the replacement implementation.
The Improved Code must be complete and directly replaceable.
Never generate pseudo code.
Never omit imports.
Never shorten functions.
Always return production-ready code.

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

Copy ONLY the exact code snippet from the uploaded file that needs refactoring.

Do NOT modify it.

--------------------------------

Recommendation

Explain what should be improved.

--------------------------------

Improved Code

Generate the COMPLETE improved implementation.

Rules:
• Preserve functionality.
• Do not use pseudo code.
• Do not use comments like "...existing code..."
• Generate production-ready code.
• Return the ENTIRE function/class.
• Include imports if required.
• The code should be directly replaceable.

--------------------------------

Implementation Notes

Explain why this implementation is better.

--------------------------------

Expected Benefit

Estimated Effort
[Provide an estimate of effort, e.g., Low (1-2 hours), Medium (half day), High (1-2 days)]

--------------------------------

[Repeat the Findings block above for each additional finding, separating them with the 32-hyphen divider: --------------------------------]

--------------------------------

### Common Functions / Routines
[Identify all reusable helper functions, API integration logic, validation logic, formatting utilities, or custom state hooks. Explicitly specify the recommended target path and filename for extraction (e.g., hooks/useAuth.ts, utils/formatters.ts, services/api.ts) and describe the extracted API surface.]

--------------------------------

### Performance Improvements
[Provide a list of runtime performance optimizations. For React, detail React.memo, useMemo, useCallback usage, virtualized lists, or chunked rendering. For general code, detail algorithmic improvements, memory footprint reduction, or async optimization.]

--------------------------------

### Complete Refactored File

Merge ALL improvements into one final version.

Generate the COMPLETE updated file.

The generated file should compile successfully.

Return ONLY code.

Do NOT explain anything.

--------------------------------

### Refactoring Priority
[Indicate the overall priority: Critical, High, Medium, or Low, based on the highest priority finding discovered. Provide a one-sentence justification.]"""

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
        prompt = self.REFACTOR_FILE_PROMPT.format(file_name=file_name)
        return self._generate_local_response(prompt, file_content)

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
