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

    GLOBAL_REFACTOR_PROMPT = """You are a Principal Software Architect.

You have received refactoring findings from multiple files. Analyze them to produce a project-level refactoring package.

Generate:

REFACTORING_GUIDE.md

COMMON_FUNCTIONS.md

CUSTOM_HOOKS.md

COMMON_PATTERNS.md

MIGRATION_PLAN.md

REFACTORING_SPEC.md

Detailed requirements for generated files:

1. COMMON_FUNCTIONS.md:
Identify duplicate helper functions, shared API logic, shared validation, shared formatting, shared table configuration, and shared calculations. Recommend extraction locations (e.g. hooks/usePCRAnalysis.ts, utils/formatters.ts).

2. CUSTOM_HOOKS.md:
Identify custom hooks that should be created (e.g. usePCRAnalysis(), useGridColumns(), useAnalysisHistory()) and explain why.

3. REFACTORING_SPEC.md:
Generate reusable engineering rules covering:
- React conventions
- Hook conventions
- State management rules
- Performance rules
- Folder structure rules
- Naming conventions
- Component structure
- API patterns
- Transformation rules

You MUST format the output as separate Markdown file blocks. Every block starts exactly with:
### File: <filename>
then its Markdown content, with no code fences.

Per-file findings:
{metadata_summary}"""

    CICD_PROMPT = """You are a Senior DevOps Engineer.
Analyze the uploaded/scanned codebase.
Generate production-ready CI/CD configuration files tailored to the detected tech stack.

You MUST format your response as a collection of separate file blocks. For each file you generate, start with a header exactly in the format:
### File: <relative_filepath>
followed by the code block.

Include:
1. ### File: Dockerfile
Generate a production-ready, multi-stage build Dockerfile. Make sure it runs as a non-root user for security and pins specific versions of base images.
2. ### File: docker-compose.yml
Generate a docker-compose.yml configuration with appropriate volume mounting, port mapping, environment variables, restart policies, and resource constraints/limits.
3. ### File: .github/workflows/ci.yml
Generate a GitHub Actions workflow for Continuous Integration (CI) that runs linting, formatting check, and automated tests. Use dependency caching for faster run times.
4. ### File: .github/workflows/cd.yml
Generate a GitHub Actions workflow for Continuous Deployment (CD) that builds the Docker image and pushes/deploys it to a production target using secure secrets.
5. ### File: README_CICD.md
Generate a README file describing the deployment architecture, required repository secrets (e.g., credentials, tokens), local run instructions, and production recommendations.

Ensure all configurations match the language, framework, dependencies, and requirements found in the analyzed code."""

    REFACTOR_FILE_PROMPT = """You are a Principal React/TypeScript Software Engineer.

Analyze ONLY the provided file.

Your goal is NOT to review code quality.

Your goal is to identify concrete refactoring opportunities that improve maintainability, readability and performance while preserving behaviour.

Focus on:

- Duplicate useEffect hooks
- Redundant API calls
- Missing useMemo
- Missing useCallback
- Incorrect dependency arrays
- Derived state stored in useState
- Repeated business logic
- Functions that should be extracted
- Reusable custom hooks
- Common helper methods
Analyze ONLY the provided file. Your goal is to identify concrete refactoring opportunities that improve maintainability, readability, and performance while preserving behaviour.

Perform a layer-by-layer engineering analysis focusing on:
- Layer 1 – Component Structure (multiple responsibilities, split candidates, large render, large files, repeated JSX)
- Layer 2 – React Hooks (duplicate useEffect, incorrect dependency arrays, stale closures, missing cleanup, missing useMemo/useCallback, derived state in useState, unnecessary updates, infinite render risks)
- Layer 3 – API Layer (duplicate/sequential requests, missing caching, repeated fetch logic, loading optimization, error handling)
- Layer 4 – Functions (duplicate helpers, redundant/dead functions, extraction candidates, shared routines, repeated business logic)
- Layer 5 – Rendering Performance (unnecessary re-renders, expensive calculations, inline objects/callbacks, missing React.memo, prop chains)
- Layer 6 – Business Logic (duplicate validation, transformations, sorting, filtering, repeated calculations)
- Layer 7 – Code Reuse (opportunities to extract custom hooks, utility functions, shared components, API services, constants, config modules)

Return your response in Markdown only using the exact structure below.

Format:

## File
{file_name}

### Finding 1

Category
[Category Name]

Problem
[Problem Description]

Evidence
[Code Evidence]

Recommendation
[Recommendation Detail]

Expected Benefit
[Expected Benefit]

Estimated Effort
[Estimated Effort]

--------------------------------

Repeat the findings block above for every finding.

--------------------------------

### Common Functions / Routines
List reusable functions.
Recommend extraction into:
hooks/
utils/
services/

--------------------------------

### Performance Improvements
List React-specific improvements.

--------------------------------

### Refactoring Priority
[Critical/High/Medium/Low]"""

    REFACTORING_SPEC_PROMPT = """You are a Principal Software Architect.
Generate a project-wide refactoring specification named REFACTORING_SPEC.md based on the provided per-file findings.

Your response should contain:
# REFACTORING_SPEC.md
- React conventions
- Hook conventions
- State management rules
- Performance rules
- Folder structure rules
- Naming conventions
- Component structure
- API patterns
- Transformation rules

You MUST start your response exactly with the header:
### File: REFACTORING_SPEC.md
followed by the markdown content, with no code fences.

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
        self.num_predict = 384
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

    def generate_cicd(self, code: str) -> str:
        return self._generate_local_response(self.CICD_PROMPT, code)

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
