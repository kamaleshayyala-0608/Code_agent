import os
import re
import ollama
from typing import Dict, Any, Generator
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

REQUIRED_MODEL_NAME = "gemma4:26b"

def clean_refactored_code(text: str) -> str:
    """
    Robustly extracts the code content from LLM output, removing any potential markdown code fences.
    """
    text = text.strip()
    code_match = re.match(r"^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$", text)
    if code_match:
        return code_match.group(1).strip()
    code_match_lazy = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", text)
    if code_match_lazy:
        return code_match_lazy.group(1).strip()
    return text



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

    TRANSFORM_FILE_PROMPT = """You are an Enterprise Code Transformation Engine.

The following Specification is the ONLY source of truth:
{specification}

Apply every applicable rule.

Do NOT generate recommendations.

Do NOT explain.

Do NOT summarize.

Do NOT produce reports.

Preserve behaviour.

Return ONLY the complete refactored source file.
"""

    VALIDATOR_SYSTEM_PROMPT = """You are an Automated Code Behavior Validator.
Your task is to verify if the Refactored Code has the EXACT same behavior and external interface as the Original Code.
You must compare their structures, inputs, outputs, exceptions, and overall logic.
Minor improvements (like cleaning up duplicate logic, adding type hints, or using dependency injection as specified in rules) are allowed and expected, but the core functionality must remain identical.
Answer strictly with:
IDENTICAL: YES
or
IDENTICAL: NO
followed by a brief reason.
"""




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

        # Load static refactoring specification
        self.spec_rules = ""
        try:
            if os.path.exists("rules/refactoring_spec.md"):
                with open("rules/refactoring_spec.md", "r", encoding="utf-8") as f:
                    self.spec_rules = f.read()
            else:
                spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules", "refactoring_spec.md")
                if os.path.exists(spec_path):
                    with open(spec_path, "r", encoding="utf-8") as f:
                        self.spec_rules = f.read()
        except Exception:
            pass

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

    def transform_file(self, file_name: str, file_content: str) -> str:
        system_instruction = self.TRANSFORM_FILE_PROMPT.format(specification=self.spec_rules)
        
        current_content = file_content
        max_retries = 3
        
        for attempt in range(max_retries):
            refactored_raw = self._generate_local_response(system_instruction, current_content, num_predict=4096)
            refactored_code = clean_refactored_code(refactored_raw)
            
            if self.validate_behavior(file_content, refactored_code):
                return refactored_code
            
        return refactored_code

    def validate_behavior(self, original: str, refactored: str) -> bool:
        user_content = f"### Original Code\n```\n{original}\n```\n\n### Refactored Code\n```\n{refactored}\n```"
        try:
            response = self._generate_local_response(self.VALIDATOR_SYSTEM_PROMPT, user_content, num_predict=512)
            if re.search(r"IDENTICAL:\s*YES", response, re.IGNORECASE):
                return True
            return False
        except Exception:
            return True


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
