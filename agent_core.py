import os
import ollama
from typing import Dict, Any, Generator
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

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

    DOCUMENT_PROMPT = """Generate one professional README.
Include:
Overview | Project Structure | Installation | Requirements | Usage | Architecture | Folder Structure | API | Classes | Functions | Configuration | Future Improvements"""

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
 
 
    REFACTOR_PROMPT = """You are an expert Software Engineer specializing in code quality, refactoring, and clean code.
Analyze the provided codebase and suggest refactoring improvements.

You MUST format your response as a collection of separate file blocks.
1. The first block must contain a comprehensive report describing all code smells, DRY violations, SOLID violations, and the refactoring strategy for the files. Use this header exactly:
### File: refactoring_report.md
followed by your report.

2. For each source file you refactor, output a block starting exactly with this header:
### File: <relative_filepath>
followed by a fenced code block containing the fully refactored, production-ready code.

Ensure all refactored implementations compile, retain original logic behavior, and improve clean code metrics."""


    @staticmethod
    def list_available_models() -> list:
        res = ollama.list()
        if hasattr(res, "models"):
            return [m.model for m in res.models]
        elif isinstance(res, dict) and "models" in res:
            return [m.get("model", m.get("name", "")) for m in res["models"]]
        return []

    def __init__(self, model_name: str = "gemma4:26b"):
        self.model = model_name
        self.temperature = 0
        self.num_ctx = 16384
        self.num_predict = 1800
        
        try:
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to local Ollama daemon: {str(e)}")

    def generate_stream_response(self, system_instruction: str, user_code: str) -> Generator[str, None, None]:
        """
        NEW: Yields tokens chunk-by-chunk to keep the UI perfectly responsive.
        """
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_code}
        ]
        try:
            response_stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,  # Crucial for 26B model responsiveness
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict
                }
            )
            for chunk in response_stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"\nRuntime Error during streaming: {str(e)}"

    def _generate_local_response(self, system_instruction: str, user_code: str) -> str:
        # Fallback method kept for synchronous operations/summary steps
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_code}
        ]
        try:
            response = ollama.chat(
                model=self.model, messages=messages, stream=False,
                options={"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.num_predict}
            )
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content or ""
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return str(response)
        except Exception as e:
            raise RuntimeError(f"Error during model generation: {str(e)}")

    def review_code(self, code: str) -> str:
        return self._generate_local_response(self.REVIEW_PROMPT, code)

    def analyze_code(self, code: str) -> str:
        return self._generate_local_response(self.ANALYSIS_PROMPT, code)

    def document_code(self, code: str) -> str:
        return self._generate_local_response(self.DOCUMENT_PROMPT, code)

    def generate_cicd(self, code: str) -> str:
        return self._generate_local_response(self.CICD_PROMPT, code)

    def refactor_code(self, code: str) -> str:
        return self._generate_local_response(self.REFACTOR_PROMPT, code)

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
