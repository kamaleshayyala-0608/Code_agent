import os
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

class ScannerAgent:
    def __init__(self):
        self.max_file_size = 500 * 1024  # 500 KB
        self.ignored_dirs = {
            '.git', '.idea', '.vscode', '__pycache__', 'node_modules',
            'env', 'venv', '.venv', '.pytest_cache', 'build', 'dist',
            'target', '.next', '.cache', 'coverage', 'vendor'
        }
        self.valid_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp',
            '.cc', '.cxx', '.h', '.hpp', '.cs', '.go', '.rs', '.php',
            '.rb', '.swift', '.kt', '.sql', '.html', '.css', '.scss',
            '.sass', '.json', '.yaml', '.yml', '.xml', '.toml', '.ini',
            '.cfg', '.md', '.txt', '.sh', '.bat', '.ps1'
        }
        self.binary_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.rar',
            '.exe', '.dll', '.mp4', '.mp3', '.wav'
        }
        self.special_files = {
            "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
            "requirements.txt", "package.json", "package-lock.json",
            "pyproject.toml", ".gitignore", ".env", ".env.example",
            "README.md", "LICENSE"
        }

    def scan(self, folder_path: str) -> Dict[str, str]:
        """
        Traverses the folder path and returns a dictionary of {relative_path: content}.
        """
        folder_path = folder_path.strip().strip('"').strip("'")
        folder_path = os.path.abspath(folder_path)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The path '{folder_path}' does not exist.")

        file_tasks: List[Tuple[str, str]] = []
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]

            for file in files:
                if file.endswith(".min.js") or file.endswith(".bundle.js") or file.endswith(".map"):
                    continue

                _, ext = os.path.splitext(file)
                if ext.lower() in self.binary_extensions:
                    continue

                if ext.lower() in self.valid_extensions or file in self.special_files:
                    full_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        continue
                    if size == 0 or size > self.max_file_size:
                        continue
                    relative_path = os.path.relpath(full_path, folder_path)
                    file_tasks.append((full_path, relative_path))

        def read_file(task: Tuple[str, str]) -> Tuple[str, str] | None:
            full_path, relative_path = task
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(500000)
                return relative_path, content
            except Exception:
                return None

        scanned_files = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            for result in executor.map(read_file, file_tasks):
                if result:
                    rel_path, content = result
                    scanned_files[rel_path] = content

        # Prioritize real source code over schema dumps/data files, and cap generously —
        # real projects can easily have 400+ source files (this one has 439), so 100-300
        # was silently dropping legitimate application code. 2000 is a safety ceiling against
        # pathological uploads (e.g. an accidentally-included node_modules), not a normal limit.
        source_exts = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.php', '.rb', '.cs', '.cpp', '.c', '.h', '.hpp'}
        prioritized = sorted(scanned_files.keys(), key=lambda k: 0 if os.path.splitext(k)[1].lower() in source_exts else 1)
        keys = prioritized[:2000]
        if len(scanned_files) > len(keys):
            print(f"[ScannerAgent] Warning: {len(scanned_files) - len(keys)} files dropped beyond the {len(keys)} cap.")
        return {k: scanned_files[k] for k in keys}
