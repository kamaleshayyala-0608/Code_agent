import os
import time
import re
import hashlib
import json
import difflib
import concurrent.futures
import streamlit as st
from agent_core import LocalCodeAgentEngine, REQUIRED_MODEL_NAME
from generators.cicd_generator import (
    parse_markdown_file_blocks,
    parse_json_from_llm,
    build_doc_markdown,
    create_zip_from_dict
)

# Initialize session state cache and persistence variables
st.session_state.setdefault("llm_cache", {})
st.session_state.setdefault("pipeline_results", {})
st.session_state.setdefault("pipeline_executed", False)
st.session_state.setdefault("executed_payload_hash", "")
st.session_state.setdefault("executed_tasks", [])
st.session_state.setdefault("pipeline_duration", 0.0)

# ---------------------------------------------------------------------------
# OPTIMIZATION: Page layout (unchanged)
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Ollama Code Intelligence", page_icon="🧠")

@st.cache_resource
def get_agent_engine(model_name: str) -> LocalCodeAgentEngine:
    try:
        return LocalCodeAgentEngine(model_name=model_name)
    except ConnectionError as e:
        st.error(f"🔌 {str(e)}")
        st.info(
            "💡 **Setup Suggestion:**\n"
            "1. Ensure you have Ollama installed.\n"
            "2. Start the local Ollama daemon (runs on `http://localhost:11434` by default).\n"
            "3. Pull a model (e.g., `ollama pull gemma4:26b`) in your terminal.\n"
            "4. Refresh this page."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected Error during initialization: {str(e)}")
        st.stop()

def _content_hash(text: str) -> str:
    """Generate a SHA-256 hash to use as a stable cache key for code payloads."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def _split_payload_into_files(payload: str) -> dict:
    """
    Parses the standard codebase payload format into a dictionary of {file_name: content}.
    If the payload doesn't contain '--- FILE:', treats it as a single snippet named 'snippet.py'.
    """
    files = {}
    if "--- FILE:" not in payload:
        if payload.strip():
            files["snippet.py"] = payload
        return files
        
    parts = payload.split("--- FILE:")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        if len(lines) < 2:
            continue
        header = lines[0].strip()
        header = header.rstrip("-").strip()
        content = lines[1].strip()
        files[header] = content
    return files

def _extract_folder_structure(payload: str) -> str:
    """
    Extracts a folder tree representation from the codebase payload.
    Returns an ASCII tree of the current folder structure.
    """
    files = _split_payload_into_files(payload)
    if not files:
        return "No files detected."
    
    tree = {}
    for fname in files.keys():
        parts = fname.replace("\\", "/").split("/")
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None
    
    lines = ["."]
    def build_tree(node, prefix, is_last_list):
        entries = sorted(node.keys())
        for i, name in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")
            child = node[name]
            if child is not None:
                extension = "    " if is_last else "│   "
                build_tree(child, prefix + extension, [])
    
    build_tree(tree, "", [])
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# OPTIMIZATION: Split large payloads into chunks <= 8k chars for faster processing
# Small projects (< 8k chars) are not split.
# ---------------------------------------------------------------------------
CHUNK_CHAR_LIMIT = 8000
DEFAULT_FAST_CHAR_LIMIT = 20000
MODEL_NAME = REQUIRED_MODEL_NAME
FAST_MODE = True
# Ollama generally schedules requests for one loaded model serially. Submitting
# many 26B requests at once increases queueing and VRAM pressure instead of
# reducing total report time.
MAX_CONCURRENT_MODEL_REQUESTS = 1
GLOBAL_COMPILER_TOKENS = 12000

def _split_into_chunks(text: str, chunk_size: int = CHUNK_CHAR_LIMIT) -> list:
    """
    Split a very large code payload on file boundaries into smaller chunks.
    Keeps files intact and avoids mid-file splits where possible.
    """
    if len(text) <= chunk_size:
        return [text]
    parts = text.split("\n--- FILE:")
    reconstructed = []
    for i in range(1, len(parts)):
        parts[i] = "--- FILE:" + parts[i]
    current = ""
    for part in parts:
        if len(current) + len(part) > chunk_size and current:
            reconstructed.append(current.strip())
            current = part
        else:
            current += ("\n" if current else "") + part
    if current.strip():
        reconstructed.append(current.strip())
    return reconstructed

def _trim_payload(text: str, max_chars: int) -> str:
    """
    Keep the prompt small enough for local models to answer quickly.
    For scanned projects, prefer complete file blocks over cutting mid-file.
    """
    if len(text) <= max_chars:
        return text

    file_blocks = text.split("\n--- FILE:")
    if len(file_blocks) == 1:
        return text[:max_chars]

    # A single large file used to consume the entire fast-mode budget, so the
    # remaining uploaded files never reached Review, Analysis, or CI/CD. Give
    # every file a fair share of the prompt instead.
    blocks = [file_blocks[0]] + ["--- FILE:" + block for block in file_blocks[1:]]
    per_file_budget = max(1, max_chars // len(blocks))
    return "\n".join(block[:per_file_budget] for block in blocks)

def detect_project_stack(payload: str) -> dict:
    """
    Parses the codebase payload to detect Language, Framework, Inference, Deployment Target, and Package Manager.
    """
    detected = {
        "Language": "Unknown",
        "Framework": "None Detected",
        "Inference": "None Detected",
        "Deployment": "Local",
        "PackageManager": "None",
        "Recommendation": "Docker + GitHub Actions"
    }
    
    if not payload:
        return detected

    payload_lower = payload.lower()

    # 1. Detect Language & Package Manager
    has_python = ".py" in payload_lower or "requirements.txt" in payload_lower or "pyproject.toml" in payload_lower
    has_nodejs = "package.json" in payload_lower or ".js" in payload_lower or ".ts" in payload_lower
    has_java = ".java" in payload_lower or "pom.xml" in payload_lower or "build.gradle" in payload_lower

    if has_python:
        detected["Language"] = "Python"
        detected["PackageManager"] = "pip"
        if "pyproject.toml" in payload_lower:
            detected["PackageManager"] = "poetry"
    elif has_nodejs:
        detected["Language"] = "JavaScript/TypeScript"
        detected["PackageManager"] = "npm"
        if "yarn.lock" in payload_lower:
            detected["PackageManager"] = "yarn"
        elif "pnpm-lock.yaml" in payload_lower:
            detected["PackageManager"] = "pnpm"
    elif has_java:
        detected["Language"] = "Java"
        if "pom.xml" in payload_lower:
            detected["PackageManager"] = "Maven"
        else:
            detected["PackageManager"] = "Gradle"

    # 2. Detect Frameworks
    frameworks = []
    if "streamlit" in payload_lower:
        frameworks.append("Streamlit")
    if "flask" in payload_lower:
        frameworks.append("Flask")
    if "django" in payload_lower:
        frameworks.append("Django")
    if "fastapi" in payload_lower:
        frameworks.append("FastAPI")
    if "react" in payload_lower or "import react" in payload_lower or '"react"' in payload_lower:
        frameworks.append("React")

    if frameworks:
        detected["Framework"] = ", ".join(frameworks)

    # 3. Detect Inference
    if "ollama" in payload_lower:
        detected["Inference"] = "Ollama"
    elif "openai" in payload_lower:
        detected["Inference"] = "OpenAI API"
    elif "transformers" in payload_lower or "torch" in payload_lower:
        detected["Inference"] = "HuggingFace/PyTorch"

    # 4. Deployment & Recommendation
    primary_fw = frameworks[0] if frameworks else "Streamlit"
    
    recs = {
        "Streamlit": "Docker + GitHub Actions",
        "Flask": "Docker + Nginx",
        "FastAPI": "Docker + Uvicorn",
        "React": "GitHub Pages or Vercel",
        "Django": "Docker + Gunicorn"
    }
    
    detected["Recommendation"] = recs.get(primary_fw, "Docker + GitHub Actions")
    
    if detected["Inference"] == "Ollama":
        detected["Deployment"] = "Local"
    else:
        detected["Deployment"] = "Cloud"

    return detected

# ---------------------------------------------------------------------------
# Ollama Connection & Model Selection
# ---------------------------------------------------------------------------
try:
    available_models = LocalCodeAgentEngine.list_available_models()
except Exception as e:
    st.error("🔌 **Could not connect to local Ollama daemon**")
    st.info(
        "💡 **Troubleshooting Steps:**\n\n"
        "1. **Start Ollama:** Ensure the Ollama application or daemon is running on your system.\n"
        "2. **Check Port:** Ollama runs on `http://localhost:11434` by default.\n"
        "3. **Refresh:** Once Ollama is started, refresh this page."
    )
    st.stop()

if MODEL_NAME not in available_models:
    st.error(f"Required Ollama model not found: `{MODEL_NAME}`")
    st.info(
        "Start Ollama, then install the required model with "
        "`ollama pull gemma4:26b`, and refresh this page."
    )
    st.stop()

fast_mode = FAST_MODE
max_code_chars = DEFAULT_FAST_CHAR_LIMIT

engine = get_agent_engine(MODEL_NAME)

# ---------------------------------------------------------------------------
# Main UI Layout (unchanged)
# ---------------------------------------------------------------------------
st.title("🧠 Ollama-Powered Code Intelligence Agent")
st.markdown("A local enterprise-grade Code Review, Analysis, and Documentation tool.")
st.markdown("Analyze single snippets, batch-uploaded files, or an entire local POC repository.")
st.caption(f"Model: `{MODEL_NAME}` · concise mode enabled for faster reporting")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ingestion Panel")

    ingest_mode = st.radio(
        "Select Code Source Input Mode:",
        ["Direct Clipboard Paste", "Upload Multiple Files", "Upload Project ZIP", "Scan Local POC Folder Path"],
        horizontal=True
    )

    final_code_payload = ""

    # ---------------- MODE 1: SNIPPET PASTE ----------------
    if ingest_mode == "Direct Clipboard Paste":
        language = st.selectbox(
            "Code Language Profiling",
            ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "Other"]
        )
        final_code_payload = st.text_area(
            "Source Code Buffer",
            height=400,
            placeholder="Paste your source code here..."
        )

    # ---------------- MODE 2: BATCH FILE UPLOAD ----------------
    elif ingest_mode == "Upload Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose source files to upload",
            accept_multiple_files=True,
            type=[
                'py', 'js', 'jsx', 'ts', 'tsx', 'java', 'c', 'cpp', 'cc', 'cxx', 'h',
                'hpp', 'cs', 'go', 'rs', 'php', 'rb', 'swift', 'kt', 'sql', 'html',
                'css', 'scss', 'sass', 'json', 'yaml', 'yml', 'xml', 'toml', 'ini',
                'cfg', 'md', 'txt', 'sh', 'bat', 'ps1'
            ]
        )
        if uploaded_files:
            file_contents = []
            for uploaded_file in uploaded_files:
                try:
                    string_data = uploaded_file.read().decode("utf-8")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    string_data = uploaded_file.read().decode(
                        "latin-1",
                        errors="ignore"
                    )
                file_contents.append(f"--- FILE: {uploaded_file.name} ---\n{string_data}\n")
            final_code_payload = "\n".join(file_contents)
            st.success(f"Successfully processed {len(uploaded_files)} file(s) into context window.")

    # ---------------- MODE 2.5: PROJECT ZIP UPLOAD ----------------
    elif ingest_mode == "Upload Project ZIP":
        uploaded_zip = st.file_uploader(
            "Choose a project ZIP file to upload",
            type=["zip"]
        )
        if uploaded_zip:
            import zipfile
            import io
            file_contents = []
            try:
                zip_data = uploaded_zip.read()
                with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                    for zip_info in z.infolist():
                        if zip_info.is_dir():
                            continue
                        
                        fname = zip_info.filename
                        # Standard exclusion checks
                        if zip_info.file_size > 500 * 1024:  # 500 KB limit
                            continue
                            
                        parts = fname.replace("\\", "/").split("/")
                        if any(p.startswith(".") and p not in (".", "..") for p in parts):
                            continue
                        if any(p in {'node_modules', '__pycache__', 'env', 'venv', '.venv', 'build', 'dist', 'target', '.next', '.cache', 'coverage', 'vendor'} for p in parts):
                            continue
                            
                        _, ext = os.path.splitext(fname.lower())
                        valid_extensions = {
                            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
                            '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.sql', '.html', '.css', '.scss',
                            '.sass', '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg', '.md', '.txt',
                            '.sh', '.bat', '.ps1'
                        }
                        if ext not in valid_extensions:
                            continue
                            
                        try:
                            string_data = z.read(zip_info).decode("utf-8")
                        except UnicodeDecodeError:
                            string_data = z.read(zip_info).decode("latin-1", errors="ignore")
                            
                        file_contents.append(f"--- FILE: {fname} ---\n{string_data}\n")
                final_code_payload = "\n".join(file_contents)
                st.success(f"Successfully processed {len(file_contents)} files from project ZIP into context window.")
            except Exception as e:
                st.error(f"Failed to read ZIP file: {str(e)}")

    # ---------------- MODE 3: LOCAL POC DIRECTORY ----------------
    elif ingest_mode == "Scan Local POC Folder Path":
        folder_path = st.text_input(
            "Enter absolute local folder directory path:",
            placeholder="/Users/username/workspace/my-poc-project"
        )
        if folder_path:
            try:
                progress = st.progress(0)
                # OPTIMIZATION: Track scan timing in session_state for the Reporting Panel
                scan_start = time.perf_counter()
                with st.spinner("Traversing directories and reading files..."):
                    with concurrent.futures.ThreadPoolExecutor() as ex:
                        fut = ex.submit(engine.scan_local_folder, folder_path)
                        pct = 0
                        while not fut.done():
                            pct = (pct + 5) % 95
                            progress.progress(pct / 100.0)
                            time.sleep(0.15)
                        final_code_payload = fut.result()
                scan_elapsed = time.perf_counter() - scan_start
                st.session_state._last_scan_time = scan_elapsed
                progress.progress(1.0)
                if final_code_payload:
                    file_count = final_code_payload.count("--- FILE:")
                    st.success(f"Successfully loaded {file_count} files.")
                else:
                    st.warning(
                        "No supported source files were found. The folder may contain only binary files or unsupported formats."
                    )
            except Exception as err:
                st.error(f"Error checking directory: {err}")

    # Detected stack summary display
    if final_code_payload.strip():
        st.markdown("---")
        st.subheader("🔍 Detected Architecture Stack")
        stack = detect_project_stack(final_code_payload)
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"**Language:** `{stack['Language']}`")
            st.markdown(f"**Framework:** `{stack['Framework']}`")
            st.markdown(f"**Inference:** `{stack['Inference']}`")
        with m2:
            st.markdown(f"**Deployment:** `{stack['Deployment']}`")
            st.markdown(f"**Package Manager:** `{stack['PackageManager']}`")
            st.markdown(f"**Recommendation:** `{stack['Recommendation']}`")

    # Task toggles
    st.markdown("### Execution Tasks")
    do_review = st.checkbox("Review Code (Audit & Security)", value=True)
    do_analyze = st.checkbox("Analyze Code (Complexity & Flow)", value=False)
    do_document = st.checkbox("Document Code (README Generation)", value=False)
    do_refactor = st.checkbox("Refactor Code (Readability & Design)", value=False)

    execute = st.button("🚀 Execute Pipeline", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Helper to render download controls and reports for each task
# ---------------------------------------------------------------------------
def _render_task_controls(task_name: str, task_data, active_code_payload: str, files_to_process: dict = None):
    if task_name == "Documentation":
        # task_data is a dict of {filepath: content}
        if task_data:
            zip_bytes = create_zip_from_dict(task_data)
            st.download_button(
                label="📥 Download Documentation Package (ZIP)",
                data=zip_bytes,
                file_name="documentation.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_doc"
            )
        
        # Render global documents in expanders
        for filename in ["docs/README.md", "docs/Architecture.md", "docs/API.md", "docs/FolderStructure.md"]:
            if filename in task_data:
                display_name = filename.replace("docs/", "")
                with st.expander(f"📖 {display_name}", expanded=(display_name == "README.md")):
                    st.markdown(task_data[filename])
                    
        # Render per-module documentation
        with st.expander("📁 Module Specifications (Per-File docs)", expanded=False):
            for filename, content in task_data.items():
                if filename.startswith("docs/modules/"):
                    module_name = filename.replace("docs/modules/", "")
                    st.markdown(f"### `{module_name}`")
                    st.markdown(content)
                    st.markdown("---")
                    
    elif task_name == "Refactor":
        if task_data:
            for full_key, content in task_data.items():
                fname = full_key.replace("Refactored_Project/", "")
                if fname == "spec.md":
                    continue

                st.markdown(f"### 📄 `{fname}`")
                _, ext = os.path.splitext(fname)
                lang = ext.replace(".", "")
                if lang in ("tsx", "jsx"):
                    lang = "typescript"
                elif lang == "js":
                    lang = "javascript"
                elif lang == "py":
                    lang = "python"

                st.code(content, language=lang)
                st.markdown("---")

            zip_bytes = create_zip_from_dict(task_data)
            st.download_button(
                label="📥 Download Refactored_Project.zip",
                data=zip_bytes,
                file_name="Refactored_Project.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_refactor"
            )


    elif task_name in ("Review", "Analysis"):
        if task_data:
            st.download_button(
                label=f"📥 Download {task_name} Report",
                data=task_data,
                file_name=f"{task_name.lower()}_report.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"download_{task_name.lower()}"
            )

# ---------------------------------------------------------------------------
# REPORTING PANEL (col2)
# ---------------------------------------------------------------------------
with col2:
    st.subheader("Reporting Panel")

    current_hash = _content_hash(final_code_payload)
    current_tasks = [do_review, do_analyze, do_document, do_refactor]

    # Clear persistence state if payload or selected tasks change
    if (st.session_state.get("executed_payload_hash") != current_hash 
        or st.session_state.get("executed_tasks") != current_tasks):
        st.session_state.pipeline_executed = False

    should_generate = execute
    if should_generate:
        if not final_code_payload.strip():
            st.warning("⚠️ Please provide source code in the buffer before executing.")
            should_generate = False
        elif not any(current_tasks):
            st.warning("Please select at least one execution task.")
            should_generate = False

    if should_generate:
        active_code_payload = final_code_payload
        if fast_mode and len(active_code_payload) > max_code_chars:
            original_chars = len(active_code_payload)
            active_code_payload = _trim_payload(active_code_payload, int(max_code_chars))
            st.warning(
                f"Fast mode is analyzing {len(active_code_payload):,} of "
                f"{original_chars:,} characters."
            )

        # Standard chunks for global/monolithic tasks
        if len(active_code_payload) < CHUNK_CHAR_LIMIT:
            chunks = [active_code_payload]
        else:
            chunks = _split_into_chunks(active_code_payload, CHUNK_CHAR_LIMIT)

        # Split into individual files for refactoring/documentation
        files_to_process = _split_payload_into_files(active_code_payload)

        start_total = time.perf_counter()
        pipeline_results = st.session_state.get("pipeline_results", {})

        # 1. Run Monolithic Tasks (Review, Analysis, CI/CD)
        mono_configs = []
        if do_review:
            mono_configs.append(("Review", engine.REVIEW_PROMPT, "### 🛡️ Code Review Audit"))
        if do_analyze:
            mono_configs.append(("Analysis", engine.ANALYSIS_PROMPT, "### 🏗️ Architecture & Complexity Analysis"))

        for task_name, system_prompt, header_text in mono_configs:
            st.markdown(header_text)
            with st.container(border=True):
                combined_result_chunks = []
                for idx, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        st.caption(f"Processing part {idx+1} of {len(chunks)}...")
                    
                    try:
                        stream_generator = engine.generate_stream_response(system_prompt, chunk)
                        output_text = st.write_stream(stream_generator)
                        if output_text:
                            combined_result_chunks.append(output_text)
                    except Exception as e:
                        st.error(f"Error during streaming: {str(e)}")
                
                full_task_text = "\n\n".join(combined_result_chunks)
                pipeline_results[task_name] = full_task_text
                _render_task_controls(task_name, full_task_text, active_code_payload)

        # 2. Run Refactoring Pipeline (Multi-Agent progress visualizer)
        if do_refactor:
            st.markdown("### ⚙️ Multi-Agent Refactoring Pipeline")
            refactor_results = {}
            pipeline_results["Refactor"] = refactor_results
            pipeline_results["Refactor_Reports"] = {}

            if not files_to_process:
                st.warning("No files found to refactor.")
            else:
                # Progress Step Badges (Item 11)
                step_cols = st.columns(7)
                step_badges = {
                    "Scanning": step_cols[0].empty(),
                    "Context": step_cols[1].empty(),
                    "Planning": step_cols[2].empty(),
                    "Refactoring": step_cols[3].empty(),
                    "Validation": step_cols[4].empty(),
                    "Retry": step_cols[5].empty(),
                    "Packaging": step_cols[6].empty(),
                }

                def update_step_ui(completed_steps=None):
                    completed_steps = completed_steps or set()
                    labels = ["Scanning", "Context", "Planning", "Refactoring", "Validation", "Retry", "Packaging"]
                    for idx, name in enumerate(labels):
                        icon = "✔" if name in completed_steps else "⏳"
                        status_str = f"**{name}** {icon}"
                        step_badges[name].markdown(status_str)

                completed_steps = set()
                update_step_ui(completed_steps)

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                log_expander = st.expander("🛠️ Detailed Execution & Timing Logs", expanded=True)
                log_area = log_expander.empty()

                log_messages = []
                file_keys = list(files_to_process.keys())
                total_files = len(file_keys)

                pipeline_gen = engine.run_multi_agent_pipeline(files_to_process)

                for update in pipeline_gen:
                    stage = update.get("stage")
                    status = update.get("status")
                    msg = update.get("message", "")
                    fname = update.get("file_name", "")

                    if stage == "parsing" and status == "completed":
                        completed_steps.add("Scanning")
                    elif stage == "context_building" and status == "completed":
                        completed_steps.add("Context")
                    elif stage == "planning" and status == "completed":
                        completed_steps.add("Planning")
                    elif stage == "refactoring" and status == "completed":
                        completed_steps.add("Refactoring")
                    elif stage == "validation" and status == "completed":
                        completed_steps.add("Validation")
                    elif stage == "retry":
                        completed_steps.add("Retry")
                    elif stage == "export" and status == "completed":
                        completed_steps.add("Packaging")

                    update_step_ui(completed_steps)

                    if fname in file_keys:
                        current_file_idx = file_keys.index(fname)
                        stage_weights = {
                            "file_start": 0.0,
                            "planning": 0.2,
                            "refactoring": 0.5,
                            "validation": 0.75,
                            "retry": 0.85,
                            "quality": 0.95,
                            "file_complete": 1.0
                        }
                        weight = stage_weights.get(stage, 0.0)
                        progress_val = (current_file_idx + weight) / total_files
                        progress_bar.progress(min(1.0, max(0.0, progress_val)))

                    log_prefix = "🤖"
                    if stage == "parsing":
                        log_prefix = "🔍 [Scanner]"
                    elif stage == "context_building":
                        log_prefix = "🌐 [Context Builder]"
                    elif stage == "planning":
                        log_prefix = "📋 [Planner]"
                    elif stage == "refactoring":
                        log_prefix = "⚙️ [5-Pass Refactor]"
                    elif stage == "validation":
                        log_prefix = "🧪 [8-Point Validator]"
                    elif stage == "retry":
                        log_prefix = "🩹 [Retry Loop]"
                    elif stage == "quality":
                        log_prefix = "📊 [Quality Agent]"
                    elif stage == "export":
                        log_prefix = "📥 [Export Agent]"
                        progress_bar.progress(1.0)

                    if msg:
                        log_line = f"{log_prefix} {msg}"
                        log_messages.append(log_line)
                        status_text.markdown(f"**Current Execution:** {msg}")
                        log_area.code("\n".join(log_messages[-14:]))

                    if status == "completed" or status == "skipped":
                        if stage == "export":
                            export_data = update.get("data", {})
                            packaged_files = export_data.get("packaged_files", {})
                            reports = export_data.get("reports", {})

                            for pk, code in packaged_files.items():
                                refactor_results[pk] = code

                            pipeline_results["Refactor_Reports"] = reports

                completed_steps.update(["Scanning", "Context", "Planning", "Refactoring", "Validation", "Retry", "Packaging"])
                update_step_ui(completed_steps)

                status_text.success("🎉 Enterprise Execution Loop completed successfully!")
                progress_bar.empty()

                _render_task_controls("Refactor", refactor_results, active_code_payload, files_to_process)

        # 3. Run Technical Documentation Pipeline (File-by-file + Global compiler)
        if do_document:
            st.markdown("### 📝 Technical Documentation")
            doc_results = {}
            pipeline_results["Documentation"] = doc_results

            if not files_to_process:
                st.warning("No files found to document.")
            else:
                parsed_jsons = {}
                if MAX_CONCURRENT_MODEL_REQUESTS > 1:
                    with st.spinner("Documenting files in parallel..."):
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MODEL_REQUESTS) as executor:
                            future_to_file = {}
                            for fname, fcontent in files_to_process.items():
                                prompt = engine.DOCUMENT_FILE_PROMPT.format(file_name=fname)
                                future = executor.submit(engine._generate_local_response, prompt, fcontent)
                                future_to_file[future] = fname
                                
                            for future in concurrent.futures.as_completed(future_to_file):
                                fname = future_to_file[future]
                                try:
                                    raw_output = future.result()
                                    parsed = parse_json_from_llm(raw_output)
                                    if "error" in parsed:
                                        st.error(f"JSON schema error for {fname}: {parsed['error']}")
                                    parsed_jsons[fname] = parsed
                                    doc_results[f"docs/modules/{fname}.md"] = build_doc_markdown(parsed)
                                except Exception as e:
                                    st.error(f"Failed to parse JSON for {fname}: {e}")
                                    doc_results[f"docs/modules/{fname}.md"] = f"Error documenting {fname}: {str(e)}"
                else:
                    for fname, fcontent in files_to_process.items():
                        st.caption(f"Documenting `{fname}`...")
                        prompt = engine.DOCUMENT_FILE_PROMPT.format(file_name=fname)
                        with st.container(border=True):
                            raw_output = engine._generate_local_response(prompt, fcontent)
                            st.write("✓ Analysis completed.")
                            try:
                                parsed = parse_json_from_llm(raw_output)
                                if "error" in parsed:
                                    st.error(f"JSON schema error for {fname}: {parsed['error']}")
                            except Exception as e:
                                st.error(f"Failed to parse JSON for {fname}: {e}")
                                continue
                            parsed_jsons[fname] = parsed
                            doc_results[f"docs/modules/{fname}.md"] = build_doc_markdown(parsed)

                # Compile summaries list
                folder_structure = _extract_folder_structure(active_code_payload)
                meta_summaries = []
                for fname, parsed in parsed_jsons.items():
                    purpose = parsed.get("purpose", "")
                    classes_list = [c.get("name", "") for c in parsed.get("classes", [])] if isinstance(parsed.get("classes"), list) else []
                    funcs_list = [f.get("name", "") for f in parsed.get("functions", [])] if isinstance(parsed.get("functions"), list) else []
                    deps_list = parsed.get("dependencies", []) if isinstance(parsed.get("dependencies"), list) else []
                    imports_list = parsed.get("imports", []) if isinstance(parsed.get("imports"), list) else []
                    config_list = parsed.get("configuration", []) if isinstance(parsed.get("configuration"), list) else []
                    flow = parsed.get("flow", "")
                    
                    meta_summaries.append(
                        f"Module: {fname}\n"
                        f"Purpose: {purpose}\n"
                        f"Imports: {', '.join(imports_list)}\n"
                        f"Dependencies: {', '.join(deps_list)}\n"
                        f"Classes: {', '.join(classes_list)}\n"
                        f"Functions: {', '.join(funcs_list)}\n"
                        f"Execution Flow: {flow}\n"
                        f"Configurations: {', '.join(config_list)}\n"
                        f"---"
                    )
                meta_summaries.append(f"Current Folder Structure:\n{folder_structure}")
                metadata_summary_text = "\n".join(meta_summaries)

                # Compile overall global docs
                st.caption("Compiling comprehensive global documentation...")
                prompt_global = engine.GLOBAL_DOCS_PROMPT.format(metadata_summary=metadata_summary_text)
                
                with st.container(border=True):
                    try:
                        stream_generator = engine.generate_stream_response(prompt_global, metadata_summary_text)
                        raw_global_output = st.write_stream(stream_generator)
                        if not raw_global_output:
                            raw_global_output = ""
                    except Exception as e:
                        st.error(f"Error during documentation compilation: {str(e)}")
                        raw_global_output = ""

                global_files = parse_markdown_file_blocks(raw_global_output)
                for path, content in global_files.items():
                    if not path.startswith("docs/"):
                        path = f"docs/{path}"
                    doc_results[path] = content

                if "docs/README.md" not in doc_results:
                    doc_results["docs/README.md"] = raw_global_output

                _render_task_controls("Documentation", doc_results, active_code_payload)

        total_elapsed = time.perf_counter() - start_total
        st.session_state.pipeline_results = pipeline_results
        st.session_state.pipeline_duration = total_elapsed
        st.session_state.executed_payload_hash = current_hash
        st.session_state.executed_tasks = current_tasks
        st.session_state.pipeline_executed = True

        # Execution Summary
        st.markdown("### 📊 Execution Summary")
        completed_tasks = []
        if do_review:
            completed_tasks.append("✔ Review Completed")
        if do_analyze:
            completed_tasks.append("✔ Analysis Completed")
        if do_document:
            completed_tasks.append("✔ Documentation Completed")
        if do_refactor:
            completed_tasks.append("✔ Refactor Completed")
        
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            for task in completed_tasks:
                st.markdown(task)
        with col_sum2:
            st.markdown(f"**Total Files:** {len(files_to_process)}")
            st.markdown(f"**Characters:** {len(active_code_payload):,}")
            st.markdown(f"**Chunks:** {len(chunks)}")
            st.markdown(f"**Pipeline Time:** {total_elapsed:.1f} sec")

        st.markdown("### ⏱️ Performance Metrics")
        scan_time = st.session_state.get("_last_scan_time", 0)
        st.metric(label="Folder Scan", value=f"{scan_time:.2f}s")
        st.metric(label="LLM Processing", value=f"{total_elapsed - scan_time:.2f}s")
        st.metric(label="Total", value=f"{total_elapsed:.2f}s")

    elif st.session_state.pipeline_executed:
        task_configs = []
        if do_review:
            task_configs.append(("Review", "### 🛡️ Code Review Audit"))
        if do_analyze:
            task_configs.append(("Analysis", "### 🏗️ Architecture & Complexity Analysis"))
        if do_document:
            task_configs.append(("Documentation", "### 📝 Technical Documentation"))
        if do_refactor:
            task_configs.append(("Refactor", "### ⚙️ Refactoring Suggestions"))

        active_code_payload = final_code_payload
        if fast_mode and len(active_code_payload) > max_code_chars:
            active_code_payload = _trim_payload(active_code_payload, int(max_code_chars))
        restored_files_to_process = _split_payload_into_files(active_code_payload)

        for task_name, header_text in task_configs:
            if task_name in st.session_state.pipeline_results:
                st.markdown(header_text)
                task_data = st.session_state.pipeline_results[task_name]
                with st.container(border=True):
                    # For dict data, we render via expanders in _render_task_controls
                    # For string data, we display markdown first
                    if isinstance(task_data, str):
                        st.markdown(task_data)
                    _render_task_controls(task_name, task_data, active_code_payload, restored_files_to_process)

        st.markdown("### ⏱️ Performance Metrics")
        scan_time = st.session_state.get("_last_scan_time", 0)
        st.metric(label="Folder Scan", value=f"{scan_time:.2f}s")
        st.metric(label="LLM Processing", value=f"{st.session_state.pipeline_duration - scan_time:.2f}s")
        st.metric(label="Total", value=f"{st.session_state.pipeline_duration:.2f}s")
    else:
        if final_code_payload.strip():
            stack = detect_project_stack(final_code_payload)
            file_count = final_code_payload.count("--- FILE:")
            if file_count == 0:
                file_count = 1  # Single snippet
            
            st.markdown("### 📋 Project Status")
            st.success("✔ Project Loaded")
            st.markdown(f"**Files:** {file_count}")
            st.markdown(f"**Language:** `{stack['Language']}`")
            st.markdown(f"**Framework:** `{stack['Framework']}`")
            st.info("Ready to Execute")
        else:
            st.info("ℹ️ Awaiting execution. Provide code on the left and click **Execute Pipeline**.")
