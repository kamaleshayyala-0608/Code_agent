import time
import re
import hashlib
import json
import concurrent.futures
import streamlit as st
from agent_core import LocalCodeAgentEngine, REQUIRED_MODEL_NAME
from generators.cicd_generator import (
    parse_markdown_file_blocks,
    parse_json_from_llm,
    build_doc_markdown,
    create_zip_from_dict,
    extract_complete_refactored_file
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
        ["Direct Clipboard Paste", "Upload Multiple Files", "Scan Local POC Folder Path"],
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
def _render_task_controls(task_name: str, task_data, active_code_payload: str):
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
        # task_data is a dict of {filepath: content}
        if task_data:
            zip_bytes = create_zip_from_dict(task_data)
            st.download_button(
                label="📥 Download Refactoring Package (ZIP)",
                data=zip_bytes,
                file_name="refactoring_suggestions.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_refactor"
            )
        
        # Render project-level guide first
        if "REFACTORING_GUIDE.md" in task_data:
            with st.expander("📋 Project Refactoring Guide", expanded=True):
                st.markdown(task_data["REFACTORING_GUIDE.md"])

        # Render report summary first
        if "SUMMARY.md" in task_data:
            with st.expander("📋 Comprehensive Refactoring Report Summary", expanded=True):
                st.markdown(task_data["SUMMARY.md"])
                
        # Render per-file recommendations separately from consulting documents.
        for filename, content in task_data.items():
            if filename.startswith("FILE_RECOMMENDATIONS/"):
                with st.expander(f"📄 Suggestions for {filename}", expanded=False):
                    st.markdown(content)

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

        # 2. Run Refactoring Pipeline (File-by-file)
        if do_refactor:
            st.markdown("### ⚙️ Refactoring Suggestions")
            refactor_results = {}
            pipeline_results["Refactor"] = refactor_results

            if not files_to_process:
                st.warning("No files found to refactor.")
            else:
                # Check if a previous REFACTORING_SPEC.md exists in session state from a prior run
                prev_spec = None
                if "pipeline_results" in st.session_state and isinstance(st.session_state.pipeline_results, dict):
                    prev_refactor = st.session_state.pipeline_results.get("Refactor")
                    if isinstance(prev_refactor, dict):
                        prev_spec = prev_refactor.get("REFACTORING_SPEC.md")

                def get_refactor_prompt(fname: str) -> str:
                    if prev_spec:
                        return f"""You are a Principal React/TypeScript Software Engineer.

Follow REFACTORING_SPEC.md.

Refactor this file using only those engineering rules.

REFACTORING_SPEC.md:
{prev_spec}

File Name:
{fname}

Format:

## File
{fname}

### Finding 1

Category

Problem

Evidence

Current Code

Recommendation

Improved Code

Implementation Notes

Expected Benefit

Estimated Effort

--------------------------------

Repeat for every finding."""
                    else:
                        return engine.REFACTOR_FILE_PROMPT.format(file_name=fname)

                raw_outputs = {}
                if MAX_CONCURRENT_MODEL_REQUESTS > 1:
                    with st.spinner("Refactoring files in parallel..."):
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MODEL_REQUESTS) as executor:
                            future_to_file = {}
                            for fname, fcontent in files_to_process.items():
                                future = executor.submit(engine.refactor_file_two_stage, fname, fcontent, prev_spec)
                                future_to_file[future] = fname
                                
                            for future in concurrent.futures.as_completed(future_to_file):
                                fname = future_to_file[future]
                                try:
                                    raw_output = future.result()
                                    raw_outputs[fname] = raw_output
                                    refactor_results[f"FILE_RECOMMENDATIONS/{fname}.md"] = raw_output
                                    refactored_code = extract_complete_refactored_file(raw_output)
                                    if refactored_code:
                                        refactor_results[f"REFACTORED_CODE/{fname}"] = refactored_code
                                except Exception as e:
                                    st.error(f"Failed to refactor {fname}: {e}")
                                    refactor_results[f"FILE_RECOMMENDATIONS/{fname}.md"] = f"Error refactoring {fname}: {str(e)}"
                else:
                    for fname, fcontent in files_to_process.items():
                        st.caption(f"Refactoring `{fname}`...")
                        with st.container(border=True):
                            try:
                                raw_output = engine.refactor_file_two_stage(fname, fcontent, prev_spec)
                                st.write("✓ Analysis completed.")
                                raw_outputs[fname] = raw_output
                                refactor_results[f"FILE_RECOMMENDATIONS/{fname}.md"] = raw_output
                                refactored_code = extract_complete_refactored_file(raw_output)
                                if refactored_code:
                                    refactor_results[f"REFACTORED_CODE/{fname}"] = refactored_code
                            except Exception as e:
                                st.error(f"Failed to refactor {fname}: {e}")
                                refactor_results[f"FILE_RECOMMENDATIONS/{fname}.md"] = f"Error refactoring {fname}: {str(e)}"

                # Compile the file-level findings into a project consulting package.
                refactor_metadata = "\n\n".join(
                    f"### Findings for {fname}\n{raw_output}"
                    for fname, raw_output in raw_outputs.items()
                )
                global_prompts = [
                    ("REFACTORING_GUIDE.md", engine.GLOBAL_REFACTORING_GUIDE_PROMPT),
                    ("COMMON_FUNCTIONS.md", engine.GLOBAL_COMMON_FUNCTIONS_PROMPT),
                    ("CUSTOM_HOOKS.md", engine.GLOBAL_CUSTOM_HOOKS_PROMPT),
                    ("COMMON_PATTERNS.md", engine.GLOBAL_COMMON_PATTERNS_PROMPT),
                    ("MIGRATION_PLAN.md", engine.GLOBAL_MIGRATION_PLAN_PROMPT),
                    ("REFACTORED_FILES.md", engine.GLOBAL_REFACTORED_FILES_PROMPT)
                ]
                for doc_name, prompt_template in global_prompts:
                    st.caption(f"Compiling project-level {doc_name}...")
                    with st.container(border=True):
                        try:
                            prompt = prompt_template.format(metadata_summary=refactor_metadata)
                            stream = engine.generate_stream_response(
                                prompt,
                                f"Generate the requested {doc_name} file.",
                                num_predict=GLOBAL_COMPILER_TOKENS,
                            )
                            raw_out = st.write_stream(stream) or ""
                            parsed_files = parse_markdown_file_blocks(raw_out)
                            if parsed_files:
                                refactor_results.update(parsed_files)
                            else:
                                clean_out = raw_out
                                if clean_out.strip().startswith("### File:"):
                                    parts = clean_out.split("\n", 1)
                                    if len(parts) > 1:
                                        clean_out = parts[1]
                                refactor_results[doc_name] = clean_out.strip()
                        except Exception as e:
                            st.error(f"Error compiling {doc_name}: {str(e)}")

                # Additional LLM call to generate REFACTORING_SPEC.md
                st.caption("Generating project-wide refactoring specification...")
                with st.container(border=True):
                    try:
                        spec_prompt = engine.REFACTORING_SPEC_PROMPT.format(metadata_summary=refactor_metadata)
                        spec_stream = engine.generate_stream_response(
                            spec_prompt,
                            "Generate the REFACTORING_SPEC.md document based on the per-file findings.",
                            num_predict=GLOBAL_COMPILER_TOKENS,
                        )
                        raw_spec_output = st.write_stream(spec_stream) or ""
                        spec_files = parse_markdown_file_blocks(raw_spec_output)
                        if "REFACTORING_SPEC.md" in spec_files:
                            refactor_results["REFACTORING_SPEC.md"] = spec_files["REFACTORING_SPEC.md"]
                        else:
                            # Heuristic fallback if header missing
                            clean_spec = raw_spec_output
                            if clean_spec.strip().startswith("### File:"):
                                parts = clean_spec.split("\n", 1)
                                if len(parts) > 1:
                                    clean_spec = parts[1]
                            refactor_results["REFACTORING_SPEC.md"] = clean_spec
                    except Exception as e:
                        st.error(f"Error during specification generation: {str(e)}")

                # Programmatically build SUMMARY.md
                summary_lines = []
                summary_lines.append("# Refactoring Summary Report\n")
                summary_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                summary_lines.append("## Files Analyzed and Recommendations\n")

                total_findings = 0
                critical_findings = 0
                high_findings = 0
                medium_findings = 0
                low_findings = 0

                for file_key, file_content in refactor_results.items():
                    if file_key.startswith("FILE_RECOMMENDATIONS/"):
                        fname = file_key.replace("FILE_RECOMMENDATIONS/", "").replace(".md", "")
                        findings_count = len(re.findall(r"###\s*Finding\s*\d+", file_content, re.IGNORECASE))
                        total_findings += findings_count
                        
                        critical_findings += len(re.findall(r"Priority\s*[\r\n]+Critical", file_content, re.IGNORECASE))
                        high_findings += len(re.findall(r"Priority\s*[\r\n]+High", file_content, re.IGNORECASE))
                        medium_findings += len(re.findall(r"Priority\s*[\r\n]+Medium", file_content, re.IGNORECASE))
                        low_findings += len(re.findall(r"Priority\s*[\r\n]+Low", file_content, re.IGNORECASE))
                        
                        has_refactored_code = f"REFACTORED_CODE/{fname}" in refactor_results
                        code_status = "Generated" if has_refactored_code else "No refactoring required"
                        summary_lines.append(f"- **{fname}**: {findings_count} finding(s) | Refactored Code: {code_status}")

                summary_lines.append("\n## Findings by Priority\n")
                summary_lines.append(f"- 🔴 **Critical**: {critical_findings}")
                summary_lines.append(f"- 🟠 **High**: {high_findings}")
                summary_lines.append(f"- 🟡 **Medium**: {medium_findings}")
                summary_lines.append(f"- 🟢 **Low**: {low_findings}")
                summary_lines.append(f"- **Total Findings**: {total_findings}\n")

                summary_lines.append("## Next Steps\n")
                summary_lines.append("1. Review `REFACTORING_GUIDE.md` for the roadmap phase details.")
                summary_lines.append("2. Adhere to coding standards defined in `REFACTORING_SPEC.md`.")
                summary_lines.append("3. Extract common functions and hooks as specified in `COMMON_FUNCTIONS.md` and `CUSTOM_HOOKS.md`.")
                summary_lines.append("4. Follow the step-by-step execution in `MIGRATION_PLAN.md`.")

                refactor_results["SUMMARY.md"] = "\n".join(summary_lines)

                # Filter keys to exactly match the requested ZIP structure
                allowed_keys = {
                    "REFACTORING_GUIDE.md",
                    "REFACTORING_SPEC.md",
                    "COMMON_FUNCTIONS.md",
                    "COMMON_PATTERNS.md",
                    "CUSTOM_HOOKS.md",
                    "MIGRATION_PLAN.md",
                    "REFACTORED_FILES.md",
                    "SUMMARY.md"
                }
                for key in list(refactor_results.keys()):
                    if (
                        not key.startswith("FILE_RECOMMENDATIONS/")
                        and not key.startswith("REFACTORED_CODE/")
                        and key not in allowed_keys
                    ):
                        refactor_results.pop(key, None)

                _render_task_controls("Refactor", refactor_results, active_code_payload)

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

        for task_name, header_text in task_configs:
            if task_name in st.session_state.pipeline_results:
                st.markdown(header_text)
                task_data = st.session_state.pipeline_results[task_name]
                with st.container(border=True):
                    # For dict data, we render via expanders in _render_task_controls
                    # For string data, we display markdown first
                    if isinstance(task_data, str):
                        st.markdown(task_data)
                    _render_task_controls(task_name, task_data, active_code_payload)

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
