import time
import hashlib
import streamlit as st
from agent_core import LocalCodeAgentEngine
import concurrent.futures

# Initialize session state cache
st.session_state.setdefault("llm_cache", {})

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

# ---------------------------------------------------------------------------
# OPTIMIZATION: Split large payloads into chunks <= 12k chars so they fit
# comfortably under the 16384-token context window of the model.
# Small projects (< 12k chars) are not split.
# ---------------------------------------------------------------------------
CHUNK_CHAR_LIMIT = 12000
DEFAULT_FAST_CHAR_LIMIT = 30000
MODEL_NAME = "gemma4:26b"
FAST_MODE = True
RUN_PARALLEL = False

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

    trimmed_blocks = []
    used = 0
    for idx, block in enumerate(file_blocks):
        if idx > 0:
            block = "--- FILE:" + block
        block_len = len(block)
        if used + block_len > max_chars:
            break
        trimmed_blocks.append(block)
        used += block_len

    if not trimmed_blocks:
        return text[:max_chars]
    return "\n".join(trimmed_blocks)

# ---------------------------------------------------------------------------
# OPTIMIZATION: Helper to run a single LLM task in a background thread.
# Uses the session-state cache so repeated analyses return instantly.
# ---------------------------------------------------------------------------
def _run_llm_task(task_name: str, engine: LocalCodeAgentEngine, code: str, cache_key: tuple, cache: dict):
    """
    Execute review / analyze / document on a code chunk.
    Returns (task_name, result_str, elapsed_time_seconds).
    Cache hit returns instantly with 0.0 elapsed time.
    """
    if cache_key in cache:
        return task_name, cache[cache_key], 0.0

    start = time.perf_counter()
    try:
        if task_name == "Review":
            result = engine.review_code(code)
        elif task_name == "Analysis":
            result = engine.analyze_code(code)
        elif task_name == "Documentation":
            result = engine.document_code(code)
        else:
            raise ValueError(f"Unknown task: {task_name}")
    except Exception as e:
        elapsed = time.perf_counter() - start
        return task_name, f"ERROR: {str(e)}", elapsed

    elapsed = time.perf_counter() - start
    cache[cache_key] = result
    return task_name, result, elapsed

# ---------------------------------------------------------------------------
# Fixed Ollama model
# ---------------------------------------------------------------------------
available_models = LocalCodeAgentEngine.list_available_models()
if MODEL_NAME not in available_models:
    st.error(f"Required Ollama model not found: `{MODEL_NAME}`")
    st.info(
        "Start Ollama, then install the required model with "
        "`ollama pull gemma4:26b`, and refresh this page."
    )
    st.stop()

fast_mode = FAST_MODE
max_code_chars = DEFAULT_FAST_CHAR_LIMIT
run_parallel = RUN_PARALLEL

engine = get_agent_engine(MODEL_NAME)

# ---------------------------------------------------------------------------
# Main UI Layout (unchanged)
# ---------------------------------------------------------------------------
st.title("🧠 Ollama-Powered Code Intelligence Agent")
st.markdown("A local enterprise-grade Code Review, Analysis, and Documentation tool.")
st.markdown("Analyze single snippets, batch-uploaded files, or an entire local POC repository.")

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

    # Task toggles
    st.markdown("### Execution Tasks")
    do_review = st.checkbox("Review Code (Audit & Security)", value=True)
    do_analyze = st.checkbox("Analyze Code (Complexity & Flow)", value=False)
    do_document = st.checkbox("Document Code (README Generation)", value=False)

    execute = st.button("🚀 Execute Pipeline", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# REPORTING PANEL (col2)
# ---------------------------------------------------------------------------
with col2:
    st.subheader("Reporting Panel")

    if not execute:
        st.info("ℹ️ Awaiting execution. Provide code on the left and click **Execute Pipeline**.")
    else:
        if not final_code_payload.strip():
            st.warning("⚠️ Please provide source code in the buffer before executing.")
        elif not any([do_review, do_analyze, do_document]):
            st.warning("Please select at least one execution task.")
        else:
            active_code_payload = final_code_payload
            if fast_mode and len(active_code_payload) > max_code_chars:
                original_chars = len(active_code_payload)
                active_code_payload = _trim_payload(active_code_payload, int(max_code_chars))
                st.warning(
                    f"Fast mode is analyzing {len(active_code_payload):,} of "
                    f"{original_chars:,} characters."
                )

            # Split payload on file boundaries
            if len(active_code_payload) < CHUNK_CHAR_LIMIT:
                chunks = [active_code_payload]
            else:
                chunks = _split_into_chunks(active_code_payload, CHUNK_CHAR_LIMIT)

            # Map user options to engine configurations
            task_configs = []
            if do_review:
                task_configs.append(("Review", engine.REVIEW_PROMPT, "### 🛡️ Code Review Audit"))
            if do_analyze:
                task_configs.append(("Analysis", engine.ANALYSIS_PROMPT, "### 🏗️ Architecture & Complexity Analysis"))
            if do_document:
                task_configs.append(("Documentation", engine.DOCUMENT_PROMPT, "### 📝 Technical Documentation"))

            start_total = time.perf_counter()
            timing_data = {"Total Processing Time": 0.0}

            # Process tasks sequentially but with full text streaming 
            for task_name, system_prompt, header_text in task_configs:
                st.markdown(header_text)
                
                # Container to hold the streaming frames
                with st.container(border=True):
                    combined_result_chunks = []
                    
                    for idx, chunk in enumerate(chunks):
                        if len(chunks) > 1:
                            st.caption(f"Processing part {idx+1} of {len(chunks)}...")
                        
                        # Trigger the live stream text engine
                        stream_generator = engine.generate_stream_response(system_prompt, chunk)
                        
                        # st.write_stream consumes the generator and paints tokens on the fly
                        output_text = st.write_stream(stream_generator)
                        combined_result_chunks.append(output_text)
                    
                    full_task_text = "\n\n".join(combined_result_chunks)
                    
                    # Provide download option for documentation immediately
                    if task_name == "Documentation":
                        st.download_button(
                            label="📥 Download Documentation",
                            data=full_task_text,                            
                            file_name="README.md",
                            mime="text/markdown",
                            use_container_width=True
                        )

            total_elapsed = time.perf_counter() - start_total
            
            # Simple single metric summary
            st.markdown("### ⏱️ Performance Metric")
            st.metric(label="Pipeline Duration", value=f"{total_elapsed:.2f} seconds")
