# 🧠 Ollama-Powered Code Intelligence Agent

An enterprise-grade local Code Review, Architecture Analysis, and Documentation tool. Built with Streamlit and powered by Ollama running local Large Language Models (defaults to `qwen3:8b`).

This tool lets you ingest source code snippets, batch-upload files, or scan entire local directory structures, sending the parsed contents to a local Ollama daemon for analysis.

---

## 🚀 Key Features

*   **Flexible Ingestion Modes:**
    *   **Direct Clipboard Paste:** Fast code snippet reviews.
    *   **Upload Multiple Files:** Batch-upload multiple source files concurrently.
    *   **Scan Local POC Folder Path:** Automatically traverse a local folder, skipping binaries, dependencies (`node_modules`, `venv`, `.git`), and minified files, parsing files up to 500 KB and compiling them into a structured context payload.
*   **Pipeline Tasks:**
    *   **Code Review (Audit & Security):** Identifies critical/medium/low-risk bugs, performance issues, and security vulnerabilities with concrete remediation steps.
    *   **Architecture & Complexity Analysis:** Outlines project modules, dependencies, design patterns, bottlenecks, and scalability.
    *   **Technical Documentation:** Generates comprehensive, high-quality, professional markdown documentation.
*   **Real-time Streaming:** Token-by-token responses rendered directly to the Streamlit UI.
*   **Smart Payload Chunking:** Automatically segments massive code payloads into smaller chunks (<= 12k chars) along file boundaries to fit under local LLM context limits without cutting off code mid-file.
*   **Response Caching:** In-memory caching prevents redundant LLM calls for unchanged code reviews.

---

## 🛠️ Prerequisites

1.  **Ollama:** Ensure the Ollama daemon is installed and running on your local machine (`http://localhost:11434` by default).
    *   Download from [ollama.com](https://ollama.com).
2.  **Model:** Pull the configured LLM:
    ```bash
    ollama pull qwen3:8b
    ```

---

## 📦 Installation & Setup

1.  **Clone / Download this repository.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Running the Application

Start the Streamlit server from the root directory:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```text
├── .streamlit/             # Streamlit configuration directory
├── agent_core.py           # Core logic handling Ollama calls, folder parsing, and LLM orchestration
├── app.py                  # Streamlit UI layout and streaming response panels
├── test_script.py          # Offline CLI verification script
└── requirements.txt        # Python dependency manifest
```
