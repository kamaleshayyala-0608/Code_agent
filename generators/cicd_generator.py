import re
import io
import zipfile
import json


def parse_markdown_file_blocks(text: str) -> dict:
    """Extract Markdown documents emitted as ``### File: path`` blocks."""
    headers = list(re.finditer(r"^###\s*File:\s*([^\r\n]+)\s*$", text, re.MULTILINE))
    files = {}
    for index, header in enumerate(headers):
        filepath = re.sub(r"[*`_]", "", header.group(1)).strip()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        content = text[header.end():end].strip()
        # Be forgiving if the model still wraps a Markdown document in fences.
        content = re.sub(r"^```(?:markdown|md)?\s*\n", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\n```\s*$", "", content).strip()
        if filepath and content:
            files[filepath] = content
    return files


def parse_json_from_llm(text: str) -> dict:
    """
    Robustly extracts and parses a JSON object from the LLM output text,
    handling markdown formatting fences.
    """
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_text": text
        }


def build_doc_markdown(doc_json: dict) -> str:
    """
    Synthesizes a JSON object into module documentation markdown.
    """
    if "error" in doc_json:
        return f"# Module Documentation Error\n\n{doc_json['raw_text']}"
        
    fn = doc_json.get("file_name", "unknown")
    purpose = doc_json.get("purpose", "")
    
    md = []
    md.append(f"# Technical Documentation for `{fn}`\n")
    md.append(f"## Module Overview\n- **Purpose:** {purpose}\n")
    
    md.append("## Responsibilities")
    for resp in doc_json.get("responsibilities", []):
        md.append(f"- {resp}")
    md.append("")
    
    md.append("## Imports")
    imports = doc_json.get("imports", [])
    if imports:
        for imp in imports:
            md.append(f"- `{imp}`")
    else:
        md.append("- No library imports.")
    md.append("")

    md.append("## Dependencies")
    deps = doc_json.get("dependencies", [])
    if deps:
        for dep in deps:
            md.append(f"- `{dep}`")
    else:
        md.append("- No external dependencies.")
    md.append("")
    
    md.append("## Classes")
    classes = doc_json.get("classes", [])
    if classes:
        for cls in classes:
            name = cls.get("name", "")
            purp = cls.get("purpose", "")
            md.append(f"### Class: `{name}`\n{purp}\n")
            methods = cls.get("methods", [])
            if methods:
                md.append("#### Methods")
                for m in methods:
                    mname = m.get("name", "")
                    mpurp = m.get("purpose", "")
                    margs = ", ".join(m.get("arguments", []))
                    mret = m.get("returns", "void")
                    md.append(f"- **`{mname}({margs})`**: {mpurp} (Returns: `{mret}`)")
            md.append("")
    else:
        md.append("- No classes defined.")
    md.append("")
    
    md.append("## Functions")
    funcs = doc_json.get("functions", [])
    if funcs:
        for f in funcs:
            fname = f.get("name", "")
            fpurp = f.get("purpose", "")
            fargs = ", ".join(f.get("arguments", []))
            fret = f.get("returns", "void")
            md.append(f"- **`{fname}({fargs})`**: {fpurp} (Returns: `{fret}`)")
    else:
        md.append("- No standalone functions defined.")
    md.append("")
    
    md.append(f"## Execution Call Flow\n{doc_json.get('flow', 'N/A')}\n")
    
    md.append("## Inputs & Outputs")
    md.append("### Inputs")
    for inp in doc_json.get("inputs", []):
        md.append(f"- {inp}")
    if not doc_json.get("inputs", []):
        md.append("- None")
    md.append("### Outputs")
    for outp in doc_json.get("outputs", []):
        md.append(f"- {outp}")
    if not doc_json.get("outputs", []):
        md.append("- None")
    md.append("")
    
    md.append("## Exception Handling")
    excs = doc_json.get("exceptions", [])
    if excs:
        for exc in excs:
            etype = exc.get("type", "")
            desc = exc.get("description", "")
            md.append(f"- **`{etype}`**: {desc}")
    else:
        md.append("- No formal exceptions cataloged.")
    md.append("")
    
    md.append("## Configuration Settings")
    configs = doc_json.get("configuration", [])
    if configs:
        for cfg in configs:
            md.append(f"- {cfg}")
    else:
        md.append("- No custom configuration settings mapped.")
    md.append("")

    md.append("## Example Usage")
    usage = doc_json.get("example_usage", "")
    if usage:
        md.append(usage)
    else:
        md.append("- No example usage provided.")
    md.append("")

    md.append("## Future Improvements")
    for imp in doc_json.get("future_improvements", []):
        md.append(f"- {imp}")
    if not doc_json.get("future_improvements", []):
        md.append("- None planned.")
        
    return "\n".join(md)

def create_zip_from_dict(files_dict: dict) -> bytes:
    """
    Generates a ZIP archive from a dictionary of {file_path: file_content}.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, content in files_dict.items():
            clean_path = filepath.replace("\\", "/")
            zip_file.writestr(clean_path, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def extract_complete_refactored_file(text: str) -> str:
    """
    Extracts the complete refactored file content from the refactoring findings.
    Looks for code blocks under the '### Complete Refactored File' section.
    """
    match = re.search(r"###\s*Complete\s*Refactored\s*File\s*([\s\S]*)", text, re.IGNORECASE)
    if not match:
        return ""
    
    content = match.group(1).strip()
    
    # Remove any trailing section (e.g. '### Refactoring Priority')
    divider_match = re.search(r"^(?:--------------------------------|###\s*\w+)", content, re.MULTILINE)
    if divider_match:
        content = content[:divider_match.start()].strip()
        
    code_match = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", content)
    if code_match:
        return code_match.group(1).strip()
    
    return content

