import re
import io
import zipfile
import json

def parse_cicd_output(text: str) -> dict:
    """
    Parses the generated CI/CD markdown text and extracts individual file contents.
    Looks for the following format:
    ### File: <filepath>
    ```<language>
    <content>
    ```
    """
    files = {}
    
    # Pattern matching "### File: <path>" followed by an optional description and a fenced code block
    # Matches markdown headers like "### File: Dockerfile" or "### File: .github/workflows/ci.yml"
    pattern = r"###\s*File:\s*([^\n\r]+)[\s\S]*?```[a-zA-Z0-9_-]*\n([\s\S]*?)```"
    matches = re.findall(pattern, text)
    
    for filepath, content in matches:
        # Clean up the path
        filepath = filepath.strip()
        filepath = re.sub(r'[*`_]', '', filepath)  # remove formatting characters
        files[filepath] = content.strip()
        
    # If no files were found with the custom header format, try a fallback heuristic
    if not files:
        # Find all fenced code blocks
        blocks = re.findall(r"```([a-zA-Z0-9_-]*)\n([\s\S]*?)```", text)
        dockerfile_found = False
        docker_compose_found = False
        ci_found = False
        cd_found = False
        
        for idx, (lang, content) in enumerate(blocks):
            content_stripped = content.strip()
            lang_lower = lang.lower()
            
            if lang_lower in ("dockerfile", "docker") or "from " in content_stripped.lower():
                if not dockerfile_found:
                    files["Dockerfile"] = content_stripped
                    dockerfile_found = True
            elif "services:" in content_stripped or "version:" in content_stripped:
                if not docker_compose_found:
                    files["docker-compose.yml"] = content_stripped
                    docker_compose_found = True
            elif "on:" in content_stripped and "jobs:" in content_stripped:
                # GitHub Action
                if "ci" in content_stripped.lower() or not ci_found:
                    files[".github/workflows/ci.yml"] = content_stripped
                    ci_found = True
                else:
                    files[".github/workflows/cd.yml"] = content_stripped
                    cd_found = True
                    
        # Always output the full text as README_CICD.md
        files["README_CICD.md"] = text
    else:
        # Ensure README_CICD.md contains the full text of the report
        files["README_CICD.md"] = text

    return files

def create_cicd_zip(files: dict) -> bytes:
    """
    Assembles a ZIP file in-memory containing the generated DevOps assets.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, content in files.items():
            # Ensure correct directory structure in ZIP (e.g. forward slashes)
            clean_path = filepath.replace("\\", "/")
            zip_file.writestr(clean_path, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def evaluate_production_score(files: dict, text: str) -> int:
    """
    Analyzes generated DevOps files to calculate a Production Readiness Score out of 100.
    Checks for compliance with common cloud-native best practices.
    """
    score = 0
    
    # 1. Existence checks (Max 70 points)
    has_dockerfile = False
    has_compose = False
    has_ci = False
    has_cd = False
    
    for path in files.keys():
        path_lower = path.lower()
        if "dockerfile" in path_lower:
            has_dockerfile = True
        elif "docker-compose" in path_lower:
            has_compose = True
        elif "ci.yml" in path_lower or "ci.yaml" in path_lower:
            has_ci = True
        elif "cd.yml" in path_lower or "cd.yaml" in path_lower:
            has_cd = True
            
    if has_dockerfile:
        score += 20
    if has_compose:
        score += 20
    if has_ci:
        score += 15
    if has_cd:
        score += 15
        
    # 2. Quality checks (Max 30 points)
    # Check Dockerfile rules
    dockerfile_content = ""
    for path, content in files.items():
        if "dockerfile" in path.lower():
            dockerfile_content = content
            break
            
    if dockerfile_content:
        # Check: Pinned version of base image (not using just 'latest' or 'python:latest')
        lines = dockerfile_content.split("\n")
        from_lines = [l for l in lines if l.strip().upper().startswith("FROM")]
        if from_lines:
            base_image = from_lines[0].split()[-1]
            if ":" in base_image and not base_image.endswith(":latest"):
                score += 10 # Pinned base image
                
        # Check: Running as non-root user (USER directive)
        if "USER " in dockerfile_content.upper():
            score += 10 # Multi-stage / secure user
            
    # Check CD workflow secrets
    cd_content = ""
    for path, content in files.items():
        if "cd.yml" in path.lower() or "cd.yaml" in path.lower():
            cd_content = content
            break
            
    if cd_content:
        # Check: Uses secrets for sensitive keys
        if "${{ secrets." in cd_content or "secrets." in cd_content:
            score += 10
            
    # Normalize score (in case of math errors, bound between 0 and 100)
    # Make sure a baseline is present: if at least some files are generated, score is minimum 50
    if files and score < 50:
        score = max(score, 50)
        
    return min(score, 100)

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

def build_refactor_markdown(refactor_json: dict) -> str:
    """
    Synthesizes a JSON object into a readable Refactoring Suggestions markdown file.
    """
    if "error" in refactor_json:
        return f"# Refactoring Report Error\n\n{refactor_json['raw_text']}"
        
    fn = refactor_json.get("file_name", "unknown")
    summary = refactor_json.get("summary", "")
    purpose = refactor_json.get("purpose", "")
    
    complexity = refactor_json.get("complexity", {})
    cc = complexity.get("cyclomatic_complexity", "N/A")
    mi = complexity.get("maintainability", "N/A")
    
    score = refactor_json.get("overall_score", 100)
    
    md = []
    md.append(f"# Refactoring Suggestions for `{fn}`")
    md.append(f"**Overall Quality Score:** `{score}/100`\n")
    md.append(f"## File Overview\n- **Purpose:** {purpose}\n- **Summary:** {summary}\n")
    md.append(f"## Complexity Metrics\n- **Cyclomatic Complexity:** {cc}\n- **Maintainability Index:** {mi}\n")
    
    md.append("## Code Smells")
    smells = refactor_json.get("smells", [])
    if smells:
        for smell in smells:
            stype = smell.get("type", "General")
            sdesc = smell.get("description", "")
            md.append(f"- **{stype}**: {sdesc}")
    else:
        md.append("- No significant code smells detected.")
    md.append("")
    
    md.append("## Performance Problems")
    perf = refactor_json.get("performance", [])
    if perf:
        for p in perf:
            issue = p.get("issue", "")
            desc = p.get("description", "")
            md.append(f"- **{issue}**: {desc}")
    else:
        md.append("- No major performance problems identified.")
    md.append("")
    
    md.append("## Security Vulnerabilities")
    sec = refactor_json.get("security", [])
    if sec:
        for s in sec:
            issue = s.get("issue", "")
            desc = s.get("description", "")
            md.append(f"- ⚠️ **{issue}**: {desc}")
    else:
        md.append("- No obvious security vulnerabilities found.")
    md.append("")
    
    md.append("## SOLID Violations")
    solid = refactor_json.get("solid_violations", [])
    if solid:
        for sv in solid:
            principle = sv.get("principle", "")
            desc = sv.get("description", "")
            md.append(f"- **{principle}**: {desc}")
    else:
        md.append("- No SOLID principles violated.")
    md.append("")
    
    md.append("## Refactoring Recommendations")
    sugs = refactor_json.get("suggestions", [])
    if sugs:
        for sug in sugs:
            ref = sug.get("refactoring", "")
            priority = sug.get("priority", "Low")
            impact = sug.get("impact", "")
            md.append(f"- **[{priority}] {ref}**  \n  *Impact:* {impact}")
    else:
        md.append("- No suggestions recommended.")
        
    return "\n".join(md)

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
