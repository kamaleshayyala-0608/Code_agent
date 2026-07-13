import re
import io
import zipfile

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

def parse_refactor_output(text: str) -> dict:
    """
    Parses the generated refactoring suggestions text and extracts refactored source files.
    """
    files = {}
    pattern = r"###\s*File:\s*([^\n\r]+)[\s\S]*?```[a-zA-Z0-9_-]*\n([\s\S]*?)```"
    matches = re.findall(pattern, text)
    
    for filepath, content in matches:
        filepath = filepath.strip()
        filepath = re.sub(r'[*`_]', '', filepath)
        files[filepath] = content.strip()
        
    # Also add/ensure the full text is saved as refactoring_report.md
    files["refactoring_report.md"] = text
    return files

def create_refactor_zip(files: dict) -> bytes:
    """
    Assembles a ZIP file in-memory containing the refactored code and report.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, content in files.items():
            clean_path = filepath.replace("\\", "/")
            zip_file.writestr(clean_path, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
