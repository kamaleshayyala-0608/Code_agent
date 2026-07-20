import io
import zipfile
from typing import Dict

class ExportAgent:
    def __init__(self):
        pass

    def package_refactored_project(self, refactored_files: Dict[str, str]) -> Dict[str, str]:
        """
        Organizes refactored code files under a uniform prefix structure.
        """
        packaged = {}
        for fname, content in refactored_files.items():
            # Add package prefix directory
            packaged[f"Refactored_Project/{fname}"] = content
        return packaged

    def build_zip_archive(self, packaged_files: Dict[str, str]) -> bytes:
        """
        Generates a ZIP archive in-memory for download.
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filepath, content in packaged_files.items():
                clean_path = filepath.replace("\\", "/")
                zip_file.writestr(clean_path, content)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
