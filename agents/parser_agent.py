from typing import Dict, Any
from utils.ast_parser import ASTParser
from utils.dependency_analyzer import DependencyAnalyzer

class ParserAgent:
    def __init__(self):
        pass

    def parse_project(self, files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Parses all code files in the project to extract structural metadata.
        """
        parsed_metadata = {}
        for fname, fcontent in files.items():
            parsed_metadata[fname] = ASTParser.parse_file(fname, fcontent)
            
        # Add dependency context
        dep_graph = DependencyAnalyzer.analyze_workspace_dependencies(files, parsed_metadata)
        for fname in files.keys():
            parsed_metadata[fname]["dependencies_context"] = dep_graph.get(fname, {"depends_on": [], "depended_on_by": []})
            
        return parsed_metadata
