import os
import json
import hashlib
from typing import Dict, Any, Tuple, Optional, List

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
AST_CACHE_DIR = os.path.join(CACHE_DIR, "ast")
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
HASH_CACHE_FILE = os.path.join(CACHE_DIR, "file_hashes.json")

class CacheManager:
    """
    Manages disk-based AST, Embedding, and SHA-256 file hash caching to enable
    incremental refactoring (skipping unchanged files).
    """

    def __init__(self):
        os.makedirs(AST_CACHE_DIR, exist_ok=True)
        os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
        self.hashes: Dict[str, str] = self._load_file_hashes()

    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

    def is_file_unchanged(self, fname: str, content: str) -> Tuple[bool, str]:
        """
        Returns (is_unchanged, content_hash).
        """
        current_hash = self.compute_hash(content)
        previous_hash = self.hashes.get(fname, "")
        if previous_hash and previous_hash == current_hash:
            return True, current_hash
        return False, current_hash

    def update_file_hash(self, fname: str, content_hash: str):
        self.hashes[fname] = content_hash
        self._save_file_hashes()

    def get_ast_cache(self, content_hash: str) -> Optional[Dict[str, Any]]:
        cache_file = os.path.join(AST_CACHE_DIR, f"{content_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def set_ast_cache(self, content_hash: str, ast_data: Dict[str, Any]):
        cache_file = os.path.join(AST_CACHE_DIR, f"{content_hash}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(ast_data, f, indent=2)
        except Exception:
            pass

    def get_embedding_cache(self, content_hash: str) -> Optional[List[float]]:
        cache_file = os.path.join(EMBEDDINGS_CACHE_DIR, f"{content_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def set_embedding_cache(self, content_hash: str, embedding_vector: List[float]):
        cache_file = os.path.join(EMBEDDINGS_CACHE_DIR, f"{content_hash}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(embedding_vector, f)
        except Exception:
            pass

    def _load_file_hashes(self) -> Dict[str, str]:
        if os.path.exists(HASH_CACHE_FILE):
            try:
                with open(HASH_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_file_hashes(self):
        try:
            with open(HASH_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.hashes, f, indent=2)
        except Exception:
            pass
