import os
import json
import numpy as np
import ollama
from typing import List, Dict, Any, Tuple

class LocalVectorDB:
    def __init__(self, spec_path: str = "rules/refactoring_spec.md", cache_path: str = "rules/refactoring_spec_embeddings.json"):
        self.spec_path = spec_path
        self.cache_path = cache_path
        self.embeddings_model = "nomic-embed-text:latest"
        self.rules: List[Dict[str, Any]] = []
        
        # Resolve absolute paths relative to execution dir
        if not os.path.exists(self.spec_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.spec_path = os.path.join(base_dir, "rules", "refactoring_spec.md")
            self.cache_path = os.path.join(base_dir, "rules", "refactoring_spec_embeddings.json")
            
        self._load_or_build_index()

    def _get_embedding(self, text: str) -> List[float]:
        try:
            response = ollama.embeddings(model=self.embeddings_model, prompt=text)
            # Response is a Pydantic model in newer ollama versions, support both direct attr and dictionary
            if hasattr(response, "embedding"):
                return response.embedding
            elif isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            else:
                # Try .get fallback or model_dump
                res_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                return res_dict.get("embedding", [])
        except Exception as e:
            # Try ollama.embed fallback
            try:
                response = ollama.embed(model=self.embeddings_model, input=text)
                if hasattr(response, "embeddings") and response.embeddings:
                    return response.embeddings[0]
                elif isinstance(response, dict) and "embeddings" in response and response["embeddings"]:
                    return response["embeddings"][0]
            except Exception as e2:
                raise RuntimeError(f"Failed to generate embedding via Ollama: {e} -> {e2}")

    def _load_or_build_index(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
                return
            except Exception:
                pass # Rebuild on corruption
                
        if not os.path.exists(self.spec_path):
            # Fallback when spec is completely missing
            self.rules = []
            return

        with open(self.spec_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into chapters/rules based on standard markdown headings
        chunks = content.split("\n## ")
        
        # First chunk is title/intro
        intro = chunks[0].strip()
        rules_to_embed = []
        
        for idx, chunk in enumerate(chunks[1:]):
            full_rule_text = "## " + chunk.strip()
            lines = chunk.strip().split("\n")
            title = lines[0].strip() if lines else f"Rule {idx+1}"
            rules_to_embed.append({
                "id": idx + 1,
                "title": title,
                "text": full_rule_text
            })

        # Generate embeddings for each rule block
        for rule in rules_to_embed:
            try:
                rule["embedding"] = self._get_embedding(rule["text"])
            except Exception:
                rule["embedding"] = []

        self.rules = rules_to_embed
        
        # Save cache
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.rules, f, indent=2)
        except Exception:
            pass

    def retrieve_relevant_rules(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves the top_k most relevant rules based on cosine similarity.
        """
        if not self.rules:
            return []
            
        try:
            query_embedding = np.array(self._get_embedding(query))
        except Exception:
            # Fallback to returning the first top_k rules if embedding fails
            return [(rule, 1.0) for rule in self.rules[:top_k]]

        results = []
        for rule in self.rules:
            if not rule.get("embedding"):
                continue
            rule_embedding = np.array(rule["embedding"])
            
            # Cosine similarity calculation
            dot_product = np.dot(query_embedding, rule_embedding)
            query_norm = np.linalg.norm(query_embedding)
            rule_norm = np.linalg.norm(rule_embedding)
            
            if query_norm > 0 and rule_norm > 0:
                similarity = float(dot_product / (query_norm * rule_norm))
            else:
                similarity = 0.0
                
            results.append((rule, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
