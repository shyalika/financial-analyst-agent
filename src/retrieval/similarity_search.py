import json
import numpy as np
from src.ingestion.vector_store import SQLiteVectorStore

class SimilaritySearch:
    def __init__(self):
        self.db = SQLiteVectorStore()

    def _cosine_similarity(self, v1, v2) -> float:
        v1, v2 = np.array(v1), np.array(v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        candidates = self.db.fetch_all_child_chunks()

        scored = []
        for candidate in candidates:
            embedding = json.loads(candidate["embedding_json"])
            similarity = self._cosine_similarity(query_embedding, embedding)
            result = {key: value for key, value in candidate.items() if key != "embedding_json"}
            result["similarity"] = similarity
            scored.append(result)

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:top_k]
