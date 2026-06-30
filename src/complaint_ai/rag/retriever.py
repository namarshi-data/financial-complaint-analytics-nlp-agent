from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PolicyRetriever:
    def __init__(
        self,
        index_path: str | Path = "models/policy_faiss.index",
        metadata_path: str | Path = "models/policy_metadata.json",
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "RAG index not found. Run the RAG build command first."
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.backend = metadata.get("backend", "sentence_transformer_faiss")
        self.docs = metadata["docs"]

        if self.backend == "tfidf":
            payload = joblib.load(self.index_path)
            self.vectorizer = payload["vectorizer"]
            self.matrix = payload["matrix"]
            self.index = None
            self.model = None

        else:
            # Lazy imports: these are loaded only when using the transformer backend.
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.index = faiss.read_index(str(self.index_path))
            self.model = SentenceTransformer(
                metadata.get(
                    "embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            )

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.backend == "tfidf":
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.matrix).ravel()
            top_indices = np.argsort(scores)[::-1][:k]

            results = []

            for idx in top_indices:
                doc = self.docs[int(idx)].copy()
                doc["score"] = float(scores[int(idx)])
                doc["retrieval_backend"] = "tfidf"
                results.append(doc)

            return results

        embedding = self.model.encode([query], normalize_embeddings=True)
        embedding = np.asarray(embedding, dtype="float32")

        scores, indices = self.index.search(embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.docs[int(idx)].copy()
            doc["score"] = float(score)
            doc["retrieval_backend"] = "sentence_transformer_faiss"
            results.append(doc)

        return results