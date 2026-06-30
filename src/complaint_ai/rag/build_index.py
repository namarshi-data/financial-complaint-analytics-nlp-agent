from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def read_policy_docs(policy_dir: str | Path) -> list[dict]:
    policy_dir = Path(policy_dir)
    docs: list[dict] = []

    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 40]

        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "source": path.name,
                    "chunk_id": idx,
                    "text": chunk,
                }
            )

    if not docs:
        raise ValueError(f"No policy markdown files found in {policy_dir}")

    return docs


def _build_sentence_transformer_faiss(
    docs: list[dict],
    index_path: Path,
    metadata_path: Path,
    embedding_model_name: str,
) -> None:
    # Lazy imports: these are only loaded if this backend is selected.
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode([doc["text"] for doc in docs], normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    metadata_path.write_text(
        json.dumps(
            {
                "backend": "sentence_transformer_faiss",
                "embedding_model": embedding_model_name,
                "docs": docs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_tfidf_index(
    docs: list[dict],
    index_path: Path,
    metadata_path: Path,
) -> None:
    texts = [doc["text"] for doc in docs]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    matrix = vectorizer.fit_transform(texts)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
        },
        index_path,
    )

    metadata_path.write_text(
        json.dumps(
            {
                "backend": "tfidf",
                "docs": docs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_faiss_index(
    policy_dir: str | Path,
    index_path: str | Path,
    metadata_path: str | Path,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    backend: str = "auto",
) -> None:
    docs = read_policy_docs(policy_dir)

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if backend not in {"auto", "sentence_transformer", "tfidf"}:
        raise ValueError("backend must be one of: auto, sentence_transformer, tfidf")

    if backend in {"auto", "sentence_transformer"}:
        try:
            _build_sentence_transformer_faiss(
                docs=docs,
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name=embedding_model_name,
            )
            print(f"Saved SentenceTransformer FAISS index to {index_path}")
            print(f"Saved metadata to {metadata_path}")
            return

        except Exception as exc:
            if backend == "sentence_transformer":
                raise

            print("SentenceTransformer/torch index build failed.")
            print("Using lightweight sklearn TF-IDF fallback instead.")
            print(f"Fallback reason: {type(exc).__name__}: {exc}")

    _build_tfidf_index(
        docs=docs,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    print(f"Saved lightweight TF-IDF retrieval index to {index_path}")
    print(f"Saved metadata to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_dir", default="knowledge_base/policies")
    parser.add_argument("--index_path", default="models/policy_faiss.index")
    parser.add_argument("--metadata_path", default="models/policy_metadata.json")
    parser.add_argument(
        "--embedding_model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "sentence_transformer", "tfidf"],
    )

    args = parser.parse_args()

    build_faiss_index(
        policy_dir=args.policy_dir,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        embedding_model_name=args.embedding_model_name,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()