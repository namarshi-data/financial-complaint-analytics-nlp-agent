from __future__ import annotations

import argparse

from complaint_ai.models.train_baseline import train_baseline
from complaint_ai.rag.build_index import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/complaints_sample_1000.csv")
    args = parser.parse_args()

    train_baseline(
        input_path=args.input,
        model_path="models/baseline_tfidf_logreg.joblib",
        metrics_path="reports/baseline_metrics.json",
    )
    build_faiss_index(
        policy_dir="knowledge_base/policies",
        index_path="models/policy_faiss.index",
        metadata_path="models/policy_metadata.json",
    )
    print("Training pipeline completed.")


if __name__ == "__main__":
    main()
