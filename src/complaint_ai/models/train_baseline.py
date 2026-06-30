from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from complaint_ai.data.ingest import filter_training_rows, load_complaints
from complaint_ai.data.preprocess import prepare_text_classification_frame
from complaint_ai.features.text_features import build_tfidf_logreg_pipeline


def train_baseline(
    input_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    sample_nrows: int | None = None,
) -> dict:
    df = load_complaints(input_path, nrows=sample_nrows)
    df = filter_training_rows(df)
    df = prepare_text_classification_frame(df)

    # Remove ultra-rare classes in small demo samples so stratified splitting is valid.
    # For the full dataset, keep more classes and use the same logic with a higher threshold if needed.
    class_counts = df["label_clean"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["label_clean"].isin(valid_classes)].reset_index(drop=True)

    stratify = df["label_clean"] if df["label_clean"].nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label_clean"],
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_tfidf_logreg_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
