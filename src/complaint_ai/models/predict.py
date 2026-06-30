from pathlib import Path
from typing import Any

import joblib

from complaint_ai.data.preprocess import clean_text


def load_classifier(model_path: str | Path) -> Any:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run `make train-baseline` first."
        )
    return joblib.load(model_path)


def predict_category(text: str, model_path: str | Path = "models/baseline_tfidf_logreg.joblib") -> dict:
    model = load_classifier(model_path)
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]
    confidence = None
    top_predictions = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)[:5]
        top_predictions = [{"label": str(cls), "probability": float(prob)} for cls, prob in ranked]
        confidence = float(max(probs))
    return {
        "label": str(label),
        "confidence": confidence,
        "top_predictions": top_predictions,
    }
