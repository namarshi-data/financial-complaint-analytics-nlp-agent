from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from complaint_ai.agents.priority import score_priority
from complaint_ai.agents.response_agent import generate_response
from complaint_ai.models.predict import predict_category
from complaint_ai.rag.retriever import PolicyRetriever

app = FastAPI(
    title="Financial Complaint Intelligence API",
    version="0.1.0",
    description="Complaint classification, escalation scoring, RAG retrieval, and response drafting.",
)


class ComplaintRequest(BaseModel):
    text: str = Field(..., min_length=20)
    generate_reply: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ComplaintRequest) -> dict:
    prediction = predict_category(request.text)
    priority = score_priority(
        request.text,
        category=prediction["label"],
        confidence=prediction.get("confidence"),
    )

    policies = []
    if Path("models/policy_faiss.index").exists():
        retriever = PolicyRetriever()
        policies = retriever.search(request.text, k=3)

    response = None
    if request.generate_reply:
        response = generate_response(
            category=prediction["label"],
            priority=priority.priority,
            complaint_text=request.text,
            retrieved_policies=[p["text"] for p in policies],
        )

    return {
        "prediction": prediction,
        "priority": priority.__dict__,
        "retrieved_policies": policies,
        "draft_response": response,
    }
