from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from complaint_ai.agents.priority import score_priority
from complaint_ai.agents.response_agent import generate_response
from complaint_ai.models.predict import predict_category
from complaint_ai.rag.retriever import PolicyRetriever

st.set_page_config(
    page_title="Financial Complaint NLP Agent",
    page_icon="💬",
    layout="wide",
)

st.title("Financial Customer Complaint Intelligence & Resolution Agent")
st.caption("LSTM/BERT-ready NLP classification + FAISS RAG + Azure OpenAI response drafting")

sample_text = """I found incorrect information on my credit report and the company keeps saying it verified the account. This is damaging my credit score and I need it fixed urgently."""

with st.sidebar:
    st.header("Demo Controls")
    model_path = st.text_input("Model path", "models/baseline_tfidf_logreg.joblib")
    use_rag = st.checkbox("Use RAG policy retrieval", value=True)
    use_llm = st.checkbox("Generate response draft", value=True)
    st.divider()
    st.markdown("Run `make train-baseline` and `make build-rag` before production demo.")

complaint = st.text_area("Paste customer complaint", value=sample_text, height=180)

if st.button("Analyze Complaint", type="primary"):
    if not Path(model_path).exists():
        st.error("Model file not found. Run `make train-baseline` first.")
        st.stop()

    prediction = predict_category(complaint, model_path=model_path)
    priority = score_priority(
        complaint,
        category=prediction["label"],
        confidence=prediction.get("confidence"),
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Category", prediction["label"].replace("_", " ").title())
    col2.metric("Priority", priority.priority)
    col3.metric("Risk Score", priority.score)

    st.subheader("Top Model Predictions")
    if prediction["top_predictions"]:
        st.dataframe(pd.DataFrame(prediction["top_predictions"]), use_container_width=True)
    else:
        st.write(prediction)

    st.subheader("Escalation Logic")
    for reason in priority.reasons:
        st.write(f"- {reason}")
    st.write(f"Human review required: **{priority.human_review_required}**")

    retrieved = []
    if use_rag:
        try:
            retriever = PolicyRetriever()
            retrieved = retriever.search(complaint, k=3)
            st.subheader("Retrieved Policy Guidance")
            for item in retrieved:
                with st.expander(f"{item['source']} · score {item['score']:.3f}"):
                    st.write(item["text"])
        except Exception as exc:  # noqa: BLE001
            st.warning(f"RAG index unavailable. Run `make build-rag`. Details: {exc}")

    if use_llm:
        st.subheader("Draft Response for Analyst Review")
        draft = generate_response(
            category=prediction["label"],
            priority=priority.priority,
            complaint_text=complaint,
            retrieved_policies=[item["text"] for item in retrieved],
        )
        st.write(draft)
