# Interview Talking Points

## 30-Second Summary

I built an end-to-end financial complaint intelligence platform that classifies customer complaints, scores escalation risk, retrieves relevant policy guidance using FAISS RAG, and drafts an analyst-reviewable customer response with Azure OpenAI. I compared classical NLP, LSTM, attention-based deep learning, and transformer models, then deployed the workflow through Streamlit, FastAPI, and Docker.

## Why This Is Business-Relevant

Complaint handling affects customer trust, regulatory risk, operational workload, and SLA performance. This project turns unstructured complaint text into structured categories, risk scores, and action recommendations for analysts.

## Technical Decisions

- I started with TF-IDF Logistic Regression as a transparent baseline.
- I added LSTM and attention models to show sequence modeling and explainability.
- I added BERT/RoBERTa to benchmark modern transformer performance.
- I used FAISS RAG so generated responses are grounded in policy documents instead of free-form LLM output.
- I kept human review for high-risk and low-confidence cases.

## Best Questions to Prepare For

1. Why did you use macro F1 instead of accuracy?
2. How did you handle class imbalance?
3. What would you monitor after deployment?
4. How would you prevent hallucinated customer responses?
5. How would you protect PII before sending data to Azure OpenAI?
6. How would this integrate with Power BI or a CRM system?
