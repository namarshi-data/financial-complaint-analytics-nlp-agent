# Architecture

## Layers

1. Data ingestion
   - Read raw CFPB-style complaint data
   - Validate required columns
   - Filter rows with usable narratives

2. Text preprocessing
   - Redaction cleanup
   - whitespace normalization
   - label normalization
   - train/test split

3. Modeling
   - baseline TF-IDF + Logistic Regression
   - RNN and LSTM sequence models
   - LSTM + Attention for explainability
   - BERT / RoBERTa transformer fine-tuning

4. Operational risk scoring
   - keyword and category-based scoring
   - low-confidence model review
   - human escalation flag

5. RAG
   - policy documents chunked into passages
   - sentence-transformer embeddings
   - FAISS vector index
   - top-k policy retrieval

6. Response agent
   - Azure OpenAI response drafting
   - fallback response template when no API key is configured
   - policy-grounded and human-review-first wording

7. Serving
   - Streamlit business demo
   - FastAPI endpoint for system integration
   - Docker deployment

## Production Considerations

- Use a private model registry for trained artifacts.
- Store Azure OpenAI keys in a secret manager, not in source code.
- Log model version, policy version, prediction confidence, and response generation trace.
- Add PII handling and redaction before sending text to an LLM API.
- Require human approval before sending generated responses to customers.
