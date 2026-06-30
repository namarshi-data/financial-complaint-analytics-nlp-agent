# Model Card

## Model Purpose

Classify financial customer complaints into product categories and support operational triage.

## Intended Users

- complaint operations analysts
- risk analysts
- customer experience teams
- compliance reporting teams
- data analysts building NLP workflows

## Not Intended For

- automated final customer decisions
- legal advice
- regulatory determinations
- unsupervised customer communication without human review

## Training Data

Public financial complaint records with complaint narratives and product labels. Rows without complaint narratives should be excluded from text model training.

## Label Imbalance

The dataset is highly imbalanced. Use macro F1 and per-class recall, not accuracy alone.

## Evaluation

Recommended metrics:

- macro F1
- weighted F1
- precision/recall by class
- confusion matrix
- top-k accuracy
- confidence calibration
- latency

## Risks

- Complaint narratives may contain sensitive information.
- Generated responses may be inaccurate without human review.
- Product categories may shift over time.
- Some labels are historical and may require normalization.

## Controls

- PII redaction before LLM calls
- human-review flag for high-risk and low-confidence cases
- audit log of model version and retrieved policy
- drift monitoring on product and issue distribution
