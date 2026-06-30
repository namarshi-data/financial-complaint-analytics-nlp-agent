# Data Dictionary

| Column | Meaning | Portfolio use |
|---|---|---|
| Date received | Date complaint was received | trend analysis and SLA aging |
| Product | Financial product category | primary classification target |
| Sub-product | More detailed product category | feature and dashboard slice |
| Issue | Complaint issue | secondary target or analytical dimension |
| Sub-issue | Detailed issue | optional fine-grained label |
| Consumer complaint narrative | Free-text complaint | NLP model input |
| Company public response | Public company statement | response analytics |
| Company | Financial company | company-level trend analysis |
| State | Consumer state | geography analysis |
| Submitted via | Complaint channel | channel analysis |
| Date sent to company | Date sent for company response | SLA timing |
| Company response to consumer | Resolution outcome | outcome prediction / dashboard |
| Timely response? | Whether company responded on time | SLA monitoring target |
| Consumer disputed? | Whether consumer disputed response | customer dissatisfaction signal |
| Complaint ID | Unique complaint key | traceability |
