# Uploaded Data Profile

The uploaded archive `Data (18).zip` contains one file: `Data/complaints.csv`.

## Profile Summary

| Metric | Value |
|---|---:|
| Total rows | 2,326,246 |
| Rows with complaint narrative | 809,343 |
| Narrative coverage | 34.79% |
| Date range | 2011-12-01 to 2021-11-02 |
| Mean words in sampled narratives | 229.8 |

## Top Products

| Product | Rows |
|---|---:|
| Credit reporting, credit repair services, or other personal consumer reports | 861,526 |
| Debt collection | 388,108 |
| Mortgage | 339,965 |
| Credit reporting | 140,431 |
| Credit card or prepaid card | 124,830 |
| Checking or savings account | 103,548 |
| Credit card | 89,190 |
| Bank account or service | 86,206 |
| Student loan | 63,561 |
| Money transfer, virtual currency, or money service | 33,810 |

## Recommended Target Variables

- Primary NLP classification target: `Product`
- Secondary NLP classification target: `Issue`
- SLA / operations target: `Timely response?`
- Resolution outcome target: `Company response to consumer`
- Analytics dimensions: `Company`, `State`, `Submitted via`, `Date received`

## Data Quality Notes

- Only rows with non-null `Consumer complaint narrative` should be used for text classification training.
- Product labels should be normalized because older categories overlap with newer categories.
- The full dataset should not be committed to GitHub because it is large.
- Build a small sample for demo and document how to recreate it.
