# Power BI Dashboard Specification

## Dashboard Goal

Show how complaint analytics can help financial-services teams monitor customer experience, product risk, escalation workload, and response performance.

## Data Tables

Use `make powerbi-export` to generate:

- `fact_complaints.csv`
- `monthly_category_kpis.csv`
- `company_category_summary.csv`

## Page 1: Executive Overview

Recommended visuals:

- KPI cards: total complaints, narrative complaints, timely response rate, high-risk complaints
- line chart: monthly complaint trend
- bar chart: top product categories
- bar chart: top issues
- slicers: date, product, company, state, submitted via

## Page 2: Risk & Escalation

Recommended visuals:

- critical/high priority complaint count
- matrix by product and issue
- trend of fraud / credit reporting / debt collection complaints
- table of complaints requiring human review

## Page 3: Company & Product Insights

Recommended visuals:

- company complaint count
- product mix by company
- response outcome distribution
- timely response rate by company

## Page 4: Model Performance

Recommended visuals:

- model comparison table
- macro F1 by model
- confusion matrix heatmap
- low-confidence complaint examples

## Page 5: RAG & Agent Audit

Recommended visuals:

- complaint ID
- predicted category
- priority score
- retrieved policy source
- human review flag
- generated response status

## DAX Measures

```DAX
Total Complaints = COUNTROWS(fact_complaints)

Timely Response Rate =
DIVIDE(
    CALCULATE(COUNTROWS(fact_complaints), fact_complaints[Timely response?] = "Yes"),
    [Total Complaints]
)

Avg Narrative Word Count = AVERAGE(fact_complaints[narrative_word_count])

Complaint MoM Change =
VAR CurrentMonth = [Total Complaints]
VAR PreviousMonth = CALCULATE([Total Complaints], DATEADD('Date'[Date], -1, MONTH))
RETURN DIVIDE(CurrentMonth - PreviousMonth, PreviousMonth)
```
