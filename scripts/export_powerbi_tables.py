from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from complaint_ai.data.ingest import filter_training_rows, load_complaints
from complaint_ai.data.preprocess import prepare_text_classification_frame


def export_powerbi_tables(input_path: str, output_dir: str = "data/processed/powerbi") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = prepare_text_classification_frame(filter_training_rows(load_complaints(input_path)))
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df["month"] = df["Date received"].dt.to_period("M").astype(str)
    df["narrative_word_count"] = df["text_clean"].str.split().str.len()

    fact_cols = [
        "Complaint ID",
        "Date received",
        "month",
        "Product",
        "label_clean",
        "Issue",
        "Company",
        "State",
        "Submitted via",
        "Company response to consumer",
        "Timely response?",
        "narrative_word_count",
    ]
    df[[c for c in fact_cols if c in df.columns]].to_csv(output / "fact_complaints.csv", index=False)

    monthly = (
        df.groupby(["month", "label_clean"], dropna=False)
        .agg(
            complaint_count=("Complaint ID", "count"),
            avg_narrative_word_count=("narrative_word_count", "mean"),
            timely_rate=("Timely response?", lambda x: (x == "Yes").mean()),
        )
        .reset_index()
    )
    monthly.to_csv(output / "monthly_category_kpis.csv", index=False)

    company_summary = (
        df.groupby(["Company", "label_clean"], dropna=False)
        .agg(
            complaint_count=("Complaint ID", "count"),
            timely_rate=("Timely response?", lambda x: (x == "Yes").mean()),
        )
        .reset_index()
        .sort_values("complaint_count", ascending=False)
    )
    company_summary.to_csv(output / "company_category_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/complaints_sample_1000.csv")
    parser.add_argument("--output_dir", default="data/processed/powerbi")
    args = parser.parse_args()
    export_powerbi_tables(args.input, args.output_dir)


if __name__ == "__main__":
    main()
