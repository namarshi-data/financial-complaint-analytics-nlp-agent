from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_sample(input_path: str, output_path: str, n: int = 1000) -> None:
    usecols = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company public response",
        "Company",
        "State",
        "Submitted via",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
        "Complaint ID",
    ]
    chunks = []
    total = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=100000, low_memory=False):
        chunk = chunk.dropna(subset=["Consumer complaint narrative", "Product"])
        chunks.append(chunk)
        total += len(chunk)
        if total >= n:
            break
    sample = pd.concat(chunks, ignore_index=True).head(n)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/sample/complaints_sample_1000.csv")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    make_sample(args.input, args.output, args.n)


if __name__ == "__main__":
    main()
