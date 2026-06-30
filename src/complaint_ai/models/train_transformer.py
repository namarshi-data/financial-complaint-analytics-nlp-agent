"""Optional BERT / RoBERTa fine-tuning script.

Install requirements-ml.txt before running this file.
Example:
python -m complaint_ai.models.train_transformer --input data/sample/complaints_sample_1000.csv --model_name distilbert-base-uncased
"""
from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from complaint_ai.data.ingest import filter_training_rows, load_complaints
from complaint_ai.data.preprocess import prepare_text_classification_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="models/transformer")
    parser.add_argument("--max_rows", type=int, default=50000)
    args = parser.parse_args()

    df = prepare_text_classification_frame(filter_training_rows(load_complaints(args.input, nrows=args.max_rows)))
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["label_clean"])

    dataset = Dataset.from_pandas(df[["text_clean", "labels"]]).train_test_split(test_size=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text_clean"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_),
    )
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(Path(args.output_dir) / "best_model")


if __name__ == "__main__":
    main()
