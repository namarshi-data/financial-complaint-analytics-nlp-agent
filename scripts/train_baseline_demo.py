from complaint_ai.models.train_baseline import train_baseline


if __name__ == "__main__":
    metrics = train_baseline(
        input_path="data/sample/complaints_sample_1000.csv",
        model_path="models/baseline_tfidf_logreg.joblib",
        metrics_path="reports/baseline_metrics.json",
    )
    print("Baseline training complete")
    print({k: v for k, v in metrics.items() if k != "classification_report"})
