from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def build_tfidf_logreg_pipeline(max_features: int = 50000) -> Pipeline:
    """Build a fast, deployment-friendly text classifier.

    The SGDClassifier with log loss behaves like a scalable logistic-regression model and
    trains quickly on large text datasets.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
