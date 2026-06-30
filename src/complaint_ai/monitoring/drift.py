import pandas as pd
from scipy.spatial.distance import jensenshannon


def category_distribution(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df[label_col].value_counts(normalize=True).sort_index()


def js_drift_score(reference: pd.Series, current: pd.Series) -> float:
    aligned = pd.concat([reference, current], axis=1).fillna(0)
    aligned.columns = ["reference", "current"]
    return float(jensenshannon(aligned["reference"], aligned["current"]))
