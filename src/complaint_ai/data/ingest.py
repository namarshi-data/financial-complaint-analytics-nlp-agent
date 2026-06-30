from pathlib import Path

import pandas as pd

from complaint_ai.data.schema import REQUIRED_COLUMNS


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_complaints(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    validate_columns(df)
    return df


def filter_training_rows(
    df: pd.DataFrame,
    text_col: str = "Consumer complaint narrative",
    label_col: str = "Product",
    min_text_chars: int = 40,
) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[text_col, label_col])
    out[text_col] = out[text_col].astype(str)
    out = out[out[text_col].str.len() >= min_text_chars]
    return out.reset_index(drop=True)
