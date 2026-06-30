import re

import pandas as pd

REDACTION_PATTERNS = [
    r"XX/XX/XXXX",
    r"X{2,}",
    r"\{\$[\d,\.]+\}",
]


def clean_text(text: str) -> str:
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    for pattern in REDACTION_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^A-Za-z0-9%$.,!?\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str) -> str:
    if label is None or pd.isna(label):
        return "unknown"
    label = str(label).strip().lower()
    replacements = {
        "credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
        "credit reporting": "credit_reporting",
        "debt collection": "debt_collection",
        "mortgage": "mortgage",
        "credit card or prepaid card": "credit_card_or_prepaid_card",
        "credit card": "credit_card_or_prepaid_card",
        "checking or savings account": "bank_account",
        "bank account or service": "bank_account",
        "student loan": "student_loan",
        "vehicle loan or lease": "vehicle_loan_or_lease",
        "money transfer, virtual currency, or money service": "money_transfer",
    }
    return replacements.get(label, re.sub(r"[^a-z0-9]+", "_", label).strip("_"))


def prepare_text_classification_frame(
    df: pd.DataFrame,
    text_col: str = "Consumer complaint narrative",
    label_col: str = "Product",
) -> pd.DataFrame:
    out = df.copy()
    out["text_clean"] = out[text_col].map(clean_text)
    out["label_clean"] = out[label_col].map(normalize_label)
    out = out[(out["text_clean"].str.len() > 40) & (out["label_clean"] != "unknown")]
    return out.reset_index(drop=True)
