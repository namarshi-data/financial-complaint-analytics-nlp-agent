from complaint_ai.data.preprocess import clean_text, normalize_label
from complaint_ai.agents.priority import score_priority


def test_clean_text_removes_redactions():
    text = "I paid {$270.00} on XX/XX/XXXX and XXXX said no."
    cleaned = clean_text(text)
    assert "XX/XX/XXXX" not in cleaned
    assert "{$270.00}" not in cleaned


def test_normalize_label_credit_reporting():
    label = "Credit reporting, credit repair services, or other personal consumer reports"
    assert normalize_label(label) == "credit_reporting"


def test_priority_detects_fraud():
    result = score_priority("There was unauthorized fraud on my account.", "bank_account", 0.8)
    assert result.priority in {"High", "Critical"}
    assert result.human_review_required is True
