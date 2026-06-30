from dataclasses import dataclass


CRITICAL_KEYWORDS = [
    "fraud",
    "identity theft",
    "unauthorized",
    "foreclosure",
    "lawsuit",
    "garnishment",
    "eviction",
    "bankruptcy",
    "discrimination",
]

HIGH_KEYWORDS = [
    "credit score",
    "collections",
    "harassment",
    "late fee",
    "incorrect report",
    "closed account",
    "chargeback",
    "escalation",
]


@dataclass
class PriorityResult:
    priority: str
    score: int
    reasons: list[str]
    human_review_required: bool


def score_priority(text: str, category: str | None = None, confidence: float | None = None) -> PriorityResult:
    t = (text or "").lower()
    score = 20
    reasons: list[str] = []

    for keyword in CRITICAL_KEYWORDS:
        if keyword in t:
            score += 35
            reasons.append(f"Critical keyword detected: {keyword}")
            break

    for keyword in HIGH_KEYWORDS:
        if keyword in t:
            score += 20
            reasons.append(f"High-risk keyword detected: {keyword}")
            break

    if category in {"mortgage", "debt_collection", "credit_reporting"}:
        score += 10
        reasons.append(f"Regulatory-sensitive category: {category}")

    if confidence is not None and confidence < 0.45:
        score += 15
        reasons.append("Low model confidence; human review recommended")

    if len(t.split()) > 250:
        score += 5
        reasons.append("Long narrative; likely complex case")

    score = min(score, 100)
    if score >= 75:
        priority = "Critical"
    elif score >= 55:
        priority = "High"
    elif score >= 35:
        priority = "Medium"
    else:
        priority = "Low"

    return PriorityResult(
        priority=priority,
        score=score,
        reasons=reasons or ["No high-risk keywords detected"],
        human_review_required=score >= 55 or (confidence is not None and confidence < 0.45),
    )
