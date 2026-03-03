import pandas as pd

THEMES = {
    "Culture",
    "Leadership & Management",
    "Role Clarity",
    "Performance & Rewards",
    "Capacity",
    "Collaboration",
    "Learning & Capability",
    "Governance",
    "Process",
    "Innovation",
    "Tools & Technology",
}

DECISION_QUEUES = ["Policy/Process Review", "People Review", "Product Review"]

def classify_category(row) -> str:
    """
    Category is your intent layer:
    - Sentiment = temperature/attitude
    - Function = actionable proposal / concrete fix request
    """
    stype = row["signal_type"]

    # Explicit proposals
    if stype == "Suggestion":
        return "Function"

    # Strong sentiment expressions
    if stype in ["Complaint", "Praise"]:
        return "Sentiment"

    return "Sentiment"


def needs_validation(row) -> bool:
    """
    Validation is a gate that prevents opinion -> action without checks.
    Trigger validation for:
    - Function category signals
    - Sentiment signals that are high severity
    """
    if row["category"] == "Function":
        return True
    if row["category"] == "Sentiment" and row["severity"] >= 5:
        return True
    return False


def route_queue(row) -> str:
    """
    Queue is where a signal goes for processing.
    Sentiment signals go to Reporting; Function signals go to Validation.
    """
    if row["category"] == "Sentiment":
        return "Reporting"

    # Function (proposal-like) goes through validation first
    return "Validation"


def post_validation_queue(row) -> str:
    """
    After validation succeeds, route to a decision body.
    Theme is informational until validation; once validated, theme can route downstream.
    """
    theme = row["theme"]

    if theme in ["Innovation", "Tools & Technology"]:
        return "Product Review"

    if theme in ["Governance", "Process"]:
        return "Policy/Process Review"

    # People Review themes
    if theme in [
        "Capacity",
        "Collaboration",
        "Learning & Capability",
        "Role Clarity",
        "Leadership & Management",
        "Performance & Rewards",
        "Culture",
    ]:
        return "People Review"

    # Fallback if a new theme appears
    return "Policy/Process Review"


def apply_engine(df: pd.DataFrame, validate_pass_rate: float = 0.65) -> pd.DataFrame:
    """
    Applies your operating model to raw signals.
    This is where you evolve logic over time.
    """
    out = df.copy()

    # 1) category
    out["category"] = out.apply(classify_category, axis=1)

    # 2) initial queue (monitoring vs validation)
    out["queue"] = out.apply(route_queue, axis=1)

    # 3) status baseline — Watching is a status, not a queue
    out["status"] = "Open"
    watching_mask = (out["queue"] == "Reporting") & (out["sentiment_score"] <= -0.4) & (out["severity"] >= 3)
    out.loc[watching_mask, "status"] = "Watching"

    # 4) validation trigger + outcome (simulated for now)
    out["needs_validation"] = out.apply(needs_validation, axis=1)

    # Only items in Validation lane are eligible
    import numpy as np
    rng = np.random.default_rng(42)
    out["validated"] = False

    mask = out["queue"] == "Validation"
    out.loc[mask, "validated"] = rng.random(mask.sum()) < validate_pass_rate

    # 5) route validated items to decision queues + status
    validated_mask = mask & out["validated"]
    out.loc[validated_mask, "queue"] = out.loc[validated_mask].apply(post_validation_queue, axis=1)
    out.loc[validated_mask, "status"] = "In Process"

    # 6) close some non-validated items (simulated disposition)
    not_validated_mask = mask & (~out["validated"])
    # keep some open, close some
    close_mask = not_validated_mask & (rng.random(len(out)) < 0.35)
    out.loc[close_mask, "status"] = "Closed"

    return out