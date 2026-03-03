import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from engine import apply_engine

# Configuration
NUM_EVENTS = 500
DEPARTMENTS = ["People", "Operations", "Engineering", "Product", "Finance", "Marketing", "Sales", "Legal"]
THEMES = ["Governance", "Capacity", "Collaboration", "Innovation", "Learning & Capability", "Process", "Role Clarity", "Tools & Technology", "Leadership & Management", "Performance & Rewards", "Culture"]

np.random.seed(42)


def generate_timestamp():
    start = datetime.now() - timedelta(days=365)
    return start + timedelta(
        seconds=np.random.randint(0, 365 * 24 * 60 * 60)
    )


def generate_signals(n=NUM_EVENTS):
    data = []

    for i in range(n):
        dept = np.random.choice(DEPARTMENTS)
        theme = np.random.choice(THEMES)
        sentiment = np.random.normal(loc=0, scale=0.5)
        sentiment = max(min(sentiment, 1), -1)
        severity = np.random.randint(1, 6)

        signal_type = np.random.choice(
            ["Question", "Complaint", "Suggestion", "Praise", "Observation"]
        )

        routed = np.random.choice([0, 1], p=[0.4, 0.6])
        snippet = f"{signal_type} about {theme} in {dept}"

        data.append(
            {
                "event_id": i,
                "timestamp": generate_timestamp(),
                "department": dept,
                "theme": theme,
                "sentiment_score": sentiment,
                "severity": severity,
                "signal_type": signal_type,
                "routed": routed,
                "snippet": snippet,
            }
        )

    df = pd.DataFrame(data)
    return apply_engine(df)


if __name__ == "__main__":
    df = generate_signals()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/signals.csv", index=False)

    print("Generated data/signals.csv")
