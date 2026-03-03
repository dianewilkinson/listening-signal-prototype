import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
if st.button("Force reload data"):
    st.cache_data.clear()

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

st.set_page_config(page_title="Listening A[i]gent Prototype", layout="wide")

st.title("Listening A[i]gent: Signal Intelligence Dashboard")

# Load data — passes mtime as arg so cache invalidates when file changes
import os

@st.cache_data
def load_data(mtime):
    return pd.read_csv("data/signals.csv", parse_dates=["timestamp"])

df = load_data(os.path.getmtime("data/signals.csv"))

# -----------------------------
# Top Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Signals", len(df))
col2.metric("Routed Signals", int(df["routed"].sum()))
col3.metric("Closed Signals", len(df[df["status"] == "Closed"]))

# -----------------------------
# Charts (Compact Layout)
# -----------------------------
col_left, col_right = st.columns([2, 2], gap="large")

with col_left:
    st.markdown("**Sentiment Distribution (Daily Share)**")

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").set_index("timestamp")
    d = d[d.index >= d.index.max() - pd.DateOffset(months=6)]

    weekly_total = d["sentiment_score"].resample("W").count()
    weekly_pos = (d["sentiment_score"] > 0).resample("W").sum()
    weekly_neg = (d["sentiment_score"] < 0).resample("W").sum()

    weekly = pd.DataFrame({
    "Positive": (weekly_pos / weekly_total).fillna(0),
    "Negative": (weekly_neg / weekly_total).fillna(0),
})

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=120)

    width = 0.35
    x = np.arange(len(weekly))

    ax.bar(x - width/2, weekly["Positive"], width=width, label="Positive", color="#4E9E9E")
    ax.bar(x + width/2, weekly["Negative"], width=width, label="Negative", color="#9E9E9E")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of signals")

    month_starts = [i for i, d in enumerate(weekly.index) if d.day <= 7]
    ax.set_xticks(x)
    ax.set_xticklabels(["" for _ in x])
    ax.set_xticks([x[i] for i in month_starts], minor=False)
    ax.set_xticklabels([weekly.index[i].strftime("%b '%y") for i in month_starts])

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_right:
    st.markdown("**Sentiment by Department (Positive vs Negative)**")

    d = df.copy()

    dept = (
        d.groupby("department")
         .agg(
             positive=("sentiment_score", lambda s: (s > 0).sum()),
             negative=("sentiment_score", lambda s: (s < 0).sum()),
             total=("sentiment_score", "count")
         )
         .sort_values("negative", ascending=False)
    )

    # Use shares so departments with more volume don't dominate
    dept["pos_share"] = dept["positive"] / dept["total"]
    dept["neg_share"] = dept["negative"] / dept["total"]

    y = np.arange(len(dept.index))

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=120)

    # Negative to the left (plot as negative values)
    ax.barh(y, -dept["neg_share"], label="Negative", color="#9E9E9E")
    ax.barh(y, dept["pos_share"], label="Positive", color="#4E9E9E")

    ax.set_yticks(y)
    ax.set_yticklabels(dept.index, fontsize=8)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Share of signals")
    ax.grid(axis="x", alpha=0.2)

    # Make x labels show as positive percentages on both sides
    ax.set_xlim(-1, 1)

    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Drilldown Table
# -----------------------------
st.subheader("Drilldown: Raw Signals")

selected_theme = st.selectbox("Filter by Theme", ["All"] + list(df["theme"].unique()))

if selected_theme != "All":
    filtered = df[df["theme"] == selected_theme]
else:
    filtered = df

# -----------------------------
# Dimension Options (unique values + counts)
# -----------------------------


st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True,
             height=320)