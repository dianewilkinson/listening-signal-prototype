# pages/2_Decision_Integrity.py
# Streamlit multipage: Decision Integrity Dashboard (synthetic demo)
# Run from repo root:
#   streamlit run app/app.py   (or streamlit run app.py — depending on your structure)

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Decision Integrity Dashboard", layout="wide")

st.title("Decision Integrity Dashboard (Synthetic Demo)")
st.caption("Purpose: make evaluation integrity visible (consistency, overrides, drift, predictive strength, false negative indicator).")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Synthetic data generator")
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)

    n_reqs = st.slider("Requisitions", 1, 12, 3)
    n_applicants = st.slider("Applicants per req", 200, 3000, 1000, step=100)

    st.subheader("Pipeline")
    qual_rate = st.slider("Must-have qualified rate", 0.02, 0.50, 0.12, step=0.01)
    interview_rate = st.slider("Interview rate within qualified", 0.05, 0.60, 0.30, step=0.01)
    hire_rate = st.slider("Hire rate within interviewed", 0.01, 0.30, 0.04, step=0.01)

    st.subheader("Scoring & drift")
    n_interviewers = st.slider("Interviewers per req", 2, 8, 4)
    severity_spread = st.slider("Interviewer severity spread", 0.1, 2.5, 1.0, step=0.1)
    drift_strength = st.slider("Pressure drift strength", 0.0, 2.0, 0.8, step=0.1)
    pressure_noise = st.slider("Pressure volatility", 0.0, 2.0, 0.8, step=0.1)

    st.subheader("Governance")
    threshold = st.slider("Rubric threshold (0–5)", 1.0, 4.5, 3.2, step=0.1)
    override_rate = st.slider("Override rate", 0.00, 0.40, 0.12, step=0.01)

    st.subheader("Outcome link")
    outcome_strength = st.slider("Predictive strength", 0.0, 2.0, 1.0, step=0.1)
    ext_window_mo = st.slider("External success window (months)", 3, 24, 12, step=1)

# ---------- Data gen ----------
np.random.seed(int(seed))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

req_ids = [f"REQ-{i+1:03d}" for i in range(int(n_reqs))]
rows = []

for r_i, req in enumerate(req_ids):
    base_pressure = (r_i / max(len(req_ids) - 1, 1))
    pressure = float(np.clip(base_pressure + np.random.normal(0, pressure_noise) * 0.2, 0, 1))

    interviewer_ids = [f"INT-{j+1:02d}" for j in range(int(n_interviewers))]
    severity = dict(zip(interviewer_ids, np.random.normal(0, severity_spread, size=len(interviewer_ids))))

    for c in range(int(n_applicants)):
        cand = f"{req}-C{c+1:04d}"
        qualified = np.random.rand() < qual_rate

        latent = np.random.normal(0.0, 1.0) + (0.8 if qualified else -0.2)

        interviewed = bool(qualified and (np.random.rand() < interview_rate))

        score = np.nan
        rater_sd = np.nan
        if interviewed:
            drift = drift_strength * (pressure - 0.5)
            true_score = 3.0 + 0.9 * latent + drift

            indiv = []
            for iid in interviewer_ids:
                indiv.append(true_score + severity[iid] + np.random.normal(0, 0.7))
            indiv = np.array(indiv)

            mean_score = float(np.mean(indiv))
            rater_sd = float(np.std(indiv, ddof=0))
            score = float(np.clip(mean_score, 0, 5))

        # rule-based decision + overrides
        override = False
        advanced = False
        if interviewed:
            rule = score >= threshold
            if np.random.rand() < override_rate:
                override = True
                advanced = not rule
            else:
                advanced = rule

        hired = bool(interviewed and advanced and (np.random.rand() < hire_rate))

        performance_good = np.nan
        if hired:
            p_good = sigmoid(outcome_strength * (0.9 * latent + 0.35 * (score - 3.0)))
            performance_good = float(np.random.rand() < p_good)

        success_elsewhere = False
        if (not hired) and qualified:
            base = sigmoid(0.9 * latent + 0.25 * ((0 if pd.isna(score) else score) - 3.0))
            window_factor = np.clip(ext_window_mo / 12.0, 0.25, 2.0)
            p_ext = float(np.clip(0.18 * base * window_factor, 0, 0.65))
            success_elsewhere = bool(np.random.rand() < p_ext)

        rows.append(
            dict(
                req_id=req,
                candidate_id=cand,
                pressure=pressure,
                qualified=qualified,
                interviewed=interviewed,
                score=score,
                rater_sd=rater_sd,
                advanced=advanced,
                override=override,
                hired=hired,
                performance_good=performance_good,
                success_elsewhere=success_elsewhere,
            )
        )

df = pd.DataFrame(rows)

# ---------- KPIs ----------
total = len(df)
qual = int(df["qualified"].sum())
interviewed = int(df["interviewed"].sum())
hires = int(df["hired"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Applicants", f"{total:,}")
k2.metric("Qualified", f"{qual:,}", f"{(qual/total)*100:.1f}%")
k3.metric("Interviewed (within qualified)", f"{interviewed:,}", f"{(interviewed/max(qual,1))*100:.1f}%")
k4.metric("Hires (within interviewed)", f"{hires:,}", f"{(hires/max(interviewed,1))*100:.1f}%")

st.divider()

# ---------- 1) Decision Consistency (proxy) ----------
st.subheader("1) Decision Consistency (proxy)")
st.caption("Proxy = average rater standard deviation (SD) within a req. Lower SD = more consistent scoring.")

cons = (
    df[df["interviewed"]]
    .groupby("req_id", as_index=False)
    .agg(interviewed=("candidate_id", "count"), avg_rater_sd=("rater_sd", "mean"))
    .sort_values("avg_rater_sd", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
ax.bar(cons["req_id"], cons["avg_rater_sd"])
ax.set_ylabel("Avg rater SD")
ax.set_xlabel("Req")
ax.grid(axis="y", alpha=0.2)
plt.xticks(rotation=0)
st.pyplot(fig, use_container_width=True)

st.divider()

# ---------- 2) Rubric adherence / overrides ----------
st.subheader("2) Rubric Adherence / Overrides")
st.caption("Override rate = share of interviewed decisions that contradict the threshold rule.")

ov = (
    df[df["interviewed"]]
    .groupby("req_id", as_index=False)
    .agg(interviewed=("candidate_id", "count"), override_rate=("override", "mean"))
    .sort_values("override_rate", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
ax.bar(ov["req_id"], ov["override_rate"])
ax.set_ylim(0, 1)
ax.set_ylabel("Override rate")
ax.set_xlabel("Req")
ax.grid(axis="y", alpha=0.2)
st.pyplot(fig, use_container_width=True)

st.divider()

# ---------- 3) Drift under pressure ----------
st.subheader("3) Calibration Drift (scores vs pressure)")
st.caption("If average scores rise/fall with pressure, your bar is drifting under load.")

by_req = (
    df[df["interviewed"]]
    .groupby("req_id", as_index=False)
    .agg(avg_score=("score", "mean"), avg_pressure=("pressure", "mean"), n=("candidate_id", "count"))
)

fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
ax.scatter(by_req["avg_pressure"], by_req["avg_score"], s=np.clip(by_req["n"] / 3, 20, 300))
ax.set_xlabel("Avg pressure (0–1)")
ax.set_ylabel("Avg score (0–5)")
ax.grid(alpha=0.2)
st.pyplot(fig, use_container_width=True)

st.divider()

# ---------- 4) Predictive strength ----------
st.subheader("4) Predictive Strength (score ↔ downstream outcome)")
st.caption("In real data: correlate interview scores/bands to performance or ramp outcomes.")

hi = df[df["hired"]].dropna(subset=["score", "performance_good"]).copy()
if len(hi) >= 5 and hi["performance_good"].nunique() > 1:
    corr = float(hi["score"].corr(hi["performance_good"]))
    st.metric("Correlation (demo)", f"{corr:.2f}")
else:
    st.metric("Correlation (demo)", "—")
    st.caption("Not enough hired rows in this synthetic run to compute a stable correlation. Increase reqs or hire rate.")

st.divider()

# ---------- 5) False negative indicator ----------
st.subheader("5) False Negative Indicator (proxy)")
strong_cut = min(5.0, threshold + 0.5)
st.caption(f"Among qualified + strong-band candidates (score ≥ {strong_cut:.1f}) who were NOT hired, what % show success elsewhere within window?")

strong_rej = df[(df["qualified"]) & (df["interviewed"]) & (df["score"] >= strong_cut) & (~df["hired"])]
rate = float(strong_rej["success_elsewhere"].mean()) if len(strong_rej) else np.nan
st.metric("False negative indicator rate", "—" if np.isnan(rate) else f"{rate*100:.1f}%")
st.caption("This is a proxy. In real data you’d define success elsewhere via LinkedIn updates, boomerang hires, or comparable-role placement signals.")

st.divider()

st.subheader("Underlying rows")
st.dataframe(
    df[[
        "req_id","candidate_id","pressure",
        "qualified","interviewed","score",
        "advanced","override","hired",
        "performance_good","success_elsewhere"
    ]].sort_values(["req_id","candidate_id"]),
    use_container_width=True,
    height=420
)