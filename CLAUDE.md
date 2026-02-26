Goal: Build a minimal Python prototype (Streamlit) for “Listening A[i]gent: Signal Intelligence & Insights”.
Constraints:
- Keep changes minimal; do not rewrite files unless asked.
- Prefer small diffs and short plans.
- Use pandas + numpy + matplotlib + streamlit only (unless requested).
- App must run locally via: streamlit run app/app.py
- Data is synthetic and stored at data/signals.csv
Deliverables:
- src/generate_signals.py generates data/signals.csv
- app/app.py renders: sentiment trend, dept×theme heatmap, and drilldown evidence table