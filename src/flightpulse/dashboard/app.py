import os
from pathlib import Path

import pandas as pd
import psycopg2
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

API = os.environ.get("FLIGHTPULSE_API", "http://localhost:8000")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REPORTS_DIR = Path(os.environ.get("FLIGHTPULSE_REPORTS", "reports"))

st.set_page_config(page_title="FlightPulse", page_icon="", layout="wide")


def _conn():
    return psycopg2.connect(
        host=os.environ.get("FLIGHTPULSE_DB_HOST", "localhost"),
        port=int(os.environ.get("FLIGHTPULSE_DB_PORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


@st.cache_data(ttl=30)
def load_jobs() -> pd.DataFrame:
    sql = """
        SELECT created_at, task, decision, window_start, window_end,
               n_rows, drift_share, drift_detected, auc, pr_auc, model_version
        FROM jobs ORDER BY created_at DESC
    """
    conn = _conn()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


@st.cache_data(ttl=30)
def load_predictions() -> pd.DataFrame:
    sql = """
        SELECT request_ts, model_version, proba, prediction
        FROM prediction_log ORDER BY request_ts DESC LIMIT 5000
    """
    conn = _conn()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


@st.cache_data(ttl=30)
def champion_info() -> dict:
    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        c = MlflowClient()
        mv = c.get_model_version_by_alias("flightpulse", "champion")
        run = c.get_run(mv.run_id)
        return {
            "version": mv.version,
            "run_name": run.data.tags.get("mlflow.runName", "?"),
            "auc": run.data.metrics.get("auc"),
            "pr_auc": run.data.metrics.get("pr_auc"),
            "features": run.data.tags.get("features", "?"),
            "leakage_check": run.data.tags.get("leakage_check", "?"),
        }
    except Exception as e:
        return {"error": str(e)}


def api_health() -> dict:
    try:
        return requests.get(f"{API}/health", timeout=5).json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


st.title("FlightPulse — Drift-Aware Flight Delay Monitoring")
st.caption("Pre-departure delay prediction with automated drift detection and retraining")

tab_overview, tab_drift, tab_preds, tab_try = st.tabs(
    ["Overview", "Drift Reports", "Predictions", "Try It"]
)
# Overview
with tab_overview:
    champ = champion_info()
    health = api_health()

    c1, c2, c3, c4 = st.columns(4)
    if "error" not in champ:
        c1.metric("Champion", f"v{champ['version']}", champ["run_name"])
        c2.metric("ROC-AUC", f"{champ['auc']:.4f}" if champ["auc"] else "—")
        c3.metric("PR-AUC", f"{champ['pr_auc']:.4f}" if champ["pr_auc"] else "—")
    else:
        c1.metric("Champion", "unavailable")
        st.warning(f"MLflow: {champ['error']}")
    c4.metric("API", health.get("status", "?"),
              f"model v{health.get('model_version', '?')}")

    if "error" not in champ:
        st.caption(f"Features: `{champ['features']}` · Leakage check: "
                   f"`{champ['leakage_check']}`")

    st.subheader("Pipeline decisions")
    jobs = load_jobs()
    if jobs.empty:
        st.info("No pipeline runs logged yet. Trigger the DAG in Airflow.")
    else:
        counts = jobs["decision"].value_counts()
        d1, d2, d3 = st.columns(3)
        d1.metric("Retrained + promoted", int(counts.get("retrained_promoted", 0)))
        d2.metric("Skipped (no drift)", int(counts.get("no_drift_skip", 0)))
        d3.metric("Halted (quality)", int(counts.get("quality_failed", 0)))

        st.dataframe(
            jobs[["created_at", "decision", "window_start", "window_end",
                  "drift_detected", "drift_share", "auc", "pr_auc", "model_version"]],
            use_container_width=True, hide_index=True,
        )

        hist = jobs.dropna(subset=["pr_auc"]).sort_values("created_at")
        if len(hist) > 1:
            st.subheader("Champion PR-AUC over retrains")
            st.line_chart(hist.set_index("created_at")[["auc", "pr_auc"]])

# Drift
with tab_drift:
    st.subheader("Evidently drift reports")
    reports = sorted(REPORTS_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime,
                     reverse=True) if REPORTS_DIR.exists() else []
    if not reports:
        st.info(f"No reports found in `{REPORTS_DIR}`. Run the DAG's drift task "
                f"or the drift_validation notebook.")
    else:
        choice = st.selectbox("Report", [p.name for p in reports])
        path = REPORTS_DIR / choice
        st.caption(f"{path} · {path.stat().st_size/1024:.0f} KB")
        components.html(path.read_text(encoding="utf-8"), height=800, scrolling=True)

# Predictions
with tab_preds:
    preds = load_predictions()
    if preds.empty:
        st.info("No predictions logged yet. Run: "
                "`python -m src.flightpulse.generate_traffic --n 300`")
    else:
        p1, p2, p3 = st.columns(3)
        p1.metric("Predictions logged", len(preds))
        p2.metric("Mean probability", f"{preds['proba'].mean():.4f}")
        p3.metric("Predicted delayed", f"{preds['prediction'].mean():.1%}")

        st.subheader("Predicted probability distribution")
        st.caption("Training base rate was ~0.17 (2019-2022); 2023 test ~0.22. "
                   "A live distribution far from these is a drift signal.")
        bins = [i / 20 for i in range(21)]
        binned = pd.cut(preds["proba"], bins=bins)
        counts = binned.value_counts().sort_index()
        # Altair can't serialize pandas Interval objects — use the bin's left edge.
        chart_df = pd.DataFrame({
            "probability": [float(iv.left) for iv in counts.index],
            "flights": counts.to_numpy(),
        }).set_index("probability")
        st.bar_chart(chart_df)

        st.subheader("Recent predictions")
        st.dataframe(preds.head(50), use_container_width=True, hide_index=True)

# Try it
with tab_try:
    st.subheader("Live prediction")
    st.caption(f"POSTs to {API}/predict — the same champion the DAG promotes.")

    c1, c2 = st.columns(2)
    with c1:
        fl_date = st.text_input("Flight date", "2023-02-15")
        airline = st.text_input("Airline code", "WN")
        origin = st.text_input("Origin", "LAX")
        dest = st.text_input("Destination", "JFK")
    with c2:
        crs_dep_time = st.number_input("Scheduled departure (hhmm)", value=1430, step=5)
        crs_elapsed_time = st.number_input("Scheduled minutes", value=330.0)
        distance = st.number_input("Distance (miles)", value=2475.0)

    if st.button("Predict", type="primary"):
        payload = {
            "fl_date": fl_date, "airline": airline, "origin": origin, "dest": dest,
            "crs_dep_time": int(crs_dep_time),
            "crs_elapsed_time": float(crs_elapsed_time),
            "distance": float(distance),
        }
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=20)
            if r.status_code == 200:
                d = r.json()
                m1, m2, m3 = st.columns(3)
                m1.metric("P(delay > 15 min)", f"{d['delay_gt_15_probability']:.2%}")
                m2.metric("Prediction", "DELAYED" if d["prediction"] else "ON TIME")
                m3.metric("Congestion", d["congestion"],
                          help="Scheduled departures from this origin in this hour "
                               "(looked up from Postgres, matching training)")
                st.json(d)
            else:
                st.error(f"{r.status_code}: {r.text[:400]}")
        except Exception as e:
            st.error(str(e))

