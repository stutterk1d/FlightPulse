import os

import pendulum
from airflow.sdk import dag, get_current_context, task



def _slack(text: str):
    # Use slack
    import requests
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print(f"[slack:noop] {text}")
        return
    try:
        requests.post(url, json={"text": text}, timeout=10)
    except Exception as e:
        print(f"[slack:error] {e} :: {text}")


@dag(
    schedule=None,
    start_date=pendulum.datetime(2019, 1, 1, tz="UTC"),
    catchup=False,
    tags=["flightpulse"],
    params={
        "reference_start": "2019-06-01",
        "reference_end": "2019-06-30",
        "window_start": "2023-01-01",
        "window_end": "2023-01-31",
    },
)
def flightpulse_pipeline():

    @task
    def extract() -> dict:
        ctx = get_current_context()
        p = ctx["params"]
        return {
            "ref_start": p["reference_start"], "ref_end": p["reference_end"],
            "win_start": p["window_start"], "win_end": p["window_end"],
            "run_id": ctx["dag_run"].run_id,
        }

    @task
    def data_quality(meta: dict) -> dict:
        from src.flightpulse.features import build_features
        from src.flightpulse.queries import load_window
        df = build_features(load_window(meta["win_start"], meta["win_end"]))
        # simple integrity checks (cheap, no Deepchecks needed for the gate)
        n = len(df)
        dup_ratio = float(df.duplicated().mean())
        passed = (n > 1000) and (dup_ratio < 0.3)
        meta.update({"n_rows": n, "dup_ratio": dup_ratio, "quality_passed": passed})
        return meta

    @task.branch
    def gate_quality(meta: dict) -> str:
        return "detect_drift" if meta["quality_passed"] else "halt"

    @task
    def detect_drift(meta: dict) -> dict:
        from src.flightpulse import drift
        from src.flightpulse.features import build_features
        from src.flightpulse.queries import load_window
        ref = build_features(load_window(meta["ref_start"], meta["ref_end"]))
        cur = build_features(load_window(meta["win_start"], meta["win_end"]))
        res = drift.evidently_drift(ref, cur, html_name=f"drift_{meta['run_id']}.html")
        meta.update({
            "drift_detected": bool(res["dataset_drift"]),
            "drift_share": float(res["drifted_share"]) if res["drifted_share"] is not None else None,
        })
        return meta

    @task.branch
    def gate_drift(meta: dict) -> str:
        return "retrain" if meta["drift_detected"] else "skip_retrain"

    @task
    def retrain(meta: dict) -> dict:
        from src.flightpulse.train import train_and_register
        results = train_and_register()  # full 2019-2022 train, 2023 test
        # pull the champion's metrics for logging
        best = max(results.values(), key=lambda r: r["test"]["pr_auc"])
        meta.update({
            "auc": float(best["test"]["auc"]),
            "pr_auc": float(best["test"]["pr_auc"]),
            "model_version": str(best["version"]),
        })
        return meta

    @task
    def promote(meta: dict):
        from src.flightpulse.jobs import log_decision
        _slack(f":white_check_mark: FlightPulse retrained & promoted. "
               f"AUC={meta['auc']:.4f} PR-AUC={meta['pr_auc']:.4f} "
               f"(drift_share={meta.get('drift_share')})")
        log_decision(meta["run_id"], "promote", "retrained_promoted",
                     meta["win_start"], meta["win_end"], n_rows=meta.get("n_rows"),
                     drift_share=meta.get("drift_share"), drift_detected=True,
                     auc=meta.get("auc"), pr_auc=meta.get("pr_auc"),
                     model_version=meta.get("model_version"))

    @task
    def skip_retrain(meta: dict):
        from src.flightpulse.jobs import log_decision
        _slack(f":white_circle: FlightPulse: no drift detected "
               f"(share={meta.get('drift_share')}); skipping retrain.")
        log_decision(meta["run_id"], "skip_retrain", "no_drift_skip",
                     meta["win_start"], meta["win_end"], n_rows=meta.get("n_rows"),
                     drift_share=meta.get("drift_share"), drift_detected=False)

    @task
    def halt(meta: dict):
        from src.flightpulse.jobs import log_decision
        _slack(f":red_circle: FlightPulse: data-quality check failed "
               f"(rows={meta.get('n_rows')}, dup={meta.get('dup_ratio')}); halting.")
        log_decision(meta["run_id"], "halt", "quality_failed",
                     meta["win_start"], meta["win_end"], n_rows=meta.get("n_rows"))

    m = extract()
    dq = data_quality(m)
    gq = gate_quality(dq)

    d = detect_drift(dq)
    gd = gate_drift(d)

    r = retrain(d)
    p = promote(r)
    s = skip_retrain(d)
    h = halt(dq)

    gq >> [d, h]
    gd >> [r, s]
    r >> p


flightpulse_pipeline()
