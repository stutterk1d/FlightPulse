import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def _conn():
    return psycopg2.connect(
        host=os.environ.get("FLIGHTPULSE_DB_HOST", "localhost"),
        port=int(os.environ.get("FLIGHTPULSE_DB_PORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )

def log_decision(dag_run_id, task, decision, window_start, window_end,
                 n_rows=None, drift_share=None, drift_detected=None,
                 auc=None, pr_auc=None, model_version=None):
    sql = """
        INSERT INTO jobs (dag_run_id, task, decision, window_start, window_end,
                          n_rows, drift_share, drift_detected, auc, pr_auc, model_version)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (dag_run_id, task, decision, window_start, window_end,
                              n_rows, drift_share, drift_detected, auc, pr_auc, model_version))
        conn.commit()
    finally:
        conn.close()
