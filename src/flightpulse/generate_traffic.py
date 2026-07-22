import argparse
import os
import time
from pathlib import Path

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

API = os.environ.get("FLIGHTPULSE_API", "http://localhost:8000")


def _conn():
    return psycopg2.connect(
        host=os.environ.get("FLIGHTPULSE_DB_HOST", "localhost"),
        port=int(os.environ.get("FLIGHTPULSE_DB_PORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def sample_flights(n: int = 300) -> pd.DataFrame:
    sql = (
        "SELECT fl_date::text AS fl_date, airline_code AS airline, origin, dest, "
        "crs_dep_time, crs_elapsed_time, distance FROM flights "
        "WHERE fl_date BETWEEN '2023-01-01' AND '2023-08-31' "
        "AND cancelled = 0 AND diverted = 0 "
        "AND crs_elapsed_time IS NOT NULL AND distance IS NOT NULL "
        "ORDER BY random() LIMIT %s"
    )
    conn = _conn()
    try:
        return pd.read_sql(sql, conn, params=(n,))
    finally:
        conn.close()


def main(n: int, delay: float):
    df = sample_flights(n)
    print(f"Sending {len(df)} real flights to {API}/predict ...")
    ok = err = 0
    for _, r in df.iterrows():
        payload = {
            "fl_date": r["fl_date"],
            "airline": str(r["airline"]),
            "origin": str(r["origin"]),
            "dest": str(r["dest"]),
            "crs_dep_time": int(r["crs_dep_time"]),
            "crs_elapsed_time": float(r["crs_elapsed_time"]),
            "distance": float(r["distance"]),
        }
        try:
            resp = requests.post(f"{API}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                ok += 1
            else:
                err += 1
                if err <= 3:
                    print(f"  [{resp.status_code}] {resp.text[:200]}")
        except Exception as e:
            err += 1
            if err <= 3:
                print(f"  [error] {e}")
        if delay:
            time.sleep(delay)
    print(f"Done: {ok} ok, {err} errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds between requests (0 = as fast as possible)")
    main(*vars(ap.parse_args()).values())

