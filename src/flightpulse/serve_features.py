import os
from pathlib import Path

import holidays
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_US_HOLIDAYS = holidays.UnitedStates()

FEATURE_COLS = [
    "route", "origin", "dest",
    "airline", "season", "dep_bucket",
    "is_weekend", "is_holiday",
    "distance", "crs_elapsed_time",
    "dep_minutes", "congestion",
]


def _conn():
    return psycopg2.connect(
        host=os.environ.get("FLIGHTPULSE_DB_HOST", "localhost"),
        port=int(os.environ.get("FLIGHTPULSE_DB_PORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def _hhmm_to_minutes(t: int) -> int:
    t = 0 if t == 2400 else int(t)
    return (t // 100) * 60 + (t % 100)


def _dep_bucket(hour: int) -> str:
    if hour <= 5:
        return "overnight"
    if hour <= 11:
        return "morning"
    if hour <= 16:
        return "midday"
    if hour <= 20:
        return "evening"
    return "night"


def lookup_congestion(origin: str, fl_date: str, dep_hour: int) -> int:
    sql_exact = (
        "SELECT count(*) FROM flights "
        "WHERE origin = %s AND fl_date = %s AND (crs_dep_time / 100) = %s"
    )
    sql_fallback = (
        "SELECT COALESCE(percentile_disc(0.5) WITHIN GROUP (ORDER BY c), 1)::int "
        "FROM (SELECT count(*) AS c FROM flights "
        "WHERE origin = %s AND (crs_dep_time / 100) = %s GROUP BY fl_date) t"
    )
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_exact, (origin, fl_date, dep_hour))
            n = cur.fetchone()[0]
            if n and n > 0:
                return int(n)
            cur.execute(sql_fallback, (origin, dep_hour))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] else 1
    finally:
        conn.close()


def build_serving_features(
    fl_date: str, airline: str, origin: str, dest: str,
    crs_dep_time: int, crs_elapsed_time: float, distance: float,
) -> pd.DataFrame:
    fd = pd.to_datetime(fl_date)
    dep_min = _hhmm_to_minutes(crs_dep_time)
    dep_hour = dep_min // 60

    row = {
        "route": f"{origin}-{dest}",
        "origin": origin,
        "dest": dest,
        "airline": airline,
        "season": {0: "winter", 1: "spring", 2: "summer", 3: "fall"}[fd.month % 12 // 3],
        "dep_bucket": _dep_bucket(dep_hour),
        "is_weekend": int(fd.dayofweek >= 5),
        "is_holiday": int(fd.date() in _US_HOLIDAYS),
        "distance": float(distance),
        "crs_elapsed_time": float(crs_elapsed_time),
        "dep_minutes": int(dep_min),
        "congestion": lookup_congestion(origin, fl_date, dep_hour),
    }
    df = pd.DataFrame([row])[FEATURE_COLS]
    df["is_weekend"] = df["is_weekend"].astype("int32")
    df["is_holiday"] = df["is_holiday"].astype("int32")
    df["dep_minutes"] = df["dep_minutes"].astype("int32")
    df["congestion"] = df["congestion"].astype("int32")
    df["distance"] = df["distance"].astype("float32")
    df["crs_elapsed_time"] = df["crs_elapsed_time"].astype("float32")
    return df


