import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / '.env')

_SELECT_COLS = [
    "fl_date", "airline", "airline_code", "origin", "dest",
    "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance",
    "arr_delay",
    "cancelled", "diverted",
]

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("FLIGHTPULSE_DB_HOST", "localhost"),
        port=int(os.environ.get("FLIGHTPULSE_DB_PORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )

def load_window(start_date: str, end_date: str) -> pd.DataFrame:
    cols = ", ".join(_SELECT_COLS)
    sql = (
        f"SELECT {cols} FROM flights "
        f"WHERE fl_date BETWEEN %s AND %s"
    )
    conn = get_conn()
    try:
        df = pd.read_sql(sql, conn, params=(start_date, end_date),
                         parse_dates=["fl_date"])
    finally:
        conn.close()
    return df

if __name__ == "__main__":
    d = load_window("2020-03-01", "2020-03-31")
    print(d.shape)
    print(d.head())


