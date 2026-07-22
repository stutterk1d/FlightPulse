import os
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = Path("data/raw/flights_sample_3m.csv")

COLUMNS = [
    "fl_date", "airline", "airline_dot", "airline_code", "dot_code",
    "fl_number", "origin", "origin_city", "dest", "dest_city",
    "crs_dep_time", "dep_time", "dep_delay", "taxi_out", "wheels_off",
    "wheels_on", "taxi_in", "crs_arr_time", "arr_time", "arr_delay",
    "cancelled", "cancellation_code", "diverted", "crs_elapsed_time",
    "elapsed_time", "air_time", "distance", "delay_due_carrier",
    "delay_due_weather", "delay_due_nas", "delay_due_security",
    "delay_due_late_aircraft",
]

def get_conn():
    return psycopg2.connect(
        host="localhost",
        port=5433,  # host-mapped port for flightpulse-db
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH.resolve()}")

    col_list = ", ".join(COLUMNS)
    copy_sql = (
        f"COPY flights ({col_list}) "
        f"FROM STDIN WITH (FORMAT csv, HEADER true)"
    )

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM flights;")
            existing = cur.fetchone()[0]
            if existing > 0:
                raise RuntimeError(
                    f"flights already has {existing:,} rows. "
                    f"Truncate first if you want a clean reload."
                )

            print(f"Loading {CSV_PATH} ...")
            t0 = time.time()
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                cur.copy_expert(copy_sql, f)
            conn.commit()
            dt = time.time() - t0

            cur.execute("SELECT count(*) FROM flights;")
            n = cur.fetchone()[0]
            print(f"Done: {n:,} rows in {dt:.1f}s ({n/dt:,.0f} rows/s)")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
