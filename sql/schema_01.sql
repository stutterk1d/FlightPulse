CREATE TABLE IF NOT EXISTS flights (
    id BIGSERIAL PRIMARY KEY,
    fl_date DATE NOT NULL,
    airline TEXT, airline_dot TEXT, airline_code TEXT,
    dot_code INT, fl_number INT,
    origin TEXT, origin_city TEXT, dest TEXT, dest_city TEXT,
    crs_dep_time SMALLINT, dep_time REAL, dep_delay REAL,
    taxi_out REAL, wheels_off REAL, wheels_on REAL, taxi_in REAL,
    crs_arr_time SMALLINT, arr_time REAL, arr_delay REAL,
    cancelled REAL, cancellation_code TEXT, diverted REAL,
    crs_elapsed_time REAL, elapsed_time REAL, air_time REAL, distance REAL,
    delay_due_carrier REAL, delay_due_weather REAL, delay_due_nas REAL,
    delay_due_security REAL, delay_due_late_aircraft REAL
);
CREATE INDEX IF NOT EXISTS idx_flights_fl_date ON flights (fl_date);

CREATE TABLE IF NOT EXISTS jobs (
    job_id BIGSERIAL PRIMARY KEY,
    dag_run_id TEXT, task TEXT, decision TEXT,
    window_start DATE, window_end DATE,
    n_rows INT, drift_share REAL, drift_detected BOOLEAN,
    auc REAL, pr_auc REAL, model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prediction_log (
    pred_id BIGSERIAL PRIMARY KEY,
    request_ts TIMESTAMPTZ DEFAULT now(),
    model_version TEXT, features JSONB, proba REAL, prediction INT
);