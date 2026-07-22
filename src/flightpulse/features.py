import holidays
import pandas as pd

LEAKAGE_COLS = [
    "dep_time", "dep_delay", "taxi_out", "wheels_off", "wheels_on",
    "taxi_in", "arr_time", "elapsed_time", "air_time",
    "delay_due_carrier", "delay_due_weather", "delay_due_nas",
    "delay_due_security", "delay_due_late_aircraft",
    "cancellation_code", "diverted", "arr_delay",
]

FEATURE_COLS = [
    "route", "origin", "dest",
    "airline", "season", "dep_bucket",
    "is_weekend", "is_holiday",
    "distance", "crs_elapsed_time",
    "dep_minutes", "congestion",
]

_US_HOLIDAYS = holidays.UnitedStates()

def _hhmm_to_minutes(t: pd.Series) -> pd.Series:
    t = t.astype("int32")
    t = t.where(t != 2400, 0)
    return ((t // 100) * 60 + (t % 100)).astype("int16")

def _dep_bucket(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 16, 20, 24]
    labels = ["overnight", "morning", "midday", "evening", "night"]
    return pd.cut(hour, bins=bins, labels=labels)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df["cancelled"].fillna(0) == 0) & (df["diverted"].fillna(0) == 0)
    df = df.loc[keep & df["arr_delay"].notna()].copy()

    df["label"] = (df["arr_delay"] > 15).astype("int8")
    dep_min = _hhmm_to_minutes(df["crs_dep_time"])
    df["dep_minutes"] = dep_min
    dep_hour = (dep_min // 60).astype("int16")
    df["dep_bucket"] = _dep_bucket(dep_hour)

    fd = pd.to_datetime(df["fl_date"])
    df["is_weekend"] = (fd.dt.dayofweek >= 5).astype("int8")
    month = fd.dt.month
    df["season"] = (month % 12 // 3).map(
        {0: "winter", 1: "spring", 2: "summer", 3: "fall"}
    ).astype("category")

    uniq_dates = fd.dt.normalize().drop_duplicates()
    holiday_dates = {d for d in uniq_dates.dt.date if d in _US_HOLIDAYS}
    df["is_holiday"] = fd.dt.date.isin(holiday_dates).astype("int8")

    df["route"] = (df["origin"].astype(str) + "-" + df["dest"].astype(str)).astype("category")
    df["_dep_hour"] = dep_hour
    df["congestion"] = (
        df.groupby(["origin", "fl_date", "_dep_hour"])["route"]
          .transform("size").astype("int32")
    )
    df.drop(columns=["_dep_hour"], inplace=True)

    df["airline"] = df["airline"].astype("category")
    df["origin"] = df["origin"].astype("category")
    df["dest"] = df["dest"].astype("category")

    df["distance"] = df["distance"].astype("float32")
    df["crs_elapsed_time"] = df["crs_elapsed_time"].astype("float32")

    out = df[FEATURE_COLS + ["label"]].copy()
    del df  # free the working frame

    present = [c for c in LEAKAGE_COLS if c in out.columns]
    assert not present, f"LEAKAGE columns present in feature frame: {present}"
    return out

if __name__ == "__main__":
    from src.flightpulse.queries import load_window
    raw = load_window("2020-03-01", "2020-03-31")
    feats = build_features(raw)
    print(feats.shape)
    print(feats.head())
    print("label rate:", round(feats["label"].mean(), 4))
    print("no leakage cols:", not any(c in feats.columns for c in LEAKAGE_COLS))

