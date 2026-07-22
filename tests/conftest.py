import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(0)

AIRLINES = ["WN", "AA", "DL", "UA"]
AIRPORTS = ["LAX", "JFK", "ORD", "ATL", "DEN"]


def _make_raw(n: int = 500, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.to_datetime(start) + pd.to_timedelta(RNG.integers(0, 28, n), unit="D")
    origin = RNG.choice(AIRPORTS, n)
    dest = RNG.choice(AIRPORTS, n)
    dest = np.where(dest == origin, np.roll(AIRPORTS, 1)[
        [AIRPORTS.index(o) for o in origin]], dest)

    return pd.DataFrame({
        "fl_date": dates,
        "airline": RNG.choice(AIRLINES, n),
        "airline_code": RNG.choice(AIRLINES, n),
        "origin": origin,
        "dest": dest,
        "crs_dep_time": RNG.choice([600, 830, 1145, 1430, 1900, 2215], n),
        "crs_arr_time": RNG.choice([900, 1130, 1445, 1730, 2200, 2359], n),
        "crs_elapsed_time": RNG.uniform(60, 400, n),
        "distance": RNG.uniform(150, 2800, n),
        "arr_delay": RNG.normal(5, 40, n),
        "cancelled": np.zeros(n),
        "diverted": np.zeros(n),
    })


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return _make_raw()


@pytest.fixture
def raw_df_with_cancellations() -> pd.DataFrame:
    df = _make_raw(200)
    df.loc[:19, "cancelled"] = 1.0
    df.loc[:19, "arr_delay"] = np.nan
    df.loc[20:29, "diverted"] = 1.0
    df.loc[20:29, "arr_delay"] = np.nan
    return df
