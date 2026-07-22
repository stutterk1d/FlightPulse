from src.flightpulse.features import FEATURE_COLS, build_features


def test_serving_and_training_feature_columns_match():
    from src.flightpulse.serve_features import FEATURE_COLS as SERVE_COLS
    assert list(SERVE_COLS) == list(FEATURE_COLS), (
        "serve_features.FEATURE_COLS drifted from features.FEATURE_COLS"
    )


def test_training_dtypes_are_narrow(raw_df):
    feats = build_features(raw_df)
    assert feats["is_weekend"].dtype.name in ("int8", "int32")
    assert feats["is_holiday"].dtype.name in ("int8", "int32")
    assert feats["distance"].dtype.name == "float32"
    assert feats["crs_elapsed_time"].dtype.name == "float32"


def test_serving_builder_casts_to_int32_not_int64(monkeypatch):
    from src.flightpulse import serve_features

    monkeypatch.setattr(serve_features, "lookup_congestion", lambda *a, **k: 7)

    df = serve_features.build_serving_features(
        fl_date="2023-02-15", airline="WN", origin="LAX", dest="JFK",
        crs_dep_time=1430, crs_elapsed_time=330.0, distance=2475.0,
    )

    assert df["is_weekend"].dtype.name == "int32"
    assert df["is_holiday"].dtype.name == "int32"
    assert df["dep_minutes"].dtype.name == "int32"
    assert df["congestion"].dtype.name == "int32"
    assert df["distance"].dtype.name == "float32"
    assert df["crs_elapsed_time"].dtype.name == "float32"


def test_serving_builder_produces_one_row_with_expected_columns(monkeypatch):
    from src.flightpulse import serve_features
    monkeypatch.setattr(serve_features, "lookup_congestion", lambda *a, **k: 7)
    df = serve_features.build_serving_features(
        fl_date="2023-07-04", airline="AA", origin="ORD", dest="DEN",
        crs_dep_time=830, crs_elapsed_time=150.0, distance=888.0,
    )
    assert len(df) == 1
    assert list(df.columns) == list(serve_features.FEATURE_COLS)


def test_hhmm_conversion_handles_2400(monkeypatch):
    from src.flightpulse.serve_features import _hhmm_to_minutes
    assert _hhmm_to_minutes(2400) == 0
    assert _hhmm_to_minutes(1430) == 870
    assert _hhmm_to_minutes(5) == 5


def test_dag_file_parses():
    import ast
    from pathlib import Path
    for name in ("flightpulse_dag.py", "flightpulse_backfill_dag.py"):
        p = Path("dags") / name
        if p.exists():
            ast.parse(p.read_text(encoding="utf-8"))
