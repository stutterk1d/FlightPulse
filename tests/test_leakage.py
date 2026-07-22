from src.flightpulse.features import FEATURE_COLS, LEAKAGE_COLS, build_features


def test_no_leakage_columns_in_feature_frame(raw_df):
    feats = build_features(raw_df)
    present = [c for c in LEAKAGE_COLS if c in feats.columns]
    assert not present, f"leakage columns survived: {present}"


def test_feature_frame_has_exactly_expected_columns(raw_df):
    feats = build_features(raw_df)
    assert set(feats.columns) == set(FEATURE_COLS + ["label"])


def test_arr_delay_is_consumed_not_exposed(raw_df):
    feats = build_features(raw_df)
    assert "arr_delay" not in feats.columns
    assert "label" in feats.columns
    assert set(feats["label"].unique()).issubset({0, 1})


def test_label_matches_15_minute_threshold(raw_df):
    feats = build_features(raw_df)
    expected = (raw_df["arr_delay"] > 15).sum()
    assert feats["label"].sum() == expected


def test_cancelled_and_diverted_rows_dropped(raw_df_with_cancellations):
    feats = build_features(raw_df_with_cancellations)
    assert len(feats) == 170  # 200 - 20 cancelled - 10 diverted
    assert feats["label"].notna().all()
