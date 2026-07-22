from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

HIGH_CARD = ["route", "origin", "dest"]
LOW_CARD  = ["airline", "season", "dep_bucket",
             "is_weekend", "is_holiday"]
NUMERIC   = ["distance", "crs_elapsed_time",
             "dep_minutes", "congestion"]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("te", TargetEncoder(random_state=0), HIGH_CARD),
            ("oh", OneHotEncoder(handle_unknown="ignore"), LOW_CARD),
            ("num", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
    )

def time_split(feats: pd.DataFrame, train_end: str = "2022-12-31"):
    raise NotImplementedError

def fit_from_windows(
    train_start="2019-01-01", train_end="2022-12-31",
    test_start="2023-01-01", test_end="2023-08-31",
):
    from src.flightpulse.features import build_features
    from src.flightpulse.queries import load_window

    print(f"[train] {train_start} .. {train_end}")
    train_feats = build_features(load_window(train_start, train_end))
    print(f"  {len(train_feats):,} rows, label rate {train_feats['label'].mean():.4f}")

    print(f"[test]  {test_start} .. {test_end}")
    test_feats = build_features(load_window(test_start, test_end))
    print(f"  {len(test_feats):,} rows, label rate {test_feats['label'].mean():.4f}")

    ytr = train_feats["label"].to_numpy()
    yte = test_feats["label"].to_numpy()
    Xtr_df = train_feats.drop(columns=["label"])
    Xte_df = test_feats.drop(columns=["label"])

    pre = build_preprocessor()
    Xtr = pre.fit_transform(Xtr_df, ytr)
    Xte = pre.transform(Xte_df)

    joblib.dump(pre, PREPROCESSOR_PATH)
    print(f"Persisted preprocessor -> {PREPROCESSOR_PATH}")
    print(f"Xtr shape {Xtr.shape}, Xte shape {Xte.shape}")
    return Xtr, ytr, Xte, yte, pre

if __name__ == "__main__":
    fit_from_windows()


