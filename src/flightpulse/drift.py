from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

NUM_COLS = ["distance", "crs_elapsed_time", "dep_minutes", "congestion"]
CAT_COLS = ["route", "origin", "dest", "airline", "season", "dep_bucket",
            "is_weekend", "is_holiday"]

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # Population Stability Index, <0.1 no shift; 0.1-0.25 moderate; >0.25 significant.
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    e = np.histogram(expected, q)[0] / len(expected)
    a = np.histogram(actual, q)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def ks_drift(expected: np.ndarray, actual: np.ndarray, alpha: float = 0.05) -> dict:
    # Two-sample Kolmogorov-Smirnov. drift=True when p < alpha.
    stat, p = ks_2samp(np.asarray(expected), np.asarray(actual))
    return {"stat": float(stat), "p": float(p), "drift": bool(p < alpha)}

def psi_report(ref: pd.DataFrame, cur: pd.DataFrame, cols=NUM_COLS) -> pd.DataFrame:
    # PSI for each numeric column; a compact interpretable table.
    rows = []
    for c in cols:
        val = psi(ref[c].to_numpy(), cur[c].to_numpy())
        level = "none" if val < 0.1 else ("moderate" if val < 0.25 else "significant")
        rows.append({"column": c, "psi": round(val, 4), "level": level})
    return pd.DataFrame(rows)

def evidently_drift(ref: pd.DataFrame, cur: pd.DataFrame,
                    drift_share: float = 0.5,
                    html_name: str | None = None) -> dict:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    # Sample for tractability
    MAX_N = 15000
    ref_s = ref.sample(min(len(ref), MAX_N), random_state=0)
    cur_s = cur.sample(min(len(cur), MAX_N), random_state=0)

    schema = DataDefinition(numerical_columns=NUM_COLS, categorical_columns=CAT_COLS)
    ref_ds = Dataset.from_pandas(ref_s[NUM_COLS + CAT_COLS], data_definition=schema)
    cur_ds = Dataset.from_pandas(cur_s[NUM_COLS + CAT_COLS], data_definition=schema)

    report = Report([DataDriftPreset(drift_share=drift_share)])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    if html_name:
        snapshot.save_html(str(REPORTS_DIR / html_name))

    d = snapshot.dict()
    share = count = None
    for m in d.get("metrics", []):
        val = m.get("value", {})
        if isinstance(val, dict) and "share" in val and "count" in val:
            share, count = val["share"], val["count"]
            break
    return {
        "dataset_drift": bool(share is not None and share > drift_share),
        "drifted_share": share,
        "drifted_count": count,
        "snapshot": snapshot,
    }

def deepchecks_drift(ref: pd.DataFrame, cur: pd.DataFrame, label_col="label"):
    try:
        from deepchecks.tabular import Dataset as DCDataset
        from deepchecks.tabular.checks import FeatureDrift, LabelDrift
    except Exception as e:
        return {"available": False, "reason": str(e)}

    ref_ds = DCDataset(ref, label=label_col, cat_features=CAT_COLS)
    cur_ds = DCDataset(cur, label=label_col, cat_features=CAT_COLS)
    fd = FeatureDrift().run(train_dataset=ref_ds, test_dataset=cur_ds)
    ld = LabelDrift().run(train_dataset=ref_ds, test_dataset=cur_ds)
    return {"available": True, "feature_drift": fd, "label_drift": ld}

def shift_distance(df: pd.DataFrame, factor: float = 1.3) -> pd.DataFrame:
    d = df.copy()
    d["distance"] = d["distance"] * factor
    return d

def resample_airline_mix(df: pd.DataFrame, boost: str = "WN", frac: float = 0.5) -> pd.DataFrame:
    # Address categorical drift from airline
    extra = df[df["airline"] == boost].sample(frac=frac, replace=True, random_state=0)
    return pd.concat([df, extra], ignore_index=True)

def inject_label_drift(df: pd.DataFrame, target_rate: float = 0.35,
                       label_col: str = "label") -> pd.DataFrame:
    # Push the positive rate up
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    n_pos = int(target_rate / (1 - target_rate) * len(neg))
    pos_s = pos.sample(n_pos, replace=True, random_state=0)
    return pd.concat([neg, pos_s], ignore_index=True)

def null_control(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    return df.sample(frac=1.0, replace=True, random_state=seed)

def psi_detector(ref: pd.DataFrame, cur: pd.DataFrame, col="distance", thresh=0.25) -> dict:
    val = psi(ref[col].to_numpy(), cur[col].to_numpy())
    return {"psi": val, "drift": bool(val > thresh)}

def validate_detector(ref: pd.DataFrame, verbose: bool = True) -> dict:
    results = {}

    ctrl = null_control(ref)
    r_ctrl = psi_detector(ref, ctrl)
    results["null_control"] = r_ctrl
    assert not r_ctrl["drift"], f"FALSE POSITIVE on null control (psi={r_ctrl['psi']:.4f})"

    drifted = shift_distance(ref, factor=1.6)
    r_drift = psi_detector(ref, drifted)
    results["injected_distance_drift"] = r_drift
    assert r_drift["drift"], f"MISSED injected distance drift (psi={r_drift['psi']:.4f})"

    if verbose:
        print(f"[null control]     psi={r_ctrl['psi']:.4f}  drift={r_ctrl['drift']}  (want False)")
        print(f"[injected drift]   psi={r_drift['psi']:.4f}  drift={r_drift['drift']}  (want True)")
        print("PASS: detector is specific (quiet on control) and sensitive (fires on drift)")
    return results

if __name__ == "__main__":
    from src.flightpulse.features import build_features
    from src.flightpulse.queries import load_window

    ref = build_features(load_window("2019-06-01", "2019-06-30"))
    print("PSI self-check (ref vs ref):")
    print(psi_report(ref, ref))
    print()
    validate_detector(ref)



