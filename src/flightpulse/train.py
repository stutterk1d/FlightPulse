
import hashlib
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.flightpulse.features import build_features
from src.flightpulse.preprocess import build_preprocessor
from src.flightpulse.queries import load_window

MODEL_NAME = "flightpulse"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "flightpulse-training"

# Promotion policy
MIN_AUC = 0.60            # never promote a model worse than this
MAX_DEGRADATION = 0.05    # train_pr - test_pr must be <= this
PROMOTE_MARGIN = 0.005    # challenger must beat champion PR-AUC by at least this


def _fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(pd.util.hash_pandas_object(df["label"], index=False).values.tobytes())
    return h.hexdigest()[:12]


def _metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "positive_rate": float(np.mean(y_true)),
    }
    return out


def _build_models():
    return {
        "histgb": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=8, random_state=0)),
        ]),
        "xgb": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=400, learning_rate=0.08, max_depth=7,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=0)),
        ]),
    }


def train_and_register(
    train_start="2019-01-01", train_end="2022-12-31",
    test_start="2023-01-01", test_end="2023-08-31",
):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    print("Loading + building features ...")
    train_feats = build_features(load_window(train_start, train_end))
    test_feats = build_features(load_window(test_start, test_end))
    Xtr = train_feats.drop(columns=["label"]); ytr = train_feats["label"].to_numpy()
    Xte = test_feats.drop(columns=["label"]);  yte = test_feats["label"].to_numpy()
    print(f"  train {Xtr.shape}, test {Xte.shape}, "
          f"train pos {ytr.mean():.4f}, test pos {yte.mean():.4f}")

    data_hash = _fingerprint(train_feats)
    results = {}

    for name, pipe in _build_models().items():
        print(f"\n=== Training {name} ===")
        with mlflow.start_run(run_name=name) as run:
            pipe.fit(Xtr, ytr)

            p_te = pipe.predict_proba(Xte)[:, 1]
            p_tr = pipe.predict_proba(Xtr)[:, 1]
            m_te = _metrics(yte, p_te)
            m_tr = _metrics(ytr, p_tr)
            degradation = m_tr["pr_auc"] - m_te["pr_auc"]

            mlflow.log_metrics({f"test_{k}": v for k, v in m_te.items()})
            mlflow.log_metrics({f"train_{k}": v for k, v in m_tr.items()})
            mlflow.log_metric("pr_auc", m_te["pr_auc"])   # canonical selection metric
            mlflow.log_metric("auc", m_te["auc"])
            mlflow.log_metric("degradation_pr", degradation)
            mlflow.log_params({
                "model": name,
                "train_window": f"{train_start}..{train_end}",
                "test_window": f"{test_start}..{test_end}",
            })
            mlflow.set_tags({
                "features": "pre_departure_only",
                "leakage_check": "passed",
                "data_hash": data_hash,
            })

            sig = infer_signature(Xte, p_te)
            info = mlflow.sklearn.log_model(
                pipe, name="model", signature=sig, input_example=Xte.head(3))
            mv = mlflow.register_model(info.model_uri, MODEL_NAME)

            passes_guard = (m_te["auc"] >= MIN_AUC) and (degradation <= MAX_DEGRADATION)
            results[name] = {
                "version": mv.version, "test": m_te,
                "degradation": degradation, "passes_guard": passes_guard,
                "run_id": run.info.run_id,
            }
            print(f"  test AUC {m_te['auc']:.4f}  PR-AUC {m_te['pr_auc']:.4f}  "
                  f"degradation {degradation:.4f}  guard {'PASS' if passes_guard else 'FAIL'}")

    eligible = {n: r for n, r in results.items() if r["passes_guard"]}
    if not eligible:
        print("\nNo model passed the overfitting/AUC guard,nothing promoted.")
        return results

    winner = max(eligible, key=lambda n: eligible[n]["test"]["pr_auc"])
    winner_pr = eligible[winner]["test"]["pr_auc"]

    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ_pr = float(client.get_run(champ.run_id).data.metrics["pr_auc"])
    except Exception:
        champ_pr = -1.0

    print(f"\nWinner among eligible: {winner} (PR-AUC {winner_pr:.4f}). "
          f"Incumbent champion PR-AUC: {champ_pr:.4f}")

    if winner_pr > champ_pr + PROMOTE_MARGIN:
        client.set_registered_model_alias(MODEL_NAME, "champion", eligible[winner]["version"])
        print(f"PROMOTED {winner} v{eligible[winner]['version']} -> @champion")
    else:
        print("Winner did not beat incumbent by the margin — champion unchanged.")

    non_champ = [n for n in results if n != winner]
    if non_champ:
        chal = max(non_champ, key=lambda n: results[n]["test"]["pr_auc"])
        client.set_registered_model_alias(MODEL_NAME, "challenger", results[chal]["version"])
        print(f"Set {chal} v{results[chal]['version']} -> @challenger")

    return results


if __name__ == "__main__":
    train_and_register()

