import json
import os
from pathlib import Path

import mlflow
import psycopg2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.flightpulse.serve_features import build_serving_features

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL_URI = "models:/flightpulse@champion"
app = FastAPI(title="FlightPulse", version="1.0")

_model = None
_model_version = None


def _conn():
    return psycopg2.connect(
        host=os.environ.get("FLIGHTPULSE_DB_HOST", "localhost"),
        port=int(os.environ.get("FLIGHTPULSE_DB_PORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def _load_champion():
    global _model, _model_version
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    try:
        from mlflow import MlflowClient
        mv = MlflowClient().get_model_version_by_alias("flightpulse", "champion")
        _model_version = mv.version
    except Exception:
        _model_version = "unknown"
    return _model_version


@app.on_event("startup")
def startup():
    try:
        v = _load_champion()
        print(f"[startup] loaded champion v{v}")
    except Exception as e:
        print(f"[startup] could not load champion: {e}")


class FlightRequest(BaseModel):
    fl_date: str = Field(..., examples=["2023-02-15"])
    airline: str = Field(..., examples=["WN"])
    origin: str = Field(..., examples=["LAX"])
    dest: str = Field(..., examples=["JFK"])
    crs_dep_time: int = Field(..., examples=[1430], description="Scheduled departure, hhmm")
    crs_elapsed_time: float = Field(..., examples=[330.0], description="Scheduled minutes")
    distance: float = Field(..., examples=[2475.0], description="Miles")


def _log_prediction(features_json: str, proba: float, prediction: int, version: str):
    sql = (
        "INSERT INTO prediction_log (model_version, features, proba, prediction) "
        "VALUES (%s, %s, %s, %s)"
    )
    try:
        conn = _conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (version, features_json, proba, prediction))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[log_prediction:error] {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "model_uri": MODEL_URI,
    }


@app.post("/reload")
def reload_model():
    try:
        v = _load_champion()
        return {"status": "reloaded", "model_version": v}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"reload failed: {e}")


@app.get("/metrics")
def metrics():
    try:
        conn = _conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*), avg(proba), sum(prediction) FROM prediction_log;")
                n, avg_p, n_pos = cur.fetchone()
        finally:
            conn.close()
        return {
            "predictions_total": int(n or 0),
            "mean_probability": float(avg_p) if avg_p is not None else None,
            "predicted_delayed": int(n_pos or 0),
            "model_version": _model_version,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict(req: FlightRequest, background: BackgroundTasks):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded; POST /reload")
    try:
        X = build_serving_features(
            fl_date=req.fl_date, airline=req.airline, origin=req.origin,
            dest=req.dest, crs_dep_time=req.crs_dep_time,
            crs_elapsed_time=req.crs_elapsed_time, distance=req.distance,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"feature build failed: {e}")

    raw = _model.predict(X)
    try:
        proba = float(_model._model_impl.predict_proba(X)[:, 1][0])
    except Exception:
        proba = float(raw[0])
    prediction = int(proba >= 0.5)

    payload = req.model_dump()
    payload["congestion"] = int(X["congestion"].iloc[0])
    background.add_task(_log_prediction, json.dumps(payload), proba,
                        prediction, str(_model_version))

    return {
        "delay_gt_15_probability": round(proba, 4),
        "prediction": prediction,
        "model_version": _model_version,
        "congestion": int(X["congestion"].iloc[0]),
    }

