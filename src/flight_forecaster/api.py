from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from flight_forecaster.schemas import (
    OnTimePrediction,
    OnTimeRequest,
    PricePrediction,
    PriceRequest,
)
from flight_forecaster.service import PredictionService
from flight_forecaster.training import ARTIFACT_FILENAME

app = FastAPI(
    title="Flight Forecast Lab",
    version="0.1.0",
    description="Future itinerary fare estimates and flight on-time probabilities.",
)


def model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", "artifacts/demo"))


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService(model_dir())


def _service_or_503() -> PredictionService:
    try:
        return get_service()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str | bool]:
    artifact_exists = (model_dir() / ARTIFACT_FILENAME).exists()
    if artifact_exists:
        try:
            get_service()
        except (FileNotFoundError, ValueError):
            return {"status": "model_not_ready", "model_ready": False}
    return {
        "status": "ok" if artifact_exists else "model_not_trained",
        "model_ready": artifact_exists,
    }


@app.get("/v1/model-info")
def model_info() -> dict:
    return _service_or_503().model_info()


@app.post("/v1/predict/price", response_model=PricePrediction)
def predict_price(request: PriceRequest) -> PricePrediction:
    return _service_or_503().predict_price(request)


@app.post("/v1/predict/on-time", response_model=OnTimePrediction)
def predict_ontime(request: OnTimeRequest) -> OnTimePrediction:
    return _service_or_503().predict_ontime(request)
