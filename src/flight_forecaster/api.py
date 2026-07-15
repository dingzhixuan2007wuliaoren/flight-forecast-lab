from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from flight_forecaster.route_info import RouteLookupError
from flight_forecaster.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    ContextDetailRequest,
    NewsDetailResponse,
    OfferDetailRequest,
    OfferDetailResponse,
    OnTimePrediction,
    OnTimeRequest,
    PricePrediction,
    PriceRequest,
    WeatherDetailResponse,
)
from flight_forecaster.service import OfferNotFoundError, PredictionService
from flight_forecaster.training import ARTIFACT_FILENAME

app = FastAPI(
    title="Flight Forecast Lab",
    version="0.2.0",
    description=(
        "Bilingual global airline/cabin model comparisons with automatic weather, "
        "airport-operations, and current-news context."
    ),
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


@app.get("/details/weather", include_in_schema=False)
def weather_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "weather.html")


@app.get("/details/news", include_in_schema=False)
def news_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "news.html")


@app.get("/details/offer", include_in_schema=False)
def offer_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "offer.html")


@app.get("/health")
def health() -> dict[str, str | bool]:
    artifact_exists = (model_dir() / ARTIFACT_FILENAME).exists()
    service: PredictionService | None = None
    if artifact_exists:
        try:
            service = get_service()
        except (FileNotFoundError, ValueError):
            return {
                "status": "model_not_ready",
                "model_ready": False,
                "fare_provider_configured": False,
                "fare_provider_environment": "disabled",
            }
    return {
        "status": "ok" if artifact_exists else "model_not_trained",
        "model_ready": artifact_exists,
        "fare_provider_configured": bool(
            service is not None and service.flight_offer_provider.configured
        ),
        "fare_provider_environment": (
            service.flight_offer_provider.environment
            if service is not None
            else "disabled"
        ),
    }


@app.get("/v1/model-info")
def model_info() -> dict:
    return _service_or_503().model_info()


@app.post("/v1/predict/price", response_model=PricePrediction)
def predict_price(request: PriceRequest) -> PricePrediction:
    try:
        return _service_or_503().predict_price(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/predict/on-time", response_model=OnTimePrediction)
def predict_ontime(request: OnTimeRequest) -> OnTimePrediction:
    try:
        return _service_or_503().predict_ontime(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/compare", response_model=ComparisonResponse)
def compare_flights(request: ComparisonRequest) -> ComparisonResponse:
    try:
        return _service_or_503().compare(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/offer-detail", response_model=OfferDetailResponse)
def offer_detail(request: OfferDetailRequest) -> OfferDetailResponse:
    try:
        return _service_or_503().offer_detail(request)
    except OfferNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/context/weather-detail", response_model=WeatherDetailResponse)
def weather_detail(request: ContextDetailRequest) -> WeatherDetailResponse:
    try:
        return _service_or_503().weather_detail(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/context/news-detail", response_model=NewsDetailResponse)
def news_detail(request: ContextDetailRequest) -> NewsDetailResponse:
    try:
        return _service_or_503().news_detail(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
