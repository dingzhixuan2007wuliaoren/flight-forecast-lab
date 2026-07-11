from pathlib import Path

from fastapi.testclient import TestClient

from flight_forecaster.api import app, get_service


def test_health_and_predictions(monkeypatch, trained_model_dir: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    get_service.cache_clear()
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True
    assert "Flight Forecast Lab" in client.get("/").text
    assert "fare_estimation" in client.get("/v1/model-info").json()["available_tasks"]

    response = client.post(
        "/v1/predict/price",
        json={
            "origin": "JFK",
            "destination": "LAX",
            "airline": "DL",
            "cabin": "economy",
            "stops": 0,
            "duration_minutes": 365,
            "distance_km": 3983,
            "quote_time": "2026-07-15T12:00:00-04:00",
            "departure_time": "2026-09-15T08:00:00-04:00",
        },
    )
    assert response.status_code == 200
    assert response.json()["estimated_price_usd"] > 0

    ontime = client.post(
        "/v1/predict/on-time",
        json={
            "origin": "JFK",
            "destination": "LAX",
            "airline": "DL",
            "distance_km": 3983,
            "scheduled_departure": "2026-09-15T08:00:00-04:00",
            "weather_severity_forecast": 0.2,
            "origin_congestion_index": 0.65,
        },
    )
    assert ontime.status_code == 200
    assert 0 <= ontime.json()["on_time_probability"] <= 1


def test_invalid_request_is_rejected(monkeypatch, trained_model_dir: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    get_service.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/v1/predict/on-time",
        json={
            "origin": "JFK",
            "destination": "JFK",
            "airline": "DL",
            "distance_km": 3983,
            "scheduled_departure": "2026-09-15T08:00:00-04:00",
            "weather_severity_forecast": 2.0,
            "origin_congestion_index": 0.6,
        },
    )
    assert response.status_code == 422
