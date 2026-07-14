from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from flight_forecaster.api import app, get_service


def test_health_and_predictions(monkeypatch, trained_model_dir: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    get_service.cache_clear()
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True
    dashboard = client.get("/").text
    assert "Flight Forecast Lab" in dashboard
    assert 'name="distance_km"' not in dashboard
    assert 'name="duration_minutes"' not in dashboard
    assert 'name="quote_time"' not in dashboard
    assert 'name="weather_severity_forecast"' not in dashboard
    assert 'name="origin_congestion_index"' not in dashboard
    assert 'name="departure_time"' in dashboard
    assert "fare_estimation" in client.get("/v1/model-info").json()["available_tasks"]
    assert (
        "global_airline_cabin_comparison"
        in client.get("/v1/model-info").json()["available_tasks"]
    )

    future_departure = (datetime.now(UTC) + timedelta(days=60)).isoformat()

    response = client.post(
        "/v1/predict/price",
        json={
            "origin": "JFK",
            "destination": "LAX",
            "airline": "DL",
            "cabin": "economy",
            "stops": 0,
            "departure_time": future_departure,
        },
    )
    assert response.status_code == 200
    assert response.json()["estimated_price_usd"] > 0
    assert response.json()["distance_km"] == 3983

    ontime = client.post(
        "/v1/predict/on-time",
        json={
            "origin": "JFK",
            "destination": "LAX",
            "airline": "DL",
            "scheduled_departure": future_departure,
        },
    )
    assert ontime.status_code == 200
    assert 0 <= ontime.json()["on_time_probability"] <= 1


def test_invalid_request_is_rejected(monkeypatch, trained_model_dir: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    get_service.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/v1/predict/on-time",
        json={
            "origin": "JFK",
            "destination": "JFK",
            "airline": "DL",
            "scheduled_departure": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
        },
    )
    assert response.status_code == 422


def test_unsupported_airport_is_rejected(monkeypatch, trained_model_dir: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    get_service.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/v1/predict/on-time",
        json={
            "origin": "ZZZ",
            "destination": "JFK",
            "airline": "DL",
            "scheduled_departure": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
        },
    )
    assert response.status_code == 422
    assert "暂不支持机场" in response.json()["detail"]


def test_global_comparison_has_all_rankings_and_labelled_fallbacks(
    monkeypatch, trained_model_dir: Path
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    get_service.cache_clear()
    client = TestClient(app)
    departure = (datetime.now(UTC) + timedelta(days=45)).isoformat()

    response = client.post(
        "/v1/compare",
        json={"origin": "YYZ", "destination": "LHR", "departure_time": departure},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["distance_km"] > 5_000
    assert len({offer["airline_code"] for offer in payload["offers"]}) == 60
    offer_ids = {offer["id"] for offer in payload["offers"]}
    assert set(payload["rankings"]["direct_first"]) == offer_ids
    assert set(payload["rankings"]["lowest_price"]) == offer_ids
    assert set(payload["rankings"]["student_first"]) == offer_ids
    assert payload["context"]["weather"]["status"] == "proxy"
    assert payload["context"]["operations"]["status"] == "proxy"
    assert payload["context"]["news"]["status"] == "neutral"
    assert payload["context"]["news"]["articles"] == []
    assert any(offer["student_status"] == "program_available" for offer in payload["offers"])
    assert all(offer["route_status"] == "model_scenario" for offer in payload["offers"])
    assert all(offer["stops"] == 1 for offer in payload["offers"])
    assert all(offer["cabin_status"] == "catalog_scenario" for offer in payload["offers"])
    assert all(
        offer["duration_minutes"] == payload["duration_minutes"] + 90
        for offer in payload["offers"]
    )
    assert all(
        offer["punctuality_basis"] == "two_leg_independence_scenario"
        for offer in payload["offers"]
    )
    assert payload["departure_timezone"] == "America/Toronto"
