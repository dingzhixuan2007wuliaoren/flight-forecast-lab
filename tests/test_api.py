from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flight_forecaster.api import app, get_service
from flight_forecaster.schemas import ComparisonResponse


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
    assert 'id="strict-mode-notice"' in dashboard
    assert 'id="timetable-reference-section"' in dashboard
    assert 'name="distance_km"' not in dashboard
    assert 'name="duration_minutes"' not in dashboard
    assert 'name="quote_time"' not in dashboard
    assert 'name="weather_severity_forecast"' not in dashboard
    assert 'name="origin_congestion_index"' not in dashboard
    assert 'name="departure_date"' in dashboard
    offer_page = client.get("/details/offer").text
    assert 'id="price-curve"' in offer_page
    assert 'id="curve-chart"' in offer_page
    assert "historical_prices_available" in offer_page
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


def test_date_only_before_origin_today_returns_422(
    monkeypatch,
    trained_model_dir: Path,
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    get_service.cache_clear()
    response = TestClient(app).post(
        "/v1/compare",
        json={
            "origin": "YYZ",
            "destination": "LHR",
            "departure_date": (
                datetime.now(ZoneInfo("America/Toronto")).date() - timedelta(days=1)
            ).isoformat(),
        },
    )
    assert response.status_code == 422
    assert "before today" in response.json()["detail"]


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


def test_strict_comparison_returns_structured_empty_result_without_fare_provider(
    monkeypatch, trained_model_dir: Path
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    get_service.cache_clear()
    client = TestClient(app)
    departure = (datetime.now(UTC) + timedelta(days=45)).date().isoformat()

    response = client.post(
        "/v1/compare",
        json={"origin": "YYZ", "destination": "LHR", "departure_date": departure},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["distance_km"] > 5_000
    assert payload["availability_mode"] == "strict_bookable_only"
    assert payload["result_status"] == "fare_provider_not_configured"
    assert payload["offers"] == []
    assert payload["rankings"] == {
        "direct_first": [],
        "lowest_price": [],
        "student_first": [],
    }
    assert payload["timetable_references"] == []
    assert "Google Flights" in payload["strict_mode_notice"]["zh"]
    assert "booking-token" in payload["strict_mode_notice"]["en"]
    assert payload["fare_search_metadata"]["status"] == "not_configured"
    assert payload["context"]["weather"]["status"] == "proxy"
    assert payload["context"]["operations"]["status"] == "proxy"
    assert payload["context"]["news"]["status"] == "neutral"
    assert payload["context"]["news"]["articles"] == []
    assert payload["departure_timezone"] == "America/Toronto"
    assert payload["departure_date"] == departure
    assert payload["departure_time_basis"] == "origin_local_noon_model_reference"
    assert ComparisonResponse.model_validate(payload).fare_search_metadata is not None
    missing_metadata = dict(payload)
    missing_metadata["fare_search_metadata"] = None
    with pytest.raises(ValidationError, match="requires fare-search metadata"):
        ComparisonResponse.model_validate(missing_metadata)
    mismatched_status = dict(payload)
    mismatched_status["result_status"] = "fare_provider_rate_limited"
    with pytest.raises(ValidationError, match="must agree"):
        ComparisonResponse.model_validate(mismatched_status)
    missing = client.post(
        "/v1/offer-detail",
        json={
            "origin": "YYZ",
            "destination": "LHR",
            "departure_date": departure,
            "offer_id": "off_000000000000000000000000",
        },
    )
    assert missing.status_code == 404
