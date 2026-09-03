from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flight_forecaster.api import app, get_service


def _request(*, destination: str = "YYZ") -> dict[str, object]:
    check_in = datetime.now(UTC).date() + timedelta(days=60)
    return {
        "destination": destination,
        "check_in": check_in.isoformat(),
        "check_out": (check_in + timedelta(days=3)).isoformat(),
        "adults": 2,
        "property_type": "hotel",
        "hotel_class": 4,
        "rating": 4.4,
        "review_count": 850,
        "distance_from_city_center_km": 2.8,
        "distance_from_airport_km": 22.0,
        "amenity_count": 12,
        "free_cancellation": True,
        "language": "zh-cn",
    }


def test_hotel_prediction_is_local_bounded_and_quota_free(
    monkeypatch: pytest.MonkeyPatch,
    trained_model_dir: Path,
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    get_service.cache_clear()

    response = TestClient(app).post("/v1/predict/hotel-price", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "model_estimate"
    assert body["currency"] == "USD"
    assert body["external_provider_called"] is False
    assert body["provider_quota_consumed"] is False
    assert body["synthetic_demo"] is True
    assert body["destination_coverage"] == "known"
    assert len(body["forecast_points"]) >= 2
    current = body["current_estimate"]
    assert 0 <= current["interval_80_low_usd"] <= current["estimated_nightly_price_usd"]
    assert current["interval_80_high_usd"] >= current["estimated_nightly_price_usd"]
    assert current["estimated_stay_total_usd"] == pytest.approx(
        current["estimated_nightly_price_usd"] * 3,
        abs=0.02,
    )


def test_verified_price_can_anchor_but_never_replace_model_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    trained_model_dir: Path,
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    payload = _request()
    payload["current_nightly_price_anchor_usd"] = 240.0

    response = TestClient(app).post("/v1/predict/hotel-price", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["anchor"]["status"] == "caller_supplied_price_anchor"
    assert body["current_estimate"]["estimated_nightly_price_usd"] == pytest.approx(
        240.0, abs=0.01
    )
    assert any("锚定" in warning for warning in body["warnings"])
    assert body["external_provider_called"] is False


def test_hotel_prediction_marks_unseen_destination_and_rejects_price_like_extras(
    monkeypatch: pytest.MonkeyPatch,
    trained_model_dir: Path,
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    client = TestClient(app)

    unseen = client.post("/v1/predict/hotel-price", json=_request(destination="PVG"))
    assert unseen.status_code == 200
    assert unseen.json()["destination_coverage"] == "unseen"

    invalid = _request()
    invalid["nightly_price_usd"] = 1.0
    rejected = client.post("/v1/predict/hotel-price", json=invalid)
    assert rejected.status_code == 422

    excessive_anchor = _request()
    excessive_anchor["current_nightly_price_anchor_usd"] = 1e308
    rejected_anchor = client.post(
        "/v1/predict/hotel-price",
        json=excessive_anchor,
    )
    assert rejected_anchor.status_code == 422
