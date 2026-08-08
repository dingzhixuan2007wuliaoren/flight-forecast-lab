from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flight_forecaster.api import app, get_service
from flight_forecaster.schemas import ComparisonResponse


def _set_optional_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def test_optional_site_basic_auth(monkeypatch) -> None:
    monkeypatch.setenv("SITE_ACCESS_USERNAME", "flight")
    monkeypatch.setenv("SITE_ACCESS_PASSWORD", "test-only-password")
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code in {200, 503}
    denied = client.get("/")
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].startswith("Basic ")
    assert client.get("/", auth=("flight", "wrong-password")).status_code == 401
    assert client.get("/", auth=("flight", "test-only-password")).status_code == 200


def test_readiness_fails_when_model_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing-model"))
    get_service.cache_clear()

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "model artifact is missing"


def test_health_and_predictions(monkeypatch, trained_model_dir: Path) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("RENDER_GIT_BRANCH", raising=False)
    get_service.cache_clear()
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model_ready": True,
        "build_sha": "unknown",
        "branch": "unknown",
    }
    dashboard = client.get("/").text
    assert "Flight Forecast Lab" in dashboard
    assert 'id="strict-mode-notice"' in dashboard
    assert 'id="fare-coverage-notice"' in dashboard
    assert 'id="timetable-reference-section"' in dashboard
    assert 'name="distance_km"' not in dashboard
    assert 'name="duration_minutes"' not in dashboard
    assert 'name="quote_time"' not in dashboard
    assert 'name="weather_severity_forecast"' not in dashboard
    assert 'name="origin_congestion_index"' not in dashboard
    assert 'name="departure_date"' in dashboard
    assert "fare_provider_processing" in dashboard
    assert "fare_provider_error" in dashboard
    assert "报价任务仍在处理中" in dashboard
    assert "Fare provider returned an error" in dashboard
    assert "fareAggregateRecoveryBody" in dashboard
    assert "暂时性失败来源每次最多受控重试一次" in dashboard
    assert "A transiently failing source is retried at most once" in dashboard
    assert "providerRuns.length > 1 && aggregateFailureStatuses[status]" in dashboard
    assert "isProcessingComparison" in dashboard
    assert "本次准点预测已忽略天气变量。" in dashboard
    assert "Weather was omitted from this on-time prediction." in dashboard
    for field in (
        "coverage_scope",
        "eligible_candidate_count",
        "verification_attempted_count",
        "verified_candidate_count",
        "strictly_rejected_candidate_count",
        "provider_failed_candidate_count",
        "quota_skipped_candidate_count",
        "deduplicated_verified_count",
        "coverage_status",
        "quota_limit",
    ):
        assert field in dashboard
    assert "attempts every eligible candidate" in dashboard
    assert "MAX_STRICT_ITINERARY_SEGMENTS = 8" in dashboard
    assert ".slice(0, MAX_STRICT_ITINERARY_SEGMENTS + 1)" in dashboard
    assert "stops >= MAX_STRICT_ITINERARY_SEGMENTS" in dashboard
    for obsolete_claim in (
        "最多 10 次",
        "at most 10 provider requests",
        "four cabin searches plus six",
    ):
        assert obsolete_claim not in dashboard
    offer_page = client.get("/details/offer").text
    assert 'id="price-curve"' in offer_page
    assert "MAX_STRICT_ITINERARY_SEGMENTS=8" in offer_page
    assert "segments.slice(0,MAX_STRICT_ITINERARY_SEGMENTS+1)" in offer_page
    assert "segments.length<=MAX_STRICT_ITINERARY_SEGMENTS" in offer_page
    assert 'id="curve-chart"' in offer_page
    assert "historical_market_context" in offer_page
    assert 'id="historical-data"' in offer_page
    assert "本次准点预测已忽略天气变量。" in offer_page
    assert "Weather was omitted from this on-time prediction." in offer_page
    assert "weather_feature_status" in offer_page
    assert "data.offers.length" in dashboard
    assert "coverageCacheSnapshot" in dashboard
    assert "600000" in offer_page
    assert "150000" not in offer_page
    assert "minimumDaySpacing=72" in offer_page
    assert "points.forEach(function(xPoint,pointIndex)" in offer_page
    assert "Math.min(5,points.length)" not in offer_page
    assert "overflow-x:auto" in offer_page
    for obsolete_claim in (
        "最多 10 次",
        "at most 10 provider requests",
        "four cabin searches plus six",
        "最多验证六个候选",
        "at most six candidates",
    ):
        assert obsolete_claim not in offer_page
    assert "fare_estimation" in client.get("/v1/model-info").json()["available_tasks"]
    assert (
        "global_airline_cabin_comparison" in client.get("/v1/model-info").json()["available_tasks"]
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


@pytest.mark.parametrize(
    ("build_sha", "branch"),
    [
        ("1234abc", "main"),
        ("0123456789abcdef0123456789abcdef01234567", "codex/provider-fix_1.2"),
    ],
)
def test_health_and_ready_expose_only_sanitized_render_build_metadata(
    monkeypatch: pytest.MonkeyPatch,
    trained_model_dir: Path,
    build_sha: str,
    branch: str,
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    monkeypatch.setenv("RENDER_GIT_COMMIT", build_sha)
    monkeypatch.setenv("RENDER_GIT_BRANCH", branch)
    monkeypatch.setenv("UNRELATED_DEPLOY_SECRET", "must-not-appear-in-health-output")
    monkeypatch.setenv(
        "RENDER_EXTERNAL_URL",
        "https://secret.example.invalid/?token=must-not-appear",
    )
    get_service.cache_clear()

    client = TestClient(app)
    health = client.get("/health")
    ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_ready": True,
        "fare_provider_configured": False,
        "fare_provider_environment": "disabled",
        "build_sha": build_sha,
        "branch": branch,
    }
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model_ready": True,
        "build_sha": build_sha,
        "branch": branch,
    }
    serialized = health.text + ready.text
    assert "must-not-appear-in-health-output" not in serialized
    assert "must-not-appear" not in serialized
    assert "RENDER_EXTERNAL_URL" not in serialized
    assert "UNRELATED_DEPLOY_SECRET" not in serialized


@pytest.mark.parametrize(
    ("build_sha", "branch", "expected_sha", "expected_branch"),
    [
        (None, None, "unknown", "unknown"),
        ("abcdef", "main", "unknown", "main"),
        ("a" * 41, "main", "unknown", "main"),
        ("12345gg", "main", "unknown", "main"),
        ("1234abc\nInjected", "main", "unknown", "main"),
        ("1234abc", "feature/<script>", "1234abc", "unknown"),
        ("1234abc", "main\nX-Injected: true", "1234abc", "unknown"),
    ],
)
def test_health_and_ready_replace_missing_or_unsafe_build_metadata_with_unknown(
    monkeypatch: pytest.MonkeyPatch,
    trained_model_dir: Path,
    build_sha: str | None,
    branch: str | None,
    expected_sha: str,
    expected_branch: str,
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    _set_optional_environment(monkeypatch, "RENDER_GIT_COMMIT", build_sha)
    _set_optional_environment(monkeypatch, "RENDER_GIT_BRANCH", branch)
    get_service.cache_clear()

    client = TestClient(app)
    health = client.get("/health")
    ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json()["build_sha"] == expected_sha
    assert health.json()["branch"] == expected_branch
    assert ready.status_code == 200
    assert ready.json()["build_sha"] == expected_sha
    assert ready.json()["branch"] == expected_branch


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
    assert "严格报价来源" in payload["strict_mode_notice"]["zh"]
    assert "secondary booking-verification identifier" in payload["strict_mode_notice"]["en"]
    assert payload["fare_search_metadata"]["status"] == "not_configured"
    assert payload["context"]["weather"]["status"] == "proxy"
    assert payload["context"]["weather_feature_status"] == "ignored"
    assert "忽略天气变量" in payload["context"]["weather_feature_notice_zh"]
    assert "weather feature" in payload["context"]["weather_feature_notice_en"]
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
