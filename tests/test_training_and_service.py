import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest

from flight_forecaster.context import (
    ContextProvider,
    NewsSignal,
    OperationsSignal,
    PredictionContext,
    WeatherSignal,
)
from flight_forecaster.route_info import AIRPORTS, RouteLookupError
from flight_forecaster.schemas import (
    ComparisonOffer,
    ComparisonRequest,
    OnTimeRequest,
    PriceRequest,
)
from flight_forecaster.service import PredictionService
from flight_forecaster.training import ARTIFACT_FILENAME


def test_training_writes_versioned_artifacts_and_beats_baselines(
    trained_model_dir: Path,
) -> None:
    bundle = joblib.load(trained_model_dir / ARTIFACT_FILENAME)
    assert bundle["artifact_schema_version"] == 3
    assert "ontime_model_without_weather" in bundle
    assert bundle["metrics"]["price"]["mae_usd"] < bundle["metrics"]["price"]["baseline_mae_usd"]
    assert (
        bundle["metrics"]["on_time"]["brier_score"]
        < bundle["metrics"]["on_time"]["baseline_brier_score"]
    )
    assert 0 <= bundle["metrics"]["on_time_without_weather"]["brier_score"] <= 1
    assert bundle["context_priors"]["source"] == "pytest_synthetic_training_average"
    assert len(bundle["context_priors"]["weather_by_month"]) == 12
    assert bundle["context_priors"]["operations_by_origin"]
    json.loads((trained_model_dir / "metrics.json").read_text(encoding="utf-8"))
    json.loads((trained_model_dir / "metadata.json").read_text(encoding="utf-8"))


def test_service_predictions_are_bounded(trained_model_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    service = PredictionService(trained_model_dir)
    price = service.predict_price(
        PriceRequest.model_validate(
            {
                "origin": "JFK",
                "destination": "LAX",
                "airline": "DL",
                "cabin": "economy",
                "stops": 0,
                "departure_time": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
            }
        )
    )
    on_time = service.predict_ontime(
        OnTimeRequest.model_validate(
            {
                "origin": "YYZ",
                "destination": "JFK",
                "airline": "XY",
                "scheduled_departure": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
            }
        )
    )
    assert 0 <= price.interval_80_low_usd <= price.estimated_price_usd
    assert price.interval_80_high_usd >= price.estimated_price_usd
    assert price.distance_km == 3983
    assert price.duration_minutes == 365
    assert 0 <= on_time.on_time_probability <= 1
    assert 500 < on_time.distance_km < 700
    assert abs(on_time.on_time_probability + on_time.disruption_probability - 1) < 0.0011
    assert on_time.weather_feature_status == "ignored"
    assert "excludes the weather feature" in on_time.weather_feature_notice_en


def test_proxy_weather_uses_the_no_weather_model(
    trained_model_dir: Path,
    monkeypatch,
) -> None:
    class _MustNotRun:
        def predict_proba(self, _features):
            raise AssertionError("weather-enhanced model must not run for proxy weather")

    class _NoWeatherModel:
        def __init__(self) -> None:
            self.columns: list[str] = []

        def predict_proba(self, features):
            self.columns = list(features.columns)
            return np.asarray([[0.25, 0.75]])

    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    service = PredictionService(trained_model_dir)
    no_weather_model = _NoWeatherModel()
    service.bundle["ontime_model"] = _MustNotRun()
    service.bundle["ontime_model_without_weather"] = no_weather_model

    prediction = service.predict_ontime(
        OnTimeRequest.model_validate(
            {
                "origin": "YYZ",
                "destination": "JFK",
                "airline": "AC",
                "scheduled_departure": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
            }
        )
    )

    assert prediction.on_time_probability == 0.75
    assert prediction.weather_feature_status == "ignored"
    assert "weather_severity_forecast" not in no_weather_model.columns


def test_forecast_weather_uses_the_weather_enhanced_model(
    trained_model_dir: Path,
    monkeypatch,
) -> None:
    class _WeatherModel:
        def __init__(self) -> None:
            self.columns: list[str] = []

        def predict_proba(self, features):
            self.columns = list(features.columns)
            return np.asarray([[0.4, 0.6]])

    class _MustNotRun:
        def predict_proba(self, _features):
            raise AssertionError("no-weather model must not run for forecast weather")

    now = datetime.now(UTC)
    context = PredictionContext(
        weather=WeatherSignal(0.35, "forecast", "test_forecast", now, "预报", "forecast"),
        operations=OperationsSignal(0.2, "proxy", "test", now, "运行", "operations"),
        news=NewsSignal(0.1, "neutral", "test", now, "新闻", "news"),
        resolved_at=now,
    )
    service = PredictionService(trained_model_dir)
    weather_model = _WeatherModel()
    service.bundle["ontime_model"] = weather_model
    service.bundle["ontime_model_without_weather"] = _MustNotRun()
    monkeypatch.setattr(service, "_context", lambda _route, _departure: context)

    prediction = service.predict_ontime(
        OnTimeRequest.model_validate(
            {
                "origin": "YYZ",
                "destination": "JFK",
                "airline": "AC",
                "scheduled_departure": (now + timedelta(days=1)).isoformat(),
            }
        )
    )

    assert prediction.on_time_probability == 0.6
    assert prediction.weather_feature_status == "used"
    assert "weather_severity_forecast" in weather_model.columns


def test_date_level_weather_is_ignored_for_flights_far_from_reference_hour() -> None:
    reference = datetime(2026, 8, 20, 16, 0, tzinfo=UTC)
    context = PredictionContext(
        weather=WeatherSignal(
            0.35,
            "forecast",
            "test_forecast",
            reference,
            "预报",
            "forecast",
        ),
        operations=OperationsSignal(0.2, "proxy", "test", reference, "运行", "operations"),
        news=NewsSignal(0.1, "neutral", "test", reference, "新闻", "news"),
        resolved_at=reference,
    )

    assert (
        PredictionService._weather_feature_status(
            context,
            target_departure=reference + timedelta(hours=2),
            weather_reference_departure=reference,
        )
        == "used"
    )
    assert (
        PredictionService._weather_feature_status(
            context,
            target_departure=reference + timedelta(hours=3),
            weather_reference_departure=reference,
        )
        == "ignored"
    )


def _student_offer(
    airline: str,
    *,
    price: float = 100.0,
    baggage: str = "unknown",
    student: str = "program_available",
    change: str = "unknown",
    refund: str = "unknown",
) -> ComparisonOffer:
    return ComparisonOffer(
        id=f"{airline}-economy-0",
        airline_code=airline,
        airline_name=airline,
        cabin="economy",
        stops=None,
        duration_minutes=120,
        estimated_price_usd=price,
        interval_80_low_usd=max(0, price - 10),
        interval_80_high_usd=price + 10,
        on_time_probability=0.8,
        risk_level="low",
        baggage_status=baggage,
        student_status=student,
        change_status=change,
        refund_status=refund,
        student_age_limit_zh="测试",
        student_age_limit_en="test",
        student_verification_zh="测试",
        student_verification_en="test",
        route_status="model_scenario",
        routing_status="model_route_unresolved",
        cabin_status="catalog_scenario",
        punctuality_basis="route_only_model",
    )


def test_student_ranking_uses_the_requested_lexicographic_priority() -> None:
    key = PredictionService._student_sort_key

    assert key(_student_offer("ZZ", price=99)) < key(
        _student_offer("TK", price=100, baggage="confirmed_free")
    )
    assert key(_student_offer("ZZ", baggage="confirmed_free")) < key(
        _student_offer("ZZ", student="confirmed_discount")
    )
    assert key(_student_offer("ZZ", student="confirmed_discount")) < key(
        _student_offer("ZZ", change="confirmed_free", refund="confirmed_free")
    )
    assert key(_student_offer("ZZ", change="confirmed_free")) < key(_student_offer("ZZ"))
    assert key(_student_offer("TK")) < key(_student_offer("QR"))
    assert key(_student_offer("LH")) < key(_student_offer("SQ"))


class _ConfirmedRouteProvider(ContextProvider):
    def route_airlines(self, origin: str, destination: str) -> set[str] | None:
        assert (origin, destination) == ("YYZ", "LHR")
        return {"AC", "BA"}


def test_legacy_route_airline_hints_do_not_bypass_strict_bookable_mode(
    trained_model_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    service = PredictionService(
        trained_model_dir,
        context_provider=_ConfirmedRouteProvider(),
    )
    local_departure = (datetime.now() + timedelta(days=60)).replace(
        hour=9,
        minute=0,
        second=0,
        microsecond=0,
    )
    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_time=local_departure,
        )
    )

    assert result.departure_timezone == "America/Toronto"
    assert result.departure_time.hour == 9
    assert result.offers == []
    assert result.timetable_references == []
    assert result.result_status == "fare_provider_not_configured"
    assert result.rankings.model_dump() == {
        "direct_first": [],
        "lowest_price": [],
        "student_first": [],
    }


def test_origin_timezone_rejects_dst_gap_and_overlap() -> None:
    with pytest.raises(RouteLookupError, match="DST gap"):
        PredictionService._departure_at_origin(
            datetime(2027, 3, 14, 2, 30),
            AIRPORTS["YYZ"],
        )
    with pytest.raises(RouteLookupError, match="ambiguous"):
        PredictionService._departure_at_origin(
            datetime(2027, 11, 7, 1, 30),
            AIRPORTS["YYZ"],
        )
