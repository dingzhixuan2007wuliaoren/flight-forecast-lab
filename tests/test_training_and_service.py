import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pytest

from flight_forecaster.context import ContextProvider
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
    assert bundle["artifact_schema_version"] == 2
    assert bundle["metrics"]["price"]["mae_usd"] < bundle["metrics"]["price"]["baseline_mae_usd"]
    assert (
        bundle["metrics"]["on_time"]["brier_score"]
        < bundle["metrics"]["on_time"]["baseline_brier_score"]
    )
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


def test_confirmed_direct_carriers_precede_connecting_scenarios(
    trained_model_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
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
    assert len({offer.airline_code for offer in result.offers}) == 60
    confirmed = [offer for offer in result.offers if offer.route_status == "provider_confirmed"]
    scenarios = [offer for offer in result.offers if offer.route_status == "model_scenario"]
    assert {offer.airline_code for offer in confirmed} == {"AC", "BA"}
    assert all(offer.stops == 0 for offer in confirmed)
    connecting_scenarios = [offer for offer in scenarios if offer.stops == 1]
    unresolved_scenarios = [offer for offer in scenarios if offer.stops is None]
    assert connecting_scenarios
    assert unresolved_scenarios
    assert all(
        service._model_hub(offer.airline_code, "YYZ", "LHR") is None
        for offer in unresolved_scenarios
    )
    assert all(offer.duration_minutes == result.duration_minutes for offer in confirmed)
    assert all(
        offer.duration_minutes > result.duration_minutes + 90
        for offer in connecting_scenarios
    )
    assert all(
        offer.duration_minutes == result.duration_minutes for offer in unresolved_scenarios
    )
    aa_offer = next(offer for offer in connecting_scenarios if offer.airline_code == "AA")
    assert aa_offer.duration_minutes == (
        service._route("YYZ", "DFW").duration_minutes
        + 90
        + service._route("DFW", "LHR").duration_minutes
    )
    assert all(offer.punctuality_basis == "direct_leg_model" for offer in confirmed)
    assert all(
        offer.punctuality_basis == "two_leg_independence_scenario"
        for offer in connecting_scenarios
    )
    assert all(
        offer.punctuality_basis == "route_only_model"
        for offer in unresolved_scenarios
    )
    offers_by_id = {offer.id: offer for offer in result.offers}
    ranked = [offers_by_id[offer_id] for offer_id in result.rankings.direct_first]
    assert all(offer.route_status == "provider_confirmed" for offer in ranked[: len(confirmed)])


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
