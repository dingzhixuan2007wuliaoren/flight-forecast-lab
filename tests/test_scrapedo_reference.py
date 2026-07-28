from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pytest

from flight_forecaster.availability import (
    ConfirmedFlightOffer,
    FlightOfferSearchResult,
    FlightOfferSegment,
)
from flight_forecaster.context import ContextProvider
from flight_forecaster.schedules import ScheduleSearchResult
from flight_forecaster.schemas import ComparisonRequest
from flight_forecaster.scrapedo_reference import (
    SCRAPE_DO_CREDITS_PER_CALL,
    SCRAPE_DO_FLIGHTS_URL,
    SCRAPE_DO_FREE_MONTHLY_CREDITS,
    ScrapeDoGoogleFlightsReferenceProvider,
    ScrapeDoReferenceResult,
)
from flight_forecaster.service import PredictionService


class _Response:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.content = json.dumps(payload).encode("utf-8")

    def json(self) -> Any:
        return self._payload


class _PlannedClient:
    def __init__(self, responses: list[_Response | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> _Response:
        assert timeout > 0
        self.calls.append((url, dict(params)))
        if not self.responses:
            raise AssertionError("unexpected fake-client call")
        planned = self.responses.pop(0)
        if isinstance(planned, Exception):
            raise planned
        return planned


def _flight_payload(selected_date: date) -> dict[str, Any]:
    return {
        "search_parameters": {
            "departure_id": "YYZ",
            "arrival_id": "LHR",
            "outbound_date": selected_date.isoformat(),
            "currency": "USD",
        },
        "best_flights": [
            {
                "type": "One way",
                "price": 421,
                "booking_token": "must-never-be-retained",
                "booking_url": "https://seller.example/secret-path",
                "flights": [
                    {
                        "departure_airport": {
                            "id": "YYZ",
                            "time": f"{selected_date.isoformat()} 09:00",
                        },
                        "arrival_airport": {
                            "id": "LHR",
                            "time": f"{selected_date.isoformat()} 21:00",
                        },
                        "flight_number": "AC 800",
                        "travel_class": "Economy",
                        "duration": 420,
                    }
                ],
            }
        ],
        "other_flights": [],
        "price_insights": {
            "price_level": "typical",
            "typical_price_range": [390, 520],
        },
    }


def _flight_response(
    selected_date: date,
    *,
    request_cost: int = SCRAPE_DO_CREDITS_PER_CALL,
    remaining_credits: int = 990,
) -> _Response:
    return _Response(
        200,
        _flight_payload(selected_date),
        headers={
            "sCrApE.Do-ReQuEsT-CoSt": str(request_cost),
            "Scrape.do-Remaining-Credits": str(remaining_credits),
        },
    )


def _provider(
    tmp_path: Path,
    client: _PlannedClient,
    *,
    limit: int = SCRAPE_DO_FREE_MONTHLY_CREDITS,
) -> ScrapeDoGoogleFlightsReferenceProvider:
    return ScrapeDoGoogleFlightsReferenceProvider(
        "secret-test-token",
        usage_path=tmp_path / "scrapedo.sqlite3",
        monthly_credit_limit=limit,
        client=client,
        cache_ttl_seconds=1_800,
    )


def test_one_economy_reference_call_accounts_ten_credits_and_drops_booking_data(
    tmp_path: Path,
) -> None:
    selected_date = date(2026, 8, 20)
    observed_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _PlannedClient([_flight_response(selected_date)])
    provider = _provider(tmp_path, client)

    result = provider.snapshot(
        "YYZ",
        "LHR",
        selected_date,
        fetched_at=observed_at,
    )

    assert result.status == "available"
    assert result.candidate_count == 1
    assert result.direct_candidate_count == 1
    assert result.lowest_price_usd == 421
    assert result.credits_reserved == SCRAPE_DO_CREDITS_PER_CALL
    assert result.monthly_credits_used == SCRAPE_DO_CREDITS_PER_CALL
    assert result.provider_reported_request_cost == SCRAPE_DO_CREDITS_PER_CALL
    assert result.provider_reported_remaining_credits == 990
    assert [url for url, _params in client.calls] == [SCRAPE_DO_FLIGHTS_URL]
    flight_params = client.calls[0][1]
    assert flight_params["type"] == 2
    assert flight_params["travel_class"] == 1
    serialized = json.dumps(result.__dict__ if hasattr(result, "__dict__") else str(result))
    assert "booking_token" not in serialized
    assert "must-never-be-retained" not in serialized
    assert "seller.example" not in serialized


def test_snapshot_cache_uses_no_http_call_and_no_new_credits(
    tmp_path: Path,
) -> None:
    selected_date = date(2026, 8, 20)
    first_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _PlannedClient([_flight_response(selected_date)])
    provider = _provider(tmp_path, client)

    first = provider.snapshot("YYZ", "LHR", selected_date, fetched_at=first_at)
    cached = provider.snapshot("YYZ", "LHR", selected_date, fetched_at=first_at)

    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert cached.credits_reserved == 0
    assert cached.monthly_credits_used == SCRAPE_DO_CREDITS_PER_CALL
    assert [url for url, _params in client.calls] == [SCRAPE_DO_FLIGHTS_URL]


def test_local_free_thousand_credit_hard_stop_prevents_any_http_call(
    tmp_path: Path,
) -> None:
    observed_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _PlannedClient([])
    provider = _provider(tmp_path, client)
    for _ in range(SCRAPE_DO_FREE_MONTHLY_CREDITS // SCRAPE_DO_CREDITS_PER_CALL):
        assert provider._ledger.reserve(  # noqa: SLF001 - verify durable hard wall
            observed_at,
            hard_limit=SCRAPE_DO_FREE_MONTHLY_CREDITS,
        ).allowed

    result = provider.snapshot(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=observed_at,
    )

    assert result.status == "quota_exhausted"
    assert result.monthly_credits_used == SCRAPE_DO_FREE_MONTHLY_CREDITS
    assert result.exception_type == "LocalFreeQuotaExhausted"
    assert client.calls == []


def test_reported_cost_above_reservation_is_added_and_saturates_local_limit(
    tmp_path: Path,
) -> None:
    selected_date = date(2026, 8, 20)
    observed_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _PlannedClient(
        [_flight_response(selected_date, request_cost=50, remaining_credits=950)]
    )
    provider = _provider(tmp_path, client, limit=30)

    result = provider.snapshot(
        "YYZ",
        "LHR",
        selected_date,
        fetched_at=observed_at,
    )

    assert result.status == "provider_unavailable"
    assert result.credits_reserved == SCRAPE_DO_CREDITS_PER_CALL
    assert result.monthly_credits_used == 30
    assert result.exception_type == "UnsafeProviderQuotaReport"
    assert result.provider_reported_request_cost == 50
    assert result.provider_reported_remaining_credits == 950
    assert provider.credits_used(observed_at) == 30
    assert [url for url, _params in client.calls] == [SCRAPE_DO_FLIGHTS_URL]

    blocked = provider.snapshot(
        "YYZ",
        "LHR",
        selected_date.replace(day=21),
        fetched_at=observed_at,
    )

    assert blocked.status == "quota_exhausted"
    assert blocked.exception_type == "LocalFreeQuotaExhausted"
    assert blocked.monthly_credits_used == 30
    assert [url for url, _params in client.calls] == [SCRAPE_DO_FLIGHTS_URL]


def test_retry_reserves_each_plugin_attempt(
    tmp_path: Path,
) -> None:
    selected_date = date(2026, 8, 20)
    client = _PlannedClient(
        [
            _Response(
                503,
                {"error": "secret backend detail"},
                headers={"Scrape.do-Request-Cost": "10"},
            ),
            _flight_response(selected_date),
        ]
    )
    provider = _provider(tmp_path, client)

    result = provider.snapshot(
        "YYZ",
        "LHR",
        selected_date,
        fetched_at=datetime(2026, 7, 16, 12, tzinfo=UTC),
    )

    assert result.status == "available"
    assert result.credits_reserved == 20
    assert result.monthly_credits_used == 20
    assert [url for url, _params in client.calls] == [
        SCRAPE_DO_FLIGHTS_URL,
        SCRAPE_DO_FLIGHTS_URL,
    ]


def test_two_transport_failures_are_sanitized_and_never_retry_a_third_time(
    tmp_path: Path,
) -> None:
    secret = "token=should-not-escape"
    client = _PlannedClient(
        [
            RuntimeError(secret),
            RuntimeError(secret),
        ]
    )
    provider = _provider(tmp_path, client)

    result = provider.snapshot(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=datetime(2026, 7, 16, 12, tzinfo=UTC),
    )

    assert result.status == "provider_error"
    assert result.exception_type == "TransportError"
    assert result.credits_reserved == 20
    assert result.http_status is None
    assert secret not in repr(result)
    assert [url for url, _params in client.calls] == [
        SCRAPE_DO_FLIGHTS_URL,
        SCRAPE_DO_FLIGHTS_URL,
    ]


class _StaticScheduleProvider:
    def search(self, *args: Any, **kwargs: Any) -> ScheduleSearchResult:
        return ScheduleSearchResult((), frozenset(), "external_context_disabled")


class _StaticFareProvider:
    configured = True
    environment = "production"

    def __init__(self, result: FlightOfferSearchResult) -> None:
        self.result = result

    def search(self, *args: Any, **kwargs: Any) -> FlightOfferSearchResult:
        return self.result


class _ReferenceSpy:
    credential_present = True
    configured = True
    monthly_credit_limit = SCRAPE_DO_FREE_MONTHLY_CREDITS

    def __init__(self) -> None:
        self.calls = 0

    def snapshot(self, *args: Any, **kwargs: Any) -> ScrapeDoReferenceResult:
        self.calls += 1
        return ScrapeDoReferenceResult(
            status="no_results",
            observed_at=kwargs["fetched_at"],
        )


def _empty_fare_result(status: str, observed_at: datetime) -> FlightOfferSearchResult:
    return FlightOfferSearchResult(
        offers=(),
        status=status,  # type: ignore[arg-type]
        observed_at=observed_at,
        environment="production",
        searched_cabins=("economy",),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
    )


def _verified_fare_result(selected_date: date, observed_at: datetime) -> FlightOfferSearchResult:
    offer = ConfirmedFlightOffer(
        provider_offer_id="verified-ac800",
        validating_airline_code="AC",
        airline_name="Air Canada",
        cabin="economy",
        total_amount_usd=500,
        base_amount_usd=440,
        last_ticketing_date=selected_date,
        number_of_bookable_seats=4,
        seat_count_capped=False,
        verified_at=observed_at,
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        booking_provider="Air Canada",
        booking_url="https://www.google.com/travel/flights/booking/ac800",
        booking_url_kind="direct_get",
        booking_verified=True,
        segments=(
            FlightOfferSegment(
                segment_id="1",
                origin="YYZ",
                destination="LHR",
                departure_at=datetime.combine(selected_date, time(9)),
                arrival_at=datetime.combine(selected_date, time(21)),
                marketing_airline_code="AC",
                operating_airline_code="AC",
                flight_number="800",
                departure_terminal="1",
                arrival_terminal="2",
                aircraft_icao="B789",
                cabin="economy",
                booking_class="Y",
                fare_basis="YFLEX",
                fare_brand="Flex",
                checked_bags_quantity=1,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            ),
        ),
        refundable_fare=True,
        no_penalty_fare=True,
        no_restriction_fare=None,
    )
    return FlightOfferSearchResult(
        offers=(offer,),
        status="confirmed_offers",
        observed_at=observed_at,
        environment="production",
        searched_cabins=("economy",),
        calls_used=2,
        cache_hit=False,
        search_calls_used=1,
        pricing_calls_used=1,
        eligible_candidate_count=1,
        verification_attempted_count=1,
        verified_candidate_count=1,
        coverage_status="complete",
    )


@pytest.mark.parametrize("strict_status", ["no_results", "provider_unavailable"])
def test_service_calls_reference_only_when_strict_result_is_empty_or_unavailable(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    strict_status: str,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    observed_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    reference = _ReferenceSpy()
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(),  # type: ignore[arg-type]
        flight_offer_provider=_StaticFareProvider(
            _empty_fare_result(strict_status, observed_at)
        ),
        fare_reference_provider=reference,
        now_provider=lambda: observed_at,
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=date(2026, 8, 20),
        )
    )

    assert reference.calls == 1
    assert result.offers == []
    assert len(result.fare_reference_snapshots) == 1
    snapshot = result.fare_reference_snapshots[0]
    assert snapshot.role == "reference_only"
    assert snapshot.can_supply_strict_offers is False
    assert not hasattr(snapshot, "booking_url")


def test_service_does_not_call_reference_when_verified_offer_exists(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    observed_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    selected_date = date(2026, 8, 20)
    reference = _ReferenceSpy()
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(),  # type: ignore[arg-type]
        flight_offer_provider=_StaticFareProvider(
            _verified_fare_result(selected_date, observed_at)
        ),
        fare_reference_provider=reference,
        now_provider=lambda: observed_at,
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
        )
    )

    assert reference.calls == 0
    assert result.fare_reference_snapshots == []
    assert len(result.offers) == 1
    assert result.offers[0].bookability_status == "booking_option_verified"
