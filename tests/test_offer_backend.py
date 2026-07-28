from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from math import log1p
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from flight_forecaster.availability import (
    Cabin,
    ConfirmedFlightOffer,
    FlightOfferSearchResult,
    FlightOfferSegment,
    ProviderDiagnostic,
    RouteCabinMarketHistory,
    RouteCabinMarketPricePoint,
)
from flight_forecaster.context import (
    AIRLABS_FREE_SAMPLE_LIMIT,
    AIRLABS_ROUTES_URL,
    AIRLABS_SCHEDULES_URL,
    ContextProvider,
)
from flight_forecaster.route_info import RouteLookupError
from flight_forecaster.schedules import (
    FlightSchedule,
    ScheduleProvider,
    ScheduleSearchResult,
)
from flight_forecaster.schemas import ComparisonRequest, OfferDetailRequest
from flight_forecaster.service import OfferNotFoundError, PredictionService


class _Response:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class _Client:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert headers["Accept"] == "application/json"
        assert timeout > 0
        self.calls.append((url, params))
        return _Response(self.payloads[url])


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"response": [], "request": {"has_more": True}}, True),
        ({"response": [], "request": {"has_more": "true"}}, True),
        ({"response": [], "request": {"has_more": "false"}}, False),
        ({"response": [{}] * AIRLABS_FREE_SAMPLE_LIMIT}, True),
    ],
)
def test_airlabs_sample_truncation_metadata_is_parsed(
    payload: dict[str, Any],
    expected: bool,
) -> None:
    assert ScheduleProvider._sample_is_truncated(payload) is expected


class _StaticScheduleProvider:
    def __init__(self, result: ScheduleSearchResult) -> None:
        self.result = result

    def search(self, *args: Any, **kwargs: Any) -> ScheduleSearchResult:
        return self.result


class _StaticFareProvider:
    configured = True
    environment = "production"

    def __init__(self, result: FlightOfferSearchResult) -> None:
        self.result = result
        self.force_refresh_values: list[bool] = []

    def search(self, *args: Any, **kwargs: Any) -> FlightOfferSearchResult:
        self.force_refresh_values.append(bool(kwargs.get("force_refresh")))
        return self.result


def _confirmed_offer(
    *,
    selected_date: date,
    verified_at: datetime,
    airline: str = "AC",
    airline_name: str = "Air Canada",
    cabin: Cabin = "economy",
    total: float = 512.34,
    flight_number: str = "801",
) -> ConfirmedFlightOffer:
    return ConfirmedFlightOffer(
        provider_offer_id=f"priced-{airline}-{flight_number}-{cabin}",
        validating_airline_code=airline,
        airline_name=airline_name,
        cabin=cabin,
        total_amount_usd=total,
        base_amount_usd=430.0,
        last_ticketing_date=selected_date,
        number_of_bookable_seats=4,
        seat_count_capped=False,
        verified_at=verified_at,
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        booking_provider=airline_name,
        booking_url=(
            "https://www.google.com/travel/flights/booking/"
            f"{airline.lower()}-{flight_number}-{cabin}"
        ),
        booking_url_kind="direct_get",
        booking_verified=True,
        segments=(
            FlightOfferSegment(
                segment_id="1",
                origin="YYZ",
                destination="LHR",
                departure_at=datetime.combine(selected_date, time(9, 0)),
                arrival_at=datetime.combine(selected_date, time(21, 0)),
                marketing_airline_code=airline,
                operating_airline_code=airline,
                flight_number=flight_number,
                departure_terminal="1",
                arrival_terminal="2",
                aircraft_icao="B789",
                cabin=cabin,
                booking_class="Y",
                fare_basis="YFLEX",
                fare_brand="FLEX",
                checked_bags_quantity=1,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            ),
        ),
        refundable_fare=True,
        no_penalty_fare=True,
        no_restriction_fare=None,
    )


def _multi_segment_confirmed_offer(
    *,
    segment_count: int,
    selected_date: date,
    verified_at: datetime,
) -> ConfirmedFlightOffer:
    airports = (
        ("YYZ", ZoneInfo("America/Toronto")),
        ("YUL", ZoneInfo("America/Toronto")),
        ("JFK", ZoneInfo("America/New_York")),
        ("BOS", ZoneInfo("America/New_York")),
        ("IAD", ZoneInfo("America/New_York")),
        ("ATL", ZoneInfo("America/New_York")),
        ("MIA", ZoneInfo("America/New_York")),
        ("DFW", ZoneInfo("America/Chicago")),
        ("DEN", ZoneInfo("America/Denver")),
        ("LAX", ZoneInfo("America/Los_Angeles")),
    )
    start = datetime.combine(selected_date, time(12), tzinfo=UTC)
    segments: list[FlightOfferSegment] = []
    for index in range(segment_count):
        departure_utc = start + timedelta(hours=index * 4)
        arrival_utc = departure_utc + timedelta(hours=2)
        origin, origin_zone = airports[index]
        destination, destination_zone = airports[index + 1]
        segments.append(
            FlightOfferSegment(
                segment_id=f"strict-{index + 1}",
                origin=origin,
                destination=destination,
                departure_at=departure_utc.astimezone(origin_zone).replace(tzinfo=None),
                arrival_at=arrival_utc.astimezone(destination_zone).replace(tzinfo=None),
                marketing_airline_code="AC",
                operating_airline_code="AC",
                flight_number=str(800 + index),
                departure_terminal=None,
                arrival_terminal=None,
                aircraft_icao=None,
                cabin="economy",
                booking_class="K",
                fare_basis="KFLEX",
                fare_brand="FLEX",
                checked_bags_quantity=1,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            )
        )
    return ConfirmedFlightOffer(
        provider_offer_id=f"priced-multi-{segment_count}",
        validating_airline_code="AC",
        airline_name="Air Canada",
        cabin="economy",
        total_amount_usd=900.0,
        base_amount_usd=800.0,
        last_ticketing_date=selected_date,
        number_of_bookable_seats=4,
        seat_count_capped=False,
        verified_at=verified_at,
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        booking_provider="Air Canada",
        booking_url="https://www.google.com/travel/flights/booking/multi-segment",
        booking_url_kind="direct_get",
        booking_verified=True,
        segments=tuple(segments),
        refundable_fare=True,
        no_penalty_fare=True,
        no_restriction_fare=None,
    )


def test_service_accepts_eight_segments_and_rejects_nine(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    generated_at = datetime(2026, 8, 1, 12, tzinfo=UTC)
    selected_date = date(2026, 8, 10)
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(ScheduleSearchResult((), frozenset())),  # type: ignore[arg-type]
        now_provider=lambda: generated_at,
    )
    accepted = _multi_segment_confirmed_offer(
        segment_count=8,
        selected_date=selected_date,
        verified_at=generated_at,
    )

    segments, distance_km, duration_minutes = service._strict_provider_segments(
        accepted,
        origin="YYZ",
        destination="DEN",
        departure_date=selected_date,
        generated_at=generated_at,
    )

    assert len(segments) == 8
    assert distance_km > 0
    assert duration_minutes == 1_800
    service.flight_offer_provider = _StaticFareProvider(
        _fare_result(accepted, observed_at=generated_at)
    )
    comparison = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="DEN",
            departure_date=selected_date,
        )
    )
    assert len(comparison.offers) == 1
    strict_offer = comparison.offers[0]
    assert strict_offer.stops == 7
    assert len(strict_offer.segments) == 8
    detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="DEN",
            departure_date=selected_date,
            offer_id=strict_offer.id,
        )
    )
    assert detail.itinerary.kind == "multi_stop"
    assert len(detail.itinerary.legs) == 8
    assert len(detail.itinerary.layovers) == 7
    rejected = _multi_segment_confirmed_offer(
        segment_count=9,
        selected_date=selected_date,
        verified_at=generated_at,
    )
    with pytest.raises(RouteLookupError, match="one to eight segments"):
        service._strict_provider_segments(
            rejected,
            origin="YYZ",
            destination="LAX",
            departure_date=selected_date,
            generated_at=generated_at,
        )


def _fare_result(
    offer: ConfirmedFlightOffer,
    *,
    observed_at: datetime,
    historical_market_contexts: tuple[RouteCabinMarketHistory, ...] = (),
) -> FlightOfferSearchResult:
    return FlightOfferSearchResult(
        offers=(offer,),
        status="confirmed_offers",
        observed_at=observed_at,
        environment="production",
        searched_cabins=(offer.cabin,),
        calls_used=2,
        cache_hit=False,
        search_calls_used=1,
        pricing_calls_used=1,
        search_monthly_limit=250,
        pricing_monthly_limit=None,
        search_monthly_used=2,
        pricing_monthly_used=None,
        eligible_candidate_count=1,
        verification_attempted_count=1,
        verified_candidate_count=1,
        coverage_status="complete",
        historical_market_contexts=historical_market_contexts,
    )


def _market_history(
    *,
    departure_date: date,
    observed_at: datetime,
    cabin: Cabin = "economy",
) -> RouteCabinMarketHistory:
    return RouteCabinMarketHistory(
        origin="YYZ",
        destination="LHR",
        departure_date=departure_date,
        cabin=cabin,
        provider_observed_at=observed_at,
        points=(
            RouteCabinMarketPricePoint(
                observed_at=observed_at - timedelta(days=2),
                price_usd=480.0,
            ),
            RouteCabinMarketPricePoint(
                observed_at=observed_at - timedelta(days=1),
                price_usd=500.0,
            ),
        ),
    )


def test_fare_metadata_explains_cached_and_combined_incomplete_coverage() -> None:
    observed_at = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    cached = FlightOfferSearchResult(
        offers=(),
        status="no_results",
        observed_at=observed_at,
        environment="production",
        searched_cabins=("economy",),
        calls_used=0,
        cache_hit=True,
        eligible_candidate_count=1,
        verification_attempted_count=1,
        strictly_rejected_candidate_count=1,
        coverage_status="complete",
    )
    combined = FlightOfferSearchResult(
        offers=(),
        status="provider_error",
        observed_at=observed_at,
        environment="production",
        searched_cabins=("economy",),
        calls_used=2,
        cache_hit=False,
        pricing_calls_used=2,
        eligible_candidate_count=3,
        verification_attempted_count=2,
        provider_failed_candidate_count=2,
        quota_skipped_candidate_count=1,
        coverage_status="quota_and_provider_incomplete",
        quota_limit="hourly",
    )

    cached_metadata = PredictionService._fare_metadata(cached)
    combined_metadata = PredictionService._fare_metadata(combined)

    assert "缓存" in cached_metadata.notice.zh
    assert "original search" in cached_metadata.notice.en
    assert "供应商错误" in combined_metadata.notice.zh
    assert "provider errors" in combined_metadata.notice.en


class _ConfirmedAirlineProvider(ContextProvider):
    def route_airlines(self, origin: str, destination: str) -> set[str] | None:
        assert (origin, destination) == ("DXB", "JFK")
        return {"AA"}


def _next_weekday(start: date, weekday: int) -> date:
    offset = (weekday - start.weekday()) % 7
    return start + timedelta(days=offset or 7)


def _live_row(
    *,
    flight: str,
    departure: datetime,
    arrival: datetime,
) -> dict[str, Any]:
    return {
        "airline_iata": "AC",
        "flight_iata": flight,
        "flight_number": flight[2:],
        "dep_iata": "YYZ",
        "arr_iata": "LHR",
        "dep_time": departure.strftime("%Y-%m-%d %H:%M"),
        "dep_time_utc": departure.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
        "dep_time_ts": int(departure.timestamp()),
        "arr_time": arrival.strftime("%Y-%m-%d %H:%M"),
        "arr_time_utc": arrival.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
        "arr_time_ts": int(arrival.timestamp()),
        "duration": round(
            (arrival.astimezone(UTC) - departure.astimezone(UTC)).total_seconds() / 60
        ),
        "dep_terminal": "1",
        "arr_terminal": "2",
        "aircraft_icao": "B789",
        "status": "scheduled",
    }


def test_live_schedules_keep_distinct_same_day_flights_and_validate_route() -> None:
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    observed = datetime.now(UTC)
    selected_date = observed.astimezone(origin_zone).date() + timedelta(days=1)
    first_departure = datetime.combine(selected_date, time(16, 0), tzinfo=origin_zone)
    second_departure = datetime.combine(selected_date, time(21, 0), tzinfo=origin_zone)
    first_arrival = (first_departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(
        destination_zone
    )
    second_arrival = (second_departure.astimezone(UTC) + timedelta(minutes=415)).astimezone(
        destination_zone
    )
    wrong_route = _live_row(
        flight="AC999",
        departure=first_departure,
        arrival=first_arrival,
    )
    wrong_route["arr_iata"] = "CDG"
    client = _Client(
        {
            AIRLABS_SCHEDULES_URL: {
                "response": [
                    _live_row(
                        flight="AC850",
                        departure=first_departure,
                        arrival=first_arrival,
                    ),
                    _live_row(
                        flight="AC856",
                        departure=second_departure,
                        arrival=second_arrival,
                    ),
                    wrong_route,
                ]
            }
        }
    )
    provider = ScheduleProvider(api_key="free-test", client=client)

    result = provider.search(
        "YYZ",
        "LHR",
        selected_date,
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=observed,
    )

    assert [flight.flight_number for flight in result.schedules] == ["AC850", "AC856"]
    assert all(flight.schedule_status == "live_schedule" for flight in result.schedules)
    assert all(flight.departure_local.tzinfo is not None for flight in result.schedules)
    assert result.route_airlines == {"AC"}
    assert client.calls[0][1]["limit"] == AIRLABS_FREE_SAMPLE_LIMIT
    assert client.calls[0][1]["arr_iata"] == "LHR"


def test_schedule_search_ors_truncation_across_airlabs_endpoints() -> None:
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    observed = datetime(2026, 7, 14, 12, tzinfo=UTC)
    selected_date = observed.astimezone(origin_zone).date()
    provider = ScheduleProvider(
        api_key="free-test",
        client=_Client(
            {
                AIRLABS_SCHEDULES_URL: {
                    "response": [],
                    "request": {"has_more": "false"},
                },
                AIRLABS_ROUTES_URL: {
                    "response": [],
                    "request": {"has_more": "true"},
                },
            }
        ),
    )

    result = provider.search(
        "YYZ",
        "LHR",
        selected_date,
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=observed,
    )

    assert result.sample_truncated is True


def test_live_canceled_past_and_completed_rows_suppress_matching_projections() -> None:
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    selected_date = date(2026, 7, 14)
    observed_local = datetime.combine(selected_date, time(12, 0), tzinfo=origin_zone)
    observed = observed_local.astimezone(UTC)
    departures = {
        "AC850": time(15, 0),  # cancelled, with an intentionally incomplete live row
        "AC851": time(9, 0),  # scheduled but already departed
        "AC852": time(18, 0),  # eligible scheduled flight
        "AC853": time(20, 0),  # undocumented status is retained and displayed
        "AC854": time(21, 0),  # active is not a future selectable schedule
    }
    live_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    statuses = {
        "AC850": "cancelled",
        "AC851": "scheduled",
        "AC852": "scheduled",
        "AC853": "mystery",
        "AC854": "active",
    }
    weekday = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")[selected_date.weekday()]
    for flight, departure_clock in departures.items():
        departure = datetime.combine(selected_date, departure_clock, tzinfo=origin_zone)
        arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(destination_zone)
        live = _live_row(flight=flight, departure=departure, arrival=arrival)
        live["status"] = statuses[flight]
        if flight == "AC850":
            for field in (
                "arr_time",
                "arr_time_utc",
                "arr_time_ts",
                "duration",
            ):
                live.pop(field)
        live_rows.append(live)
        route_rows.append(
            {
                "airline_iata": "AC",
                "flight_iata": flight,
                "flight_number": flight[2:],
                "dep_iata": "YYZ",
                "arr_iata": "LHR",
                "dep_time": departure.strftime("%H:%M"),
                "arr_time": arrival.strftime("%H:%M"),
                "duration": 420,
                "days": [weekday],
            }
        )
    provider = ScheduleProvider(
        api_key="free-test",
        client=_Client(
            {
                AIRLABS_SCHEDULES_URL: {"response": live_rows},
                AIRLABS_ROUTES_URL: {"response": route_rows},
            }
        ),
    )

    result = provider.search(
        "YYZ",
        "LHR",
        selected_date,
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=observed,
    )

    assert [schedule.flight_number for schedule in result.schedules] == ["AC852", "AC853"]
    assert all(schedule.schedule_status == "live_schedule" for schedule in result.schedules)
    assert [schedule.provider_flight_status for schedule in result.schedules] == [
        "scheduled",
        "mystery",
    ]


def test_recurring_timetable_filters_weekday_and_resolves_overnight_arrival() -> None:
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    selected_date = _next_weekday(date.today() + timedelta(days=20), 0)
    departure = datetime.combine(selected_date, time(22, 0), tzinfo=origin_zone)
    arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(destination_zone)
    monday = {
        "airline_iata": "AC",
        "flight_iata": "AC858",
        "flight_number": "858",
        "dep_iata": "YYZ",
        "arr_iata": "LHR",
        "dep_time": "22:00",
        "arr_time": arrival.strftime("%H:%M"),
        "duration": 420,
        "days": ["mon"],
        "dep_terminals": ["1"],
        "arr_terminals": ["2"],
        "aircraft_icao": "B789",
        "updated": "2026-07-01T12:00:00Z",
    }
    tuesday = {**monday, "flight_iata": "AC860", "flight_number": "860", "days": ["tue"]}
    client = _Client({AIRLABS_ROUTES_URL: {"response": [monday, tuesday]}})
    provider = ScheduleProvider(api_key="free-test", client=client)

    result = provider.search(
        "YYZ",
        "LHR",
        selected_date,
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=datetime.now(UTC),
    )

    assert len(result.schedules) == 1
    schedule = result.schedules[0]
    assert schedule.flight_number == "AC858"
    assert schedule.schedule_status == "recurring_timetable_projection"
    assert schedule.departure_local.date() == selected_date
    assert schedule.arrival_utc > schedule.departure_utc
    assert schedule.arrival_local.strftime("%H:%M") == monday["arr_time"]
    assert schedule.departure_terminal is None
    assert schedule.arrival_terminal is None
    assert schedule.aircraft_icao is None


def test_strict_compare_moves_recurring_rows_to_reference_section(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    selected_date = date.today() + timedelta(days=30)
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    departures = [
        datetime.combine(selected_date, time(9, 0), tzinfo=origin_zone),
        datetime.combine(selected_date, time(20, 0), tzinfo=origin_zone),
    ]
    schedules = []
    for index, departure in enumerate(departures, start=1):
        arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(destination_zone)
        schedules.append(
            FlightSchedule(
                airline_code="AC",
                flight_number=f"AC{800 + index}",
                schedule_status="recurring_timetable_projection",
                source="airlabs_routes",
                departure_local=departure,
                arrival_local=arrival,
                departure_utc=departure.astimezone(UTC),
                arrival_utc=arrival.astimezone(UTC),
                duration_minutes=420,
                departure_terminal="1",
                arrival_terminal="2",
                aircraft_icao="B789",
                observed_at=datetime(2026, 7, 1, tzinfo=UTC),
            )
        )
    schedule_provider = _StaticScheduleProvider(
        ScheduleSearchResult(
            tuple(schedules),
            frozenset({"AC"}),
            sample_truncated=True,
        )
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=schedule_provider,  # type: ignore[arg-type]
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
        )
    )

    assert result.offers == []
    assert result.rankings.direct_first == []
    assert result.rankings.lowest_price == []
    assert result.rankings.student_first == []
    assert result.result_status == "fare_provider_not_configured"
    assert result.availability_mode == "strict_bookable_only"
    assert result.fare_search_metadata is not None
    assert result.fare_search_metadata.status == "not_configured"
    assert len(result.timetable_references) == 2
    assert {item.flight_number for item in result.timetable_references} == {
        "AC801",
        "AC802",
    }
    assert all(
        item.schedule_status == "recurring_timetable_projection"
        for item in result.timetable_references
    )
    assert all(item.bookability_status == "unverified" for item in result.timetable_references)
    assert all("不证明" in item.reference_reason.zh for item in result.timetable_references)
    assert result.departure_date == selected_date
    assert result.departure_time.hour == 12
    assert result.departure_time_basis == "origin_local_noon_model_reference"
    assert result.schedule_sample_truncated is True
    assert result.schedule_sample_limit == AIRLABS_FREE_SAMPLE_LIMIT
    assert "最多返回 50 行" in result.warnings.zh
    assert "at most 50 rows" in result.warnings.en

    with pytest.raises(OfferNotFoundError, match="does not exist"):
        service.offer_detail(
            OfferDetailRequest(
                origin="YYZ",
                destination="LHR",
                departure_date=selected_date,
                offer_id="off_000000000000000000000000",
            )
        )


def test_strict_priced_offer_has_detail_and_daily_model_price_curve(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    selected_date = date(2026, 7, 16)
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    departure = datetime.combine(selected_date, time(9, 0), tzinfo=origin_zone)
    arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(destination_zone)
    schedule = FlightSchedule(
        airline_code="AC",
        flight_number="AC801",
        schedule_status="live_schedule",
        source="airlabs_schedules",
        departure_local=departure,
        arrival_local=arrival,
        departure_utc=departure.astimezone(UTC),
        arrival_utc=arrival.astimezone(UTC),
        duration_minutes=420,
        departure_terminal="1",
        arrival_terminal="2",
        aircraft_icao="B789",
        provider_flight_status="scheduled",
        observed_at=generated_at,
    )
    fare_provider = _StaticFareProvider(
        _fare_result(
            _confirmed_offer(selected_date=selected_date, verified_at=generated_at),
            observed_at=generated_at,
            historical_market_contexts=(
                _market_history(
                    departure_date=selected_date,
                    observed_at=generated_at,
                ),
            ),
        )
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(
            ScheduleSearchResult((schedule,), frozenset({"AC"}))
        ),  # type: ignore[arg-type]
        flight_offer_provider=fare_provider,
        now_provider=lambda: generated_at,
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
        )
    )

    assert result.result_status == "verified_offers_found"
    assert len(result.timetable_references) == 1
    assert result.timetable_references[0].flight_number == "AC801"
    assert len(result.offers) == 1
    assert {offer.flight_number for offer in result.offers} == {"AC801"}
    assert {offer.cabin for offer in result.offers} == {"economy"}
    assert all(offer.schedule_status == "priced_offer" for offer in result.offers)
    assert all(offer.bookability_status == "booking_option_verified" for offer in result.offers)
    assert result.offers[0].live_fare is not None
    assert result.offers[0].live_fare.total_amount == 512.34
    assert result.offers[0].live_fare.taxes_included is None
    assert result.offers[0].live_fare.provider_cache_hit is False
    assert result.offers[0].live_fare.provider_cache_age_seconds == 0
    assert result.fare_search_metadata is not None
    assert result.fare_search_metadata.monthly_call_limit == 250
    assert result.fare_search_metadata.monthly_calls_used == 2
    assert result.fare_search_metadata.search_monthly_limit == 250
    assert result.fare_search_metadata.pricing_monthly_limit is None
    assert len(result.historical_market_contexts) == 1
    comparison_history = result.historical_market_contexts[0]
    assert comparison_history.scope == "route_departure_date_cabin_market"
    assert (
        comparison_history.relation_to_offer
        == "market_context_not_selected_offer_history"
    )
    assert comparison_history.cabin == "economy"
    assert [point.price_usd for point in comparison_history.points] == [480.0, 500.0]
    assert "不是所选航班" in comparison_history.notice.zh
    assert "not price history for the selected flight" in comparison_history.notice.en

    selected = next(offer for offer in result.offers if offer.cabin == "economy")
    detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
            offer_id=selected.id,
        )
    )

    assert detail.itinerary.time_basis == "provider_schedule"
    assert detail.itinerary.legs[0].flight_number == "AC801"
    assert detail.itinerary.legs[0].departure_terminal == "1"
    assert detail.itinerary.legs[0].cabin == "economy"
    assert fare_provider.force_refresh_values == [False, False]
    refreshed_detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
            offer_id=selected.id,
            force_refresh=True,
        )
    )
    assert refreshed_detail.offer.id == detail.offer.id
    assert fare_provider.force_refresh_values == [False, False, True]
    curve = detail.price_curve
    assert curve.status == "model_projection"
    assert curve.basis == "verified_fare_anchored_synthetic_trajectory"
    assert curve.calibration_method == "log1p_offset_to_verified_fare"
    assert curve.interval_basis == "synthetic_demo_interval_log1p_shifted"
    assert curve.historical_prices_available is False
    assert curve.anchor_price_usd == 512.34
    assert curve.anchor_verified_at == generated_at
    assert curve.anchor_provider_code == "serpapi_google_flights"
    assert curve.raw_model_start_price_usd == selected.estimated_price_usd
    assert curve.calibration_log1p_offset == pytest.approx(
        log1p(512.34) - log1p(selected.estimated_price_usd)
    )
    assert curve.start_date == date(2026, 7, 14)
    assert curve.end_date == selected_date
    assert [point.quote_date for point in curve.points] == [
        date(2026, 7, 14),
        date(2026, 7, 15),
        date(2026, 7, 16),
    ]
    assert curve.points[0].estimated_price_usd == 512.34
    assert all(
        point.interval_80_low_usd <= point.estimated_price_usd <= point.interval_80_high_usd
        for point in curve.points
    )
    assert all(
        first.quote_time < second.quote_time
        for first, second in zip(curve.points, curve.points[1:], strict=False)
    )
    assert curve.points[-1].quote_time.astimezone(UTC) < departure.astimezone(UTC)
    assert "第一个点精确等于本次已验证实时报价" in curve.notice.zh
    assert "first point exactly equals" in curve.notice.en
    assert detail.historical_market_context == comparison_history


def test_priced_connection_preserves_each_leg_layover_and_live_fare_ranking(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    selected_date = date(2026, 7, 20)
    connecting = ConfirmedFlightOffer(
        provider_offer_id="priced-lh-connection",
        validating_airline_code="LH",
        airline_name="Lufthansa",
        cabin="business",
        total_amount_usd=480.0,
        base_amount_usd=390.0,
        last_ticketing_date=selected_date,
        number_of_bookable_seats=9,
        seat_count_capped=True,
        verified_at=generated_at,
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        booking_provider="Lufthansa",
        booking_url="https://www.google.com/travel/flights/booking/lh-connection",
        booking_url_kind="direct_get",
        booking_verified=True,
        segments=(
            FlightOfferSegment(
                segment_id="1",
                origin="YYZ",
                destination="FRA",
                departure_at=datetime.combine(selected_date, time(9, 0)),
                arrival_at=datetime.combine(selected_date, time(22, 0)),
                marketing_airline_code="LH",
                operating_airline_code="AC",
                flight_number="471",
                departure_terminal="1",
                arrival_terminal="1",
                aircraft_icao="A333",
                cabin="business",
                booking_class="J",
                fare_basis="JFLEX",
                fare_brand="BUSINESS FLEX",
                checked_bags_quantity=2,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            ),
            FlightOfferSegment(
                segment_id="2",
                origin="FRA",
                destination="LHR",
                departure_at=datetime.combine(selected_date + timedelta(days=1), time(8, 0)),
                arrival_at=datetime.combine(selected_date + timedelta(days=1), time(8, 30)),
                marketing_airline_code="LH",
                operating_airline_code="LH",
                flight_number="900",
                departure_terminal="1",
                arrival_terminal="2",
                aircraft_icao="A320",
                cabin="business",
                booking_class="J",
                fare_basis="JFLEX",
                fare_brand="BUSINESS FLEX",
                checked_bags_quantity=2,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            ),
        ),
        refundable_fare=True,
        no_penalty_fare=False,
        no_restriction_fare=False,
    )
    direct = _confirmed_offer(
        selected_date=selected_date,
        verified_at=generated_at,
        total=820.0,
    )
    fare_result = FlightOfferSearchResult(
        offers=(direct, connecting),
        status="confirmed_offers",
        observed_at=generated_at,
        environment="production",
        searched_cabins=("economy", "business"),
        calls_used=3,
        cache_hit=False,
        search_calls_used=2,
        pricing_calls_used=1,
        search_monthly_limit=250,
        pricing_monthly_limit=None,
        search_monthly_used=3,
        pricing_monthly_used=None,
        eligible_candidate_count=2,
        verification_attempted_count=2,
        verified_candidate_count=2,
        coverage_status="complete",
    )
    fare_provider = _StaticFareProvider(fare_result)
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(ScheduleSearchResult((), frozenset())),  # type: ignore[arg-type]
        flight_offer_provider=fare_provider,
        now_provider=lambda: generated_at,
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
        )
    )

    assert len(result.offers) == 2
    by_id = {offer.id: offer for offer in result.offers}
    assert by_id[result.rankings.direct_first[0]].routing_status == "provider_direct"
    cheapest = by_id[result.rankings.lowest_price[0]]
    assert cheapest.routing_status == "provider_itinerary"
    assert cheapest.live_fare is not None
    assert cheapest.live_fare.total_amount == 480.0
    assert cheapest.stops == 1
    assert [segment.flight_number for segment in cheapest.segments] == [
        "LH471",
        "LH900",
    ]

    detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
            offer_id=cheapest.id,
        )
    )
    assert detail.itinerary.kind == "one_stop"
    assert len(detail.itinerary.legs) == 2
    assert len(detail.itinerary.layovers) == 1
    assert detail.itinerary.layovers[0].airport == "FRA"
    assert detail.itinerary.layovers[0].duration_minutes == 600
    assert detail.itinerary.total_duration_minutes == 1_110
    assert (
        detail.price_curve.points[0].estimated_price_usd
        == cheapest.live_fare.total_amount
    )
    assert (
        detail.price_curve.raw_model_start_price_usd
        == cheapest.estimated_price_usd
    )


def test_unknown_airline_requires_a_priced_cabin_and_is_not_expanded(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    selected_date = date(2026, 7, 15)
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    departure = datetime.combine(selected_date, time(9, 0), tzinfo=origin_zone)
    arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(destination_zone)
    schedule = FlightSchedule(
        airline_code="8M",
        flight_number="8M750",
        schedule_status="live_schedule",
        source="airlabs_schedules",
        departure_local=departure,
        arrival_local=arrival,
        departure_utc=departure.astimezone(UTC),
        arrival_utc=arrival.astimezone(UTC),
        duration_minutes=420,
        provider_flight_status="scheduled",
        observed_at=generated_at,
    )
    unknown_offer = _confirmed_offer(
        selected_date=selected_date,
        verified_at=generated_at,
        airline="8M",
        airline_name="Myanmar Airways International",
        flight_number="750",
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(
            ScheduleSearchResult((schedule,), frozenset({"8M"}))
        ),  # type: ignore[arg-type]
        flight_offer_provider=_StaticFareProvider(
            _fare_result(unknown_offer, observed_at=generated_at)
        ),
        now_provider=lambda: generated_at,
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
        )
    )

    assert len(result.offers) == 1
    assert result.offers[0].airline_code == "8M"
    assert result.offers[0].cabin == "economy"
    assert result.offers[0].cabin_status == "provider_confirmed"
    assert len(result.timetable_references) == 1
    assert result.timetable_references[0].flight_number == "8M750"
    assert "严格报价来源及其二次购票验证标识" in (
        result.timetable_references[0].reference_reason.zh
    )


def test_no_key_strict_mode_returns_empty_instead_of_model_flights(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    service = PredictionService(trained_model_dir, context_provider=ContextProvider())
    selected_date = date.today() + timedelta(days=35)
    comparison = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
        )
    )
    assert comparison.offers == []
    assert comparison.timetable_references == []
    assert comparison.result_status == "fare_provider_not_configured"
    assert comparison.rankings.model_dump() == {
        "direct_first": [],
        "lowest_price": [],
        "student_first": [],
    }

    with pytest.raises(OfferNotFoundError, match="does not exist"):
        service.offer_detail(
            OfferDetailRequest(
                origin="YYZ",
                destination="LHR",
                departure_date=selected_date,
                offer_id="off_000000000000000000000000",
            )
        )


@pytest.mark.parametrize(
    ("provider_status", "result_status", "notice_fragment"),
    [
        ("provider_processing", "fare_provider_processing", "有界轮询"),
        ("provider_error", "fare_provider_error", "终态错误"),
        ("no_results", "no_verified_offer", "没有返回"),
    ],
)
def test_empty_provider_outcomes_remain_distinct_across_service_contract(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_status: str,
    result_status: str,
    notice_fragment: str,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    generated_at = datetime(2026, 7, 15, 12, tzinfo=UTC)
    diagnostic = (
        (
            ProviderDiagnostic(
                observed_at=generated_at,
                stage="search_archive",
                http_status=200,
                exception_type=(
                    "ProviderProcessingError"
                    if provider_status == "provider_processing"
                    else "ProviderSearchError"
                ),
                search_id="pendingsearch01",
            ),
        )
        if provider_status != "no_results"
        else ()
    )
    fare_result = FlightOfferSearchResult(
        offers=(),
        status=provider_status,  # type: ignore[arg-type]
        observed_at=generated_at,
        environment="production",
        searched_cabins=("economy", "premium_economy", "business", "first"),
        calls_used=4,
        cache_hit=False,
        search_calls_used=4,
        search_monthly_limit=250,
        search_monthly_used=4,
        archive_poll_count=(2 if provider_status == "provider_processing" else 0),
        diagnostics=diagnostic,
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(ScheduleSearchResult((), frozenset())),  # type: ignore[arg-type]
        flight_offer_provider=_StaticFareProvider(fare_result),
        now_provider=lambda: generated_at,
    )

    comparison = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=date(2026, 7, 22),
        )
    )

    assert comparison.offers == []
    assert comparison.result_status == result_status
    assert comparison.fare_search_metadata is not None
    assert comparison.fare_search_metadata.status == provider_status
    assert notice_fragment in comparison.fare_search_metadata.notice.zh
    assert "本次没有价格通过严格购票选项验证" in comparison.warnings.zh
    assert "No price passed strict booking-option verification" in comparison.warnings.en
    assert len(comparison.fare_search_metadata.diagnostics) == len(diagnostic)


def test_bounded_empty_subset_is_reported_as_coverage_limited(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    generated_at = datetime(2026, 7, 15, 12, tzinfo=UTC)
    fare_result = FlightOfferSearchResult(
        offers=(),
        status="no_results",
        observed_at=generated_at,
        environment="production",
        searched_cabins=("economy", "premium_economy", "business", "first"),
        calls_used=10,
        cache_hit=False,
        search_calls_used=4,
        pricing_calls_used=6,
        search_monthly_limit=250,
        search_monthly_used=10,
        eligible_candidate_count=8,
        verification_attempted_count=6,
        strictly_rejected_candidate_count=6,
        quota_skipped_candidate_count=2,
        coverage_status="quota_limited",
        quota_limit="provider_specific",
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_StaticScheduleProvider(ScheduleSearchResult((), frozenset())),  # type: ignore[arg-type]
        flight_offer_provider=_StaticFareProvider(fare_result),
        now_provider=lambda: generated_at,
    )

    comparison = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=date(2026, 7, 22),
        )
    )

    assert comparison.offers == []
    assert comparison.result_status == "fare_provider_coverage_limited"
    assert comparison.fare_search_metadata is not None
    assert comparison.fare_search_metadata.status == "no_results"
    assert comparison.fare_search_metadata.coverage_status == "quota_limited"
    assert "leaving 2 unverified" in comparison.fare_search_metadata.notice.en


def test_departure_date_rejects_past_and_more_than_370_days(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    service = PredictionService(trained_model_dir, context_provider=ContextProvider())
    origin_today = datetime.now(ZoneInfo("America/Toronto")).date()
    for invalid_date, message in (
        (origin_today - timedelta(days=1), "before today"),
        (origin_today + timedelta(days=372), "370 days"),
    ):
        with pytest.raises(RouteLookupError, match=message):
            service.compare(
                ComparisonRequest(
                    origin="YYZ",
                    destination="LHR",
                    departure_date=invalid_date,
                )
            )


@pytest.mark.parametrize(
    ("local_clock", "expected_basis", "expected_hour"),
    [
        (time(8, 0), "origin_local_noon_model_reference", 12),
        (time(20, 15, 45), "origin_local_remaining_day_model_reference", 20),
    ],
)
def test_origin_local_today_uses_a_future_safe_model_reference(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_clock: time,
    expected_basis: str,
    expected_hour: int,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    origin_zone = ZoneInfo("America/Toronto")
    today = date(2026, 7, 14)
    local_now = datetime.combine(today, local_clock, tzinfo=origin_zone)
    generated_at = local_now.astimezone(UTC)
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        now_provider=lambda: generated_at,
    )

    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=today,
        )
    )

    assert result.departure_time_basis == expected_basis
    assert result.departure_time.hour == expected_hour
    assert result.departure_time.astimezone(UTC) > generated_at
    if expected_basis == "origin_local_remaining_day_model_reference":
        assert result.departure_time.astimezone(UTC) == generated_at + timedelta(minutes=30)


def test_origin_local_today_near_midnight_has_no_safe_reference(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    origin_zone = ZoneInfo("America/Toronto")
    today = date(2026, 7, 14)
    local_now = datetime.combine(today, time(23, 45), tzinfo=origin_zone)
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        now_provider=lambda: local_now.astimezone(UTC),
    )

    with pytest.raises(RouteLookupError, match="no safe same-day model reference"):
        service.compare(
            ComparisonRequest(
                origin="YYZ",
                destination="LHR",
                departure_date=today,
            )
        )


def test_same_day_safe_reference_uses_utc_timeline_across_spring_forward(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    origin_zone = ZoneInfo("America/Toronto")
    generated_at = datetime(2027, 3, 14, 6, 50, tzinfo=UTC)

    safe_reference = PredictionService._same_day_safe_reference(
        generated_at,
        origin_zone,
    )

    assert safe_reference.isoformat() == "2027-03-14T03:20:00-04:00"
    assert safe_reference.tzinfo is not None
    assert safe_reference.astimezone(UTC) > generated_at

    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        now_provider=lambda: generated_at,
    )
    result = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=date(2027, 3, 14),
        )
    )
    assert result.departure_time_basis == "origin_local_noon_model_reference"
    assert result.departure_time.isoformat() == "2027-03-14T12:00:00-04:00"


def test_legacy_route_airline_hints_do_not_create_strict_offers(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    service = PredictionService(
        trained_model_dir,
        context_provider=_ConfirmedAirlineProvider(),
    )
    selected_date = date.today() + timedelta(days=30)
    comparison = service.compare(
        ComparisonRequest(
            origin="DXB",
            destination="JFK",
            departure_date=selected_date,
        )
    )

    assert comparison.offers == []
    assert comparison.timetable_references == []
    assert comparison.result_status == "fare_provider_not_configured"
