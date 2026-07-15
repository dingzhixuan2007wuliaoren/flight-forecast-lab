from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

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
    weekday = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")[
        selected_date.weekday()
    ]
    for flight, departure_clock in departures.items():
        departure = datetime.combine(selected_date, departure_clock, tzinfo=origin_zone)
        arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(
            destination_zone
        )
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


def test_date_only_compare_expands_every_provider_flight_per_catalog_cabin(
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
        arrival = (departure.astimezone(UTC) + timedelta(minutes=420)).astimezone(
            destination_zone
        )
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

    ac_offers = [offer for offer in result.offers if offer.airline_code == "AC"]
    assert len(ac_offers) == 6  # two flights x AC's three catalog cabin scenarios
    assert len({offer.id for offer in ac_offers}) == 6
    assert {offer.flight_number for offer in ac_offers} == {"AC801", "AC802"}
    assert {offer.scheduled_departure_local.hour for offer in ac_offers} == {9, 20}
    assert all(offer.cabin_status == "catalog_scenario" for offer in ac_offers)
    assert all(offer.routing_status == "provider_direct" for offer in ac_offers)
    assert result.departure_date == selected_date
    assert result.departure_time.hour == 12
    assert result.departure_time_basis == "origin_local_noon_model_reference"
    assert result.schedule_sample_truncated is True
    assert result.schedule_sample_limit == AIRLABS_FREE_SAMPLE_LIMIT
    assert "最多返回 50 行" in result.warnings.zh
    assert "at most 50 rows" in result.warnings.en

    selected = next(offer for offer in ac_offers if offer.flight_number == "AC801")
    detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
            offer_id=selected.id,
        )
    )
    assert detail.schedule_status == "recurring_timetable_projection"
    assert detail.itinerary.time_basis == "provider_schedule"
    assert detail.itinerary.legs[0].flight_number == "AC801"
    assert detail.itinerary.legs[0].departure_terminal is None
    assert detail.itinerary.legs[0].arrival_terminal is None
    assert detail.itinerary.legs[0].aircraft_icao is None
    assert detail.itinerary.legs[0].departure_utc is not None
    assert detail.schedule_sample_truncated is True
    assert detail.schedule_sample_limit == AIRLABS_FREE_SAMPLE_LIMIT
    assert "最多返回 50 行" in detail.notice.zh
    assert "at most 50 rows" in detail.notice.en


def test_no_key_fallback_has_no_fake_flight_number_or_clock_time_and_model_hub_legs(
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
    assert comparison.offers
    assert all(offer.schedule_status == "model_scenario" for offer in comparison.offers)
    assert all(offer.flight_number is None for offer in comparison.offers)
    assert all(offer.scheduled_departure_local is None for offer in comparison.offers)
    assert all(offer.scheduled_arrival_local is None for offer in comparison.offers)

    selected = next(offer for offer in comparison.offers if offer.airline_code == "AA")
    detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=selected_date,
            offer_id=selected.id,
        )
    )
    assert detail.itinerary.kind == "one_stop"
    assert detail.itinerary.layover_airport == "DFW"
    assert detail.itinerary.layover_minutes == 90
    assert len(detail.itinerary.legs) == 2
    leg_minutes = sum(leg.duration_minutes for leg in detail.itinerary.legs)
    assert leg_minutes + 90 == selected.duration_minutes
    assert (
        detail.itinerary.legs[0].duration_minutes
        == service._route("YYZ", "DFW").duration_minutes
    )
    assert (
        detail.itinerary.legs[1].duration_minutes
        == service._route("DFW", "LHR").duration_minutes
    )
    assert selected.duration_minutes > comparison.duration_minutes + 90
    assert all(leg.flight_number is None for leg in detail.itinerary.legs)
    assert all(leg.departure_local is None for leg in detail.itinerary.legs)
    assert detail.fallback_reason is not None

    with pytest.raises(OfferNotFoundError, match="does not exist"):
        service.offer_detail(
            OfferDetailRequest(
                origin="YYZ",
                destination="LHR",
                departure_date=selected_date,
                offer_id="off_000000000000000000000000",
            )
        )


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


def test_endpoint_hub_is_not_replaced_with_an_unrelated_airline_hub(
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

    ek_offers = [offer for offer in comparison.offers if offer.airline_code == "EK"]
    assert ek_offers
    assert all(offer.route_status == "model_scenario" for offer in ek_offers)
    assert all(offer.stops is None for offer in ek_offers)
    assert all(offer.routing_status == "model_route_unresolved" for offer in ek_offers)
    assert all(offer.duration_minutes == comparison.duration_minutes for offer in ek_offers)
    assert all(offer.flight_number is None for offer in ek_offers)

    selected = next(offer for offer in ek_offers if offer.cabin == "economy")
    detail = service.offer_detail(
        OfferDetailRequest(
            origin="DXB",
            destination="JFK",
            departure_date=selected_date,
            offer_id=selected.id,
        )
    )
    assert detail.itinerary.kind == "route_unresolved"
    assert detail.itinerary.layover_airport is None
    assert detail.itinerary.legs == []
    assert detail.itinerary.total_distance_km == comparison.distance_km
    assert detail.itinerary.total_duration_minutes == comparison.duration_minutes

    offers_by_id = {offer.id: offer for offer in comparison.offers}
    ranked = [offers_by_id[offer_id] for offer_id in comparison.rankings.direct_first]
    routing_ranks = [service._routing_rank(offer) for offer in ranked]
    assert routing_ranks == sorted(routing_ranks)
    assert ranked[0].routing_status == "provider_direct"
    assert ranked[-1].routing_status == "model_route_unresolved"
