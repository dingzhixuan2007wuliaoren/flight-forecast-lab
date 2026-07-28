from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flight_forecaster.availability import (
    SERPAPI_ACCOUNT_URL,
    SERPAPI_SEARCH_URL,
)
from flight_forecaster.destination_guide import (
    DestinationGuideService,
    _serpapi_transit_option,
)
from flight_forecaster.route_info import Airport
from flight_forecaster.serpapi_transit import (
    SERPAPI_TRANSIT_POLL_DELAYS_SECONDS,
    SERPAPI_TRANSIT_SOURCE_URL,
    SerpApiTransitDirectionsProvider,
    SerpApiTransitLeg,
    SerpApiTransitResult,
    _country_code,
)


def test_google_maps_market_uses_uk_for_iso_gb() -> None:
    assert _country_code("GB") == "UK"
    assert _country_code("ca") == "CA"


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return self.payload


class _Client:
    def __init__(
        self,
        *,
        accounts: list[Any],
        searches: list[Any],
        archives: list[Any] | None = None,
    ) -> None:
        self.accounts = accounts
        self.searches = searches
        self.archives = archives or []
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if url == SERPAPI_ACCOUNT_URL:
            return self._reply(self.accounts)
        if url == SERPAPI_SEARCH_URL:
            return self._reply(self.searches)
        assert url.startswith("https://serpapi.com/searches/")
        return self._reply(self.archives)

    @staticmethod
    def _reply(values: list[Any]) -> _Response:
        if not values:
            raise AssertionError("unexpected extra provider request")
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, _Response) else _Response(value)


NOW = datetime(2026, 7, 26, 15, 0, tzinfo=UTC)
DEPARTURE = datetime(2026, 8, 1, 4, 0, tzinfo=UTC)
ORIGIN = "31.144300,121.808300"
DESTINATION = "31.230400,121.473700"
SEARCH_ID = "safe-search-id-123"


def _account() -> dict[str, Any]:
    return {
        "account_status": "Active",
        "plan_renewal_date": "2026-08-15",
        "searches_per_month": 250,
        "this_month_usage": 20,
        "this_hour_searches": 2,
        "account_rate_limit_per_hour": 50,
        "api_key": "provider-returned-secret",
    }


def _search(*, status: str = "Success", search_id: str = SEARCH_ID) -> dict[str, Any]:
    return {
        "search_metadata": {"id": search_id, "status": status},
        "search_parameters": {
            "engine": "google_maps_directions",
            "start_coords": ORIGIN,
            "end_coords": DESTINATION,
            "travel_mode": 3,
            "time": f"depart_at:{int(DEPARTURE.timestamp())}",
        },
        "directions": [
            {
                "travel_mode": "Transit",
                "duration": 4_200,
                "distance": 31_200,
                "start_time": "12:00 PM",
                "end_time": "1:10 PM",
                "trips": [
                    {
                        "travel_mode": "Transit",
                        "duration": 3_900,
                        "distance": 30_800,
                        "title": "Shanghai Metro Line 2",
                        "headsign": "East Xujing",
                        "start_stop": {
                            "name": "Pudong International Airport",
                            "time": "12:00 PM",
                        },
                        "end_stop": {
                            "name": "People's Square",
                            "time": "1:05 PM",
                        },
                        "stops": [
                            {"name": "Guanglan Road", "time": "12:35 PM"},
                        ],
                        "service_run_by": {"name": "Shanghai Metro"},
                    }
                ],
            }
        ],
    }


def _pending(search_id: str) -> dict[str, Any]:
    return {
        "search_metadata": {"id": search_id, "status": "Processing"},
    }


def _provider(tmp_path: Path, client: _Client) -> SerpApiTransitDirectionsProvider:
    return SerpApiTransitDirectionsProvider(
        "private-test-key",
        usage_path=tmp_path / "serpapi-usage.sqlite3",
        client=client,
        now_provider=lambda: NOW,
        poll_delays_seconds=(0.0,),
        sleep_provider=lambda _: None,
    )


def _run(
    provider: SerpApiTransitDirectionsProvider,
):
    return provider.search(
        origin_latitude=31.1443,
        origin_longitude=121.8083,
        destination_latitude=31.2304,
        destination_longitude=121.4737,
        departure_at=DEPARTURE,
        country_code="CN",
    )


def test_available_transit_keeps_provider_clock_labels_without_inventing_timezone(
    tmp_path: Path,
) -> None:
    client = _Client(accounts=[_account()], searches=[_search()])
    result = _run(_provider(tmp_path, client))

    assert result.status == "available"
    assert result.duration_minutes == 70
    assert result.distance_km == 31.2
    assert result.departure_at is None
    assert result.arrival_at is None
    assert result.departure_label == "12:00 PM"
    assert result.arrival_label == "1:10 PM"
    assert result.legs[0].line_name == "Shanghai Metro Line 2"
    assert result.legs[0].agency_name == "Shanghai Metro"
    assert result.legs[0].intermediate_stops == ("Guanglan Road · 12:35 PM",)
    search_call = next(call for call in client.calls if call["url"] == SERPAPI_SEARCH_URL)
    assert search_call["params"]["travel_mode"] == "3"
    assert search_call["params"]["time"] == f"depart_at:{int(DEPARTURE.timestamp())}"
    assert search_call["params"]["async"] == "true"
    assert sum(SERPAPI_TRANSIT_POLL_DELAYS_SECONDS) >= 10
    ledger_bytes = (tmp_path / "serpapi-usage.sqlite3").read_bytes()
    assert b"private-test-key" not in ledger_bytes
    assert SEARCH_ID.encode() not in ledger_bytes


def test_pending_search_polls_archive_without_reserving_another_call(
    tmp_path: Path,
) -> None:
    client = _Client(
        accounts=[_account()],
        searches=[_pending(SEARCH_ID)],
        archives=[_search()],
    )

    result = _run(_provider(tmp_path, client))

    assert result.status == "available"
    assert sum(call["url"] == SERPAPI_ACCOUNT_URL for call in client.calls) == 1
    assert sum(call["url"] == SERPAPI_SEARCH_URL for call in client.calls) == 1
    assert sum("/searches/" in call["url"] for call in client.calls) == 1


def test_still_processing_gets_only_one_separately_reserved_retry(
    tmp_path: Path,
) -> None:
    first_id = "safe-first-search-id"
    second_id = "safe-second-search-id"
    client = _Client(
        accounts=[_account(), _account()],
        searches=[_pending(first_id), _pending(second_id)],
        archives=[_pending(first_id), _pending(second_id)],
    )

    result = _run(_provider(tmp_path, client))

    assert result.status == "provider_processing"
    assert sum(call["url"] == SERPAPI_ACCOUNT_URL for call in client.calls) == 2
    assert sum(call["url"] == SERPAPI_SEARCH_URL for call in client.calls) == 2
    assert sum("/searches/" in call["url"] for call in client.calls) == 2


def test_no_result_is_classified_and_converts_to_truthful_unavailability(
    tmp_path: Path,
) -> None:
    payload = _search()
    payload["directions"] = []
    result = _run(_provider(tmp_path, _Client(accounts=[_account()], searches=[payload])))

    option = _serpapi_transit_option(
        result,
        requested_departure_at=DEPARTURE,
        departure_time_basis="user_supplied",
        expires_at=datetime(2026, 7, 26, 15, 30, tzinfo=UTC),
    )

    assert result.status == "no_results"
    assert option.status == "unavailable"
    assert option.coverage_status == "no_itinerary"
    assert option.duration_minutes is None
    assert option.data_source == "serpapi_google_maps_directions"
    assert option.source_url == SERPAPI_TRANSIT_SOURCE_URL


def test_not_configured_makes_no_external_request(tmp_path: Path) -> None:
    client = _Client(accounts=[], searches=[])
    provider = SerpApiTransitDirectionsProvider(
        None,
        usage_path=tmp_path / "usage.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = _run(provider)

    assert result.status == "not_configured"
    assert client.calls == []


def test_destination_service_uses_serpapi_only_for_explicit_live_transit() -> None:
    class _TransitousClient:
        def __init__(self) -> None:
            self.calls = 0

        def request_json(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.calls += 1
            return {"itineraries": []}

    class _Fallback:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def configured(self) -> bool:
            return True

        def search(self, **_: Any) -> SerpApiTransitResult:
            self.calls += 1
            return SerpApiTransitResult(
                status="available",
                observed_at=NOW,
                message="Source-backed provider itinerary.",
                duration_minutes=70,
                distance_km=31.2,
                departure_label="12:00 PM",
                arrival_label="1:10 PM",
                transfers=0,
                legs=(
                    SerpApiTransitLeg(
                        mode="SUBWAY",
                        from_name="Pudong International Airport",
                        to_name="People's Square",
                        duration_minutes=65,
                        distance_km=30.8,
                        line_name="Shanghai Metro Line 2",
                        headsign="East Xujing",
                        agency_name="Shanghai Metro",
                        intermediate_stops=("Guanglan Road · 12:35 PM",),
                        departure_label="12:00 PM",
                        arrival_label="1:05 PM",
                    ),
                ),
            )

    client = _TransitousClient()
    fallback = _Fallback()
    service = DestinationGuideService(
        client=client,
        wall_clock=lambda: NOW,
        monotonic_clock=lambda: 0.0,
        sleeper=lambda _: None,
        serpapi_transit_provider=fallback,  # type: ignore[arg-type]
    )
    airport = Airport(
        "PVG",
        "ZSPD",
        "Shanghai Pudong International",
        "large_airport",
        "CN",
        31.1443,
        121.8083,
    )

    default_result = service._transit_option(  # noqa: SLF001
        airport,
        31.2304,
        121.4737,
        transit_departure_at=DEPARTURE,
    )
    first = service._transit_option(  # noqa: SLF001
        airport,
        31.2304,
        121.4737,
        transit_departure_at=DEPARTURE,
        include_live_transit=True,
    )
    cached = service._transit_option(  # noqa: SLF001
        airport,
        31.2304,
        121.4737,
        transit_departure_at=DEPARTURE,
        include_live_transit=True,
    )

    assert default_result.status == "unavailable"
    assert default_result.data_source == "transitous_motis"
    assert first == cached
    assert first.status == "available"
    assert first.data_source == "serpapi_google_maps_directions"
    assert first.departure_time_label == "12:00 PM"
    assert first.legs[0].line_name == "Shanghai Metro Line 2"
    assert client.calls == 2
    assert fallback.calls == 1
