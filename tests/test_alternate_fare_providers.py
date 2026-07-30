from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
from fastapi.testclient import TestClient

from flight_forecaster import api as api_module
from flight_forecaster.alternate_fare_providers import (
    IGNAV_BOOKING_LINKS_URL,
    IGNAV_ONE_WAY_URL,
    SEARCHAPI_SEARCH_URL,
    FallbackFlightOfferProvider,
    IgnavQuarantineFlightOfferProvider,
    SearchApiFlightOfferProvider,
    _parse_ignav_segments,
    _parse_searchapi_segments,
)
from flight_forecaster.availability import (
    AGGREGATE_PROVIDER_CODE,
    AGGREGATE_PROVIDER_NAME,
    IGNAV_QUARANTINE_PROVIDER_CODE,
    IGNAV_QUARANTINE_PROVIDER_NAME,
    IGNAV_VERIFIED_PROVIDER_CODE,
    IGNAV_VERIFIED_PROVIDER_NAME,
    SEARCHAPI_PROVIDER_CODE,
    SEARCHAPI_PROVIDER_NAME,
    SERPAPI_PROVIDER_CODE,
    SERPAPI_PROVIDER_NAME,
    ConfirmedFlightOffer,
    FlightOfferSearchResult,
    FlightOfferSegment,
    RouteCabinMarketHistory,
    RouteCabinMarketPricePoint,
)
from flight_forecaster.context import ContextProvider
from flight_forecaster.schedules import ScheduleSearchResult
from flight_forecaster.schemas import ComparisonRequest, ComparisonResponse, OfferDetailRequest
from flight_forecaster.service import PredictionService

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)
DEPARTURE_DATE = date(2026, 8, 2)
_TRAVEL_CLASSES = {
    "economy": "economy",
    "premium_economy": "premium_economy",
    "business": "business",
    "first": "first_class",
}


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self) -> Any:
        return self.payload


class _SearchApiClient:
    def __init__(
        self,
        *,
        include_candidate: bool = True,
        candidate_cabins: set[str] | None = None,
        candidates_per_cabin: int = 1,
    ) -> None:
        self.candidate_cabins = (
            set(candidate_cabins)
            if candidate_cabins is not None
            else ({"economy"} if include_candidate else set())
        )
        self.candidates_per_cabin = candidates_per_cabin
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert headers["Authorization"] == "Bearer searchapi-test-key"
        assert timeout > 0
        self.calls.append((url, dict(params)))
        assert url == SEARCHAPI_SEARCH_URL
        cabin = str(params["travel_class"])
        if "booking_token" in params:
            return _Response(_searchapi_booking_payload(cabin, params))
        return _Response(
            _searchapi_search_payload(
                cabin,
                params,
                include_candidate=cabin in self.candidate_cabins,
                candidate_count=self.candidates_per_cabin,
            )
        )


class _ConcurrentSearchApiClient(_SearchApiClient):
    def __init__(self) -> None:
        super().__init__(candidate_cabins=set(_TRAVEL_CLASSES.values()))
        self.search_barrier = Barrier(4, timeout=2)
        self.booking_barrier = Barrier(4, timeout=2)

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        if url == SEARCHAPI_SEARCH_URL:
            barrier = (
                self.booking_barrier
                if "booking_token" in params
                else self.search_barrier
            )
            barrier.wait()
        return super().get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )


class _PartiallyFailingSearchApiClient(_SearchApiClient):
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        if (
            url == SEARCHAPI_SEARCH_URL
            and "booking_token" not in params
            and params.get("travel_class") == "business"
        ):
            self.calls.append((url, dict(params)))
            return _Response({"error": "temporary provider failure"}, 503)
        return super().get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )


class _FailingSearchApiClient(_SearchApiClient):
    def __init__(self, status_code: int) -> None:
        super().__init__(include_candidate=False)
        self.status_code = status_code

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert url == SEARCHAPI_SEARCH_URL
        assert "booking_token" not in params
        assert headers["Authorization"] == "Bearer searchapi-test-key"
        assert timeout > 0
        self.calls.append((url, dict(params)))
        return _Response({"error": "provider failure"}, self.status_code)


class _RejectingBoundedSearchApiClient(_SearchApiClient):
    def __init__(self) -> None:
        super().__init__(
            candidate_cabins=set(_TRAVEL_CLASSES.values()),
            candidates_per_cabin=3,
        )

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        response = super().get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        if url == SEARCHAPI_SEARCH_URL and "booking_token" in params:
            response.payload["booking_options"] = []
            response.content = json.dumps(response.payload).encode("utf-8")
            response.text = response.content.decode("utf-8")
        return response


class _IgnavClient:
    def __init__(self, *, booking_url: str) -> None:
        self.booking_url = booking_url
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert headers["X-Api-Key"] == "ignav-test-key"
        assert timeout > 0
        self.calls.append((url, dict(json)))
        if url == IGNAV_ONE_WAY_URL:
            cabin = str(json["cabin_class"])
            return _Response(
                {
                    "origin": "YYZ",
                    "destination": "LHR",
                    "departure_date": DEPARTURE_DATE.isoformat(),
                    "itineraries": (
                        [_ignav_candidate()] if cabin == "economy" else []
                    ),
                }
            )
        assert url == IGNAV_BOOKING_LINKS_URL
        assert json == {"ignav_id": "ignavcandidate01"}
        return _Response(_ignav_booking_payload(self.booking_url))


class _BoundedConcurrentIgnavClient:
    def __init__(
        self,
        *,
        booking_url: str = "https://www.aircanada.com/booking/test",
    ) -> None:
        self.booking_url = booking_url
        self.search_barrier = Barrier(4)
        self.booking_barrier = Barrier(6)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.candidates: dict[str, tuple[str, float]] = {}
        for cabin_index, cabin in enumerate(_TRAVEL_CLASSES):
            for candidate_index in range(3):
                candidate_id = f"ignav_{cabin}_{candidate_index:02d}"
                self.candidates[candidate_id] = (
                    cabin,
                    500.0 + cabin_index * 100 + candidate_index,
                )

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert headers["X-Api-Key"] == "ignav-test-key"
        assert timeout > 0
        self.calls.append((url, dict(json)))
        if url == IGNAV_ONE_WAY_URL:
            self.search_barrier.wait()
            cabin = str(json["cabin_class"])
            rows = [
                _ignav_candidate(
                    ignav_id=candidate_id,
                    cabin=candidate_cabin,
                    amount=amount,
                )
                for candidate_id, (candidate_cabin, amount) in self.candidates.items()
                if candidate_cabin == cabin
            ]
            return _Response(
                {
                    "origin": "YYZ",
                    "destination": "LHR",
                    "departure_date": DEPARTURE_DATE.isoformat(),
                    "itineraries": rows,
                }
            )
        assert url == IGNAV_BOOKING_LINKS_URL
        self.booking_barrier.wait()
        candidate_id = str(json["ignav_id"])
        cabin, amount = self.candidates[candidate_id]
        return _Response(
            _ignav_booking_payload(
                self.booking_url,
                cabin=cabin,
                amount=amount,
            )
        )


class _NoCallClient:
    def get(self, *_args: Any, **_kwargs: Any) -> _Response:
        raise AssertionError("provider network must not be called")

    def post(self, *_args: Any, **_kwargs: Any) -> _Response:
        raise AssertionError("provider network must not be called")


def _searchapi_parameters(cabin: str) -> dict[str, Any]:
    return {
        "engine": "google_flights",
        "flight_type": "one_way",
        "departure_id": "YYZ",
        "arrival_id": "LHR",
        "outbound_date": DEPARTURE_DATE.isoformat(),
        "travel_class": cabin,
        "currency": "USD",
    }


def _searchapi_flight(cabin: str) -> dict[str, Any]:
    label = "First" if cabin == "first_class" else cabin.replace("_", " ").title()
    return {
        "departure_airport": {
            "id": "YYZ",
            "date": DEPARTURE_DATE.isoformat(),
            "time": "09:00",
            "terminal": "1",
        },
        "arrival_airport": {
            "id": "LHR",
            "date": DEPARTURE_DATE.isoformat(),
            "time": "21:00",
            "terminal": "2",
        },
        "airline": "Air Canada",
        "flight_number": "AC 801",
        "travel_class": label,
        "duration": 420,
    }


def _multi_segment_searchapi_rows(count: int) -> list[dict[str, Any]]:
    airports = ("YYZ", "YUL", "JFK", "BOS", "IAD", "ATL", "MIA", "DFW", "DEN", "LAX")
    start = datetime(2026, 8, 2, 8)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        departure = start + timedelta(hours=index * 3)
        arrival = departure + timedelta(hours=1)
        rows.append(
            {
                "departure_airport": {
                    "id": airports[index],
                    "date": departure.date().isoformat(),
                    "time": departure.strftime("%H:%M"),
                },
                "arrival_airport": {
                    "id": airports[index + 1],
                    "date": arrival.date().isoformat(),
                    "time": arrival.strftime("%H:%M"),
                },
                "airline": "Air Canada",
                "flight_number": f"AC {800 + index}",
                "travel_class": "Economy",
                "duration": 60,
            }
        )
    return rows


def _multi_segment_ignav_rows(count: int) -> list[dict[str, Any]]:
    airports = ("YYZ", "YUL", "JFK", "BOS", "IAD", "ATL", "MIA", "DFW", "DEN", "LAX")
    start = datetime(2026, 8, 2, 8, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        departure = start + timedelta(hours=index * 3)
        arrival = departure + timedelta(hours=1)
        rows.append(
            {
                "departure_airport": airports[index],
                "arrival_airport": airports[index + 1],
                "marketing_carrier_code": "AC",
                "flight_number": str(800 + index),
                "departure_time_local": departure.replace(tzinfo=None).isoformat(
                    timespec="minutes"
                ),
                "arrival_time_local": arrival.replace(tzinfo=None).isoformat(
                    timespec="minutes"
                ),
                "departure_time_utc": departure.isoformat().replace("+00:00", "Z"),
                "arrival_time_utc": arrival.isoformat().replace("+00:00", "Z"),
                "departure_timezone": "UTC",
                "arrival_timezone": "UTC",
                "duration_minutes": 60,
            }
        )
    return rows


def test_alternate_segment_parsers_accept_eight_and_reject_nine_segments() -> None:
    searchapi = _parse_searchapi_segments(_multi_segment_searchapi_rows(8), "economy")
    ignav = _parse_ignav_segments(_multi_segment_ignav_rows(8), "economy")

    assert len(searchapi) == 8
    assert ignav is not None and len(ignav[0]) == 8
    assert _parse_searchapi_segments(_multi_segment_searchapi_rows(9), "economy") == ()
    assert _parse_ignav_segments(_multi_segment_ignav_rows(9), "economy") is None


def _metadata(suffix: str) -> dict[str, Any]:
    return {
        "status": "Success",
        "created_at": NOW.isoformat(),
        "id": f"searchid{suffix:0>4}"[-16:],
        "google_flights_url": (
            "https://www.google.com/travel/flights/search?tfs=test"
        ),
    }


def _searchapi_search_payload(
    cabin: str,
    params: dict[str, Any],
    *,
    include_candidate: bool,
    candidate_count: int = 1,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "search_metadata": _metadata(cabin),
        "search_parameters": _searchapi_parameters(cabin),
        "best_flights": [],
        "other_flights": [],
    }
    assert params["flight_type"] == "one_way"
    if include_candidate:
        payload["best_flights"] = [
            {
                "type": "One way",
                "price": 512.34 + index,
                "booking_token": f"bookingtoken{index + 1:04d}",
                "flights": [_searchapi_flight(cabin)],
            }
            for index in range(candidate_count)
        ]
    return payload


def _searchapi_booking_payload(
    cabin: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    parameters = _searchapi_parameters(cabin)
    parameters["booking_token"] = params["booking_token"]
    return {
        "search_metadata": _metadata(f"booking-{cabin}"),
        "search_parameters": parameters,
        "selected_flights": [
            {"type": "One way", "flights": [_searchapi_flight(cabin)]}
        ],
        "booking_options": [
            {
                "book_with": "Air Canada",
                "price": 519.99,
                "flight_numbers": ["AC 801"],
                "booking_request": {
                    "url": "https://www.google.com/travel/clk/f?token=test"
                },
                "option_title": "Standard",
                "extensions": ["1 free checked bag", "Free changes"],
            }
        ],
    }


def _ignav_segment() -> dict[str, Any]:
    return {
        "departure_airport": "YYZ",
        "arrival_airport": "LHR",
        "marketing_carrier_code": "AC",
        "flight_number": "801",
        "departure_time_local": f"{DEPARTURE_DATE.isoformat()}T09:00",
        "arrival_time_local": f"{DEPARTURE_DATE.isoformat()}T21:00",
        "departure_time_utc": f"{DEPARTURE_DATE.isoformat()}T13:00:00Z",
        "arrival_time_utc": f"{DEPARTURE_DATE.isoformat()}T20:00:00Z",
        "departure_timezone": "America/Toronto",
        "arrival_timezone": "Europe/London",
        "duration_minutes": 420,
    }


def _ignav_outbound() -> dict[str, Any]:
    return {
        "duration_minutes": 420,
        "carrier": "Air Canada",
        "segments": [_ignav_segment()],
    }


def _ignav_candidate(
    *,
    ignav_id: str = "ignavcandidate01",
    cabin: str = "economy",
    amount: float = 510.0,
) -> dict[str, Any]:
    return {
        "ignav_id": ignav_id,
        "cabin_class": cabin,
        "price": {"amount": amount, "currency": "USD", "status": "verified"},
        "outbound": _ignav_outbound(),
        "bags": {"checked": 1},
    }


def _ignav_booking_payload(
    booking_url: str,
    *,
    cabin: str = "economy",
    amount: float = 510.0,
) -> dict[str, Any]:
    return {
        "itinerary": {
            "cabin_class": cabin,
            "price": {
                "amount": amount,
                "currency": "USD",
                "status": "verified",
            },
            "outbound": _ignav_outbound(),
        },
        "booking_options": [
            {
                "legs": ["outbound"],
                "links": [
                    {
                        "provider_name": "Air Canada",
                        "provider_type": "airline",
                        "price": {
                            "amount": 515.0,
                            "currency": "USD",
                            "status": "verified",
                        },
                        "url": booking_url,
                        "fare_name": "Standard",
                    }
                ],
            }
        ],
    }


def test_searchapi_strictly_searches_four_one_way_cabins_and_verifies_booking(
    tmp_path: Path,
) -> None:
    client = _SearchApiClient()
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search(
        "YYZ",
        "LHR",
        DEPARTURE_DATE,
        fetched_at=NOW,
    )

    assert result.status == "confirmed_offers"
    assert result.provider_code == SEARCHAPI_PROVIDER_CODE
    assert result.provider_name == SEARCHAPI_PROVIDER_NAME
    assert result.searched_cabins == (
        "economy",
        "premium_economy",
        "business",
        "first",
    )
    assert result.search_calls_used == 4
    assert result.pricing_calls_used == 1
    assert result.verified_candidate_count == 1
    assert len(result.offers) == 1
    offer = result.offers[0]
    assert offer.provider_code == SEARCHAPI_PROVIDER_CODE
    assert offer.provider_name == SEARCHAPI_PROVIDER_NAME
    assert offer.booking_verified is True
    assert offer.booking_url.startswith("https://www.google.com/travel/clk/")
    assert offer.total_amount_usd == 519.99
    cabin_calls = [
        params
        for url, params in client.calls
        if url == SEARCHAPI_SEARCH_URL and "booking_token" not in params
    ]
    assert sorted(call["travel_class"] for call in cabin_calls) == sorted(
        _TRAVEL_CLASSES.values()
    )
    assert all(call["flight_type"] == "one_way" for call in cabin_calls)
    assert all(call["gl"] == "ca" for call in cabin_calls)
    booking_calls = [
        params
        for url, params in client.calls
        if url == SEARCHAPI_SEARCH_URL and "booking_token" in params
    ]
    assert all(call["gl"] == "ca" for call in booking_calls)
    assert len(client.calls) == 5
    assert all(url == SEARCHAPI_SEARCH_URL for url, _ in client.calls)


def test_searchapi_runs_cabin_and_booking_requests_with_bounded_concurrency(
    tmp_path: Path,
) -> None:
    client = _ConcurrentSearchApiClient()
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage-concurrent.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.search_calls_used == 4
    assert result.pricing_calls_used == 4
    assert result.verified_candidate_count == 4
    assert len(result.offers) == 4
    assert client.search_barrier.broken is False
    assert client.booking_barrier.broken is False


def test_searchapi_reserves_all_eligible_booking_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _SearchApiClient(
        candidate_cabins=set(_TRAVEL_CLASSES.values()),
        candidates_per_cabin=3,
    )
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage-batches.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )
    requested_reservations: list[int] = []
    original_reserve = provider._ledger.reserve

    def recording_reserve(
        provider_code: str,
        calls: int,
        *,
        hard_limit: int,
        require_all: bool = False,
    ) -> int:
        requested_reservations.append(calls)
        return original_reserve(
            provider_code,
            calls,
            hard_limit=hard_limit,
            require_all=require_all,
        )

    monkeypatch.setattr(provider._ledger, "reserve", recording_reserve)

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.eligible_candidate_count == 12
    assert result.pricing_calls_used == 12
    assert result.quota_skipped_candidate_count == 0
    assert result.coverage_status == "complete"
    assert result.quota_limit is None
    assert requested_reservations == [4, 12]


def test_searchapi_partial_rejections_do_not_claim_complete_no_results(
    tmp_path: Path,
) -> None:
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage-rejected.sqlite3",
        client=_RejectingBoundedSearchApiClient(),
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "no_results"
    assert result.offers == ()
    assert result.eligible_candidate_count == 12
    assert result.verification_attempted_count == 12
    assert result.strictly_rejected_candidate_count == 12
    assert result.quota_skipped_candidate_count == 0
    assert result.coverage_status == "complete"
    assert result.quota_limit is None


def test_searchapi_business_authentication_failure_is_reserved(
    tmp_path: Path,
) -> None:
    client = _FailingSearchApiClient(401)
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage-auth.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "authentication_failed"
    assert result.offers == ()
    assert result.search_calls_used == 4
    assert result.search_monthly_used == 4
    assert len(client.calls) == 4
    assert all(url == SEARCHAPI_SEARCH_URL for url, _ in client.calls)
    assert provider._ledger.snapshot(provider.ledger_provider_code) == 4


def test_searchapi_missing_key_fails_closed_without_network(tmp_path: Path) -> None:
    provider = SearchApiFlightOfferProvider(
        None,
        usage_path=tmp_path / "usage.sqlite3",
        client=_NoCallClient(),
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "not_configured"
    assert result.offers == ()


def test_searchapi_lifetime_hard_limit_survives_new_searches(tmp_path: Path) -> None:
    client = _SearchApiClient(include_candidate=False)
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage.sqlite3",
        monthly_limit=4,
        client=client,
        now_provider=lambda: NOW,
    )

    first = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    calls_after_first = len(client.calls)
    second = provider.search("YUL", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert first.status == "no_results"
    assert first.search_calls_used == 4
    assert first.search_monthly_used == 4
    assert second.status == "budget_exhausted"
    assert second.search_calls_used == 0
    assert second.search_monthly_used == 4
    assert len(client.calls) == calls_after_first
    assert all(url == SEARCHAPI_SEARCH_URL for url, _ in client.calls)


def test_searchapi_four_cabin_reservation_is_all_or_nothing(tmp_path: Path) -> None:
    client = _SearchApiClient(include_candidate=False)
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage.sqlite3",
        monthly_limit=5,
        client=client,
        now_provider=lambda: NOW,
    )
    assert provider._ledger.reserve(
        provider.ledger_provider_code,
        2,
        hard_limit=5,
    ) == 2

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "budget_exhausted"
    assert result.search_calls_used == 0
    assert provider._ledger.snapshot(provider.ledger_provider_code) == 2
    assert client.calls == []


def test_searchapi_hourly_wall_is_not_reported_as_lifetime_exhaustion(
    tmp_path: Path,
) -> None:
    client = _FailingSearchApiClient(429)
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "rate_limited"
    assert result.quota_limit == "hourly"
    assert result.search_calls_used == 4
    assert result.search_monthly_used == 4
    assert len(client.calls) == 4
    assert all(url == SEARCHAPI_SEARCH_URL for url, _ in client.calls)


def test_searchapi_cache_hit_makes_no_request_or_reservation(
    tmp_path: Path,
) -> None:
    client = _SearchApiClient(include_candidate=False)
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage-cache.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    first = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    calls_after_first = len(client.calls)
    used_after_first = provider._ledger.snapshot(provider.ledger_provider_code)
    second = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert first.status == "no_results"
    assert second.status == "no_results"
    assert second.cache_hit is True
    assert second.search_calls_used == 0
    assert len(client.calls) == calls_after_first
    assert provider._ledger.snapshot(provider.ledger_provider_code) == used_after_first


def test_searchapi_keeps_verified_offer_when_one_cabin_search_fails(
    tmp_path: Path,
) -> None:
    client = _PartiallyFailingSearchApiClient()
    provider = SearchApiFlightOfferProvider(
        "searchapi-test-key",
        usage_path=tmp_path / "usage.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert len(result.offers) == 1
    assert result.search_calls_used == 4
    assert result.search_failed_cabin_count == 1
    assert result.coverage_status == "provider_incomplete"
    assert result.provider_failed_candidate_count == 0
    assert {item.exception_type for item in result.diagnostics} == {"ProviderError"}


def test_ignav_is_quarantined_without_both_operator_attestations(
    tmp_path: Path,
) -> None:
    no_attestation = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "not-attested.sqlite3",
        release_verified=True,
        free_account_attested=False,
        client=_NoCallClient(),
        now_provider=lambda: NOW,
    )
    not_released = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "not-released.sqlite3",
        release_verified=False,
        free_account_attested=True,
        client=_NoCallClient(),
        now_provider=lambda: NOW,
    )

    assert (
        no_attestation.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW).status
        == "budget_not_configured"
    )
    assert (
        not_released.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW).status
        == "test_environment_rejected"
    )
    assert no_attestation.strict_release_enabled is False
    assert not_released.strict_release_enabled is False
    assert no_attestation.provider_code == IGNAV_QUARANTINE_PROVIDER_CODE
    assert no_attestation.provider_name == IGNAV_QUARANTINE_PROVIDER_NAME
    assert not_released.provider_code == IGNAV_QUARANTINE_PROVIDER_CODE


def test_ignav_four_cabin_reservation_is_all_or_nothing(tmp_path: Path) -> None:
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "usage.sqlite3",
        release_verified=True,
        free_account_attested=True,
        lifetime_limit=5,
        client=_NoCallClient(),
        now_provider=lambda: NOW,
    )
    assert provider._ledger.reserve(
        provider.ledger_provider_code,
        2,
        hard_limit=5,
    ) == 2

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "budget_exhausted"
    assert result.search_calls_used == 0
    assert provider._ledger.snapshot(provider.ledger_provider_code) == 2


@pytest.mark.parametrize(
    ("booking_url", "expected_status"),
    [
        ("http://www.aircanada.com/booking/test", "no_results"),
        ("https://www.aircanada.com/booking/test", "confirmed_offers"),
    ],
)
def test_ignav_release_still_requires_a_verified_public_https_booking_link(
    tmp_path: Path,
    booking_url: str,
    expected_status: str,
) -> None:
    client = _IgnavClient(booking_url=booking_url)
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-{expected_status}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == expected_status
    assert result.search_calls_used == 4
    assert result.pricing_calls_used == 1
    if expected_status == "confirmed_offers":
        assert len(result.offers) == 1
        assert result.provider_code == IGNAV_VERIFIED_PROVIDER_CODE
        assert result.provider_name == IGNAV_VERIFIED_PROVIDER_NAME
        assert result.offers[0].provider_code == IGNAV_VERIFIED_PROVIDER_CODE
        assert result.offers[0].provider_name == IGNAV_VERIFIED_PROVIDER_NAME
        assert result.offers[0].booking_url == booking_url
    else:
        assert result.offers == ()
        assert result.strictly_rejected_candidate_count == 1


def test_ignav_verifies_all_candidates_with_bounded_concurrency(
    tmp_path: Path,
) -> None:
    client = _BoundedConcurrentIgnavClient()
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "ignav-bounded.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.search_calls_used == 4
    assert result.eligible_candidate_count == 12
    assert result.verification_attempted_count == 12
    assert result.pricing_calls_used == 12
    assert result.quota_skipped_candidate_count == 0
    assert result.coverage_status == "complete"
    assert result.quota_limit is None
    assert provider._ledger.snapshot(provider.ledger_provider_code) == 16
    assert sum(url == IGNAV_ONE_WAY_URL for url, _ in client.calls) == 4
    assert sum(url == IGNAV_BOOKING_LINKS_URL for url, _ in client.calls) == 12


def test_ignav_partial_rejections_do_not_claim_complete_no_results(
    tmp_path: Path,
) -> None:
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "ignav-rejected.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=_BoundedConcurrentIgnavClient(
            booking_url="http://www.aircanada.com/booking/test"
        ),
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "no_results"
    assert result.offers == ()
    assert result.eligible_candidate_count == 12
    assert result.verification_attempted_count == 12
    assert result.strictly_rejected_candidate_count == 12
    assert result.quota_skipped_candidate_count == 0
    assert result.coverage_status == "complete"
    assert result.quota_limit is None


class _FallbackStub:
    configured = True
    environment = "production"

    def __init__(self, name: str, result: FlightOfferSearchResult, log: list[str]) -> None:
        self.name = name
        self.result = result
        self.log = log

    def search(self, *_args: Any, **_kwargs: Any) -> FlightOfferSearchResult:
        self.log.append(self.name)
        return self.result


class _SequencedFallbackStub(_FallbackStub):
    def __init__(
        self,
        name: str,
        results: tuple[FlightOfferSearchResult, ...],
        log: list[str],
    ) -> None:
        super().__init__(name, results[-1], log)
        self.results = results
        self.calls = 0

    def search(self, *_args: Any, **kwargs: Any) -> FlightOfferSearchResult:
        self.log.append(
            f"{self.name}:{str(bool(kwargs.get('force_refresh'))).lower()}"
        )
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class _RaisingStrictProvider:
    configured = True
    environment = "production"
    provider_code = SEARCHAPI_PROVIDER_CODE
    provider_name = SEARCHAPI_PROVIDER_NAME

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def search(self, *_args: Any, **_kwargs: Any) -> FlightOfferSearchResult:
        self.log.append("raising-searchapi")
        raise sqlite3.OperationalError("sensitive provider detail must not escape")


def _offer_result(
    *,
    provider_code: str = SERPAPI_PROVIDER_CODE,
    provider_name: str = SERPAPI_PROVIDER_NAME,
) -> FlightOfferSearchResult:
    offer = ConfirmedFlightOffer(
        provider_offer_id=f"{provider_code}-offer",
        validating_airline_code="AC",
        airline_name="Air Canada",
        cabin="economy",
        total_amount_usd=519.99,
        base_amount_usd=None,
        last_ticketing_date=None,
        number_of_bookable_seats=None,
        seat_count_capped=False,
        verified_at=NOW,
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        segments=(
            FlightOfferSegment(
                segment_id="AC801-202608020900-0",
                origin="YYZ",
                destination="LHR",
                departure_at=datetime(2026, 8, 2, 9),
                arrival_at=datetime(2026, 8, 2, 21),
                marketing_airline_code="AC",
                operating_airline_code=None,
                flight_number="801",
                departure_terminal="1",
                arrival_terminal="2",
                aircraft_icao=None,
                cabin="economy",
                booking_class=None,
                fare_basis=None,
                fare_brand="Standard",
                checked_bags_quantity=1,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            ),
        ),
        refundable_fare=None,
        no_penalty_fare=True,
        no_restriction_fare=None,
        booking_url="https://www.google.com/travel/clk/f?token=test",
        booking_url_kind="direct_get",
        booking_provider="Air Canada",
        provider_code=provider_code,
        provider_name=provider_name,
    )
    historical_market_contexts = (
        (
            RouteCabinMarketHistory(
                origin="YYZ",
                destination="LHR",
                departure_date=DEPARTURE_DATE,
                cabin="economy",
                provider_observed_at=NOW,
                points=(
                    RouteCabinMarketPricePoint(
                        observed_at=NOW - timedelta(days=1),
                        price_usd=499.0,
                    ),
                ),
            ),
        )
        if provider_code == SERPAPI_PROVIDER_CODE
        else ()
    )
    return FlightOfferSearchResult(
        offers=(offer,),
        status="confirmed_offers",
        observed_at=NOW,
        environment="production",
        searched_cabins=("economy",),
        calls_used=2,
        cache_hit=False,
        search_calls_used=1,
        pricing_calls_used=1,
        search_monthly_limit=(100 if provider_code == SEARCHAPI_PROVIDER_CODE else 250),
        search_monthly_used=2,
        eligible_candidate_count=1,
        verification_attempted_count=1,
        verified_candidate_count=1,
        coverage_status="complete",
        provider_code=provider_code,
        provider_name=provider_name,
        historical_market_contexts=historical_market_contexts,
    )


def test_fallback_queries_every_strict_provider_and_aggregates_confirmed_runs() -> None:
    call_log: list[str] = []
    serpapi = _FallbackStub("serpapi", _offer_result(), call_log)
    searchapi = _FallbackStub(
        "searchapi",
        _offer_result(
            provider_code=SEARCHAPI_PROVIDER_CODE,
            provider_name=SEARCHAPI_PROVIDER_NAME,
        ),
        call_log,
    )
    provider = FallbackFlightOfferProvider((serpapi, searchapi))

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.provider_code == AGGREGATE_PROVIDER_CODE
    assert result.provider_name == AGGREGATE_PROVIDER_NAME
    assert result.status == "confirmed_offers"
    assert len(result.offers) == 1
    assert len(result.provider_runs) == 2
    assert result.historical_market_contexts == ()
    serpapi_run = next(
        run
        for run in result.provider_runs
        if run.provider_code == SERPAPI_PROVIDER_CODE
    )
    assert len(serpapi_run.historical_market_contexts) == 1
    assert (
        serpapi_run.historical_market_contexts[0].scope
        == "route_departure_date_cabin_market"
    )
    assert {run.provider_code for run in result.provider_runs} == {
        SERPAPI_PROVIDER_CODE,
        SEARCHAPI_PROVIDER_CODE,
    }
    assert sorted(call_log) == ["searchapi", "serpapi"]


def test_fallback_runs_strict_providers_in_configured_order() -> None:
    call_log: list[str] = []
    provider = FallbackFlightOfferProvider(
        (
            _SequencedFallbackStub("serpapi", (_offer_result(),), call_log),
            _SequencedFallbackStub(
                "searchapi",
                (
                    _offer_result(
                        provider_code=SEARCHAPI_PROVIDER_CODE,
                        provider_name=SEARCHAPI_PROVIDER_NAME,
                    ),
                ),
                call_log,
            ),
        )
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert len(result.provider_runs) == 2
    assert call_log == ["serpapi:false", "searchapi:false"]


def test_fallback_retries_one_transient_provider_failure_once() -> None:
    call_log: list[str] = []
    transient = FlightOfferSearchResult(
        offers=(),
        status="provider_error",
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
        provider_code=SERPAPI_PROVIDER_CODE,
        provider_name=SERPAPI_PROVIDER_NAME,
    )
    provider = FallbackFlightOfferProvider(
        (
            _SequencedFallbackStub(
                "serpapi",
                (transient, _offer_result()),
                call_log,
            ),
        )
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.calls_used == 3
    assert result.search_calls_used == 2
    assert result.pricing_calls_used == 1
    assert result.cache_hit is False
    assert call_log == ["serpapi:false", "serpapi:true"]
    assert any(
        diagnostic.exception_type == "ControlledProviderRetry"
        for diagnostic in result.diagnostics
    )


@pytest.mark.parametrize(
    "status",
    (
        "authentication_failed",
        "rate_limited",
        "budget_exhausted",
        "no_results",
    ),
)
def test_fallback_does_not_spend_retry_calls_on_terminal_source_status(
    status: str,
) -> None:
    call_log: list[str] = []
    terminal = (
        _complete_no_results(SERPAPI_PROVIDER_CODE, SERPAPI_PROVIDER_NAME)
        if status == "no_results"
        else FlightOfferSearchResult(
            offers=(),
            status=status,
            observed_at=NOW,
            environment="production",
            searched_cabins=(),
            calls_used=0,
            cache_hit=False,
            provider_code=SERPAPI_PROVIDER_CODE,
            provider_name=SERPAPI_PROVIDER_NAME,
        )
    )
    provider = FallbackFlightOfferProvider(
        (_SequencedFallbackStub("serpapi", (terminal,), call_log),)
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == status
    assert call_log == ["serpapi:false"]


def test_fallback_isolates_unexpected_later_provider_exception() -> None:
    call_log: list[str] = []
    serpapi = _FallbackStub("serpapi", _offer_result(), call_log)
    provider = FallbackFlightOfferProvider(
        (serpapi, _RaisingStrictProvider(call_log))
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert len(result.offers) == 1
    assert result.coverage_status == "provider_incomplete"
    assert [run.status for run in result.provider_runs] == [
        "confirmed_offers",
        "provider_unavailable",
    ]
    assert result.provider_runs[1].diagnostics[0].exception_type == (
        "UnexpectedProviderError"
    )
    assert "sensitive provider detail" not in repr(result)
    assert call_log == ["serpapi", "raising-searchapi", "raising-searchapi"]


def _complete_no_results(
    provider_code: str,
    provider_name: str,
) -> FlightOfferSearchResult:
    return FlightOfferSearchResult(
        offers=(),
        status="no_results",
        observed_at=NOW,
        environment="production",
        searched_cabins=("economy",),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
        coverage_status="complete",
        provider_code=provider_code,
        provider_name=provider_name,
    )


def test_fallback_continues_after_complete_no_results_to_expand_coverage() -> None:
    call_log: list[str] = []
    serpapi = _FallbackStub(
        "serpapi",
        _complete_no_results(SERPAPI_PROVIDER_CODE, SERPAPI_PROVIDER_NAME),
        call_log,
    )
    searchapi = _FallbackStub(
        "searchapi",
        _offer_result(
            provider_code=SEARCHAPI_PROVIDER_CODE,
            provider_name=SEARCHAPI_PROVIDER_NAME,
        ),
        call_log,
    )

    result = FallbackFlightOfferProvider((serpapi, searchapi)).search(
        "YYZ",
        "LHR",
        DEPARTURE_DATE,
        fetched_at=NOW,
    )

    assert result.provider_code == AGGREGATE_PROVIDER_CODE
    assert result.status == "confirmed_offers"
    assert result.offers[0].provider_code == SEARCHAPI_PROVIDER_CODE
    assert [run.status for run in result.provider_runs] == [
        "no_results",
        "confirmed_offers",
    ]
    assert sorted(call_log) == ["searchapi", "serpapi"]


def test_fallback_single_source_result_remains_backward_compatible() -> None:
    call_log: list[str] = []
    original = _offer_result()

    result = FallbackFlightOfferProvider(
        (_FallbackStub("serpapi", original, call_log),)
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result is original
    assert result.provider_runs == ()
    assert call_log == ["serpapi"]


def test_aggregate_deduplicates_actual_itinerary_and_cabin_to_lowest_price() -> None:
    call_log: list[str] = []
    serpapi_result = _offer_result()
    searchapi_result = _offer_result(
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )
    cheaper_searchapi_offer = replace(
        searchapi_result.offers[0],
        validating_airline_code="UA",
        airline_name="United Airlines",
        total_amount_usd=410.0,
        booking_provider="Different seller",
    )
    searchapi_result = replace(searchapi_result, offers=(cheaper_searchapi_offer,))

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub("serpapi", serpapi_result, call_log),
            _FallbackStub("searchapi", searchapi_result, call_log),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert len(result.offers) == 1
    assert result.offers[0].provider_code == SEARCHAPI_PROVIDER_CODE
    assert result.offers[0].total_amount_usd == 410.0
    assert result.verified_candidate_count == 2
    assert result.deduplicated_verified_count == 1


def test_aggregate_retains_distinct_complete_itineraries_from_each_provider() -> None:
    call_log: list[str] = []
    serpapi_result = _offer_result()
    searchapi_result = _offer_result(
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )
    distinct_segment = replace(
        searchapi_result.offers[0].segments[0],
        segment_id="AC802-202608021000-0",
        flight_number="802",
        departure_at=datetime(2026, 8, 2, 10),
        arrival_at=datetime(2026, 8, 2, 22),
    )
    searchapi_result = replace(
        searchapi_result,
        offers=(
            replace(
                searchapi_result.offers[0],
                total_amount_usd=480.0,
                segments=(distinct_segment,),
            ),
        ),
    )

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub("serpapi", serpapi_result, call_log),
            _FallbackStub("searchapi", searchapi_result, call_log),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert len(result.offers) == 2
    assert {offer.provider_code for offer in result.offers} == {
        SERPAPI_PROVIDER_CODE,
        SEARCHAPI_PROVIDER_CODE,
    }


def test_aggregate_distinguishes_provider_failure_from_complete_no_results() -> None:
    call_log: list[str] = []
    provider_error = FlightOfferSearchResult(
        offers=(),
        status="provider_error",
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub(
                "serpapi",
                _complete_no_results(SERPAPI_PROVIDER_CODE, SERPAPI_PROVIDER_NAME),
                call_log,
            ),
            _FallbackStub("searchapi", provider_error, call_log),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_error"
    assert result.offers == ()
    assert [run.status for run in result.provider_runs] == [
        "no_results",
        "provider_error",
    ]


def test_aggregate_marks_quota_limited_empty_runs_as_budget_exhausted() -> None:
    call_log: list[str] = []

    def bounded_empty(provider_code: str, provider_name: str) -> FlightOfferSearchResult:
        return FlightOfferSearchResult(
            offers=(),
            status="no_results",
            observed_at=NOW,
            environment="production",
            searched_cabins=("economy", "premium_economy", "business", "first"),
            calls_used=10,
            cache_hit=False,
            search_calls_used=4,
            pricing_calls_used=6,
            eligible_candidate_count=12,
            verification_attempted_count=6,
            strictly_rejected_candidate_count=6,
            quota_skipped_candidate_count=6,
            coverage_status="quota_limited",
            quota_limit="provider_specific",
            provider_code=provider_code,
            provider_name=provider_name,
        )

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub(
                "serpapi",
                bounded_empty(SERPAPI_PROVIDER_CODE, SERPAPI_PROVIDER_NAME),
                call_log,
            ),
            _FallbackStub(
                "searchapi",
                bounded_empty(SEARCHAPI_PROVIDER_CODE, SEARCHAPI_PROVIDER_NAME),
                call_log,
            ),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "budget_exhausted"
    assert result.offers == ()
    assert result.eligible_candidate_count == 24
    assert result.verification_attempted_count == 12
    assert result.strictly_rejected_candidate_count == 12
    assert result.quota_skipped_candidate_count == 12
    assert result.coverage_status == "quota_limited"
    assert result.quota_limit == "provider_specific"
    assert [run.status for run in result.provider_runs] == ["no_results", "no_results"]


def test_aggregate_marks_confirmed_plus_budget_exhausted_source_quota_limited() -> None:
    call_log: list[str] = []
    budget_exhausted = FlightOfferSearchResult(
        offers=(),
        status="budget_exhausted",
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=0,
        cache_hit=False,
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub("serpapi", _offer_result(), call_log),
            _FallbackStub("searchapi", budget_exhausted, call_log),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert len(result.offers) == 1
    assert result.coverage_status == "quota_and_provider_incomplete"
    assert result.quota_limit == "provider_specific"
    assert result.retry_quota_limited is False
    metadata = PredictionService._fare_metadata(result)
    assert metadata.coverage_status == "quota_and_provider_incomplete"
    assert metadata.quota_limit == "provider_specific"
    assert [run.status for run in metadata.provider_runs] == [
        "confirmed_offers",
        "budget_exhausted",
    ]


def test_aggregate_authentication_failure_plus_quota_wall_is_structured(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: list[str] = []
    authentication_failed = FlightOfferSearchResult(
        offers=(),
        status="authentication_failed",
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
        provider_code=SERPAPI_PROVIDER_CODE,
        provider_name=SERPAPI_PROVIDER_NAME,
    )
    budget_exhausted = FlightOfferSearchResult(
        offers=(),
        status="budget_exhausted",
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=0,
        cache_hit=False,
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub("serpapi", authentication_failed, call_log),
            _FallbackStub("searchapi", budget_exhausted, call_log),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "authentication_failed"
    assert result.offers == ()
    assert result.coverage_status == "quota_and_provider_incomplete"
    assert result.quota_limit == "provider_specific"
    assert result.retry_quota_limited is False
    assert [run.status for run in result.provider_runs] == [
        "authentication_failed",
        "budget_exhausted",
    ]
    metadata = PredictionService._fare_metadata(result)
    assert metadata.status == "authentication_failed"
    assert metadata.coverage_status == "quota_and_provider_incomplete"
    assert metadata.quota_limit == "provider_specific"
    assert [run.status for run in metadata.provider_runs] == [
        "authentication_failed",
        "budget_exhausted",
    ]
    assert call_log == ["serpapi", "searchapi"]

    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "auto")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-only-serpapi-key")
    monkeypatch.setenv("SEARCHAPI_API_KEY", "test-only-searchapi-key")
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_EmptyScheduleProvider(),  # type: ignore[arg-type]
        flight_offer_provider=FallbackFlightOfferProvider(
            (
                _FallbackStub("serpapi", authentication_failed, []),
                _FallbackStub("searchapi", budget_exhausted, []),
            )
        ),
        now_provider=lambda: NOW,
    )
    monkeypatch.setattr(api_module, "get_service", lambda: service)

    response = TestClient(api_module.app).post(
        "/v1/compare",
        json={
            "origin": "YYZ",
            "destination": "LHR",
            "departure_date": DEPARTURE_DATE.isoformat(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_status"] == "fare_provider_authentication_failed"
    assert payload["offers"] == []
    assert payload["fare_search_metadata"]["status"] == "authentication_failed"
    assert (
        payload["fare_search_metadata"]["coverage_status"]
        == "quota_and_provider_incomplete"
    )
    assert payload["fare_search_metadata"]["quota_limit"] == "provider_specific"
    assert [
        run["status"] for run in payload["fare_search_metadata"]["provider_runs"]
    ] == ["authentication_failed", "budget_exhausted"]


def test_search_result_rejects_offer_from_a_different_provider() -> None:
    searchapi_offer = _offer_result(
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    ).offers[0]

    with pytest.raises(ValueError, match="every fare-search offer"):
        FlightOfferSearchResult(
            offers=(searchapi_offer,),
            status="confirmed_offers",
            observed_at=NOW,
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
            provider_code=SERPAPI_PROVIDER_CODE,
            provider_name=SERPAPI_PROVIDER_NAME,
        )


def test_quarantine_identity_cannot_construct_a_strict_offer() -> None:
    verified = _offer_result(
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    ).offers[0]

    with pytest.raises(ValueError, match="confirmed offer provider is invalid"):
        replace(
            verified,
            provider_code=IGNAV_QUARANTINE_PROVIDER_CODE,
            provider_name=IGNAV_QUARANTINE_PROVIDER_NAME,
        )


class _EmptyScheduleProvider:
    def search(self, *_args: Any, **_kwargs: Any) -> ScheduleSearchResult:
        return ScheduleSearchResult((), frozenset())


def test_searchapi_attribution_survives_service_and_schema_serialization(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    searchapi_result = _offer_result(
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_EmptyScheduleProvider(),  # type: ignore[arg-type]
        flight_offer_provider=_FallbackStub("searchapi", searchapi_result, []),
        now_provider=lambda: NOW,
    )

    comparison = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=DEPARTURE_DATE,
        )
    )
    serialized = comparison.model_dump(mode="json")

    assert comparison.fare_search_metadata is not None
    assert comparison.fare_search_metadata.provider_code == SEARCHAPI_PROVIDER_CODE
    assert comparison.fare_search_metadata.provider_name == SEARCHAPI_PROVIDER_NAME
    assert comparison.offers[0].live_fare is not None
    assert comparison.offers[0].live_fare.provider_code == SEARCHAPI_PROVIDER_CODE
    assert comparison.offers[0].live_fare.provider_name == SEARCHAPI_PROVIDER_NAME
    assert serialized["fare_search_metadata"]["provider_code"] == SEARCHAPI_PROVIDER_CODE
    assert serialized["offers"][0]["live_fare"]["provider_code"] == SEARCHAPI_PROVIDER_CODE
    assert serialized["fare_search_metadata"]["quota_unit"] == "lifetime_requests"

    mismatched = json.loads(json.dumps(serialized))
    mismatched["fare_search_metadata"]["provider_code"] = SERPAPI_PROVIDER_CODE
    mismatched["fare_search_metadata"]["provider_name"] = SERPAPI_PROVIDER_NAME
    mismatched["fare_search_metadata"]["quota_unit"] = "billing_period_requests"
    with pytest.raises(ValueError, match="comparison metadata"):
        ComparisonResponse.model_validate(mismatched)

    detail = service.offer_detail(
        OfferDetailRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=DEPARTURE_DATE,
            offer_id=comparison.offers[0].id,
        )
    )
    assert detail.fare_search_metadata is not None
    assert detail.fare_search_metadata.provider_code == SEARCHAPI_PROVIDER_CODE
    assert all(
        leg.data_basis == "searchapi_booking_confirmed"
        for leg in detail.itinerary.legs
    )


def test_aggregate_attribution_survives_comparison_and_each_offer_detail(
    trained_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    serpapi_result = _offer_result()
    searchapi_result = _offer_result(
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
    )
    second_segment = replace(
        searchapi_result.offers[0].segments[0],
        segment_id="AC802-202608021000-0",
        flight_number="802",
        departure_at=datetime(2026, 8, 2, 10),
        arrival_at=datetime(2026, 8, 2, 22),
    )
    searchapi_result = replace(
        searchapi_result,
        offers=(replace(searchapi_result.offers[0], segments=(second_segment,)),),
    )
    provider = FallbackFlightOfferProvider(
        (
            _FallbackStub("serpapi", serpapi_result, []),
            _FallbackStub("searchapi", searchapi_result, []),
        )
    )
    service = PredictionService(
        trained_model_dir,
        context_provider=ContextProvider(),
        schedule_provider=_EmptyScheduleProvider(),  # type: ignore[arg-type]
        flight_offer_provider=provider,
        now_provider=lambda: NOW,
    )

    comparison = service.compare(
        ComparisonRequest(
            origin="YYZ",
            destination="LHR",
            departure_date=DEPARTURE_DATE,
        )
    )
    serialized = comparison.model_dump(mode="json")

    assert comparison.fare_search_metadata is not None
    assert comparison.fare_search_metadata.provider_code == AGGREGATE_PROVIDER_CODE
    assert len(comparison.fare_search_metadata.provider_runs) == 2
    assert len(comparison.offers) == 2
    assert len(comparison.historical_market_contexts) == 1
    assert comparison.historical_market_contexts[0].provider_code == (
        SERPAPI_PROVIDER_CODE
    )
    assert (
        comparison.historical_market_contexts[0].relation_to_offer
        == "market_context_not_selected_offer_history"
    )
    assert {offer.live_fare.provider_code for offer in comparison.offers} == {
        SERPAPI_PROVIDER_CODE,
        SEARCHAPI_PROVIDER_CODE,
    }
    ComparisonResponse.model_validate(serialized)

    for offer in comparison.offers:
        detail = service.offer_detail(
            OfferDetailRequest(
                origin="YYZ",
                destination="LHR",
                departure_date=DEPARTURE_DATE,
                offer_id=offer.id,
            )
        )
        assert detail.fare_search_metadata is not None
        assert detail.fare_search_metadata.provider_code == AGGREGATE_PROVIDER_CODE
        assert detail.historical_market_context is not None
        assert detail.historical_market_context.provider_code == SERPAPI_PROVIDER_CODE
        assert (
            detail.historical_market_context.relation_to_offer
            == "market_context_not_selected_offer_history"
        )
        expected_basis = {
            SERPAPI_PROVIDER_CODE: "serpapi_booking_confirmed",
            SEARCHAPI_PROVIDER_CODE: "searchapi_booking_confirmed",
        }[detail.offer.live_fare.provider_code]
        assert all(leg.data_basis == expected_basis for leg in detail.itinerary.legs)
