from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import Any

import pytest
from fastapi.testclient import TestClient

from flight_forecaster import api as api_module
from flight_forecaster.alternate_fare_providers import (
    IGNAV_BOOKING_LINKS_URL,
    IGNAV_ONE_WAY_URL,
    SCRAPPA_BOOKING_DETAILS_URL,
    SCRAPPA_ONE_WAY_URL,
    SEARCHAPI_SEARCH_URL,
    FallbackFlightOfferProvider,
    IgnavQuarantineFlightOfferProvider,
    ScrappaFlightOfferProvider,
    SearchApiFlightOfferProvider,
    _parse_ignav_segments,
    _parse_scrappa_segments,
    _parse_searchapi_segments,
    _scrappa_search_parameters_match,
)
from flight_forecaster.availability import (
    AGGREGATE_PROVIDER_CODE,
    AGGREGATE_PROVIDER_NAME,
    IGNAV_QUARANTINE_PROVIDER_CODE,
    IGNAV_QUARANTINE_PROVIDER_NAME,
    IGNAV_VERIFIED_PROVIDER_CODE,
    IGNAV_VERIFIED_PROVIDER_NAME,
    SCRAPPA_PROVIDER_CODE,
    SCRAPPA_PROVIDER_NAME,
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
    def __init__(
        self,
        payload: Any,
        status_code: int = 200,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = dict(headers or {})
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


class _ConfigurableIgnavClient:
    def __init__(
        self,
        *,
        candidate_transform: Any = None,
        booking_transform: Any = None,
        first_search_status: int | None = None,
        first_search_transport_error: bool = False,
        search_failure_attempts: int = 1,
        search_failure_cabin: str = "economy",
        first_booking_status: int | None = None,
        first_booking_transport_error: bool = False,
        booking_failure_attempts: int = 1,
    ) -> None:
        self.candidate_transform = candidate_transform
        self.booking_transform = booking_transform
        self.first_search_status = first_search_status
        self.first_search_transport_error = first_search_transport_error
        self.search_failure_attempts = search_failure_attempts
        self.search_failure_cabin = search_failure_cabin
        self.first_booking_status = first_booking_status
        self.first_booking_transport_error = first_booking_transport_error
        self.booking_failure_attempts = booking_failure_attempts
        self.search_attempts: dict[str, int] = {}
        self.booking_attempts = 0
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
            attempt = self.search_attempts.get(cabin, 0) + 1
            self.search_attempts[cabin] = attempt
            if (
                cabin == self.search_failure_cabin
                and attempt == 1
                and self.first_search_transport_error
            ):
                raise TimeoutError("simulated Ignav cabin-search transport failure")
            if (
                cabin == self.search_failure_cabin
                and self.first_search_status is not None
                and attempt <= self.search_failure_attempts
            ):
                search_id = f"ignav-search-retry-{self.first_search_status}"
                return _Response(
                    {"error": "simulated Ignav cabin-search failure"},
                    self.first_search_status,
                    headers={"x-request-id": search_id},
                )
            rows: list[dict[str, Any]] = []
            if cabin == "economy":
                candidate = _ignav_candidate()
                if self.candidate_transform is not None:
                    self.candidate_transform(candidate)
                rows.append(candidate)
            return _Response(
                {
                    "origin": "YYZ",
                    "destination": "LHR",
                    "departure_date": DEPARTURE_DATE.isoformat(),
                    "itineraries": rows,
                }
            )
        assert url == IGNAV_BOOKING_LINKS_URL
        assert json == {"ignav_id": "ignavcandidate01"}
        self.booking_attempts += 1
        if self.booking_attempts == 1 and self.first_booking_transport_error:
            raise TimeoutError("simulated Ignav booking transport failure")
        if (
            self.first_booking_status is not None
            and self.booking_attempts <= self.booking_failure_attempts
        ):
            search_id = f"ignav-retry-{self.first_booking_status}"
            return _Response(
                {"error": "simulated Ignav booking failure"},
                self.first_booking_status,
                headers={"x-request-id": search_id},
            )
        payload = _ignav_booking_payload(
            "https://www.aircanada.com/booking/test"
        )
        if self.booking_transform is not None:
            self.booking_transform(payload)
        return _Response(payload)


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


class _ScrappaClient:
    def __init__(
        self,
        *,
        candidates_per_cabin: int = 2,
        candidate_counts_by_cabin: dict[str, int] | None = None,
        zero_search_price: bool = False,
        search_error_envelope: bool = False,
        booking_error_envelope: bool = False,
        candidate_cabins: set[str] | None = None,
        first_booking_status: int | None = None,
        first_booking_status_tokens: set[str] | None = None,
        booking_failure_attempts: int = 1,
        first_booking_transport_error: bool = False,
    ) -> None:
        self.candidates_per_cabin = candidates_per_cabin
        self.candidate_counts_by_cabin = (
            dict(candidate_counts_by_cabin)
            if candidate_counts_by_cabin is not None
            else None
        )
        self.zero_search_price = zero_search_price
        self.search_error_envelope = search_error_envelope
        self.booking_error_envelope = booking_error_envelope
        self.candidate_cabins = candidate_cabins
        self.first_booking_status = first_booking_status
        self.first_booking_status_tokens = (
            set(first_booking_status_tokens)
            if first_booking_status_tokens is not None
            else None
        )
        self.booking_failure_attempts = booking_failure_attempts
        self.first_booking_transport_error = first_booking_transport_error
        self.booking_attempts: dict[str, int] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert headers["x-api-key"] == "scrappa-test-key"
        assert timeout > 0
        self.calls.append((url, dict(params)))
        cabin = str(params["cabin_class"])
        if url == SCRAPPA_ONE_WAY_URL:
            candidate_count = (
                self.candidate_counts_by_cabin.get(cabin, 0)
                if self.candidate_counts_by_cabin is not None
                else self.candidates_per_cabin
            )
            if self.search_error_envelope:
                return _Response(
                    {
                        "error": "upstream search failed",
                        "service": "google_flights_one_way",
                        "request_id": f"scrappa-search-error-{cabin}",
                    }
                )
            return _Response(
                {
                    "flights": [
                        {
                            "booking_token": f"scrappa-token-{cabin}-{index}",
                            "price": (
                                0 if self.zero_search_price else 500 + index
                            ),
                            "currency": "USD",
                            "total_duration_minutes": 720,
                            "legs": [
                                {
                                    "departure_airport": "YYZ",
                                    "arrival_airport": "LHR",
                                    "departure_time": f"{DEPARTURE_DATE.isoformat()}T09:00",
                                    "arrival_time": f"{DEPARTURE_DATE.isoformat()}T21:00",
                                    "duration_minutes": 420,
                                    "airline": "AC",
                                    "airline_name": "Air Canada",
                                    "flight_number": "801",
                                    "stops": 0,
                                }
                            ],
                        }
                        for index in range(candidate_count)
                        if self.candidate_cabins is None
                        or cabin in self.candidate_cabins
                    ],
                    "search_metadata": {
                        "origin": "YYZ",
                        "destination": "LHR",
                        "departure_date": DEPARTURE_DATE.isoformat(),
                        "cabin_class": cabin,
                        "passengers": {
                            "adults": 1,
                            "children": 0,
                            "infants_in_seat": 0,
                            "infants_on_lap": 0,
                        },
                        "request_id": f"scrappa-search-{cabin}",
                    },
                }
            )
        assert url == SCRAPPA_BOOKING_DETAILS_URL
        booking_token = str(params["booking_token"])
        attempt = self.booking_attempts.get(booking_token, 0) + 1
        self.booking_attempts[booking_token] = attempt
        if attempt == 1 and self.first_booking_transport_error:
            raise TimeoutError("simulated booking transport failure")
        if (
            attempt <= self.booking_failure_attempts
            and self.first_booking_status is not None
            and (
                self.first_booking_status_tokens is None
                or booking_token in self.first_booking_status_tokens
            )
        ):
            search_id = f"scrappa-retry-{self.first_booking_status}"
            return _Response(
                {"error": "transient booking failure", "request_id": search_id},
                self.first_booking_status,
                headers={"x-request-id": search_id},
            )
        if self.booking_error_envelope:
            return _Response(
                {
                    "error": "upstream booking verification failed",
                    "service": "google_flights_booking_details",
                    "failed_stage": "booking_request_exhausted",
                    "reason": "cookie_session_unavailable",
                    "request_id": f"scrappa-booking-error-{cabin}",
                }
            )
        index = int(booking_token.rsplit("-", 1)[1])
        return _Response(
            {
                "flight_details": {
                    "airline_code": "AC",
                    "airline_name": "Air Canada",
                    "total_duration_minutes": 720,
                    "leg": {"flight_number": "801"},
                },
                "fare_options": [
                    {
                        "provider": "Air Canada",
                        "price": 510 + index,
                        "currency": "USD",
                        "booking_url": "https://www.aircanada.com/booking/scrappa-test",
                        "flight_numbers": ["AC 801"],
                    }
                ],
                "booking_metadata": {
                    "origin": "YYZ",
                    "destination": "LHR",
                    "departure_date": DEPARTURE_DATE.isoformat(),
                    "airline": "AC",
                    "flight_number": "801",
                    "request_id": f"scrappa-booking-{cabin}-{index}",
                },
            }
        )

    def post(self, *_args: Any, **_kwargs: Any) -> _Response:
        raise AssertionError("provider network must not be called")


class _ScrappaBookingShapeClient(_ScrappaClient):
    """Return controlled booking payload variants without external requests."""

    def __init__(
        self,
        *,
        array_flight_details: bool = False,
        include_booking_metadata: bool = True,
        empty_fare_options: bool = False,
    ) -> None:
        super().__init__(candidates_per_cabin=1)
        self.array_flight_details = array_flight_details
        self.include_booking_metadata = include_booking_metadata
        self.empty_fare_options = empty_fare_options

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
        if url != SCRAPPA_BOOKING_DETAILS_URL:
            return response

        payload = dict(response.payload)
        if self.array_flight_details:
            payload["flight_details"] = [payload["flight_details"]]
        if not self.include_booking_metadata:
            payload.pop("booking_metadata", None)
        if self.empty_fare_options:
            payload["fare_options"] = []
        payload["price_insights"] = []
        payload["baggage_info"] = []
        return _Response(payload)


class _ScrappaContradictoryBookingClient(_ScrappaClient):
    """Inject one contradictory booking-details shape per strict regression."""

    def __init__(self, variant: str) -> None:
        if variant not in {
            "extra_unparseable_detail",
            "contradictory_airline",
            "scalar_flight_numbers",
        }:
            raise ValueError("unsupported Scrappa contradiction variant")
        super().__init__(candidates_per_cabin=1)
        self.variant = variant

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
        if url != SCRAPPA_BOOKING_DETAILS_URL:
            return response

        payload = dict(response.payload)
        details = dict(payload["flight_details"])
        option = dict(payload["fare_options"][0])
        if self.variant == "extra_unparseable_detail":
            payload["flight_details"] = [details, {}]
            option.pop("flight_numbers", None)
        elif self.variant == "contradictory_airline":
            leg = dict(details["leg"])
            leg["flight_number"] = "AC 801"
            details["leg"] = leg
            details["airline_code"] = "BA"
            payload["flight_details"] = details
            option.pop("flight_numbers", None)
        else:
            option["flight_numbers"] = "BA 999"
        payload["fare_options"] = [option]
        return _Response(payload)


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


@pytest.mark.parametrize("missing_link_price", (False, True))
def test_ignav_accepts_official_nullable_summaries_and_itinerary_price_fallback(
    tmp_path: Path,
    missing_link_price: bool,
) -> None:
    def nullable_candidate(candidate: dict[str, Any]) -> None:
        candidate["cabin_class"] = None
        candidate["outbound"]["duration_minutes"] = None

    def nullable_booking(payload: dict[str, Any]) -> None:
        payload["itinerary"]["cabin_class"] = None
        payload["itinerary"]["outbound"]["duration_minutes"] = None
        link = payload["booking_options"][0]["links"][0]
        if missing_link_price:
            link.pop("price")
        else:
            link["price"] = None

    client = _ConfigurableIgnavClient(
        candidate_transform=nullable_candidate,
        booking_transform=nullable_booking,
    )
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-nullable-{missing_link_price}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.verified_candidate_count == 1
    assert result.offers[0].total_amount_usd == 510.0
    assert result.offers[0].booking_url.startswith("https://")


@pytest.mark.parametrize(
    ("stage", "field", "value", "verification_attempts"),
    (
        ("candidate", "cabin_class", "business", 0),
        ("candidate", "duration_minutes", 421, 0),
        ("booking", "cabin_class", "business", 1),
        ("booking", "duration_minutes", 421, 1),
    ),
)
def test_ignav_rejects_present_summary_contradictions(
    tmp_path: Path,
    stage: str,
    field: str,
    value: Any,
    verification_attempts: int,
) -> None:
    def candidate_transform(candidate: dict[str, Any]) -> None:
        if stage == "candidate":
            target = candidate if field == "cabin_class" else candidate["outbound"]
            target[field] = value

    def booking_transform(payload: dict[str, Any]) -> None:
        if stage == "booking":
            itinerary = payload["itinerary"]
            target = itinerary if field == "cabin_class" else itinerary["outbound"]
            target[field] = value

    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-contradiction-{stage}-{field}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=_ConfigurableIgnavClient(
            candidate_transform=candidate_transform,
            booking_transform=booking_transform,
        ),
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.offers == ()
    assert result.verification_attempted_count == verification_attempts
    assert result.status in {"no_results", "provider_unavailable"}


def test_ignav_does_not_replace_an_explicit_invalid_link_price(
    tmp_path: Path,
) -> None:
    def invalid_link_price(payload: dict[str, Any]) -> None:
        payload["booking_options"][0]["links"][0]["price"] = {
            "amount": 515.0,
            "currency": "CAD",
            "status": "verified",
        }

    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "ignav-invalid-link-price.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=_ConfigurableIgnavClient(booking_transform=invalid_link_price),
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "no_results"
    assert result.offers == ()
    assert result.strictly_rejected_candidate_count == 1


@pytest.mark.parametrize(
    ("first_search_status", "transport_error"),
    ((424, False), (503, False), (None, True)),
)
def test_ignav_retries_one_transient_failure_for_only_the_affected_cabin(
    tmp_path: Path,
    first_search_status: int | None,
    transport_error: bool,
) -> None:
    sleep_delays: list[float] = []
    client = _ConfigurableIgnavClient(
        first_search_status=first_search_status,
        first_search_transport_error=transport_error,
    )
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-search-retry-{first_search_status}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.search_calls_used == 5
    assert result.pricing_calls_used == 1
    assert result.calls_used == 6
    assert result.search_monthly_used == 6
    assert result.retry_quota_limited is False
    assert client.search_attempts == {
        "economy": 2,
        "premium_economy": 1,
        "business": 1,
        "first": 1,
    }
    assert sleep_delays == [pytest.approx(0.15)]
    assert "ControlledCabinRetry" in {
        item.exception_type for item in result.diagnostics
    }


@pytest.mark.parametrize("terminal_status", (400, 401, 402, 403, 404, 429, 500))
def test_ignav_does_not_retry_terminal_cabin_search_statuses(
    tmp_path: Path,
    terminal_status: int,
) -> None:
    sleep_delays: list[float] = []
    client = _ConfigurableIgnavClient(first_search_status=terminal_status)
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-search-terminal-{terminal_status}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.search_calls_used == 4
    assert result.pricing_calls_used == 0
    assert result.calls_used == 4
    assert result.search_failed_cabin_count == 1
    assert client.search_attempts["economy"] == 1
    assert sleep_delays == []
    assert "ControlledCabinRetry" not in {
        item.exception_type for item in result.diagnostics
    }


@pytest.mark.parametrize(
    ("lifetime_limit", "expected_attempts", "retry_quota_limited"),
    ((1_000, 2, False), (4, 1, True)),
)
def test_ignav_cabin_retry_stops_after_once_or_at_the_lifetime_wall(
    tmp_path: Path,
    lifetime_limit: int,
    expected_attempts: int,
    retry_quota_limited: bool,
) -> None:
    sleep_delays: list[float] = []
    client = _ConfigurableIgnavClient(
        first_search_status=424,
        search_failure_attempts=10,
    )
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-search-wall-{lifetime_limit}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        lifetime_limit=lifetime_limit,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.0,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_unavailable"
    assert result.search_calls_used == 3 + expected_attempts
    assert result.pricing_calls_used == 0
    assert result.calls_used == 3 + expected_attempts
    assert result.search_failed_cabin_count == 1
    assert result.retry_quota_limited is retry_quota_limited
    assert result.coverage_status == (
        "quota_and_provider_incomplete"
        if retry_quota_limited
        else "provider_incomplete"
    )
    assert result.quota_limit == ("lifetime" if retry_quota_limited else None)
    assert client.search_attempts["economy"] == expected_attempts
    assert len(sleep_delays) == (1 if expected_attempts == 2 else 0)


def test_ignav_handled_cabin_retry_is_not_replayed_by_fallback(
    tmp_path: Path,
) -> None:
    client = _ConfigurableIgnavClient(
        first_search_status=503,
        search_failure_attempts=10,
    )
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / "ignav-search-fallback.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=lambda _delay: None,
        retry_jitter_provider=lambda: 0.0,
    )

    result = FallbackFlightOfferProvider((provider,)).search(
        "YYZ",
        "LHR",
        DEPARTURE_DATE,
        fetched_at=NOW,
    )

    assert result.status == "provider_unavailable"
    assert result.search_calls_used == 5
    assert client.search_attempts["economy"] == 2
    assert "ControlledProviderRetry" not in {
        item.exception_type for item in result.diagnostics
    }


@pytest.mark.parametrize(
    ("first_booking_status", "transport_error"),
    ((424, False), (503, False), (None, True)),
)
def test_ignav_retries_one_transient_booking_failure(
    tmp_path: Path,
    first_booking_status: int | None,
    transport_error: bool,
) -> None:
    sleep_delays: list[float] = []
    client = _ConfigurableIgnavClient(
        first_booking_status=first_booking_status,
        first_booking_transport_error=transport_error,
    )
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-retry-{first_booking_status}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.verification_attempted_count == 1
    assert result.provider_failed_candidate_count == 0
    assert result.pricing_calls_used == 2
    assert result.calls_used == 6
    assert client.booking_attempts == 2
    assert sleep_delays == [pytest.approx(0.15)]
    assert "ControlledCandidateRetry" in {
        item.exception_type for item in result.diagnostics
    }


@pytest.mark.parametrize("terminal_status", (400, 401, 402, 403, 404, 429, 500))
def test_ignav_does_not_retry_terminal_booking_statuses(
    tmp_path: Path,
    terminal_status: int,
) -> None:
    sleep_delays: list[float] = []
    client = _ConfigurableIgnavClient(first_booking_status=terminal_status)
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-terminal-{terminal_status}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.provider_failed_candidate_count == 1
    assert result.pricing_calls_used == 1
    assert result.calls_used == 5
    assert client.booking_attempts == 1
    assert sleep_delays == []
    assert "ControlledCandidateRetry" not in {
        item.exception_type for item in result.diagnostics
    }


@pytest.mark.parametrize(
    ("lifetime_limit", "expected_attempts", "retry_quota_limited"),
    ((1_000, 2, False), (5, 1, True)),
)
def test_ignav_transient_failure_stops_after_one_retry_or_lifetime_wall(
    tmp_path: Path,
    lifetime_limit: int,
    expected_attempts: int,
    retry_quota_limited: bool,
) -> None:
    sleep_delays: list[float] = []
    client = _ConfigurableIgnavClient(
        first_booking_status=424,
        booking_failure_attempts=10,
    )
    provider = IgnavQuarantineFlightOfferProvider(
        "ignav-test-key",
        usage_path=tmp_path / f"ignav-retry-wall-{lifetime_limit}.sqlite3",
        release_verified=True,
        free_account_attested=True,
        lifetime_limit=lifetime_limit,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.0,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_unavailable"
    assert result.provider_failed_candidate_count == 1
    assert result.pricing_calls_used == expected_attempts
    assert result.retry_quota_limited is retry_quota_limited
    assert result.quota_limit == ("lifetime" if retry_quota_limited else None)
    assert client.booking_attempts == expected_attempts
    assert len(sleep_delays) == (1 if expected_attempts == 2 else 0)


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


class _CircuitFallbackStub(_SequencedFallbackStub):
    provider_code = SERPAPI_PROVIDER_CODE
    provider_name = SERPAPI_PROVIDER_NAME


class _RecoverableAuthenticationCircuitStub(_CircuitFallbackStub):
    authentication_recheck_seconds = 300.0


class _BlockingHalfOpenFallbackStub(_CircuitFallbackStub):
    def __init__(
        self,
        name: str,
        results: tuple[FlightOfferSearchResult, ...],
        log: list[str],
    ) -> None:
        super().__init__(name, results, log)
        self.probe_started = Event()
        self.release_probe = Event()

    def search(self, *_args: Any, **kwargs: Any) -> FlightOfferSearchResult:
        result = super().search(*_args, **kwargs)
        if self.calls == 3:
            self.probe_started.set()
            if not self.release_probe.wait(timeout=2):
                raise TimeoutError("half-open probe was not released")
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


def _circuit_failure_result(status: str) -> FlightOfferSearchResult:
    return FlightOfferSearchResult(
        offers=(),
        status=status,
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
        provider_code=SERPAPI_PROVIDER_CODE,
        provider_name=SERPAPI_PROVIDER_NAME,
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


def test_fallback_circuit_skips_authentication_failure_on_later_queries() -> None:
    clock = [0.0]
    call_log: list[str] = []
    provider_stub = _CircuitFallbackStub(
        "serpapi",
        (_circuit_failure_result("authentication_failed"),),
        call_log,
    )
    provider = FallbackFlightOfferProvider(
        (provider_stub,),
        circuit_clock=lambda: clock[0],
    )

    first = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    clock[0] = 10_000.0
    skipped = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert first.status == "authentication_failed"
    assert skipped.status == "authentication_failed"
    assert skipped.calls_used == 0
    assert skipped.cache_hit is True
    assert call_log == ["serpapi:false"]
    assert any(
        diagnostic.exception_type == "ProviderCircuitOpen"
        for diagnostic in skipped.diagnostics
    )


def test_fallback_authentication_circuit_uses_account_only_recovery_window() -> None:
    clock = [0.0]
    call_log: list[str] = []
    provider_stub = _RecoverableAuthenticationCircuitStub(
        "serpapi",
        (
            _circuit_failure_result("authentication_failed"),
            _offer_result(),
        ),
        call_log,
    )
    provider = FallbackFlightOfferProvider(
        (provider_stub,),
        circuit_clock=lambda: clock[0],
    )

    first = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    clock[0] = 299.999
    skipped = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    clock[0] = 300.0
    recovered = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert first.status == "authentication_failed"
    assert skipped.calls_used == 0
    assert skipped.cache_hit is True
    assert recovered.status == "confirmed_offers"
    assert call_log == ["serpapi:false", "serpapi:false"]


def test_fallback_circuit_skips_transient_provider_error_for_90_seconds() -> None:
    clock = [0.0]
    call_log: list[str] = []
    transient = _circuit_failure_result("provider_error")
    provider_stub = _CircuitFallbackStub(
        "serpapi",
        (transient, transient),
        call_log,
    )
    provider = FallbackFlightOfferProvider(
        (provider_stub,),
        circuit_clock=lambda: clock[0],
    )

    first = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    clock[0] = 89.999
    skipped = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert first.status == "provider_error"
    assert skipped.status == "provider_error"
    assert skipped.calls_used == 0
    assert skipped.cache_hit is True
    assert call_log == ["serpapi:false", "serpapi:true"]
    assert any(
        diagnostic.exception_type == "ProviderCircuitOpen"
        for diagnostic in skipped.diagnostics
    )


def test_fallback_circuit_allows_one_half_open_probe_and_closes_on_success() -> None:
    clock = [0.0]
    call_log: list[str] = []
    transient = _circuit_failure_result("provider_error")
    provider_stub = _BlockingHalfOpenFallbackStub(
        "serpapi",
        (transient, transient, _offer_result()),
        call_log,
    )
    provider = FallbackFlightOfferProvider(
        (provider_stub,),
        circuit_clock=lambda: clock[0],
    )
    provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    clock[0] = 90.0
    probe_results: list[FlightOfferSearchResult] = []
    probe_errors: list[BaseException] = []

    def run_probe() -> None:
        try:
            probe_results.append(
                provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            probe_errors.append(exc)

    probe_thread = Thread(target=run_probe)
    probe_thread.start()
    assert provider_stub.probe_started.wait(timeout=2)

    concurrent = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    assert provider_stub.calls == 3
    assert any(
        diagnostic.exception_type == "ProviderCircuitOpen"
        for diagnostic in concurrent.diagnostics
    )

    provider_stub.release_probe.set()
    probe_thread.join(timeout=2)
    assert not probe_thread.is_alive()
    assert probe_errors == []
    assert len(probe_results) == 1
    assert probe_results[0].status == "confirmed_offers"

    after_success = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert after_success.status == "confirmed_offers"
    assert provider_stub.calls == 4
    assert call_log == [
        "serpapi:false",
        "serpapi:true",
        "serpapi:false",
        "serpapi:false",
    ]


@pytest.mark.parametrize(
    (
        "verification_attempted_count",
        "provider_failed_candidate_count",
        "quota_skipped_candidate_count",
        "coverage_status",
        "quota_limit",
    ),
    (
        (0, 0, 1, "quota_and_provider_incomplete", "provider_specific"),
        (1, 1, 0, "provider_incomplete", None),
    ),
)
def test_fallback_does_not_retry_a_provider_after_candidate_work_started(
    verification_attempted_count: int,
    provider_failed_candidate_count: int,
    quota_skipped_candidate_count: int,
    coverage_status: str,
    quota_limit: str | None,
) -> None:
    call_log: list[str] = []
    partial = FlightOfferSearchResult(
        offers=(),
        status="provider_error",
        observed_at=NOW,
        environment="production",
        searched_cabins=("economy",),
        calls_used=1 + verification_attempted_count,
        cache_hit=False,
        search_calls_used=1,
        pricing_calls_used=verification_attempted_count,
        eligible_candidate_count=1,
        verification_attempted_count=verification_attempted_count,
        provider_failed_candidate_count=provider_failed_candidate_count,
        search_failed_cabin_count=(1 if verification_attempted_count == 0 else 0),
        quota_skipped_candidate_count=quota_skipped_candidate_count,
        coverage_status=coverage_status,
        quota_limit=quota_limit,
        provider_code=SCRAPPA_PROVIDER_CODE,
        provider_name=SCRAPPA_PROVIDER_NAME,
    )
    provider = FallbackFlightOfferProvider(
        (
            _SequencedFallbackStub(
                "scrappa",
                (
                    partial,
                    _offer_result(
                        provider_code=SCRAPPA_PROVIDER_CODE,
                        provider_name=SCRAPPA_PROVIDER_NAME,
                    ),
                ),
                call_log,
            ),
        )
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_error"
    assert call_log == ["scrappa:false"]


def test_aggregate_failure_notice_explains_bounded_recovery_in_both_languages() -> None:
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
    provider_unavailable = FlightOfferSearchResult(
        offers=(),
        status="provider_unavailable",
        observed_at=NOW,
        environment="production",
        searched_cabins=(),
        calls_used=1,
        cache_hit=False,
        search_calls_used=1,
        provider_code=IGNAV_VERIFIED_PROVIDER_CODE,
        provider_name=IGNAV_VERIFIED_PROVIDER_NAME,
    )
    result = FallbackFlightOfferProvider(
        (
            _SequencedFallbackStub(
                "serpapi",
                (authentication_failed,),
                call_log,
            ),
            _SequencedFallbackStub(
                "ignav",
                (provider_unavailable, provider_unavailable),
                call_log,
            ),
        )
    ).search("YYZ", "PVG", DEPARTURE_DATE, fetched_at=NOW)

    metadata = PredictionService._fare_metadata(result)

    assert result.status == "authentication_failed"
    assert call_log == ["serpapi:false", "ignav:false", "ignav:true"]
    assert "SerpApi Google Flights：认证失败" in metadata.notice.zh
    assert "Ignav Verified Fares：供应商暂不可用" in metadata.notice.zh
    assert "最多完成一次额度受控重试" in metadata.notice.zh
    assert "不会对其盲目重复请求" in metadata.notice.zh
    assert "authentication_failed" not in metadata.notice.zh
    assert "provider_unavailable" not in metadata.notice.zh
    assert "SerpApi Google Flights: authentication failed" in metadata.notice.en
    assert "Ignav Verified Fares: provider temporarily unavailable" in metadata.notice.en
    assert "at most one quota-controlled retry" in metadata.notice.en
    assert "is not retried blindly" in metadata.notice.en


def test_scrappa_verifies_every_candidate_without_a_six_offer_cap(tmp_path: Path) -> None:
    client = _ScrappaClient(candidates_per_cabin=2)
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.provider_code == SCRAPPA_PROVIDER_CODE
    assert result.provider_name == SCRAPPA_PROVIDER_NAME
    assert result.eligible_candidate_count == 8
    assert result.verification_attempted_count == 8
    assert result.verified_candidate_count == 8
    assert result.deduplicated_verified_count == 4
    assert len(result.offers) == 4
    assert result.calls_used == 12
    assert sum(url == SCRAPPA_ONE_WAY_URL for url, _ in client.calls) == 4
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 8
    assert all(offer.total_amount_usd == 510 for offer in result.offers)


@pytest.mark.parametrize(
    ("first_booking_status", "transport_error"),
    (
        (None, True),
        (429, False),
        (500, False),
        (502, False),
        (503, False),
    ),
)
def test_scrappa_retries_one_transient_failure_per_booking_token(
    tmp_path: Path,
    first_booking_status: int | None,
    transport_error: bool,
) -> None:
    sleep_delays: list[float] = []
    client = _ScrappaClient(
        candidates_per_cabin=1,
        candidate_cabins={"economy"},
        first_booking_status=first_booking_status,
        first_booking_transport_error=transport_error,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
        retry_base_delay_seconds=0.1,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.eligible_candidate_count == 1
    assert result.verification_attempted_count == 1
    assert result.verified_candidate_count == 1
    assert result.provider_failed_candidate_count == 0
    assert result.pricing_calls_used == 2
    assert result.calls_used == 6
    assert result.search_monthly_used == 6
    assert result.retry_quota_limited is False
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 2
    assert sleep_delays == [pytest.approx(0.15)]
    assert "ControlledCandidateRetry" in {
        item.exception_type for item in result.diagnostics
    }
    if first_booking_status is not None:
        search_id = f"scrappa-retry-{first_booking_status}"
        assert any(
            item.http_status == first_booking_status
            and item.search_id == search_id
            for item in result.diagnostics
        )
    assert all(
        item.search_id is None or "scrappa-token" not in item.search_id
        for item in result.diagnostics
    )


def test_scrappa_recovers_seventeen_of_twenty_six_transient_booking_failures(
    tmp_path: Path,
) -> None:
    usage_path = tmp_path / "alternate.sqlite3"
    candidate_counts = {
        "economy": 7,
        "premium_economy": 7,
        "business": 6,
        "first": 6,
    }
    all_tokens = [
        f"scrappa-token-{cabin}-{index}"
        for cabin, count in candidate_counts.items()
        for index in range(count)
    ]
    transient_tokens = set(all_tokens[:17])
    sleep_delays: list[float] = []
    client = _ScrappaClient(
        candidate_counts_by_cabin=candidate_counts,
        first_booking_status=502,
        first_booking_status_tokens=transient_tokens,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=usage_path,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.0,
        retry_base_delay_seconds=0.1,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.eligible_candidate_count == 26
    assert result.verification_attempted_count == 26
    assert result.verified_candidate_count == 26
    assert result.strictly_rejected_candidate_count == 0
    assert result.provider_failed_candidate_count == 0
    assert result.quota_skipped_candidate_count == 0
    assert result.coverage_status == "complete"
    assert result.retry_quota_limited is False
    assert result.deduplicated_verified_count == 22
    assert len(result.offers) == 4
    assert set(client.booking_attempts) == set(all_tokens)
    assert {
        token for token, attempts in client.booking_attempts.items() if attempts == 2
    } == transient_tokens
    assert all(
        attempts == (2 if token in transient_tokens else 1)
        for token, attempts in client.booking_attempts.items()
    )
    assert len(sleep_delays) == 17
    assert all(delay == pytest.approx(0.1) for delay in sleep_delays)
    search_calls = [
        params
        for url, params in client.calls
        if url == SCRAPPA_ONE_WAY_URL
    ]
    assert len(search_calls) == 4
    assert {str(params["cabin_class"]) for params in search_calls} == set(
        candidate_counts
    )
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 43
    assert result.pricing_calls_used == 43
    assert result.calls_used == 47
    assert result.search_monthly_used == 47

    with sqlite3.connect(usage_path) as connection:
        diagnostics = connection.execute(
            """
            SELECT exception_type, http_status, COUNT(*)
            FROM alternate_provider_diagnostics
            WHERE provider_code = ?
            GROUP BY exception_type, http_status
            """,
            (SCRAPPA_PROVIDER_CODE,),
        ).fetchall()
        usage = connection.execute(
            """
            SELECT reserved_calls
            FROM alternate_provider_monthly_usage
            WHERE provider_code = ? AND period_key = ?
            """,
            (SCRAPPA_PROVIDER_CODE, "2026-08"),
        ).fetchone()

    assert {
        (str(exception_type), int(http_status), int(count))
        for exception_type, http_status, count in diagnostics
    } == {
        ("ProviderError", 502, 17),
        ("ControlledCandidateRetry", 502, 17),
    }
    assert usage == (47,)


@pytest.mark.parametrize("terminal_status", (400, 401, 402, 403, 422))
def test_scrappa_does_not_retry_terminal_booking_http_statuses(
    tmp_path: Path,
    terminal_status: int,
) -> None:
    sleep_delays: list[float] = []
    client = _ScrappaClient(
        candidates_per_cabin=1,
        candidate_cabins={"economy"},
        first_booking_status=terminal_status,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.eligible_candidate_count == 1
    assert result.verification_attempted_count == 1
    assert result.provider_failed_candidate_count == 1
    assert result.pricing_calls_used == 1
    assert result.calls_used == 5
    assert result.search_monthly_used == 5
    assert result.retry_quota_limited is False
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 1
    assert sleep_delays == []
    assert "ControlledCandidateRetry" not in {
        item.exception_type for item in result.diagnostics
    }


def test_scrappa_retry_requires_an_atomic_monthly_credit_reservation(
    tmp_path: Path,
) -> None:
    sleep_delays: list[float] = []
    client = _ScrappaClient(
        candidates_per_cabin=1,
        candidate_cabins={"economy"},
        first_booking_status=500,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        monthly_limit=5,
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.5,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_error"
    assert result.eligible_candidate_count == 1
    assert result.verification_attempted_count == 1
    assert result.provider_failed_candidate_count == 1
    assert result.pricing_calls_used == 1
    assert result.calls_used == 5
    assert result.search_monthly_used == 5
    assert result.retry_quota_limited is True
    assert result.quota_limit == "monthly"
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 1
    assert sleep_delays == []
    assert "RetryQuotaLimited" in {
        item.exception_type for item in result.diagnostics
    }
    assert "ControlledCandidateRetry" not in {
        item.exception_type for item in result.diagnostics
    }


def test_scrappa_never_retries_a_booking_token_more_than_once(tmp_path: Path) -> None:
    sleep_delays: list[float] = []
    client = _ScrappaClient(
        candidates_per_cabin=1,
        candidate_cabins={"economy"},
        first_booking_status=503,
        booking_failure_attempts=10,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
        retry_sleep_provider=sleep_delays.append,
        retry_jitter_provider=lambda: 0.0,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_error"
    assert result.provider_failed_candidate_count == 1
    assert result.pricing_calls_used == 2
    assert result.calls_used == 6
    assert result.search_monthly_used == 6
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 2
    assert sleep_delays == [pytest.approx(0.1)]
    assert sum(
        item.exception_type == "ControlledCandidateRetry"
        for item in result.diagnostics
    ) == 1


def test_scrappa_numeric_flight_number_and_zero_search_price_are_verified(
    tmp_path: Path,
) -> None:
    client = _ScrappaClient(candidates_per_cabin=1, zero_search_price=True)
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.eligible_candidate_count == 4
    assert result.verification_attempted_count == 4
    assert result.verified_candidate_count == 4
    assert len(result.offers) == 4
    assert all(offer.total_amount_usd == 510 for offer in result.offers)
    assert all(offer.segments[0].flight_number == "801" for offer in result.offers)
    assert sum(url == SCRAPPA_BOOKING_DETAILS_URL for url, _ in client.calls) == 4


def test_scrappa_accepts_official_top_level_booking_arrays(tmp_path: Path) -> None:
    client = _ScrappaBookingShapeClient(
        array_flight_details=True,
        include_booking_metadata=False,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.eligible_candidate_count == 4
    assert result.verification_attempted_count == 4
    assert result.verified_candidate_count == 4
    assert len(result.offers) == 4
    assert all(offer.total_amount_usd == 510 for offer in result.offers)
    assert all(offer.booking_provider == "Air Canada" for offer in result.offers)
    assert all(offer.booking_url.startswith("https://") for offer in result.offers)
    assert all(offer.segments[0].flight_number == "801" for offer in result.offers)


def test_scrappa_booking_metadata_is_optional(tmp_path: Path) -> None:
    client = _ScrappaBookingShapeClient(include_booking_metadata=False)
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "confirmed_offers"
    assert result.verified_candidate_count == 4
    assert len(result.offers) == 4


def test_scrappa_reports_specific_sanitized_strict_rejection_reason(
    tmp_path: Path,
) -> None:
    client = _ScrappaBookingShapeClient(
        array_flight_details=True,
        include_booking_metadata=False,
        empty_fare_options=True,
    )
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "no_results"
    assert result.strictly_rejected_candidate_count == 4
    assert result.provider_failed_candidate_count == 0
    assert {item.exception_type for item in result.diagnostics} == {
        "MissingFareOptions"
    }
    assert "StrictCandidateRejected" not in {
        item.exception_type for item in result.diagnostics
    }


def _assert_scrappa_itinerary_contradiction_is_rejected(
    tmp_path: Path,
    variant: str,
) -> None:
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=_ScrappaContradictoryBookingClient(variant),
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "no_results"
    assert result.offers == ()
    assert result.strictly_rejected_candidate_count == 4
    assert result.provider_failed_candidate_count == 0
    assert {item.exception_type for item in result.diagnostics} == {
        "ItineraryMismatch"
    }


def test_scrappa_rejects_an_extra_unparseable_booking_detail(tmp_path: Path) -> None:
    _assert_scrappa_itinerary_contradiction_is_rejected(
        tmp_path,
        "extra_unparseable_detail",
    )


def test_scrappa_rejects_a_full_flight_number_with_a_conflicting_airline(
    tmp_path: Path,
) -> None:
    _assert_scrappa_itinerary_contradiction_is_rejected(
        tmp_path,
        "contradictory_airline",
    )


def test_scrappa_rejects_scalar_contradictory_option_flight_numbers(
    tmp_path: Path,
) -> None:
    _assert_scrappa_itinerary_contradiction_is_rejected(
        tmp_path,
        "scalar_flight_numbers",
    )


def test_scrappa_accepts_documented_minimal_search_metadata() -> None:
    payload = {
        "flights": [{"booking_token": "redacted-test-token"}],
        "search_metadata": {
            "origin": "YYZ",
            "destination": "LHR",
            "departure_date": DEPARTURE_DATE.isoformat(),
            "response_time_ms": 842,
        },
    }

    assert _scrappa_search_parameters_match(
        payload,
        origin="YYZ",
        destination="LHR",
        departure_date=DEPARTURE_DATE,
        cabin="economy",
    )


def test_scrappa_validates_optional_flat_passenger_metadata() -> None:
    payload = {
        "flights": [{"booking_token": "redacted-test-token"}],
        "search_metadata": {
            "origin": "YYZ",
            "destination": "LHR",
            "departure_date": DEPARTURE_DATE.isoformat(),
            "cabin_class": "economy",
            "adults": "1",
            "children": 0,
        },
    }

    assert _scrappa_search_parameters_match(
        payload,
        origin="YYZ",
        destination="LHR",
        departure_date=DEPARTURE_DATE,
        cabin="economy",
    )


@pytest.mark.parametrize(
    "metadata_patch",
    (
        {"destination": "JFK"},
        {"cabin_class": "business"},
        {"adults": 2},
        {"passengers": {"adults": 1, "children": 1}},
    ),
)
def test_scrappa_rejects_contradictory_optional_search_metadata(
    metadata_patch: dict[str, Any],
) -> None:
    metadata: dict[str, Any] = {
        "origin": "YYZ",
        "destination": "LHR",
        "departure_date": DEPARTURE_DATE.isoformat(),
    }
    metadata.update(metadata_patch)

    assert not _scrappa_search_parameters_match(
        {"flights": [{"booking_token": "redacted-test-token"}], "search_metadata": metadata},
        origin="YYZ",
        destination="LHR",
        departure_date=DEPARTURE_DATE,
        cabin="economy",
    )


def test_scrappa_accepts_documented_empty_result_metadata_array() -> None:
    assert _scrappa_search_parameters_match(
        {"flights": [], "search_metadata": []},
        origin="YYZ",
        destination="LHR",
        departure_date=DEPARTURE_DATE,
        cabin="economy",
    )
    assert not _scrappa_search_parameters_match(
        {"flights": [{"booking_token": "redacted-test-token"}], "search_metadata": []},
        origin="YYZ",
        destination="LHR",
        departure_date=DEPARTURE_DATE,
        cabin="economy",
    )


def test_scrappa_http_200_search_error_envelope_is_provider_error(
    tmp_path: Path,
) -> None:
    client = _ScrappaClient(search_error_envelope=True)
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_error"
    assert result.search_failed_cabin_count == 4
    assert result.eligible_candidate_count == 0
    assert result.coverage_status == "provider_incomplete"
    assert all(item.exception_type == "ProviderError" for item in result.diagnostics)


def test_scrappa_http_200_booking_error_envelope_is_not_a_strict_rejection(
    tmp_path: Path,
) -> None:
    client = _ScrappaClient(candidates_per_cabin=1, booking_error_envelope=True)
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        client=client,
        now_provider=lambda: NOW,
    )

    result = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "provider_error"
    assert result.eligible_candidate_count == 4
    assert result.verification_attempted_count == 4
    assert result.provider_failed_candidate_count == 4
    assert result.strictly_rejected_candidate_count == 0
    assert result.verified_candidate_count == 0
    assert result.pricing_calls_used == 4
    assert sum(client.booking_attempts.values()) == 4
    assert "ControlledCandidateRetry" not in {
        item.exception_type for item in result.diagnostics
    }
    assert all(item.exception_type == "ProviderError" for item in result.diagnostics)


def test_scrappa_accepts_numeric_flight_number_with_separate_airline_code() -> None:
    segments = _parse_scrappa_segments(
        [
            {
                "departure_airport": "YYZ",
                "arrival_airport": "HKG",
                "departure_time": f"{DEPARTURE_DATE.isoformat()}T09:00",
                "arrival_time": f"{DEPARTURE_DATE.isoformat()}T21:00",
                "duration_minutes": 720,
                "airline": "CX",
                "flight_number": "821",
                "stops": 0,
            }
        ],
        "economy",
    )

    assert len(segments) == 1
    assert segments[0].marketing_airline_code == "CX"
    assert segments[0].flight_number == "821"


@pytest.mark.parametrize(
    ("airline", "flight_number"),
    (
        (None, "821"),
        ("CX", "AC 821"),
        ("Cathay Pacific", "821"),
        ("CX", "CX821"),
    ),
)
def test_scrappa_rejects_ambiguous_or_mismatched_flight_numbers(
    airline: str | None,
    flight_number: str,
) -> None:
    legs = [
        {
            "departure_airport": "YYZ",
            "arrival_airport": "HKG",
            "departure_time": f"{DEPARTURE_DATE.isoformat()}T09:00",
            "arrival_time": f"{DEPARTURE_DATE.isoformat()}T21:00",
            "duration_minutes": 720,
            "airline": airline,
            "flight_number": flight_number,
            "stops": 0,
        }
    ]

    assert _parse_scrappa_segments(legs, "economy") == ()


def test_scrappa_local_hard_wall_resets_by_utc_calendar_month(tmp_path: Path) -> None:
    clock = {"now": datetime(2026, 8, 31, 23, 59, tzinfo=UTC)}
    client = _ScrappaClient(candidates_per_cabin=0)
    provider = ScrappaFlightOfferProvider(
        "scrappa-test-key",
        usage_path=tmp_path / "alternate.sqlite3",
        monthly_limit=4,
        client=client,
        now_provider=lambda: clock["now"],
    )

    august = provider.search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)
    august_exhausted = provider.search(
        "YYZ", "CDG", DEPARTURE_DATE, fetched_at=NOW, force_refresh=True
    )
    clock["now"] = datetime(2026, 9, 1, tzinfo=UTC)
    september = provider.search(
        "YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW, force_refresh=True
    )

    assert august.status == "no_results"
    assert august_exhausted.status == "budget_exhausted"
    assert august_exhausted.quota_limit == "monthly"
    assert september.status == "no_results"
    assert september.search_monthly_used == 4


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
    tmp_path: Path,
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
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "clean-model"))
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
    serpapi_status = next(
        provider
        for provider in payload["provider_statuses"]
        if provider["code"] == SERPAPI_PROVIDER_CODE
    )
    assert serpapi_status["status"] == "authentication_failed"
    assert serpapi_status["active"] is True
    searchapi_status = next(
        provider
        for provider in payload["provider_statuses"]
        if provider["code"] == SEARCHAPI_PROVIDER_CODE
    )
    assert searchapi_status["active"] is True
    assert searchapi_status["status"] == "configured"
    assert searchapi_status["quota_status"] == "unknown"
    assert searchapi_status["quota_data_basis"] == "unavailable"
    assert searchapi_status["quota_used"] is None
    assert searchapi_status["quota_remaining"] is None
    assert searchapi_status["quota_observed_at"] is None


@pytest.mark.parametrize("other_status", ("provider_error", "provider_unavailable"))
def test_aggregate_does_not_hide_actionable_authentication_failure(
    other_status: str,
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
    transient_failure = FlightOfferSearchResult(
        offers=(),
        status=other_status,  # type: ignore[arg-type]
        observed_at=NOW,
        environment="production",
        searched_cabins=("economy",),
        calls_used=2,
        cache_hit=False,
        search_calls_used=1,
        pricing_calls_used=1,
        eligible_candidate_count=1,
        verification_attempted_count=1,
        provider_failed_candidate_count=1,
        coverage_status="provider_incomplete",
        provider_code=SCRAPPA_PROVIDER_CODE,
        provider_name=SCRAPPA_PROVIDER_NAME,
    )

    result = FallbackFlightOfferProvider(
        (
            _FallbackStub("serpapi", authentication_failed, call_log),
            _FallbackStub("scrappa", transient_failure, call_log),
        )
    ).search("YYZ", "LHR", DEPARTURE_DATE, fetched_at=NOW)

    assert result.status == "authentication_failed"
    assert result.offers == ()
    assert [run.status for run in result.provider_runs] == [
        "authentication_failed",
        other_status,
    ]
    assert call_log == ["serpapi", "scrappa"]


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
