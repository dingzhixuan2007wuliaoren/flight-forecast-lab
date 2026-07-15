from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from flight_forecaster.availability import (
    SERPAPI_ACCOUNT_URL,
    SERPAPI_SEARCH_URL,
    NullFlightOfferProvider,
    SerpApiFlightOfferProvider,
    flight_offer_provider_from_env,
)

_FETCHED_AT = datetime(2026, 7, 15, 12, tzinfo=UTC)
_PROVIDER_CREATED_AT = "2026-07-15 12:00:00 UTC"
_BILLING_CYCLE_KEY = "renewal:2026-08-01"
_HOUR_BUCKET_KEY = "2026-07-15T12"


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


def _flight(cabin: str, suffix: int = 0) -> dict[str, Any]:
    class_number = {
        "Economy": 100,
        "Premium economy": 200,
        "Business": 300,
        "First": 400,
    }[cabin]
    return {
        "departure_airport": {
            "name": "Toronto Pearson International Airport",
            "id": "YYZ",
            "time": f"2026-08-20 {18 + suffix:02d}:15",
            "terminal": "1",
        },
        "arrival_airport": {
            "name": "Heathrow Airport",
            "id": "LHR",
            "time": f"2026-08-21 {6 + suffix:02d}:20",
            "terminal": "2",
        },
        "duration": 425,
        "airplane": "Boeing 787",
        "airline": "Air Canada",
        "travel_class": cabin,
        "flight_number": f"AC {class_number + suffix}",
    }


def _search_payload(travel_class: int) -> dict[str, Any]:
    cabin = {
        1: "Economy",
        2: "Premium economy",
        3: "Business",
        4: "First",
    }[travel_class]
    amount = {1: 300, 2: 500, 3: 900, 4: 1500}[travel_class]
    return {
        "search_metadata": {
            "status": "Success",
            "created_at": _PROVIDER_CREATED_AT,
            "google_flights_url": "https://www.google.com/travel/flights?selected=1",
        },
        "search_parameters": {
            "engine": "google_flights",
            "departure_id": "YYZ",
            "arrival_id": "LHR",
            "outbound_date": "2026-08-20",
            "type": 2,
            "travel_class": travel_class,
            "currency": "USD",
        },
        "best_flights": [
            {
                "flights": [_flight(cabin)],
                "price": amount,
                "type": "One way",
                "booking_token": f"booking-token-{travel_class}",
            }
        ],
        "other_flights": [
            {
                "flights": [_flight(cabin, suffix=1)],
                "price": amount + 100,
                "type": "One way",
                "booking_token": f"booking-token-{travel_class}-high",
            }
        ],
    }


def _booking_payload(travel_class: int, *, invalid: str | None = None) -> dict[str, Any]:
    cabin = {
        1: "Economy",
        2: "Premium economy",
        3: "Business",
        4: "First",
    }[travel_class]
    amount = {1: 310, 2: 510, 3: 910, 4: 1510}[travel_class]
    flight = _flight(cabin)
    if invalid == "wrong_time":
        flight["departure_airport"]["time"] = "2026-08-20 19:15"
    if invalid == "wrong_cabin":
        flight["travel_class"] = "Business" if cabin != "Business" else "Economy"
    option: dict[str, Any] = {
        "book_with": "Air Canada",
        "airline": True,
        "marketed_as": [flight["flight_number"]],
        "price": amount,
        "option_title": "Standard",
        "extensions": ["Refundable", "Free changes"],
        "booking_request": {
            "url": "https://www.google.com/travel/clk/f",
            "post_data": "opaque=form-data",
        },
    }
    if invalid == "no_link":
        option.pop("booking_request")
    if invalid == "wrong_marketed_as":
        option["marketed_as"] = ["AC 999"]
    return {
        "search_metadata": {
            "status": "Success",
            "created_at": _PROVIDER_CREATED_AT,
            "google_flights_url": (
                "https://www.google.com/travel/flights?selected=" + str(travel_class)
            ),
        },
        "search_parameters": {
            "engine": "google_flights",
            "booking_token": f"booking-token-{travel_class}",
            "currency": "USD",
        },
        "selected_flights": [
            {
                "flights": [flight],
                "type": "One way",
            }
        ],
        "booking_options": [{"together": option}],
    }


class _Client:
    def __init__(
        self,
        *,
        invalid: str | None = None,
        account_status: int = 200,
        search_status: int = 200,
        account_usage: int = 0,
        hourly_usage: int = 0,
        hourly_limit: int = 50,
        provider_monthly_limit: int = 250,
        account_state: Any = "Active",
        plan_renewal_date: Any = "2026-08-01",
        search_created_at: str = _PROVIDER_CREATED_AT,
        booking_created_at: str = _PROVIDER_CREATED_AT,
        search_metadata_status: str = "Success",
        booking_metadata_status: str = "Success",
    ) -> None:
        self.invalid = invalid
        self.account_status = account_status
        self.search_status = search_status
        self.account_usage = account_usage
        self.hourly_usage = hourly_usage
        self.hourly_limit = hourly_limit
        self.provider_monthly_limit = provider_monthly_limit
        self.account_state = account_state
        self.plan_renewal_date = plan_renewal_date
        self.search_created_at = search_created_at
        self.booking_created_at = booking_created_at
        self.search_metadata_status = search_metadata_status
        self.booking_metadata_status = booking_metadata_status
        self.account_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.booking_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def get(self, url: str, **kwargs: Any) -> _Response:
        if url == SERPAPI_ACCOUNT_URL:
            self.account_calls.append(kwargs)
            if self.account_status != 200:
                return _Response({"error": "account error"}, self.account_status)
            return _Response(
                {
                    "account_status": self.account_state,
                    "plan_renewal_date": self.plan_renewal_date,
                    "searches_per_month": self.provider_monthly_limit,
                    "this_month_usage": self.account_usage,
                    "this_hour_searches": self.hourly_usage,
                    "account_rate_limit_per_hour": self.hourly_limit,
                }
            )
        assert url == SERPAPI_SEARCH_URL
        params = kwargs["params"]
        if self.search_status != 200:
            return _Response({"error": "search error"}, self.search_status)
        if "booking_token" in params:
            with self._lock:
                self.booking_calls.append(kwargs)
            token = str(params["booking_token"])
            travel_class = int(token.split("-")[2])
            payload = _booking_payload(travel_class, invalid=self.invalid)
            payload["search_metadata"]["created_at"] = self.booking_created_at
            payload["search_metadata"]["status"] = self.booking_metadata_status
            return _Response(payload)
        with self._lock:
            self.search_calls.append(kwargs)
        payload = _search_payload(int(params["travel_class"]))
        payload["search_metadata"]["created_at"] = self.search_created_at
        payload["search_metadata"]["status"] = self.search_metadata_status
        return _Response(payload)


def _provider(
    tmp_path: Path,
    client: Any,
    *,
    monthly_limit: int | None = 250,
) -> SerpApiFlightOfferProvider:
    return SerpApiFlightOfferProvider(
        "serpapi-key",
        usage_path=tmp_path / "private" / "usage.sqlite3",
        monthly_limit=monthly_limit,
        client=client,
        now_provider=lambda: _FETCHED_AT,
    )


def _ledger_calls(
    path: Path,
    *,
    scope: str = "billing_cycle",
    period_key: str = _BILLING_CYCLE_KEY,
) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT calls FROM serpapi_quota_usage
            WHERE scope = ? AND period_key = ?
            """,
            (scope, period_key),
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_search_then_booking_options_returns_only_strictly_verified_offers(
    tmp_path: Path,
) -> None:
    client = _Client()
    provider = _provider(tmp_path, client)
    fetched_at = _FETCHED_AT

    result = provider.search("yyz", "lhr", date(2026, 8, 20), fetched_at=fetched_at)

    assert result.status == "confirmed_offers"
    assert len(result.offers) == 4
    assert {offer.cabin for offer in result.offers} == {
        "economy",
        "premium_economy",
        "business",
        "first",
    }
    assert result.calls_used == 10
    assert result.search_calls_used == 4
    assert result.pricing_calls_used == 6
    assert result.search_monthly_limit == 250
    assert result.pricing_monthly_limit is None
    assert result.search_monthly_used == 10
    assert len(client.account_calls) == 1
    assert len(client.search_calls) == 4
    assert len(client.booking_calls) == 6
    assert client.account_calls[0]["timeout"] == 8.0
    for call in client.search_calls:
        assert call["params"]["type"] == 2
        assert call["params"]["adults"] == 1
        assert call["params"]["currency"] == "USD"
        assert call["params"]["show_hidden"] == "true"
        assert call["params"]["deep_search"] == "true"
        assert "no_cache" not in call["params"]
        assert call["params"]["api_key"] == "serpapi-key"
        assert call["timeout"] == 25.0
    for call in client.booking_calls:
        assert "booking_token" in call["params"]
        assert call["params"]["engine"] == "google_flights"
        assert "no_cache" not in call["params"]
        assert call["timeout"] == 25.0

    offer = result.offers[0]
    assert offer.provider_code == "serpapi_google_flights"
    assert offer.provider_name == "SerpApi Google Flights"
    assert offer.environment == "production"
    assert offer.booking_verified is True
    assert offer.booking_provider == "Air Canada"
    assert offer.booking_url.startswith("https://www.google.com/travel/flights?")
    assert offer.booking_url_kind == "google_flights_itinerary"
    assert offer.source_url == offer.booking_url
    assert offer.verified_at == fetched_at
    assert offer.provider_cache_hit is False
    assert offer.provider_cache_age_seconds == 0
    assert offer.total_amount_usd == 310
    assert offer.refundable_fare is True
    assert offer.no_penalty_fare is True
    assert offer.segments[0].departure_at.tzinfo is None
    assert offer.segments[0].flight_number == "100"
    assert offer.segments[0].fare_brand == "Standard"
    assert len(offer.fingerprint) == 64
    assert replace(offer, provider_offer_id="new", total_amount_usd=999).fingerprint == (
        offer.fingerprint
    )

    assert _ledger_calls(tmp_path / "private" / "usage.sqlite3") == 10
    assert (
        _ledger_calls(
            tmp_path / "private" / "usage.sqlite3",
            scope="hour",
            period_key=_HOUR_BUCKET_KEY,
        )
        == 10
    )


def test_candidate_requires_booking_token_and_booking_response_requires_evidence(
    tmp_path: Path,
) -> None:
    client = _Client(invalid="no_link")
    provider = _provider(tmp_path, client)
    result = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == "no_results"
    assert result.offers == ()

    class _NoTokenClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            if url == SERPAPI_SEARCH_URL and "booking_token" not in kwargs["params"]:
                for key in ("best_flights", "other_flights"):
                    for row in response.payload[key]:
                        row.pop("booking_token")
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    no_token = _NoTokenClient()
    no_token_result = _provider(tmp_path / "other", no_token).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert no_token_result.status == "no_results"
    assert no_token_result.offers == ()
    assert no_token.booking_calls == []


def test_non_google_booking_evidence_is_rejected(tmp_path: Path) -> None:
    class _UnsafeLinkClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            if url == SERPAPI_SEARCH_URL and "booking_token" in kwargs["params"]:
                response.payload["booking_options"][0]["together"]["booking_request"][
                    "url"
                ] = "https://example.test/travel/clk/f"
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    result = _provider(tmp_path, _UnsafeLinkClient()).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == "no_results"
    assert result.offers == ()


def test_optional_booking_extensions_may_be_absent(tmp_path: Path) -> None:
    class _NoExtensionsClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            if url == SERPAPI_SEARCH_URL and "booking_token" in kwargs["params"]:
                response.payload["booking_options"][0]["together"].pop("extensions")
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    result = _provider(tmp_path, _NoExtensionsClient()).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == "confirmed_offers"
    assert result.offers
    assert result.offers[0].refundable_fare is None


def test_get_booking_action_is_exposed_as_direct_redirect(tmp_path: Path) -> None:
    class _GetBookingClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            if url == SERPAPI_SEARCH_URL and "booking_token" in kwargs["params"]:
                response.payload["booking_options"][0]["together"]["booking_request"].pop(
                    "post_data"
                )
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    result = _provider(tmp_path, _GetBookingClient()).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == "confirmed_offers"
    assert result.offers[0].booking_url_kind == "direct_get"
    assert result.offers[0].booking_url == "https://www.google.com/travel/clk/f"


def test_post_booking_action_falls_back_to_initial_search_results_url(
    tmp_path: Path,
) -> None:
    class _MissingBookingPageClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            if url == SERPAPI_SEARCH_URL and "booking_token" in kwargs["params"]:
                response.payload["search_metadata"].pop("google_flights_url")
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    result = _provider(tmp_path, _MissingBookingPageClient()).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "confirmed_offers"
    assert result.offers
    assert all(
        offer.booking_url == "https://www.google.com/travel/flights?selected=1"
        for offer in result.offers
    )
    assert all(
        offer.booking_url_kind == "google_flights_itinerary"
        for offer in result.offers
    )


@pytest.mark.parametrize("invalid", ["wrong_time", "wrong_cabin", "wrong_marketed_as"])
def test_booking_confirmation_must_match_search_itinerary(
    tmp_path: Path,
    invalid: str,
) -> None:
    result = _provider(tmp_path, _Client(invalid=invalid)).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == "no_results"
    assert result.offers == ()


def test_shared_monthly_limit_defaults_and_clamps_to_250(tmp_path: Path) -> None:
    assert _provider(tmp_path, _Client(), monthly_limit=None).monthly_limit == 250
    assert _provider(tmp_path, _Client(), monthly_limit=500).monthly_limit == 250
    assert _provider(tmp_path, _Client(), monthly_limit=25).monthly_limit == 25

    client = _Client(account_usage=247)
    exhausted = _provider(tmp_path / "exhausted", client, monthly_limit=250).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert exhausted.status == "budget_exhausted"
    assert exhausted.calls_used == 0
    assert exhausted.search_monthly_used == 247
    assert client.search_calls == []
    assert client.booking_calls == []


@pytest.mark.parametrize("account_state", [None, "", "Suspended"])
def test_account_status_must_be_explicitly_active(
    tmp_path: Path,
    account_state: Any,
) -> None:
    client = _Client(account_state=account_state)
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )
    assert result.status == "authentication_failed"
    assert client.search_calls == []


@pytest.mark.parametrize(
    "renewal_date",
    [None, "", "not-a-date", "2026-07-14", "2027-08-01"],
)
def test_invalid_or_missing_plan_renewal_date_fails_closed(
    tmp_path: Path,
    renewal_date: Any,
) -> None:
    client = _Client(plan_renewal_date=renewal_date)
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )
    assert result.status == "provider_unavailable"
    assert client.search_calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"account_usage": None},
        {"hourly_usage": None},
        {"hourly_limit": None},
        {"provider_monthly_limit": None},
    ],
)
def test_missing_account_quota_fields_fail_closed(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    client = _Client(**overrides)
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )
    assert result.status == "provider_unavailable"
    assert client.search_calls == []


def test_billing_cycle_uses_provider_renewal_date_not_calendar_month(
    tmp_path: Path,
) -> None:
    first = _provider(tmp_path, _Client(plan_renewal_date="2026-08-01"))
    second = _provider(tmp_path, _Client(plan_renewal_date="2026-09-01"))

    first_result = first.search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )
    second_result = second.search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert first_result.status == "confirmed_offers"
    assert second_result.status == "confirmed_offers"
    ledger_path = tmp_path / "private" / "usage.sqlite3"
    assert _ledger_calls(ledger_path, period_key="renewal:2026-08-01") == 10
    assert _ledger_calls(ledger_path, period_key="renewal:2026-09-01") == 10
    with sqlite3.connect(ledger_path) as connection:
        natural_month = connection.execute(
            """
            SELECT calls FROM serpapi_quota_usage
            WHERE scope = 'billing_cycle' AND period_key = '2026-07'
            """
        ).fetchone()
    assert natural_month is None


def test_hourly_hard_limit_blocks_four_cabin_search_without_partial_reservation(
    tmp_path: Path,
) -> None:
    client = _Client(hourly_usage=47, hourly_limit=100)
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "rate_limited"
    assert result.calls_used == 0
    assert client.search_calls == []
    ledger_path = tmp_path / "private" / "usage.sqlite3"
    assert _ledger_calls(ledger_path) == 0
    assert (
        _ledger_calls(
            ledger_path,
            scope="hour",
            period_key=_HOUR_BUCKET_KEY,
        )
        == 47
    )


def test_booking_verification_uses_only_remaining_hourly_capacity(
    tmp_path: Path,
) -> None:
    client = _Client(hourly_usage=45, hourly_limit=100)
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "confirmed_offers"
    assert result.calls_used == 5
    assert result.search_calls_used == 4
    assert result.pricing_calls_used == 1
    assert len(client.search_calls) == 4
    assert len(client.booking_calls) == 1
    assert len(result.offers) == 1
    ledger_path = tmp_path / "private" / "usage.sqlite3"
    assert _ledger_calls(ledger_path) == 5
    assert (
        _ledger_calls(
            ledger_path,
            scope="hour",
            period_key=_HOUR_BUCKET_KEY,
        )
        == 50
    )


def test_two_provider_instances_share_sqlite_limit_without_overwrite(
    tmp_path: Path,
) -> None:
    first_client = _Client()
    second_client = _Client()
    first_provider = _provider(tmp_path, first_client, monthly_limit=10)
    second_provider = _provider(tmp_path, second_client, monthly_limit=10)
    observed = _FETCHED_AT
    first = first_provider.search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=observed
    )
    second = second_provider.search(
        "YYZ",
        "CDG",
        date(2026, 8, 21),
        fetched_at=observed,
        force_refresh=True,
    )
    assert first.status == "confirmed_offers"
    assert first.search_monthly_used == 10
    assert second.status == "budget_exhausted"
    assert second_client.search_calls == []
    assert _ledger_calls(tmp_path / "private" / "usage.sqlite3") == 10


def test_two_provider_instances_reserve_sqlite_quota_concurrently(
    tmp_path: Path,
) -> None:
    search_barrier = threading.Barrier(8)

    class _SynchronizedSearchClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            params = kwargs.get("params", {})
            if url == SERPAPI_SEARCH_URL and "booking_token" not in params:
                search_barrier.wait(timeout=3)
            return super().get(url, **kwargs)

    first_client = _SynchronizedSearchClient(hourly_limit=8)
    second_client = _SynchronizedSearchClient(hourly_limit=8)
    first_provider = _provider(tmp_path, first_client, monthly_limit=250)
    second_provider = _provider(tmp_path, second_client, monthly_limit=250)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                provider.search,
                "YYZ",
                "LHR",
                date(2026, 8, 20),
                fetched_at=_FETCHED_AT,
            )
            for provider in (first_provider, second_provider)
        ]
        results = [future.result() for future in futures]

    assert {result.status for result in results} == {"rate_limited"}
    assert len(first_client.search_calls) == 4
    assert len(second_client.search_calls) == 4
    assert first_client.booking_calls == []
    assert second_client.booking_calls == []
    assert _ledger_calls(tmp_path / "private" / "usage.sqlite3") == 8
    assert (
        _ledger_calls(
            tmp_path / "private" / "usage.sqlite3",
            scope="hour",
            period_key=_HOUR_BUCKET_KEY,
        )
        == 8
    )


def test_five_minute_cache_and_force_refresh(tmp_path: Path) -> None:
    client = _Client()
    provider = _provider(tmp_path, client)
    fetched_at = _FETCHED_AT
    first = provider.search("YYZ", "LHR", date(2026, 8, 20), fetched_at=fetched_at)
    cached = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=fetched_at + timedelta(minutes=4),
    )
    refreshed = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=fetched_at + timedelta(minutes=4),
        force_refresh=True,
    )
    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert cached.calls_used == 0
    assert cached.offers
    assert all(offer.provider_cache_hit for offer in cached.offers)
    assert all(offer.provider_cache_age_seconds == 240 for offer in cached.offers)
    assert refreshed.cache_hit is False
    assert refreshed.calls_used == 10
    assert len(client.search_calls) == 8
    assert len(client.booking_calls) == 12
    assert all("no_cache" not in call["params"] for call in client.search_calls[:4])
    assert all(call["params"]["no_cache"] == "true" for call in client.search_calls[4:])
    assert all("no_cache" not in call["params"] for call in client.booking_calls[:6])
    assert all(
        call["params"]["no_cache"] == "true" for call in client.booking_calls[6:]
    )


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-07-15 10:54:59 UTC",
        "2026-07-15 12:05:01 UTC",
    ],
)
def test_search_metadata_outside_freshness_window_is_rejected(
    tmp_path: Path,
    created_at: str,
) -> None:
    client = _Client(search_created_at=created_at)

    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "provider_unavailable"
    assert result.offers == ()
    assert client.booking_calls == []


def test_provider_observation_uses_response_receive_clock_not_request_start(
    tmp_path: Path,
) -> None:
    result = _provider(tmp_path, _Client()).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT - timedelta(hours=12),
    )

    assert result.status == "confirmed_offers"
    assert result.offers
    assert all(offer.verified_at == _FETCHED_AT for offer in result.offers)
    assert all(offer.provider_cache_age_seconds == 0 for offer in result.offers)
    assert all(not offer.provider_cache_hit for offer in result.offers)


def test_explicit_cached_search_metadata_is_accepted_and_marked_exactly(
    tmp_path: Path,
) -> None:
    client = _Client(
        search_metadata_status="Cached",
        booking_metadata_status="Cached",
    )
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "confirmed_offers"
    assert result.search_monthly_used == 10
    assert result.offers
    assert all(offer.provider_cache_hit for offer in result.offers)
    assert all(offer.provider_cache_age_seconds == 0 for offer in result.offers)


def test_booking_provider_time_drives_verification_age(tmp_path: Path) -> None:
    cached_client = _Client(booking_created_at="2026-07-15 11:58:59 UTC")
    cached = _provider(tmp_path / "cached", cached_client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )
    assert cached.status == "confirmed_offers"
    assert cached.offers
    assert all(offer.verified_at == _FETCHED_AT - timedelta(seconds=61) for offer in cached.offers)
    assert all(offer.provider_cache_hit for offer in cached.offers)
    assert all(offer.provider_cache_age_seconds == 61 for offer in cached.offers)

    stale_client = _Client(booking_created_at="2026-07-15 10:54:59 UTC")
    stale = _provider(tmp_path / "stale", stale_client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )
    assert stale.status == "provider_unavailable"
    assert stale.offers == ()


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (["1 free carry-on", "1 free checked bag", "2nd checked bag: US$45"], 1),
        (["1 free carry-on", "1st checked bag: US$35"], 0),
        (["1 free carry-on", "1st checked bag: USD 35"], 0),
        (["1 free carry-on", "No checked bags"], 0),
        (["1 free carry-on", "Bag policy varies by seller"], None),
    ],
)
def test_chosen_booking_option_baggage_is_parsed_conservatively(
    tmp_path: Path,
    policy: list[str],
    expected: int | None,
) -> None:
    class _BaggageClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            if url == SERPAPI_SEARCH_URL and "booking_token" in kwargs["params"]:
                response.payload["booking_options"][0]["together"][
                    "baggage_prices"
                ] = policy
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    result = _provider(tmp_path, _BaggageClient()).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "confirmed_offers"
    assert result.offers
    assert all(
        segment.checked_bags_quantity == expected
        for offer in result.offers
        for segment in offer.segments
    )


def test_additional_candidates_prioritize_direct_new_airlines(tmp_path: Path) -> None:
    class _DiverseClient(_Client):
        def get(self, url: str, **kwargs: Any) -> _Response:
            response = super().get(url, **kwargs)
            params = kwargs.get("params", {})
            if (
                url == SERPAPI_SEARCH_URL
                and "booking_token" not in params
                and int(params["travel_class"]) == 1
            ):
                for suffix, code, airline, price in (
                    (2, "UA", "United Airlines", 350),
                    (3, "BA", "British Airways", 360),
                ):
                    flight = _flight("Economy", suffix=suffix)
                    flight["airline"] = airline
                    flight["flight_number"] = f"{code} {500 + suffix}"
                    response.payload["other_flights"].append(
                        {
                            "flights": [flight],
                            "price": price,
                            "type": "One way",
                            "booking_token": f"booking-token-1-{code.lower()}",
                        }
                    )
                response.content = json.dumps(response.payload).encode("utf-8")
            return response

    client = _DiverseClient()
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "confirmed_offers"
    tokens = {call["params"]["booking_token"] for call in client.booking_calls}
    assert {"booking-token-1-ua", "booking-token-1-ba"} <= tokens
    assert "booking-token-1-high" not in tokens


def test_booking_token_verification_runs_in_parallel(tmp_path: Path) -> None:
    class _ParallelClient(_Client):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(6)
            self.thread_ids: set[int] = set()
            self.thread_ids_lock = threading.Lock()

        def get(self, url: str, **kwargs: Any) -> _Response:
            if url == SERPAPI_SEARCH_URL and "booking_token" in kwargs["params"]:
                with self.thread_ids_lock:
                    self.thread_ids.add(threading.get_ident())
                self.barrier.wait(timeout=3)
            return super().get(url, **kwargs)

    client = _ParallelClient()
    result = _provider(tmp_path, client).search(
        "YYZ", "LHR", date(2026, 8, 20), fetched_at=_FETCHED_AT
    )

    assert result.status == "confirmed_offers"
    assert len(client.thread_ids) == 6


@pytest.mark.parametrize(
    ("account_status", "search_status", "expected"),
    [
        (401, 200, "authentication_failed"),
        (200, 401, "authentication_failed"),
        (200, 429, "rate_limited"),
        (503, 200, "provider_unavailable"),
    ],
)
def test_provider_errors_are_distinguished_and_fail_closed(
    tmp_path: Path,
    account_status: int,
    search_status: int,
    expected: str,
) -> None:
    client = _Client(account_status=account_status, search_status=search_status)
    result = _provider(tmp_path, client).search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == expected
    assert result.offers == ()


def test_env_factory_is_fail_closed_and_uses_serpapi_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FLIGHT_OFFER_PROVIDER", raising=False)
    assert isinstance(
        flight_offer_provider_from_env(tmp_path / "usage.sqlite3"),
        NullFlightOfferProvider,
    )

    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "serpapi")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    unconfigured = flight_offer_provider_from_env(tmp_path / "usage.sqlite3")
    assert isinstance(unconfigured, SerpApiFlightOfferProvider)
    assert unconfigured.environment == "disabled"
    result = unconfigured.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        fetched_at=_FETCHED_AT,
    )
    assert result.status == "not_configured"
    assert result.environment == "disabled"

    monkeypatch.setenv("SERPAPI_API_KEY", "key")
    monkeypatch.setenv("SERPAPI_MONTHLY_LIMIT", "999")
    configured = flight_offer_provider_from_env(tmp_path / "usage.sqlite3")
    assert isinstance(configured, SerpApiFlightOfferProvider)
    assert configured.configured is True
    assert configured.environment == "production"
    assert configured.monthly_limit == 250
