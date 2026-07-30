from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from flight_forecaster.alternate_fare_providers import (
    SEARCHAPI_SEARCH_URL,
    _FreeCallLedger,
)
from flight_forecaster.availability import (
    SEARCHAPI_PROVIDER_CODE,
    SERPAPI_ACCOUNT_URL,
    SERPAPI_SEARCH_ARCHIVE_URL,
    SERPAPI_SEARCH_URL,
    _NoRedirectHandler,
)
from flight_forecaster.hotel_prices import (
    HOTEL_PRICE_MAX_RESPONSE_BYTES,
    HotelPriceError,
    HotelPriceValidationError,
    SearchApiHotelPriceProvider,
    SerpApiHotelPriceProvider,
    StrictHotelPriceProviderPool,
    hotel_price_provider_from_env,
)


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


class _OversizedResponse(_Response):
    def __init__(self) -> None:
        super().__init__({"ignored": True})
        self.content = b"x" * (HOTEL_PRICE_MAX_RESPONSE_BYTES + 1)


class _Client:
    def __init__(
        self,
        account: Any,
        search: Any,
        *,
        account_status: int = 200,
        search_status: int = 200,
        archive: Any = None,
    ) -> None:
        self.account = account
        self.search = search
        self.account_status = account_status
        self.search_status = search_status
        self.archive = archive
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if url == SERPAPI_ACCOUNT_URL:
            return self._reply(self._take("account"), self.account_status)
        if url == SERPAPI_SEARCH_URL:
            return self._reply(self._take("search"), self.search_status)
        assert url.startswith("https://serpapi.com/searches/")
        return self._reply(self._take("archive"), 200)

    def _take(self, name: str) -> Any:
        value = getattr(self, name)
        if isinstance(value, list):
            if not value:
                raise AssertionError(f"unexpected extra {name} request")
            return value.pop(0)
        if value is None:
            raise AssertionError(f"unexpected {name} request")
        return value

    @staticmethod
    def _reply(value: Any, status: int) -> _Response:
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, _Response):
            return value
        return _Response(value, status)


class _SearchApiClient:
    def __init__(self, payload: Any, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        assert url == SEARCHAPI_SEARCH_URL
        value = self.payload.pop(0) if isinstance(self.payload, list) else self.payload
        if isinstance(value, BaseException):
            raise value
        return _Response(value, self.status)


NOW = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
CHECK_IN = date(2026, 8, 10)
CHECK_OUT = date(2026, 8, 12)


def test_shared_serpapi_http_client_refuses_redirects() -> None:
    handler = _NoRedirectHandler()

    assert handler.redirect_request(None, None, 302, "Found", {}, "https://evil.test") is None


def _account(*, monthly: int = 10, hourly: int = 2, limit: int = 250) -> dict[str, Any]:
    return {
        "account_status": "Active",
        "plan_renewal_date": "2026-08-15",
        "searches_per_month": limit,
        "this_month_usage": monthly,
        "this_hour_searches": hourly,
        "account_rate_limit_per_hour": 50,
        # Deliberately sensitive account fields must never be persisted.
        "api_key": "provider-returned-secret",
        "account_email": "private@example.test",
    }


def _search_payload(
    *,
    properties: list[Any] | None = None,
    query: str = "Toronto",
) -> dict[str, Any]:
    if properties is None:
        properties = [
            {
                "type": "hotel",
                "name": "Harbour Test Hotel",
                "description": "Waterfront rooms and a pool.",
                "link": "https://hotel.example.test/book",
                "property_token": "do-not-persist-property-token",
                "gps_coordinates": {"latitude": 43.6426, "longitude": -79.3871},
                "extracted_hotel_class": 4,
                "overall_rating": 4.6,
                "reviews": 1234,
                "rate_per_night": {"extracted_lowest": 210},
                "total_rate": {"extracted_lowest": 420},
                "prices": [
                    {
                        "source": "Safe Hotel Seller",
                        "rate_per_night": {"extracted_lowest": 190},
                        "total_rate": {"extracted_lowest": 380},
                        "free_cancellation": True,
                        "link": "https://seller.example.test/private-provider-link",
                    },
                    {
                        "source": "Higher Seller",
                        "rate_per_night": {"extracted_lowest": 205},
                        "total_rate": {"extracted_lowest": 410},
                    },
                ],
                "amenities": ["Free Wi-Fi", "Pool", "Free Wi-Fi"],
                "serpapi_property_details_link": (
                    "https://serpapi.com/search.json?property_token=do-not-persist"
                ),
            },
            {
                "type": "hostel",
                "name": "City Hostel",
                "link": "https://serpapi.com/search.json?property_token=secret",
                "property_token": "another-secret-token",
                "gps_coordinates": {"latitude": 43.65, "longitude": -79.38},
                "hotel_class": "2-star hotel",
                "rate_per_night": {"extracted_lowest": 80},
                "total_rate": {"extracted_lowest": 160},
            },
        ]
    return {
        "search_metadata": {"id": "safe-search-id", "status": "Success"},
        "search_parameters": {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": CHECK_IN.isoformat(),
            "check_out_date": CHECK_OUT.isoformat(),
            "adults": 2,
            "currency": "USD",
            "hl": "en",
        },
        "properties": properties,
    }


def _property_detail_payload(
    *,
    name: str = "Harbour Test Hotel",
    latitude: float = 43.6426,
    longitude: float = -79.3871,
    property_token: str = "do-not-persist-property-token",
) -> dict[str, Any]:
    return {
        "search_metadata": {"id": "safe-detail-search-id", "status": "Success"},
        "search_parameters": {
            "engine": "google_hotels",
            "property_token": property_token,
            "check_in_date": CHECK_IN.isoformat(),
            "check_out_date": CHECK_OUT.isoformat(),
            "adults": 2,
            "currency": "USD",
            "hl": "en",
        },
        "type": "hotel",
        "name": name,
        "address": "100 Harbour Street, Toronto, ON",
        "phone": "+1 416 555 0100",
        "check_in_time": "3:00 PM",
        "check_out_time": "11:00 AM",
        "thumbnail": "https://images.example.test/hotel-thumb.jpg",
        "images": [
            {
                "thumbnail": "https://images.example.test/hotel-1-thumb.jpg",
                "original_image": "https://images.example.test/hotel-1.jpg",
            },
            {
                "thumbnail": "https://images.example.test/hotel-2-thumb.jpg",
                "original_image": "https://images.example.test/hotel-2.jpg",
            },
        ],
        "property_token": property_token,
        "gps_coordinates": {"latitude": latitude, "longitude": longitude},
        "overall_rating": 4.7,
        "reviews": 1500,
        "featured_prices": [
            {
                "source": "Hotel Direct",
                "official": True,
                "link": "https://hotel.example.test/rooms",
                "rooms": [
                    {
                        "name": "Deluxe King",
                        "num_guests": 2,
                        "rates": [
                            {
                                "link": "https://hotel.example.test/rooms/deluxe",
                                "num_guests": 2,
                                "free_cancellation": True,
                                "free_cancellation_until_date": "Aug 8",
                                "free_cancellation_until_time": "6:00 PM",
                                "breakfast_included": True,
                                "beds": [{"type": "King", "count": 1}],
                                "rate_per_night": {
                                    "extracted_lowest": 240,
                                    "extracted_before_taxes_fees": 210,
                                },
                                "total_rate": {
                                    "extracted_lowest": 480,
                                    "extracted_before_taxes_fees": 420,
                                },
                                "inclusions": ["Breakfast", "Free Wi-Fi"],
                            }
                        ],
                    },
                    {
                        "name": "Twin City View",
                        "link": "https://hotel.example.test/rooms/twin",
                        "num_guests": 2,
                        "rate_per_night": {"extracted_lowest": 220},
                        "total_rate": {"extracted_lowest": 440},
                    },
                ],
            },
            {
                "source": "Booking Platform",
                "official": False,
                "rooms": [
                    {
                        "name": "Deluxe King",
                        "rates": [
                            {
                                "rate_per_night": {"extracted_lowest": 235},
                                "total_rate": {"extracted_lowest": 470},
                            }
                        ],
                    }
                ],
            },
        ],
        "other_reviews": [
            {
                "source": "Tripadvisor",
                "source_rating": {"score": 4.5, "max_score": 5},
                "reviews": 321,
                "user_review": {
                    "username": "Verified guest",
                    "date": "2 weeks ago",
                    "rating": {"score": 5, "max_score": 5},
                    "comment": "Quiet room and helpful staff.",
                    "link": "https://reviews.example.test/tripadvisor/123",
                },
            },
            {
                "source": "Trip.com",
                "source_rating": {"score": 4.2, "max_score": 5},
                "reviews": 98,
                "user_review": {
                    "comment": "Convenient location.",
                    "link": "https://reviews.example.test/trip/456",
                },
            },
        ],
    }


def _with_search_status(
    payload: dict[str, Any],
    status: str,
    search_id: str,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    copied["search_metadata"] = {"id": search_id, "status": status}
    return copied


def _provider(
    tmp_path: Path,
    client: _Client,
    *,
    now: datetime = NOW,
    api_key: str | None = "test-serpapi-secret",
    monthly_limit: int = 250,
    poll_delays_seconds: tuple[float, ...] = (0.0,),
) -> SerpApiHotelPriceProvider:
    return SerpApiHotelPriceProvider(
        api_key,
        usage_path=tmp_path / "artifacts" / "runtime" / "serpapi-usage.sqlite3",
        monthly_limit=monthly_limit,
        client=client,
        now_provider=lambda: now,
        poll_delays_seconds=poll_delays_seconds,
        sleep_provider=lambda _: None,
    )


def _search(provider: SerpApiHotelPriceProvider, **changes: Any):
    values = {
        "city_query": "Toronto",
        "destination_iata": "YYZ",
        "check_in": CHECK_IN,
        "check_out": CHECK_OUT,
        "adults": 2,
        "language": "en",
        "explicit": True,
    }
    values.update(changes)
    return provider.search(**values)


def _ledger_calls(path: Path, scope: str, period: str) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT calls FROM serpapi_quota_usage WHERE scope = ? AND period_key = ?",
            (scope, period),
        ).fetchone()
    return int(row[0]) if row else 0


def test_live_search_uses_exact_google_hotels_params_and_sanitizes_cache(
    tmp_path: Path,
) -> None:
    client = _Client(_account(), _search_payload())
    provider = _provider(tmp_path, client)

    result = _search(provider)

    assert result.status == "available"
    assert result.cache_hit is False
    assert result.calls_reserved == 1
    assert result.quota_monthly_used == 11
    assert result.quota_hourly_used == 3
    assert len(result.offers) == 2
    hotel = result.offers[0]
    assert hotel.hotel_id.startswith("gh_")
    assert hotel.latitude == 43.6426
    assert hotel.longitude == -79.3871
    assert hotel.nightly_price == 190
    assert hotel.total_price == 380
    assert hotel.price_source == "Safe Hotel Seller"
    assert hotel.free_cancellation is True
    assert hotel.amenities == ("Free Wi-Fi", "Pool")
    assert hotel.website_url == "https://hotel.example.test/book"
    assert result.offers[1].website_url.startswith("https://www.google.com/travel/search?")

    assert [call["url"] for call in client.calls] == [
        SERPAPI_ACCOUNT_URL,
        SERPAPI_SEARCH_URL,
    ]
    params = client.calls[1]["params"]
    assert params == {
        "engine": "google_hotels",
        "q": "Toronto",
        "check_in_date": "2026-08-10",
        "check_out_date": "2026-08-12",
        "adults": 2,
        "currency": "USD",
        "hl": "en",
        "api_key": "test-serpapi-secret",
    }
    assert "no_cache" not in params
    assert client.calls[0]["timeout"] <= 8
    assert client.calls[1]["timeout"] <= 30

    ledger = provider.usage_path
    persisted = ledger.read_bytes()
    for secret in (
        b"test-serpapi-secret",
        b"provider-returned-secret",
        b"private@example.test",
        b"do-not-persist-property-token",
        b"another-secret-token",
        b"property_token",
        b"serpapi.com",
    ):
        assert secret not in persisted
    assert _ledger_calls(ledger, "billing_cycle", "renewal:2026-08-15") == 11
    assert _ledger_calls(ledger, "hour", "2026-07-21T14") == 3


def test_price_evidence_never_cross_fills_between_seller_and_property(
    tmp_path: Path,
) -> None:
    priced = {
        "type": "hotel",
        "name": "Single Seller Evidence",
        "property_token": "single-seller-token",
        "gps_coordinates": {"latitude": 43.64, "longitude": -79.38},
        "rate_per_night": {"extracted_lowest": 155},
        "total_rate": {"extracted_lowest": 620},
        "free_cancellation": True,
        "prices": [
            {
                "source": "Nightly Only Seller",
                "rate_per_night": {"extracted_lowest": 120},
                # No total or cancellation evidence from this seller.
            }
        ],
    }
    unpriced = {
        "type": "hotel",
        "name": "No Positive Price",
        "property_token": "unpriced-token",
        "gps_coordinates": {"latitude": 43.65, "longitude": -79.37},
        "rate_per_night": {"extracted_lowest": 0},
        "total_rate": {"extracted_lowest": "unknown"},
        "prices": [{"source": "No Price Seller"}],
    }
    provider = _provider(
        tmp_path,
        _Client(_account(monthly=0, hourly=0), _search_payload(properties=[priced, unpriced])),
    )

    result = _search(provider)

    assert len(result.offers) == 1
    offer = result.offers[0]
    assert offer.name == "Single Seller Evidence"
    assert offer.price_source == "Nightly Only Seller"
    assert offer.nightly_price == 120
    assert offer.total_price is None
    assert offer.free_cancellation is None


def test_safe_properties_without_positive_prices_return_no_real_price_results(
    tmp_path: Path,
) -> None:
    unpriced = {
        "type": "hostel",
        "name": "Real Place But No Quote",
        "property_token": "no-quote-token",
        "gps_coordinates": {"latitude": 43.65, "longitude": -79.37},
        "rate_per_night": {"extracted_lowest": -1},
    }
    provider = _provider(
        tmp_path,
        _Client(_account(monthly=0, hourly=0), _search_payload(properties=[unpriced])),
    )

    result = _search(provider)

    assert result.status == "no_results"
    assert result.offers == ()


def test_cache_hit_and_detail_use_no_network_and_no_quota(tmp_path: Path) -> None:
    client = _Client(_account(monthly=0, hourly=0), _search_payload())
    provider = _provider(tmp_path, client)
    first = _search(provider)
    calls_after_first = list(client.calls)

    cached = _search(provider, language="zh-CN")
    detail = provider.detail(
        first.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="zh-CN",
    )

    assert cached.cache_hit is True
    assert cached.calls_reserved == 0
    assert cached.quota_monthly_used is None
    assert detail == first.offers[0]
    assert client.calls == calls_after_first
    assert _ledger_calls(
        provider.usage_path, "billing_cycle", "renewal:2026-08-15"
    ) == 1


def test_detail_requires_exact_stay_cache_key(tmp_path: Path) -> None:
    provider = _provider(tmp_path, _Client(_account(), _search_payload()))
    result = _search(provider)

    assert (
        provider.detail(
            result.offers[0].hotel_id,
            "Toronto",
            "YYZ",
            CHECK_IN,
            CHECK_OUT + timedelta(days=1),
            adults=2,
            language="en",
        )
        is None
    )


def test_explicit_action_is_required_before_any_network_or_ledger(tmp_path: Path) -> None:
    client = _Client(_account(), _search_payload())
    provider = _provider(tmp_path, client)

    with pytest.raises(HotelPriceValidationError) as error_info:
        _search(provider, explicit=False)

    assert error_info.value.code == "validation_error"
    assert client.calls == []
    assert not provider.usage_path.exists()


def test_missing_key_fails_without_network_or_quota(tmp_path: Path) -> None:
    client = _Client(_account(), _search_payload())
    provider = _provider(tmp_path, client, api_key=None)

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == "not_configured"
    assert client.calls == []
    assert not provider.usage_path.exists()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"city_query": "\x00"}, "city"),
        ({"destination_iata": "Y1Z"}, "airport"),
        ({"check_in": "2026-07-20"}, "370"),
        ({"check_out": CHECK_IN}, "dates"),
        ({"check_out": "2027-07-30"}, "370"),
        ({"adults": 0}, "Adults"),
        ({"adults": 9}, "Adults"),
        ({"language": "fr"}, "Language"),
    ],
)
def test_stay_validation_prevents_network(
    tmp_path: Path,
    changes: dict[str, Any],
    message: str,
) -> None:
    client = _Client(_account(), _search_payload())
    provider = _provider(tmp_path, client)

    with pytest.raises(HotelPriceValidationError, match=message):
        _search(provider, **changes)

    assert client.calls == []


@pytest.mark.parametrize(
    ("account", "monthly_limit", "expected_scope"),
    [
        (_account(monthly=250, hourly=0), 250, "monthly"),
        (_account(monthly=0, hourly=50), 250, "hourly"),
    ],
)
def test_shared_quota_exhaustion_blocks_business_request(
    tmp_path: Path,
    account: dict[str, Any],
    monthly_limit: int,
    expected_scope: str,
) -> None:
    client = _Client(account, _search_payload())
    provider = _provider(tmp_path, client, monthly_limit=monthly_limit)

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == "quota_exhausted"
    assert error_info.value.quota_scope == expected_scope
    assert [call["url"] for call in client.calls] == [SERPAPI_ACCOUNT_URL]


def test_existing_flight_ledger_reservation_cannot_be_bypassed(tmp_path: Path) -> None:
    client = _Client(_account(monthly=0, hourly=0, limit=1), _search_payload())
    provider = _provider(tmp_path, client, monthly_limit=1)
    first = _search(provider)
    assert first.calls_reserved == 1

    # A different stay misses the hotel cache but sees the same shared quota row.
    client.search = _search_payload(properties=[])
    client.search["search_parameters"]["check_in_date"] = "2026-08-11"
    client.search["search_parameters"]["check_out_date"] = "2026-08-13"
    with pytest.raises(HotelPriceError) as error_info:
        _search(
            provider,
            check_in=date(2026, 8, 11),
            check_out=date(2026, 8, 13),
        )

    assert error_info.value.code == "quota_exhausted"
    assert [call["url"] for call in client.calls].count(SERPAPI_SEARCH_URL) == 1


@pytest.mark.parametrize(
    ("account_status", "search_status", "expected"),
    [
        (401, 200, "authentication_failed"),
        (429, 200, "rate_limited"),
        (503, 200, "provider_unavailable"),
        (200, 400, "provider_error"),
        (200, 429, "rate_limited"),
        (200, 503, "provider_unavailable"),
    ],
)
def test_http_errors_have_safe_categories(
    tmp_path: Path,
    account_status: int,
    search_status: int,
    expected: str,
) -> None:
    client = _Client(
        {"provider_error": "raw secret account error"},
        {"provider_error": "raw secret search error"},
        account_status=account_status,
        search_status=search_status,
    )
    if account_status == 200:
        client.account = _account(monthly=0, hourly=0)
    provider = _provider(tmp_path, client)

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == expected
    assert "raw secret" not in str(error_info.value)


def test_transport_timeout_is_classified_without_raw_error(tmp_path: Path) -> None:
    client = _Client(TimeoutError("secret socket details"), _search_payload())
    provider = _provider(tmp_path, client)

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == "provider_unavailable"
    assert "secret socket" not in str(error_info.value)


def test_oversized_response_is_rejected(tmp_path: Path) -> None:
    client = _Client(_account(monthly=0, hourly=0), _OversizedResponse())
    provider = _provider(tmp_path, client)

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == "response_invalid"


def test_mismatched_echo_is_response_invalid(tmp_path: Path) -> None:
    mismatched = _search_payload()
    mismatched["search_parameters"]["q"] = "Vancouver"
    provider = _provider(
        tmp_path,
        _Client(_account(monthly=0, hourly=0), mismatched),
    )
    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)
    assert error_info.value.code == "response_invalid"


def test_queued_search_polls_strict_archive_without_another_reservation(
    tmp_path: Path,
) -> None:
    search_id = "hotelSearch_001"
    queued = _with_search_status(_search_payload(), "Queued", search_id)
    still_processing = _with_search_status(_search_payload(), "Processing", search_id)
    archived_success = _with_search_status(_search_payload(), "Success", search_id)
    client = _Client(
        _account(monthly=0, hourly=0),
        queued,
        archive=[still_processing, archived_success],
    )
    provider = _provider(tmp_path, client, poll_delays_seconds=(0.0, 0.0))

    result = _search(provider)

    archive_url = SERPAPI_SEARCH_ARCHIVE_URL.format(search_id=search_id)
    assert result.status == "available"
    assert result.calls_reserved == 1
    assert [call["url"] for call in client.calls] == [
        SERPAPI_ACCOUNT_URL,
        SERPAPI_SEARCH_URL,
        archive_url,
        archive_url,
    ]
    assert client.calls[2]["params"] == {"api_key": "test-serpapi-secret"}
    assert client.calls[2]["timeout"] <= 2
    assert _ledger_calls(
        provider.usage_path, "billing_cycle", "renewal:2026-08-15"
    ) == 1


def test_pending_search_gets_at_most_one_quota_reserved_resubmission(
    tmp_path: Path,
) -> None:
    first_id = "hotelSearch_101"
    queued = _with_search_status(_search_payload(), "Queued", first_id)
    pending = _with_search_status(_search_payload(), "Processing", first_id)
    retried_success = _with_search_status(_search_payload(), "Success", "hotelSearch_102")
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=0, hourly=0)],
        [queued, retried_success],
        archive=[pending],
    )
    provider = _provider(tmp_path, client)

    result = _search(provider)

    assert result.calls_reserved == 2
    assert [call["url"] for call in client.calls].count(SERPAPI_SEARCH_URL) == 2
    assert [call["url"] for call in client.calls].count(SERPAPI_ACCOUNT_URL) == 2
    assert _ledger_calls(
        provider.usage_path, "billing_cycle", "renewal:2026-08-15"
    ) == 2
    for call in (item for item in client.calls if item["url"] == SERPAPI_SEARCH_URL):
        assert "no_cache" not in call["params"]


def test_twice_pending_is_guarded_without_ids_urls_or_repeat_quota(
    tmp_path: Path,
) -> None:
    first_id = "hotelSearch_201"
    second_id = "hotelSearch_202"
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=0, hourly=0)],
        [
            _with_search_status(_search_payload(), "Queued", first_id),
            _with_search_status(_search_payload(), "Processing", second_id),
        ],
        archive=[
            _with_search_status(_search_payload(), "Processing", first_id),
            _with_search_status(_search_payload(), "Queued", second_id),
        ],
    )
    provider = _provider(tmp_path, client)

    with pytest.raises(HotelPriceError) as first_error:
        _search(provider)
    calls_after_bounded_workflow = list(client.calls)
    with pytest.raises(HotelPriceError) as guarded_error:
        _search(provider, language="zh-CN")

    assert first_error.value.code == "provider_processing"
    assert guarded_error.value.code == "provider_processing"
    assert client.calls == calls_after_bounded_workflow
    assert [call["url"] for call in client.calls].count(SERPAPI_SEARCH_URL) == 2
    assert _ledger_calls(
        provider.usage_path, "billing_cycle", "renewal:2026-08-15"
    ) == 2
    persisted = provider.usage_path.read_bytes()
    assert first_id.encode() not in persisted
    assert second_id.encode() not in persisted
    assert b"/searches/" not in persisted


def test_pending_search_id_and_archive_identity_are_strictly_validated(
    tmp_path: Path,
) -> None:
    unsafe_id = "../../account"
    unsafe = _with_search_status(_search_payload(), "Queued", unsafe_id)
    unsafe_client = _Client(_account(monthly=0, hourly=0), unsafe)
    provider = _provider(tmp_path / "unsafe", unsafe_client)
    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)
    assert error_info.value.code == "response_invalid"
    assert len(unsafe_client.calls) == 2

    expected_id = "hotelSearch_301"
    mismatched_id = "hotelSearch_302"
    mismatch_client = _Client(
        _account(monthly=0, hourly=0),
        _with_search_status(_search_payload(), "Queued", expected_id),
        archive=[_with_search_status(_search_payload(), "Processing", mismatched_id)],
    )
    provider = _provider(tmp_path / "mismatch", mismatch_client)
    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)
    assert error_info.value.code == "response_invalid"
    assert mismatch_client.calls[2]["url"] == SERPAPI_SEARCH_ARCHIVE_URL.format(
        search_id=expected_id
    )


def test_controlled_retry_cannot_bypass_shared_quota(tmp_path: Path) -> None:
    search_id = "hotelSearch_401"
    client = _Client(
        [_account(monthly=0, hourly=0, limit=1), _account(monthly=0, hourly=0, limit=1)],
        [_with_search_status(_search_payload(), "Queued", search_id)],
        archive=[_with_search_status(_search_payload(), "Processing", search_id)],
    )
    provider = _provider(tmp_path, client, monthly_limit=1)

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == "quota_exhausted"
    assert [call["url"] for call in client.calls].count(SERPAPI_SEARCH_URL) == 1
    assert [call["url"] for call in client.calls].count(SERPAPI_ACCOUNT_URL) == 2
    assert _ledger_calls(
        provider.usage_path, "billing_cycle", "renewal:2026-08-15"
    ) == 1


def test_nonempty_but_unusable_properties_are_not_reported_as_no_results(
    tmp_path: Path,
) -> None:
    payload = _search_payload(properties=[{"name": "Missing coordinates"}])
    provider = _provider(tmp_path, _Client(_account(monthly=0, hourly=0), payload))

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    assert error_info.value.code == "response_invalid"


def test_real_empty_result_is_cached_for_one_hour(tmp_path: Path) -> None:
    client = _Client(_account(monthly=0, hourly=0), _search_payload(properties=[]))
    provider = _provider(tmp_path, client)

    first = _search(provider)
    second = _search(provider)

    assert first.status == "no_results"
    assert first.calls_reserved == 1
    assert second.status == "no_results"
    assert second.cache_hit is True
    assert second.calls_reserved == 0
    assert len(client.calls) == 2


def test_local_cache_expires_after_one_hour_and_then_reserves_again(tmp_path: Path) -> None:
    clock = [NOW]
    client = _Client(_account(monthly=0, hourly=0), _search_payload())
    provider = SerpApiHotelPriceProvider(
        "test-serpapi-secret",
        usage_path=tmp_path / "shared.sqlite3",
        client=client,
        now_provider=lambda: clock[0],
    )

    first = _search(provider)
    clock[0] = NOW + timedelta(minutes=59)
    cached = _search(provider)
    clock[0] = NOW + timedelta(hours=1, seconds=1)
    refreshed = _search(provider)

    assert first.calls_reserved == 1
    assert cached.cache_hit is True
    assert refreshed.cache_hit is False
    assert refreshed.calls_reserved == 1
    assert [call["url"] for call in client.calls] == [
        SERPAPI_ACCOUNT_URL,
        SERPAPI_SEARCH_URL,
        SERPAPI_ACCOUNT_URL,
        SERPAPI_SEARCH_URL,
    ]
    assert _ledger_calls(
        provider.usage_path, "billing_cycle", "renewal:2026-08-15"
    ) == 2


def test_env_factory_uses_only_existing_serpapi_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "factory-key")
    monkeypatch.setenv("SERPAPI_MONTHLY_LIMIT", "99")
    path = tmp_path / "shared.sqlite3"

    provider = hotel_price_provider_from_env(path)

    assert provider.configured is True
    assert provider.usage_path == path


def _searchapi_payload(*, properties: list[Any] | None = None) -> dict[str, Any]:
    return {
        "search_metadata": {"id": "searchapi-hotel-001", "status": "Success"},
        "search_parameters": {
            "engine": "google_hotels",
            "q": "Toronto",
            "check_in_date": CHECK_IN.isoformat(),
            "check_out_date": CHECK_OUT.isoformat(),
            "adults": 2,
            "currency": "USD",
            "hl": "en",
        },
        "properties": (
            [
                {
                    "type": "hotel",
                    "property_token": "searchapi-property-token-not-persisted",
                    "data_id": "0x123:0x456",
                    "name": "SearchAPI Harbour Hotel",
                    "link": "https://hotel.example.test/searchapi-book",
                    "gps_coordinates": {
                        "latitude": 43.641,
                        "longitude": -79.381,
                    },
                    "price_per_night": {
                        "extracted_price": 175,
                        "extracted_price_before_taxes": 150,
                    },
                    "total_price": {
                        "extracted_price": 350,
                        "extracted_price_before_taxes": 300,
                    },
                    "rating": 4.4,
                    "reviews": 321,
                }
            ]
            if properties is None
            else properties
        ),
    }


def _searchapi_provider(
    tmp_path: Path,
    client: _SearchApiClient,
    *,
    api_key: str | None = "searchapi-secret-not-for-output",
    limit: int = 100,
) -> SearchApiHotelPriceProvider:
    return SearchApiHotelPriceProvider(
        api_key,
        usage_path=tmp_path / "shared.sqlite3",
        lifetime_limit=limit,
        client=client,
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )


def test_env_factory_shares_searchapi_hotel_and_flight_lifetime_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setenv("SEARCHAPI_API_KEY", "configured-secret-not-for-output")
    monkeypatch.setenv("SEARCHAPI_LIFETIME_LIMIT", "100")
    monkeypatch.setenv("SEARCHAPI_MONTHLY_LIMIT", "100")
    serpapi_path = tmp_path / "runtime" / "serpapi-usage.sqlite3"
    alternate_path = serpapi_path.with_name("alternate-provider-usage.sqlite3")
    ledger = _FreeCallLedger(alternate_path)
    assert ledger.reserve(
        SEARCHAPI_PROVIDER_CODE,
        100,
        hard_limit=100,
        require_all=True,
    ) == 100
    provider = hotel_price_provider_from_env(serpapi_path)
    searchapi = provider.providers[1]
    client = _SearchApiClient(_searchapi_payload())
    searchapi._client = client

    with pytest.raises(HotelPriceError) as error_info:
        _search(provider)

    searchapi_run = next(
        run
        for run in error_info.value.provider_runs
        if run.provider_code == "searchapi_google_hotels"
    )
    assert searchapi.usage_path == alternate_path
    assert error_info.value.code == "quota_exhausted"
    assert searchapi_run.status == "quota_exhausted"
    assert searchapi_run.error_code == "quota_exhausted"
    assert searchapi_run.calls_reserved == 0
    assert client.calls == []


def test_searchapi_hotel_search_is_strict_and_uses_shared_lifetime_ledger(
    tmp_path: Path,
) -> None:
    client = _SearchApiClient(_searchapi_payload())
    provider = _searchapi_provider(tmp_path, client)

    result = _search(provider)

    assert result.status == "available"
    assert result.provider_code == "searchapi_google_hotels"
    assert result.provider_name == "SearchAPI.io Google Hotels"
    assert result.calls_reserved == 1
    assert result.quota_monthly_used == 1
    assert result.quota_monthly_limit == 100
    assert result.quota_scope == "lifetime"
    assert result.quota_hourly_used is None
    assert result.quota_hourly_limit is None
    assert result.offers[0].nightly_price == 175
    assert result.offers[0].total_price == 350
    assert result.offers[0].price_source == "SearchAPI.io Google Hotels"
    assert client.calls[0]["url"] == SEARCHAPI_SEARCH_URL
    assert "api_key" not in client.calls[0]["params"]
    assert client.calls[0]["headers"]["Authorization"].startswith("Bearer ")
    persisted = provider.usage_path.read_bytes()
    assert b"searchapi-secret-not-for-output" not in persisted
    assert b"searchapi-property-token-not-persisted" not in persisted


def test_transient_hotel_failure_gets_exactly_one_reserved_retry(
    tmp_path: Path,
) -> None:
    client = _SearchApiClient(
        [
            TimeoutError("sensitive upstream timeout"),
            _searchapi_payload(),
        ]
    )
    provider = _searchapi_provider(tmp_path, client)

    result = _search(provider)

    assert result.status == "available"
    assert result.calls_reserved == 2
    assert len(client.calls) == 2
    assert all(call["url"] == SEARCHAPI_SEARCH_URL for call in client.calls)


def test_pool_fails_over_from_serpapi_quota_to_searchapi_strict_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.sqlite3"
    serp_client = _Client(_account(monthly=250, hourly=0), _search_payload())
    serp = SerpApiHotelPriceProvider(
        "serp-secret-not-for-output",
        usage_path=path,
        client=serp_client,
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )
    search_client = _SearchApiClient(_searchapi_payload())
    searchapi = SearchApiHotelPriceProvider(
        "search-secret-not-for-output",
        usage_path=path,
        client=search_client,
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )
    pool = StrictHotelPriceProviderPool((serp, searchapi))

    result = _search(pool)

    assert result.status == "available"
    assert result.provider_code == "searchapi_google_hotels"
    assert [run.status for run in result.provider_runs] == [
        "quota_exhausted",
        "available",
    ]
    assert result.provider_runs[0].error_code == "quota_exhausted"
    assert len(search_client.calls) == 1
    serialized = json.dumps(
        [run.as_safe_dict() for run in result.provider_runs]
    )
    assert "serp-secret-not-for-output" not in serialized
    assert "search-secret-not-for-output" not in serialized


def test_pool_fails_over_from_serpapi_authentication_to_searchapi(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.sqlite3"
    serp = SerpApiHotelPriceProvider(
        "bad-serp-key",
        usage_path=path,
        client=_Client(
            {"error": "secret authentication detail"},
            _search_payload(),
            account_status=401,
        ),
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )
    searchapi = SearchApiHotelPriceProvider(
        "valid-search-key",
        usage_path=path,
        client=_SearchApiClient(_searchapi_payload()),
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )

    result = _search(StrictHotelPriceProviderPool((serp, searchapi)))

    assert result.status == "available"
    assert [run.status for run in result.provider_runs] == [
        "authentication_failed",
        "available",
    ]
    assert "secret authentication detail" not in json.dumps(
        [run.as_safe_dict() for run in result.provider_runs]
    )


def test_provider_run_reports_reserved_call_after_safe_provider_failure(
    tmp_path: Path,
) -> None:
    provider = SerpApiHotelPriceProvider(
        "serp-key",
        usage_path=tmp_path / "shared.sqlite3",
        client=_Client(
            _account(monthly=0, hourly=0),
            {"error": "secret upstream response"},
            search_status=503,
        ),
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )

    with pytest.raises(HotelPriceError) as error_info:
        _search(StrictHotelPriceProviderPool((provider,)))

    assert error_info.value.provider_runs[0].calls_reserved == 2
    assert error_info.value.provider_runs[0].status == "provider_unavailable"
    assert "secret upstream response" not in str(error_info.value)


def test_searchapi_property_detail_accepts_strict_property_engine_echo(
    tmp_path: Path,
) -> None:
    detail = {
        "search_metadata": {"id": "searchapi-detail-001", "status": "Success"},
        "search_parameters": {
            "engine": "google_hotels_property",
            "property_token": "searchapi-property-token-not-persisted",
            "check_in_date": CHECK_IN.isoformat(),
            "check_out_date": CHECK_OUT.isoformat(),
            "adults": 2,
            "currency": "USD",
            "hl": "en",
        },
        "property": {
            "type": "hotel",
            "property_token": "searchapi-property-token-not-persisted",
            "name": "SearchAPI Harbour Hotel",
            "link": "https://hotel.example.test/searchapi-book",
            "gps_coordinates": {"latitude": 43.641, "longitude": -79.381},
            "price_per_night": {"extracted_price": 175},
            "total_price": {"extracted_price": 350},
            "rating": 4.5,
            "reviews": 400,
            "featured_offers": [
                {
                    "source": "Hotel Direct",
                    "is_official": True,
                    "link": "https://hotel.example.test/searchapi-book",
                    "rooms": [
                        {
                            "name": "King Room",
                            "price_per_night": {
                                "extracted_price": 180,
                                "extracted_price_before_taxes": 155,
                            },
                            "total_price": {
                                "extracted_price": 360,
                                "extracted_price_before_taxes": 310,
                            },
                            "has_free_cancellation": True,
                        }
                    ],
                }
            ],
        },
    }
    client = _SearchApiClient([_searchapi_payload(), detail])
    provider = _searchapi_provider(tmp_path, client)
    listing = _search(provider)

    offer = provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert offer is not None
    assert offer.room_rates_status == "available"
    assert offer.room_rates[0].room_name == "King Room"
    assert offer.room_rates[0].total_price == 360
    assert client.calls[1]["params"]["engine"] == "google_hotels_property"


def test_pool_reports_true_no_results_only_after_all_configured_sources_complete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.sqlite3"
    serp = SerpApiHotelPriceProvider(
        "serp-key",
        usage_path=path,
        client=_Client(_account(monthly=0, hourly=0), _search_payload(properties=[])),
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )
    searchapi = SearchApiHotelPriceProvider(
        "search-key",
        usage_path=path,
        client=_SearchApiClient(_searchapi_payload(properties=[])),
        now_provider=lambda: NOW,
        sleep_provider=lambda _: None,
    )
    pool = StrictHotelPriceProviderPool((serp, searchapi))

    result = _search(pool)

    assert result.status == "no_results"
    assert result.provider_code == "strict_hotel_price_failover"
    assert [run.status for run in result.provider_runs] == [
        "no_results",
        "no_results",
    ]


def test_exact_property_is_not_reported_missing_when_one_source_failed() -> None:
    class _ExactProvider:
        configured = True

        def __init__(self, code: str, outcome: object) -> None:
            self.provider_code = code
            self.provider_name = code
            self.outcome = outcome

        def exact_property_detail(self, *args: Any, **kwargs: Any) -> Any:
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    pool = StrictHotelPriceProviderPool(
        (
            _ExactProvider("completed_empty", None),  # type: ignore[arg-type]
            _ExactProvider(
                "failed_source",
                HotelPriceError(
                    "provider_unavailable",
                    "sanitized provider failure",
                ),
            ),  # type: ignore[arg-type]
        )
    )

    with pytest.raises(HotelPriceError) as error_info:
        pool.exact_property_detail()

    assert error_info.value.code == "provider_unavailable"


def test_explicit_detail_returns_real_room_rates_and_platform_reviews(
    tmp_path: Path,
) -> None:
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [_search_payload(), _property_detail_payload()],
    )
    provider = _provider(tmp_path, client)
    listing = _search(provider)

    detail = provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    assert detail.room_rates_status == "available"
    assert detail.review_sources_status == "available"
    assert detail.address == "100 Harbour Street, Toronto, ON"
    assert detail.phone == "+1 416 555 0100"
    assert detail.check_in_time == "3:00 PM"
    assert detail.check_out_time == "11:00 AM"
    assert detail.thumbnail == "https://images.example.test/hotel-thumb.jpg"
    assert detail.images == (
        "https://images.example.test/hotel-1.jpg",
        "https://images.example.test/hotel-2.jpg",
    )
    assert len(detail.room_rates) == 3
    direct = next(
        item
        for item in detail.room_rates
        if item.room_name == "Deluxe King" and item.source == "Hotel Direct"
    )
    assert direct.nightly_price == 240
    assert direct.total_price == 480
    assert direct.nightly_before_taxes == 210
    assert direct.total_before_taxes == 420
    assert direct.beds == ("1 × King",)
    assert direct.breakfast_included is True
    assert direct.free_cancellation is True
    assert direct.free_cancellation_until == "Aug 8 6:00 PM"
    assert {item.source for item in detail.review_sources} == {
        "Google",
        "Tripadvisor",
        "Trip.com",
    }
    tripadvisor = next(
        item for item in detail.review_sources if item.source == "Tripadvisor"
    )
    assert tripadvisor.score == 4.5
    assert tripadvisor.review_count == 321
    assert tripadvisor.sample_comment == "Quiet room and helpful staff."
    detail_call = [
        call for call in client.calls if call["url"] == SERPAPI_SEARCH_URL
    ][1]
    assert detail_call["params"]["property_token"] == "do-not-persist-property-token"
    assert "q" not in detail_call["params"]
    assert _ledger_calls(
        provider.usage_path,
        "billing_cycle",
        "renewal:2026-08-15",
    ) == 2
    persisted = provider.usage_path.read_bytes()
    assert b"do-not-persist-property-token" not in persisted
    assert b"property_token" not in persisted
    assert b"Quiet room and helpful staff." in persisted


def test_enriched_detail_is_cache_only_after_first_explicit_fetch(tmp_path: Path) -> None:
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [_search_payload(), _property_detail_payload()],
    )
    provider = _provider(tmp_path, client)
    listing = _search(provider)
    first = provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )
    calls = list(client.calls)

    second = provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="zh-cn",
        explicit=True,
    )

    assert second == first
    assert client.calls == calls


@pytest.mark.parametrize(
    ("name", "latitude", "longitude"),
    [
        ("Different Hotel", 43.6426, -79.3871),
        ("Harbour Test Hotel", 43.6526, -79.3871),
        ("Harbour Test Hotel", 43.6426, -79.3971),
    ],
)
def test_detail_rejects_cross_hotel_identity_even_with_same_property_token(
    tmp_path: Path,
    name: str,
    latitude: float,
    longitude: float,
) -> None:
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [
            _search_payload(),
            _property_detail_payload(
                name=name,
                latitude=latitude,
                longitude=longitude,
            ),
        ],
    )
    provider = _provider(tmp_path, client)
    listing = _search(provider)

    with pytest.raises(HotelPriceError) as error_info:
        provider.detail(
            listing.offers[0].hotel_id,
            "Toronto",
            "YYZ",
            CHECK_IN,
            CHECK_OUT,
            adults=2,
            language="en",
            explicit=True,
        )

    assert error_info.value.code == "response_invalid"
    assert "same property identity" in str(error_info.value)


def test_cached_listing_reacquires_token_once_after_process_restart(
    tmp_path: Path,
) -> None:
    first_client = _Client(_account(monthly=0, hourly=0), _search_payload())
    first_provider = _provider(tmp_path, first_client)
    listing = _search(first_provider)
    exact_query = "Harbour Test Hotel, Toronto"
    second_client = _Client(
        [_account(monthly=1, hourly=1), _account(monthly=2, hourly=2)],
        [
            _search_payload(query=exact_query),
            _property_detail_payload(),
        ],
    )
    restarted_provider = _provider(tmp_path, second_client)

    detail = restarted_provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    assert detail.room_rates_status == "available"
    assert detail.review_sources_status == "available"
    searches = [
        call for call in second_client.calls if call["url"] == SERPAPI_SEARCH_URL
    ]
    assert len(searches) == 2
    assert searches[0]["params"]["q"] == exact_query
    assert searches[1]["params"]["property_token"] == (
        "do-not-persist-property-token"
    )


def test_reacquisition_never_fuzzy_merges_a_different_hotel(tmp_path: Path) -> None:
    first_provider = _provider(
        tmp_path,
        _Client(_account(monthly=0, hourly=0), _search_payload()),
    )
    listing = _search(first_provider)
    wrong_property = _search_payload(
        query="Harbour Test Hotel, Toronto",
        properties=[
            {
                "type": "hotel",
                "name": "Harbour Test Hotel Annex",
                "property_token": "different-property-token",
                "gps_coordinates": {"latitude": 43.6426, "longitude": -79.3871},
                "rate_per_night": {"extracted_lowest": 120},
            }
        ],
    )
    client = _Client(_account(monthly=1, hourly=1), wrong_property)
    restarted_provider = _provider(tmp_path, client)

    detail = restarted_provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    assert detail.room_rates_status == "temporarily_unavailable"
    assert detail.review_sources_status == "available"
    assert [call["url"] for call in client.calls].count(SERPAPI_SEARCH_URL) == 1


def test_room_aggregate_price_never_inherits_unpriced_rate_benefits(
    tmp_path: Path,
) -> None:
    detail_payload = _property_detail_payload()
    detail_payload["featured_prices"] = [
        {
            "source": "Hotel Direct",
            "rooms": [
                {
                    "name": "Strict King",
                    "rate_per_night": {"extracted_lowest": 199},
                    "total_rate": {"extracted_lowest": 398},
                    "rates": [
                        {
                            "free_cancellation": True,
                            "breakfast_included": True,
                        }
                    ],
                }
            ],
        }
    ]
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [_search_payload(), detail_payload],
    )
    provider = _provider(tmp_path, client)
    listing = _search(provider)

    detail = provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    assert len(detail.room_rates) == 1
    room = detail.room_rates[0]
    assert room.nightly_price == 199
    assert room.total_price == 398
    assert room.free_cancellation is None
    assert room.breakfast_included is None


def test_official_top_level_property_search_is_parsed_and_token_verified(
    tmp_path: Path,
) -> None:
    exact_query = "Harbour Test Hotel, Toronto"
    listing_payload = _search_payload(query=exact_query)
    top_level_payload = {
        "search_metadata": listing_payload["search_metadata"],
        "search_parameters": listing_payload["search_parameters"],
        **listing_payload["properties"][0],
    }
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [top_level_payload, _property_detail_payload()],
    )
    provider = _provider(tmp_path, client)

    listing = _search(provider, city_query=exact_query)
    detail = provider.detail(
        listing.offers[0].hotel_id,
        exact_query,
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert len(listing.offers) == 1
    assert detail is not None
    assert detail.detail_fetch_complete is True
    assert detail.room_rates_status == "available"


def test_property_wrapped_exact_search_is_not_misclassified_as_empty(
    tmp_path: Path,
) -> None:
    exact_query = "Harbour Test Hotel, Toronto"
    listing_payload = _search_payload(query=exact_query)
    property_row = dict(listing_payload["properties"][0])
    property_row.update(
        {
            "address": "100 Harbour Street, Toronto, ON",
            "phone": "+1 416 555 0100",
            "check_in_time": "3:00 PM",
            "check_out_time": "11:00 AM",
            "thumbnail": "https://images.example.test/listing-thumb.jpg",
            "images": [
                {
                    "thumbnail": "https://images.example.test/listing-image-thumb.jpg",
                    "original_image": "https://images.example.test/listing-image.jpg",
                }
            ],
        }
    )
    wrapped_payload = {
        "search_metadata": listing_payload["search_metadata"],
        "search_parameters": listing_payload["search_parameters"],
        "property": property_row,
    }
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [wrapped_payload, _property_detail_payload()],
    )
    provider = _provider(tmp_path, client)

    listing = _search(provider, city_query=exact_query)

    assert listing.status == "available"
    assert len(listing.offers) == 1
    offer = listing.offers[0]
    assert offer.name == "Harbour Test Hotel"
    assert offer.address == "100 Harbour Street, Toronto, ON"
    assert offer.phone == "+1 416 555 0100"
    assert offer.check_in_time == "3:00 PM"
    assert offer.check_out_time == "11:00 AM"
    assert offer.thumbnail == "https://images.example.test/listing-thumb.jpg"
    assert offer.images == ("https://images.example.test/listing-image.jpg",)

    detail = provider.detail(
        offer.hotel_id,
        exact_query,
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    assert detail.detail_fetch_complete is True
    detail_call = [
        call for call in client.calls if call["url"] == SERPAPI_SEARCH_URL
    ][1]
    assert detail_call["params"]["property_token"] == (
        "do-not-persist-property-token"
    )


def test_provider_metadata_never_accepts_unsafe_image_urls(tmp_path: Path) -> None:
    payload = _search_payload()
    payload["properties"][0].update(
        {
            "thumbnail": "https://serpapi.com/private.jpg?api_key=secret",
            "images": [
                {"original_image": "http://images.example.test/insecure.jpg"},
                {
                    "original_image": (
                        "https://images.example.test/private.jpg?token=secret"
                    )
                },
                {
                    "original_image": "https://images.example.test/public.jpg"
                },
            ],
        }
    )
    provider = _provider(
        tmp_path,
        _Client(_account(monthly=0, hourly=0), payload),
    )

    result = _search(provider)

    assert result.offers[0].thumbnail is None
    assert result.offers[0].images == (
        "https://images.example.test/public.jpg",
    )
    persisted = provider.usage_path.read_bytes()
    assert b"api_key=secret" not in persisted
    assert b"token=secret" not in persisted


def test_partial_listing_evidence_does_not_skip_explicit_property_detail(
    tmp_path: Path,
) -> None:
    listing_payload = _search_payload()
    listing_payload["properties"][0]["featured_prices"] = (
        _property_detail_payload()["featured_prices"][:1]
    )
    client = _Client(
        [_account(monthly=0, hourly=0), _account(monthly=1, hourly=1)],
        [listing_payload, _property_detail_payload()],
    )
    provider = _provider(tmp_path, client)
    listing = _search(provider)

    assert listing.offers[0].room_rates
    assert {item.source for item in listing.offers[0].review_sources} == {"Google"}
    assert listing.offers[0].detail_fetch_complete is False

    detail = provider.detail(
        listing.offers[0].hotel_id,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    assert detail.detail_fetch_complete is True
    assert {item.source for item in detail.review_sources} == {
        "Google",
        "Tripadvisor",
        "Trip.com",
    }
    assert [call["url"] for call in client.calls].count(SERPAPI_SEARCH_URL) == 2


def test_osm_exact_property_restart_reuses_idempotent_query_for_token(
    tmp_path: Path,
) -> None:
    exact_query = "Harbour Test Hotel, Toronto"
    first_provider = _provider(
        tmp_path,
        _Client(
            _account(monthly=0, hourly=0),
            _search_payload(query=exact_query),
        ),
    )
    _search(first_provider, city_query=exact_query)
    second_client = _Client(
        [_account(monthly=1, hourly=1), _account(monthly=2, hourly=2)],
        [
            _search_payload(query=exact_query),
            _property_detail_payload(),
        ],
    )
    restarted_provider = _provider(tmp_path, second_client)

    detail = restarted_provider.exact_property_detail(
        ("Harbour Test Hotel",),
        43.6426,
        -79.3871,
        "Toronto",
        "YYZ",
        CHECK_IN,
        CHECK_OUT,
        adults=2,
        language="en",
        explicit=True,
    )

    assert detail is not None
    searches = [
        call for call in second_client.calls if call["url"] == SERPAPI_SEARCH_URL
    ]
    assert searches[0]["params"]["q"] == exact_query
    assert "Harbour Test Hotel, Harbour Test Hotel" not in searches[0]["params"]["q"]
    assert searches[1]["params"]["property_token"] == (
        "do-not-persist-property-token"
    )
