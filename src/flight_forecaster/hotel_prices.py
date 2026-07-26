"""Explicit, quota-safe Google Hotels price lookups through SerpApi.

This module deliberately shares the SerpApi quota ledger used by strict flight
searches.  It never performs network I/O during construction or ``detail``
lookups, never requests ``no_cache=true``, and persists only sanitized hotel
fields in its one-hour local cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any, Literal
from urllib import parse

from flight_forecaster.availability import (
    ACCOUNT_REQUEST_TIMEOUT_SECONDS,
    MAX_PROVIDER_RESPONSE_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    SEARCH_ARCHIVE_REQUEST_TIMEOUT_SECONDS,
    SERPAPI_ACCOUNT_URL,
    SERPAPI_DEFAULT_MONTHLY_LIMIT,
    SERPAPI_MAX_HOURLY_LIMIT,
    SERPAPI_MAX_MONTHLY_LIMIT,
    SERPAPI_SEARCH_ARCHIVE_URL,
    SERPAPI_SEARCH_URL,
    _UrllibClient,
    _UsageError,
    _UsageLedger,
)

HotelPriceStatus = Literal["available", "no_results"]
HotelDetailEvidenceStatus = Literal[
    "not_requested",
    "available",
    "source_not_provided",
    "temporarily_unavailable",
]
HotelPriceErrorCode = Literal[
    "validation_error",
    "not_configured",
    "authentication_failed",
    "quota_exhausted",
    "rate_limited",
    "provider_processing",
    "provider_error",
    "provider_unavailable",
    "response_invalid",
    "quota_ledger_unavailable",
]
QuotaScope = Literal["monthly", "hourly"]

HOTEL_PRICE_PROVIDER_CODE = "serpapi_google_hotels"
HOTEL_PRICE_PROVIDER_NAME = "SerpApi Google Hotels"
HOTEL_PRICE_CACHE_TTL_SECONDS = 60 * 60
HOTEL_PRICE_FAILURE_GUARD_SECONDS = 60 * 60
HOTEL_PRICE_POLL_DELAYS_SECONDS = (0.25, 0.5, 1.0)
HOTEL_PRICE_MAX_RESULTS = 60
HOTEL_PRICE_MAX_ADULTS = 8
HOTEL_PRICE_MAX_FUTURE_DAYS = 370
HOTEL_PRICE_MAX_RESPONSE_BYTES = min(MAX_PROVIDER_RESPONSE_BYTES, 4_000_000)
HOTEL_PRICE_DEFAULT_USAGE_PATH = (
    Path("artifacts") / "runtime" / "serpapi-usage.sqlite3"
)

_IATA_PATTERN = re.compile(r"^[A-Z]{3}$")
_HOTEL_ID_PATTERN = re.compile(r"^gh_[a-f0-9]{32}$")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$")
_SEARCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_SAFE_CURRENCY = "USD"


class HotelPriceError(RuntimeError):
    """A classified error whose public text contains no provider payload."""

    code: HotelPriceErrorCode
    quota_scope: QuotaScope | None

    def __init__(
        self,
        code: HotelPriceErrorCode,
        message: str,
        *,
        quota_scope: QuotaScope | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.quota_scope = quota_scope


class HotelPriceValidationError(HotelPriceError):
    def __init__(self, message: str) -> None:
        super().__init__("validation_error", message)


@dataclass(frozen=True, slots=True)
class HotelRoomRate:
    """One source-backed room/rate row returned for the exact stay."""

    room_name: str
    source: str
    nightly_price: float | None
    total_price: float | None
    nightly_before_taxes: float | None
    total_before_taxes: float | None
    currency: Literal["USD"]
    guests: int | None
    official: bool | None
    free_cancellation: bool | None
    free_cancellation_until: str | None
    breakfast_included: bool | None
    beds: tuple[str, ...]
    inclusions: tuple[str, ...]
    booking_url: str | None
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "beds", tuple(self.beds))
        object.__setattr__(self, "inclusions", tuple(self.inclusions))
        object.__setattr__(self, "observed_at", _as_utc(self.observed_at))
        if _safe_text(self.room_name, 200) is None or _safe_text(self.source, 160) is None:
            raise ValueError("hotel room identity is invalid")
        values = (
            self.nightly_price,
            self.total_price,
            self.nightly_before_taxes,
            self.total_before_taxes,
        )
        if all(value is None for value in values):
            raise ValueError("hotel room rate requires price evidence")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in values
        ):
            raise ValueError("hotel room price is invalid")
        if self.currency != _SAFE_CURRENCY:
            raise ValueError("hotel room currency is invalid")
        if self.guests is not None and not 1 <= self.guests <= 50:
            raise ValueError("hotel room guest count is invalid")
        if self.booking_url is not None and _safe_public_url(self.booking_url) is None:
            raise ValueError("hotel room booking URL is invalid")
        if len(self.beds) > 12 or any(_safe_text(item, 100) is None for item in self.beds):
            raise ValueError("hotel room beds are invalid")
        if len(self.inclusions) > 20 or any(
            _safe_text(item, 160) is None for item in self.inclusions
        ):
            raise ValueError("hotel room inclusions are invalid")

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "room_name": self.room_name,
            "source": self.source,
            "nightly_price": self.nightly_price,
            "total_price": self.total_price,
            "nightly_before_taxes": self.nightly_before_taxes,
            "total_before_taxes": self.total_before_taxes,
            "currency": self.currency,
            "guests": self.guests,
            "official": self.official,
            "free_cancellation": self.free_cancellation,
            "free_cancellation_until": self.free_cancellation_until,
            "breakfast_included": self.breakfast_included,
            "beds": list(self.beds),
            "inclusions": list(self.inclusions),
            "booking_url": self.booking_url,
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class HotelReviewSource:
    """Rating and optional review excerpt tied to one named review platform."""

    source: str
    score: float | None
    max_score: float | None
    review_count: int | None
    sample_author: str | None
    sample_date: str | None
    sample_score: float | None
    sample_max_score: float | None
    sample_comment: str | None
    review_url: str | None
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _as_utc(self.observed_at))
        if _safe_text(self.source, 160) is None:
            raise ValueError("hotel review source is invalid")
        if (self.score is None) != (self.max_score is None):
            raise ValueError("hotel review rating is incomplete")
        if self.score is not None and (
            not math.isfinite(self.score)
            or not math.isfinite(self.max_score or 0)
            or self.max_score is None
            or self.max_score <= 0
            or self.score < 0
            or self.score > self.max_score
        ):
            raise ValueError("hotel review rating is invalid")
        if self.review_count is not None and self.review_count < 0:
            raise ValueError("hotel review count is invalid")
        if (self.sample_score is None) != (self.sample_max_score is None):
            raise ValueError("hotel sample review rating is incomplete")
        if self.sample_score is not None and (
            not math.isfinite(self.sample_score)
            or not math.isfinite(self.sample_max_score or 0)
            or self.sample_max_score is None
            or self.sample_max_score <= 0
            or self.sample_score < 0
            or self.sample_score > self.sample_max_score
        ):
            raise ValueError("hotel sample review rating is invalid")
        if self.review_url is not None and _safe_public_url(self.review_url) is None:
            raise ValueError("hotel review URL is invalid")

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": self.score,
            "max_score": self.max_score,
            "review_count": self.review_count,
            "sample_author": self.sample_author,
            "sample_date": self.sample_date,
            "sample_score": self.sample_score,
            "sample_max_score": self.sample_max_score,
            "sample_comment": self.sample_comment,
            "review_url": self.review_url,
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class HotelPriceOffer:
    """Sanitized, cache-safe price evidence for one Google Hotels property."""

    hotel_id: str
    name: str
    property_type: str
    latitude: float
    longitude: float
    description: str | None
    hotel_class: int | None
    rating: float | None
    review_count: int | None
    nightly_price: float | None
    total_price: float | None
    currency: Literal["USD"]
    price_source: str | None
    free_cancellation: bool | None
    amenities: tuple[str, ...]
    website_url: str
    observed_at: datetime
    room_rates: tuple[HotelRoomRate, ...] = ()
    review_sources: tuple[HotelReviewSource, ...] = ()
    room_rates_status: HotelDetailEvidenceStatus = "not_requested"
    review_sources_status: HotelDetailEvidenceStatus = "not_requested"
    detail_observed_at: datetime | None = None
    detail_fetch_complete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "amenities", tuple(self.amenities))
        object.__setattr__(self, "room_rates", tuple(self.room_rates))
        object.__setattr__(self, "review_sources", tuple(self.review_sources))
        object.__setattr__(self, "observed_at", _as_utc(self.observed_at))
        if self.detail_observed_at is not None:
            object.__setattr__(
                self,
                "detail_observed_at",
                _as_utc(self.detail_observed_at),
            )
        if not _HOTEL_ID_PATTERN.fullmatch(self.hotel_id):
            raise ValueError("hotel ID is invalid")
        if _safe_text(self.name, 200) is None or _safe_text(self.property_type, 80) is None:
            raise ValueError("hotel identity is invalid")
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("hotel coordinates are invalid")
        if self.hotel_class is not None and not 1 <= self.hotel_class <= 5:
            raise ValueError("hotel class is invalid")
        if self.rating is not None and not 0 <= self.rating <= 5:
            raise ValueError("hotel rating is invalid")
        if self.review_count is not None and self.review_count < 0:
            raise ValueError("hotel review count is invalid")
        for value in (self.nightly_price, self.total_price):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("hotel price is invalid")
        if self.currency != _SAFE_CURRENCY:
            raise ValueError("hotel currency is invalid")
        if _safe_public_url(self.website_url) is None:
            raise ValueError("hotel website URL is invalid")
        if len(self.amenities) > 40 or any(
            _safe_text(item, 100) is None for item in self.amenities
        ):
            raise ValueError("hotel amenities are invalid")
        valid_detail_statuses = {
            "not_requested",
            "available",
            "source_not_provided",
            "temporarily_unavailable",
        }
        if (
            self.room_rates_status not in valid_detail_statuses
            or self.review_sources_status not in valid_detail_statuses
        ):
            raise ValueError("hotel detail evidence status is invalid")
        if bool(self.room_rates) != (self.room_rates_status == "available"):
            raise ValueError("hotel room evidence status is inconsistent")
        if bool(self.review_sources) != (self.review_sources_status == "available"):
            raise ValueError("hotel review evidence status is inconsistent")
        if self.detail_fetch_complete and self.detail_observed_at is None:
            raise ValueError("completed hotel detail requires an observation time")

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "hotel_id": self.hotel_id,
            "name": self.name,
            "property_type": self.property_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "description": self.description,
            "hotel_class": self.hotel_class,
            "rating": self.rating,
            "review_count": self.review_count,
            "nightly_price": self.nightly_price,
            "total_price": self.total_price,
            "currency": self.currency,
            "price_source": self.price_source,
            "free_cancellation": self.free_cancellation,
            "amenities": list(self.amenities),
            "website_url": self.website_url,
            "observed_at": self.observed_at.isoformat(),
            "room_rates": [item.as_safe_dict() for item in self.room_rates],
            "review_sources": [item.as_safe_dict() for item in self.review_sources],
            "room_rates_status": self.room_rates_status,
            "review_sources_status": self.review_sources_status,
            "detail_observed_at": (
                self.detail_observed_at.isoformat()
                if self.detail_observed_at is not None
                else None
            ),
            "detail_fetch_complete": self.detail_fetch_complete,
        }


@dataclass(frozen=True, slots=True)
class HotelPriceSearchResult:
    offers: tuple[HotelPriceOffer, ...]
    status: HotelPriceStatus
    observed_at: datetime
    cache_hit: bool
    calls_reserved: int
    quota_monthly_used: int | None
    quota_monthly_limit: int | None
    quota_hourly_used: int | None
    quota_hourly_limit: int | None
    provider_code: str = HOTEL_PRICE_PROVIDER_CODE
    provider_name: str = HOTEL_PRICE_PROVIDER_NAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "offers", tuple(self.offers))
        object.__setattr__(self, "observed_at", _as_utc(self.observed_at))
        if self.status == "available" and not self.offers:
            raise ValueError("available hotel result requires offers")
        if self.status == "no_results" and self.offers:
            raise ValueError("no-results hotel result cannot contain offers")
        if self.calls_reserved not in {0, 1, 2}:
            raise ValueError("hotel reservation count is invalid")
        if self.cache_hit and self.calls_reserved:
            raise ValueError("cached hotel result cannot reserve quota")
        quota_values = (
            self.quota_monthly_used,
            self.quota_monthly_limit,
            self.quota_hourly_used,
            self.quota_hourly_limit,
        )
        if any(value is not None and value < 0 for value in quota_values):
            raise ValueError("hotel quota snapshot is invalid")


@dataclass(frozen=True, slots=True)
class _Stay:
    city_query: str
    destination_iata: str
    check_in: date
    check_out: date
    adults: int
    language: str

    @property
    def cache_key(self) -> str:
        encoded = json.dumps(
            {
                "city": self.city_query.casefold(),
                "destination": self.destination_iata,
                "check_in": self.check_in.isoformat(),
                "check_out": self.check_out.isoformat(),
                "adults": self.adults,
                "currency": _SAFE_CURRENCY,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _AccountQuota:
    billing_cycle_key: str
    hour_bucket_key: str
    monthly_used: int
    hourly_used: int
    monthly_limit: int
    hourly_limit: int


class _HotelCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load_search(
        self,
        query_key: str,
        *,
        now: datetime,
    ) -> HotelPriceSearchResult | None:
        if not self.path.is_file():
            return None
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(create=False)
            row = connection.execute(
                """
                SELECT payload_json, expires_at
                FROM serpapi_hotel_search_cache
                WHERE query_key = ?
                """,
                (query_key,),
            ).fetchone()
            if row is None or _cached_deadline(row[1]) <= now:
                return None
            payload = json.loads(str(row[0]))
            return _result_from_cache(payload)
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            return None
        finally:
            if connection is not None:
                connection.close()

    def load_detail(
        self,
        query_key: str,
        hotel_id: str,
        *,
        now: datetime,
    ) -> HotelPriceOffer | None:
        if not self.path.is_file():
            return None
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(create=False)
            row = connection.execute(
                """
                SELECT payload_json, expires_at
                FROM serpapi_hotel_detail_cache
                WHERE query_key = ? AND hotel_id = ?
                """,
                (query_key, hotel_id),
            ).fetchone()
            if row is None or _cached_deadline(row[1]) <= now:
                return None
            payload = json.loads(str(row[0]))
            return _offer_from_dict(payload)
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            return None
        finally:
            if connection is not None:
                connection.close()

    def load_failure(
        self,
        query_key: str,
        *,
        now: datetime,
    ) -> HotelPriceErrorCode | None:
        """Return a recent classified failure without retaining provider IDs."""

        if not self.path.is_file():
            return None
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(create=False)
            row = connection.execute(
                """
                SELECT error_code, expires_at
                FROM serpapi_hotel_failure_guard
                WHERE query_key = ?
                """,
                (query_key,),
            ).fetchone()
            if row is None or _cached_deadline(row[1]) <= now:
                return None
            return _failure_code(row[0])
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return None
        finally:
            if connection is not None:
                connection.close()

    def store(
        self,
        query_key: str,
        result: HotelPriceSearchResult,
        *,
        expires_at: datetime,
    ) -> None:
        payload = {
            "status": result.status,
            "observed_at": result.observed_at.isoformat(),
            "offers": [offer.as_safe_dict() for offer in result.offers],
        }
        serialized = _safe_cache_json(payload)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(create=True)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO serpapi_hotel_search_cache(
                    query_key, observed_at, expires_at, payload_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(query_key) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    expires_at = excluded.expires_at,
                    payload_json = excluded.payload_json
                """,
                (
                    query_key,
                    result.observed_at.isoformat(),
                    expires_at.isoformat(),
                    serialized,
                ),
            )
            connection.execute(
                "DELETE FROM serpapi_hotel_detail_cache WHERE query_key = ?",
                (query_key,),
            )
            connection.execute(
                "DELETE FROM serpapi_hotel_failure_guard WHERE query_key = ?",
                (query_key,),
            )
            for offer in result.offers:
                connection.execute(
                    """
                    INSERT INTO serpapi_hotel_detail_cache(
                        query_key, hotel_id, observed_at, expires_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        query_key,
                        offer.hotel_id,
                        offer.observed_at.isoformat(),
                        expires_at.isoformat(),
                        _safe_cache_json(offer.as_safe_dict()),
                    ),
                )
            connection.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise HotelPriceError(
                "quota_ledger_unavailable",
                "The local hotel cache is unavailable.",
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def store_detail(
        self,
        query_key: str,
        offer: HotelPriceOffer,
        *,
        expires_at: datetime,
    ) -> None:
        """Replace one sanitized detail row without retaining provider tokens."""

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(create=True)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO serpapi_hotel_detail_cache(
                    query_key, hotel_id, observed_at, expires_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(query_key, hotel_id) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    expires_at = excluded.expires_at,
                    payload_json = excluded.payload_json
                """,
                (
                    query_key,
                    offer.hotel_id,
                    offer.observed_at.isoformat(),
                    expires_at.isoformat(),
                    _safe_cache_json(offer.as_safe_dict()),
                ),
            )
            connection.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise HotelPriceError(
                "quota_ledger_unavailable",
                "The local hotel detail cache is unavailable.",
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def store_failure(
        self,
        query_key: str,
        error_code: HotelPriceErrorCode,
        *,
        observed_at: datetime,
        expires_at: datetime,
    ) -> None:
        """Persist only the safe category and timestamps after a spent request."""

        safe_code = _failure_code(error_code)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(create=True)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO serpapi_hotel_failure_guard(
                    query_key, error_code, observed_at, expires_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(query_key) DO UPDATE SET
                    error_code = excluded.error_code,
                    observed_at = excluded.observed_at,
                    expires_at = excluded.expires_at
                """,
                (
                    query_key,
                    safe_code,
                    observed_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            connection.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise HotelPriceError(
                "quota_ledger_unavailable",
                "The local hotel failure guard is unavailable.",
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def _connect(self, *, create: bool) -> sqlite3.Connection:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        if not create:
            connection.execute("PRAGMA query_only = ON")
            return connection
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS serpapi_hotel_search_cache (
                query_key TEXT PRIMARY KEY,
                observed_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS serpapi_hotel_detail_cache (
                query_key TEXT NOT NULL,
                hotel_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(query_key, hotel_id)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS serpapi_hotel_failure_guard (
                query_key TEXT PRIMARY KEY,
                error_code TEXT NOT NULL CHECK(
                    error_code IN (
                        'authentication_failed',
                        'quota_exhausted',
                        'rate_limited',
                        'provider_processing',
                        'provider_error',
                        'provider_unavailable',
                        'response_invalid',
                        'quota_ledger_unavailable'
                    )
                ),
                observed_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        return connection


class SerpApiHotelPriceProvider:
    """One explicit Google Hotels lookup backed by the shared flight quota."""

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: Path = HOTEL_PRICE_DEFAULT_USAGE_PATH,
        monthly_limit: int | None = SERPAPI_DEFAULT_MONTHLY_LIMIT,
        client: Any = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        now_provider: Any = None,
        poll_delays_seconds: tuple[float, ...] = HOTEL_PRICE_POLL_DELAYS_SECONDS,
        sleep_provider: Any = None,
    ) -> None:
        self._api_key = _safe_api_key(api_key)
        self._usage_path = Path(usage_path)
        self._monthly_limit = _bounded_monthly_limit(monthly_limit)
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("hotel provider timeout is invalid") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("hotel provider timeout is invalid")
        self._timeout_seconds = min(timeout, 30.0)
        self._client = client or _UrllibClient()
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        try:
            delays = tuple(float(value) for value in poll_delays_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("hotel provider poll delays are invalid") from exc
        if (
            not delays
            or len(delays) > 5
            or any(not math.isfinite(value) or value < 0 or value > 5 for value in delays)
        ):
            raise ValueError("hotel provider poll delays are invalid")
        self._poll_delays_seconds = delays
        self._sleep_provider = sleep_provider or sleep
        self._ledger = _UsageLedger(self._usage_path)
        self._cache = _HotelCache(self._usage_path)
        # Property tokens are opaque provider identifiers.  They are kept only
        # in this bounded process-local map so a deliberate detail click can
        # retrieve exact-property evidence without writing tokens to disk.
        self._property_tokens: dict[tuple[str, str], str] = {}

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    @property
    def usage_path(self) -> Path:
        return self._usage_path

    def search(
        self,
        city_query: str,
        destination_iata: str,
        check_in: date | str,
        check_out: date | str,
        *,
        adults: int,
        language: str,
        explicit: bool = False,
        refresh_local_cache: bool = False,
    ) -> HotelPriceSearchResult:
        """Fetch prices only after an explicit user action.

        A valid one-hour local cache hit returns before the free Account API and
        before any shared-quota reservation.
        """

        now = self._now()
        stay = _validate_stay(
            city_query,
            destination_iata,
            check_in,
            check_out,
            adults=adults,
            language=language,
            today=now.date(),
        )
        if explicit is not True:
            raise HotelPriceValidationError("Hotel prices require an explicit user request.")
        cached = self._cache.load_search(stay.cache_key, now=now)
        if cached is not None and refresh_local_cache is not True:
            return replace(
                cached,
                cache_hit=True,
                calls_reserved=0,
                quota_monthly_used=None,
                quota_monthly_limit=None,
                quota_hourly_used=None,
                quota_hourly_limit=None,
            )
        if not self.configured:
            raise HotelPriceError(
                "not_configured",
                "Live hotel prices are not configured.",
            )
        guarded_code = self._cache.load_failure(stay.cache_key, now=now)
        if guarded_code is not None:
            raise _guarded_failure(guarded_code)

        account = self._account_quota()
        reservation = self._reserve_one(account)
        calls_reserved = 1
        params: dict[str, Any] = {
            "engine": "google_hotels",
            "q": stay.city_query,
            "check_in_date": stay.check_in.isoformat(),
            "check_out_date": stay.check_out.isoformat(),
            "adults": stay.adults,
            "currency": _SAFE_CURRENCY,
            "hl": stay.language,
            "api_key": self._api_key,
        }
        try:
            payload, observed_at = self._request_json(
                SERPAPI_SEARCH_URL,
                params=params,
                account_request=False,
                allow_pending=True,
            )
            if _search_status(payload) in {"processing", "queued"}:
                payload, observed_at = self._poll_pending_search(payload)
            if _search_status(payload) in {"processing", "queued"}:
                # One controlled re-submission only. Account synchronization is
                # free, while the second business attempt gets its own atomic
                # reservation in the exact ledger shared with flight searches.
                account = self._account_quota()
                reservation = self._reserve_one(account)
                calls_reserved = 2
                payload, observed_at = self._request_json(
                    SERPAPI_SEARCH_URL,
                    params=params,
                    account_request=False,
                    allow_pending=True,
                )
                if _search_status(payload) in {"processing", "queued"}:
                    payload, observed_at = self._poll_pending_search(payload)
                if _search_status(payload) in {"processing", "queued"}:
                    raise HotelPriceError(
                        "provider_processing",
                        "The hotel price provider is still processing the request.",
                    )
            _validate_search_echo(payload, stay)
            offers = _parse_offers(payload, stay, observed_at)
            self._remember_property_tokens(payload, stay, offers)
            raw_properties = payload.get("properties")
            if (
                isinstance(raw_properties, list)
                and raw_properties
                and not offers
                and not any(_safe_property_identity(row) for row in raw_properties)
            ):
                raise HotelPriceError(
                    "response_invalid",
                    "The hotel provider response did not contain safe property records.",
                )
        except HotelPriceError as exc:
            failure_time = self._now()
            self._cache.store_failure(
                stay.cache_key,
                exc.code,
                observed_at=failure_time,
                expires_at=failure_time
                + timedelta(seconds=HOTEL_PRICE_FAILURE_GUARD_SECONDS),
            )
            raise
        result = HotelPriceSearchResult(
            offers=offers,
            status="available" if offers else "no_results",
            observed_at=observed_at,
            cache_hit=False,
            calls_reserved=calls_reserved,
            quota_monthly_used=reservation.monthly_used,
            quota_monthly_limit=account.monthly_limit,
            quota_hourly_used=reservation.hourly_used,
            quota_hourly_limit=account.hourly_limit,
        )
        self._cache.store(
            stay.cache_key,
            result,
            expires_at=observed_at + timedelta(seconds=HOTEL_PRICE_CACHE_TTL_SECONDS),
        )
        return result

    def detail(
        self,
        hotel_id: str,
        city_query: str,
        destination_iata: str,
        check_in: date | str,
        check_out: date | str,
        *,
        adults: int,
        language: str,
        explicit: bool = False,
    ) -> HotelPriceOffer | None:
        """Read or explicitly enrich an exact-stay hotel from the safe cache.

        The default remains a zero-network cache lookup.  ``explicit=True`` is
        reserved for a user's hotel-detail action and may consume one shared
        SerpApi search credit.  The opaque property token is never persisted.
        """

        now = self._now()
        if not isinstance(hotel_id, str) or not _HOTEL_ID_PATTERN.fullmatch(hotel_id):
            raise HotelPriceValidationError("Hotel ID is invalid.")
        stay = _validate_stay(
            city_query,
            destination_iata,
            check_in,
            check_out,
            adults=adults,
            language=language,
            today=now.date(),
        )
        cached = self._cache.load_detail(stay.cache_key, hotel_id, now=now)
        if cached is None or explicit is not True:
            return cached
        if cached.detail_fetch_complete:
            return cached
        token = self._property_tokens.get((stay.cache_key, hotel_id))
        if token is None:
            exact_query = _exact_property_query(cached.name, stay.city_query)
            if exact_query is None:
                return _detail_temporarily_unavailable(cached)
            reacquired = self.search(
                exact_query,
                stay.destination_iata,
                stay.check_in,
                stay.check_out,
                adults=stay.adults,
                language=stay.language,
                explicit=True,
                refresh_local_cache=True,
            )
            matches = [
                offer
                for offer in reacquired.offers
                if _same_hotel_identity(offer, cached)
            ]
            if len(matches) != 1:
                return _detail_temporarily_unavailable(cached)
            exact_stay = _validate_stay(
                exact_query,
                stay.destination_iata,
                stay.check_in,
                stay.check_out,
                adults=stay.adults,
                language=stay.language,
                today=now.date(),
            )
            token = self._property_tokens.get(
                (exact_stay.cache_key, matches[0].hotel_id)
            )
            if token is None:
                return _detail_temporarily_unavailable(cached)
            self._property_tokens[(stay.cache_key, hotel_id)] = token
        if not self.configured:
            raise HotelPriceError(
                "not_configured",
                "Live hotel details are not configured.",
            )
        account = self._account_quota()
        self._reserve_one(account)
        params: dict[str, Any] = {
            "engine": "google_hotels",
            "property_token": token,
            "check_in_date": stay.check_in.isoformat(),
            "check_out_date": stay.check_out.isoformat(),
            "adults": stay.adults,
            "currency": _SAFE_CURRENCY,
            "hl": stay.language,
            "api_key": self._api_key,
        }
        payload, observed_at = self._request_json(
            SERPAPI_SEARCH_URL,
            params=params,
            account_request=False,
            allow_pending=True,
        )
        if _search_status(payload) in {"processing", "queued"}:
            payload, observed_at = self._poll_pending_search(payload)
        if _search_status(payload) in {"processing", "queued"}:
            return _detail_temporarily_unavailable(cached)
        _validate_detail_echo(payload, stay, token)
        detail_row = _verified_property_detail(payload, token, cached)
        room_rates = _parse_room_rates(detail_row, observed_at)
        review_sources = _parse_review_sources(detail_row, observed_at)
        enriched = replace(
            cached,
            description=_safe_text(detail_row.get("description"), 1_200)
            or cached.description,
            rating=_bounded_float(detail_row.get("overall_rating"), 0, 5)
            if detail_row.get("overall_rating") is not None
            else cached.rating,
            review_count=_nonnegative_int(detail_row.get("reviews"))
            if detail_row.get("reviews") is not None
            else cached.review_count,
            amenities=(
                _safe_text_list(
                    detail_row.get("amenities"),
                    limit=40,
                    item_limit=100,
                )
                if isinstance(detail_row.get("amenities"), list)
                else cached.amenities
            ),
            room_rates=room_rates,
            review_sources=review_sources,
            room_rates_status="available" if room_rates else "source_not_provided",
            review_sources_status=(
                "available" if review_sources else "source_not_provided"
            ),
            detail_observed_at=observed_at,
            detail_fetch_complete=True,
        )
        self._cache.store_detail(
            stay.cache_key,
            enriched,
            expires_at=observed_at + timedelta(seconds=HOTEL_PRICE_CACHE_TTL_SECONDS),
        )
        return enriched

    def exact_property_detail(
        self,
        hotel_names: tuple[str, ...],
        latitude: float,
        longitude: float,
        city_query: str,
        destination_iata: str,
        check_in: date | str,
        check_out: date | str,
        *,
        adults: int,
        language: str,
        explicit: bool = False,
    ) -> HotelPriceOffer | None:
        """Resolve an OSM hotel only with exact-name and tight-coordinate proof.

        This deliberately refuses fuzzy matching.  A successful explicit call
        can use one property search plus one property-detail search, both
        protected by the same SerpApi quota ledger as flight searches.
        """

        names = tuple(
            dict.fromkeys(
                text
                for text in (
                    _safe_text(item, 200)
                    for item in tuple(hotel_names)[:4]
                )
                if text is not None
            )
        )
        expected_names = {_normalized_hotel_name(item) for item in names}
        expected_latitude = _coordinate(latitude, -90, 90)
        expected_longitude = _coordinate(longitude, -180, 180)
        city = _safe_text(city_query, 160)
        if (
            not expected_names
            or expected_latitude is None
            or expected_longitude is None
            or city is None
        ):
            raise HotelPriceValidationError("Exact hotel identity is invalid.")
        exact_query = _exact_property_query(names[0], city)
        if exact_query is None:
            raise HotelPriceValidationError("Exact hotel query is invalid.")
        result = self.search(
            exact_query,
            destination_iata,
            check_in,
            check_out,
            adults=adults,
            language=language,
            explicit=explicit,
        )
        matches = [
            offer
            for offer in result.offers
            if _normalized_hotel_name(offer.name) in expected_names
            and abs(offer.latitude - expected_latitude) <= 0.0015
            and abs(offer.longitude - expected_longitude) <= 0.0015
        ]
        if len(matches) != 1:
            return None
        return self.detail(
            matches[0].hotel_id,
            exact_query,
            destination_iata,
            check_in,
            check_out,
            adults=adults,
            language=language,
            explicit=explicit,
        )

    def _remember_property_tokens(
        self,
        payload: dict[str, Any],
        stay: _Stay,
        offers: tuple[HotelPriceOffer, ...],
    ) -> None:
        properties = payload.get("properties")
        if properties is None:
            properties = [payload] if _safe_top_level_property(payload) else []
        elif not isinstance(properties, list):
            return
        known_ids = {offer.hotel_id for offer in offers}
        for row in properties[:HOTEL_PRICE_MAX_RESULTS]:
            if not isinstance(row, dict):
                continue
            token = _safe_text(row.get("property_token"), 2_000)
            if token is None:
                continue
            hotel_id = _hotel_id(row, stay)
            if hotel_id in known_ids:
                self._property_tokens[(stay.cache_key, hotel_id)] = token
        while len(self._property_tokens) > 300:
            del self._property_tokens[next(iter(self._property_tokens))]

    def _account_quota(self) -> _AccountQuota:
        payload, received_at = self._request_json(
            SERPAPI_ACCOUNT_URL,
            params={"api_key": self._api_key},
            account_request=True,
        )
        if str(payload.get("account_status", "")).strip().lower() != "active":
            raise HotelPriceError(
                "authentication_failed",
                "The hotel price provider account is not active.",
            )
        monthly_used = _nonnegative_int(payload.get("this_month_usage"))
        hourly_used = _nonnegative_int(payload.get("this_hour_searches"))
        provider_monthly_limit = _positive_int(payload.get("searches_per_month"))
        provider_hourly_limit = _positive_int(payload.get("account_rate_limit_per_hour"))
        renewal = _renewal_date(payload.get("plan_renewal_date"), received_at)
        if None in (
            monthly_used,
            hourly_used,
            provider_monthly_limit,
            provider_hourly_limit,
            renewal,
        ):
            raise HotelPriceError(
                "response_invalid",
                "The provider account quota metadata is incomplete.",
            )
        assert isinstance(renewal, date)
        return _AccountQuota(
            billing_cycle_key=f"renewal:{renewal.isoformat()}",
            hour_bucket_key=received_at.strftime("%Y-%m-%dT%H"),
            monthly_used=monthly_used,
            hourly_used=hourly_used,
            monthly_limit=min(self._monthly_limit, provider_monthly_limit),
            hourly_limit=min(SERPAPI_MAX_HOURLY_LIMIT, provider_hourly_limit),
        )

    def _reserve_one(self, account: _AccountQuota) -> Any:
        try:
            reservation = self._ledger.synchronize_and_reserve(
                account.billing_cycle_key,
                account.hour_bucket_key,
                calls=1,
                monthly_limit=account.monthly_limit,
                hourly_limit=account.hourly_limit,
                provider_monthly_usage=account.monthly_used,
                provider_hourly_usage=account.hourly_used,
                require_all=True,
            )
        except _UsageError as exc:
            raise HotelPriceError(
                "quota_ledger_unavailable",
                "The shared SerpApi quota ledger is unavailable.",
            ) from exc
        if reservation.reserved_calls != 1:
            scope: QuotaScope = (
                "hourly" if reservation.limiting_quota == "hourly" else "monthly"
            )
            raise HotelPriceError(
                "quota_exhausted",
                "The shared free SerpApi quota is exhausted.",
                quota_scope=scope,
            )
        return reservation

    def _poll_pending_search(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], datetime]:
        search_id = _payload_search_id(payload)
        if search_id is None:
            raise HotelPriceError(
                "response_invalid",
                "The pending hotel search did not contain a safe search ID.",
            )
        archive_url = SERPAPI_SEARCH_ARCHIVE_URL.format(search_id=search_id)
        if _archive_search_id(archive_url) != search_id:
            raise HotelPriceError(
                "response_invalid",
                "The pending hotel search archive URL is invalid.",
            )
        last_payload = payload
        last_observed = self._now()
        for delay_seconds in self._poll_delays_seconds:
            try:
                self._sleep_provider(delay_seconds)
            except Exception as exc:
                raise HotelPriceError(
                    "provider_unavailable",
                    "The hotel price provider polling wait failed.",
                ) from exc
            archived, observed_at = self._request_json(
                archive_url,
                params={"api_key": self._api_key},
                account_request=False,
                allow_pending=True,
            )
            if _payload_search_id(archived) != search_id:
                raise HotelPriceError(
                    "response_invalid",
                    "The hotel search archive response did not match the request.",
                )
            last_payload = archived
            last_observed = observed_at
            if _search_status(archived) in {"success", "cached"}:
                return archived, observed_at
        return last_payload, last_observed

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        account_request: bool,
        allow_pending: bool = False,
    ) -> tuple[dict[str, Any], datetime]:
        archive_id = _archive_search_id(url)
        if account_request:
            allowed_url = url == SERPAPI_ACCOUNT_URL
        else:
            allowed_url = url == SERPAPI_SEARCH_URL or archive_id is not None
        if not allowed_url:
            raise HotelPriceError(
                "response_invalid",
                "The hotel provider request URL is not allowlisted.",
            )
        try:
            response = self._client.get(
                url,
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "flight-forecast-lab/0.2.0 (explicit hotel prices)",
                },
                timeout=(
                    min(self._timeout_seconds, ACCOUNT_REQUEST_TIMEOUT_SECONDS)
                    if account_request
                    else min(
                        self._timeout_seconds,
                        SEARCH_ARCHIVE_REQUEST_TIMEOUT_SECONDS,
                    )
                    if archive_id is not None
                    else self._timeout_seconds
                ),
            )
        except Exception as exc:
            raise HotelPriceError(
                "provider_unavailable",
                "The hotel price provider is temporarily unavailable.",
            ) from exc
        observed_at = self._now()
        status = _response_status(response)
        if status in {401, 403}:
            raise HotelPriceError(
                "authentication_failed",
                "The hotel price provider rejected authentication.",
            )
        if status == 429:
            raise HotelPriceError(
                "rate_limited",
                "The hotel price provider rate limit was reached.",
            )
        if status in {408, 425} or 500 <= status <= 599:
            raise HotelPriceError(
                "provider_unavailable",
                "The hotel price provider is temporarily unavailable.",
            )
        if not 200 <= status <= 299:
            raise HotelPriceError(
                "provider_error",
                "The hotel price provider rejected the request.",
            )
        payload = _response_json(response)
        if not isinstance(payload, dict):
            raise HotelPriceError(
                "response_invalid",
                "The hotel price provider returned an invalid response.",
            )
        if not account_request:
            search_status = _search_status(payload)
            if search_status in {"processing", "queued"}:
                if allow_pending:
                    return payload, observed_at
                raise HotelPriceError(
                    "provider_processing",
                    "The hotel price provider is still processing the request.",
                )
            if search_status not in {"success", "cached"}:
                code: HotelPriceErrorCode = (
                    "provider_error" if search_status == "error" else "response_invalid"
                )
                raise HotelPriceError(code, "The hotel price provider search failed.")
        return payload, observed_at

    def _now(self) -> datetime:
        try:
            return _as_utc(self._now_provider())
        except (TypeError, ValueError, OverflowError) as exc:
            raise HotelPriceError(
                "response_invalid",
                "The hotel price provider clock is invalid.",
            ) from exc


def hotel_price_provider_from_env(
    usage_path: Path = HOTEL_PRICE_DEFAULT_USAGE_PATH,
) -> SerpApiHotelPriceProvider:
    """Construct the optional provider without performing any network I/O."""

    return SerpApiHotelPriceProvider(
        os.getenv("SERPAPI_API_KEY"),
        usage_path=usage_path,
        monthly_limit=os.getenv("SERPAPI_MONTHLY_LIMIT"),
    )


def _validate_stay(
    city_query: Any,
    destination_iata: Any,
    check_in: date | str,
    check_out: date | str,
    *,
    adults: Any,
    language: Any,
    today: date,
) -> _Stay:
    city = _safe_text(city_query, 120)
    if city is None or len(city) < 2:
        raise HotelPriceValidationError("Destination city is invalid.")
    iata = str(destination_iata or "").strip().upper()
    if not _IATA_PATTERN.fullmatch(iata):
        raise HotelPriceValidationError("Destination airport code is invalid.")
    start = _date_value(check_in)
    end = _date_value(check_out)
    if start is None or end is None or end <= start:
        raise HotelPriceValidationError("Hotel stay dates are invalid.")
    horizon = today + timedelta(days=HOTEL_PRICE_MAX_FUTURE_DAYS)
    if start < today or start > horizon or end > horizon:
        raise HotelPriceValidationError("Hotel stay must be within the next 370 days.")
    adult_count = _positive_int(adults)
    if adult_count is None or adult_count > HOTEL_PRICE_MAX_ADULTS:
        raise HotelPriceValidationError("Adults must be between 1 and 8.")
    lang = str(language or "").strip().lower().replace("_", "-")
    if not _LANGUAGE_PATTERN.fullmatch(lang) or lang not in {"en", "zh-cn"}:
        raise HotelPriceValidationError("Language must be en or zh-CN.")
    return _Stay(city, iata, start, end, adult_count, lang)


def _validate_search_echo(payload: dict[str, Any], stay: _Stay) -> None:
    parameters = payload.get("search_parameters")
    if not isinstance(parameters, dict):
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider did not confirm the search parameters.",
        )
    echoed_city = " ".join(str(parameters.get("q", "")).split()).casefold()
    checks = (
        str(parameters.get("engine", "")).strip() == "google_hotels",
        echoed_city == stay.city_query.casefold(),
        str(parameters.get("check_in_date", "")).strip() == stay.check_in.isoformat(),
        str(parameters.get("check_out_date", "")).strip() == stay.check_out.isoformat(),
        _positive_int(parameters.get("adults")) == stay.adults,
        str(parameters.get("currency", "")).strip().upper() == _SAFE_CURRENCY,
    )
    if not all(checks):
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider returned a mismatched search response.",
        )


def _validate_detail_echo(
    payload: dict[str, Any],
    stay: _Stay,
    property_token: str,
) -> None:
    parameters = payload.get("search_parameters")
    if not isinstance(parameters, dict):
        raise HotelPriceError(
            "response_invalid",
            "The hotel detail provider did not confirm the stay parameters.",
        )
    checks = (
        str(parameters.get("engine", "")).strip() == "google_hotels",
        _safe_text(parameters.get("property_token"), 2_000) == property_token,
        str(parameters.get("check_in_date", "")).strip() == stay.check_in.isoformat(),
        str(parameters.get("check_out_date", "")).strip() == stay.check_out.isoformat(),
        _positive_int(parameters.get("adults")) == stay.adults,
        str(parameters.get("currency", "")).strip().upper() == _SAFE_CURRENCY,
    )
    if not all(checks):
        raise HotelPriceError(
            "response_invalid",
            "The hotel detail provider returned a mismatched stay response.",
        )


def _parse_offers(
    payload: dict[str, Any],
    stay: _Stay,
    observed_at: datetime,
) -> tuple[HotelPriceOffer, ...]:
    properties = payload.get("properties")
    if properties is None:
        properties = [payload] if _safe_top_level_property(payload) else []
    if not isinstance(properties, list):
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider property list is invalid.",
        )
    deduplicated: dict[str, HotelPriceOffer] = {}
    for row in properties[:HOTEL_PRICE_MAX_RESULTS]:
        offer = _parse_offer(row, stay, observed_at)
        if offer is None:
            continue
        current = deduplicated.get(offer.hotel_id)
        if current is None or _price_rank(offer) < _price_rank(current):
            deduplicated[offer.hotel_id] = offer
    return tuple(deduplicated.values())


def _safe_property_identity(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    coordinates = row.get("gps_coordinates")
    return bool(
        _safe_text(row.get("name"), 200)
        and _safe_text(row.get("type"), 80)
        and isinstance(coordinates, dict)
        and _coordinate(coordinates.get("latitude"), -90, 90) is not None
        and _coordinate(coordinates.get("longitude"), -180, 180) is not None
    )


def _safe_top_level_property(payload: Any) -> bool:
    return bool(
        _safe_property_identity(payload)
        and isinstance(payload, dict)
        and _safe_text(payload.get("property_token"), 2_000)
    )


def _parse_offer(
    row: Any,
    stay: _Stay,
    observed_at: datetime,
) -> HotelPriceOffer | None:
    if not isinstance(row, dict):
        return None
    name = _safe_text(row.get("name"), 200)
    property_type = _safe_text(row.get("type"), 80)
    coordinates = row.get("gps_coordinates")
    if name is None or property_type is None or not isinstance(coordinates, dict):
        return None
    latitude = _coordinate(coordinates.get("latitude"), -90, 90)
    longitude = _coordinate(coordinates.get("longitude"), -180, 180)
    if latitude is None or longitude is None:
        return None
    hotel_id = _hotel_id(row, stay)
    selected_price = _selected_price(row)
    if selected_price is not None:
        # A price source is an indivisible evidence row. Never fill a missing
        # value or cancellation flag from the property aggregate or another
        # seller while retaining this seller's label.
        nightly = _nested_price(selected_price, "rate_per_night")
        total = _nested_price(selected_price, "total_rate")
        price_source = _safe_text(selected_price.get("source"), 160)
        free_cancellation = _optional_bool(selected_price.get("free_cancellation"))
    else:
        nightly = _nested_price(row, "rate_per_night")
        total = _nested_price(row, "total_rate")
        price_source = None
        free_cancellation = _optional_bool(row.get("free_cancellation"))
    if nightly is None and total is None:
        return None
    if price_source is None:
        price_source = "Google Hotels"
    hotel_class = _hotel_class(row)
    rating = _bounded_float(row.get("overall_rating"), 0, 5)
    review_count = _nonnegative_int(row.get("reviews"))
    description = _safe_text(row.get("description"), 1_200)
    amenities = _safe_text_list(row.get("amenities"), limit=40, item_limit=100)
    public_link = _safe_public_url(row.get("link")) or _google_hotels_url(name, stay.city_query)
    room_rates = _parse_room_rates(row, observed_at)
    review_sources = _parse_review_sources(row, observed_at)
    return HotelPriceOffer(
        hotel_id=hotel_id,
        name=name,
        property_type=property_type,
        latitude=latitude,
        longitude=longitude,
        description=description,
        hotel_class=hotel_class,
        rating=rating,
        review_count=review_count,
        nightly_price=nightly,
        total_price=total,
        currency=_SAFE_CURRENCY,
        price_source=price_source,
        free_cancellation=free_cancellation,
        amenities=amenities,
        website_url=public_link,
        observed_at=observed_at,
        room_rates=room_rates,
        review_sources=review_sources,
        room_rates_status="available" if room_rates else "not_requested",
        review_sources_status="available" if review_sources else "not_requested",
        detail_observed_at=None,
        detail_fetch_complete=False,
    )


def _hotel_id(row: dict[str, Any], stay: _Stay) -> str:
    name = _safe_text(row.get("name"), 200) or ""
    property_type = _safe_text(row.get("type"), 80) or ""
    coordinates = row.get("gps_coordinates")
    latitude = (
        _coordinate(coordinates.get("latitude"), -90, 90)
        if isinstance(coordinates, dict)
        else None
    )
    longitude = (
        _coordinate(coordinates.get("longitude"), -180, 180)
        if isinstance(coordinates, dict)
        else None
    )
    token = _safe_text(row.get("property_token"), 2_000)
    identity = token or json.dumps(
        [
            stay.destination_iata,
            name.casefold(),
            property_type.casefold(),
            latitude,
            longitude,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "gh_" + hashlib.sha256(
        ("serpapi-google-hotels-v1\0" + identity).encode("utf-8")
    ).hexdigest()[:32]


def _selected_price(row: dict[str, Any]) -> dict[str, Any] | None:
    prices = row.get("prices")
    if not isinstance(prices, list):
        return None
    safe_rows = [item for item in prices[:30] if isinstance(item, dict)]
    candidates = [
        item
        for item in safe_rows
        if _safe_text(item.get("source"), 160) is not None
        and (
            _nested_price(item, "rate_per_night") is not None
            or _nested_price(item, "total_rate") is not None
        )
    ]
    return min(
        candidates,
        key=lambda item: (
            _nested_price(item, "total_rate") or math.inf,
            _nested_price(item, "rate_per_night") or math.inf,
        ),
        default=None,
    )


def _verified_property_detail(
    payload: dict[str, Any],
    expected_token: str,
    expected_offer: HotelPriceOffer,
) -> dict[str, Any]:
    """Return only a property-detail row that proves the same hotel identity."""

    candidates: list[dict[str, Any]] = []
    if isinstance(payload.get("property"), dict):
        candidates.append(payload["property"])
    if isinstance(payload.get("properties"), list):
        candidates.extend(
            item for item in payload["properties"][:10] if isinstance(item, dict)
        )
    if _safe_text(payload.get("name"), 200) is not None:
        candidates.append(payload)
    expected_name = _normalized_hotel_name(expected_offer.name)
    for row in candidates:
        if _safe_text(row.get("property_token"), 2_000) != expected_token:
            continue
        if _normalized_hotel_name(_safe_text(row.get("name"), 200) or "") != expected_name:
            continue
        coordinates = row.get("gps_coordinates")
        if not isinstance(coordinates, dict):
            continue
        latitude = _coordinate(coordinates.get("latitude"), -90, 90)
        longitude = _coordinate(coordinates.get("longitude"), -180, 180)
        if latitude is None or longitude is None:
            continue
        if abs(latitude - expected_offer.latitude) > 0.0015:
            continue
        if abs(longitude - expected_offer.longitude) > 0.0015:
            continue
        return row
    raise HotelPriceError(
        "response_invalid",
        "The hotel detail provider did not prove the same property identity.",
    )


def _parse_room_rates(
    row: dict[str, Any],
    observed_at: datetime,
) -> tuple[HotelRoomRate, ...]:
    featured = row.get("featured_prices")
    if not isinstance(featured, list):
        return ()
    deduplicated: dict[tuple[Any, ...], HotelRoomRate] = {}
    for seller in featured[:20]:
        if not isinstance(seller, dict):
            continue
        source = _safe_text(seller.get("source"), 160)
        rooms = seller.get("rooms")
        if source is None or not isinstance(rooms, list):
            continue
        seller_link = _safe_public_url(seller.get("link"))
        official = _optional_bool(seller.get("official"))
        for room in rooms[:30]:
            if not isinstance(room, dict):
                continue
            room_name = _safe_text(room.get("name"), 200)
            if room_name is None:
                continue
            room_link = _safe_public_url(room.get("link")) or seller_link
            raw_rates = room.get("rates")
            nested_rates = (
                [item for item in raw_rates[:20] if isinstance(item, dict)]
                if isinstance(raw_rates, list)
                else []
            )
            priced_nested_rates = [
                item
                for item in nested_rates
                if any(
                    value is not None
                    for value in (
                        _nested_price(item, "rate_per_night"),
                        _nested_price(item, "total_rate"),
                        _nested_before_taxes(item, "rate_per_night"),
                        _nested_before_taxes(item, "total_rate"),
                    )
                )
            ]
            # A room-level amount is a separate aggregate. Never attach it to
            # a nested rate's cancellation/breakfast terms when that rate did
            # not return its own price evidence.
            rate_rows = priced_nested_rates or [room]
            for rate in rate_rows:
                nightly = _nested_price(rate, "rate_per_night")
                total = _nested_price(rate, "total_rate")
                nightly_before = _nested_before_taxes(rate, "rate_per_night")
                total_before = _nested_before_taxes(rate, "total_rate")
                if all(
                    value is None
                    for value in (nightly, total, nightly_before, total_before)
                ):
                    continue
                guests = (
                    _bounded_positive_int(rate.get("num_guests"), 50)
                    or _bounded_positive_int(room.get("num_guests"), 50)
                    or _bounded_positive_int(seller.get("num_guests"), 50)
                )
                cancellation_date = _safe_text(
                    rate.get("free_cancellation_until_date"),
                    80,
                )
                cancellation_time = _safe_text(
                    rate.get("free_cancellation_until_time"),
                    80,
                )
                cancellation_until = " ".join(
                    item for item in (cancellation_date, cancellation_time) if item
                ) or None
                beds = _parse_beds(rate.get("beds"))
                inclusions = _safe_text_list(
                    rate.get("inclusions"),
                    limit=20,
                    item_limit=160,
                )
                booking_url = _safe_public_url(rate.get("link")) or room_link
                parsed = HotelRoomRate(
                    room_name=room_name,
                    source=source,
                    nightly_price=nightly,
                    total_price=total,
                    nightly_before_taxes=nightly_before,
                    total_before_taxes=total_before,
                    currency=_SAFE_CURRENCY,
                    guests=guests,
                    official=official,
                    free_cancellation=_optional_bool(rate.get("free_cancellation")),
                    free_cancellation_until=cancellation_until,
                    breakfast_included=_optional_bool(rate.get("breakfast_included")),
                    beds=beds,
                    inclusions=inclusions,
                    booking_url=booking_url,
                    observed_at=observed_at,
                )
                key = (
                    parsed.source.casefold(),
                    parsed.room_name.casefold(),
                    parsed.nightly_price,
                    parsed.total_price,
                    parsed.guests,
                    parsed.free_cancellation,
                    parsed.breakfast_included,
                )
                deduplicated[key] = parsed
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (
                item.room_name.casefold(),
                item.total_price or math.inf,
                item.nightly_price or math.inf,
                item.source.casefold(),
            ),
        )[:120]
    )


def _parse_review_sources(
    row: dict[str, Any],
    observed_at: datetime,
) -> tuple[HotelReviewSource, ...]:
    sources: list[HotelReviewSource] = []
    google_score = _bounded_float(row.get("overall_rating"), 0, 5)
    google_count = _nonnegative_int(row.get("reviews"))
    if google_score is not None or google_count is not None:
        sources.append(
            HotelReviewSource(
                source="Google",
                score=google_score,
                max_score=5.0 if google_score is not None else None,
                review_count=google_count,
                sample_author=None,
                sample_date=None,
                sample_score=None,
                sample_max_score=None,
                sample_comment=None,
                review_url=None,
                observed_at=observed_at,
            )
        )
    other_reviews = row.get("other_reviews")
    if not isinstance(other_reviews, list):
        return tuple(sources)
    for item in other_reviews[:20]:
        if not isinstance(item, dict):
            continue
        source = _safe_text(item.get("source"), 160)
        if source is None or source.casefold() == "google":
            continue
        source_rating = item.get("source_rating")
        score = (
            _nonnegative_float(source_rating.get("score"))
            if isinstance(source_rating, dict)
            else None
        )
        max_score = (
            _positive_float(source_rating.get("max_score"))
            if isinstance(source_rating, dict)
            else None
        )
        if score is None or max_score is None or score > max_score or max_score > 100:
            score = None
            max_score = None
        user_review = item.get("user_review")
        if not isinstance(user_review, dict):
            user_review = {}
        sample_rating = user_review.get("rating")
        sample_score = (
            _nonnegative_float(sample_rating.get("score"))
            if isinstance(sample_rating, dict)
            else None
        )
        sample_max = (
            _positive_float(sample_rating.get("max_score"))
            if isinstance(sample_rating, dict)
            else None
        )
        if (
            sample_score is None
            or sample_max is None
            or sample_score > sample_max
            or sample_max > 100
        ):
            sample_score = None
            sample_max = None
        comment = _safe_text(user_review.get("comment"), 800)
        review_count = _nonnegative_int(item.get("reviews"))
        if score is None and review_count is None and comment is None:
            continue
        sources.append(
            HotelReviewSource(
                source=source,
                score=score,
                max_score=max_score,
                review_count=review_count,
                sample_author=_safe_text(user_review.get("username"), 160),
                sample_date=_safe_text(user_review.get("date"), 100),
                sample_score=sample_score,
                sample_max_score=sample_max,
                sample_comment=comment,
                review_url=_safe_public_url(user_review.get("link")),
                observed_at=observed_at,
            )
        )
    deduplicated: dict[str, HotelReviewSource] = {}
    for item in sources:
        current = deduplicated.get(item.source.casefold())
        if current is None or _review_evidence_rank(item) > _review_evidence_rank(current):
            deduplicated[item.source.casefold()] = item
    return tuple(deduplicated.values())


def _parse_beds(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    beds: list[str] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        bed_type = _safe_text(item.get("type"), 80)
        count = _bounded_positive_int(item.get("count"), 20)
        if bed_type is None:
            continue
        beds.append(f"{count} × {bed_type}" if count is not None else bed_type)
    return tuple(dict.fromkeys(beds))


def _nested_before_taxes(container: dict[str, Any], key: str) -> float | None:
    value = container.get(key)
    if not isinstance(value, dict):
        return None
    return _positive_float(value.get("extracted_before_taxes_fees"))


def _review_evidence_rank(item: HotelReviewSource) -> tuple[int, int, int]:
    return (
        int(item.sample_comment is not None),
        int(item.score is not None),
        item.review_count or 0,
    )


def _normalized_hotel_name(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _exact_property_query(name: str, city_query: str) -> str | None:
    hotel_name = _safe_text(name, 120)
    city = _safe_text(city_query, 120)
    if hotel_name is None or city is None:
        return None
    if city.casefold() == hotel_name.casefold() or city.casefold().startswith(
        f"{hotel_name.casefold()},"
    ):
        return city
    combined = f"{hotel_name}, {city}"
    return combined if len(combined) <= 120 else hotel_name


def _same_hotel_identity(
    candidate: HotelPriceOffer,
    expected: HotelPriceOffer,
) -> bool:
    return bool(
        _normalized_hotel_name(candidate.name)
        == _normalized_hotel_name(expected.name)
        and abs(candidate.latitude - expected.latitude) <= 0.0015
        and abs(candidate.longitude - expected.longitude) <= 0.0015
    )


def _detail_temporarily_unavailable(offer: HotelPriceOffer) -> HotelPriceOffer:
    return replace(
        offer,
        room_rates_status=(
            offer.room_rates_status
            if offer.room_rates_status != "not_requested"
            else "temporarily_unavailable"
        ),
        review_sources_status=(
            offer.review_sources_status
            if offer.review_sources_status != "not_requested"
            else "temporarily_unavailable"
        ),
    )


def _nested_price(container: dict[str, Any], key: str) -> float | None:
    value = container.get(key)
    if not isinstance(value, dict):
        return None
    return _positive_float(value.get("extracted_lowest"))


def _price_rank(offer: HotelPriceOffer) -> tuple[float, float, str]:
    return (
        offer.total_price or math.inf,
        offer.nightly_price or math.inf,
        offer.name.casefold(),
    )


def _hotel_class(row: dict[str, Any]) -> int | None:
    value = _positive_int(row.get("extracted_hotel_class"))
    if value is not None and value <= 5:
        return value
    label = _safe_text(row.get("hotel_class"), 80)
    if label is None:
        return None
    match = re.search(r"\b([1-5])(?:-star|\s*star)", label.casefold())
    return int(match.group(1)) if match else None


def _safe_text_list(value: Any, *, limit: int, item_limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _safe_text(item, item_limit)
        if text is None or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


def _safe_text(value: Any, max_length: int) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return None
    text = " ".join(str(value).split())
    if not text or len(text) > max_length or _CONTROL_PATTERN.search(text):
        return None
    return text


def _safe_public_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 2_048 or _CONTROL_PATTERN.search(candidate):
        return None
    try:
        parts = parse.urlsplit(candidate)
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    hostname = (parts.hostname or "").lower().rstrip(".")
    query_names = {name.casefold() for name, _ in parse.parse_qsl(parts.query)}
    sensitive_query = any(
        marker in name
        for name in query_names
        for marker in ("api_key", "token", "secret", "authorization")
    )
    if (
        parts.scheme.lower() != "https"
        or not hostname
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or hostname == "serpapi.com"
        or hostname.endswith(".serpapi.com")
        or sensitive_query
    ):
        return None
    return candidate


def _google_hotels_url(name: str, city_query: str) -> str:
    query = parse.urlencode({"q": f"{name}, {city_query}"})
    return f"https://www.google.com/travel/search?{query}"


def _safe_api_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip()
    return key if key and len(key) <= 512 and not _CONTROL_PATTERN.search(key) else None


def _search_status(payload: Any) -> str:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    return (
        str(metadata.get("status", "")).strip().lower()
        if isinstance(metadata, dict)
        else ""
    )


def _payload_search_id(payload: Any) -> str | None:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("id")
    return value if isinstance(value, str) and _SEARCH_ID_PATTERN.fullmatch(value) else None


def _archive_search_id(url: Any) -> str | None:
    if not isinstance(url, str) or len(url) > 256:
        return None
    try:
        parts = parse.urlsplit(url)
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    if (
        parts.scheme != "https"
        or parts.hostname != "serpapi.com"
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or parts.query
        or parts.fragment
        or not parts.path.startswith("/searches/")
        or not parts.path.endswith(".json")
    ):
        return None
    search_id = parts.path.removeprefix("/searches/").removesuffix(".json")
    return search_id if _SEARCH_ID_PATTERN.fullmatch(search_id) else None


def _failure_code(value: Any) -> HotelPriceErrorCode:
    allowed = {
        "authentication_failed",
        "quota_exhausted",
        "rate_limited",
        "provider_processing",
        "provider_error",
        "provider_unavailable",
        "response_invalid",
        "quota_ledger_unavailable",
    }
    if value not in allowed:
        raise ValueError("hotel failure guard code is invalid")
    return value


def _guarded_failure(code: HotelPriceErrorCode) -> HotelPriceError:
    messages = {
        "authentication_failed": "The recent hotel price request failed authentication.",
        "quota_exhausted": "The shared free SerpApi quota was recently exhausted.",
        "rate_limited": "The hotel price provider was recently rate limited.",
        "provider_processing": "The hotel price provider is still processing this stay.",
        "provider_error": "The hotel price provider recently rejected this stay.",
        "provider_unavailable": "The hotel price provider was recently unavailable.",
        "response_invalid": "The recent hotel price response could not be verified.",
        "quota_ledger_unavailable": "The shared quota ledger was recently unavailable.",
    }
    message = messages.get(code)
    if message is None:
        raise ValueError("hotel failure guard code is invalid")
    return HotelPriceError(code, message)


def _response_status(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", getattr(response, "status", 0)))
    except (TypeError, ValueError):
        return 0


def _response_json(response: Any) -> Any:
    try:
        content = getattr(response, "content", None)
        text = getattr(response, "text", None)
    except Exception as exc:
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider returned an unreadable response.",
        ) from exc
    if isinstance(content, bytes) and len(content) > HOTEL_PRICE_MAX_RESPONSE_BYTES:
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider response exceeded the safety limit.",
        )
    if isinstance(text, str) and len(text.encode("utf-8")) > HOTEL_PRICE_MAX_RESPONSE_BYTES:
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider response exceeded the safety limit.",
        )
    try:
        payload = response.json()
        size = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except Exception as exc:
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider returned invalid JSON.",
        ) from exc
    if size > HOTEL_PRICE_MAX_RESPONSE_BYTES:
        raise HotelPriceError(
            "response_invalid",
            "The hotel price provider response exceeded the safety limit.",
        )
    return payload


def _safe_cache_json(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    lowered = serialized.casefold()
    if any(token in lowered for token in ("api_key", "property_token", "serpapi.com")):
        raise ValueError("hotel cache payload contains a forbidden provider field")
    return serialized


def _result_from_cache(payload: Any) -> HotelPriceSearchResult:
    if not isinstance(payload, dict):
        raise ValueError("hotel cache result is invalid")
    raw_offers = payload.get("offers")
    if not isinstance(raw_offers, list):
        raise ValueError("hotel cache offers are invalid")
    offers = tuple(_offer_from_dict(item) for item in raw_offers)
    status = str(payload.get("status", ""))
    if status not in {"available", "no_results"}:
        raise ValueError("hotel cache status is invalid")
    return HotelPriceSearchResult(
        offers=offers,
        status=status,
        observed_at=_datetime_value(payload.get("observed_at")),
        cache_hit=True,
        calls_reserved=0,
        quota_monthly_used=None,
        quota_monthly_limit=None,
        quota_hourly_used=None,
        quota_hourly_limit=None,
    )


def _offer_from_dict(payload: Any) -> HotelPriceOffer:
    if not isinstance(payload, dict):
        raise ValueError("hotel cache detail is invalid")
    return HotelPriceOffer(
        hotel_id=str(payload.get("hotel_id", "")),
        name=str(payload.get("name", "")),
        property_type=str(payload.get("property_type", "")),
        latitude=float(payload.get("latitude")),
        longitude=float(payload.get("longitude")),
        description=(
            str(payload["description"]) if payload.get("description") is not None else None
        ),
        hotel_class=_optional_int(payload.get("hotel_class")),
        rating=_optional_float(payload.get("rating")),
        review_count=_optional_int(payload.get("review_count")),
        nightly_price=_optional_float(payload.get("nightly_price")),
        total_price=_optional_float(payload.get("total_price")),
        currency=str(payload.get("currency", "")),
        price_source=(
            str(payload["price_source"]) if payload.get("price_source") is not None else None
        ),
        free_cancellation=_optional_bool(payload.get("free_cancellation")),
        amenities=tuple(payload.get("amenities") or ()),
        website_url=str(payload.get("website_url", "")),
        observed_at=_datetime_value(payload.get("observed_at")),
        room_rates=tuple(
            _room_rate_from_dict(item)
            for item in (payload.get("room_rates") or ())
        ),
        review_sources=tuple(
            _review_source_from_dict(item)
            for item in (payload.get("review_sources") or ())
        ),
        room_rates_status=str(payload.get("room_rates_status", "not_requested")),
        review_sources_status=str(
            payload.get("review_sources_status", "not_requested")
        ),
        detail_observed_at=(
            _datetime_value(payload.get("detail_observed_at"))
            if payload.get("detail_observed_at") is not None
            else None
        ),
        detail_fetch_complete=payload.get("detail_fetch_complete") is True,
    )


def _room_rate_from_dict(payload: Any) -> HotelRoomRate:
    if not isinstance(payload, dict):
        raise ValueError("cached hotel room rate is invalid")
    return HotelRoomRate(
        room_name=str(payload.get("room_name", "")),
        source=str(payload.get("source", "")),
        nightly_price=_optional_float(payload.get("nightly_price")),
        total_price=_optional_float(payload.get("total_price")),
        nightly_before_taxes=_optional_float(payload.get("nightly_before_taxes")),
        total_before_taxes=_optional_float(payload.get("total_before_taxes")),
        currency=str(payload.get("currency", "")),
        guests=_optional_int(payload.get("guests")),
        official=_optional_bool(payload.get("official")),
        free_cancellation=_optional_bool(payload.get("free_cancellation")),
        free_cancellation_until=(
            str(payload["free_cancellation_until"])
            if payload.get("free_cancellation_until") is not None
            else None
        ),
        breakfast_included=_optional_bool(payload.get("breakfast_included")),
        beds=tuple(payload.get("beds") or ()),
        inclusions=tuple(payload.get("inclusions") or ()),
        booking_url=(
            str(payload["booking_url"])
            if payload.get("booking_url") is not None
            else None
        ),
        observed_at=_datetime_value(payload.get("observed_at")),
    )


def _review_source_from_dict(payload: Any) -> HotelReviewSource:
    if not isinstance(payload, dict):
        raise ValueError("cached hotel review source is invalid")
    return HotelReviewSource(
        source=str(payload.get("source", "")),
        score=_optional_float(payload.get("score")),
        max_score=_optional_float(payload.get("max_score")),
        review_count=_optional_int(payload.get("review_count")),
        sample_author=(
            str(payload["sample_author"])
            if payload.get("sample_author") is not None
            else None
        ),
        sample_date=(
            str(payload["sample_date"])
            if payload.get("sample_date") is not None
            else None
        ),
        sample_score=_optional_float(payload.get("sample_score")),
        sample_max_score=_optional_float(payload.get("sample_max_score")),
        sample_comment=(
            str(payload["sample_comment"])
            if payload.get("sample_comment") is not None
            else None
        ),
        review_url=(
            str(payload["review_url"])
            if payload.get("review_url") is not None
            else None
        ),
        observed_at=_datetime_value(payload.get("observed_at")),
    )


def _cached_deadline(value: Any) -> datetime:
    return _datetime_value(value)


def _datetime_value(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("cached datetime is invalid")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("cached datetime is invalid")
    return parsed.astimezone(UTC)


def _renewal_date(value: Any, received_at: datetime) -> date | None:
    if not isinstance(value, str) or len(value.strip()) != 10:
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        return None
    days = (parsed - received_at.date()).days
    return parsed if 0 <= days <= 62 and parsed.isoformat() == value.strip() else None


def _date_value(value: date | str) -> date | None:
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or len(value.strip()) != 10:
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value.strip() else None


def _coordinate(value: Any, low: float, high: float) -> float | None:
    number = _optional_float(value)
    return number if number is not None and low <= number <= high else None


def _positive_float(value: Any) -> float | None:
    number = _optional_float(value)
    return round(number, 2) if number is not None and number > 0 else None


def _nonnegative_float(value: Any) -> float | None:
    number = _optional_float(value)
    return round(number, 2) if number is not None and number >= 0 else None


def _bounded_float(value: Any, low: float, high: float) -> float | None:
    number = _optional_float(value)
    return round(number, 2) if number is not None and low <= number <= high else None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _positive_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _bounded_positive_int(value: Any, high: int) -> int | None:
    parsed = _positive_int(value)
    return parsed if parsed is not None and parsed <= high else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    try:
        equivalent = float(value) == parsed
    except (TypeError, ValueError):
        equivalent = False
    return parsed if equivalent else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _bounded_monthly_limit(value: Any) -> int:
    parsed = _positive_int(value)
    return min(parsed or SERPAPI_DEFAULT_MONTHLY_LIMIT, SERPAPI_MAX_MONTHLY_LIMIT)


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("hotel provider clock must be a datetime")
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
