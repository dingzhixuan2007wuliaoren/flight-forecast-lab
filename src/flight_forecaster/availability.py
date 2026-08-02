"""Strict verification of bookable Google Flights offers through SerpApi.

The adapter deliberately fails closed.  A Google Flights search result is only
a candidate: it must contain a ``booking_token`` and a second SerpApi request
must return the identical selected itinerary plus a concrete booking option.
No flight number, clock time, price, or booking link is synthesized here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Literal
from urllib import error, parse, request

from flight_forecaster.quota_status import QuotaLedgerSnapshot
from flight_forecaster.route_info import AIRPORTS
from flight_forecaster.schemas import MAX_STRICT_ITINERARY_SEGMENTS

Cabin = Literal["economy", "premium_economy", "business", "first"]
BookingUrlKind = Literal["direct_get", "google_flights_itinerary"]
SearchStatus = Literal[
    "confirmed_offers",
    "no_results",
    "not_configured",
    "test_environment_rejected",
    "authentication_failed",
    "rate_limited",
    "budget_not_configured",
    "budget_exhausted",
    "provider_processing",
    "provider_error",
    "provider_unavailable",
]
CandidateCoverageStatus = Literal[
    "not_evaluated",
    "complete",
    "quota_limited",
    "provider_incomplete",
    "quota_and_provider_incomplete",
]

SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
SERPAPI_ACCOUNT_URL = "https://serpapi.com/account.json"
SERPAPI_SEARCH_ARCHIVE_URL = "https://serpapi.com/searches/{search_id}.json"
SERPAPI_PROVIDER_CODE = "serpapi_google_flights"
SERPAPI_PROVIDER_NAME = "SerpApi Google Flights"
SEARCHAPI_PROVIDER_CODE = "searchapi_google_flights"
SEARCHAPI_PROVIDER_NAME = "SearchAPI.io Google Flights"
SCRAPPA_PROVIDER_CODE = "scrappa_google_flights"
SCRAPPA_PROVIDER_NAME = "Scrappa Google Flights"
IGNAV_QUARANTINE_PROVIDER_CODE = "ignav_quarantine"
IGNAV_QUARANTINE_PROVIDER_NAME = "Ignav (strict quarantine)"
IGNAV_VERIFIED_PROVIDER_CODE = "ignav_verified_fares"
IGNAV_VERIFIED_PROVIDER_NAME = "Ignav Verified Fares"
AGGREGATE_PROVIDER_CODE = "strict_fare_aggregate"
AGGREGATE_PROVIDER_NAME = "Strict Fare Provider Aggregate"
# Backward-compatible aliases deliberately continue to mean the quarantined
# identity.  Quarantined evidence can never construct a ConfirmedFlightOffer.
IGNAV_PROVIDER_CODE = IGNAV_QUARANTINE_PROVIDER_CODE
IGNAV_PROVIDER_NAME = IGNAV_QUARANTINE_PROVIDER_NAME
SERPAPI_DEFAULT_MONTHLY_LIMIT = 250
SERPAPI_MAX_MONTHLY_LIMIT = 250
FLIGHT_OFFER_CACHE_TTL_SECONDS = 300.0
MAX_PROVIDER_RESPONSE_BYTES = 5_000_000
REQUEST_TIMEOUT_SECONDS = 15.0
# SerpApi documents ``deep_search=true`` as the browser-equivalent, fuller
# search path and warns that it takes longer.  Keep booking-option requests at
# the ordinary timeout, but do not turn a slow deep cabin search into a false
# provider failure after only 15 seconds.
DEEP_SEARCH_REQUEST_TIMEOUT_SECONDS = 45.0
ACCOUNT_REQUEST_TIMEOUT_SECONDS = 8.0
SEARCH_ARCHIVE_REQUEST_TIMEOUT_SECONDS = 2.0
MAX_CACHE_ENTRIES = 128
# This bounds parallel network pressure only. It is not a candidate-count cap:
# every eligible provider-returned candidate for which quota can be reserved is
# submitted through this worker pool.
MAX_BOOKING_WORKERS = 6
SERPAPI_MAX_HOURLY_LIMIT = 50
MAX_PROVIDER_CACHE_AGE_SECONDS = 65 * 60
MAX_PROVIDER_FUTURE_SKEW_SECONDS = 5 * 60
PROVIDER_CACHE_HIT_AGE_SECONDS = 60
MAX_SERPAPI_PRICE_HISTORY_POINTS = 400
MIN_SERPAPI_PRICE_HISTORY_TIMESTAMP = int(datetime(2000, 1, 1, tzinfo=UTC).timestamp())
# Poll only the original allowlisted Search ID.  The roughly 15-second bounded
# window is long enough for asynchronous deep searches without submitting a
# second paid search just because the ordinary five-second window was short.
PROVIDER_POLL_DELAYS_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0)
MAX_PROVIDER_DIAGNOSTICS = 10
MAX_PERSISTED_PROVIDER_DIAGNOSTICS = 500

_CABINS: tuple[Cabin, ...] = (
    "economy",
    "premium_economy",
    "business",
    "first",
)
_SERPAPI_TRAVEL_CLASSES: dict[Cabin, int] = {
    "economy": 1,
    "premium_economy": 2,
    "business": 3,
    "first": 4,
}
_GOOGLE_TRAVEL_CLASSES: dict[str, Cabin] = {
    "economy": "economy",
    "premium economy": "premium_economy",
    "business": "business",
    "first": "first",
}
_IATA_PATTERN = re.compile(r"^[A-Z]{3}$")
_AIRLINE_PATTERN = re.compile(r"^[A-Z0-9]{2,3}$")
_FLIGHT_NUMBER_PATTERN = re.compile(r"^[A-Z0-9]{1,8}$")
_FULL_FLIGHT_NUMBER_PATTERN = re.compile(r"^([A-Z0-9]{2,3})\s+([A-Z0-9]{1,8})$")
_SAFE_SHORT_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]+$")
_SEARCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_EXCEPTION_TYPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _google_market_for_origin(origin: str) -> str:
    """Return a safe Google Flights ``gl`` market from the origin airport.

    Google uses ``uk`` for the United Kingdom even though the airport catalog
    carries the ISO country code ``GB``.  Unknown airports deliberately retain
    the prior US-market fallback rather than guessing from the destination.
    """

    code = _iata(origin)
    airport = AIRPORTS.get(code) if code is not None else None
    country = airport.country.strip().upper() if airport is not None else ""
    if country == "GB":
        return "uk"
    if re.fullmatch(r"[A-Z]{2}", country):
        return country.lower()
    return "us"

DiagnosticStage = Literal[
    "account",
    "cabin_search",
    "booking_options",
    "search_archive",
    "validation",
]


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """A deliberately small, secret-free provider diagnostic record."""

    observed_at: datetime
    stage: DiagnosticStage
    http_status: int | None
    exception_type: str
    search_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ValueError("provider diagnostic HTTP status is invalid")
        if not _EXCEPTION_TYPE_PATTERN.fullmatch(self.exception_type):
            raise ValueError("provider diagnostic exception type is invalid")
        if self.search_id is not None and not _SEARCH_ID_PATTERN.fullmatch(self.search_id):
            raise ValueError("provider diagnostic search ID is invalid")


@dataclass(frozen=True, slots=True)
class FlightOfferSegment:
    segment_id: str
    origin: str
    destination: str
    departure_at: datetime
    arrival_at: datetime
    marketing_airline_code: str
    operating_airline_code: str | None
    flight_number: str
    departure_terminal: str | None
    arrival_terminal: str | None
    aircraft_icao: str | None
    cabin: Cabin
    booking_class: str | None
    fare_basis: str | None
    fare_brand: str | None
    checked_bags_quantity: int | None
    checked_bags_weight: int | None
    checked_bags_weight_unit: str | None

    def __post_init__(self) -> None:
        if self.departure_at.tzinfo is not None or self.arrival_at.tzinfo is not None:
            raise ValueError("segment times must be naive provider-local datetimes")
        if self.cabin not in _CABINS:
            raise ValueError("unsupported cabin")


@dataclass(frozen=True, slots=True)
class ConfirmedFlightOffer:
    provider_offer_id: str
    validating_airline_code: str
    airline_name: str
    cabin: Cabin
    total_amount_usd: float
    base_amount_usd: float | None
    last_ticketing_date: date | None
    number_of_bookable_seats: int | None
    seat_count_capped: bool
    verified_at: datetime
    provider_cache_hit: bool
    provider_cache_age_seconds: int
    segments: tuple[FlightOfferSegment, ...]
    refundable_fare: bool | None
    no_penalty_fare: bool | None
    no_restriction_fare: bool | None
    booking_url: str
    booking_url_kind: BookingUrlKind
    booking_provider: str
    booking_verified: bool = True
    provider_code: str = SERPAPI_PROVIDER_CODE
    provider_name: str = SERPAPI_PROVIDER_NAME
    environment: str = "production"

    def __post_init__(self) -> None:
        object.__setattr__(self, "verified_at", _utc(self.verified_at))
        object.__setattr__(self, "segments", tuple(self.segments))
        if self.cabin not in _CABINS or not self.segments:
            raise ValueError("confirmed offer is incomplete")
        if _finite_amount(self.total_amount_usd) is None or self.total_amount_usd <= 0:
            raise ValueError("confirmed offer price is invalid")
        if not self.booking_verified:
            raise ValueError("confirmed offer requires verified booking evidence")
        allowed_provider_names = {
            SERPAPI_PROVIDER_CODE: SERPAPI_PROVIDER_NAME,
            SEARCHAPI_PROVIDER_CODE: SEARCHAPI_PROVIDER_NAME,
            SCRAPPA_PROVIDER_CODE: SCRAPPA_PROVIDER_NAME,
            IGNAV_VERIFIED_PROVIDER_CODE: IGNAV_VERIFIED_PROVIDER_NAME,
        }
        if allowed_provider_names.get(self.provider_code) != self.provider_name:
            raise ValueError("confirmed offer provider is invalid")
        if self.environment != "production":
            raise ValueError("confirmed offer environment is invalid")
        if not isinstance(self.provider_cache_hit, bool):
            raise ValueError("confirmed offer provider cache flag is invalid")
        if (
            isinstance(self.provider_cache_age_seconds, bool)
            or not isinstance(self.provider_cache_age_seconds, int)
            or not 0 <= self.provider_cache_age_seconds <= MAX_PROVIDER_CACHE_AGE_SECONDS
        ):
            raise ValueError("confirmed offer provider cache age is invalid")
        if not _safe_https_url(self.booking_url):
            raise ValueError("confirmed offer booking URL is invalid")
        if self.booking_url_kind not in {"direct_get", "google_flights_itinerary"}:
            raise ValueError("confirmed offer booking URL kind is invalid")
        if not _short_text(self.booking_provider, max_length=160):
            raise ValueError("confirmed offer booking provider is invalid")

    @property
    def source_url(self) -> str:
        """Return the verified, user-openable booking evidence URL."""

        return self.booking_url

    @property
    def fingerprint(self) -> str:
        """Return a provider-id-independent identity for the itinerary and fare."""

        identity = {
            "validating_airline": self.validating_airline_code,
            "cabin": self.cabin,
            "segments": [
                {
                    "origin": segment.origin,
                    "destination": segment.destination,
                    "departure": segment.departure_at.isoformat(timespec="minutes"),
                    "arrival": segment.arrival_at.isoformat(timespec="minutes"),
                    "marketing_airline": segment.marketing_airline_code,
                    "operating_airline": segment.operating_airline_code,
                    "flight_number": segment.flight_number,
                    "cabin": segment.cabin,
                    "booking_class": segment.booking_class,
                    "fare_basis": segment.fare_basis,
                    "fare_brand": segment.fare_brand,
                }
                for segment in self.segments
            ],
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def lowest_price_group_key(self) -> tuple[Any, ...]:
        """Group equivalent flight-and-cabin offers before choosing a seller.

        Seller, fare brand, booking URL, and price are deliberately absent: a
        provider response may expose the same dated itinerary through multiple
        booking tokens or sellers, and strict mode keeps only the cheapest
        successfully verified option for that flight and cabin.
        """

        return (
            self.cabin,
            tuple(
                (
                    segment.origin,
                    segment.destination,
                    segment.departure_at,
                    segment.arrival_at,
                    segment.marketing_airline_code,
                    segment.operating_airline_code,
                    segment.flight_number,
                    segment.cabin,
                )
                for segment in self.segments
            ),
        )


@dataclass(frozen=True, slots=True)
class RouteCabinMarketPricePoint:
    """One sanitized query-level Google Flights market-history observation."""

    observed_at: datetime
    price_usd: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        amount = _finite_amount(self.price_usd)
        if amount is None or amount <= 0:
            raise ValueError("market-history price must be a positive finite amount")
        object.__setattr__(self, "price_usd", amount)


@dataclass(frozen=True, slots=True)
class RouteCabinMarketHistory:
    """Sanitized SerpApi query history, never a selected-offer price history."""

    origin: str
    destination: str
    departure_date: date
    cabin: Cabin
    provider_observed_at: datetime
    points: tuple[RouteCabinMarketPricePoint, ...]
    provider_code: Literal["serpapi_google_flights"] = SERPAPI_PROVIDER_CODE
    provider_name: Literal["SerpApi Google Flights"] = SERPAPI_PROVIDER_NAME
    currency: Literal["USD"] = "USD"
    scope: Literal["route_departure_date_cabin_market"] = (
        "route_departure_date_cabin_market"
    )

    def __post_init__(self) -> None:
        origin = _iata(self.origin)
        destination = _iata(self.destination)
        if (
            origin is None
            or destination is None
            or origin == destination
            or isinstance(self.departure_date, datetime)
            or not isinstance(self.departure_date, date)
            or self.cabin not in _CABINS
        ):
            raise ValueError("market-history query identity is invalid")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "destination", destination)
        observed_at = _utc(self.provider_observed_at)
        object.__setattr__(self, "provider_observed_at", observed_at)
        points = tuple(self.points)
        object.__setattr__(self, "points", points)
        if not 1 <= len(points) <= MAX_SERPAPI_PRICE_HISTORY_POINTS:
            raise ValueError("market-history point count is invalid")
        point_times = [point.observed_at for point in points]
        if point_times != sorted(set(point_times)):
            raise ValueError("market-history timestamps must be unique and increasing")
        latest_allowed = observed_at.timestamp() + MAX_PROVIDER_FUTURE_SKEW_SECONDS
        if any(point.observed_at.timestamp() > latest_allowed for point in points):
            raise ValueError("market-history contains a future observation")
        if (
            self.provider_code != SERPAPI_PROVIDER_CODE
            or self.provider_name != SERPAPI_PROVIDER_NAME
            or self.currency != "USD"
            or self.scope != "route_departure_date_cabin_market"
        ):
            raise ValueError("market-history provider attribution is invalid")


@dataclass(frozen=True, slots=True)
class FlightOfferSearchResult:
    """Strict offers plus conservative local request-attempt accounting.

    The ``calls`` and ``monthly_used`` fields are safety-ledger values. They can
    exceed SerpApi's billed searches because cached, failed, and ambiguous
    requests are deliberately reserved too.
    """

    offers: tuple[ConfirmedFlightOffer, ...]
    status: SearchStatus
    observed_at: datetime
    environment: str
    searched_cabins: tuple[Cabin, ...]
    calls_used: int
    cache_hit: bool
    search_calls_used: int = 0
    pricing_calls_used: int = 0
    search_monthly_limit: int | None = None
    pricing_monthly_limit: int | None = None
    search_monthly_used: int | None = None
    pricing_monthly_used: int | None = None
    archive_poll_count: int = 0
    diagnostics: tuple[ProviderDiagnostic, ...] = ()
    coverage_scope: Literal["provider_returned_booking_verification_candidates"] = (
        "provider_returned_booking_verification_candidates"
    )
    eligible_candidate_count: int = 0
    verification_attempted_count: int = 0
    verified_candidate_count: int = 0
    strictly_rejected_candidate_count: int = 0
    provider_failed_candidate_count: int = 0
    search_failed_cabin_count: int = 0
    quota_skipped_candidate_count: int = 0
    deduplicated_verified_count: int = 0
    coverage_status: CandidateCoverageStatus = "not_evaluated"
    quota_limit: QuotaLimit | None = None
    retry_quota_limited: bool = False
    provider_code: str = SERPAPI_PROVIDER_CODE
    provider_name: str = SERPAPI_PROVIDER_NAME
    provider_runs: tuple[FlightOfferSearchResult, ...] = ()
    historical_market_contexts: tuple[RouteCabinMarketHistory, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "offers", tuple(self.offers))
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        object.__setattr__(self, "searched_cabins", tuple(self.searched_cabins))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "provider_runs", tuple(self.provider_runs))
        object.__setattr__(
            self,
            "historical_market_contexts",
            tuple(self.historical_market_contexts),
        )
        allowed_provider_names = {
            "none": "No strict fare provider",
            SERPAPI_PROVIDER_CODE: SERPAPI_PROVIDER_NAME,
            SEARCHAPI_PROVIDER_CODE: SEARCHAPI_PROVIDER_NAME,
            SCRAPPA_PROVIDER_CODE: SCRAPPA_PROVIDER_NAME,
            IGNAV_QUARANTINE_PROVIDER_CODE: IGNAV_QUARANTINE_PROVIDER_NAME,
            IGNAV_VERIFIED_PROVIDER_CODE: IGNAV_VERIFIED_PROVIDER_NAME,
            AGGREGATE_PROVIDER_CODE: AGGREGATE_PROVIDER_NAME,
        }
        if allowed_provider_names.get(self.provider_code) != self.provider_name:
            raise ValueError("fare-search provider identity is invalid")
        if self.provider_code == AGGREGATE_PROVIDER_CODE:
            if len(self.provider_runs) < 2:
                raise ValueError("aggregate fare-search results require at least two provider runs")
            if any(run.provider_runs for run in self.provider_runs):
                raise ValueError("aggregate provider runs cannot be nested")
            run_identities = [
                (run.provider_code, run.provider_name) for run in self.provider_runs
            ]
            if len(run_identities) != len(set(run_identities)):
                raise ValueError("aggregate provider runs must identify unique providers")
            if any(
                not any(
                    offer in run.offers
                    and (offer.provider_code, offer.provider_name)
                    == (run.provider_code, run.provider_name)
                    for run in self.provider_runs
                )
                for offer in self.offers
            ):
                raise ValueError("every aggregate offer must retain a matching provider run")
            if self.status == "confirmed_offers" and not self.offers:
                raise ValueError("confirmed aggregate results require at least one offer")
            if self.offers and self.status != "confirmed_offers":
                raise ValueError("aggregate offers require confirmed-offers status")
        else:
            if self.provider_runs:
                raise ValueError("single-provider results cannot contain provider runs")
            if any(
                (offer.provider_code, offer.provider_name)
                != (self.provider_code, self.provider_name)
                for offer in self.offers
            ):
                raise ValueError("every fare-search offer must match the result provider")
        market_context_keys = [
            (
                context.provider_code,
                context.origin,
                context.destination,
                context.departure_date,
                context.cabin,
            )
            for context in self.historical_market_contexts
        ]
        if len(market_context_keys) != len(set(market_context_keys)):
            raise ValueError("fare-search market histories must be unique")
        if self.provider_code == AGGREGATE_PROVIDER_CODE:
            if self.historical_market_contexts:
                raise ValueError(
                    "aggregate market histories must remain attributed to provider runs"
                )
        elif self.provider_code == SERPAPI_PROVIDER_CODE:
            if any(
                context.provider_code != SERPAPI_PROVIDER_CODE
                for context in self.historical_market_contexts
            ):
                raise ValueError("SerpApi result contains another provider's market history")
        elif self.historical_market_contexts:
            raise ValueError("non-SerpApi results cannot contain SerpApi market history")
        if self.provider_code == IGNAV_QUARANTINE_PROVIDER_CODE and self.offers:
            raise ValueError("quarantined Ignav evidence cannot supply strict offers")
        if self.coverage_scope != "provider_returned_booking_verification_candidates":
            raise ValueError("candidate coverage scope is invalid")
        count_fields = {
            "eligible candidates": self.eligible_candidate_count,
            "verification attempts": self.verification_attempted_count,
            "verified candidates": self.verified_candidate_count,
            "strictly rejected candidates": self.strictly_rejected_candidate_count,
            "provider-failed candidates": self.provider_failed_candidate_count,
            "failed cabin searches": self.search_failed_cabin_count,
            "quota-skipped candidates": self.quota_skipped_candidate_count,
            "deduplicated verified candidates": self.deduplicated_verified_count,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in count_fields.values()
        ):
            raise ValueError("candidate coverage counts must be nonnegative integers")
        if not isinstance(self.retry_quota_limited, bool):
            raise ValueError("retry quota flag must be boolean")
        if self.retry_quota_limited and not (
            self.provider_failed_candidate_count or self.search_failed_cabin_count
        ):
            raise ValueError("retry quota limitation requires a failed provider attempt")
        if self.verification_attempted_count != (
            self.verified_candidate_count
            + self.strictly_rejected_candidate_count
            + self.provider_failed_candidate_count
        ):
            raise ValueError("verification attempts must equal all verification outcomes")
        if self.verified_candidate_count != (len(self.offers) + self.deduplicated_verified_count):
            raise ValueError("verified candidates must equal returned plus deduplicated offers")
        if self.coverage_status == "not_evaluated":
            if self.retry_quota_limited:
                raise ValueError("unevaluated candidate coverage cannot be retry-quota limited")
            if any(
                (
                    self.verification_attempted_count,
                    self.verified_candidate_count,
                    self.strictly_rejected_candidate_count,
                    self.provider_failed_candidate_count,
                    self.search_failed_cabin_count,
                    self.quota_skipped_candidate_count,
                    self.deduplicated_verified_count,
                )
            ):
                raise ValueError("unevaluated candidate coverage cannot report outcomes")
        elif self.eligible_candidate_count != (
            self.verification_attempted_count + self.quota_skipped_candidate_count
        ):
            raise ValueError("eligible candidates must equal attempted plus quota-skipped")

        provider_run_incomplete = bool(
            self.provider_runs
            and any(
                run.status not in {"confirmed_offers", "no_results"}
                or run.coverage_status
                in {"provider_incomplete", "quota_and_provider_incomplete"}
                for run in self.provider_runs
            )
        )
        provider_run_quota_limited = bool(
            self.provider_runs
            and any(
                run.status in {"rate_limited", "budget_exhausted"}
                or run.coverage_status
                in {"quota_limited", "quota_and_provider_incomplete"}
                for run in self.provider_runs
            )
        )
        expected_status = _candidate_coverage_status(
            evaluated=self.coverage_status != "not_evaluated",
            provider_failed=(
                self.provider_failed_candidate_count
                or self.search_failed_cabin_count
                or int(provider_run_incomplete)
            ),
            quota_skipped=self.quota_skipped_candidate_count,
            retry_quota_limited=(
                self.retry_quota_limited or provider_run_quota_limited
            ),
        )
        if self.coverage_status != expected_status:
            raise ValueError("candidate coverage status does not match its counts")
        quota_limited = self.coverage_status in {
            "quota_limited",
            "quota_and_provider_incomplete",
        }
        unevaluated_quota_wall = (
            self.coverage_status == "not_evaluated"
            and self.status in {"rate_limited", "budget_exhausted"}
        )
        if not unevaluated_quota_wall and quota_limited != (self.quota_limit is not None):
            raise ValueError("candidate quota limit must match quota-truncated coverage")

    @property
    def configured_search_monthly_limit(self) -> int | None:
        return self.search_monthly_limit

    @property
    def configured_pricing_monthly_limit(self) -> int | None:
        return self.pricing_monthly_limit


class _ProviderError(RuntimeError):
    pass


class _AuthenticationError(_ProviderError):
    pass


class _RateLimitError(_ProviderError):
    pass


class _PayloadError(_ProviderError):
    pass


class _UsageError(_ProviderError):
    pass


class _ProviderHttpError(_ProviderError):
    pass


class _TransientProviderHttpError(_ProviderHttpError):
    pass


class _ProviderSearchError(_ProviderError):
    pass


class _ProviderProcessingError(_ProviderError):
    pass


class _TransportError(_ProviderError):
    pass


class _ControlledRetryQuotaError(_ProviderError):
    def __init__(self, limiting_quota: QuotaLimit | None) -> None:
        super().__init__("controlled provider retry would exceed the quota wall")
        self.limiting_quota = limiting_quota


QuotaLimit = Literal["monthly", "hourly", "lifetime", "provider_specific"]


class _UrllibResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.content = body

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.text) if self.content else {}


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """Refuse redirects so provider credentials never leave the fixed endpoint."""

    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class _UrllibClient:
    """Minimal sync client matching the subset of httpx used by the adapter."""

    def __init__(self, opener: Any | None = None) -> None:
        self._opener = opener or request.build_opener(_NoRedirectHandler())

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> _UrllibResponse:
        query = parse.urlencode(params or {})
        target = f"{url}?{query}" if query else url
        outbound = request.Request(target, headers=headers or {}, method="GET")
        try:
            with self._opener.open(outbound, timeout=timeout) as response:
                response_body = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                status = int(response.status)
        except error.HTTPError as exc:
            response_body = exc.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
            status = int(exc.code)
        if len(response_body) > MAX_PROVIDER_RESPONSE_BYTES:
            raise _PayloadError("provider response exceeded safety limit")
        return _UrllibResponse(status, response_body)


@dataclass(frozen=True, slots=True)
class _QuotaReservation:
    reserved_calls: int
    monthly_used: int
    hourly_used: int
    limiting_quota: QuotaLimit | None


@dataclass(frozen=True, slots=True)
class _AccountQuota:
    billing_cycle_key: str
    hour_bucket_key: str
    monthly_used: int
    hourly_used: int
    monthly_limit: int
    hourly_limit: int
    observed_at: datetime


class _UsageLedger:
    """Cross-process conservative attempt reservations for two provider quotas.

    These counters are not SerpApi billing records. Cached and failed provider
    requests can be free, but reserving every attempt prevents ambiguous network
    outcomes or another process from exceeding either configured hard stop.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def snapshot(self, scope: Literal["billing_cycle", "hour"], period_key: str) -> int:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                """
                SELECT calls FROM serpapi_quota_usage
                WHERE scope = ? AND period_key = ?
                """,
                (scope, period_key),
            ).fetchone()
            return _nonnegative_int(row[0]) if row is not None else 0
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise _UsageError("usage ledger is unreadable") from exc
        finally:
            if connection is not None:
                connection.close()

    def synchronize_and_reserve(
        self,
        billing_cycle_key: str,
        hour_bucket_key: str,
        *,
        calls: int,
        monthly_limit: int,
        hourly_limit: int,
        provider_monthly_usage: int = 0,
        provider_hourly_usage: int = 0,
        require_all: bool,
    ) -> _QuotaReservation:
        """Synchronize both baselines and reserve attempts in one transaction."""

        if (
            calls < 0
            or monthly_limit < 1
            or hourly_limit < 1
            or provider_monthly_usage < 0
            or provider_hourly_usage < 0
            or not billing_cycle_key
            or not hour_bucket_key
        ):
            raise _UsageError("usage ledger reservation is invalid")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            monthly_baseline = max(
                self._read_counter(connection, "billing_cycle", billing_cycle_key),
                provider_monthly_usage,
            )
            hourly_baseline = max(
                self._read_counter(connection, "hour", hour_bucket_key),
                provider_hourly_usage,
            )
            monthly_available = max(0, monthly_limit - monthly_baseline)
            hourly_available = max(0, hourly_limit - hourly_baseline)
            capacity = min(monthly_available, hourly_available)
            if require_all and capacity < calls:
                reserved_calls = 0
            else:
                reserved_calls = min(calls, capacity)
            limiting_quota: QuotaLimit | None = None
            if reserved_calls < calls:
                limiting_quota = "monthly" if monthly_available <= hourly_available else "hourly"
            monthly_used = monthly_baseline + reserved_calls
            hourly_used = hourly_baseline + reserved_calls
            self._write_counter(
                connection,
                "billing_cycle",
                billing_cycle_key,
                monthly_used,
            )
            self._write_counter(connection, "hour", hour_bucket_key, hourly_used)
            connection.commit()
            return _QuotaReservation(
                reserved_calls=reserved_calls,
                monthly_used=monthly_used,
                hourly_used=hourly_used,
                limiting_quota=limiting_quota,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise _UsageError("usage ledger is unwritable") from exc
        finally:
            if connection is not None:
                connection.close()

    def record_diagnostic(self, diagnostic: ProviderDiagnostic) -> None:
        """Persist only the allowlisted diagnostic fields with bounded retention."""

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO serpapi_provider_diagnostics(
                    observed_at,
                    stage,
                    http_status,
                    exception_type,
                    search_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    diagnostic.observed_at.isoformat(),
                    diagnostic.stage,
                    diagnostic.http_status,
                    diagnostic.exception_type,
                    diagnostic.search_id,
                ),
            )
            connection.execute(
                """
                DELETE FROM serpapi_provider_diagnostics
                WHERE diagnostic_id NOT IN (
                    SELECT diagnostic_id
                    FROM serpapi_provider_diagnostics
                    ORDER BY diagnostic_id DESC
                    LIMIT ?
                )
                """,
                (MAX_PERSISTED_PROVIDER_DIAGNOSTICS,),
            )
            connection.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise _UsageError("provider diagnostic ledger is unwritable") from exc
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS serpapi_quota_usage (
                scope TEXT NOT NULL CHECK(scope IN ('billing_cycle', 'hour')),
                period_key TEXT NOT NULL,
                calls INTEGER NOT NULL CHECK(calls >= 0),
                PRIMARY KEY(scope, period_key)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS serpapi_provider_diagnostics (
                diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL,
                stage TEXT NOT NULL CHECK(
                    stage IN (
                        'account',
                        'cabin_search',
                        'booking_options',
                        'search_archive',
                        'validation'
                    )
                ),
                http_status INTEGER CHECK(
                    http_status IS NULL OR http_status BETWEEN 100 AND 599
                ),
                exception_type TEXT NOT NULL,
                search_id TEXT
            )
            """
        )
        return connection

    @staticmethod
    def _read_counter(
        connection: sqlite3.Connection,
        scope: Literal["billing_cycle", "hour"],
        period_key: str,
    ) -> int:
        row = connection.execute(
            """
            SELECT calls FROM serpapi_quota_usage
            WHERE scope = ? AND period_key = ?
            """,
            (scope, period_key),
        ).fetchone()
        return _nonnegative_int(row[0]) if row is not None else 0

    @staticmethod
    def _write_counter(
        connection: sqlite3.Connection,
        scope: Literal["billing_cycle", "hour"],
        period_key: str,
        calls: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO serpapi_quota_usage(scope, period_key, calls)
            VALUES (?, ?, ?)
            ON CONFLICT(scope, period_key) DO UPDATE SET calls = excluded.calls
            """,
            (scope, period_key, calls),
        )


def read_serpapi_quota_snapshot(
    path: str | Path,
    *,
    hard_limit: int,
    now: datetime,
) -> QuotaLedgerSnapshot:
    """Read the nearest non-expired SerpApi renewal ledger without network I/O."""

    ledger_path = Path(path)
    if not ledger_path.is_file():
        return QuotaLedgerSnapshot.unavailable()
    observed = _utc(now)
    limit = min(max(int(hard_limit), 1), SERPAPI_MAX_MONTHLY_LIMIT)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(ledger_path), timeout=1.0)
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT period_key, calls FROM serpapi_quota_usage
            WHERE scope = 'billing_cycle' AND period_key LIKE 'renewal:____-__-__'
            """
        ).fetchall()
        candidates: list[tuple[date, int, str]] = []
        for raw_key, raw_calls in rows:
            key = str(raw_key)
            try:
                renewal = date.fromisoformat(key.removeprefix("renewal:"))
                used = int(raw_calls)
            except (TypeError, ValueError):
                continue
            days = (renewal - observed.date()).days
            if 0 <= days <= 62 and used >= 0:
                candidates.append((renewal, used, key))
        if not candidates:
            return QuotaLedgerSnapshot.unavailable()
        renewal, raw_used, period_key = min(candidates, key=lambda item: item[0])
        used = min(raw_used, limit)
        return QuotaLedgerSnapshot(
            available=True,
            used=used,
            limit=limit,
            remaining=max(0, limit - used),
            period_key=period_key,
            data_basis="conservative_minimum",
            observed_at=observed,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return QuotaLedgerSnapshot.unavailable()
    finally:
        if connection is not None:
            connection.close()


class _DiagnosticCollector:
    """Collect per-comparison diagnostics and persist a bounded local history."""

    def __init__(self, ledger: _UsageLedger) -> None:
        self._ledger = ledger
        self._events: list[ProviderDiagnostic] = []
        self._archive_poll_count = 0
        self._controlled_retry_counts: dict[DiagnosticStage, int] = {}
        self._controlled_retry_monthly_used = 0
        self._lock = threading.Lock()

    def note_archive_poll(self) -> None:
        with self._lock:
            self._archive_poll_count += 1

    @property
    def archive_poll_count(self) -> int:
        with self._lock:
            return self._archive_poll_count

    def note_controlled_retry(
        self,
        stage: Literal["cabin_search", "booking_options"],
        *,
        monthly_used: int,
    ) -> None:
        """Count only a quota-reserved re-submission, never an archive poll."""

        with self._lock:
            self._controlled_retry_counts[stage] = (
                self._controlled_retry_counts.get(stage, 0) + 1
            )
            self._controlled_retry_monthly_used = max(
                self._controlled_retry_monthly_used,
                monthly_used,
            )

    def controlled_retry_count(
        self,
        stage: Literal["cabin_search", "booking_options"],
    ) -> int:
        with self._lock:
            return self._controlled_retry_counts.get(stage, 0)

    def conservative_monthly_used(self, baseline: int | None) -> int | None:
        with self._lock:
            retry_usage = self._controlled_retry_monthly_used
        if baseline is None:
            return retry_usage or None
        return max(baseline, retry_usage)

    def record(
        self,
        *,
        observed_at: datetime,
        stage: DiagnosticStage,
        http_status: int | None,
        exception_type: str,
        search_id: str | None,
    ) -> None:
        diagnostic = ProviderDiagnostic(
            observed_at=observed_at,
            stage=stage,
            http_status=(
                http_status if http_status is not None and 100 <= http_status <= 599 else None
            ),
            exception_type=_safe_exception_type(exception_type),
            search_id=_safe_search_id(search_id),
        )
        with self._lock:
            if diagnostic not in self._events:
                self._events.append(diagnostic)
        try:
            self._ledger.record_diagnostic(diagnostic)
        except _UsageError:
            # Diagnostics must never weaken the provider's fail-closed behavior.
            pass

    def snapshot(self) -> tuple[ProviderDiagnostic, ...]:
        with self._lock:
            events = tuple(self._events)
        ordered = sorted(
            events,
            key=lambda item: (
                item.observed_at,
                item.stage,
                item.http_status or 0,
                item.exception_type,
                item.search_id or "",
            ),
        )
        return tuple(ordered[-MAX_PROVIDER_DIAGNOSTICS:])


@dataclass(frozen=True, slots=True)
class _SearchCandidate:
    booking_token: str
    google_flights_url: str | None
    cabin: Cabin
    search_price_usd: float
    segments: tuple[FlightOfferSegment, ...]
    airline_name: str

    @property
    def primary_airline_code(self) -> str:
        return self.segments[0].marketing_airline_code

    @property
    def identity(self) -> tuple[Any, ...]:
        return tuple(
            (
                segment.origin,
                segment.destination,
                segment.departure_at,
                segment.arrival_at,
                segment.marketing_airline_code,
                segment.flight_number,
                segment.cabin,
            )
            for segment in self.segments
        )


@dataclass(frozen=True, slots=True)
class _BookingEvidence:
    amount_usd: float
    booking_url: str
    booking_url_kind: BookingUrlKind
    booking_provider: str
    fare_brand: str | None
    refundable: bool | None
    no_penalty: bool | None
    no_restriction: bool | None
    checked_bags_quantity: int | None


@dataclass(frozen=True, slots=True)
class _ProviderObservation:
    created_at: datetime
    cache_hit: bool
    cache_age_seconds: int


class NullFlightOfferProvider:
    @property
    def configured(self) -> bool:
        return False

    @property
    def environment(self) -> str:
        return "disabled"

    def search(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        *,
        fetched_at: datetime,
        force_refresh: bool = False,
    ) -> FlightOfferSearchResult:
        del origin, destination, departure_date, force_refresh
        return FlightOfferSearchResult(
            offers=(),
            status="not_configured",
            observed_at=_utc(fetched_at),
            environment=self.environment,
            searched_cabins=(),
            calls_used=0,
            cache_hit=False,
            provider_code="none",
            provider_name="No strict fare provider",
        )


class SerpApiFlightOfferProvider:
    """Google Flights Search-to-Booking adapter with a 250-call hard ceiling."""

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: Path,
        monthly_limit: int | None = SERPAPI_DEFAULT_MONTHLY_LIMIT,
        client: Any = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        now_provider: Callable[[], datetime] | None = None,
        poll_delays_seconds: tuple[float, ...] = PROVIDER_POLL_DELAYS_SECONDS,
        sleep_provider: Callable[[float], None] | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._monthly_limit = _bounded_monthly_limit(monthly_limit)
        self._client = client or _UrllibClient()
        self._timeout_seconds = max(0.1, min(float(timeout_seconds), 30.0))
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        delays = tuple(float(value) for value in poll_delays_seconds)
        if (
            not delays
            or len(delays) > 8
            or any(not math.isfinite(value) or value < 0 or value > 10 for value in delays)
        ):
            raise ValueError("provider poll delays are invalid")
        self._poll_delays_seconds = delays
        self._sleep_provider = sleep_provider or sleep
        self._ledger = _UsageLedger(Path(usage_path))
        self._operation_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: dict[
            tuple[str, str, date],
            tuple[float, FlightOfferSearchResult],
        ] = {}

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    @property
    def environment(self) -> str:
        return "production" if self.configured else "disabled"

    @property
    def monthly_limit(self) -> int:
        return self._monthly_limit

    def search(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        *,
        fetched_at: datetime,
        force_refresh: bool = False,
    ) -> FlightOfferSearchResult:
        observed_at = _utc(fetched_at)
        route = (_iata(origin), _iata(destination))
        if not self.configured:
            return self._result("not_configured", observed_at)
        if (
            route[0] is None
            or route[1] is None
            or route[0] == route[1]
            or not isinstance(departure_date, date)
            or isinstance(departure_date, datetime)
        ):
            return self._result("no_results", observed_at)

        key = (route[0], route[1], departure_date)
        with self._operation_lock:
            cached = self._cached(key, observed_at, force_refresh=force_refresh)
            if cached is not None:
                return cached
            result = self._search_uncached(
                route[0],
                route[1],
                departure_date,
                observed_at,
                force_refresh=force_refresh,
            )
            if result.status in {"confirmed_offers", "no_results"}:
                self._remember(key, result)
            return result

    def _search_uncached(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        observed_at: datetime,
        *,
        force_refresh: bool,
    ) -> FlightOfferSearchResult:
        diagnostics = _DiagnosticCollector(self._ledger)
        try:
            account = self._account_quota(diagnostics)
        except _AuthenticationError:
            return self._diagnostic_result("authentication_failed", observed_at, diagnostics)
        except _RateLimitError:
            return self._diagnostic_result("rate_limited", observed_at, diagnostics)
        except (_ProviderHttpError, _ProviderSearchError, _TransportError):
            return self._diagnostic_result("provider_error", observed_at, diagnostics)
        except Exception:
            return self._diagnostic_result("provider_unavailable", observed_at, diagnostics)

        try:
            initial_reservation = self._ledger.synchronize_and_reserve(
                account.billing_cycle_key,
                account.hour_bucket_key,
                calls=len(_CABINS),
                monthly_limit=account.monthly_limit,
                hourly_limit=account.hourly_limit,
                provider_monthly_usage=account.monthly_used,
                provider_hourly_usage=account.hourly_used,
                require_all=True,
            )
        except _UsageError:
            return self._diagnostic_result("provider_unavailable", observed_at, diagnostics)
        if initial_reservation.reserved_calls != len(_CABINS):
            status: SearchStatus = (
                "rate_limited"
                if initial_reservation.limiting_quota == "hourly"
                else "budget_exhausted"
            )
            return self._diagnostic_result(
                status,
                observed_at,
                diagnostics,
                conservative_monthly_used=initial_reservation.monthly_used,
            )

        payloads: dict[Cabin, dict[str, Any]] = {}
        failures: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="serpapi-search") as pool:
            futures = {
                pool.submit(
                    self._search_cabin,
                    origin,
                    destination,
                    departure_date,
                    cabin,
                    force_refresh,
                    diagnostics,
                    account,
                ): cabin
                for cabin in _CABINS
            }
            for future in as_completed(futures):
                cabin = futures[future]
                try:
                    payloads[cabin] = future.result()
                except Exception as exc:  # all futures must be collected
                    failures.append(exc)

        search_calls_used = len(_CABINS) + diagnostics.controlled_retry_count("cabin_search")
        search_monthly_used = diagnostics.conservative_monthly_used(
            initial_reservation.monthly_used
        )
        search_failed_cabin_count = len(failures)
        search_retry_quota_failures = tuple(
            failure
            for failure in failures
            if isinstance(failure, _ControlledRetryQuotaError)
        )
        search_retry_quota_limited = bool(search_retry_quota_failures)
        search_quota_limit: QuotaLimit | None = None
        if search_retry_quota_limited:
            search_quota_limit = (
                "hourly"
                if any(
                    failure.limiting_quota == "hourly"
                    for failure in search_retry_quota_failures
                )
                else "monthly"
            )
        failure_status = _failure_status(failures)
        if not payloads:
            return self._diagnostic_result(
                failure_status or "provider_unavailable",
                observed_at,
                diagnostics,
                searched_cabins=_CABINS,
                search_calls_used=search_calls_used,
                conservative_monthly_used=search_monthly_used,
                search_failed_cabin_count=search_failed_cabin_count,
                coverage_status=_candidate_coverage_status(
                    evaluated=True,
                    provider_failed=0,
                    quota_skipped=0,
                    retry_quota_limited=search_retry_quota_limited,
                    search_failed_cabins=search_failed_cabin_count,
                ),
                quota_limit=search_quota_limit,
                retry_quota_limited=search_retry_quota_limited,
            )

        historical_market_contexts = tuple(
            context
            for cabin, payload in payloads.items()
            if (
                context := _parse_serpapi_market_history(
                    payload,
                    origin=origin,
                    destination=destination,
                    departure_date=departure_date,
                    cabin=cabin,
                )
            )
            is not None
        )
        candidates = self._select_candidates(payloads, origin, destination, departure_date)
        if not candidates:
            coverage_status = _candidate_coverage_status(
                evaluated=True,
                provider_failed=0,
                quota_skipped=0,
                retry_quota_limited=search_retry_quota_limited,
                search_failed_cabins=search_failed_cabin_count,
            )
            return self._diagnostic_result(
                (failure_status or "provider_unavailable") if failures else "no_results",
                observed_at,
                diagnostics,
                searched_cabins=_CABINS,
                search_calls_used=search_calls_used,
                conservative_monthly_used=search_monthly_used,
                search_failed_cabin_count=search_failed_cabin_count,
                coverage_status=coverage_status,
                quota_limit=search_quota_limit,
                retry_quota_limited=search_retry_quota_limited,
                historical_market_contexts=historical_market_contexts,
            )

        confirmed_by_position: dict[int, ConfirmedFlightOffer] = {}
        booking_failures: list[BaseException] = []
        # Ask the atomic ledger for every eligible returned candidate.  The
        # reservation may still be smaller when the provider/account billing-
        # cycle or hourly quota is genuinely exhausted, but there is no
        # application-defined per-comparison candidate ceiling.
        requested_booking_calls = len(candidates)
        try:
            booking_reservation = self._ledger.synchronize_and_reserve(
                account.billing_cycle_key,
                account.hour_bucket_key,
                calls=requested_booking_calls,
                monthly_limit=account.monthly_limit,
                hourly_limit=account.hourly_limit,
                provider_monthly_usage=account.monthly_used,
                provider_hourly_usage=account.hourly_used,
                require_all=False,
            )
        except _UsageError:
            if search_failed_cabin_count:
                return self._diagnostic_result(
                    "provider_unavailable",
                    observed_at,
                    diagnostics,
                    searched_cabins=_CABINS,
                    search_calls_used=search_calls_used,
                    conservative_monthly_used=search_monthly_used,
                    search_failed_cabin_count=search_failed_cabin_count,
                    coverage_status=_candidate_coverage_status(
                        evaluated=True,
                        provider_failed=0,
                        quota_skipped=0,
                        retry_quota_limited=search_retry_quota_limited,
                        search_failed_cabins=search_failed_cabin_count,
                    ),
                    quota_limit=search_quota_limit,
                    retry_quota_limited=search_retry_quota_limited,
                    historical_market_contexts=historical_market_contexts,
                )
            return self._diagnostic_result(
                "provider_unavailable",
                observed_at,
                diagnostics,
                searched_cabins=_CABINS,
                search_calls_used=search_calls_used,
                conservative_monthly_used=search_monthly_used,
                eligible_candidate_count=len(candidates),
                historical_market_contexts=historical_market_contexts,
            )
        reserved_candidates = list(enumerate(candidates[: booking_reservation.reserved_calls]))

        booking_calls = len(reserved_candidates)
        if reserved_candidates:
            with ThreadPoolExecutor(
                max_workers=min(MAX_BOOKING_WORKERS, booking_calls),
                thread_name_prefix="serpapi-booking",
            ) as pool:
                futures = {
                    pool.submit(
                        self._booking_options,
                        candidate,
                        origin,
                        destination,
                        departure_date,
                        force_refresh,
                        diagnostics,
                        account,
                    ): (position, candidate)
                    for position, candidate in reserved_candidates
                }
                for future in as_completed(futures):
                    position, candidate = futures[future]
                    payload: dict[str, Any] | None = None
                    provider_http_status: int | None = None
                    try:
                        payload, provider_observation, provider_http_status = future.result()
                        offer = _parse_booking_confirmation(
                            payload,
                            candidate,
                            origin,
                            destination,
                            departure_date,
                            provider_observation,
                        )
                    except Exception as exc:
                        if isinstance(exc, _PayloadError) and payload is not None:
                            diagnostics.record(
                                observed_at=self._provider_now(),
                                stage="validation",
                                http_status=provider_http_status,
                                exception_type="PayloadError",
                                search_id=_payload_search_id(payload),
                            )
                        booking_failures.append(exc)
                        continue
                    if offer is not None:
                        confirmed_by_position[position] = offer

        pricing_calls_used = booking_calls + diagnostics.controlled_retry_count(
            "booking_options"
        )
        conservative_monthly_used = diagnostics.conservative_monthly_used(
            booking_reservation.monthly_used
        )
        verified_candidate_count = len(confirmed_by_position)
        provider_failed_candidate_count = len(booking_failures)
        strictly_rejected_candidate_count = (
            booking_calls - verified_candidate_count - provider_failed_candidate_count
        )
        quota_skipped_candidate_count = len(candidates) - booking_calls
        booking_retry_quota_failures = tuple(
            failure
            for failure in booking_failures
            if isinstance(failure, _ControlledRetryQuotaError)
        )
        retry_quota_failures = search_retry_quota_failures + booking_retry_quota_failures
        retry_quota_limited = bool(retry_quota_failures)
        coverage_status = _candidate_coverage_status(
            evaluated=True,
            provider_failed=provider_failed_candidate_count,
            quota_skipped=quota_skipped_candidate_count,
            retry_quota_limited=retry_quota_limited,
            search_failed_cabins=search_failed_cabin_count,
        )
        quota_limit: QuotaLimit | None = None
        if quota_skipped_candidate_count:
            quota_limit = booking_reservation.limiting_quota
        if retry_quota_limited:
            quota_limit = (
                "hourly"
                if any(failure.limiting_quota == "hourly" for failure in retry_quota_failures)
                else "monthly"
            )

        # A single itinerary can be returned through multiple booking tokens and
        # sellers. Every quota-reserved candidate is verified first; only then
        # is each flight-and-cabin group reduced to its cheapest confirmed
        # price. Candidates skipped because actual provider/account quota was
        # unavailable remain explicitly unverified.
        best_by_group: dict[tuple[Any, ...], tuple[int, ConfirmedFlightOffer]] = {}
        for position in range(booking_calls):
            offer = confirmed_by_position.get(position)
            if offer is None:
                continue
            group_key = offer.lowest_price_group_key
            current = best_by_group.get(group_key)
            if current is None:
                best_by_group[group_key] = (position, offer)
            elif _verified_offer_preference(offer) < _verified_offer_preference(current[1]):
                # Preserve the group's first position while replacing its fare.
                best_by_group[group_key] = (current[0], offer)
        confirmed = [offer for _, offer in sorted(best_by_group.values(), key=lambda item: item[0])]
        deduplicated_verified_count = verified_candidate_count - len(confirmed)

        if confirmed:
            return self._diagnostic_result(
                "confirmed_offers",
                observed_at,
                diagnostics,
                offers=tuple(confirmed),
                searched_cabins=_CABINS,
                search_calls_used=search_calls_used,
                pricing_calls_used=pricing_calls_used,
                conservative_monthly_used=conservative_monthly_used,
                eligible_candidate_count=len(candidates),
                verification_attempted_count=booking_calls,
                verified_candidate_count=verified_candidate_count,
                strictly_rejected_candidate_count=strictly_rejected_candidate_count,
                provider_failed_candidate_count=provider_failed_candidate_count,
                search_failed_cabin_count=search_failed_cabin_count,
                quota_skipped_candidate_count=quota_skipped_candidate_count,
                deduplicated_verified_count=deduplicated_verified_count,
                coverage_status=coverage_status,
                quota_limit=quota_limit,
                retry_quota_limited=retry_quota_limited,
                historical_market_contexts=historical_market_contexts,
            )
        failure_status = _failure_status([*failures, *booking_failures])
        if failure_status is not None:
            status = failure_status
        elif booking_reservation.limiting_quota == "hourly":
            status = "rate_limited"
        elif booking_reservation.limiting_quota == "monthly":
            status = "budget_exhausted"
        else:
            status = "no_results"
        return self._diagnostic_result(
            status,
            observed_at,
            diagnostics,
            searched_cabins=_CABINS,
            search_calls_used=search_calls_used,
            pricing_calls_used=pricing_calls_used,
            conservative_monthly_used=conservative_monthly_used,
            eligible_candidate_count=len(candidates),
            verification_attempted_count=booking_calls,
            verified_candidate_count=verified_candidate_count,
            strictly_rejected_candidate_count=strictly_rejected_candidate_count,
            provider_failed_candidate_count=provider_failed_candidate_count,
            search_failed_cabin_count=search_failed_cabin_count,
            quota_skipped_candidate_count=quota_skipped_candidate_count,
            deduplicated_verified_count=deduplicated_verified_count,
            coverage_status=coverage_status,
            quota_limit=quota_limit,
            retry_quota_limited=retry_quota_limited,
            historical_market_contexts=historical_market_contexts,
        )

    def _account_quota(self, diagnostics: _DiagnosticCollector) -> _AccountQuota:
        payload, received_at, http_status = self._request_json(
            SERPAPI_ACCOUNT_URL,
            params={"api_key": self._api_key},
            require_search_success=False,
            stage="account",
            diagnostics=diagnostics,
            timeout_seconds=min(
                self._timeout_seconds,
                ACCOUNT_REQUEST_TIMEOUT_SECONDS,
            ),
        )
        account_status = str(payload.get("account_status", "")).strip().lower()
        if account_status != "active":
            diagnostics.record(
                observed_at=received_at,
                stage="account",
                http_status=http_status,
                exception_type="AuthenticationError",
                search_id=None,
            )
            raise _AuthenticationError("provider account is not active")
        monthly_usage = _optional_nonnegative_int(payload.get("this_month_usage"))
        hourly_usage = _optional_nonnegative_int(payload.get("this_hour_searches"))
        provider_monthly_limit = _optional_positive_int(payload.get("searches_per_month"))
        provider_hourly_limit = _optional_positive_int(payload.get("account_rate_limit_per_hour"))
        renewal_date = _plan_renewal_date(
            payload.get("plan_renewal_date"),
            received_at=received_at,
        )
        if (
            monthly_usage is None
            or hourly_usage is None
            or provider_monthly_limit is None
            or provider_hourly_limit is None
            or renewal_date is None
        ):
            diagnostics.record(
                observed_at=received_at,
                stage="account",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=None,
            )
            raise _PayloadError("provider account quota metadata is missing or invalid")
        return _AccountQuota(
            billing_cycle_key=f"renewal:{renewal_date.isoformat()}",
            hour_bucket_key=received_at.strftime("%Y-%m-%dT%H"),
            monthly_used=monthly_usage,
            hourly_used=hourly_usage,
            monthly_limit=min(self._monthly_limit, provider_monthly_limit),
            hourly_limit=min(SERPAPI_MAX_HOURLY_LIMIT, provider_hourly_limit),
            observed_at=received_at,
        )

    def _search_cabin(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        cabin: Cabin,
        force_refresh: bool,
        diagnostics: _DiagnosticCollector,
        account: _AccountQuota,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": departure_date.isoformat(),
            "type": 2,
            "travel_class": _SERPAPI_TRAVEL_CLASSES[cabin],
            "adults": 1,
            "currency": "USD",
            "hl": "en",
            "gl": _google_market_for_origin(origin),
            "show_hidden": "true",
            "deep_search": "true",
            "async": "true",
            "api_key": self._api_key,
        }
        if force_refresh:
            params["no_cache"] = "true"
        payload, received_at, http_status = self._request_json_with_controlled_retry(
            SERPAPI_SEARCH_URL,
            params=params,
            stage="cabin_search",
            diagnostics=diagnostics,
            account=account,
            timeout_seconds=DEEP_SEARCH_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            _provider_observation(payload, received_at)
            if not _search_parameters_match(
                payload,
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                cabin=cabin,
            ):
                raise _PayloadError("search parameters were not echoed correctly")
        except _PayloadError:
            diagnostics.record(
                observed_at=received_at,
                stage="validation",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=_payload_search_id(payload),
            )
            raise
        return payload

    def _booking_options(
        self,
        candidate: _SearchCandidate,
        origin: str,
        destination: str,
        departure_date: date,
        force_refresh: bool,
        diagnostics: _DiagnosticCollector,
        account: _AccountQuota,
    ) -> tuple[dict[str, Any], _ProviderObservation, int]:
        # SerpApi documents booking_token as the itinerary selector, but its live
        # Google Flights endpoint also validates the originating search context.
        # Replay the core one-way query so a valid token is not rejected with a
        # misleading HTTP 400 "missing departure_id" response.
        params: dict[str, Any] = {
            "engine": "google_flights",
            "booking_token": candidate.booking_token,
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": departure_date.isoformat(),
            "type": 2,
            "travel_class": _SERPAPI_TRAVEL_CLASSES[candidate.cabin],
            "adults": 1,
            "currency": "USD",
            "hl": "en",
            "gl": _google_market_for_origin(origin),
            "api_key": self._api_key,
        }
        if force_refresh:
            params["no_cache"] = "true"
        payload, received_at, http_status = self._request_json_with_controlled_retry(
            SERPAPI_SEARCH_URL,
            params=params,
            stage="booking_options",
            diagnostics=diagnostics,
            account=account,
        )
        try:
            parameters = payload.get("search_parameters")
            if (
                not isinstance(parameters, dict)
                or str(parameters.get("engine", "")).strip().lower() != "google_flights"
                or str(parameters.get("currency", "")).strip().upper() != "USD"
            ):
                raise _PayloadError("booking parameters were not echoed correctly")
            observation = _provider_observation(payload, received_at)
        except _PayloadError:
            diagnostics.record(
                observed_at=received_at,
                stage="validation",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=_payload_search_id(payload),
            )
            raise
        return payload, observation, http_status

    def _request_json_with_controlled_retry(
        self,
        url: str,
        *,
        params: dict[str, Any],
        stage: Literal["cabin_search", "booking_options"],
        diagnostics: _DiagnosticCollector,
        account: _AccountQuota,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], datetime, int]:
        """Submit once, then re-submit at most once after a transient failure.

        Each submission may independently use the bounded Search Archive polling
        path.  The retry is only started after another cross-process quota
        reservation succeeds, so an exhausted free allowance is a hard wall.
        """

        try:
            return self._request_json(
                url,
                params=params,
                stage=stage,
                diagnostics=diagnostics,
                timeout_seconds=timeout_seconds,
            )
        except (
            _ProviderProcessingError,
            _ProviderSearchError,
            _TransientProviderHttpError,
            _TransportError,
        ) as first_failure:
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
            except _UsageError:
                raise
            if reservation.reserved_calls != 1:
                diagnostics.record(
                    observed_at=self._provider_now(),
                    stage=stage,
                    http_status=None,
                    exception_type="RetryBudgetExhausted",
                    search_id=None,
                )
                raise _ControlledRetryQuotaError(reservation.limiting_quota) from first_failure
            diagnostics.note_controlled_retry(
                stage,
                monthly_used=reservation.monthly_used,
            )
            # Deliberately outside another retry-catching block: the second
            # submission may poll, but it can never trigger a third submission.
            return self._request_json(
                url,
                params=params,
                stage=stage,
                diagnostics=diagnostics,
                timeout_seconds=timeout_seconds,
            )

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        stage: DiagnosticStage,
        diagnostics: _DiagnosticCollector,
        require_search_success: bool = True,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], datetime, int]:
        try:
            response = self._client.get(
                url,
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": ("flight-forecast-lab/0.2.0 (strict booking verification)"),
                },
                timeout=(self._timeout_seconds if timeout_seconds is None else timeout_seconds),
            )
        except Exception as exc:
            received_at = self._provider_now()
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=None,
                exception_type="TransportError",
                search_id=None,
            )
            raise _TransportError("provider transport failed") from exc
        received_at = self._provider_now()
        status = _status_code(response)
        if status in {401, 403}:
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="AuthenticationError",
                search_id=_response_search_id(response),
            )
            raise _AuthenticationError("provider authentication failed")
        if status == 429:
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="RateLimitError",
                search_id=_response_search_id(response),
            )
            raise _RateLimitError("provider rate limit reached")
        if status in {408, 425} or 500 <= status <= 599:
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="TransientProviderHttpError",
                search_id=_response_search_id(response),
            )
            raise _TransientProviderHttpError("provider request failed transiently")
        if status < 200 or status >= 300:
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="ProviderHttpError",
                search_id=_response_search_id(response),
            )
            raise _ProviderHttpError("provider request failed")
        try:
            payload = _safe_response_json(response)
        except _PayloadError:
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="PayloadError",
                search_id=None,
            )
            raise
        if not isinstance(payload, dict):
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="PayloadError",
                search_id=None,
            )
            raise _PayloadError("provider payload must be an object")
        if require_search_success:
            search_status = _provider_search_status(payload)
            search_id = _payload_search_id(payload)
            if search_status in {"success", "cached"}:
                return payload, received_at, status
            if search_status in {"processing", "queued"}:
                if search_id is None:
                    diagnostics.record(
                        observed_at=received_at,
                        stage=stage,
                        http_status=status,
                        exception_type="PayloadError",
                        search_id=None,
                    )
                    raise _PayloadError("pending provider search has no valid search ID")
                diagnostics.record(
                    observed_at=received_at,
                    stage=stage,
                    http_status=status,
                    exception_type="ProviderPending",
                    search_id=search_id,
                )
                return self._poll_search_archive(search_id, diagnostics)
            if search_status == "error":
                diagnostics.record(
                    observed_at=received_at,
                    stage=stage,
                    http_status=status,
                    exception_type="ProviderSearchError",
                    search_id=search_id,
                )
                raise _ProviderSearchError("provider search failed")
            diagnostics.record(
                observed_at=received_at,
                stage=stage,
                http_status=status,
                exception_type="PayloadError",
                search_id=search_id,
            )
            raise _PayloadError("provider search status is missing or invalid")
        return payload, received_at, status

    def _poll_search_archive(
        self,
        search_id: str,
        diagnostics: _DiagnosticCollector,
    ) -> tuple[dict[str, Any], datetime, int]:
        archive_url = SERPAPI_SEARCH_ARCHIVE_URL.format(search_id=search_id)
        last_received_at = self._provider_now()
        last_http_status: int | None = None
        for delay_seconds in self._poll_delays_seconds:
            self._sleep_provider(delay_seconds)
            diagnostics.note_archive_poll()
            payload, received_at, http_status = self._request_json(
                archive_url,
                params={"api_key": self._api_key},
                stage="search_archive",
                diagnostics=diagnostics,
                require_search_success=False,
                timeout_seconds=min(
                    self._timeout_seconds,
                    SEARCH_ARCHIVE_REQUEST_TIMEOUT_SECONDS,
                ),
            )
            last_received_at = received_at
            last_http_status = http_status
            archived_id = _payload_search_id(payload)
            search_status = _provider_search_status(payload)
            if archived_id != search_id:
                diagnostics.record(
                    observed_at=received_at,
                    stage="search_archive",
                    http_status=http_status,
                    exception_type="SearchIdMismatch",
                    search_id=archived_id,
                )
                raise _PayloadError("provider archive search ID did not match")
            if search_status in {"success", "cached"}:
                return payload, received_at, http_status
            if search_status == "error":
                diagnostics.record(
                    observed_at=received_at,
                    stage="search_archive",
                    http_status=http_status,
                    exception_type="ProviderSearchError",
                    search_id=search_id,
                )
                raise _ProviderSearchError("provider archive search failed")
            if search_status not in {"processing", "queued"}:
                diagnostics.record(
                    observed_at=received_at,
                    stage="search_archive",
                    http_status=http_status,
                    exception_type="PayloadError",
                    search_id=search_id,
                )
                raise _PayloadError("provider archive status is invalid")
            diagnostics.record(
                observed_at=received_at,
                stage="search_archive",
                http_status=http_status,
                exception_type="ProviderPending",
                search_id=search_id,
            )
        diagnostics.record(
            observed_at=last_received_at,
            stage="search_archive",
            http_status=last_http_status,
            exception_type="ProviderProcessingError",
            search_id=search_id,
        )
        raise _ProviderProcessingError("provider search is still processing")

    def _provider_now(self) -> datetime:
        try:
            return _utc(self._now_provider())
        except (TypeError, ValueError, OverflowError) as exc:
            raise _PayloadError("provider response clock is invalid") from exc

    @staticmethod
    def _select_candidates(
        payloads: dict[Cabin, dict[str, Any]],
        origin: str,
        destination: str,
        departure_date: date,
    ) -> tuple[_SearchCandidate, ...]:
        by_cabin: dict[Cabin, list[_SearchCandidate]] = {cabin: [] for cabin in _CABINS}
        for cabin in _CABINS:
            payload = payloads.get(cabin, {})
            google_flights_url = _metadata_google_flights_url(payload)
            rows: list[Any] = []
            for key in ("best_flights", "other_flights"):
                section = payload.get(key)
                if isinstance(section, list):
                    rows.extend(section)
            by_token: dict[str, _SearchCandidate] = {}
            conflicting_tokens: set[str] = set()
            for row in rows:
                candidate = _parse_search_candidate(
                    row,
                    cabin,
                    origin,
                    destination,
                    departure_date,
                    google_flights_url,
                )
                if candidate is None or candidate.booking_token in conflicting_tokens:
                    continue
                existing = by_token.get(candidate.booking_token)
                if existing is None:
                    by_token[candidate.booking_token] = candidate
                elif existing.identity != candidate.identity:
                    # A token is only safe when every occurrence identifies the
                    # same dated itinerary in the same requested cabin.
                    by_token.pop(candidate.booking_token, None)
                    conflicting_tokens.add(candidate.booking_token)
                elif candidate.search_price_usd < existing.search_price_usd:
                    by_token[candidate.booking_token] = candidate
            by_cabin[cabin].extend(by_token.values())
        return _direct_first_cabin_round_robin(by_cabin)

    def _cached(
        self,
        key: tuple[str, str, date],
        observed_at: datetime,
        *,
        force_refresh: bool,
    ) -> FlightOfferSearchResult | None:
        if force_refresh:
            return None
        with self._cache_lock:
            entry = self._cache.get(key)
        if entry is None:
            return None
        stored_tick, result = entry
        wall_age = (observed_at - result.observed_at).total_seconds()
        if (
            monotonic() - stored_tick > FLIGHT_OFFER_CACHE_TTL_SECONDS
            or wall_age < 0
            or wall_age > FLIGHT_OFFER_CACHE_TTL_SECONDS
        ):
            return None
        refreshed_offers: list[ConfirmedFlightOffer] = []
        for offer in result.offers:
            cache_age = _provider_cache_age_seconds(offer.verified_at, observed_at)
            if cache_age is None:
                return None
            refreshed_offers.append(
                replace(
                    offer,
                    provider_cache_hit=True,
                    provider_cache_age_seconds=cache_age,
                )
            )
        return replace(
            result,
            offers=tuple(refreshed_offers),
            cache_hit=True,
            calls_used=0,
            search_calls_used=0,
            pricing_calls_used=0,
        )

    def _remember(
        self,
        key: tuple[str, str, date],
        result: FlightOfferSearchResult,
    ) -> None:
        with self._cache_lock:
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                oldest = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest, None)
            self._cache[key] = (monotonic(), result)

    def _result(
        self,
        status: SearchStatus,
        observed_at: datetime,
        *,
        offers: tuple[ConfirmedFlightOffer, ...] = (),
        searched_cabins: tuple[Cabin, ...] = (),
        search_calls_used: int = 0,
        pricing_calls_used: int = 0,
        conservative_monthly_used: int | None = None,
        archive_poll_count: int = 0,
        diagnostics: tuple[ProviderDiagnostic, ...] = (),
        coverage_scope: Literal["provider_returned_booking_verification_candidates"] = (
            "provider_returned_booking_verification_candidates"
        ),
        eligible_candidate_count: int = 0,
        verification_attempted_count: int = 0,
        verified_candidate_count: int = 0,
        strictly_rejected_candidate_count: int = 0,
        provider_failed_candidate_count: int = 0,
        search_failed_cabin_count: int = 0,
        quota_skipped_candidate_count: int = 0,
        deduplicated_verified_count: int = 0,
        coverage_status: CandidateCoverageStatus = "not_evaluated",
        quota_limit: QuotaLimit | None = None,
        retry_quota_limited: bool = False,
        historical_market_contexts: tuple[RouteCabinMarketHistory, ...] = (),
    ) -> FlightOfferSearchResult:
        return FlightOfferSearchResult(
            offers=offers,
            status=status,
            observed_at=observed_at,
            environment=("disabled" if status == "not_configured" else self.environment),
            searched_cabins=searched_cabins,
            calls_used=search_calls_used + pricing_calls_used,
            cache_hit=False,
            search_calls_used=search_calls_used,
            pricing_calls_used=pricing_calls_used,
            # SerpApi has one shared monthly quota. This is a conservative local
            # attempt-reservation snapshot, not an exact provider billing count.
            search_monthly_limit=self._monthly_limit,
            pricing_monthly_limit=None,
            search_monthly_used=conservative_monthly_used,
            pricing_monthly_used=None,
            archive_poll_count=archive_poll_count,
            diagnostics=diagnostics,
            coverage_scope=coverage_scope,
            eligible_candidate_count=eligible_candidate_count,
            verification_attempted_count=verification_attempted_count,
            verified_candidate_count=verified_candidate_count,
            strictly_rejected_candidate_count=strictly_rejected_candidate_count,
            provider_failed_candidate_count=provider_failed_candidate_count,
            search_failed_cabin_count=search_failed_cabin_count,
            quota_skipped_candidate_count=quota_skipped_candidate_count,
            deduplicated_verified_count=deduplicated_verified_count,
            coverage_status=coverage_status,
            quota_limit=quota_limit,
            retry_quota_limited=retry_quota_limited,
            historical_market_contexts=historical_market_contexts,
        )

    def _diagnostic_result(
        self,
        status: SearchStatus,
        observed_at: datetime,
        diagnostics: _DiagnosticCollector,
        **kwargs: Any,
    ) -> FlightOfferSearchResult:
        return self._result(
            status,
            observed_at,
            archive_poll_count=diagnostics.archive_poll_count,
            diagnostics=diagnostics.snapshot(),
            **kwargs,
        )


def flight_offer_provider_from_env(
    usage_path: Path,
) -> Any:
    provider = os.getenv("FLIGHT_OFFER_PROVIDER", "none").strip().lower()
    if provider == "serpapi":
        return SerpApiFlightOfferProvider(
            os.getenv("SERPAPI_API_KEY"),
            usage_path=usage_path,
            monthly_limit=_environment_monthly_limit(),
        )
    if provider in {
        "searchapi",
        "scrappa",
        "ignav",
        "ignav_quarantine",
        "ignav_verified_fares",
        "auto",
        "serpapi_searchapi",
    }:
        # Imported lazily to keep the mature SerpApi adapter independent from
        # optional fallback providers and to avoid an import cycle.
        from flight_forecaster.alternate_fare_providers import (
            FallbackFlightOfferProvider,
            IgnavQuarantineFlightOfferProvider,
            ScrappaFlightOfferProvider,
            SearchApiFlightOfferProvider,
        )

        alternate_usage_path = Path(usage_path).with_name("alternate-provider-usage.sqlite3")
        searchapi = SearchApiFlightOfferProvider(
            os.getenv("SEARCHAPI_API_KEY"),
            usage_path=alternate_usage_path,
            monthly_limit=(
                os.getenv("SEARCHAPI_LIFETIME_LIMIT")
                or os.getenv("SEARCHAPI_MONTHLY_LIMIT", "100")
            ),
        )
        if provider == "searchapi":
            return searchapi
        scrappa = ScrappaFlightOfferProvider(
            os.getenv("SCRAPPA_API_KEY"),
            usage_path=alternate_usage_path,
            monthly_limit=os.getenv("SCRAPPA_MONTHLY_LIMIT", "500"),
        )
        if provider == "scrappa":
            return scrappa
        ignav = IgnavQuarantineFlightOfferProvider(
            os.getenv("IGNAV_API_KEY") or os.getenv("IGNAV_TOKEN"),
            usage_path=alternate_usage_path,
            release_verified=(
                provider != "ignav_quarantine"
                and os.getenv("IGNAV_STRICT_RELEASE", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            free_account_attested=(
                os.getenv("IGNAV_FREE_ACCOUNT_ATTESTED", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            lifetime_limit=os.getenv("IGNAV_LIFETIME_LIMIT", "1000"),
        )
        if provider in {"ignav", "ignav_quarantine", "ignav_verified_fares"}:
            return ignav
        serpapi = SerpApiFlightOfferProvider(
            os.getenv("SERPAPI_API_KEY"),
            usage_path=usage_path,
            monthly_limit=_environment_monthly_limit(),
        )
        providers = (serpapi, searchapi, scrappa)
        if provider == "auto":
            providers += (ignav,)
        return FallbackFlightOfferProvider(providers)
    return NullFlightOfferProvider()


def _parse_serpapi_market_history(
    payload: dict[str, Any],
    *,
    origin: str,
    destination: str,
    departure_date: date,
    cabin: Cabin,
) -> RouteCabinMarketHistory | None:
    """Fail closed on malformed optional history without rejecting strict fares.

    ``price_insights.price_history`` belongs to the route/date/cabin search, not
    to any selected itinerary or booking seller.  The provider payload has
    already passed search-parameter and freshness validation before this helper
    is called.  Optional malformed history is discarded as a whole so it cannot
    weaken or contaminate the separately verified strict-offer path.
    """

    insights = payload.get("price_insights")
    if insights is None:
        return None
    if not isinstance(insights, dict):
        return None
    raw_history = insights.get("price_history")
    if raw_history is None:
        return None
    if (
        not isinstance(raw_history, list)
        or not 1 <= len(raw_history) <= MAX_SERPAPI_PRICE_HISTORY_POINTS
    ):
        return None
    metadata = payload.get("search_metadata")
    provider_observed_at = (
        _provider_created_at(metadata.get("created_at"))
        if isinstance(metadata, dict)
        else None
    )
    if provider_observed_at is None:
        return None
    latest_allowed = int(
        provider_observed_at.timestamp() + MAX_PROVIDER_FUTURE_SKEW_SECONDS
    )
    by_timestamp: dict[int, float] = {}
    for raw_point in raw_history:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            return None
        raw_timestamp, raw_price = raw_point
        if (
            isinstance(raw_timestamp, bool)
            or not isinstance(raw_timestamp, int)
            or raw_timestamp < MIN_SERPAPI_PRICE_HISTORY_TIMESTAMP
            or raw_timestamp > latest_allowed
            or isinstance(raw_price, bool)
            or not isinstance(raw_price, (int, float))
        ):
            return None
        price = _finite_amount(raw_price)
        if price is None or price <= 0:
            return None
        existing = by_timestamp.get(raw_timestamp)
        if existing is not None and existing != price:
            return None
        by_timestamp[raw_timestamp] = price
    try:
        points = tuple(
            RouteCabinMarketPricePoint(
                observed_at=datetime.fromtimestamp(timestamp, tz=UTC),
                price_usd=price,
            )
            for timestamp, price in sorted(by_timestamp.items())
        )
        return RouteCabinMarketHistory(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            cabin=cabin,
            provider_observed_at=provider_observed_at,
            points=points,
        )
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _parse_search_candidate(
    row: Any,
    cabin: Cabin,
    origin: str,
    destination: str,
    departure_date: date,
    google_flights_url: str | None = None,
) -> _SearchCandidate | None:
    if not isinstance(row, dict) or _normalized_phrase(row.get("type")) != "one way":
        return None
    token = _opaque_token(row.get("booking_token"))
    price = _finite_amount(row.get("price"))
    raw_segments = row.get("flights")
    if token is None or price is None or price <= 0 or not isinstance(raw_segments, list):
        return None
    segments = _parse_google_segments(raw_segments, cabin)
    if not _segments_match_request(segments, origin, destination, departure_date):
        return None
    return _SearchCandidate(
        booking_token=token,
        google_flights_url=google_flights_url,
        cabin=cabin,
        search_price_usd=price,
        segments=segments,
        airline_name=_short_text(raw_segments[0].get("airline"), max_length=160)
        or segments[0].marketing_airline_code,
    )


def _parse_booking_confirmation(
    payload: dict[str, Any],
    candidate: _SearchCandidate,
    origin: str,
    destination: str,
    departure_date: date,
    provider_observation: _ProviderObservation,
) -> ConfirmedFlightOffer | None:
    selected = payload.get("selected_flights")
    if not isinstance(selected, list) or len(selected) != 1:
        return None
    itinerary = selected[0]
    if (
        not isinstance(itinerary, dict)
        or _normalized_phrase(itinerary.get("type")) != "one way"
        or not isinstance(itinerary.get("flights"), list)
    ):
        return None
    segments = _parse_google_segments(itinerary["flights"], candidate.cabin)
    if (
        not _segments_match_request(segments, origin, destination, departure_date)
        or _segment_identity(segments) != candidate.identity
    ):
        return None
    evidence = _booking_evidence(
        payload,
        segments,
        fallback_google_flights_url=candidate.google_flights_url,
    )
    if evidence is None:
        return None

    if evidence.fare_brand is not None:
        segments = tuple(replace(segment, fare_brand=evidence.fare_brand) for segment in segments)
    if evidence.checked_bags_quantity is not None:
        segments = tuple(
            replace(
                segment,
                checked_bags_quantity=evidence.checked_bags_quantity,
            )
            for segment in segments
        )
    action_identity = {
        "token": candidate.booking_token,
        "seller": evidence.booking_provider,
        "url": evidence.booking_url,
        "price": evidence.amount_usd,
        "segments": _segment_identity(segments),
    }
    digest = hashlib.sha256(
        json.dumps(action_identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return ConfirmedFlightOffer(
        provider_offer_id=f"serpapi-{digest}",
        validating_airline_code=segments[0].marketing_airline_code,
        airline_name=(
            _short_text(itinerary["flights"][0].get("airline"), max_length=160)
            or candidate.airline_name
        ),
        cabin=candidate.cabin,
        total_amount_usd=evidence.amount_usd,
        base_amount_usd=None,
        last_ticketing_date=None,
        number_of_bookable_seats=None,
        seat_count_capped=False,
        verified_at=provider_observation.created_at,
        provider_cache_hit=provider_observation.cache_hit,
        provider_cache_age_seconds=provider_observation.cache_age_seconds,
        segments=segments,
        refundable_fare=evidence.refundable,
        no_penalty_fare=evidence.no_penalty,
        no_restriction_fare=evidence.no_restriction,
        booking_url=evidence.booking_url,
        booking_url_kind=evidence.booking_url_kind,
        booking_provider=evidence.booking_provider,
        booking_verified=True,
    )


def _booking_evidence(
    payload: dict[str, Any],
    segments: tuple[FlightOfferSegment, ...],
    *,
    fallback_google_flights_url: str | None = None,
) -> _BookingEvidence | None:
    options = payload.get("booking_options")
    if not isinstance(options, list):
        return None
    expected_numbers = [_compact_flight_number(segment) for segment in segments]
    booking_response_flights_url = _metadata_google_flights_url(payload)
    evidence: list[_BookingEvidence] = []
    for option in options:
        if not isinstance(option, dict) or option.get("separate_tickets") is True:
            continue
        together = option.get("together")
        if not isinstance(together, dict):
            continue
        provider = _short_text(together.get("book_with"), max_length=160)
        amount = _finite_amount(together.get("price"))
        marketed = together.get("marketed_as")
        request_data = together.get("booking_request")
        if (
            provider is None
            or amount is None
            or amount <= 0
            or not isinstance(marketed, list)
            or not isinstance(request_data, dict)
        ):
            continue
        normalized_marketed = [_normalized_full_flight_number(value) for value in marketed]
        if any(value is None for value in normalized_marketed):
            continue
        if normalized_marketed != expected_numbers:
            continue
        action_url = _safe_google_url(request_data.get("url"), path_prefix="/travel/clk/")
        if action_url is None:
            continue
        post_data = request_data.get("post_data")
        if post_data is None:
            booking_url = action_url
            booking_url_kind: BookingUrlKind = "direct_get"
        elif isinstance(post_data, str) and post_data.strip():
            # The action itself requires POST and cannot safely be exposed as an
            # anchor. Fall back to a provider-returned Google Flights results page;
            # this is navigation evidence, not a claim that an itinerary is selected.
            booking_url = booking_response_flights_url or fallback_google_flights_url
            booking_url_kind = "google_flights_itinerary"
        else:
            continue
        if booking_url is None:
            continue
        extensions = together.get("extensions")
        phrases = (
            [_normalized_phrase(item) for item in extensions]
            if isinstance(extensions, list)
            else []
        )
        refundable = _phrase_flag(
            phrases,
            positive=("refundable", "refunds allowed"),
            negative=("no refunds", "nonrefundable", "non-refundable"),
        )
        no_penalty = _phrase_flag(
            phrases,
            positive=("free changes", "changes permitted without fee"),
            negative=("changes for a fee", "ticket changes for a fee", "no changes"),
        )
        no_restriction = _phrase_flag(
            phrases,
            positive=("no restrictions",),
            negative=("restrictions apply",),
        )
        evidence.append(
            _BookingEvidence(
                amount_usd=amount,
                booking_url=booking_url,
                booking_url_kind=booking_url_kind,
                booking_provider=provider,
                fare_brand=_short_text(together.get("option_title"), max_length=80),
                refundable=refundable,
                no_penalty=no_penalty,
                no_restriction=no_restriction,
                checked_bags_quantity=_checked_bags_quantity(together.get("baggage_prices")),
            )
        )
    if not evidence:
        return None
    return min(evidence, key=lambda item: (item.amount_usd, item.booking_provider))


def _parse_google_segments(
    rows: list[Any],
    searched_cabin: Cabin,
) -> tuple[FlightOfferSegment, ...]:
    if not 1 <= len(rows) <= MAX_STRICT_ITINERARY_SEGMENTS:
        return ()
    segments: list[FlightOfferSegment] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            return ()
        departure = row.get("departure_airport")
        arrival = row.get("arrival_airport")
        if not isinstance(departure, dict) or not isinstance(arrival, dict):
            return ()
        origin = _iata(departure.get("id"))
        destination = _iata(arrival.get("id"))
        departure_at = _google_local_datetime(departure.get("time"))
        arrival_at = _google_local_datetime(arrival.get("time"))
        full_number = _split_full_flight_number(row.get("flight_number"))
        cabin = _google_cabin(row.get("travel_class"))
        duration = _optional_positive_int(row.get("duration"))
        if (
            origin is None
            or destination is None
            or origin == destination
            or departure_at is None
            or arrival_at is None
            or full_number is None
            or cabin != searched_cabin
            or duration is None
        ):
            return ()
        if segments:
            previous = segments[-1]
            if previous.destination != origin or departure_at <= previous.arrival_at:
                return ()
        marketing_code, number = full_number
        segment_key = f"{marketing_code}{number}-{departure_at:%Y%m%d%H%M}-{index}"
        segments.append(
            FlightOfferSegment(
                segment_id=segment_key,
                origin=origin,
                destination=destination,
                departure_at=departure_at,
                arrival_at=arrival_at,
                marketing_airline_code=marketing_code,
                operating_airline_code=None,
                flight_number=number,
                departure_terminal=_short_text(departure.get("terminal"), max_length=40),
                arrival_terminal=_short_text(arrival.get("terminal"), max_length=40),
                # Google returns a human-readable airplane model, not an ICAO code.
                aircraft_icao=None,
                cabin=searched_cabin,
                booking_class=None,
                fare_basis=None,
                fare_brand=None,
                checked_bags_quantity=None,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            )
        )
    return tuple(segments)


def _segments_match_request(
    segments: tuple[FlightOfferSegment, ...],
    origin: str,
    destination: str,
    departure_date: date,
) -> bool:
    return bool(
        segments
        and segments[0].origin == origin
        and segments[-1].destination == destination
        and segments[0].departure_at.date() == departure_date
    )


def _segment_identity(segments: tuple[FlightOfferSegment, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            segment.origin,
            segment.destination,
            segment.departure_at,
            segment.arrival_at,
            segment.marketing_airline_code,
            segment.flight_number,
            segment.cabin,
        )
        for segment in segments
    )


def _search_parameters_match(
    payload: dict[str, Any],
    *,
    origin: str,
    destination: str,
    departure_date: date,
    cabin: Cabin,
) -> bool:
    parameters = payload.get("search_parameters")
    if not isinstance(parameters, dict):
        return False
    travel_class = _optional_positive_int(parameters.get("travel_class"))
    flight_type = _optional_positive_int(parameters.get("type"))
    return bool(
        str(parameters.get("engine", "")).strip().lower() == "google_flights"
        and _iata(parameters.get("departure_id")) == origin
        and _iata(parameters.get("arrival_id")) == destination
        and str(parameters.get("outbound_date", "")) == departure_date.isoformat()
        and str(parameters.get("currency", "")).strip().upper() == "USD"
        and travel_class == _SERPAPI_TRAVEL_CLASSES[cabin]
        and flight_type == 2
    )


def _metadata_google_flights_url(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("search_metadata")
    value = metadata.get("google_flights_url") if isinstance(metadata, dict) else None
    return _safe_google_url(value, path_prefix="/travel/flights")


def _provider_observation(payload: dict[str, Any], received_at: datetime) -> _ProviderObservation:
    metadata = payload.get("search_metadata")
    status = str(metadata.get("status", "")).strip().lower() if isinstance(metadata, dict) else ""
    if status not in {"success", "cached"}:
        raise _PayloadError("provider response status is invalid")
    created_at = (
        _provider_created_at(metadata.get("created_at")) if isinstance(metadata, dict) else None
    )
    if created_at is None:
        raise _PayloadError("provider creation time is missing or invalid")
    cache_age = _provider_cache_age_seconds(created_at, received_at)
    if cache_age is None:
        raise _PayloadError("provider response time is outside the freshness window")
    return _ProviderObservation(
        created_at=created_at,
        cache_hit=(status == "cached" or cache_age > PROVIDER_CACHE_HIT_AGE_SECONDS),
        cache_age_seconds=cache_age,
    )


def _provider_created_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not 20 <= len(value.strip()) <= 40:
        return None
    text = value.strip()
    try:
        if text.upper().endswith(" UTC"):
            parsed = datetime.fromisoformat(text[:-4].strip())
            if parsed.tzinfo is not None:
                return None
            return parsed.replace(tzinfo=UTC)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _plan_renewal_date(value: Any, *, received_at: datetime) -> date | None:
    if not isinstance(value, str) or len(value.strip()) != 10:
        return None
    text = value.strip()
    try:
        renewal_date = date.fromisoformat(text)
    except ValueError:
        return None
    received_date = _utc(received_at).date()
    if (
        renewal_date.isoformat() != text
        or renewal_date < received_date
        or (renewal_date - received_date).days > 62
    ):
        return None
    return renewal_date


def _provider_cache_age_seconds(
    created_at: datetime,
    observed_at: datetime,
) -> int | None:
    age = (_utc(observed_at) - _utc(created_at)).total_seconds()
    if age < -MAX_PROVIDER_FUTURE_SKEW_SECONDS or age > MAX_PROVIDER_CACHE_AGE_SECONDS:
        return None
    return max(0, int(age))


def _checked_bags_quantity(value: Any) -> int | None:
    """Return included checked bags, with zero reserved for an explicit paid first bag."""

    if not isinstance(value, list):
        return None
    free_counts: set[int] = set()
    first_bag_paid = False
    contradictory_paid_policy = False
    for item in value:
        if not isinstance(item, str):
            continue
        phrase = _normalized_phrase(item)
        if not phrase:
            continue
        for pattern in (
            r"\b(\d{1,2})\s+free\s+checked\s+bags?\b",
            r"\b(\d{1,2})\s+checked\s+bags?\s+(?:free|included)\b",
        ):
            match = re.search(pattern, phrase)
            if match is not None:
                count = int(match.group(1))
                if 1 <= count <= 9:
                    free_counts.add(count)
        first_price = re.search(
            r"\b(?:1st|first)\s+checked\s+bag\s*:\s*"
            r"(?:[a-z]{2,3}\s*)?\$?(\d+(?:\.\d{1,2})?)",
            phrase,
        )
        if first_price is not None:
            if float(first_price.group(1)) == 0:
                free_counts.add(1)
            else:
                first_bag_paid = True
        if re.search(
            r"\b(?:1st|first)\s+checked\s+bag\s*:\s*(?:free|included)\b",
            phrase,
        ):
            free_counts.add(1)
        if any(
            marker in phrase
            for marker in (
                "checked baggage for a fee",
                "checked bag for a fee",
                "checked baggage fees apply",
                "no free checked bag",
                "no checked bags",
            )
        ):
            contradictory_paid_policy = True

    if len(free_counts) > 1:
        return None
    if free_counts:
        if first_bag_paid or contradictory_paid_policy:
            return None
        return next(iter(free_counts))
    if first_bag_paid or contradictory_paid_policy:
        return 0
    return None


def _compact_flight_number(segment: FlightOfferSegment) -> str:
    return f"{segment.marketing_airline_code}{segment.flight_number}".upper()


def _normalized_full_flight_number(value: Any) -> str | None:
    parsed = _split_full_flight_number(value)
    return f"{parsed[0]}{parsed[1]}" if parsed is not None else None


def _split_full_flight_number(value: Any) -> tuple[str, str] | None:
    text = str(value or "").strip().upper()
    match = _FULL_FLIGHT_NUMBER_PATTERN.fullmatch(text)
    if match is None:
        return None
    code = _airline_code(match.group(1))
    number = _flight_number(match.group(2))
    return (code, number) if code is not None and number is not None else None


def _google_cabin(value: Any) -> Cabin | None:
    return _GOOGLE_TRAVEL_CLASSES.get(_normalized_phrase(value))


def _google_local_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not 16 <= len(value) <= 19:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None or parsed.second != 0 or parsed.microsecond != 0:
        return None
    return parsed


def _opaque_token(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > 16_000 or not _SAFE_SHORT_PATTERN.fullmatch(token):
        return None
    return token


def _opaque_token_digest(token: str) -> str:
    """Return a deterministic sort key without retaining or exposing a token."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _safe_https_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 2_048 or not _SAFE_SHORT_PATTERN.fullmatch(candidate):
        return None
    try:
        parts = parse.urlsplit(candidate)
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    if (
        parts.scheme.lower() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
    ):
        return None
    return candidate


def _safe_google_url(value: Any, *, path_prefix: str) -> str | None:
    candidate = _safe_https_url(value)
    if candidate is None:
        return None
    parts = parse.urlsplit(candidate)
    if parts.hostname not in {"google.com", "www.google.com"}:
        return None
    return candidate if parts.path.startswith(path_prefix) else None


def _phrase_flag(
    phrases: list[str],
    *,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
) -> bool | None:
    if any(marker in phrase for phrase in phrases for marker in negative):
        return False
    if any(marker in phrase for phrase in phrases for marker in positive):
        return True
    return None


def _normalized_phrase(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _safe_search_id(value: Any) -> str | None:
    candidate = str(value or "").strip()
    return candidate if _SEARCH_ID_PATTERN.fullmatch(candidate) else None


def _safe_exception_type(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if _EXCEPTION_TYPE_PATTERN.fullmatch(candidate) else "UnknownError"


def _payload_search_id(payload: Any) -> str | None:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    return _safe_search_id(metadata.get("id")) if isinstance(metadata, dict) else None


def _provider_search_status(payload: Any) -> str:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    return str(metadata.get("status", "")).strip().lower() if isinstance(metadata, dict) else ""


def _response_search_id(response: Any) -> str | None:
    try:
        return _payload_search_id(_safe_response_json(response))
    except _PayloadError:
        return None


def _safe_response_json(response: Any) -> Any:
    content = getattr(response, "content", None)
    if isinstance(content, bytes) and len(content) > MAX_PROVIDER_RESPONSE_BYTES:
        raise _PayloadError("provider response exceeded safety limit")
    text = getattr(response, "text", None)
    if isinstance(text, str) and len(text.encode("utf-8")) > MAX_PROVIDER_RESPONSE_BYTES:
        raise _PayloadError("provider response exceeded safety limit")
    try:
        payload = response.json()
    except Exception as exc:
        raise _PayloadError("provider response is not JSON") from exc
    try:
        encoded_size = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _PayloadError("provider response is not serializable") from exc
    if encoded_size > MAX_PROVIDER_RESPONSE_BYTES:
        raise _PayloadError("provider response exceeded safety limit")
    return payload


def _failure_status(failures: list[BaseException]) -> SearchStatus | None:
    if any(isinstance(exc, _RateLimitError) for exc in failures):
        return "rate_limited"
    retry_quota_failures = [
        exc for exc in failures if isinstance(exc, _ControlledRetryQuotaError)
    ]
    if retry_quota_failures:
        if any(exc.limiting_quota == "hourly" for exc in retry_quota_failures):
            return "rate_limited"
        return "budget_exhausted"
    if any(isinstance(exc, _AuthenticationError) for exc in failures):
        return "authentication_failed"
    if any(
        isinstance(
            exc,
            (_ProviderHttpError, _ProviderSearchError, _TransportError),
        )
        for exc in failures
    ):
        return "provider_error"
    if any(isinstance(exc, _ProviderProcessingError) for exc in failures):
        return "provider_processing"
    if failures:
        return "provider_unavailable"
    return None


def _direct_first_cabin_round_robin(
    by_cabin: dict[Cabin, list[Any]],
) -> tuple[Any, ...]:
    """Order every candidate without imposing a result-count ceiling.

    A provider/account quota can still admit only a prefix of this sequence.
    Put all nonstop itineraries before connections so an expensive nonstop is
    not hidden behind cheaper connecting trips, and round-robin cabins inside
    each routing tier so one cabin cannot consume the whole remaining quota.
    """

    ordered: list[Any] = []
    for direct_only in (True, False):
        buckets: dict[Cabin, list[Any]] = {}
        for cabin in _CABINS:
            candidates = [
                candidate
                for candidate in by_cabin.get(cabin, [])
                if (len(candidate.segments) == 1) is direct_only
            ]
            candidates.sort(
                key=lambda item: (
                    len(item.segments),
                    item.search_price_usd,
                    item.identity,
                    _opaque_token_digest(
                        str(
                            getattr(
                                item,
                                "booking_token",
                                getattr(item, "ignav_id", ""),
                            )
                        )
                    ),
                )
            )
            buckets[cabin] = candidates
        round_count = max((len(items) for items in buckets.values()), default=0)
        for position in range(round_count):
            for cabin in _CABINS:
                candidates = buckets[cabin]
                if position < len(candidates):
                    ordered.append(candidates[position])
    return tuple(ordered)


def _candidate_coverage_status(
    *,
    evaluated: bool,
    provider_failed: int,
    quota_skipped: int,
    retry_quota_limited: bool = False,
    search_failed_cabins: int = 0,
) -> CandidateCoverageStatus:
    if not evaluated:
        return "not_evaluated"
    quota_limited = bool(quota_skipped or retry_quota_limited)
    provider_incomplete = bool(provider_failed or search_failed_cabins)
    if provider_incomplete and quota_limited:
        return "quota_and_provider_incomplete"
    if provider_incomplete:
        return "provider_incomplete"
    if quota_limited:
        return "quota_limited"
    return "complete"


def _verified_offer_preference(offer: ConfirmedFlightOffer) -> tuple[Any, ...]:
    """Choose the cheapest verified seller, preferring a direct GET on ties."""

    return (
        offer.total_amount_usd,
        offer.booking_url_kind != "direct_get",
        offer.booking_provider.casefold(),
        offer.provider_offer_id,
    )


def _status_code(response: Any) -> int:
    value = getattr(response, "status_code", getattr(response, "status", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _finite_amount(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount):
        return None
    return round(amount, 2)


def _iata(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if _IATA_PATTERN.fullmatch(text) else None


def _airline_code(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if _AIRLINE_PATTERN.fullmatch(text) else None


def _flight_number(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if _FLIGHT_NUMBER_PATTERN.fullmatch(text) else None


def _short_text(value: Any, *, max_length: int) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text or len(text) > max_length or not _SAFE_SHORT_PATTERN.fullmatch(text):
        return None
    return text


def _nonnegative_int(value: Any) -> int:
    parsed = _optional_nonnegative_int(value)
    return parsed if parsed is not None else 0


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and str(value).strip() in {str(parsed), f"{parsed}.0"} else None


def _optional_positive_int(value: Any) -> int | None:
    parsed = _optional_nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _bounded_monthly_limit(value: Any) -> int:
    parsed = _optional_positive_int(value)
    if parsed is None:
        return SERPAPI_DEFAULT_MONTHLY_LIMIT
    return min(parsed, SERPAPI_MAX_MONTHLY_LIMIT)


def _environment_monthly_limit() -> int:
    return _bounded_monthly_limit(os.getenv("SERPAPI_MONTHLY_LIMIT"))


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("fetched_at must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
