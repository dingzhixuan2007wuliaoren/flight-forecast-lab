"""Fail-closed free-quota adapters for alternate live-fare providers.

SearchAPI.io is a strict Google Flights fallback.  Ignav is deliberately kept
in quarantine unless an operator explicitly confirms both strict release and
that the dedicated account has no billing method.  Every potentially billable
request is reserved in a durable, atomic lifetime ledger before it is sent.

The adapters never turn search hints into offers.  A candidate must survive a
second provider request that returns the same dated segments, cabin, a positive
USD price, and a user-openable HTTPS purchase path.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from urllib import parse, request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flight_forecaster.availability import (
    AGGREGATE_PROVIDER_CODE,
    AGGREGATE_PROVIDER_NAME,
    FLIGHT_OFFER_CACHE_TTL_SECONDS,
    IGNAV_QUARANTINE_PROVIDER_CODE,
    IGNAV_QUARANTINE_PROVIDER_NAME,
    IGNAV_VERIFIED_PROVIDER_CODE,
    IGNAV_VERIFIED_PROVIDER_NAME,
    MAX_CACHE_ENTRIES,
    MAX_PROVIDER_RESPONSE_BYTES,
    SCRAPPA_PROVIDER_CODE,
    SCRAPPA_PROVIDER_NAME,
    SEARCHAPI_PROVIDER_CODE,
    SEARCHAPI_PROVIDER_NAME,
    SERPAPI_PROVIDER_CODE,
    SERPAPI_PROVIDER_NAME,
    BookingUrlKind,
    Cabin,
    ConfirmedFlightOffer,
    FlightOfferSearchResult,
    FlightOfferSegment,
    ProviderDiagnostic,
    SearchStatus,
    _airline_code,
    _candidate_coverage_status,
    _checked_bags_quantity,
    _direct_first_cabin_round_robin,
    _finite_amount,
    _google_market_for_origin,
    _iata,
    _normalized_phrase,
    _opaque_token,
    _opaque_token_digest,
    _optional_nonnegative_int,
    _optional_positive_int,
    _provider_cache_age_seconds,
    _provider_observation,
    _safe_google_url,
    _safe_response_json,
    _safe_search_id,
    _short_text,
    _split_full_flight_number,
    _status_code,
    _utc,
    _verified_offer_preference,
)
from flight_forecaster.quota_status import QuotaLedgerSnapshot
from flight_forecaster.schemas import MAX_STRICT_ITINERARY_SEGMENTS

SEARCHAPI_SEARCH_URL = "https://www.searchapi.io/api/v1/search"
SEARCHAPI_FREE_CALL_LIMIT = 100

SCRAPPA_ONE_WAY_URL = "https://scrappa.co/api/flights/one-way"
SCRAPPA_BOOKING_DETAILS_URL = "https://scrappa.co/api/flights/booking-details"
SCRAPPA_FREE_MONTHLY_LIMIT = 500

IGNAV_ONE_WAY_URL = "https://ignav.com/api/fares/one-way"
IGNAV_BOOKING_LINKS_URL = "https://ignav.com/api/fares/booking-links"
IGNAV_FREE_CALL_LIMIT = 1_000
IGNAV_MAX_ITINERARIES_PER_CABIN = 1_000
# Concurrency guard only; all eligible candidates covered by the real account
# quota are queued through this bounded worker pool.
MAX_ALTERNATE_BOOKING_WORKERS = 6

_CABINS: tuple[Cabin, ...] = (
    "economy",
    "premium_economy",
    "business",
    "first",
)
_SEARCHAPI_TRAVEL_CLASSES: dict[Cabin, str] = {
    "economy": "economy",
    "premium_economy": "premium_economy",
    "business": "business",
    "first": "first_class",
}
_SEARCHAPI_CABINS: dict[str, Cabin] = {
    "economy": "economy",
    "premium economy": "premium_economy",
    "business": "business",
    "first": "first",
    "first class": "first",
}
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_SAFE_EXCEPTION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_FREE_LEDGER_PROVIDER_CODES = {
    SEARCHAPI_PROVIDER_CODE,
    # Keep the historical quarantine code as the single durable Ignav account
    # ledger key.  This prevents an upgrade to the verified result identity
    # from resetting the one-time free allowance.
    IGNAV_QUARANTINE_PROVIDER_CODE,
}
_STRICT_PROVIDER_NAMES = {
    SERPAPI_PROVIDER_CODE: SERPAPI_PROVIDER_NAME,
    SEARCHAPI_PROVIDER_CODE: SEARCHAPI_PROVIDER_NAME,
    SCRAPPA_PROVIDER_CODE: SCRAPPA_PROVIDER_NAME,
    IGNAV_QUARANTINE_PROVIDER_CODE: IGNAV_QUARANTINE_PROVIDER_NAME,
    IGNAV_VERIFIED_PROVIDER_CODE: IGNAV_VERIFIED_PROVIDER_NAME,
}

_MONTHLY_LEDGER_PROVIDER_CODES = {SCRAPPA_PROVIDER_CODE}


class _AdapterError(RuntimeError):
    status: SearchStatus = "provider_unavailable"
    exception_type = "ProviderUnavailable"


class _AuthenticationError(_AdapterError):
    status: SearchStatus = "authentication_failed"
    exception_type = "AuthenticationError"


class _RateLimitError(_AdapterError):
    status: SearchStatus = "rate_limited"
    exception_type = "RateLimitError"


class _BudgetError(_AdapterError):
    status: SearchStatus = "budget_exhausted"
    exception_type = "BudgetExhausted"


class _ProviderError(_AdapterError):
    status: SearchStatus = "provider_error"
    exception_type = "ProviderError"


class _PayloadError(_AdapterError):
    status: SearchStatus = "provider_unavailable"
    exception_type = "PayloadError"


class _TransportError(_AdapterError):
    status: SearchStatus = "provider_error"
    exception_type = "TransportError"


class _UrllibResponse:
    def __init__(
        self,
        status_code: int,
        body: bytes,
        headers: Any = None,
    ) -> None:
        self.status_code = status_code
        self.content = body
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class _UrllibClient:
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _UrllibResponse:
        query = parse.urlencode(params)
        target = f"{url}?{query}" if query else url
        call = request.Request(target, headers=headers, method="GET")
        try:
            with request.urlopen(call, timeout=timeout) as response:  # noqa: S310
                body = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                return _UrllibResponse(int(response.status), body, response.headers)
        except Exception as exc:
            status = getattr(exc, "code", None)
            if isinstance(status, int):
                body = getattr(exc, "read", lambda *_args: b"")(
                    MAX_PROVIDER_RESPONSE_BYTES + 1
                )
                return _UrllibResponse(status, body, getattr(exc, "headers", None))
            raise

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _UrllibResponse:
        body = globals()["json"].dumps(
            json,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        call = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(call, timeout=timeout) as response:  # noqa: S310
                content = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                return _UrllibResponse(int(response.status), content, response.headers)
        except Exception as exc:
            status = getattr(exc, "code", None)
            if isinstance(status, int):
                content = getattr(exc, "read", lambda *_args: b"")(
                    MAX_PROVIDER_RESPONSE_BYTES + 1
                )
                return _UrllibResponse(status, content, getattr(exc, "headers", None))
            raise


class _FreeCallLedger:
    """Conservatively reserve free calls across processes before network I/O."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alternate_provider_free_usage (
                    provider_code TEXT PRIMARY KEY,
                    reserved_calls INTEGER NOT NULL CHECK (reserved_calls >= 0)
                );
                CREATE TABLE IF NOT EXISTS alternate_provider_diagnostics (
                    diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_code TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    http_status INTEGER,
                    exception_type TEXT NOT NULL,
                    search_id TEXT
                );
                """
            )

    def snapshot(self, provider_code: str) -> int:
        self._validate_provider(provider_code)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT reserved_calls FROM alternate_provider_free_usage
                WHERE provider_code = ?
                """,
                (provider_code,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def reserve(
        self,
        provider_code: str,
        calls: int,
        *,
        hard_limit: int,
        require_all: bool = False,
    ) -> int:
        self._validate_provider(provider_code)
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("free-call reservation is invalid")
        if isinstance(hard_limit, bool) or not isinstance(hard_limit, int) or hard_limit < 0:
            raise ValueError("free-call hard limit is invalid")
        if not isinstance(require_all, bool):
            raise ValueError("free-call reservation mode is invalid")
        if calls == 0:
            return 0
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT reserved_calls FROM alternate_provider_free_usage
                WHERE provider_code = ?
                """,
                (provider_code,),
            ).fetchone()
            used = int(row[0]) if row is not None else 0
            available = max(0, hard_limit - used)
            reserved = calls if require_all and available >= calls else (
                0 if require_all else min(calls, available)
            )
            if row is None:
                connection.execute(
                    """
                    INSERT INTO alternate_provider_free_usage(provider_code, reserved_calls)
                    VALUES (?, ?)
                    """,
                    (provider_code, reserved),
                )
            elif reserved:
                connection.execute(
                    """
                    UPDATE alternate_provider_free_usage
                    SET reserved_calls = reserved_calls + ?
                    WHERE provider_code = ?
                    """,
                    (reserved, provider_code),
                )
            connection.commit()
        return reserved

    def record(self, provider_code: str, diagnostic: ProviderDiagnostic) -> None:
        self._validate_provider(provider_code)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alternate_provider_diagnostics(
                    provider_code, observed_at, stage, http_status,
                    exception_type, search_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_code,
                    diagnostic.observed_at.isoformat(),
                    diagnostic.stage,
                    diagnostic.http_status,
                    diagnostic.exception_type,
                    diagnostic.search_id,
                ),
            )
            connection.execute(
                """
                DELETE FROM alternate_provider_diagnostics
                WHERE diagnostic_id IN (
                    SELECT diagnostic_id FROM alternate_provider_diagnostics
                    ORDER BY diagnostic_id DESC LIMIT -1 OFFSET 500
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _validate_provider(provider_code: str) -> None:
        if provider_code not in _FREE_LEDGER_PROVIDER_CODES:
            raise ValueError("unsupported free-call ledger provider")


class _MonthlyFreeCallLedger:
    """UTC calendar-month hard wall for documented recurring free credits.

    Scrappa documents 500 free credits "every month" but does not expose a
    public balance/reset endpoint or an exact reset timestamp.  This ledger is
    therefore deliberately labelled as a local ceiling: it uses the provider
    clock's UTC calendar month and still stops immediately on provider 402/429
    evidence.  It must never be presented as the supplier-reported balance.
    """

    def __init__(self, path: Path, now_provider: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now_provider = now_provider
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alternate_provider_monthly_usage (
                    provider_code TEXT NOT NULL,
                    period_key TEXT NOT NULL,
                    reserved_calls INTEGER NOT NULL CHECK (reserved_calls >= 0),
                    PRIMARY KEY(provider_code, period_key)
                );
                CREATE TABLE IF NOT EXISTS alternate_provider_diagnostics (
                    diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_code TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    http_status INTEGER,
                    exception_type TEXT NOT NULL,
                    search_id TEXT
                );
                """
            )

    def period_key(self) -> str:
        value = self._now_provider()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("monthly free-call period clock is invalid")
        return value.astimezone(UTC).strftime("%Y-%m")

    def snapshot(self, provider_code: str) -> int:
        self._validate_provider(provider_code)
        period_key = self.period_key()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT reserved_calls FROM alternate_provider_monthly_usage
                WHERE provider_code = ? AND period_key = ?
                """,
                (provider_code, period_key),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def reserve(
        self,
        provider_code: str,
        calls: int,
        *,
        hard_limit: int,
        require_all: bool = False,
    ) -> int:
        self._validate_provider(provider_code)
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("monthly free-call reservation is invalid")
        if isinstance(hard_limit, bool) or not isinstance(hard_limit, int) or hard_limit < 0:
            raise ValueError("monthly free-call hard limit is invalid")
        if not isinstance(require_all, bool):
            raise ValueError("monthly free-call reservation mode is invalid")
        if calls == 0:
            return 0
        period_key = self.period_key()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT reserved_calls FROM alternate_provider_monthly_usage
                WHERE provider_code = ? AND period_key = ?
                """,
                (provider_code, period_key),
            ).fetchone()
            used = int(row[0]) if row is not None else 0
            available = max(0, hard_limit - used)
            reserved = calls if require_all and available >= calls else (
                0 if require_all else min(calls, available)
            )
            if row is None:
                connection.execute(
                    """
                    INSERT INTO alternate_provider_monthly_usage(
                        provider_code, period_key, reserved_calls
                    ) VALUES (?, ?, ?)
                    """,
                    (provider_code, period_key, reserved),
                )
            elif reserved:
                connection.execute(
                    """
                    UPDATE alternate_provider_monthly_usage
                    SET reserved_calls = reserved_calls + ?
                    WHERE provider_code = ? AND period_key = ?
                    """,
                    (reserved, provider_code, period_key),
                )
            connection.commit()
        return reserved

    def record(self, provider_code: str, diagnostic: ProviderDiagnostic) -> None:
        self._validate_provider(provider_code)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alternate_provider_diagnostics(
                    provider_code, observed_at, stage, http_status,
                    exception_type, search_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_code,
                    diagnostic.observed_at.isoformat(),
                    diagnostic.stage,
                    diagnostic.http_status,
                    diagnostic.exception_type,
                    diagnostic.search_id,
                ),
            )
            connection.execute(
                """
                DELETE FROM alternate_provider_diagnostics
                WHERE diagnostic_id IN (
                    SELECT diagnostic_id FROM alternate_provider_diagnostics
                    ORDER BY diagnostic_id DESC LIMIT -1 OFFSET 500
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _validate_provider(provider_code: str) -> None:
        if provider_code not in _MONTHLY_LEDGER_PROVIDER_CODES:
            raise ValueError("unsupported monthly free-call ledger provider")


def read_alternate_provider_quota_snapshot(
    path: str | Path,
    *,
    provider_code: str,
    hard_limit: int,
    now: datetime,
) -> QuotaLedgerSnapshot:
    """Read a lifetime reservation counter without creating or changing its DB."""

    ledger_path = Path(path)
    if provider_code not in _FREE_LEDGER_PROVIDER_CODES or not ledger_path.is_file():
        return QuotaLedgerSnapshot.unavailable()
    try:
        limit = max(0, int(hard_limit))
    except (TypeError, ValueError):
        return QuotaLedgerSnapshot.unavailable()
    if limit < 1:
        return QuotaLedgerSnapshot.unavailable()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(ledger_path), timeout=1.0)
        connection.execute("PRAGMA query_only = ON")
        row = connection.execute(
            """
            SELECT reserved_calls FROM alternate_provider_free_usage
            WHERE provider_code = ?
            """,
            (provider_code,),
        ).fetchone()
        raw_used = int(row[0]) if row is not None else 0
        if raw_used < 0:
            return QuotaLedgerSnapshot.unavailable()
        used = min(raw_used, limit)
        return QuotaLedgerSnapshot(
            available=True,
            used=used,
            limit=limit,
            remaining=max(0, limit - used),
            period_key="lifetime",
            data_basis="local_hard_limit",
            observed_at=_utc(now),
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return QuotaLedgerSnapshot.unavailable()
    finally:
        if connection is not None:
            connection.close()


def read_monthly_provider_quota_snapshot(
    path: str | Path,
    *,
    provider_code: str,
    hard_limit: int,
    now: datetime,
) -> QuotaLedgerSnapshot:
    """Read the current UTC-month local ceiling without mutating its DB."""

    ledger_path = Path(path)
    if provider_code not in _MONTHLY_LEDGER_PROVIDER_CODES or not ledger_path.is_file():
        return QuotaLedgerSnapshot.unavailable()
    try:
        limit = max(0, int(hard_limit))
        observed_at = _utc(now)
    except (TypeError, ValueError):
        return QuotaLedgerSnapshot.unavailable()
    if limit < 1:
        return QuotaLedgerSnapshot.unavailable()
    period_key = observed_at.strftime("%Y-%m")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(ledger_path), timeout=1.0)
        connection.execute("PRAGMA query_only = ON")
        row = connection.execute(
            """
            SELECT reserved_calls FROM alternate_provider_monthly_usage
            WHERE provider_code = ? AND period_key = ?
            """,
            (provider_code, period_key),
        ).fetchone()
        raw_used = int(row[0]) if row is not None else 0
        if raw_used < 0:
            return QuotaLedgerSnapshot.unavailable()
        used = min(raw_used, limit)
        return QuotaLedgerSnapshot(
            available=True,
            used=used,
            limit=limit,
            remaining=max(0, limit - used),
            period_key=period_key,
            data_basis="local_hard_limit",
            observed_at=observed_at,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return QuotaLedgerSnapshot.unavailable()
    finally:
        if connection is not None:
            connection.close()


class _Diagnostics:
    def __init__(
        self,
        provider_code: str,
        ledger: Any,
        now_provider: Any,
    ) -> None:
        self.provider_code = provider_code
        self.ledger = ledger
        self.now_provider = now_provider
        self.items: list[ProviderDiagnostic] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        stage: str,
        http_status: int | None,
        exception_type: str,
        search_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> None:
        safe_stage = stage if stage in {
            "account",
            "cabin_search",
            "booking_options",
            "search_archive",
            "validation",
        } else "validation"
        safe_type = (
            exception_type
            if _SAFE_EXCEPTION_PATTERN.fullmatch(exception_type)
            else "UnknownError"
        )
        safe_id = search_id if search_id and _SAFE_ID_PATTERN.fullmatch(search_id) else None
        diagnostic = ProviderDiagnostic(
            observed_at=_utc(observed_at or self.now_provider()),
            stage=safe_stage,  # type: ignore[arg-type]
            http_status=(http_status if http_status and 100 <= http_status <= 599 else None),
            exception_type=safe_type,
            search_id=safe_id,
        )
        with self._lock:
            self.ledger.record(self.provider_code, diagnostic)
            if len(self.items) < 10:
                self.items.append(diagnostic)

    def snapshot(self) -> tuple[ProviderDiagnostic, ...]:
        return tuple(self.items)


@dataclass(frozen=True, slots=True)
class _SearchApiCandidate:
    booking_token: str
    cabin: Cabin
    search_price_usd: float
    segments: tuple[FlightOfferSegment, ...]
    airline_name: str
    google_flights_url: str | None

    @property
    def identity(self) -> tuple[Any, ...]:
        return _segment_identity(self.segments)


@dataclass(frozen=True, slots=True)
class _ScrappaCandidate:
    booking_token: str
    cabin: Cabin
    search_price_usd: float
    total_duration_minutes: int
    segments: tuple[FlightOfferSegment, ...]
    airline_name: str

    @property
    def identity(self) -> tuple[Any, ...]:
        return _segment_identity(self.segments)

    @property
    def lead_airline_code(self) -> str:
        return self.segments[0].marketing_airline_code

    @property
    def lead_flight_number(self) -> str:
        return self.segments[0].flight_number


def _searchapi_failed_cabin_status(statuses: list[SearchStatus]) -> SearchStatus:
    """Return a stable sanitized status for a partially failed cabin sweep."""

    priority: tuple[SearchStatus, ...] = (
        "authentication_failed",
        "rate_limited",
        "budget_exhausted",
        "provider_error",
        "provider_unavailable",
    )
    observed = set(statuses)
    return next((status for status in priority if status in observed), "provider_unavailable")


@dataclass(frozen=True, slots=True)
class _IgnavCandidate:
    ignav_id: str
    cabin: Cabin
    search_price_usd: float
    segments: tuple[FlightOfferSegment, ...]
    airline_name: str
    checked_bags_quantity: int | None

    @property
    def identity(self) -> tuple[Any, ...]:
        return _segment_identity(self.segments)


@dataclass(frozen=True, slots=True)
class _BookingEvidence:
    amount_usd: float
    booking_url: str
    booking_url_kind: BookingUrlKind
    booking_provider: str
    fare_brand: str | None = None
    refundable: bool | None = None
    no_penalty: bool | None = None
    no_restriction: bool | None = None
    checked_bags_quantity: int | None = None


class _AdapterBase:
    provider_code: str
    provider_name: str
    ledger_provider_code: str
    free_call_limit: int

    def __init__(
        self,
        *,
        usage_path: Path,
        client: Any = None,
        timeout_seconds: float = 25.0,
        now_provider: Any = None,
    ) -> None:
        self._client = client or _UrllibClient()
        self._timeout_seconds = max(0.1, min(float(timeout_seconds), 30.0))
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._ledger = _FreeCallLedger(usage_path)
        self._operation_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: dict[
            tuple[str, str, date],
            tuple[float, FlightOfferSearchResult],
        ] = {}

    def _provider_now(self) -> datetime:
        try:
            return _utc(self._now_provider())
        except (TypeError, ValueError, OverflowError) as exc:
            raise _PayloadError("provider clock is invalid") from exc

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        stage: str,
        diagnostics: _Diagnostics,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], datetime, int]:
        try:
            if method == "GET":
                response = self._client.get(
                    url,
                    params=params or {},
                    headers=headers or {},
                    timeout=self._timeout_seconds,
                )
            elif method == "POST":
                response = self._client.post(
                    url,
                    json=body or {},
                    headers=headers or {},
                    timeout=self._timeout_seconds,
                )
            else:
                raise ValueError("unsupported HTTP method")
        except Exception as exc:
            diagnostics.record(
                stage=stage,
                http_status=None,
                exception_type="TransportError",
            )
            raise _TransportError("provider transport failed") from exc

        received_at = self._provider_now()
        status = _status_code(response)
        search_id = _response_search_id(response)
        if status in {401, 403}:
            error: _AdapterError = _AuthenticationError("provider authentication failed")
        elif status == 429:
            error = _RateLimitError("provider rate limit reached")
        elif status == 402:
            error = _BudgetError("provider free allowance is exhausted")
        elif status < 200 or status >= 300:
            error = _ProviderError("provider request failed")
        else:
            error = None  # type: ignore[assignment]
        if error is not None:
            diagnostics.record(
                stage=stage,
                http_status=status,
                exception_type=error.exception_type,
                search_id=search_id,
                observed_at=received_at,
            )
            raise error
        try:
            payload = _safe_response_json(response)
        except Exception as exc:
            diagnostics.record(
                stage=stage,
                http_status=status,
                exception_type="PayloadError",
                observed_at=received_at,
            )
            raise _PayloadError("provider response is not safe JSON") from exc
        if not isinstance(payload, dict):
            diagnostics.record(
                stage=stage,
                http_status=status,
                exception_type="PayloadError",
                observed_at=received_at,
            )
            raise _PayloadError("provider payload must be an object")
        return payload, received_at, status

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
        refreshed: list[ConfirmedFlightOffer] = []
        for offer in result.offers:
            age = _provider_cache_age_seconds(offer.verified_at, observed_at)
            if age is None:
                return None
            refreshed.append(
                replace(
                    offer,
                    provider_cache_hit=True,
                    provider_cache_age_seconds=age,
                )
            )
        return replace(
            result,
            offers=tuple(refreshed),
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
        diagnostics: tuple[ProviderDiagnostic, ...] = (),
        eligible_candidate_count: int = 0,
        verification_attempted_count: int = 0,
        verified_candidate_count: int = 0,
        strictly_rejected_candidate_count: int = 0,
        provider_failed_candidate_count: int = 0,
        search_failed_cabin_count: int = 0,
        quota_skipped_candidate_count: int = 0,
        deduplicated_verified_count: int = 0,
        coverage_status: str = "not_evaluated",
        quota_limit: str | None = None,
        retry_quota_limited: bool = False,
    ) -> FlightOfferSearchResult:
        used = self._ledger.snapshot(self.ledger_provider_code)
        return FlightOfferSearchResult(
            offers=offers,
            status=status,
            observed_at=observed_at,
            environment=self.environment,
            searched_cabins=searched_cabins,
            calls_used=search_calls_used + pricing_calls_used,
            cache_hit=False,
            search_calls_used=search_calls_used,
            pricing_calls_used=pricing_calls_used,
            search_monthly_limit=self.free_call_limit,
            pricing_monthly_limit=None,
            search_monthly_used=used,
            pricing_monthly_used=None,
            diagnostics=diagnostics,
            eligible_candidate_count=eligible_candidate_count,
            verification_attempted_count=verification_attempted_count,
            verified_candidate_count=verified_candidate_count,
            strictly_rejected_candidate_count=strictly_rejected_candidate_count,
            provider_failed_candidate_count=provider_failed_candidate_count,
            search_failed_cabin_count=search_failed_cabin_count,
            quota_skipped_candidate_count=quota_skipped_candidate_count,
            deduplicated_verified_count=deduplicated_verified_count,
            coverage_status=coverage_status,  # type: ignore[arg-type]
            quota_limit=quota_limit,  # type: ignore[arg-type]
            retry_quota_limited=retry_quota_limited,
            provider_code=self.provider_code,
            provider_name=self.provider_name,
        )


def _response_search_id(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is not None:
        for key in ("x-request-id", "x-search-id"):
            try:
                candidate = str(headers.get(key, "")).strip()
            except Exception:
                candidate = ""
            if _SAFE_ID_PATTERN.fullmatch(candidate):
                return candidate
    try:
        payload = _safe_response_json(response)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("search_metadata")
    if isinstance(metadata, dict):
        return _safe_search_id(metadata.get("id"))
    for key in ("request_id", "search_id"):
        candidate = str(payload.get(key, "")).strip()
        if _SAFE_ID_PATTERN.fullmatch(candidate):
            return candidate
    return None


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


def _lowest_verified_offers(
    confirmed_by_position: dict[int, ConfirmedFlightOffer],
    attempted: int,
) -> tuple[tuple[ConfirmedFlightOffer, ...], int]:
    groups: dict[tuple[Any, ...], tuple[int, ConfirmedFlightOffer]] = {}
    for position in range(attempted):
        offer = confirmed_by_position.get(position)
        if offer is None:
            continue
        key = offer.lowest_price_group_key
        current = groups.get(key)
        if current is None:
            groups[key] = (position, offer)
        elif _verified_offer_preference(offer) < _verified_offer_preference(current[1]):
            groups[key] = (current[0], offer)
    offers = tuple(offer for _, offer in sorted(groups.values(), key=lambda item: item[0]))
    return offers, len(confirmed_by_position) - len(offers)


def _safe_public_https_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 2_048 or any(ord(char) < 32 for char in candidate):
        return None
    try:
        parts = parse.urlsplit(candidate)
        hostname = parts.hostname
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    if (
        parts.scheme.lower() != "https"
        or not hostname
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or hostname.lower() == "localhost"
        or hostname.lower().endswith((".localhost", ".local"))
    ):
        return None
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return None
    return candidate


def _clean_id(value: Any) -> str | None:
    candidate = str(value or "").strip()
    return candidate if _SAFE_ID_PATTERN.fullmatch(candidate) else None


def _positive_duration(value: Any) -> int | None:
    parsed = _optional_positive_int(value)
    return parsed if parsed is not None and parsed <= 2_880 else None


def _naive_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not 16 <= len(value.strip()) <= 19:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is not None or parsed.microsecond != 0:
        return None
    return parsed.replace(second=0)


def _aware_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not 20 <= len(value.strip()) <= 40:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).replace(microsecond=0)


def _local_matches_utc(local: datetime, zone_name: Any, utc_value: datetime) -> bool:
    if not isinstance(zone_name, str) or not 1 <= len(zone_name.strip()) <= 80:
        return False
    try:
        zone = ZoneInfo(zone_name.strip())
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return any(
        local.replace(tzinfo=zone, fold=fold).astimezone(UTC).replace(microsecond=0)
        == utc_value
        for fold in (0, 1)
    )


class SearchApiFlightOfferProvider(_AdapterBase):
    """Strict SearchAPI.io Google Flights adapter with a free-only hard wall."""

    provider_code = SEARCHAPI_PROVIDER_CODE
    provider_name = SEARCHAPI_PROVIDER_NAME
    ledger_provider_code = SEARCHAPI_PROVIDER_CODE

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: Path,
        monthly_limit: int | None = SEARCHAPI_FREE_CALL_LIMIT,
        client: Any = None,
        timeout_seconds: float = 25.0,
        now_provider: Any = None,
    ) -> None:
        super().__init__(
            usage_path=usage_path,
            client=client,
            timeout_seconds=timeout_seconds,
            now_provider=now_provider,
        )
        self._api_key = (api_key or "").strip() or None
        parsed_limit = _optional_positive_int(monthly_limit)
        self.free_call_limit = min(
            parsed_limit or SEARCHAPI_FREE_CALL_LIMIT,
            SEARCHAPI_FREE_CALL_LIMIT,
        )

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    @property
    def environment(self) -> str:
        return "production" if self.configured else "disabled"

    @property
    def monthly_limit(self) -> int:
        return self.free_call_limit

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
            result = self._search_uncached(route[0], route[1], departure_date, observed_at)
            if result.status in {"confirmed_offers", "no_results"}:
                self._remember(key, result)
            return result

    def _search_uncached(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        observed_at: datetime,
    ) -> FlightOfferSearchResult:
        diagnostics = _Diagnostics(self.ledger_provider_code, self._ledger, self._provider_now)
        local_used = self._ledger.snapshot(self.ledger_provider_code)
        lifetime_capacity = max(0, self.free_call_limit - local_used)
        if lifetime_capacity < len(_CABINS):
            return self._result(
                "budget_exhausted",
                observed_at,
                diagnostics=diagnostics.snapshot(),
                quota_limit="lifetime",
            )
        hard_limit = self.free_call_limit
        search_reservation = self._ledger.reserve(
            self.ledger_provider_code,
            len(_CABINS),
            hard_limit=hard_limit,
            require_all=True,
        )
        if search_reservation != len(_CABINS):
            return self._result(
                "budget_exhausted",
                observed_at,
                diagnostics=diagnostics.snapshot(),
                quota_limit="lifetime",
            )

        payloads: dict[Cabin, dict[str, Any]] = {}
        searched_cabins: list[Cabin] = list(_CABINS)
        failed_cabin_statuses: list[SearchStatus] = []
        with ThreadPoolExecutor(
            max_workers=len(_CABINS),
            thread_name_prefix="searchapi-search",
        ) as pool:
            futures = {
                pool.submit(
                    self._search_cabin,
                    origin,
                    destination,
                    departure_date,
                    cabin,
                    diagnostics,
                ): cabin
                for cabin in _CABINS
            }
            for future in as_completed(futures):
                cabin = futures[future]
                try:
                    payloads[cabin] = future.result()
                except _AdapterError as exc:
                    failed_cabin_statuses.append(exc.status)
                except Exception:
                    failed_cabin_statuses.append("provider_unavailable")
                    diagnostics.record(
                        stage="cabin_search",
                        http_status=None,
                        exception_type="PayloadError",
                    )

        candidates = _select_searchapi_candidates(
            payloads,
            origin,
            destination,
            departure_date,
        )
        if not candidates:
            failed_count = len(failed_cabin_statuses)
            status: SearchStatus = (
                _searchapi_failed_cabin_status(failed_cabin_statuses)
                if failed_count
                else "no_results"
            )
            rate_limited = status == "rate_limited"
            return self._result(
                status,
                observed_at,
                searched_cabins=tuple(searched_cabins),
                search_calls_used=len(searched_cabins),
                diagnostics=diagnostics.snapshot(),
                search_failed_cabin_count=failed_count,
                coverage_status=(
                    "quota_and_provider_incomplete"
                    if rate_limited
                    else ("provider_incomplete" if failed_count else "complete")
                ),
                quota_limit=("hourly" if rate_limited else None),
                retry_quota_limited=rate_limited,
            )

        # Reserve every eligible candidate atomically. The lifetime ledger may
        # return a smaller reservation when the real account quota is exhausted;
        # concurrency remains bounded independently of candidate count.
        confirmed_by_position: dict[int, ConfirmedFlightOffer] = {}
        provider_failures = 0
        strict_rejections = 0
        requested = len(candidates)
        reserved = self._ledger.reserve(
            self.ledger_provider_code,
            requested,
            hard_limit=hard_limit,
        )
        attempted_candidates = candidates[:reserved]
        if attempted_candidates:
            with ThreadPoolExecutor(
                max_workers=min(MAX_ALTERNATE_BOOKING_WORKERS, len(attempted_candidates)),
                thread_name_prefix="searchapi-booking",
            ) as pool:
                futures = {
                    pool.submit(
                        self._booking_options,
                        candidate,
                        origin,
                        destination,
                        departure_date,
                        diagnostics,
                    ): (position, candidate)
                    for position, candidate in enumerate(attempted_candidates)
                }
                for future in as_completed(futures):
                    position, candidate = futures[future]
                    try:
                        payload, received_at, http_status = future.result()
                        offer = _parse_searchapi_booking_confirmation(
                            payload,
                            candidate,
                            origin,
                            destination,
                            departure_date,
                            received_at,
                        )
                    except _AdapterError:
                        provider_failures += 1
                        continue
                    except Exception:
                        provider_failures += 1
                        diagnostics.record(
                            stage="validation",
                            http_status=None,
                            exception_type="PayloadError",
                        )
                        continue
                    if offer is None:
                        strict_rejections += 1
                        diagnostics.record(
                            stage="validation",
                            http_status=http_status,
                            exception_type="StrictCandidateRejected",
                            search_id=_searchapi_payload_id(payload),
                            observed_at=received_at,
                        )
                    else:
                        confirmed_by_position[position] = offer

        verified_count = len(confirmed_by_position)
        quota_skipped = len(candidates) - len(attempted_candidates)
        coverage_status = _candidate_coverage_status(
            evaluated=True,
            provider_failed=provider_failures,
            quota_skipped=quota_skipped,
            search_failed_cabins=len(failed_cabin_statuses),
        )
        offers, deduplicated = _lowest_verified_offers(
            confirmed_by_position,
            len(attempted_candidates),
        )
        quota_limit = "lifetime" if quota_skipped else None
        common: dict[str, Any] = {
            "searched_cabins": tuple(searched_cabins),
            "search_calls_used": len(searched_cabins),
            "pricing_calls_used": len(attempted_candidates),
            "diagnostics": diagnostics.snapshot(),
            "eligible_candidate_count": len(candidates),
            "verification_attempted_count": len(attempted_candidates),
            "verified_candidate_count": verified_count,
            "strictly_rejected_candidate_count": strict_rejections,
            "provider_failed_candidate_count": provider_failures,
            "search_failed_cabin_count": len(failed_cabin_statuses),
            "quota_skipped_candidate_count": quota_skipped,
            "deduplicated_verified_count": deduplicated,
            "coverage_status": coverage_status,
            "quota_limit": quota_limit,
        }
        if offers:
            return self._result(
                "confirmed_offers",
                observed_at,
                offers=offers,
                **common,
            )
        if quota_skipped:
            status = "budget_exhausted"
        elif provider_failures:
            status = "provider_error"
        else:
            status = "no_results"
        return self._result(status, observed_at, **common)

    def _search_cabin(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        cabin: Cabin,
        diagnostics: _Diagnostics,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_flights",
            "flight_type": "one_way",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": departure_date.isoformat(),
            "travel_class": _SEARCHAPI_TRAVEL_CLASSES[cabin],
            "adults": 1,
            "currency": "USD",
            "hl": "en",
            "gl": _google_market_for_origin(origin),
            "show_hidden_flights": "true",
        }
        payload, received_at, http_status = self._request_json(
            "GET",
            SEARCHAPI_SEARCH_URL,
            stage="cabin_search",
            diagnostics=diagnostics,
            params=params,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "flight-forecast-lab/0.2.0 (strict booking verification)",
            },
        )
        try:
            _provider_observation(payload, received_at)
            if not _searchapi_parameters_match(
                payload,
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                cabin=cabin,
            ):
                raise _PayloadError("SearchAPI search parameters did not match")
        except Exception as exc:
            diagnostics.record(
                stage="validation",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=_searchapi_payload_id(payload),
                observed_at=received_at,
            )
            if isinstance(exc, _AdapterError):
                raise
            raise _PayloadError("SearchAPI search payload failed validation") from exc
        return payload

    def _booking_options(
        self,
        candidate: _SearchApiCandidate,
        origin: str,
        destination: str,
        departure_date: date,
        diagnostics: _Diagnostics,
    ) -> tuple[dict[str, Any], datetime, int]:
        params: dict[str, Any] = {
            "engine": "google_flights",
            "flight_type": "one_way",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": departure_date.isoformat(),
            "travel_class": _SEARCHAPI_TRAVEL_CLASSES[candidate.cabin],
            "adults": 1,
            "currency": "USD",
            "hl": "en",
            "gl": _google_market_for_origin(origin),
            "booking_token": candidate.booking_token,
        }
        payload, received_at, http_status = self._request_json(
            "GET",
            SEARCHAPI_SEARCH_URL,
            stage="booking_options",
            diagnostics=diagnostics,
            params=params,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "flight-forecast-lab/0.2.0 (strict booking verification)",
            },
        )
        try:
            _provider_observation(payload, received_at)
            if not _searchapi_booking_parameters_match(
                payload,
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                cabin=candidate.cabin,
            ):
                raise _PayloadError("SearchAPI booking parameters did not match")
        except Exception as exc:
            diagnostics.record(
                stage="validation",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=_searchapi_payload_id(payload),
                observed_at=received_at,
            )
            if isinstance(exc, _AdapterError):
                raise
            raise _PayloadError("SearchAPI booking payload failed validation") from exc
        return payload, received_at, http_status


def _searchapi_parameters_match(
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
    return bool(
        _normalized_phrase(parameters.get("engine")) == "google flights"
        and _normalized_phrase(parameters.get("flight_type")) == "one way"
        and _iata(parameters.get("departure_id")) == origin
        and _iata(parameters.get("arrival_id")) == destination
        and str(parameters.get("outbound_date", "")) == departure_date.isoformat()
        and str(parameters.get("currency", "")).strip().upper() == "USD"
        and _normalized_phrase(parameters.get("travel_class"))
        == _normalized_phrase(_SEARCHAPI_TRAVEL_CLASSES[cabin])
    )


def _searchapi_booking_parameters_match(
    payload: dict[str, Any],
    *,
    origin: str,
    destination: str,
    departure_date: date,
    cabin: Cabin,
) -> bool:
    parameters = payload.get("search_parameters")
    if not isinstance(parameters, dict) or _opaque_token(parameters.get("booking_token")) is None:
        return False
    return _searchapi_parameters_match(
        payload,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        cabin=cabin,
    )


def _searchapi_payload_id(payload: Any) -> str | None:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    return _safe_search_id(metadata.get("id")) if isinstance(metadata, dict) else None


def _searchapi_google_flights_url(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("search_metadata")
    if not isinstance(metadata, dict):
        return None
    for key in ("request_url", "google_flights_url"):
        candidate = _safe_google_url(metadata.get(key), path_prefix="/travel/flights")
        if candidate is not None:
            return candidate
    return None


def _select_searchapi_candidates(
    payloads: dict[Cabin, dict[str, Any]],
    origin: str,
    destination: str,
    departure_date: date,
) -> tuple[_SearchApiCandidate, ...]:
    by_cabin: dict[Cabin, list[_SearchApiCandidate]] = {cabin: [] for cabin in _CABINS}
    for cabin in _CABINS:
        payload = payloads.get(cabin, {})
        google_url = _searchapi_google_flights_url(payload)
        rows: list[Any] = []
        for key in ("best_flights", "other_flights"):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(value)
        by_token: dict[str, _SearchApiCandidate] = {}
        conflicting: set[str] = set()
        for row in rows:
            candidate = _parse_searchapi_candidate(
                row,
                cabin,
                origin,
                destination,
                departure_date,
                google_url,
            )
            if candidate is None or candidate.booking_token in conflicting:
                continue
            existing = by_token.get(candidate.booking_token)
            if existing is None:
                by_token[candidate.booking_token] = candidate
            elif existing.identity != candidate.identity:
                by_token.pop(candidate.booking_token, None)
                conflicting.add(candidate.booking_token)
            elif candidate.search_price_usd < existing.search_price_usd:
                by_token[candidate.booking_token] = candidate
        by_cabin[cabin] = list(by_token.values())
    return _direct_first_cabin_round_robin(by_cabin)


def _parse_searchapi_candidate(
    row: Any,
    cabin: Cabin,
    origin: str,
    destination: str,
    departure_date: date,
    google_flights_url: str | None,
) -> _SearchApiCandidate | None:
    if not isinstance(row, dict) or _normalized_phrase(row.get("type")) != "one way":
        return None
    token = _opaque_token(row.get("booking_token"))
    amount = _finite_amount(row.get("price"))
    raw_segments = row.get("flights")
    if token is None or amount is None or amount <= 0 or not isinstance(raw_segments, list):
        return None
    segments = _parse_searchapi_segments(raw_segments, cabin)
    if not _segments_match_request(segments, origin, destination, departure_date):
        return None
    return _SearchApiCandidate(
        booking_token=token,
        cabin=cabin,
        search_price_usd=amount,
        segments=segments,
        airline_name=_short_text(raw_segments[0].get("airline"), max_length=160)
        or segments[0].marketing_airline_code,
        google_flights_url=google_flights_url,
    )


def _parse_searchapi_segments(
    rows: list[Any],
    searched_cabin: Cabin,
) -> tuple[FlightOfferSegment, ...]:
    if not 1 <= len(rows) <= MAX_STRICT_ITINERARY_SEGMENTS:
        return ()
    segments: list[FlightOfferSegment] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return ()
        departure = row.get("departure_airport")
        arrival = row.get("arrival_airport")
        if not isinstance(departure, dict) or not isinstance(arrival, dict):
            return ()
        origin = _iata(departure.get("id"))
        destination = _iata(arrival.get("id"))
        departure_at = _searchapi_airport_datetime(departure)
        arrival_at = _searchapi_airport_datetime(arrival)
        parsed_number = _split_full_flight_number(row.get("flight_number"))
        returned_cabin = _SEARCHAPI_CABINS.get(_normalized_phrase(row.get("travel_class")))
        duration = _positive_duration(row.get("duration"))
        if (
            origin is None
            or destination is None
            or origin == destination
            or departure_at is None
            or arrival_at is None
            or parsed_number is None
            or returned_cabin != searched_cabin
            or duration is None
        ):
            return ()
        if segments:
            previous = segments[-1]
            if previous.destination != origin or departure_at <= previous.arrival_at:
                return ()
        airline_code, number = parsed_number
        segments.append(
            FlightOfferSegment(
                segment_id=f"{airline_code}{number}-{departure_at:%Y%m%d%H%M}-{index}",
                origin=origin,
                destination=destination,
                departure_at=departure_at,
                arrival_at=arrival_at,
                marketing_airline_code=airline_code,
                operating_airline_code=None,
                flight_number=number,
                departure_terminal=_short_text(departure.get("terminal"), max_length=40),
                arrival_terminal=_short_text(arrival.get("terminal"), max_length=40),
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


def _searchapi_airport_datetime(value: dict[str, Any]) -> datetime | None:
    date_text = str(value.get("date", "")).strip()
    time_text = str(value.get("time", "")).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text) or not re.fullmatch(
        r"\d{2}:\d{2}", time_text
    ):
        return None
    try:
        return datetime.fromisoformat(f"{date_text}T{time_text}")
    except ValueError:
        return None


def _parse_searchapi_booking_confirmation(
    payload: dict[str, Any],
    candidate: _SearchApiCandidate,
    origin: str,
    destination: str,
    departure_date: date,
    received_at: datetime,
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
    segments = _parse_searchapi_segments(itinerary["flights"], candidate.cabin)
    if (
        not _segments_match_request(segments, origin, destination, departure_date)
        or _segment_identity(segments) != candidate.identity
    ):
        return None
    observation = _provider_observation(payload, received_at)
    evidence = _searchapi_booking_evidence(
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
            replace(segment, checked_bags_quantity=evidence.checked_bags_quantity)
            for segment in segments
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "token": _opaque_token_digest(candidate.booking_token),
                "seller": evidence.booking_provider,
                "url": evidence.booking_url,
                "price": evidence.amount_usd,
                "segments": _segment_identity(segments),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return ConfirmedFlightOffer(
        provider_offer_id=f"searchapi-{digest}",
        validating_airline_code=segments[0].marketing_airline_code,
        airline_name=_short_text(itinerary["flights"][0].get("airline"), max_length=160)
        or candidate.airline_name,
        cabin=candidate.cabin,
        total_amount_usd=evidence.amount_usd,
        base_amount_usd=None,
        last_ticketing_date=None,
        number_of_bookable_seats=None,
        seat_count_capped=False,
        verified_at=observation.created_at,
        provider_cache_hit=observation.cache_hit,
        provider_cache_age_seconds=observation.cache_age_seconds,
        segments=segments,
        refundable_fare=evidence.refundable,
        no_penalty_fare=evidence.no_penalty,
        no_restriction_fare=evidence.no_restriction,
        booking_url=evidence.booking_url,
        booking_url_kind=evidence.booking_url_kind,
        booking_provider=evidence.booking_provider,
        provider_code=SEARCHAPI_PROVIDER_CODE,
        provider_name=SEARCHAPI_PROVIDER_NAME,
        environment="production",
    )


def _searchapi_booking_evidence(
    payload: dict[str, Any],
    segments: tuple[FlightOfferSegment, ...],
    *,
    fallback_google_flights_url: str | None,
) -> _BookingEvidence | None:
    options = payload.get("booking_options")
    if not isinstance(options, list):
        return None
    expected_numbers = [
        f"{segment.marketing_airline_code}{segment.flight_number}".upper()
        for segment in segments
    ]
    response_google_url = _searchapi_google_flights_url(payload)
    evidence: list[_BookingEvidence] = []
    for option in options:
        if not isinstance(option, dict) or option.get("is_split_booking") is True:
            continue
        provider = _short_text(option.get("book_with"), max_length=160)
        amount = _finite_amount(option.get("price"))
        raw_numbers = option.get("flight_numbers")
        request_data = option.get("booking_request")
        if (
            provider is None
            or amount is None
            or amount <= 0
            or not isinstance(raw_numbers, list)
            or not isinstance(request_data, dict)
        ):
            continue
        normalized_numbers: list[str] = []
        invalid_number = False
        for value in raw_numbers:
            parsed_number = _split_full_flight_number(value)
            if parsed_number is None:
                invalid_number = True
                break
            normalized_numbers.append(f"{parsed_number[0]}{parsed_number[1]}")
        if invalid_number or normalized_numbers != expected_numbers:
            continue
        action_url = _safe_google_url(request_data.get("url"), path_prefix="/travel/clk/")
        if action_url is None:
            continue
        post_data = request_data.get("post_data")
        if post_data is None:
            booking_url = action_url
            kind: BookingUrlKind = "direct_get"
        elif isinstance(post_data, str) and post_data.strip():
            booking_url = response_google_url or fallback_google_flights_url
            kind = "google_flights_itinerary"
        else:
            continue
        if booking_url is None:
            continue
        phrases = (
            [_normalized_phrase(value) for value in option.get("extensions", [])]
            if isinstance(option.get("extensions"), list)
            else []
        )
        evidence.append(
            _BookingEvidence(
                amount_usd=amount,
                booking_url=booking_url,
                booking_url_kind=kind,
                booking_provider=provider,
                fare_brand=_short_text(
                    option.get("option_title") or option.get("fare_type"),
                    max_length=80,
                ),
                refundable=_phrase_flag(
                    phrases,
                    positive=("refundable", "refunds allowed"),
                    negative=("no refunds", "nonrefundable", "non refundable"),
                ),
                no_penalty=_phrase_flag(
                    phrases,
                    positive=("free changes", "changes permitted without fee"),
                    negative=("changes for a fee", "no ticket changes", "no changes"),
                ),
                no_restriction=_phrase_flag(
                    phrases,
                    positive=("no restrictions",),
                    negative=("restrictions apply",),
                ),
                checked_bags_quantity=_checked_bags_quantity(option.get("baggage_prices")),
            )
        )
    return (
        min(evidence, key=lambda item: (item.amount_usd, item.booking_provider))
        if evidence
        else None
    )


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


class ScrappaFlightOfferProvider(_AdapterBase):
    """Strict Scrappa Google Flights adapter with a UTC-month local hard wall.

    Scrappa documents 500 recurring free credits per month.  Its public API
    does not expose the account reset timestamp or remaining balance, so the
    counter below is intentionally conservative and is never represented as a
    provider-reported balance.  HTTP 402/429 evidence stops the current sweep.
    """

    provider_code = SCRAPPA_PROVIDER_CODE
    provider_name = SCRAPPA_PROVIDER_NAME
    ledger_provider_code = SCRAPPA_PROVIDER_CODE

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: Path,
        monthly_limit: int | None = SCRAPPA_FREE_MONTHLY_LIMIT,
        client: Any = None,
        timeout_seconds: float = 25.0,
        now_provider: Any = None,
    ) -> None:
        super().__init__(
            usage_path=usage_path,
            client=client,
            timeout_seconds=timeout_seconds,
            now_provider=now_provider,
        )
        self._api_key = (api_key or "").strip() or None
        parsed_limit = _optional_positive_int(monthly_limit)
        self.free_call_limit = min(
            parsed_limit or SCRAPPA_FREE_MONTHLY_LIMIT,
            SCRAPPA_FREE_MONTHLY_LIMIT,
        )
        self._ledger = _MonthlyFreeCallLedger(usage_path, self._now_provider)

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    @property
    def environment(self) -> str:
        return "production" if self.configured else "disabled"

    @property
    def monthly_limit(self) -> int:
        return self.free_call_limit

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
            result = self._search_uncached(route[0], route[1], departure_date, observed_at)
            if result.status in {"confirmed_offers", "no_results"}:
                self._remember(key, result)
            return result

    def _search_uncached(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        observed_at: datetime,
    ) -> FlightOfferSearchResult:
        diagnostics = _Diagnostics(self.ledger_provider_code, self._ledger, self._provider_now)
        if self.free_call_limit - self._ledger.snapshot(self.ledger_provider_code) < len(_CABINS):
            return self._result(
                "budget_exhausted",
                observed_at,
                diagnostics=diagnostics.snapshot(),
                quota_limit="monthly",
            )
        reserved_searches = self._ledger.reserve(
            self.ledger_provider_code,
            len(_CABINS),
            hard_limit=self.free_call_limit,
            require_all=True,
        )
        if reserved_searches != len(_CABINS):
            return self._result(
                "budget_exhausted",
                observed_at,
                diagnostics=diagnostics.snapshot(),
                quota_limit="monthly",
            )

        payloads: dict[Cabin, dict[str, Any]] = {}
        searched_cabins: list[Cabin] = []
        failed_statuses: list[SearchStatus] = []
        for cabin in _CABINS:
            searched_cabins.append(cabin)
            try:
                payloads[cabin] = self._search_cabin(
                    origin,
                    destination,
                    departure_date,
                    cabin,
                    diagnostics,
                )
            except _AdapterError as exc:
                failed_statuses.append(exc.status)
                # Provider quota/rate evidence is authoritative for this run.
                if exc.status in {"budget_exhausted", "rate_limited"}:
                    is_rate_limited = exc.status == "rate_limited"
                    return self._result(
                        exc.status,
                        observed_at,
                        searched_cabins=tuple(searched_cabins),
                        search_calls_used=len(searched_cabins),
                        diagnostics=diagnostics.snapshot(),
                        search_failed_cabin_count=(1 if is_rate_limited else 0),
                        coverage_status=(
                            "quota_and_provider_incomplete"
                            if is_rate_limited
                            else "not_evaluated"
                        ),
                        quota_limit=("hourly" if is_rate_limited else "monthly"),
                        retry_quota_limited=is_rate_limited,
                    )
            except Exception:
                failed_statuses.append("provider_unavailable")
                diagnostics.record(
                    stage="cabin_search",
                    http_status=None,
                    exception_type="PayloadError",
                )

        candidates = _select_scrappa_candidates(
            payloads,
            origin,
            destination,
            departure_date,
        )
        if not candidates:
            status = (
                _searchapi_failed_cabin_status(failed_statuses)
                if failed_statuses
                else "no_results"
            )
            return self._result(
                status,
                observed_at,
                searched_cabins=tuple(searched_cabins),
                search_calls_used=len(searched_cabins),
                diagnostics=diagnostics.snapshot(),
                search_failed_cabin_count=len(failed_statuses),
                coverage_status=("provider_incomplete" if failed_statuses else "complete"),
            )

        # A partial verification sweep would silently omit bookable flights.
        # Reserve the whole candidate set atomically or fail closed before any
        # booking-details call.  The worker count controls concurrency only.
        reserved_candidates = self._ledger.reserve(
            self.ledger_provider_code,
            len(candidates),
            hard_limit=self.free_call_limit,
            require_all=True,
        )
        if reserved_candidates != len(candidates):
            return self._result(
                "budget_exhausted",
                observed_at,
                searched_cabins=tuple(searched_cabins),
                search_calls_used=len(searched_cabins),
                diagnostics=diagnostics.snapshot(),
                eligible_candidate_count=len(candidates),
                quota_skipped_candidate_count=len(candidates),
                search_failed_cabin_count=len(failed_statuses),
                coverage_status=(
                    "quota_and_provider_incomplete" if failed_statuses else "quota_limited"
                ),
                quota_limit="monthly",
            )

        confirmed_by_position: dict[int, ConfirmedFlightOffer] = {}
        provider_failures = 0
        strict_rejections = 0
        booking_statuses: list[SearchStatus] = []
        with ThreadPoolExecutor(
            max_workers=min(MAX_ALTERNATE_BOOKING_WORKERS, len(candidates)),
            thread_name_prefix="scrappa-booking",
        ) as pool:
            futures = {
                pool.submit(
                    self._booking_details,
                    candidate,
                    origin,
                    destination,
                    departure_date,
                    diagnostics,
                ): (position, candidate)
                for position, candidate in enumerate(candidates)
            }
            stop_for_quota = False
            for future in as_completed(futures):
                position, candidate = futures[future]
                if future.cancelled():
                    continue
                try:
                    payload, received_at, http_status = future.result()
                    offer = _parse_scrappa_booking_confirmation(
                        payload,
                        candidate,
                        origin,
                        destination,
                        departure_date,
                        received_at,
                    )
                except _AdapterError as exc:
                    provider_failures += 1
                    booking_statuses.append(exc.status)
                    if exc.status in {"budget_exhausted", "rate_limited"}:
                        stop_for_quota = True
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                    continue
                except Exception:
                    provider_failures += 1
                    booking_statuses.append("provider_unavailable")
                    diagnostics.record(
                        stage="validation",
                        http_status=None,
                        exception_type="PayloadError",
                    )
                    continue
                if offer is None:
                    strict_rejections += 1
                    diagnostics.record(
                        stage="validation",
                        http_status=http_status,
                        exception_type="StrictCandidateRejected",
                        search_id=_scrappa_payload_id(payload),
                        observed_at=received_at,
                    )
                else:
                    confirmed_by_position[position] = offer
            if stop_for_quota:
                # Running requests cannot be forcefully interrupted safely; no
                # new work or controlled retry is issued after the evidence.
                pass

        attempted = provider_failures + strict_rejections + len(confirmed_by_position)
        cancelled = len(candidates) - attempted
        offers, deduplicated = _lowest_verified_offers(confirmed_by_position, len(candidates))
        coverage_status = _candidate_coverage_status(
            evaluated=True,
            provider_failed=provider_failures,
            quota_skipped=cancelled,
            search_failed_cabins=len(failed_statuses),
        )
        common: dict[str, Any] = {
            "searched_cabins": tuple(searched_cabins),
            "search_calls_used": len(searched_cabins),
            "pricing_calls_used": attempted,
            "diagnostics": diagnostics.snapshot(),
            "eligible_candidate_count": len(candidates),
            "verification_attempted_count": attempted,
            "verified_candidate_count": len(confirmed_by_position),
            "strictly_rejected_candidate_count": strict_rejections,
            "provider_failed_candidate_count": provider_failures,
            "search_failed_cabin_count": len(failed_statuses),
            "quota_skipped_candidate_count": cancelled,
            "deduplicated_verified_count": deduplicated,
            "coverage_status": coverage_status,
            "quota_limit": (
                (
                    "hourly"
                    if "rate_limited" in booking_statuses
                    else "monthly"
                )
                if cancelled
                else None
            ),
        }
        if offers:
            return self._result("confirmed_offers", observed_at, offers=offers, **common)
        statuses = failed_statuses + booking_statuses
        status = _searchapi_failed_cabin_status(statuses) if statuses else "no_results"
        return self._result(status, observed_at, **common)

    def _search_cabin(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        cabin: Cabin,
        diagnostics: _Diagnostics,
    ) -> dict[str, Any]:
        payload, received_at, http_status = self._request_json(
            "GET",
            SCRAPPA_ONE_WAY_URL,
            stage="cabin_search",
            diagnostics=diagnostics,
            params={
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date.isoformat(),
                "adults": 1,
                "cabin_class": cabin,
                "max_stops": "any",
                "hl": "en",
                "gl": _google_market_for_origin(origin),
                "currency": "USD",
            },
            headers={
                "Accept": "application/json",
                "x-api-key": str(self._api_key),
                "User-Agent": "flight-forecast-lab/0.2.0 (strict booking verification)",
            },
        )
        if not _scrappa_search_parameters_match(
            payload,
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            cabin=cabin,
        ):
            diagnostics.record(
                stage="validation",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=_scrappa_payload_id(payload),
                observed_at=received_at,
            )
            raise _PayloadError("Scrappa search parameters did not match")
        return payload

    def _booking_details(
        self,
        candidate: _ScrappaCandidate,
        origin: str,
        destination: str,
        departure_date: date,
        diagnostics: _Diagnostics,
    ) -> tuple[dict[str, Any], datetime, int]:
        return self._request_json(
            "GET",
            SCRAPPA_BOOKING_DETAILS_URL,
            stage="booking_options",
            diagnostics=diagnostics,
            params={
                "booking_token": candidate.booking_token,
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date.isoformat(),
                "airline": candidate.lead_airline_code,
                "flight_number": candidate.lead_flight_number,
                "adults": 1,
                "cabin_class": candidate.cabin,
                "hl": "en",
                "gl": _google_market_for_origin(origin),
                "currency": "USD",
            },
            headers={
                "Accept": "application/json",
                "x-api-key": str(self._api_key),
                "User-Agent": "flight-forecast-lab/0.2.0 (strict booking verification)",
            },
        )


def _scrappa_payload_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for container_name in ("search_metadata", "booking_metadata"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            for key in ("request_id", "search_id", "id"):
                candidate = _safe_search_id(container.get(key))
                if candidate is not None:
                    return candidate
    for key in ("request_id", "search_id"):
        candidate = _safe_search_id(payload.get(key))
        if candidate is not None:
            return candidate
    return None


def _scrappa_search_parameters_match(
    payload: dict[str, Any],
    *,
    origin: str,
    destination: str,
    departure_date: date,
    cabin: Cabin,
) -> bool:
    metadata = payload.get("search_metadata")
    if not isinstance(metadata, dict):
        return False
    passengers = metadata.get("passengers")
    return bool(
        _iata(metadata.get("origin")) == origin
        and _iata(metadata.get("destination")) == destination
        and str(metadata.get("departure_date", "")) == departure_date.isoformat()
        and _normalized_phrase(metadata.get("cabin_class"))
        == _normalized_phrase(cabin)
        and isinstance(passengers, dict)
        and passengers.get("adults") == 1
        and all(passengers.get(key, 0) == 0 for key in (
            "children", "infants_in_seat", "infants_on_lap"
        ))
    )


def _select_scrappa_candidates(
    payloads: dict[Cabin, dict[str, Any]],
    origin: str,
    destination: str,
    departure_date: date,
) -> tuple[_ScrappaCandidate, ...]:
    by_cabin: dict[Cabin, list[_ScrappaCandidate]] = {cabin: [] for cabin in _CABINS}
    for cabin in _CABINS:
        rows = payloads.get(cabin, {}).get("flights")
        if not isinstance(rows, list):
            continue
        by_token: dict[str, _ScrappaCandidate] = {}
        conflicts: set[str] = set()
        for row in rows:
            candidate = _parse_scrappa_candidate(
                row,
                cabin,
                origin,
                destination,
                departure_date,
            )
            if candidate is None or candidate.booking_token in conflicts:
                continue
            previous = by_token.get(candidate.booking_token)
            if previous is None:
                by_token[candidate.booking_token] = candidate
            elif previous.identity != candidate.identity:
                by_token.pop(candidate.booking_token, None)
                conflicts.add(candidate.booking_token)
            elif candidate.search_price_usd < previous.search_price_usd:
                by_token[candidate.booking_token] = candidate
        by_cabin[cabin] = list(by_token.values())
    return _direct_first_cabin_round_robin(by_cabin)


def _parse_scrappa_candidate(
    row: Any,
    cabin: Cabin,
    origin: str,
    destination: str,
    departure_date: date,
) -> _ScrappaCandidate | None:
    if not isinstance(row, dict):
        return None
    token = _opaque_token(row.get("booking_token"))
    amount = _finite_amount(row.get("price"))
    currency = str(row.get("currency", "")).strip().upper()
    raw_legs = row.get("legs")
    total_duration = _positive_duration(row.get("total_duration_minutes"))
    if (
        token is None
        or len(token) > 4_096
        or amount is None
        or amount <= 0
        or currency != "USD"
        or total_duration is None
        or not isinstance(raw_legs, list)
    ):
        return None
    segments = _parse_scrappa_segments(raw_legs, cabin)
    if not _segments_match_request(segments, origin, destination, departure_date):
        return None
    return _ScrappaCandidate(
        booking_token=token,
        cabin=cabin,
        search_price_usd=amount,
        total_duration_minutes=total_duration,
        segments=segments,
        airline_name=_short_text(
            raw_legs[0].get("airline_name") or raw_legs[0].get("airline"),
            max_length=160,
        ) or segments[0].marketing_airline_code,
    )


def _parse_scrappa_segments(
    rows: list[Any],
    cabin: Cabin,
) -> tuple[FlightOfferSegment, ...]:
    if not 1 <= len(rows) <= MAX_STRICT_ITINERARY_SEGMENTS:
        return ()
    segments: list[FlightOfferSegment] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("stops", 0) != 0:
            return ()
        origin = _iata(row.get("departure_airport"))
        destination = _iata(row.get("arrival_airport"))
        departure_at = _naive_iso_datetime(row.get("departure_time"))
        arrival_at = _naive_iso_datetime(row.get("arrival_time"))
        parsed_number = _split_full_flight_number(row.get("flight_number"))
        duration = _positive_duration(row.get("duration_minutes"))
        if (
            origin is None
            or destination is None
            or origin == destination
            or departure_at is None
            or arrival_at is None
            or arrival_at <= departure_at
            or parsed_number is None
            or duration is None
        ):
            return ()
        if segments and (
            segments[-1].destination != origin
            or departure_at <= segments[-1].arrival_at
        ):
            return ()
        airline_code, number = parsed_number
        returned_airline = _airline_code(row.get("airline"))
        if returned_airline is not None and returned_airline != airline_code:
            return ()
        segments.append(
            FlightOfferSegment(
                segment_id=f"{airline_code}{number}-{departure_at:%Y%m%d%H%M}-{index}",
                origin=origin,
                destination=destination,
                departure_at=departure_at,
                arrival_at=arrival_at,
                marketing_airline_code=airline_code,
                operating_airline_code=None,
                flight_number=number,
                departure_terminal=None,
                arrival_terminal=None,
                aircraft_icao=_short_text(row.get("aircraft"), max_length=40),
                cabin=cabin,
                booking_class=None,
                fare_basis=None,
                fare_brand=None,
                checked_bags_quantity=None,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            )
        )
    return tuple(segments)


def _parse_scrappa_booking_confirmation(
    payload: dict[str, Any],
    candidate: _ScrappaCandidate,
    origin: str,
    destination: str,
    departure_date: date,
    received_at: datetime,
) -> ConfirmedFlightOffer | None:
    metadata = payload.get("booking_metadata")
    details = payload.get("flight_details")
    if not isinstance(metadata, dict) or not isinstance(details, dict):
        return None
    if (
        _iata(metadata.get("origin")) != origin
        or _iata(metadata.get("destination")) != destination
        or str(metadata.get("departure_date", "")) != departure_date.isoformat()
        or _airline_code(metadata.get("airline")) != candidate.lead_airline_code
        or str(metadata.get("flight_number", "")).strip().upper()
        != candidate.lead_flight_number
    ):
        return None
    detail_code = _airline_code(details.get("airline_code"))
    detail_leg = details.get("leg")
    if detail_code != candidate.lead_airline_code or not isinstance(detail_leg, dict):
        return None
    detail_number = str(detail_leg.get("flight_number", "")).strip().upper()
    parsed_detail = _split_full_flight_number(detail_number)
    normalized_detail_number = (
        parsed_detail[1] if parsed_detail is not None else detail_number
    )
    if normalized_detail_number != candidate.lead_flight_number:
        return None
    total_duration = _positive_duration(details.get("total_duration_minutes"))
    if total_duration is None or abs(total_duration - candidate.total_duration_minutes) > 5:
        return None
    evidence = _scrappa_booking_evidence(payload, candidate)
    if evidence is None:
        return None
    segments = candidate.segments
    if evidence.fare_brand is not None:
        segments = tuple(replace(segment, fare_brand=evidence.fare_brand) for segment in segments)
    digest = hashlib.sha256(
        json.dumps(
            {
                "token": _opaque_token_digest(candidate.booking_token),
                "seller": evidence.booking_provider,
                "url": evidence.booking_url,
                "price": evidence.amount_usd,
                "segments": _segment_identity(segments),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return ConfirmedFlightOffer(
        provider_offer_id=f"scrappa-{digest}",
        validating_airline_code=candidate.lead_airline_code,
        airline_name=_short_text(details.get("airline_name"), max_length=160)
        or candidate.airline_name,
        cabin=candidate.cabin,
        total_amount_usd=evidence.amount_usd,
        base_amount_usd=None,
        last_ticketing_date=None,
        number_of_bookable_seats=None,
        seat_count_capped=False,
        verified_at=_utc(received_at),
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        segments=segments,
        refundable_fare=evidence.refundable,
        no_penalty_fare=evidence.no_penalty,
        no_restriction_fare=evidence.no_restriction,
        booking_url=evidence.booking_url,
        booking_url_kind=evidence.booking_url_kind,
        booking_provider=evidence.booking_provider,
        provider_code=SCRAPPA_PROVIDER_CODE,
        provider_name=SCRAPPA_PROVIDER_NAME,
        environment="production",
    )


def _scrappa_booking_evidence(
    payload: dict[str, Any],
    candidate: _ScrappaCandidate,
) -> _BookingEvidence | None:
    options = payload.get("fare_options")
    if not isinstance(options, list):
        return None
    expected_numbers = [
        f"{segment.marketing_airline_code}{segment.flight_number}"
        for segment in candidate.segments
    ]
    evidence: list[_BookingEvidence] = []
    for option in options:
        if not isinstance(option, dict) or option.get("is_split_booking") is True:
            continue
        currency = str(option.get("currency", "")).strip().upper()
        amount = _finite_amount(
            option.get("price")
            if option.get("price") is not None
            else option.get("total_price")
        )
        provider = _short_text(
            option.get("provider") or option.get("book_with") or option.get("seller"),
            max_length=160,
        )
        url = _safe_public_https_url(
            option.get("booking_url") or option.get("url") or option.get("deeplink")
        )
        request_data = option.get("booking_request")
        if url is None and isinstance(request_data, dict) and request_data.get("post_data") is None:
            url = _safe_public_https_url(request_data.get("url"))
        raw_numbers = option.get("flight_numbers")
        if (
            currency != "USD"
            or amount is None
            or amount <= 0
            or provider is None
            or url is None
            or not isinstance(raw_numbers, list)
        ):
            continue
        normalized_numbers: list[str] = []
        for value in raw_numbers:
            parsed = _split_full_flight_number(value)
            if parsed is None:
                break
            normalized_numbers.append(f"{parsed[0]}{parsed[1]}")
        if normalized_numbers != expected_numbers:
            continue
        phrases = (
            [_normalized_phrase(value) for value in option.get("extensions", [])]
            if isinstance(option.get("extensions"), list)
            else []
        )
        evidence.append(
            _BookingEvidence(
                amount_usd=amount,
                booking_url=url,
                booking_url_kind="direct_get",
                booking_provider=provider,
                fare_brand=_short_text(
                    option.get("fare_brand") or option.get("fare_class"),
                    max_length=80,
                ),
                refundable=_phrase_flag(
                    phrases,
                    positive=("refundable", "refunds allowed"),
                    negative=("nonrefundable", "non refundable", "no refunds"),
                ),
                no_penalty=_phrase_flag(
                    phrases,
                    positive=("free changes", "changes permitted without fee"),
                    negative=("changes for a fee", "no changes"),
                ),
                no_restriction=_phrase_flag(
                    phrases,
                    positive=("no restrictions",),
                    negative=("restrictions apply",),
                ),
            )
        )
    return (
        min(evidence, key=lambda item: (item.amount_usd, item.booking_provider))
        if evidence
        else None
    )


class IgnavQuarantineFlightOfferProvider(_AdapterBase):
    """Opt-in Ignav adapter protected by a non-renewing free-call wall.

    Ignav becomes billable after its initial allowance.  Consequently, having
    an API key is not enough to enable this adapter: the operator must also
    attest that the dedicated account is still free and has no payment method,
    and explicitly release the experimental schema adapter.  Every successful
    or failed attempt is conservatively charged to a local lifetime ledger.
    """

    provider_code = IGNAV_QUARANTINE_PROVIDER_CODE
    provider_name = IGNAV_QUARANTINE_PROVIDER_NAME
    ledger_provider_code = IGNAV_QUARANTINE_PROVIDER_CODE

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: Path,
        release_verified: bool = False,
        free_account_attested: bool = False,
        lifetime_limit: int | None = IGNAV_FREE_CALL_LIMIT,
        client: Any = None,
        timeout_seconds: float = 25.0,
        now_provider: Any = None,
    ) -> None:
        super().__init__(
            usage_path=usage_path,
            client=client,
            timeout_seconds=timeout_seconds,
            now_provider=now_provider,
        )
        self._api_key = (api_key or "").strip() or None
        self._release_verified = release_verified is True
        self._free_account_attested = free_account_attested is True
        if self.strict_release_enabled:
            self.provider_code = IGNAV_VERIFIED_PROVIDER_CODE
            self.provider_name = IGNAV_VERIFIED_PROVIDER_NAME
        parsed_limit = _optional_positive_int(lifetime_limit)
        self.free_call_limit = min(
            parsed_limit or IGNAV_FREE_CALL_LIMIT,
            IGNAV_FREE_CALL_LIMIT,
        )

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    @property
    def strict_release_enabled(self) -> bool:
        return bool(
            self.configured
            and self._release_verified
            and self._free_account_attested
        )

    @property
    def environment(self) -> str:
        return "production" if self.strict_release_enabled else "disabled"

    @property
    def monthly_limit(self) -> int:
        """Compatibility alias; the underlying wall is a lifetime limit."""

        return self.free_call_limit

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
        if not self._free_account_attested:
            return self._result("budget_not_configured", observed_at)
        if not self._release_verified:
            return self._result("test_environment_rejected", observed_at)
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
    ) -> FlightOfferSearchResult:
        diagnostics = _Diagnostics(self.ledger_provider_code, self._ledger, self._provider_now)
        search_reservation = self._ledger.reserve(
            self.ledger_provider_code,
            len(_CABINS),
            hard_limit=self.free_call_limit,
            require_all=True,
        )
        if search_reservation != len(_CABINS):
            return self._result(
                "budget_exhausted",
                observed_at,
                diagnostics=diagnostics.snapshot(),
            )

        payloads: dict[Cabin, dict[str, Any]] = {}
        failed_cabin_statuses: list[SearchStatus] = []
        with ThreadPoolExecutor(
            max_workers=len(_CABINS),
            thread_name_prefix="ignav-search",
        ) as pool:
            futures = {
                pool.submit(
                    self._search_cabin,
                    origin,
                    destination,
                    departure_date,
                    cabin,
                    diagnostics,
                ): cabin
                for cabin in _CABINS
            }
            for future in as_completed(futures):
                cabin = futures[future]
                try:
                    payloads[cabin] = future.result()
                except _AdapterError as exc:
                    failed_cabin_statuses.append(exc.status)
                except Exception:
                    failed_cabin_statuses.append("provider_unavailable")
                    diagnostics.record(
                        stage="cabin_search",
                        http_status=None,
                        exception_type="PayloadError",
                    )

        candidates, invalid_rows = _select_ignav_candidates(
            payloads,
            origin,
            destination,
            departure_date,
        )
        if invalid_rows and not candidates:
            diagnostics.record(
                stage="validation",
                http_status=None,
                exception_type="PayloadError",
            )
            return self._result(
                "provider_unavailable",
                observed_at,
                searched_cabins=_CABINS,
                search_calls_used=len(_CABINS),
                diagnostics=diagnostics.snapshot(),
                search_failed_cabin_count=len(failed_cabin_statuses),
                coverage_status="provider_incomplete",
            )
        if not candidates:
            failed_count = len(failed_cabin_statuses)
            return self._result(
                (
                    _searchapi_failed_cabin_status(failed_cabin_statuses)
                    if failed_count
                    else "no_results"
                ),
                observed_at,
                searched_cabins=_CABINS,
                search_calls_used=len(_CABINS),
                diagnostics=diagnostics.snapshot(),
                search_failed_cabin_count=failed_count,
                coverage_status=("provider_incomplete" if failed_count else "complete"),
            )

        requested_candidates = len(candidates)
        pricing_reservation = self._ledger.reserve(
            self.ledger_provider_code,
            requested_candidates,
            hard_limit=self.free_call_limit,
        )
        attempted_candidates = candidates[:pricing_reservation]
        confirmed_by_position: dict[int, ConfirmedFlightOffer] = {}
        provider_failures = 0
        strict_rejections = 0
        if attempted_candidates:
            with ThreadPoolExecutor(
                max_workers=min(MAX_ALTERNATE_BOOKING_WORKERS, len(attempted_candidates)),
                thread_name_prefix="ignav-booking",
            ) as pool:
                futures = {
                    pool.submit(
                        self._booking_links,
                        candidate,
                        diagnostics,
                    ): (position, candidate)
                    for position, candidate in enumerate(attempted_candidates)
                }
                for future in as_completed(futures):
                    position, candidate = futures[future]
                    try:
                        payload, received_at, http_status = future.result()
                        offer = _parse_ignav_booking_confirmation(
                            payload,
                            candidate,
                            origin,
                            destination,
                            departure_date,
                            received_at,
                        )
                    except _AdapterError:
                        provider_failures += 1
                        continue
                    except Exception:
                        provider_failures += 1
                        diagnostics.record(
                            stage="validation",
                            http_status=None,
                            exception_type="PayloadError",
                        )
                        continue
                    if offer is None:
                        strict_rejections += 1
                        diagnostics.record(
                            stage="validation",
                            http_status=http_status,
                            exception_type="StrictCandidateRejected",
                            search_id=_ignav_payload_id(payload),
                            observed_at=received_at,
                        )
                    else:
                        confirmed_by_position[position] = offer

        verified_count = len(confirmed_by_position)
        quota_skipped = len(candidates) - len(attempted_candidates)
        coverage_status = _candidate_coverage_status(
            evaluated=True,
            provider_failed=provider_failures,
            quota_skipped=quota_skipped,
            search_failed_cabins=len(failed_cabin_statuses),
        )
        offers, deduplicated = _lowest_verified_offers(
            confirmed_by_position,
            len(attempted_candidates),
        )
        quota_limit = "lifetime" if quota_skipped else None
        common: dict[str, Any] = {
            "searched_cabins": _CABINS,
            "search_calls_used": len(_CABINS),
            "pricing_calls_used": len(attempted_candidates),
            "diagnostics": diagnostics.snapshot(),
            "eligible_candidate_count": len(candidates),
            "verification_attempted_count": len(attempted_candidates),
            "verified_candidate_count": verified_count,
            "strictly_rejected_candidate_count": strict_rejections,
            "provider_failed_candidate_count": provider_failures,
            "search_failed_cabin_count": len(failed_cabin_statuses),
            "quota_skipped_candidate_count": quota_skipped,
            "deduplicated_verified_count": deduplicated,
            "coverage_status": coverage_status,
            "quota_limit": quota_limit,
        }
        if offers:
            return self._result(
                "confirmed_offers",
                observed_at,
                offers=offers,
                **common,
            )
        if quota_skipped:
            status: SearchStatus = "budget_exhausted"
        elif provider_failures:
            status = "provider_error"
        else:
            status = "no_results"
        return self._result(status, observed_at, **common)

    def _search_cabin(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        cabin: Cabin,
        diagnostics: _Diagnostics,
    ) -> dict[str, Any]:
        payload, received_at, http_status = self._request_json(
            "POST",
            IGNAV_ONE_WAY_URL,
            stage="cabin_search",
            diagnostics=diagnostics,
            body={
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date.isoformat(),
                "adults": 1,
                "cabin_class": cabin,
                "market": "US",
                "allow_self_transfer": False,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Api-Key": str(self._api_key),
                "User-Agent": "flight-forecast-lab/0.2.0 (free-only quarantine)",
            },
        )
        if not _ignav_search_parameters_match(
            payload,
            origin=origin,
            destination=destination,
            departure_date=departure_date,
        ):
            diagnostics.record(
                stage="validation",
                http_status=http_status,
                exception_type="PayloadError",
                search_id=_ignav_payload_id(payload),
                observed_at=received_at,
            )
            raise _PayloadError("Ignav search payload did not match the request")
        return payload

    def _booking_links(
        self,
        candidate: _IgnavCandidate,
        diagnostics: _Diagnostics,
    ) -> tuple[dict[str, Any], datetime, int]:
        return self._request_json(
            "POST",
            IGNAV_BOOKING_LINKS_URL,
            stage="booking_options",
            diagnostics=diagnostics,
            body={"ignav_id": candidate.ignav_id},
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Api-Key": str(self._api_key),
                "User-Agent": "flight-forecast-lab/0.2.0 (strict booking verification)",
            },
        )


class FallbackFlightOfferProvider:
    """Run every strict provider in order and merge verified offers.

    Provider/account failures are isolated per source.  A transient source
    failure receives exactly one whole-provider retry with ``force_refresh``;
    authentication, quota, and complete no-result responses are terminal for
    that source and immediately advance the chain.  This keeps failover honest
    without spending scarce free calls retrying a known quota wall.
    """

    _PASSIVE_STATUSES = {
        "not_configured",
        "budget_not_configured",
        "test_environment_rejected",
    }
    _CONTROLLED_RETRY_STATUSES = {
        "provider_processing",
        "provider_error",
        "provider_unavailable",
    }

    def __init__(self, providers: tuple[Any, ...]) -> None:
        if not providers or any(not callable(getattr(item, "search", None)) for item in providers):
            raise ValueError("at least one valid fallback provider is required")
        self.providers = tuple(providers)

    @property
    def configured(self) -> bool:
        return any(bool(getattr(provider, "configured", False)) for provider in self.providers)

    @property
    def environment(self) -> str:
        return "production" if self.configured else "disabled"

    def search(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        *,
        fetched_at: datetime,
        force_refresh: bool = False,
    ) -> FlightOfferSearchResult:
        def run_provider(
            provider: Any,
            *,
            refresh: bool,
        ) -> FlightOfferSearchResult:
            try:
                return provider.search(
                    origin,
                    destination,
                    departure_date,
                    fetched_at=fetched_at,
                    force_refresh=refresh,
                )
            except Exception:
                return _unexpected_provider_result(provider, fetched_at)

        results: list[FlightOfferSearchResult] = []
        for provider in self.providers:
            first = run_provider(provider, refresh=force_refresh)
            if first.status in self._CONTROLLED_RETRY_STATUSES:
                second = run_provider(provider, refresh=True)
                first = _controlled_provider_retry_result(first, second)
            results.append(first)

        attempted = [
            result
            for result in results
            if result.status not in self._PASSIVE_STATUSES
        ]
        if len(attempted) == 1:
            return attempted[0]
        if len(attempted) > 1:
            return _aggregate_strict_provider_results(attempted)
        return results[-1]


def _controlled_provider_retry_result(
    first: FlightOfferSearchResult,
    second: FlightOfferSearchResult,
) -> FlightOfferSearchResult:
    """Return the retry's evidence with conservative two-attempt accounting."""

    if (
        first.provider_code,
        first.provider_name,
    ) != (
        second.provider_code,
        second.provider_name,
    ):
        return first

    marker_time = max(first.observed_at, second.observed_at)
    marker = ProviderDiagnostic(
        observed_at=marker_time,
        stage="validation",
        http_status=None,
        exception_type="ControlledProviderRetry",
        search_id=None,
    )
    diagnostics = tuple(
        [*first.diagnostics[:4], marker, *second.diagnostics[-5:]]
    )

    def maximum_optional(left: int | None, right: int | None) -> int | None:
        values = [value for value in (left, right) if value is not None]
        return max(values) if values else None

    return replace(
        second,
        observed_at=marker_time,
        calls_used=first.calls_used + second.calls_used,
        cache_hit=False,
        search_calls_used=(
            first.search_calls_used + second.search_calls_used
        ),
        pricing_calls_used=(
            first.pricing_calls_used + second.pricing_calls_used
        ),
        search_monthly_used=maximum_optional(
            first.search_monthly_used,
            second.search_monthly_used,
        ),
        pricing_monthly_used=maximum_optional(
            first.pricing_monthly_used,
            second.pricing_monthly_used,
        ),
        archive_poll_count=(
            first.archive_poll_count + second.archive_poll_count
        ),
        diagnostics=diagnostics,
    )


def _unexpected_provider_result(
    provider: Any,
    observed_at: datetime,
) -> FlightOfferSearchResult:
    """Isolate an unexpected provider exception without retaining its message."""

    try:
        candidate_code = getattr(provider, "provider_code", None)
        candidate_name = getattr(provider, "provider_name", None)
    except Exception:
        candidate_code = None
        candidate_name = None
    expected_name = _STRICT_PROVIDER_NAMES.get(candidate_code)
    if expected_name is None or candidate_name != expected_name:
        provider_code = "none"
        provider_name = "No strict fare provider"
    else:
        provider_code = candidate_code
        provider_name = expected_name
    safe_observed_at = _utc(observed_at)
    return FlightOfferSearchResult(
        offers=(),
        status="provider_unavailable",
        observed_at=safe_observed_at,
        environment="production",
        searched_cabins=(),
        calls_used=0,
        cache_hit=False,
        diagnostics=(
            ProviderDiagnostic(
                observed_at=safe_observed_at,
                stage="validation",
                http_status=None,
                exception_type="UnexpectedProviderError",
                search_id=None,
            ),
        ),
        provider_code=provider_code,
        provider_name=provider_name,
    )


def _aggregate_strict_provider_results(
    results: list[FlightOfferSearchResult],
) -> FlightOfferSearchResult:
    """Combine strict source runs while preserving their independent evidence.

    Each source has already performed its own second-stage booking verification.
    Cross-source aggregation only removes equivalent dated itinerary-and-cabin
    rows and never promotes a search hint, timetable, or failed candidate.
    """

    grouped: dict[tuple[Any, ...], tuple[int, ConfirmedFlightOffer]] = {}
    position = 0
    for result in results:
        for offer in result.offers:
            key = offer.lowest_price_group_key
            current = grouped.get(key)
            if current is None:
                grouped[key] = (position, offer)
            elif _verified_offer_preference(offer) < _verified_offer_preference(current[1]):
                grouped[key] = (current[0], offer)
            position += 1
    offers = tuple(offer for _, offer in sorted(grouped.values(), key=lambda item: item[0]))

    if offers:
        status: SearchStatus = "confirmed_offers"
    elif all(
        result.status == "no_results" and result.coverage_status == "complete"
        for result in results
    ):
        status = "no_results"
    else:
        priority: tuple[SearchStatus, ...] = (
            "provider_processing",
            "provider_error",
            "provider_unavailable",
            "authentication_failed",
            "rate_limited",
            "budget_exhausted",
            "budget_not_configured",
            "test_environment_rejected",
            "no_results",
        )
        statuses = {result.status for result in results}
        status = next(
            (candidate for candidate in priority if candidate in statuses),
            "provider_error",
        )
        if status == "no_results":
            quota_limited_results = [
                result
                for result in results
                if result.coverage_status
                in {"quota_limited", "quota_and_provider_incomplete"}
            ]
            status = "budget_exhausted" if quota_limited_results else "provider_error"

    verified_candidate_count = sum(
        result.verified_candidate_count for result in results
    )
    quota_skipped_candidate_count = sum(
        result.quota_skipped_candidate_count for result in results
    )
    provider_failed_candidate_count = sum(
        result.provider_failed_candidate_count for result in results
    )
    search_failed_cabin_count = sum(
        result.search_failed_cabin_count for result in results
    )
    # An aggregate has evaluated cross-provider coverage once two or more active
    # sources have returned terminal run results, even when neither source
    # reached candidate verification.  The individual runs retain their own
    # ``not_evaluated`` status; the aggregate must still expose that provider
    # failure and/or quota walls made the combined coverage incomplete.
    evaluated = bool(results)
    provider_run_incomplete = any(
        result.status not in {"confirmed_offers", "no_results"}
        or result.coverage_status
        in {"provider_incomplete", "quota_and_provider_incomplete"}
        for result in results
    )
    retry_quota_limited = any(result.retry_quota_limited for result in results)
    quota_limited_runs = [
        result
        for result in results
        if result.status in {"rate_limited", "budget_exhausted"}
        or result.coverage_status in {"quota_limited", "quota_and_provider_incomplete"}
    ]
    coverage_status = _candidate_coverage_status(
        evaluated=evaluated,
        provider_failed=(
            provider_failed_candidate_count
            or search_failed_cabin_count
            or int(provider_run_incomplete)
        ),
        quota_skipped=quota_skipped_candidate_count,
        retry_quota_limited=bool(retry_quota_limited or quota_limited_runs),
    )
    quota_limits = {
        result.quota_limit
        for result in quota_limited_runs
        if result.quota_limit is not None
    }
    quota_limit = None
    quota_truncated_coverage = coverage_status in {
        "quota_limited",
        "quota_and_provider_incomplete",
    }
    unevaluated_quota_wall = (
        coverage_status == "not_evaluated"
        and status in {"rate_limited", "budget_exhausted"}
    )
    if quota_truncated_coverage or unevaluated_quota_wall:
        quota_limit = (
            next(iter(quota_limits))
            if len(quota_limits) == 1
            and all(result.quota_limit is not None for result in quota_limited_runs)
            else "provider_specific"
        )

    searched_cabins = tuple(
        cabin
        for cabin in _CABINS
        if any(cabin in result.searched_cabins for result in results)
    )
    return FlightOfferSearchResult(
        offers=offers,
        status=status,
        observed_at=max(result.observed_at for result in results),
        environment="production",
        searched_cabins=searched_cabins,
        calls_used=sum(result.calls_used for result in results),
        cache_hit=all(result.cache_hit for result in results),
        search_calls_used=sum(result.search_calls_used for result in results),
        pricing_calls_used=sum(result.pricing_calls_used for result in results),
        archive_poll_count=sum(result.archive_poll_count for result in results),
        eligible_candidate_count=sum(
            result.eligible_candidate_count for result in results
        ),
        verification_attempted_count=sum(
            result.verification_attempted_count for result in results
        ),
        verified_candidate_count=verified_candidate_count,
        strictly_rejected_candidate_count=sum(
            result.strictly_rejected_candidate_count for result in results
        ),
        provider_failed_candidate_count=provider_failed_candidate_count,
        search_failed_cabin_count=search_failed_cabin_count,
        quota_skipped_candidate_count=quota_skipped_candidate_count,
        deduplicated_verified_count=verified_candidate_count - len(offers),
        coverage_status=coverage_status,
        quota_limit=quota_limit,
        retry_quota_limited=retry_quota_limited,
        provider_code=AGGREGATE_PROVIDER_CODE,
        provider_name=AGGREGATE_PROVIDER_NAME,
        provider_runs=tuple(results),
    )


def _ignav_payload_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("request_id", "search_id"):
        candidate = _clean_id(payload.get(key))
        if candidate is not None:
            return candidate
    return None


def _ignav_search_parameters_match(
    payload: dict[str, Any],
    *,
    origin: str,
    destination: str,
    departure_date: date,
) -> bool:
    itineraries = payload.get("itineraries")
    return bool(
        _iata(payload.get("origin")) == origin
        and _iata(payload.get("destination")) == destination
        and str(payload.get("departure_date", "")) == departure_date.isoformat()
        and isinstance(itineraries, list)
        and len(itineraries) <= IGNAV_MAX_ITINERARIES_PER_CABIN
    )


def _select_ignav_candidates(
    payloads: dict[Cabin, dict[str, Any]],
    origin: str,
    destination: str,
    departure_date: date,
) -> tuple[tuple[_IgnavCandidate, ...], int]:
    by_cabin: dict[Cabin, list[_IgnavCandidate]] = {cabin: [] for cabin in _CABINS}
    invalid_rows = 0
    seen_ids: dict[str, tuple[Any, ...]] = {}
    conflicting_ids: set[str] = set()
    for cabin in _CABINS:
        rows = payloads.get(cabin, {}).get("itineraries")
        if not isinstance(rows, list):
            invalid_rows += 1
            continue
        for row in rows:
            candidate = _parse_ignav_candidate(
                row,
                cabin,
                origin,
                destination,
                departure_date,
            )
            if candidate is None:
                invalid_rows += 1
                continue
            if candidate.ignav_id in conflicting_ids:
                invalid_rows += 1
                continue
            existing_identity = seen_ids.get(candidate.ignav_id)
            if existing_identity is not None and existing_identity != candidate.identity:
                conflicting_ids.add(candidate.ignav_id)
                invalid_rows += 1
                by_cabin = {
                    item_cabin: [
                        item
                        for item in items
                        if item.ignav_id != candidate.ignav_id
                    ]
                    for item_cabin, items in by_cabin.items()
                }
                continue
            if existing_identity is not None:
                invalid_rows += 1
                continue
            seen_ids[candidate.ignav_id] = candidate.identity
            by_cabin[cabin].append(candidate)

    return _direct_first_cabin_round_robin(by_cabin), invalid_rows


def _parse_ignav_candidate(
    row: Any,
    cabin: Cabin,
    origin: str,
    destination: str,
    departure_date: date,
) -> _IgnavCandidate | None:
    if not isinstance(row, dict) or row.get("cabin_class") != cabin:
        return None
    ignav_id = _clean_id(row.get("ignav_id"))
    price = row.get("price")
    outbound = row.get("outbound")
    if not isinstance(price, dict) or not isinstance(outbound, dict):
        return None
    amount = _verified_usd_amount(price)
    raw_segments = outbound.get("segments")
    if (
        ignav_id is None
        or amount is None
        or not isinstance(raw_segments, list)
        or _positive_duration(outbound.get("duration_minutes")) is None
    ):
        return None
    parsed = _parse_ignav_segments(raw_segments, cabin)
    if parsed is None:
        return None
    segments, elapsed_minutes = parsed
    if (
        not _segments_match_request(segments, origin, destination, departure_date)
        or _positive_duration(outbound.get("duration_minutes")) != elapsed_minutes
    ):
        return None
    bags = row.get("bags")
    checked = (
        _optional_nonnegative_int(bags.get("checked"))
        if isinstance(bags, dict)
        else None
    )
    if checked is not None and checked > 9:
        checked = None
    return _IgnavCandidate(
        ignav_id=ignav_id,
        cabin=cabin,
        search_price_usd=amount,
        segments=segments,
        airline_name=_short_text(outbound.get("carrier"), max_length=160)
        or segments[0].marketing_airline_code,
        checked_bags_quantity=checked,
    )


def _parse_ignav_segments(
    rows: list[Any],
    cabin: Cabin,
) -> tuple[tuple[FlightOfferSegment, ...], int] | None:
    if not 1 <= len(rows) <= MAX_STRICT_ITINERARY_SEGMENTS:
        return None
    segments: list[FlightOfferSegment] = []
    first_departure_utc: datetime | None = None
    previous_arrival_utc: datetime | None = None
    last_arrival_utc: datetime | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        origin = _iata(row.get("departure_airport"))
        destination = _iata(row.get("arrival_airport"))
        marketing_code = _airline_code(row.get("marketing_carrier_code"))
        flight_number = str(row.get("flight_number") or "").strip().upper()
        departure_at = _naive_iso_datetime(row.get("departure_time_local"))
        arrival_at = _naive_iso_datetime(row.get("arrival_time_local"))
        departure_utc = _aware_utc_datetime(row.get("departure_time_utc"))
        arrival_utc = _aware_utc_datetime(row.get("arrival_time_utc"))
        duration = _positive_duration(row.get("duration_minutes"))
        if (
            origin is None
            or destination is None
            or origin == destination
            or marketing_code is None
            or re.fullmatch(r"[A-Z0-9]{1,8}", flight_number) is None
            or departure_at is None
            or arrival_at is None
            or departure_utc is None
            or arrival_utc is None
            or duration is None
            or not _local_matches_utc(
                departure_at,
                row.get("departure_timezone"),
                departure_utc,
            )
            or not _local_matches_utc(
                arrival_at,
                row.get("arrival_timezone"),
                arrival_utc,
            )
            or arrival_utc <= departure_utc
            or int((arrival_utc - departure_utc).total_seconds()) != duration * 60
        ):
            return None
        if previous_arrival_utc is not None:
            if segments[-1].destination != origin or departure_utc <= previous_arrival_utc:
                return None
        if first_departure_utc is None:
            first_departure_utc = departure_utc
        previous_arrival_utc = arrival_utc
        last_arrival_utc = arrival_utc
        segments.append(
            FlightOfferSegment(
                segment_id=(
                    f"{marketing_code}{flight_number}-{departure_at:%Y%m%d%H%M}-{index}"
                ),
                origin=origin,
                destination=destination,
                departure_at=departure_at,
                arrival_at=arrival_at,
                marketing_airline_code=marketing_code,
                operating_airline_code=None,
                flight_number=flight_number,
                departure_terminal=None,
                arrival_terminal=None,
                aircraft_icao=None,
                cabin=cabin,
                booking_class=None,
                fare_basis=None,
                fare_brand=None,
                checked_bags_quantity=None,
                checked_bags_weight=None,
                checked_bags_weight_unit=None,
            )
        )
    if first_departure_utc is None or last_arrival_utc is None:
        return None
    elapsed = (last_arrival_utc - first_departure_utc).total_seconds()
    if elapsed <= 0 or elapsed % 60:
        return None
    return tuple(segments), int(elapsed // 60)


def _verified_usd_amount(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    amount = _finite_amount(value.get("amount"))
    return (
        amount
        if amount is not None
        and amount > 0
        and str(value.get("currency", "")).strip().upper() == "USD"
        and _normalized_phrase(value.get("status")) == "verified"
        else None
    )


def _parse_ignav_booking_confirmation(
    payload: dict[str, Any],
    candidate: _IgnavCandidate,
    origin: str,
    destination: str,
    departure_date: date,
    received_at: datetime,
) -> ConfirmedFlightOffer | None:
    itinerary = payload.get("itinerary")
    if not isinstance(itinerary, dict) or itinerary.get("cabin_class") != candidate.cabin:
        return None
    outbound = itinerary.get("outbound")
    price = itinerary.get("price")
    if not isinstance(outbound, dict) or _verified_usd_amount(price) is None:
        return None
    raw_segments = outbound.get("segments")
    if not isinstance(raw_segments, list):
        return None
    parsed = _parse_ignav_segments(raw_segments, candidate.cabin)
    if parsed is None:
        return None
    segments, elapsed_minutes = parsed
    if (
        not _segments_match_request(segments, origin, destination, departure_date)
        or _segment_identity(segments) != candidate.identity
        or _positive_duration(outbound.get("duration_minutes")) != elapsed_minutes
    ):
        return None
    evidence = _ignav_booking_evidence(payload)
    if evidence is None:
        return None
    if evidence.fare_brand is not None:
        segments = tuple(replace(segment, fare_brand=evidence.fare_brand) for segment in segments)
    checked = candidate.checked_bags_quantity
    if checked is not None:
        segments = tuple(
            replace(segment, checked_bags_quantity=checked)
            for segment in segments
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "id": _opaque_token_digest(candidate.ignav_id),
                "seller": evidence.booking_provider,
                "url": evidence.booking_url,
                "price": evidence.amount_usd,
                "segments": _segment_identity(segments),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return ConfirmedFlightOffer(
        provider_offer_id=f"ignav-{digest}",
        validating_airline_code=segments[0].marketing_airline_code,
        airline_name=_short_text(outbound.get("carrier"), max_length=160)
        or candidate.airline_name,
        cabin=candidate.cabin,
        total_amount_usd=evidence.amount_usd,
        base_amount_usd=None,
        last_ticketing_date=None,
        number_of_bookable_seats=None,
        seat_count_capped=False,
        verified_at=_utc(received_at),
        provider_cache_hit=False,
        provider_cache_age_seconds=0,
        segments=segments,
        refundable_fare=None,
        no_penalty_fare=None,
        no_restriction_fare=None,
        booking_url=evidence.booking_url,
        booking_url_kind="direct_get",
        booking_provider=evidence.booking_provider,
        provider_code=IGNAV_VERIFIED_PROVIDER_CODE,
        provider_name=IGNAV_VERIFIED_PROVIDER_NAME,
        environment="production",
    )


def _ignav_booking_evidence(payload: dict[str, Any]) -> _BookingEvidence | None:
    raw_options = payload.get("booking_options")
    if not isinstance(raw_options, list):
        return None
    evidence: list[_BookingEvidence] = []
    for option in raw_options:
        if not isinstance(option, dict) or option.get("legs") != ["outbound"]:
            continue
        links = option.get("links")
        if not isinstance(links, list):
            continue
        for link in links:
            if not isinstance(link, dict):
                continue
            provider = _short_text(link.get("provider_name"), max_length=160)
            provider_type = _normalized_phrase(link.get("provider_type"))
            amount = _verified_usd_amount(link.get("price"))
            booking_url = _safe_public_https_url(link.get("url"))
            if (
                provider is None
                or provider_type not in {"airline", "third party"}
                or amount is None
                or booking_url is None
            ):
                continue
            evidence.append(
                _BookingEvidence(
                    amount_usd=amount,
                    booking_url=booking_url,
                    booking_url_kind="direct_get",
                    booking_provider=provider,
                    fare_brand=_short_text(link.get("fare_name"), max_length=80),
                )
            )
    return (
        min(evidence, key=lambda item: (item.amount_usd, item.booking_provider.casefold()))
        if evidence
        else None
    )
