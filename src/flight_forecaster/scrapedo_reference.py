"""Quota-safe Scrape.do Google Flights coverage snapshots.

This module is intentionally reference-only.  Scrape.do's Google Flights
plugin returns listing data and opaque booking tokens, but it does not expose
the booking expansion needed by this project's strict purchase-path check.
The adapter therefore retains only aggregate, sanitized coverage facts and has
no offer-construction API.

Every flight-plugin attempt reserves ten credits in a durable calendar-month
ledger before it is sent.  The local ceiling is clamped to the documented
1,000-credit free allowance, so configuration can never opt this adapter into
paid usage.  Provider-reported request costs above the reservation are added to
the ledger, saturating at the local ceiling.  Cached snapshots do not reserve
additional credits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib import parse, request

from flight_forecaster.quota_status import QuotaLedgerSnapshot

SCRAPE_DO_FLIGHTS_URL = "https://api.scrape.do/plugin/google/flights"
SCRAPE_DO_PROVIDER_CODE = "scrape_do_google_flights_reference"
SCRAPE_DO_PROVIDER_NAME = "Scrape.do Google Flights"
SCRAPE_DO_CREDITS_PER_CALL = 10
SCRAPE_DO_FREE_MONTHLY_CREDITS = 1_000
SCRAPE_DO_CACHE_TTL_SECONDS = 30 * 60
SCRAPE_DO_REQUEST_TIMEOUT_SECONDS = 25.0
SCRAPE_DO_MAX_RESPONSE_BYTES = 5_000_000
SCRAPE_DO_MAX_CACHE_ROWS = 256
SCRAPE_DO_MAX_ATTEMPTS = 2

_IATA_PATTERN = re.compile(r"^[A-Z]{3}$")
_SAFE_EXCEPTION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_STATUSES = {
    "available",
    "no_results",
    "not_configured",
    "quota_exhausted",
    "authentication_failed",
    "rate_limited",
    "provider_error",
    "provider_unavailable",
}

ScrapeDoReferenceStatus = Literal[
    "available",
    "no_results",
    "not_configured",
    "quota_exhausted",
    "authentication_failed",
    "rate_limited",
    "provider_error",
    "provider_unavailable",
]


@dataclass(frozen=True, slots=True)
class ScrapeDoReferenceResult:
    """Sanitized aggregate result; no token, URL, or itinerary is retained."""

    status: ScrapeDoReferenceStatus
    observed_at: datetime
    candidate_count: int = 0
    direct_candidate_count: int = 0
    lowest_price_usd: float | None = None
    price_level: Literal["low", "typical", "high"] | None = None
    typical_price_low_usd: float | None = None
    typical_price_high_usd: float | None = None
    cache_hit: bool = False
    credits_reserved: int = 0
    monthly_credits_used: int = 0
    monthly_credit_limit: int = SCRAPE_DO_FREE_MONTHLY_CREDITS
    http_status: int | None = None
    exception_type: str | None = None
    provider_reported_request_cost: int | None = None
    provider_reported_remaining_credits: int | None = None

    def __post_init__(self) -> None:
        observed_at = self.observed_at
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("Scrape.do reference observation must include a timezone")
        object.__setattr__(self, "observed_at", observed_at.astimezone(UTC))
        if self.status not in _STATUSES:
            raise ValueError("Scrape.do reference status is invalid")
        if not 0 <= self.candidate_count <= 30:
            raise ValueError("Scrape.do reference candidate count is invalid")
        if not 0 <= self.direct_candidate_count <= self.candidate_count:
            raise ValueError("Scrape.do direct candidate count is invalid")
        if self.status == "available" and self.candidate_count == 0:
            raise ValueError("available Scrape.do reference requires candidates")
        if self.status != "available" and self.candidate_count:
            raise ValueError("unavailable Scrape.do reference cannot retain candidates")
        if self.lowest_price_usd is not None and not _positive_amount(
            self.lowest_price_usd
        ):
            raise ValueError("Scrape.do reference price is invalid")
        if self.status != "available" and self.lowest_price_usd is not None:
            raise ValueError("non-available Scrape.do reference cannot retain a price")
        if (self.typical_price_low_usd is None) != (
            self.typical_price_high_usd is None
        ):
            raise ValueError("Scrape.do typical price range must be complete")
        if self.typical_price_low_usd is not None:
            if not _positive_amount(self.typical_price_low_usd) or not _positive_amount(
                self.typical_price_high_usd
            ):
                raise ValueError("Scrape.do typical price range is invalid")
            assert self.typical_price_high_usd is not None
            if self.typical_price_high_usd < self.typical_price_low_usd:
                raise ValueError("Scrape.do typical price range is reversed")
        if self.status != "available" and (
            self.price_level is not None or self.typical_price_low_usd is not None
        ):
            raise ValueError("non-available reference cannot retain price insights")
        if self.credits_reserved not in {
            0,
            SCRAPE_DO_CREDITS_PER_CALL,
            SCRAPE_DO_CREDITS_PER_CALL * SCRAPE_DO_MAX_ATTEMPTS,
        }:
            raise ValueError("Scrape.do reserved credits are invalid")
        if not 0 <= self.monthly_credit_limit <= SCRAPE_DO_FREE_MONTHLY_CREDITS:
            raise ValueError("Scrape.do local credit limit exceeds the free allowance")
        if not 0 <= self.monthly_credits_used <= self.monthly_credit_limit:
            raise ValueError("Scrape.do local credit usage is invalid")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ValueError("Scrape.do HTTP status is invalid")
        if self.exception_type is not None and not _SAFE_EXCEPTION_PATTERN.fullmatch(
            self.exception_type
        ):
            raise ValueError("Scrape.do exception type is invalid")
        if self.provider_reported_request_cost is not None and (
            self.provider_reported_request_cost < 0
        ):
            raise ValueError("Scrape.do reported request cost is invalid")
        if self.provider_reported_remaining_credits is not None and not (
            0
            <= self.provider_reported_remaining_credits
            <= SCRAPE_DO_FREE_MONTHLY_CREDITS
        ):
            raise ValueError("Scrape.do reported remaining credits are invalid")


@dataclass(frozen=True, slots=True)
class _Reservation:
    allowed: bool
    used_after: int
    period_key: str


class _ScrapeDoLedger:
    """Atomic monthly reservations plus a persistent sanitized snapshot cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scrapedo_reference_usage (
                    period_key TEXT PRIMARY KEY,
                    reserved_credits INTEGER NOT NULL
                        CHECK (reserved_credits >= 0 AND reserved_credits <= 1000)
                );
                CREATE TABLE IF NOT EXISTS scrapedo_reference_cache (
                    cache_key TEXT PRIMARY KEY,
                    observed_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    sanitized_json TEXT NOT NULL
                );
                """
            )

    def snapshot(self, observed_at: datetime) -> int:
        key = _period_key(observed_at)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT reserved_credits FROM scrapedo_reference_usage WHERE period_key = ?",
                (key,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def reserve(self, observed_at: datetime, *, hard_limit: int) -> _Reservation:
        limit = _clamp_credit_limit(hard_limit)
        key = _period_key(observed_at)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT reserved_credits FROM scrapedo_reference_usage WHERE period_key = ?",
                (key,),
            ).fetchone()
            used = int(row[0]) if row is not None else 0
            allowed = used + SCRAPE_DO_CREDITS_PER_CALL <= limit
            used_after = used + SCRAPE_DO_CREDITS_PER_CALL if allowed else used
            if row is None:
                connection.execute(
                    "INSERT INTO scrapedo_reference_usage(period_key, reserved_credits) "
                    "VALUES (?, ?)",
                    (key, used_after),
                )
            elif allowed:
                connection.execute(
                    "UPDATE scrapedo_reference_usage SET reserved_credits = ? "
                    "WHERE period_key = ?",
                    (used_after, key),
                )
            connection.execute(
                """
                DELETE FROM scrapedo_reference_usage
                WHERE period_key NOT IN (
                    SELECT period_key FROM scrapedo_reference_usage
                    ORDER BY period_key DESC LIMIT 24
                )
                """
            )
            connection.commit()
        return _Reservation(allowed=allowed, used_after=used_after, period_key=key)

    def reconcile_reported_cost(
        self,
        observed_at: datetime,
        *,
        reported_cost: int,
        hard_limit: int,
    ) -> int:
        """Atomically add any cost above the per-attempt reservation.

        Reservations are never refunded when the provider reports a lower
        cost.  If the extra charge cannot fit below the configured hard limit,
        the ledger is saturated so no later attempt can underestimate usage.
        """

        limit = _clamp_credit_limit(hard_limit)
        if reported_cost <= SCRAPE_DO_CREDITS_PER_CALL:
            return min(self.snapshot(observed_at), limit)
        extra = reported_cost - SCRAPE_DO_CREDITS_PER_CALL
        key = _period_key(observed_at)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT reserved_credits FROM scrapedo_reference_usage "
                "WHERE period_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                # This method is called only after reserve().  A missing row is
                # therefore inconsistent, so fail closed at the local ceiling.
                used_after = limit
                connection.execute(
                    "INSERT INTO scrapedo_reference_usage(period_key, reserved_credits) "
                    "VALUES (?, ?)",
                    (key, used_after),
                )
            else:
                used = int(row[0])
                used_after = min(limit, used + extra)
                connection.execute(
                    "UPDATE scrapedo_reference_usage SET reserved_credits = ? "
                    "WHERE period_key = ?",
                    (used_after, key),
                )
            connection.commit()
        return used_after

    def cached(
        self,
        cache_key: str,
        *,
        now: datetime,
        monthly_credit_limit: int,
    ) -> ScrapeDoReferenceResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT expires_at, sanitized_json FROM scrapedo_reference_cache "
                "WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            try:
                expires_at = datetime.fromisoformat(str(row[0])).astimezone(UTC)
            except (TypeError, ValueError):
                connection.execute(
                    "DELETE FROM scrapedo_reference_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None
            if expires_at <= _utc(now):
                connection.execute(
                    "DELETE FROM scrapedo_reference_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None
            try:
                payload = json.loads(str(row[1]))
                payload["observed_at"] = datetime.fromisoformat(payload["observed_at"])
                payload["monthly_credits_used"] = self.snapshot(now)
                payload["monthly_credit_limit"] = monthly_credit_limit
                payload["credits_reserved"] = 0
                payload["cache_hit"] = True
                return ScrapeDoReferenceResult(**payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                connection.execute(
                    "DELETE FROM scrapedo_reference_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None

    def store(
        self,
        cache_key: str,
        result: ScrapeDoReferenceResult,
        *,
        expires_at: datetime,
    ) -> None:
        if result.status not in {"available", "no_results"}:
            return
        payload = asdict(
            replace(
                result,
                cache_hit=False,
                credits_reserved=0,
                http_status=None,
                exception_type=None,
            )
        )
        payload["observed_at"] = result.observed_at.isoformat()
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO scrapedo_reference_cache(
                    cache_key, observed_at, expires_at, sanitized_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    expires_at = excluded.expires_at,
                    sanitized_json = excluded.sanitized_json
                """,
                (
                    cache_key,
                    result.observed_at.isoformat(),
                    _utc(expires_at).isoformat(),
                    serialized,
                ),
            )
            connection.execute(
                "DELETE FROM scrapedo_reference_cache WHERE expires_at <= ?",
                (result.observed_at.isoformat(),),
            )
            connection.execute(
                """
                DELETE FROM scrapedo_reference_cache
                WHERE cache_key NOT IN (
                    SELECT cache_key FROM scrapedo_reference_cache
                    ORDER BY observed_at DESC LIMIT ?
                )
                """,
                (SCRAPE_DO_MAX_CACHE_ROWS,),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


def read_scrapedo_quota_snapshot(
    path: str | Path,
    *,
    hard_limit: int,
    now: datetime,
) -> QuotaLedgerSnapshot:
    """Read the current UTC-month credit counter without provider access."""

    ledger_path = Path(path)
    if not ledger_path.is_file():
        return QuotaLedgerSnapshot.unavailable()
    limit = _clamp_credit_limit(hard_limit)
    if limit < 1:
        return QuotaLedgerSnapshot.unavailable()
    observed = _utc(now)
    period_key = _period_key(observed)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(ledger_path), timeout=1.0)
        connection.execute("PRAGMA query_only = ON")
        row = connection.execute(
            """
            SELECT reserved_credits FROM scrapedo_reference_usage
            WHERE period_key = ?
            """,
            (period_key,),
        ).fetchone()
        raw_used = int(row[0]) if row is not None else 0
        if raw_used < 0:
            return QuotaLedgerSnapshot.unavailable()
        used = min(raw_used, limit)
        reset_at = (
            datetime(observed.year + 1, 1, 1, tzinfo=UTC)
            if observed.month == 12
            else datetime(observed.year, observed.month + 1, 1, tzinfo=UTC)
        )
        return QuotaLedgerSnapshot(
            available=True,
            used=used,
            limit=limit,
            remaining=max(0, limit - used),
            period_key=period_key,
            data_basis="local_hard_limit",
            observed_at=observed,
            reset_at=reset_at,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return QuotaLedgerSnapshot.unavailable()
    finally:
        if connection is not None:
            connection.close()


class _UrllibResponse:
    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class _UrllibClient:
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> _UrllibResponse:
        target = f"{url}?{parse.urlencode(params)}"
        outbound = request.Request(
            target,
            headers={"Accept": "application/json", "User-Agent": "flight-forecast-lab/0.2"},
            method="GET",
        )
        try:
            with request.urlopen(outbound, timeout=timeout) as response:  # noqa: S310
                content = response.read(SCRAPE_DO_MAX_RESPONSE_BYTES + 1)
                return _UrllibResponse(
                    int(response.status),
                    content,
                    dict(response.headers.items()),
                )
        except Exception as exc:
            status = getattr(exc, "code", None)
            if isinstance(status, int):
                content = getattr(exc, "read", lambda *_args: b"")(
                    SCRAPE_DO_MAX_RESPONSE_BYTES + 1
                )
                raw_headers = getattr(exc, "headers", None)
                headers = (
                    dict(raw_headers.items())
                    if raw_headers is not None and hasattr(raw_headers, "items")
                    else {}
                )
                return _UrllibResponse(status, content, headers)
            raise


class ScrapeDoGoogleFlightsReferenceProvider:
    """Fetch one economy coverage snapshot without producing bookable offers."""

    def __init__(
        self,
        api_token: str | None,
        *,
        usage_path: str | Path,
        monthly_credit_limit: int = SCRAPE_DO_FREE_MONTHLY_CREDITS,
        client: Any | None = None,
        cache_ttl_seconds: int = SCRAPE_DO_CACHE_TTL_SECONDS,
    ) -> None:
        self._api_token = api_token.strip() if api_token and api_token.strip() else None
        self.monthly_credit_limit = _clamp_credit_limit(monthly_credit_limit)
        self._client = client or _UrllibClient()
        self._ledger = _ScrapeDoLedger(usage_path)
        self._cache_ttl_seconds = max(0, min(int(cache_ttl_seconds), 6 * 60 * 60))

    @property
    def configured(self) -> bool:
        return self._api_token is not None and self.monthly_credit_limit > 0

    @property
    def credential_present(self) -> bool:
        """Report presence only; never expose or derive the credential value."""

        return self._api_token is not None

    def credits_used(self, observed_at: datetime) -> int:
        return min(self._ledger.snapshot(observed_at), self.monthly_credit_limit)

    def snapshot(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        *,
        fetched_at: datetime,
    ) -> ScrapeDoReferenceResult:
        observed_at = _utc(fetched_at)
        origin_code = _iata(origin)
        destination_code = _iata(destination)
        if origin_code == destination_code:
            raise ValueError("Scrape.do reference origin and destination must differ")
        cache_key = _cache_key(origin_code, destination_code, departure_date)
        cached = self._ledger.cached(
            cache_key,
            now=observed_at,
            monthly_credit_limit=self.monthly_credit_limit,
        )
        if cached is not None:
            return cached

        used = self.credits_used(observed_at)
        if not self.configured:
            status: ScrapeDoReferenceStatus = (
                "quota_exhausted" if self._api_token and self.monthly_credit_limit == 0
                else "not_configured"
            )
            return self._empty_result(status, observed_at, used=used)
        if used + SCRAPE_DO_CREDITS_PER_CALL > self.monthly_credit_limit:
            return self._empty_result(
                "quota_exhausted",
                observed_at,
                used=used,
                exception_type="LocalFreeQuotaExhausted",
            )

        params = {
            "token": self._api_token,
            "departure_id": origin_code,
            "arrival_id": destination_code,
            "outbound_date": departure_date.isoformat(),
            "type": 2,
            "adults": 1,
            "travel_class": 1,
            "sort_by": 2,
            "currency": "USD",
            "hl": "en",
            "gl": "us",
        }
        credits_reserved = 0
        response: Any | None = None
        http_status: int | None = None
        reported_cost: int | None = None
        reported_remaining: int | None = None
        for attempt in range(SCRAPE_DO_MAX_ATTEMPTS):
            reservation = self._ledger.reserve(
                observed_at,
                hard_limit=self.monthly_credit_limit,
            )
            if not reservation.allowed:
                return self._empty_result(
                    "quota_exhausted",
                    observed_at,
                    used=reservation.used_after,
                    credits_reserved=credits_reserved,
                    exception_type="LocalFreeQuotaExhausted",
                )
            used = reservation.used_after
            credits_reserved += SCRAPE_DO_CREDITS_PER_CALL
            try:
                response = self._client.get(
                    SCRAPE_DO_FLIGHTS_URL,
                    params=params,
                    timeout=SCRAPE_DO_REQUEST_TIMEOUT_SECONDS,
                )
            except Exception:
                if attempt + 1 < SCRAPE_DO_MAX_ATTEMPTS:
                    continue
                return self._empty_result(
                    "provider_error",
                    observed_at,
                    used=used,
                    credits_reserved=credits_reserved,
                    exception_type="TransportError",
                )

            http_status = _status_code(response)
            reported_cost = _header_nonnegative_int(
                response,
                "Scrape.do-Request-Cost",
            )
            reported_remaining = _header_nonnegative_int(
                response,
                "Scrape.do-Remaining-Credits",
            )
            if reported_cost is not None:
                used = self._ledger.reconcile_reported_cost(
                    observed_at,
                    reported_cost=reported_cost,
                    hard_limit=self.monthly_credit_limit,
                )
            if (
                reported_cost is not None
                and reported_cost > SCRAPE_DO_CREDITS_PER_CALL
            ) or (
                reported_remaining is not None
                and reported_remaining > SCRAPE_DO_FREE_MONTHLY_CREDITS
            ):
                return self._empty_result(
                    "provider_unavailable",
                    observed_at,
                    used=used,
                    credits_reserved=credits_reserved,
                    http_status=http_status or None,
                    exception_type="UnsafeProviderQuotaReport",
                    provider_reported_request_cost=reported_cost,
                    provider_reported_remaining_credits=reported_remaining,
                )
            if http_status == 200:
                break
            if attempt + 1 < SCRAPE_DO_MAX_ATTEMPTS and _transient_status(http_status):
                continue
            status, exception_type = _error_status(http_status)
            return self._empty_result(
                status,
                observed_at,
                used=used,
                credits_reserved=credits_reserved,
                http_status=http_status,
                exception_type=exception_type,
                provider_reported_request_cost=reported_cost,
                provider_reported_remaining_credits=reported_remaining,
            )

        if response is None or http_status != 200:
            return self._empty_result(
                "provider_unavailable",
                observed_at,
                used=used,
                credits_reserved=credits_reserved,
                exception_type="RetryBoundaryError",
            )
        try:
            content = getattr(response, "content", b"")
            if isinstance(content, (bytes, bytearray)) and len(content) > (
                SCRAPE_DO_MAX_RESPONSE_BYTES
            ):
                raise ValueError("oversized")
            payload = response.json()
            parsed = _parse_snapshot(
                payload,
                origin=origin_code,
                destination=destination_code,
                departure_date=departure_date,
            )
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return self._empty_result(
                "provider_unavailable",
                observed_at,
                used=used,
                credits_reserved=credits_reserved,
                http_status=200,
                exception_type="PayloadError",
                provider_reported_request_cost=reported_cost,
                provider_reported_remaining_credits=reported_remaining,
            )

        result = ScrapeDoReferenceResult(
            status="available" if parsed["candidate_count"] else "no_results",
            observed_at=observed_at,
            candidate_count=parsed["candidate_count"],
            direct_candidate_count=parsed["direct_candidate_count"],
            lowest_price_usd=parsed["lowest_price_usd"],
            price_level=parsed["price_level"],
            typical_price_low_usd=parsed["typical_price_low_usd"],
            typical_price_high_usd=parsed["typical_price_high_usd"],
            credits_reserved=credits_reserved,
            monthly_credits_used=used,
            monthly_credit_limit=self.monthly_credit_limit,
            http_status=200,
            provider_reported_request_cost=reported_cost,
            provider_reported_remaining_credits=reported_remaining,
        )
        self._ledger.store(
            cache_key,
            result,
            expires_at=observed_at + timedelta(seconds=self._cache_ttl_seconds),
        )
        return result

    def _empty_result(
        self,
        status: ScrapeDoReferenceStatus,
        observed_at: datetime,
        *,
        used: int,
        credits_reserved: int = 0,
        http_status: int | None = None,
        exception_type: str | None = None,
        provider_reported_request_cost: int | None = None,
        provider_reported_remaining_credits: int | None = None,
    ) -> ScrapeDoReferenceResult:
        return ScrapeDoReferenceResult(
            status=status,
            observed_at=observed_at,
            credits_reserved=credits_reserved,
            monthly_credits_used=min(max(used, 0), self.monthly_credit_limit),
            monthly_credit_limit=self.monthly_credit_limit,
            http_status=http_status,
            exception_type=exception_type,
            provider_reported_request_cost=provider_reported_request_cost,
            provider_reported_remaining_credits=provider_reported_remaining_credits,
        )


def scrapedo_reference_provider_from_env(
    usage_path: str | Path | None = None,
) -> ScrapeDoGoogleFlightsReferenceProvider:
    """Build the reference adapter from aliases without exposing a credential."""

    token = next(
        (
            value.strip()
            for name in (
                "SCRAPE_DO_API_TOKEN",
                "SCRAPEDO_API_TOKEN",
                "SCRAPE_DO_TOKEN",
                "SCRAPE_DO_API_KEY",
                "SCRAPEDO_API_KEY",
            )
            if (value := os.getenv(name, "")).strip()
        ),
        None,
    )
    raw_limit = next(
        (
            os.getenv(name, "").strip()
            for name in (
                "SCRAPE_DO_MONTHLY_CREDIT_LIMIT",
                "SCRAPEDO_MONTHLY_CREDIT_LIMIT",
            )
            if os.getenv(name, "").strip()
        ),
        None,
    )
    if raw_limit is None:
        limit = SCRAPE_DO_FREE_MONTHLY_CREDITS
    else:
        try:
            limit = int(raw_limit)
        except ValueError:
            limit = 0
    path = Path(
        os.getenv(
            "SCRAPE_DO_USAGE_DB",
            str(usage_path or Path("runtime") / "scrapedo-reference-usage.sqlite3"),
        )
    )
    return ScrapeDoGoogleFlightsReferenceProvider(
        token,
        usage_path=path,
        monthly_credit_limit=limit,
    )


def _parse_snapshot(
    payload: Any,
    *,
    origin: str,
    destination: str,
    departure_date: date,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("invalid payload")
    parameters = payload.get("search_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("missing request echo")
    if (
        str(parameters.get("departure_id", "")).strip().upper() != origin
        or str(parameters.get("arrival_id", "")).strip().upper() != destination
        or str(parameters.get("outbound_date", "")).strip() != departure_date.isoformat()
        or str(parameters.get("currency", "")).strip().upper() != "USD"
    ):
        raise ValueError("request echo mismatch")

    candidates: list[tuple[float, bool]] = []
    for key in ("best_flights", "other_flights"):
        rows = payload.get(key, [])
        if not isinstance(rows, list):
            raise ValueError("invalid flight list")
        for row in rows:
            parsed = _reference_candidate(
                row,
                origin=origin,
                destination=destination,
                departure_date=departure_date,
            )
            if parsed is not None:
                candidates.append(parsed)
    candidates = candidates[:30]
    prices = [price for price, _direct in candidates]

    price_level: Literal["low", "typical", "high"] | None = None
    typical_low: float | None = None
    typical_high: float | None = None
    insights = payload.get("price_insights")
    if insights is not None and not isinstance(insights, dict):
        raise ValueError("invalid price insights")
    if isinstance(insights, dict):
        raw_level = str(insights.get("price_level", "")).strip().lower()
        if raw_level in {"low", "typical", "high"}:
            price_level = raw_level  # type: ignore[assignment]
        raw_range = insights.get("typical_price_range")
        if isinstance(raw_range, list) and len(raw_range) == 2:
            low = _amount(raw_range[0])
            high = _amount(raw_range[1])
            if low is not None and high is not None and high >= low:
                typical_low, typical_high = low, high

    if not candidates:
        price_level = None
        typical_low = None
        typical_high = None
    return {
        "candidate_count": len(candidates),
        "direct_candidate_count": sum(direct for _price, direct in candidates),
        "lowest_price_usd": min(prices) if prices else None,
        "price_level": price_level,
        "typical_price_low_usd": typical_low,
        "typical_price_high_usd": typical_high,
    }


def _reference_candidate(
    row: Any,
    *,
    origin: str,
    destination: str,
    departure_date: date,
) -> tuple[float, bool] | None:
    if not isinstance(row, dict) or str(row.get("type", "")).strip().lower() != "one way":
        return None
    price = _amount(row.get("price"))
    flights = row.get("flights")
    if price is None or not isinstance(flights, list) or not flights or len(flights) > 4:
        return None
    previous_destination: str | None = None
    first_origin: str | None = None
    last_destination: str | None = None
    for index, flight in enumerate(flights):
        if not isinstance(flight, dict):
            return None
        departure = flight.get("departure_airport")
        arrival = flight.get("arrival_airport")
        if not isinstance(departure, dict) or not isinstance(arrival, dict):
            return None
        departure_id = str(departure.get("id", "")).strip().upper()
        arrival_id = str(arrival.get("id", "")).strip().upper()
        if not _IATA_PATTERN.fullmatch(departure_id) or not _IATA_PATTERN.fullmatch(arrival_id):
            return None
        if previous_destination is not None and departure_id != previous_destination:
            return None
        if index == 0:
            first_origin = departure_id
            raw_time = str(departure.get("time", "")).strip()
            if not raw_time.startswith(departure_date.isoformat()):
                return None
        previous_destination = arrival_id
        last_destination = arrival_id
        if not str(flight.get("flight_number", "")).strip():
            return None
        if str(flight.get("travel_class", "")).strip().lower() != "economy":
            return None
        if _positive_int(flight.get("duration")) is None:
            return None
    if first_origin != origin or last_destination != destination:
        return None
    return price, len(flights) == 1


def _cache_key(origin: str, destination: str, departure_date: date) -> str:
    raw = f"{origin}|{destination}|{departure_date.isoformat()}|economy|USD"
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _iata(value: str) -> str:
    code = str(value).strip().upper()
    if not _IATA_PATTERN.fullmatch(code):
        raise ValueError("Scrape.do reference airport code is invalid")
    return code


def _period_key(value: datetime) -> str:
    observed_at = _utc(value)
    return f"{observed_at.year:04d}-{observed_at.month:02d}"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Scrape.do reference time must include a timezone")
    return value.astimezone(UTC)


def _clamp_credit_limit(value: int) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return min(max(parsed, 0), SCRAPE_DO_FREE_MONTHLY_CREDITS)


def _amount(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    return round(amount, 2)


def _positive_amount(value: Any) -> bool:
    return _amount(value) is not None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _header_nonnegative_int(response: Any, name: str) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "items"):
        return None
    expected = name.lower()
    for raw_name, raw_value in headers.items():
        if str(raw_name).strip().lower() == expected:
            return _nonnegative_int(str(raw_value).strip())
    return None


def _transient_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code <= 599


def _status_code(response: Any) -> int:
    for name in ("status_code", "status"):
        value = getattr(response, name, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return 0


def _error_status(
    status_code: int,
) -> tuple[ScrapeDoReferenceStatus, str]:
    if status_code in {401, 403}:
        return "authentication_failed", "AuthenticationError"
    if status_code == 402:
        return "quota_exhausted", "ProviderQuotaExhausted"
    if status_code == 429:
        return "rate_limited", "RateLimitError"
    if 400 <= status_code <= 499:
        return "provider_error", "RequestRejected"
    if 500 <= status_code <= 599:
        return "provider_error", "ProviderHttpError"
    return "provider_unavailable", "InvalidHttpStatus"
