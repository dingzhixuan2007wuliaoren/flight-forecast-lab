"""Quota-safe supplemental aviation data adapters.

OpenSky is used only as a current traffic-density proxy. AeroDataBox is used
only as a dated timetable reference. Neither adapter proves fare inventory or
bookability, and neither may populate the strict fare comparison list.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any
from urllib import error, parse, request
from zoneinfo import ZoneInfo

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
AERODATABOX_API_HOST = "aerodatabox.p.rapidapi.com"
AERODATABOX_AIRPORT_FLIGHTS_URL = (
    "https://aerodatabox.p.rapidapi.com/flights/airports/iata/"
    "{origin}/{start_local}/{end_local}"
)

REQUEST_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 5_000_000
MAX_OPENSKY_ANONYMOUS_DAILY_CREDITS = 400
MAX_OPENSKY_REGISTERED_DAILY_CREDITS = 4_000
MAX_AERODATABOX_MONTHLY_UNITS = 600
MAX_SCHEDULE_ROWS = 50
MAX_FLIGHT_DURATION_MINUTES = 2_160
OPENSKY_MAX_AGE_MINUTES = 5
_CODE_PATTERN = re.compile(r"^[A-Z0-9]{2,3}$")
_FLIGHT_PATTERN = re.compile(r"^[A-Z0-9]{3,12}$")
_TERMINAL_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,40}$")


class SupplementalProviderError(RuntimeError):
    """A supplemental provider failed without affecting the prediction."""


class SupplementalQuotaExhausted(SupplementalProviderError):
    """The conservative local hard stop rejected a provider call."""


class SupplementalProviderResponseError(SupplementalProviderError):
    """A failed response carrying only normalized, non-secret quota headers."""

    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__("supplemental provider response failed")
        self.headers = headers


@dataclass(frozen=True, slots=True)
class QuotaReservation:
    reserved: bool
    used: int
    limit: int
    period_key: str


@dataclass(frozen=True, slots=True)
class ProviderQuotaObservation:
    """A provider-authenticated free quota window, expressed in API units."""

    remaining: int
    limit: int
    reset_at: datetime
    evidence: str


class SupplementalUsageLedger:
    """Cross-process conservative request-unit reservations.

    Every outbound provider attempt is reserved before it starts. Failed and
    provider-cached calls therefore still count locally; this intentionally
    favours never crossing the configured free-tier hard stop.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def reserve(
        self,
        provider: str,
        period_key: str,
        *,
        units: int,
        hard_limit: int,
    ) -> QuotaReservation:
        if not re.fullmatch(r"[a-z0-9_]{2,40}", provider):
            raise SupplementalProviderError("invalid quota provider scope")
        if not period_key or len(period_key) > 40 or units < 1 or hard_limit < 1:
            raise SupplementalProviderError("invalid quota reservation")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT units FROM supplemental_aviation_usage
                WHERE provider = ? AND period_key = ?
                """,
                (provider, period_key),
            ).fetchone()
            used = max(0, int(row[0])) if row is not None else 0
            if used + units > hard_limit:
                connection.commit()
                return QuotaReservation(False, used, hard_limit, period_key)
            used += units
            connection.execute(
                """
                INSERT INTO supplemental_aviation_usage(provider, period_key, units)
                VALUES (?, ?, ?)
                ON CONFLICT(provider, period_key) DO UPDATE SET units = excluded.units
                """,
                (provider, period_key, used),
            )
            connection.commit()
            return QuotaReservation(True, used, hard_limit, period_key)
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise SupplementalProviderError("supplemental quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def snapshot(self, provider: str, period_key: str) -> int:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                """
                SELECT units FROM supplemental_aviation_usage
                WHERE provider = ? AND period_key = ?
                """,
                (provider, period_key),
            ).fetchone()
            return max(0, int(row[0])) if row is not None else 0
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise SupplementalProviderError("supplemental quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def reserve_provider_window(
        self,
        provider: str,
        *,
        units: int,
        hard_limit: int,
        observed_at: datetime,
    ) -> QuotaReservation:
        """Atomically reserve from a trusted window or the lifetime fail-safe.

        A trusted provider reset may open exactly one provisional reservation in
        the next cycle. If that request does not return new trusted headers, the
        provisional state is discarded and later attempts fall back to the
        installation-wide lifetime ceiling.
        """

        if not re.fullmatch(r"[a-z0-9_]{2,40}", provider):
            raise SupplementalProviderError("invalid quota provider scope")
        if units < 1 or hard_limit < 1:
            raise SupplementalProviderError("invalid quota reservation")
        now = _as_utc(observed_at)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            lifetime_row_used = self._usage_in_connection(connection, provider, "lifetime")
            legacy_row = connection.execute(
                """
                SELECT COALESCE(SUM(units), 0)
                FROM supplemental_aviation_usage
                WHERE provider = ? AND period_key <> 'lifetime'
                """,
                (provider,),
            ).fetchone()
            legacy_used = max(0, int(legacy_row[0])) if legacy_row is not None else 0
            lifetime_used = lifetime_row_used + legacy_used
            row = connection.execute(
                """
                SELECT period_key, hard_limit, remaining, reset_at, evidence
                FROM supplemental_provider_quota_windows
                WHERE provider = ?
                """,
                (provider,),
            ).fetchone()

            if row is not None:
                period_key, stored_limit, remaining, reset_text, evidence = row
                stored_limit = min(hard_limit, max(1, int(stored_limit)))
                remaining = max(0, min(stored_limit, int(remaining)))
                reset_at = _stored_utc_datetime(reset_text)
                if evidence == "trusted_headers" and reset_at is not None:
                    if now < reset_at:
                        if remaining < units:
                            connection.commit()
                            return QuotaReservation(
                                False,
                                stored_limit - remaining,
                                stored_limit,
                                str(period_key),
                            )
                        remaining -= units
                        connection.execute(
                            """
                            UPDATE supplemental_provider_quota_windows
                            SET remaining = ?, updated_at = ?
                            WHERE provider = ?
                            """,
                            (remaining, _utc_text(now), provider),
                        )
                        self._increment_usage_in_connection(
                            connection, provider, "lifetime", units
                        )
                        connection.commit()
                        return QuotaReservation(
                            True,
                            stored_limit - remaining,
                            stored_limit,
                            str(period_key),
                        )

                    # The provider-authenticated reset has passed. Reserve one
                    # bounded probe in the new period, then require fresh headers.
                    if units > hard_limit:
                        connection.commit()
                        return QuotaReservation(False, 0, hard_limit, "lifetime")
                    provisional_period = f"rapidapi-after:{_utc_text(reset_at)}"
                    provisional_remaining = hard_limit - units
                    connection.execute(
                        """
                        UPDATE supplemental_provider_quota_windows
                        SET period_key = ?, hard_limit = ?, remaining = ?,
                            reset_at = NULL, evidence = 'provisional_after_reset',
                            updated_at = ?
                        WHERE provider = ?
                        """,
                        (
                            provisional_period,
                            hard_limit,
                            provisional_remaining,
                            _utc_text(now),
                            provider,
                        ),
                    )
                    self._increment_usage_in_connection(
                        connection, provider, "lifetime", units
                    )
                    connection.commit()
                    return QuotaReservation(
                        True,
                        units,
                        hard_limit,
                        provisional_period,
                    )

                # A process may have stopped after making the one provisional
                # request. Do not assume another monthly window without headers.
                if evidence == "provisional_after_reset":
                    connection.execute(
                        "DELETE FROM supplemental_provider_quota_windows WHERE provider = ?",
                        (provider,),
                    )

            if lifetime_used + units > hard_limit:
                connection.commit()
                return QuotaReservation(False, lifetime_used, hard_limit, "lifetime")
            lifetime_row_used = self._increment_usage_in_connection(
                connection, provider, "lifetime", units
            )
            lifetime_used = legacy_used + lifetime_row_used
            connection.commit()
            return QuotaReservation(True, lifetime_used, hard_limit, "lifetime")
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise SupplementalProviderError("supplemental quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def record_provider_window(
        self,
        provider: str,
        observation: ProviderQuotaObservation,
        *,
        reservation: QuotaReservation,
        observed_at: datetime,
    ) -> QuotaReservation:
        """Persist a trusted response window without ever increasing capacity."""

        now = _as_utc(observed_at)
        safe_limit = min(reservation.limit, observation.limit)
        reservation_remaining = max(0, reservation.limit - reservation.used)
        safe_remaining = min(
            safe_limit,
            reservation_remaining,
            max(0, observation.remaining),
        )
        reset_at = _as_utc(observation.reset_at)
        period_key = f"rapidapi-reset:{_utc_text(reset_at)}"
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT period_key, hard_limit, remaining, reset_at, evidence
                FROM supplemental_provider_quota_windows
                WHERE provider = ?
                """,
                (provider,),
            ).fetchone()
            if current is not None:
                current_reset = _stored_utc_datetime(current[3])
                if (
                    current[4] == "trusted_headers"
                    and current_reset is not None
                    and now < current_reset
                ):
                    # Reset countdowns can drift by seconds between responses.
                    # Retain the later boundary and never increase remaining.
                    if abs((reset_at - current_reset).total_seconds()) > 86_400:
                        connection.commit()
                        return QuotaReservation(
                            True,
                            int(current[1]) - int(current[2]),
                            int(current[1]),
                            str(current[0]),
                        )
                    reset_at = max(reset_at, current_reset)
                    period_key = f"rapidapi-reset:{_utc_text(reset_at)}"
                    safe_limit = min(safe_limit, int(current[1]))
                    safe_remaining = min(safe_remaining, int(current[2]), safe_limit)
            connection.execute(
                """
                INSERT INTO supplemental_provider_quota_windows(
                    provider, period_key, hard_limit, remaining, reset_at,
                    evidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'trusted_headers', ?)
                ON CONFLICT(provider) DO UPDATE SET
                    period_key = excluded.period_key,
                    hard_limit = excluded.hard_limit,
                    remaining = excluded.remaining,
                    reset_at = excluded.reset_at,
                    evidence = excluded.evidence,
                    updated_at = excluded.updated_at
                """,
                (
                    provider,
                    period_key,
                    safe_limit,
                    safe_remaining,
                    _utc_text(reset_at),
                    _utc_text(now),
                ),
            )
            connection.commit()
            return QuotaReservation(
                True,
                safe_limit - safe_remaining,
                safe_limit,
                period_key,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise SupplementalProviderError("supplemental quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def clear_provisional_window(self, provider: str, period_key: str) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                DELETE FROM supplemental_provider_quota_windows
                WHERE provider = ? AND period_key = ?
                    AND evidence = 'provisional_after_reset'
                """,
                (provider, period_key),
            )
            connection.commit()
        except (OSError, sqlite3.Error) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise SupplementalProviderError("supplemental quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _usage_in_connection(
        connection: sqlite3.Connection,
        provider: str,
        period_key: str,
    ) -> int:
        row = connection.execute(
            """
            SELECT units FROM supplemental_aviation_usage
            WHERE provider = ? AND period_key = ?
            """,
            (provider, period_key),
        ).fetchone()
        return max(0, int(row[0])) if row is not None else 0

    @classmethod
    def _increment_usage_in_connection(
        cls,
        connection: sqlite3.Connection,
        provider: str,
        period_key: str,
        units: int,
    ) -> int:
        used = cls._usage_in_connection(connection, provider, period_key) + units
        connection.execute(
            """
            INSERT INTO supplemental_aviation_usage(provider, period_key, units)
            VALUES (?, ?, ?)
            ON CONFLICT(provider, period_key) DO UPDATE SET units = excluded.units
            """,
            (provider, period_key, used),
        )
        return used

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS supplemental_aviation_usage (
                provider TEXT NOT NULL,
                period_key TEXT NOT NULL,
                units INTEGER NOT NULL CHECK(units >= 0),
                PRIMARY KEY(provider, period_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS supplemental_provider_quota_windows (
                provider TEXT PRIMARY KEY,
                period_key TEXT NOT NULL,
                hard_limit INTEGER NOT NULL CHECK(hard_limit > 0),
                remaining INTEGER NOT NULL CHECK(remaining >= 0),
                reset_at TEXT,
                evidence TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        return connection


class _UrllibResponse:
    def __init__(self, status_code: int, body: bytes, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers

    def raise_for_status(self) -> None:
        if not 200 <= self.status_code < 300:
            raise SupplementalProviderError(f"provider returned HTTP {self.status_code}")

    def json(self) -> Any:
        try:
            return json.loads(self._body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SupplementalProviderError("provider returned invalid JSON") from exc


class _UrllibClient:
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> _UrllibResponse:
        query = parse.urlencode(params or {})
        target = f"{url}?{query}" if query else url
        return self._request(request.Request(target, headers=headers or {}), timeout)

    def post(
        self,
        url: str,
        *,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> _UrllibResponse:
        body = parse.urlencode(data).encode("ascii")
        return self._request(request.Request(url, data=body, headers=headers or {}), timeout)

    @staticmethod
    def _request(http_request: request.Request, timeout: float) -> _UrllibResponse:
        try:
            with request.urlopen(http_request, timeout=timeout) as response:  # noqa: S310
                body = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
                headers = dict(response.headers.items())
        except error.HTTPError as exc:
            body = exc.read(MAX_RESPONSE_BYTES + 1)
            status = int(exc.code)
            headers = dict(exc.headers.items()) if exc.headers is not None else {}
        except (error.URLError, OSError, TimeoutError) as exc:
            raise SupplementalProviderError("provider transport failed") from exc
        if len(body) > MAX_RESPONSE_BYTES:
            raise SupplementalProviderError("provider response exceeded safety limit")
        return _UrllibResponse(status, body, headers)


@dataclass(frozen=True, slots=True)
class OperationsDensitySnapshot:
    value: float
    source: str
    observed_at: datetime
    aircraft_count: int
    density_denominator: int
    authentication_mode: str
    quota_used: int
    quota_limit: int
    quota_period: str
    cache_hit: bool = False


class OpenSkyOperationsProvider:
    """Current OpenSky aircraft density, explicitly not an airport-delay feed."""

    def __init__(
        self,
        *,
        usage_path: str | Path,
        client_id: str | None = None,
        client_secret: str | None = None,
        daily_credit_limit: int = MAX_OPENSKY_ANONYMOUS_DAILY_CREDITS,
        cache_ttl_seconds: float = 300.0,
        client: Any = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        supplied_id = (client_id or "").strip()
        supplied_secret = (client_secret or "").strip()
        self._registered = bool(supplied_id and supplied_secret)
        self._client_id = supplied_id if self._registered else None
        self._client_secret = supplied_secret if self._registered else None
        maximum = (
            MAX_OPENSKY_REGISTERED_DAILY_CREDITS
            if self._registered
            else MAX_OPENSKY_ANONYMOUS_DAILY_CREDITS
        )
        self.daily_credit_limit = _bounded_positive_int(
            daily_credit_limit,
            default=maximum,
            maximum=maximum,
        )
        self.cache_ttl_seconds = _bounded_float(
            cache_ttl_seconds,
            default=300.0,
            minimum=60.0,
            maximum=900.0,
        )
        self._client = client or _UrllibClient()
        self._ledger = SupplementalUsageLedger(usage_path)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._cache: dict[tuple[float, float, str], tuple[float, OperationsDensitySnapshot]] = {}
        self._cache_lock = threading.Lock()
        self._token: str | None = None
        self._token_expires_at = datetime.min.replace(tzinfo=UTC)
        self._token_lock = threading.Lock()

    @property
    def authentication_mode(self) -> str:
        return "registered_oauth2" if self._registered else "anonymous"

    def snapshot(
        self,
        latitude: float,
        longitude: float,
        airport_type: str,
        fetched_at: datetime | None = None,
    ) -> OperationsDensitySnapshot:
        latitude_value = _finite_coordinate(latitude, minimum=-90, maximum=90)
        longitude_value = _finite_coordinate(longitude, minimum=-180, maximum=180)
        quota_now = _as_utc(self._now_provider())
        observed_request = _as_utc(fetched_at or quota_now)
        cache_key = (round(latitude_value, 2), round(longitude_value, 2), airport_type)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None and monotonic() - cached[0] <= self.cache_ttl_seconds:
                return replace(cached[1], cache_hit=True)

        headers = {
            "Accept": "application/json",
            "User-Agent": "flight-forecast-lab/0.2 (OpenSky density proxy)",
        }
        mode = self.authentication_mode
        if self._registered:
            try:
                headers["Authorization"] = f"Bearer {self._bearer_token(quota_now)}"
            except SupplementalProviderError:
                mode = "anonymous_after_oauth_failure"

        # OAuth failure deliberately falls back to anonymous access, but it must
        # also fall back to the anonymous 400-credit ceiling. Keep the historic
        # ledger scope so credits reserved by an older build remain visible.
        effective_limit = self.daily_credit_limit
        if mode != "registered_oauth2":
            effective_limit = min(effective_limit, MAX_OPENSKY_ANONYMOUS_DAILY_CREDITS)
        period = quota_now.strftime("%Y-%m-%d")
        reservation = self._ledger.reserve(
            "opensky",
            period,
            units=1,
            hard_limit=effective_limit,
        )
        if not reservation.reserved:
            raise SupplementalQuotaExhausted("OpenSky daily credit hard stop reached")

        latitude_delta = 0.5
        longitude_delta = 0.5
        params = {
            "lamin": round(max(-90.0, latitude_value - latitude_delta), 4),
            "lomin": round(max(-180.0, longitude_value - longitude_delta), 4),
            "lamax": round(min(90.0, latitude_value + latitude_delta), 4),
            "lomax": round(min(180.0, longitude_value + longitude_delta), 4),
        }
        payload = _get_json(self._client, OPENSKY_STATES_URL, params=params, headers=headers)
        if not isinstance(payload, dict):
            raise SupplementalProviderError("OpenSky response object is missing")
        rows = payload.get("states")
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise SupplementalProviderError("OpenSky states list is invalid")
        aircraft_count = sum(
            1
            for row in rows
            if _valid_opensky_state(row, params=params)
        )
        provider_time = _timestamp(payload.get("time"))
        if provider_time is not None:
            age = observed_request - provider_time
            if not timedelta(minutes=-2) <= age <= timedelta(minutes=OPENSKY_MAX_AGE_MINUTES):
                raise SupplementalProviderError("OpenSky states snapshot is stale")
        snapshot_time = provider_time or observed_request
        denominator = {
            "large_airport": 75,
            "medium_airport": 35,
            "small_airport": 15,
        }.get(airport_type, 35)
        pressure = round(max(0.0, min(1.0, 0.05 + 0.95 * min(aircraft_count / denominator, 1))), 4)
        result = OperationsDensitySnapshot(
            value=pressure,
            source="opensky_states",
            observed_at=snapshot_time,
            aircraft_count=aircraft_count,
            density_denominator=denominator,
            authentication_mode=mode,
            quota_used=reservation.used,
            quota_limit=reservation.limit,
            quota_period=reservation.period_key,
        )
        with self._cache_lock:
            if len(self._cache) >= 256:
                oldest = min(self._cache, key=lambda key: self._cache[key][0])
                self._cache.pop(oldest, None)
            self._cache[cache_key] = (monotonic(), result)
        return result

    def _bearer_token(self, now: datetime) -> str:
        with self._token_lock:
            if self._token is not None and now + timedelta(seconds=60) < self._token_expires_at:
                return self._token
            response = self._client.post(
                OPENSKY_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "flight-forecast-lab/0.2 (OpenSky OAuth client)",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            payload = response.json()
            token = payload.get("access_token") if isinstance(payload, dict) else None
            expires_in = _bounded_positive_int(
                payload.get("expires_in") if isinstance(payload, dict) else None,
                default=300,
                maximum=86_400,
            )
            if not isinstance(token, str) or not token.strip() or len(token) > 8_192:
                raise SupplementalProviderError("OpenSky OAuth response is invalid")
            self._token = token.strip()
            self._token_expires_at = now + timedelta(seconds=expires_in)
            return self._token


@dataclass(frozen=True, slots=True)
class SupplementalSchedule:
    airline_code: str
    flight_number: str
    departure_local: datetime
    arrival_local: datetime
    departure_utc: datetime
    arrival_utc: datetime
    duration_minutes: int
    departure_terminal: str | None = None
    arrival_terminal: str | None = None
    aircraft_icao: str | None = None
    provider_flight_status: str | None = None
    observed_at: datetime | None = None

    @property
    def identity(self) -> tuple[str, str, datetime]:
        return (self.airline_code, self.flight_number, self.departure_utc)


@dataclass(frozen=True, slots=True)
class SupplementalScheduleResult:
    schedules: tuple[SupplementalSchedule, ...]
    status: str
    observed_at: datetime
    source: str = "aerodatabox_schedule"
    fallback_code: str | None = None
    sample_truncated: bool = False
    cache_hit: bool = False
    quota_used: int | None = None
    quota_limit: int | None = None
    quota_period: str | None = None


class AeroDataBoxScheduleProvider:
    """AeroDataBox dated schedule references; never fare or inventory data."""

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: str | Path,
        monthly_unit_limit: int = MAX_AERODATABOX_MONTHLY_UNITS,
        request_units: int = 2,
        cache_ttl_seconds: float = 21_600.0,
        client: Any = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self.monthly_unit_limit = _bounded_positive_int(
            monthly_unit_limit,
            default=MAX_AERODATABOX_MONTHLY_UNITS,
            maximum=MAX_AERODATABOX_MONTHLY_UNITS,
        )
        self.request_units = _bounded_positive_int(request_units, default=2, maximum=6)
        self.cache_ttl_seconds = _bounded_float(
            cache_ttl_seconds,
            default=21_600.0,
            minimum=300.0,
            maximum=86_400.0,
        )
        self._client = client or _UrllibClient()
        self._ledger = SupplementalUsageLedger(usage_path)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._cache: dict[
            tuple[str, str, date], tuple[float, SupplementalScheduleResult]
        ] = {}
        self._cache_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    def search(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        *,
        origin_timezone: ZoneInfo,
        destination_timezone: ZoneInfo,
        fetched_at: datetime | None = None,
    ) -> SupplementalScheduleResult:
        quota_now = _as_utc(self._now_provider())
        observed = _as_utc(fetched_at or quota_now)
        origin_code = _airport_code(origin)
        destination_code = _airport_code(destination)
        if (
            len(origin_code) != 3
            or len(destination_code) != 3
            or origin_code == destination_code
            or not isinstance(departure_date, date)
            or isinstance(departure_date, datetime)
        ):
            return SupplementalScheduleResult(
                (), "invalid_request", observed, fallback_code="aerodatabox_invalid_request"
            )
        if not self.configured:
            return SupplementalScheduleResult(
                (), "not_configured", observed, fallback_code="aerodatabox_api_key_not_configured"
            )
        key = (origin_code, destination_code, departure_date)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None and monotonic() - cached[0] <= self.cache_ttl_seconds:
                return replace(cached[1], cache_hit=True)

        # The airport endpoint accepts at most a 12-hour local interval. Reserve
        # both half-day requests atomically before transport. RapidAPI billing
        # cycles need not align with calendar months, so trusted response headers
        # define the period. Until they do, one lifetime 600-unit fail-safe is used.
        total_request_units = self.request_units * 2
        reservation = self._ledger.reserve_provider_window(
            "aerodatabox",
            units=total_request_units,
            hard_limit=self.monthly_unit_limit,
            observed_at=quota_now,
        )
        if not reservation.reserved:
            return SupplementalScheduleResult(
                (),
                "quota_exhausted",
                observed,
                fallback_code="aerodatabox_quota_exhausted",
                quota_used=reservation.used,
                quota_limit=reservation.limit,
                quota_period=reservation.period_key,
            )

        headers = {
            "Accept": "application/json",
            "X-RapidAPI-Key": self._api_key or "",
            "X-RapidAPI-Host": AERODATABOX_API_HOST,
            "User-Agent": "flight-forecast-lab/0.2 (timetable reference client)",
        }
        rows: list[dict[str, Any]] = []
        quota_headers: dict[str, str] = {}
        completed_requests = 0
        try:
            for start_clock, end_clock in (("00:00", "11:59"), ("12:00", "23:59")):
                url = AERODATABOX_AIRPORT_FLIGHTS_URL.format(
                    origin=origin_code,
                    start_local=f"{departure_date.isoformat()}T{start_clock}",
                    end_local=f"{departure_date.isoformat()}T{end_clock}",
                )
                payload, response_headers = _get_json_with_headers(
                    self._client,
                    url,
                    params={
                        "direction": "Departure",
                        "withLocation": "false",
                        "withCancelled": "false",
                        "withCodeshared": "true",
                        "withCargo": "false",
                        "withPrivate": "false",
                    },
                    headers=headers,
                )
                completed_requests += 1
                if response_headers:
                    quota_headers = response_headers
                departures = payload.get("departures") if isinstance(payload, dict) else None
                if not isinstance(departures, list):
                    raise SupplementalProviderError("AeroDataBox departures list is missing")
                rows.extend(row for row in departures if isinstance(row, dict))
        except Exception as exc:
            if isinstance(exc, SupplementalProviderResponseError) and exc.headers:
                quota_headers = exc.headers
            reservation = self._apply_aerodatabox_quota_headers(
                quota_headers,
                reservation,
                quota_now,
                pending_units=max(
                    0,
                    total_request_units - completed_requests * self.request_units,
                ),
            )
            return SupplementalScheduleResult(
                (),
                "provider_unavailable",
                observed,
                fallback_code="aerodatabox_provider_unavailable",
                quota_used=reservation.used,
                quota_limit=reservation.limit,
                quota_period=reservation.period_key,
            )

        reservation = self._apply_aerodatabox_quota_headers(
            quota_headers,
            reservation,
            quota_now,
            pending_units=0,
        )

        parsed = [
            schedule
            for row in rows
            if (
                schedule := self._parse_row(
                    row,
                    origin_code,
                    destination_code,
                    departure_date,
                    origin_timezone,
                    destination_timezone,
                    observed,
                )
            )
            is not None
        ]
        unique = {schedule.identity: schedule for schedule in parsed}
        ordered_all = sorted(
            unique.values(),
            key=lambda item: (item.departure_utc, item.airline_code, item.flight_number),
        )
        truncated = len(ordered_all) > MAX_SCHEDULE_ROWS
        ordered = tuple(ordered_all[:MAX_SCHEDULE_ROWS])
        result = SupplementalScheduleResult(
            ordered,
            "dated_schedule_references" if ordered else "no_results",
            observed,
            fallback_code=None if ordered else "aerodatabox_no_complete_schedule",
            sample_truncated=truncated,
            quota_used=reservation.used,
            quota_limit=reservation.limit,
            quota_period=reservation.period_key,
        )
        with self._cache_lock:
            if len(self._cache) >= 256:
                oldest = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest, None)
            self._cache[key] = (monotonic(), result)
        return result

    def _apply_aerodatabox_quota_headers(
        self,
        headers: dict[str, str],
        reservation: QuotaReservation,
        observed_at: datetime,
        *,
        pending_units: int,
    ) -> QuotaReservation:
        observation = _aerodatabox_quota_observation(
            headers,
            observed_at=observed_at,
            local_hard_limit=self.monthly_unit_limit,
            request_units=self.request_units,
            pending_units=pending_units,
        )
        if observation is not None:
            return self._ledger.record_provider_window(
                "aerodatabox",
                observation,
                reservation=reservation,
                observed_at=observed_at,
            )
        if reservation.period_key.startswith("rapidapi-after:"):
            self._ledger.clear_provisional_window(
                "aerodatabox",
                reservation.period_key,
            )
        return reservation

    @classmethod
    def _parse_row(
        cls,
        row: dict[str, Any],
        origin: str,
        destination: str,
        requested_date: date,
        origin_timezone: ZoneInfo,
        destination_timezone: ZoneInfo,
        observed_at: datetime,
    ) -> SupplementalSchedule | None:
        departure = row.get("departure")
        arrival = row.get("arrival")
        airline = row.get("airline")
        if not isinstance(departure, dict) or not isinstance(arrival, dict):
            return None
        airline_code = _airport_code(airline.get("iata") if isinstance(airline, dict) else None)
        if not _CODE_PATTERN.fullmatch(airline_code):
            return None
        row_origin = _nested_airport_iata(departure) or origin
        row_destination = _nested_airport_iata(arrival)
        if row_origin != origin or row_destination != destination:
            return None
        flight_number = re.sub(r"[^A-Z0-9]", "", str(row.get("number") or "").upper())
        if not _FLIGHT_PATTERN.fullmatch(flight_number) or not flight_number.startswith(
            airline_code
        ):
            return None
        departure_times = _complete_times(departure, origin_timezone)
        arrival_times = _complete_times(arrival, destination_timezone)
        if departure_times is None or arrival_times is None:
            return None
        departure_local, departure_utc = departure_times
        arrival_local, arrival_utc = arrival_times
        duration = round((arrival_utc - departure_utc).total_seconds() / 60)
        if (
            departure_local.date() != requested_date
            or departure_utc <= observed_at
            or duration <= 0
            or duration > MAX_FLIGHT_DURATION_MINUTES
        ):
            return None
        status = _provider_status(row.get("status"))
        if status in {
            "active",
            "canceled",
            "cancelled",
            "departed",
            "diverted",
            "landed",
        }:
            return None
        aircraft = row.get("aircraft")
        aircraft_icao = None
        if isinstance(aircraft, dict):
            candidate = str(aircraft.get("icao") or aircraft.get("modelCode") or "").upper()
            if re.fullmatch(r"[A-Z0-9-]{2,12}", candidate):
                aircraft_icao = candidate
        provider_observed = _iso_datetime(row.get("lastUpdatedUtc"), UTC) or observed_at
        return SupplementalSchedule(
            airline_code=airline_code,
            flight_number=flight_number,
            departure_local=departure_local,
            arrival_local=arrival_local,
            departure_utc=departure_utc,
            arrival_utc=arrival_utc,
            duration_minutes=duration,
            departure_terminal=_terminal(departure.get("terminal")),
            arrival_terminal=_terminal(arrival.get("terminal")),
            aircraft_icao=aircraft_icao,
            provider_flight_status=status,
            observed_at=provider_observed,
        )


def opensky_provider_from_env(usage_path: str | Path) -> OpenSkyOperationsProvider | None:
    if not _enabled(os.getenv("OPENSKY_ENABLED", "1")):
        return None
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    registered = bool((client_id or "").strip() and (client_secret or "").strip())
    default_limit = (
        MAX_OPENSKY_REGISTERED_DAILY_CREDITS
        if registered
        else MAX_OPENSKY_ANONYMOUS_DAILY_CREDITS
    )
    return OpenSkyOperationsProvider(
        usage_path=usage_path,
        client_id=client_id,
        client_secret=client_secret,
        daily_credit_limit=_environment_int("OPENSKY_DAILY_CREDIT_LIMIT", default_limit),
        cache_ttl_seconds=_environment_float("OPENSKY_CACHE_TTL_SECONDS", 300.0),
    )


def aerodatabox_provider_from_env(
    usage_path: str | Path,
) -> AeroDataBoxScheduleProvider | None:
    api_key = (os.getenv("AERODATABOX_API_KEY") or "").strip()
    if not api_key:
        return None
    return AeroDataBoxScheduleProvider(
        api_key,
        usage_path=usage_path,
        monthly_unit_limit=_environment_int(
            "AERODATABOX_MONTHLY_UNIT_LIMIT", MAX_AERODATABOX_MONTHLY_UNITS
        ),
        request_units=_environment_int("AERODATABOX_SCHEDULE_REQUEST_UNITS", 2),
        cache_ttl_seconds=_environment_float("AERODATABOX_CACHE_TTL_SECONDS", 21_600.0),
    )


def supplemental_usage_path(default: str | Path) -> Path:
    configured = (os.getenv("SUPPLEMENTAL_AVIATION_USAGE_DB") or "").strip()
    return Path(configured).expanduser() if configured else Path(default)


def _get_json(
    client: Any,
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
) -> Any:
    payload, _ = _get_json_with_headers(
        client,
        url,
        params=params,
        headers=headers,
    )
    return payload


def _get_json_with_headers(
    client: Any,
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
) -> tuple[Any, dict[str, str]]:
    try:
        response = client.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response_headers = _normalized_response_headers(response)
        try:
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise SupplementalProviderResponseError(response_headers) from exc
        return payload, response_headers
    except SupplementalProviderResponseError:
        raise
    except SupplementalProviderError:
        raise
    except Exception as exc:
        raise SupplementalProviderError("supplemental provider request failed") from exc


def _normalized_response_headers(response: Any) -> dict[str, str]:
    raw_headers = getattr(response, "headers", None)
    if raw_headers is None or not hasattr(raw_headers, "items"):
        return {}
    normalized: dict[str, str] = {}
    try:
        items = raw_headers.items()
    except Exception:
        return {}
    for key, value in items:
        name = str(key).strip().lower()
        text = str(value).strip()
        if name and len(name) <= 120 and text and len(text) <= 120:
            normalized[name] = text
    return normalized


def _aerodatabox_quota_observation(
    headers: dict[str, str],
    *,
    observed_at: datetime,
    local_hard_limit: int,
    request_units: int,
    pending_units: int,
) -> ProviderQuotaObservation | None:
    """Extract only free-capacity evidence that is safe for cycle resets.

    Generic ``x-ratelimit-requests-*`` headers are deliberately ignored because
    RapidAPI documents that they may describe either a daily or monthly window.
    Trust is limited to the explicit monthly free-plan hard limit or a bounded
    custom API-unit billing object with a reset value.
    """

    if not headers:
        return None
    free_base = "x-rate-limit-rapid-free-plans-hard-limit"
    free_values = _quota_header_values(headers, free_base)

    candidates: list[tuple[int, str, int, int, str | None, int]] = []
    for name in headers:
        if not name.endswith("-limit"):
            continue
        base = name[: -len("-limit")]
        values = _quota_header_values(headers, base)
        if values is None:
            continue
        raw_limit, raw_remaining, reset_value = values
        if base == free_base:
            # This header is emitted only for RapidAPI free plans and is an
            # explicit monthly hard stop. Convert remaining requests to this
            # endpoint's conservative API-unit cost.
            candidates.append(
                (1, base, raw_limit, raw_remaining, reset_value, request_units)
            )
            continue
        billing_name = base.removeprefix("x-ratelimit-").removeprefix("x-rapidapi-")
        if not any(token in billing_name for token in ("unit", "quota")):
            continue
        if free_values is None:
            # A custom quota alone can belong to a paid or non-monthly plan.
            # Require RapidAPI's explicit free-plan marker before resetting.
            continue
        if raw_limit > local_hard_limit:
            # A larger billing object can include paid capacity; never adopt it.
            continue
        if reset_value is None and free_values is not None:
            reset_value = free_values[2]
        candidates.append((0, base, raw_limit, raw_remaining, reset_value, 1))

    for _, base, raw_limit, raw_remaining, reset_value, multiplier in sorted(candidates):
        if raw_limit < 1 or raw_remaining < 0 or raw_remaining > raw_limit:
            continue
        reset_at = _provider_reset_at(reset_value, observed_at)
        if reset_at is None:
            continue
        safe_limit = min(local_hard_limit, raw_limit * multiplier)
        safe_remaining = min(safe_limit, raw_remaining * multiplier)
        safe_remaining = max(0, safe_remaining - max(0, pending_units))
        return ProviderQuotaObservation(
            remaining=safe_remaining,
            limit=safe_limit,
            reset_at=reset_at,
            evidence=base,
        )
    return None


def _quota_header_values(
    headers: dict[str, str],
    base: str,
) -> tuple[int, int, str | None] | None:
    try:
        limit = int(headers[f"{base}-limit"])
        remaining = int(headers[f"{base}-remaining"])
    except (KeyError, TypeError, ValueError):
        return None
    return limit, remaining, headers.get(f"{base}-reset")


def _provider_reset_at(value: str | None, observed_at: datetime) -> datetime | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    observed = _as_utc(observed_at)
    if numeric >= observed.timestamp() - 86_400:
        target = datetime.fromtimestamp(numeric, tz=UTC)
    elif numeric <= 45 * 24 * 60 * 60:
        target = observed + timedelta(seconds=numeric)
    else:
        return None
    if target <= observed or target - observed > timedelta(days=45):
        return None
    # Never reset early because of sub-second drift or clock rounding.
    rounded_timestamp = math.ceil(target.timestamp() / 60) * 60
    return datetime.fromtimestamp(rounded_timestamp, tz=UTC)


def _utc_text(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stored_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed_value.tzinfo is None or parsed_value.utcoffset() is None:
        return None
    return parsed_value.astimezone(UTC)


def _valid_opensky_state(row: Any, *, params: dict[str, float]) -> bool:
    if not isinstance(row, (list, tuple)) or len(row) < 8:
        return False
    try:
        longitude = float(row[5])
        latitude = float(row[6])
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and params["lamin"] <= latitude <= params["lamax"]
        and params["lomin"] <= longitude <= params["lomax"]
    )


def _complete_times(
    container: dict[str, Any],
    timezone: ZoneInfo,
) -> tuple[datetime, datetime] | None:
    block = container.get("scheduledTime")
    if not isinstance(block, dict):
        return None
    local = _iso_datetime(block.get("local"), timezone)
    utc_value = _iso_datetime(block.get("utc"), UTC)
    if local is None or utc_value is None:
        return None
    utc_value = utc_value.astimezone(UTC)
    if abs((local.astimezone(UTC) - utc_value).total_seconds()) > 120:
        return None
    localized = utc_value.astimezone(timezone)
    if abs((localized.replace(tzinfo=None) - local.replace(tzinfo=None)).total_seconds()) > 120:
        return None
    return localized, utc_value


def _iso_datetime(value: Any, default_timezone: ZoneInfo) -> datetime | None:
    if not isinstance(value, str) or len(value) > 48:
        return None
    try:
        parsed_value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed_value.tzinfo is None or parsed_value.utcoffset() is None:
        candidate = parsed_value.replace(tzinfo=default_timezone, fold=0)
        round_trip = candidate.astimezone(UTC).astimezone(default_timezone).replace(tzinfo=None)
        if round_trip != parsed_value:
            return None
        alternate = parsed_value.replace(tzinfo=default_timezone, fold=1)
        if alternate.utcoffset() != candidate.utcoffset():
            return None
        return candidate
    return parsed_value


def _nested_airport_iata(container: dict[str, Any]) -> str:
    airport = container.get("airport")
    return _airport_code(airport.get("iata") if isinstance(airport, dict) else None)


def _terminal(value: Any) -> str | None:
    terminal = str(value or "").strip()
    return terminal if _TERMINAL_PATTERN.fullmatch(terminal) else None


def _provider_status(value: Any) -> str | None:
    status = str(value or "").strip().lower().replace(" ", "_")
    return status if re.fullmatch(r"[a-z_\-]{2,40}", status) else None


def _airport_code(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())[:4]


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    try:
        return datetime.fromtimestamp(number, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("fetched_at must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _finite_coordinate(value: Any, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SupplementalProviderError("invalid airport coordinate") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise SupplementalProviderError("invalid airport coordinate")
    return number


def _bounded_positive_int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed_value = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return min(parsed_value, maximum) if parsed_value > 0 else default


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed_value):
        return default
    return max(minimum, min(maximum, parsed_value))


def _enabled(value: Any) -> bool:
    return str(value or "").strip().lower() not in {"0", "false", "no", "off"}


def _environment_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip())
    except ValueError:
        return default


def _environment_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or "").strip())
    except ValueError:
        return default
