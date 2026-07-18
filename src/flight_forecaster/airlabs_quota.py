"""Shared fail-closed quota enforcement for every AirLabs transport.

AirLabs includes account usage metadata in ``request.key``.  The provider's
``limits_by_month`` value is the monthly ceiling and ``limits_total`` is the
month-to-date call count.  Only those two integers are persisted; API keys,
account identifiers, request parameters, and response data are never stored.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_AIRLABS_MONTHLY_CALLS = 1_000
AIRLABS_USAGE_DB_ENV = "AIRLABS_USAGE_DB"
AIRLABS_MONTHLY_CALL_LIMIT_ENV = "AIRLABS_MONTHLY_CALL_LIMIT"


class AirLabsQuotaError(RuntimeError):
    """AirLabs quota safety could not authorize an outbound request."""


class AirLabsQuotaNotConfigured(AirLabsQuotaError):
    """The required local free-tier ceiling is missing or invalid."""


class AirLabsQuotaExhausted(AirLabsQuotaError):
    """The shared local/provider-aware monthly hard stop was reached."""


@dataclass(frozen=True, slots=True)
class AirLabsQuotaReservation:
    used: int
    limit: int
    period_key: str


@dataclass(frozen=True, slots=True)
class AirLabsAccountSnapshot:
    period_key: str
    limits_by_month: int
    limits_total: int
    remaining: int
    observed_at: datetime


def configured_airlabs_monthly_limit(value: Any = None) -> int | None:
    """Return a safe configured limit, or ``None`` to fail closed.

    Supplying no explicit value reads ``AIRLABS_MONTHLY_CALL_LIMIT``.  Values
    above the known free ceiling are clamped, while missing, boolean, malformed,
    and non-positive values do not silently enable provider calls.
    """

    raw = os.getenv(AIRLABS_MONTHLY_CALL_LIMIT_ENV) if value is None else value
    if raw is None or isinstance(raw, bool):
        return None
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if parsed < 1:
        return None
    return min(parsed, MAX_AIRLABS_MONTHLY_CALLS)


def airlabs_usage_path(default: str | Path | None = None) -> Path:
    configured = os.getenv(AIRLABS_USAGE_DB_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path(default or Path("runtime") / "airlabs-usage.sqlite3")


class AirLabsQuotaLedger:
    """Atomic cross-process monthly reservations and sanitized account state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def reserve(self, *, hard_limit: int | None, now: datetime) -> AirLabsQuotaReservation:
        if hard_limit is None:
            raise AirLabsQuotaNotConfigured(
                "AIRLABS_MONTHLY_CALL_LIMIT is required for AirLabs requests"
            )
        limit = configured_airlabs_monthly_limit(hard_limit)
        if limit is None:
            raise AirLabsQuotaNotConfigured("AirLabs monthly call limit is invalid")
        observed = _as_utc(now)
        period_key = observed.strftime("%Y-%m")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            usage_row = connection.execute(
                """
                SELECT calls FROM airlabs_monthly_usage WHERE period_key = ?
                """,
                (period_key,),
            ).fetchone()
            used = max(0, int(usage_row[0])) if usage_row is not None else 0

            account_row = connection.execute(
                """
                SELECT limits_by_month, limits_total
                FROM airlabs_account_snapshot WHERE period_key = ?
                """,
                (period_key,),
            ).fetchone()
            if account_row is not None:
                provider_limit = max(0, int(account_row[0]))
                provider_used = max(0, int(account_row[1]))
                # A paid account must not expand this integration beyond its
                # explicit, free-tier-sized local ceiling.
                limit = min(limit, provider_limit, MAX_AIRLABS_MONTHLY_CALLS)
                used = max(used, provider_used)

            if limit < 1 or used + 1 > limit:
                connection.commit()
                raise AirLabsQuotaExhausted("AirLabs monthly free-call hard stop reached")

            used += 1
            connection.execute(
                """
                INSERT INTO airlabs_monthly_usage(period_key, calls)
                VALUES (?, ?)
                ON CONFLICT(period_key) DO UPDATE SET calls = excluded.calls
                """,
                (period_key, used),
            )
            connection.commit()
            return AirLabsQuotaReservation(used=used, limit=limit, period_key=period_key)
        except AirLabsQuotaExhausted:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise AirLabsQuotaError("AirLabs quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def observe_payload(
        self,
        payload: Any,
        *,
        observed_at: datetime,
    ) -> AirLabsAccountSnapshot | None:
        """Persist only valid provider quota counters from an AirLabs response."""

        counters = _quota_counters(payload)
        if counters is None:
            return None
        provider_limit, provider_used = counters
        observed = _as_utc(observed_at)
        period_key = observed.strftime("%Y-%m")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT limits_by_month, limits_total
                FROM airlabs_account_snapshot WHERE period_key = ?
                """,
                (period_key,),
            ).fetchone()
            if existing is not None:
                # Concurrent responses can arrive out of order.  Within one
                # month, never let a stale response increase available quota.
                provider_limit = min(provider_limit, max(0, int(existing[0])))
                provider_used = max(provider_used, max(0, int(existing[1])))
            remaining = max(
                0,
                min(provider_limit, MAX_AIRLABS_MONTHLY_CALLS) - provider_used,
            )
            connection.execute(
                """
                INSERT INTO airlabs_account_snapshot(
                    period_key, limits_by_month, limits_total, remaining, observed_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(period_key) DO UPDATE SET
                    limits_by_month = excluded.limits_by_month,
                    limits_total = excluded.limits_total,
                    remaining = excluded.remaining,
                    observed_at = excluded.observed_at
                """,
                (
                    period_key,
                    provider_limit,
                    provider_used,
                    remaining,
                    observed.isoformat(),
                ),
            )
            connection.commit()
            return AirLabsAccountSnapshot(
                period_key=period_key,
                limits_by_month=provider_limit,
                limits_total=provider_used,
                remaining=remaining,
                observed_at=observed,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise AirLabsQuotaError("AirLabs account quota snapshot is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def calls_used(self, *, now: datetime) -> int:
        period_key = _as_utc(now).strftime("%Y-%m")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT calls FROM airlabs_monthly_usage WHERE period_key = ?",
                (period_key,),
            ).fetchone()
            return max(0, int(row[0])) if row is not None else 0
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise AirLabsQuotaError("AirLabs quota ledger is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def account_snapshot(self, *, now: datetime) -> AirLabsAccountSnapshot | None:
        period_key = _as_utc(now).strftime("%Y-%m")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                """
                SELECT limits_by_month, limits_total, remaining, observed_at
                FROM airlabs_account_snapshot WHERE period_key = ?
                """,
                (period_key,),
            ).fetchone()
            if row is None:
                return None
            return AirLabsAccountSnapshot(
                period_key=period_key,
                limits_by_month=max(0, int(row[0])),
                limits_total=max(0, int(row[1])),
                remaining=max(0, int(row[2])),
                observed_at=_as_utc(datetime.fromisoformat(str(row[3]))),
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise AirLabsQuotaError("AirLabs account quota snapshot is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 30000")
        # Do not switch journal modes on every connection.  Concurrent first
        # callers can otherwise contend inside ``PRAGMA journal_mode`` before
        # the busy handler protects the actual reservation transaction.  The
        # default journal mode plus ``BEGIN IMMEDIATE`` still serializes every
        # cross-process quota reservation atomically.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS airlabs_monthly_usage (
                period_key TEXT PRIMARY KEY,
                calls INTEGER NOT NULL CHECK(calls >= 0)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS airlabs_account_snapshot (
                period_key TEXT PRIMARY KEY,
                limits_by_month INTEGER NOT NULL CHECK(limits_by_month >= 0),
                limits_total INTEGER NOT NULL CHECK(limits_total >= 0),
                remaining INTEGER NOT NULL CHECK(remaining >= 0),
                observed_at TEXT NOT NULL
            )
            """
        )
        return connection


class AirLabsQuotaGate:
    """Reserve a shared call, execute one transport, and record quota metadata."""

    def __init__(
        self,
        *,
        ledger: AirLabsQuotaLedger,
        monthly_call_limit: int | None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.ledger = ledger
        self.monthly_call_limit = configured_airlabs_monthly_limit(monthly_call_limit)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    @classmethod
    def from_env(
        cls,
        *,
        default_usage_path: str | Path | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> AirLabsQuotaGate:
        return cls(
            ledger=AirLabsQuotaLedger(airlabs_usage_path(default_usage_path)),
            monthly_call_limit=configured_airlabs_monthly_limit(),
            now_provider=now_provider,
        )

    def get_json(
        self,
        client: Any,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        now = _as_utc(self._now_provider())
        self.ledger.reserve(hard_limit=self.monthly_call_limit, now=now)
        response = client.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        payload = response.json()
        self.ledger.observe_payload(payload, observed_at=now)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        return payload


def _quota_counters(payload: Any) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None
    request = payload.get("request")
    if not isinstance(request, dict):
        return None
    key = request.get("key")
    if not isinstance(key, dict):
        return None
    monthly = _non_negative_int(key.get("limits_by_month"))
    total = _non_negative_int(key.get("limits_total"))
    if monthly is None or total is None:
        return None
    return monthly, total


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
