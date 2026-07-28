from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from flight_forecaster.airlabs_quota import (
    AirLabsQuotaExhausted,
    AirLabsQuotaGate,
    AirLabsQuotaLedger,
    configured_airlabs_monthly_limit,
)
from flight_forecaster.context import (
    AIRLABS_FREE_SAMPLE_LIMIT,
    AIRLABS_ROUTES_URL,
    AIRLABS_SCHEDULES_URL,
    ContextProvider,
)
from flight_forecaster.schedules import ScheduleProvider


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if not 200 <= self.status_code < 300:
            raise RuntimeError(f"HTTP {self.status_code}")


class _QueueClient:
    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self.responses = {url: list(payloads) for url, payloads in responses.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._lock = Lock()

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert params["api_key"] == "test-secret-key"
        assert headers["Accept"] == "application/json"
        assert timeout > 0
        with self._lock:
            self.calls.append((url, dict(params)))
            payload = self.responses[url].pop(0)
        return _Response(payload)


def _payload(*, total: int, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "request": {
            "key": {
                "api_key": "test-secret-key",
                "id": 12345,
                "type": "free",
                "limits_by_month": 1000,
                "limits_total": total,
            }
        },
        "response": rows or [],
    }


def _gate(path: Path, now: datetime, *, limit: int = 1000) -> AirLabsQuotaGate:
    return AirLabsQuotaGate(
        ledger=AirLabsQuotaLedger(path),
        monthly_call_limit=limit,
        now_provider=lambda: now,
    )


def test_schedule_endpoints_and_context_operations_share_one_sanitized_ledger(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")
    selected_date = now.astimezone(origin_zone).date()
    scheduled = now + timedelta(minutes=30)
    client = _QueueClient(
        {
            AIRLABS_SCHEDULES_URL: [
                _payload(total=1),
                _payload(
                    total=3,
                    rows=[
                        {
                            "dep_iata": "YYZ",
                            "dep_time_utc": scheduled.isoformat(),
                            "status": "scheduled",
                            "dep_delayed": 0,
                        }
                    ],
                ),
            ],
            AIRLABS_ROUTES_URL: [_payload(total=2)],
        }
    )
    usage_path = tmp_path / "shared-airlabs.sqlite3"
    gate = _gate(usage_path, now)
    schedule_provider = ScheduleProvider(
        api_key="test-secret-key",
        client=client,
        airlabs_quota_gate=gate,
    )
    context_provider = ContextProvider(
        airlabs_api_key="test-secret-key",
        client=client,
        airlabs_quota_gate=gate,
    )

    schedule_provider.search(
        "YYZ",
        "LHR",
        selected_date,
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=now,
    )
    current, _ = context_provider._airlabs_operations_snapshots(  # noqa: SLF001
        "YYZ",
        scheduled,
        now,
    )

    assert [url for url, _ in client.calls] == [
        AIRLABS_SCHEDULES_URL,
        AIRLABS_ROUTES_URL,
        AIRLABS_SCHEDULES_URL,
    ]
    assert current is not None
    assert current.source == "airlabs_schedules"
    assert current.sample_limit == AIRLABS_FREE_SAMPLE_LIMIT
    assert gate.ledger.calls_used(now=now) == 3
    snapshot = gate.ledger.account_snapshot(now=now)
    assert snapshot is not None
    assert snapshot.limits_by_month == 1000
    assert snapshot.limits_total == 3
    assert snapshot.remaining == 997
    database_bytes = usage_path.read_bytes()
    assert b"test-secret-key" not in database_bytes
    assert b"api_key" not in database_bytes


def test_provider_reported_usage_hard_stops_the_next_transport(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    path = tmp_path / "provider-remaining.sqlite3"
    gate = _gate(path, now)
    client = _QueueClient(
        {AIRLABS_ROUTES_URL: [_payload(total=1000), _payload(total=1000)]}
    )
    params = {
        "api_key": "test-secret-key",
        "dep_iata": "YYZ",
        "arr_iata": "LHR",
    }
    headers = {"Accept": "application/json"}

    gate.get_json(
        client,
        AIRLABS_ROUTES_URL,
        params=params,
        headers=headers,
        timeout=3,
    )
    with pytest.raises(AirLabsQuotaExhausted):
        gate.get_json(
            client,
            AIRLABS_ROUTES_URL,
            params=params,
            headers=headers,
            timeout=3,
        )

    assert len(client.calls) == 1
    snapshot = gate.ledger.account_snapshot(now=now)
    assert snapshot is not None
    assert snapshot.remaining == 0
    assert gate.ledger.calls_used(now=now) == 1


def test_atomic_reservations_across_independent_ledger_instances(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    path = tmp_path / "concurrent-airlabs.sqlite3"

    def reserve_once(_: int) -> bool:
        ledger = AirLabsQuotaLedger(path)
        try:
            ledger.reserve(hard_limit=10, now=now)
        except AirLabsQuotaExhausted:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(reserve_once, range(40)))

    assert sum(results) == 10
    assert AirLabsQuotaLedger(path).calls_used(now=now) == 10


def test_missing_limit_fails_closed_before_schedule_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AIRLABS_MONTHLY_CALL_LIMIT", raising=False)
    monkeypatch.setenv("AIRLABS_USAGE_DB", str(tmp_path / "missing-limit.sqlite3"))
    client = _QueueClient(
        {
            AIRLABS_SCHEDULES_URL: [_payload(total=1)],
            AIRLABS_ROUTES_URL: [_payload(total=2)],
        }
    )
    now = datetime.now(UTC)
    provider = ScheduleProvider(api_key="test-secret-key", client=client)

    result = provider.search(
        "YYZ",
        "LHR",
        date.today(),
        origin_timezone=ZoneInfo("America/Toronto"),
        destination_timezone=ZoneInfo("Europe/London"),
        fetched_at=now,
    )

    assert not client.calls
    assert result.schedules == ()
    assert result.fallback_code in {
        "airlabs_schedules_unavailable",
        "airlabs_routes_unavailable",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("invalid", None),
        (0, None),
        (-1, None),
        (1, 1),
        (1000, 1000),
        (5000, 1000),
    ],
)
def test_airlabs_monthly_limit_is_fail_closed_and_capped(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
    expected: int | None,
) -> None:
    monkeypatch.delenv("AIRLABS_MONTHLY_CALL_LIMIT", raising=False)
    assert configured_airlabs_monthly_limit(value) == expected
