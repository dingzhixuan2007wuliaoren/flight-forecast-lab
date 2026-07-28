from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from flight_forecaster.context import ADSB_LOL_URL, ContextProvider
from flight_forecaster.schedules import ScheduleProvider
from flight_forecaster.service import PredictionService
from flight_forecaster.supplemental_aviation import (
    AERODATABOX_AIRPORT_FLIGHTS_URL,
    OPENSKY_STATES_URL,
    AeroDataBoxScheduleProvider,
    OpenSkyOperationsProvider,
    SupplementalProviderError,
    SupplementalQuotaExhausted,
    SupplementalUsageLedger,
)


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
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.payload


class _OpenSkyClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert url == OPENSKY_STATES_URL
        assert "Authorization" not in headers
        assert timeout > 0
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _Response(self.payload)


class _OAuthFailureOpenSkyClient(_OpenSkyClient):
    def post(
        self,
        url: str,
        *,
        data: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert data["grant_type"] == "client_credentials"
        assert headers["Content-Type"] == "application/x-www-form-urlencoded"
        assert timeout > 0
        raise SupplementalProviderError("simulated OAuth failure")


class _AdsbClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert url.startswith("https://api.adsb.lol/")
        assert not params
        assert headers["Accept"] == "application/json"
        assert timeout > 0
        self.calls.append(url)
        return _Response(self.payload)


class _AeroDataBoxClient:
    def __init__(
        self,
        row: dict[str, Any],
        *,
        response_headers: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.row = row
        self.response_headers = response_headers or {}
        self.status_code = status_code
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert url.startswith(AERODATABOX_AIRPORT_FLIGHTS_URL.split("{origin}", 1)[0])
        assert headers["X-RapidAPI-Key"] == "test-key"
        assert params["direction"] == "Departure"
        assert timeout > 0
        self.calls.append({"url": url, "params": params})
        departures = [self.row] if "T00:00" in url else []
        return _Response(
            {"departures": departures},
            self.status_code,
            headers=self.response_headers,
        )


class _NoCallClient:
    def get(self, *args: Any, **kwargs: Any) -> _Response:
        raise AssertionError("AirLabs must not be called without a configured key")


def _opensky_payload(observed: datetime) -> dict[str, Any]:
    return {
        "time": int(observed.timestamp()),
        "states": [
            ["abc123", "ACA850", "Canada", None, None, -79.63, 43.68, 500.0],
            ["def456", "WJA420", "Canada", None, None, -79.55, 43.70, 700.0],
            ["bad", "NOCOORD", "Canada", None, None, None, None, None],
        ],
    }


def _aerodatabox_row() -> dict[str, Any]:
    return {
        "number": "AC 850",
        "status": "Expected",
        "airline": {"iata": "AC"},
        "departure": {
            "airport": {"iata": "YYZ"},
            "scheduledTime": {
                "local": "2026-07-20T20:00:00-04:00",
                "utc": "2026-07-21T00:00:00Z",
            },
            "terminal": "1",
        },
        "arrival": {
            "airport": {"iata": "LHR"},
            "scheduledTime": {
                "local": "2026-07-21T07:30:00+01:00",
                "utc": "2026-07-21T06:30:00Z",
            },
            "terminal": "2",
        },
        "aircraft": {"icao": "B789"},
        "lastUpdatedUtc": "2026-07-16T11:55:00Z",
    }


def test_opensky_anonymous_cache_and_daily_hard_stop_use_real_clock(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _OpenSkyClient(_opensky_payload(now))
    provider = OpenSkyOperationsProvider(
        usage_path=tmp_path / "usage.sqlite3",
        daily_credit_limit=1,
        client=client,
        now_provider=lambda: now,
    )

    first = provider.snapshot(43.6777, -79.6248, "large_airport", now)
    cached = provider.snapshot(43.6777, -79.6248, "large_airport", now)

    assert first.authentication_mode == "anonymous"
    assert first.aircraft_count == 2
    assert first.quota_used == first.quota_limit == 1
    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert len(client.calls) == 1

    # A caller-supplied next-day timestamp cannot reset today's local hard stop.
    with pytest.raises(SupplementalQuotaExhausted, match="daily credit hard stop"):
        provider.snapshot(
            40.6413,
            -73.7781,
            "large_airport",
            now + timedelta(days=1),
        )
    assert len(client.calls) == 1


def test_context_falls_back_to_adsb_when_opensky_quota_is_exhausted(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    opensky_client = _OpenSkyClient(_opensky_payload(now))
    opensky = OpenSkyOperationsProvider(
        usage_path=tmp_path / "usage.sqlite3",
        daily_credit_limit=1,
        client=opensky_client,
        now_provider=lambda: now,
    )
    opensky.snapshot(43.6777, -79.6248, "large_airport", now)
    adsb_client = _AdsbClient(
        {
            "now": int(now.timestamp() * 1000),
            "ac": [{"hex": "a1"}, {"hex": "a2"}, {"hex": "a3"}],
        }
    )
    context = ContextProvider(client=adsb_client, opensky_provider=opensky)

    signal = context._operations(  # noqa: SLF001
        "YUL",
        now + timedelta(minutes=30),
        45.4706,
        -73.7408,
        "large_airport",
        now,
        "CA",
    )

    assert signal.source == "adsb_lol"
    assert signal.current_snapshot is not None
    assert signal.current_snapshot.source == "adsb_lol"
    assert signal.fallback_reason is not None
    assert "opensky_daily_quota_exhausted" in signal.fallback_reason
    expected_url = ADSB_LOL_URL.format(latitude=45.4706, longitude=-73.7408)
    assert adsb_client.calls == [expected_url]
    assert len(opensky_client.calls) == 1


def test_opensky_oauth_failure_uses_anonymous_ceiling(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _OAuthFailureOpenSkyClient(_opensky_payload(now))
    provider = OpenSkyOperationsProvider(
        usage_path=tmp_path / "usage.sqlite3",
        client_id="test-client",
        client_secret="test-secret",
        daily_credit_limit=4_000,
        client=client,
        now_provider=lambda: now,
    )

    snapshot = provider.snapshot(43.6777, -79.6248, "large_airport", now)

    assert snapshot.authentication_mode == "anonymous_after_oauth_failure"
    assert snapshot.quota_used == 1
    assert snapshot.quota_limit == 400
    assert "Authorization" not in client.calls[0]["headers"]


def test_aerodatabox_is_cached_reference_only_and_stops_at_monthly_units(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    clock = {"now": now}
    client = _AeroDataBoxClient(_aerodatabox_row())
    provider = AeroDataBoxScheduleProvider(
        "test-key",
        usage_path=tmp_path / "usage.sqlite3",
        monthly_unit_limit=4,
        request_units=2,
        client=client,
        now_provider=lambda: clock["now"],
    )
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")

    first = provider.search(
        "YYZ",
        "LHR",
        date(2026, 7, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=now,
    )
    cached = provider.search(
        "YYZ",
        "LHR",
        date(2026, 7, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=now,
    )

    assert first.status == "dated_schedule_references"
    assert first.quota_used == first.quota_limit == 4
    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert len(client.calls) == 2
    assert [item.flight_number for item in first.schedules] == ["AC850"]

    # Without trusted provider reset headers, even the real clock crossing a
    # calendar-month boundary cannot reset the lifetime fail-safe.
    clock["now"] = datetime(2026, 8, 1, 12, tzinfo=UTC)
    exhausted = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    assert exhausted.status == "quota_exhausted"
    assert exhausted.fallback_code == "aerodatabox_quota_exhausted"
    assert exhausted.quota_period == "lifetime"
    assert len(client.calls) == 2

    schedule_result = ScheduleProvider(
        api_key=None,
        client=_NoCallClient(),
        aerodatabox_provider=provider,
    ).search(
        "YYZ",
        "LHR",
        date(2026, 7, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
        fetched_at=now,
    )
    assert schedule_result.fallback_code is None
    assert schedule_result.route_airlines == {"AC"}
    assert len(schedule_result.schedules) == 1
    reference = schedule_result.schedules[0]
    assert reference.schedule_status == "future_schedule_reference"
    assert reference.source == "aerodatabox_schedule"
    assert not hasattr(reference, "price")
    assert len(client.calls) == 2


def test_aerodatabox_uses_persisted_rapidapi_reset_not_calendar_month(
    tmp_path: Path,
) -> None:
    clock = {"now": datetime(2026, 7, 16, 12, tzinfo=UTC)}
    reset_headers = {
        "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Limit": "300",
        "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Remaining": "298",
        "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Reset": "1728000",
    }
    usage_path = tmp_path / "usage.sqlite3"
    first_client = _AeroDataBoxClient(
        _aerodatabox_row(),
        response_headers=reset_headers,
    )
    provider = AeroDataBoxScheduleProvider(
        "test-key",
        usage_path=usage_path,
        monthly_unit_limit=4,
        request_units=2,
        client=first_client,
        now_provider=lambda: clock["now"],
    )
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")

    first = provider.search(
        "YYZ",
        "LHR",
        date(2026, 7, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert first.status == "dated_schedule_references"
    assert first.quota_period is not None
    assert first.quota_period.startswith("rapidapi-reset:")

    # Calendar August has started, but the authenticated RapidAPI reset has not.
    clock["now"] = datetime(2026, 8, 1, 0, 30, tzinfo=UTC)
    before_provider_reset = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert before_provider_reset.status == "quota_exhausted"
    assert len(first_client.calls) == 2

    # Once the persisted provider reset passes, exactly one pre-reserved probe
    # can establish the next provider-authenticated billing period.
    clock["now"] = datetime(2026, 8, 5, 12, 1, tzinfo=UTC)
    august_row = _aerodatabox_row()
    august_row["departure"]["scheduledTime"] = {
        "local": "2026-08-20T20:00:00-04:00",
        "utc": "2026-08-21T00:00:00Z",
    }
    august_row["arrival"]["scheduledTime"] = {
        "local": "2026-08-21T07:30:00+01:00",
        "utc": "2026-08-21T06:30:00Z",
    }
    second_client = _AeroDataBoxClient(
        august_row,
        response_headers={
            **reset_headers,
            "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Reset": "2592000",
        },
    )
    restarted = AeroDataBoxScheduleProvider(
        "test-key",
        usage_path=usage_path,
        monthly_unit_limit=4,
        request_units=2,
        client=second_client,
        now_provider=lambda: clock["now"],
    )
    after_provider_reset = restarted.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert after_provider_reset.status == "dated_schedule_references"
    assert after_provider_reset.quota_period is not None
    assert after_provider_reset.quota_period.startswith("rapidapi-reset:")
    assert after_provider_reset.quota_period != first.quota_period
    assert len(second_client.calls) == 2


def test_aerodatabox_rejects_paid_or_unknown_quota_window_and_keeps_lifetime_wall(
    tmp_path: Path,
) -> None:
    clock = {"now": datetime(2026, 7, 16, 12, tzinfo=UTC)}
    client = _AeroDataBoxClient(
        _aerodatabox_row(),
        response_headers={
            "X-RapidAPI-Api-Units-Limit": "6000",
            "X-RapidAPI-Api-Units-Remaining": "5996",
            "X-RapidAPI-Api-Units-Reset": "2592000",
        },
    )
    provider = AeroDataBoxScheduleProvider(
        "test-key",
        usage_path=tmp_path / "usage.sqlite3",
        monthly_unit_limit=4,
        request_units=2,
        client=client,
        now_provider=lambda: clock["now"],
    )
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")

    first = provider.search(
        "YYZ",
        "LHR",
        date(2026, 7, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert first.quota_period == "lifetime"

    clock["now"] = datetime(2026, 8, 20, 12, tzinfo=UTC)
    exhausted = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert exhausted.status == "quota_exhausted"
    assert exhausted.quota_period == "lifetime"
    assert len(client.calls) == 2


def test_aerodatabox_persists_free_quota_headers_from_error_response(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    client = _AeroDataBoxClient(
        _aerodatabox_row(),
        status_code=429,
        response_headers={
            "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Limit": "300",
            "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Remaining": "0",
            "X-Rate-Limit-Rapid-Free-Plans-Hard-Limit-Reset": "2592000",
        },
    )
    provider = AeroDataBoxScheduleProvider(
        "test-key",
        usage_path=tmp_path / "usage.sqlite3",
        monthly_unit_limit=8,
        request_units=2,
        client=client,
        now_provider=lambda: now,
    )
    origin_zone = ZoneInfo("America/Toronto")
    destination_zone = ZoneInfo("Europe/London")

    failed = provider.search(
        "YYZ",
        "LHR",
        date(2026, 7, 20),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert failed.status == "provider_unavailable"
    assert failed.quota_period is not None
    assert failed.quota_period.startswith("rapidapi-reset:")

    blocked = provider.search(
        "YYZ",
        "LHR",
        date(2026, 7, 21),
        origin_timezone=origin_zone,
        destination_timezone=destination_zone,
    )
    assert blocked.status == "quota_exhausted"
    assert blocked.quota_period == failed.quota_period
    assert len(client.calls) == 1


def test_aerodatabox_legacy_calendar_rows_are_migrated_into_lifetime_wall(
    tmp_path: Path,
) -> None:
    usage_path = tmp_path / "usage.sqlite3"
    legacy = SupplementalUsageLedger(usage_path)
    legacy_reservation = legacy.reserve(
        "aerodatabox",
        "2026-07",
        units=4,
        hard_limit=4,
    )
    assert legacy_reservation.reserved is True

    client = _AeroDataBoxClient(_aerodatabox_row())
    provider = AeroDataBoxScheduleProvider(
        "test-key",
        usage_path=usage_path,
        monthly_unit_limit=4,
        request_units=2,
        client=client,
        now_provider=lambda: datetime(2026, 8, 20, 12, tzinfo=UTC),
    )
    blocked = provider.search(
        "YYZ",
        "LHR",
        date(2026, 8, 20),
        origin_timezone=ZoneInfo("America/Toronto"),
        destination_timezone=ZoneInfo("Europe/London"),
    )

    assert blocked.status == "quota_exhausted"
    assert blocked.quota_period == "lifetime"
    assert client.calls == []


def test_prediction_service_wires_supplemental_providers_without_network_calls(
    trained_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSKY_ENABLED", "1")
    monkeypatch.setenv("AERODATABOX_API_KEY", "test-key")
    monkeypatch.setenv("SUPPLEMENTAL_AVIATION_USAGE_DB", str(tmp_path / "usage.sqlite3"))

    service = PredictionService(trained_model_dir)

    assert isinstance(service.context_provider.opensky_provider, OpenSkyOperationsProvider)
    assert isinstance(
        service.schedule_provider.aerodatabox_provider,
        AeroDataBoxScheduleProvider,
    )
