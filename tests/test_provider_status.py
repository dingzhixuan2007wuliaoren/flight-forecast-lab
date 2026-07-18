from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from flight_forecaster.airlabs_quota import AirLabsQuotaLedger
from flight_forecaster.api import _runtime_provider_status, app


def _providers_by_code(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    providers = payload["providers"]
    assert isinstance(providers, list)
    return {str(item["code"]): item for item in providers if isinstance(item, dict)}


def test_provider_status_reports_true_quota_scopes_without_secrets(monkeypatch) -> None:
    secrets = {
        "SERPAPI_API_KEY": "serp-secret-not-for-response",
        "SEARCHAPI_API_KEY": "search-secret-not-for-response",
        "IGNAV_API_KEY": "ignav-secret-not-for-response",
        "SCRAPE_DO_API_TOKEN": "scrapedo-secret-not-for-response",
        "AIRLABS_API_KEY": "airlabs-secret-not-for-response",
        "AERODATABOX_API_KEY": "aerodatabox-secret-not-for-response",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "auto")
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    monkeypatch.setenv("OPENSKY_ENABLED", "1")

    response = TestClient(app).get("/v1/provider-status")

    assert response.status_code == 200
    payload = response.json()
    providers = _providers_by_code(payload)
    assert set(providers) == {
        "serpapi_google_flights",
        "searchapi_google_flights",
        "ignav_quarantine",
        "scrape_do_google_flights_reference",
        "airlabs_reference",
        "aerodatabox_reference",
        "opensky_reference",
    }

    serpapi = providers["serpapi_google_flights"]
    assert (serpapi["active"], serpapi["quota_limit"], serpapi["quota_unit"]) == (
        True,
        250,
        "billing_period_requests",
    )

    searchapi = providers["searchapi_google_flights"]
    assert (searchapi["active"], searchapi["quota_limit"], searchapi["quota_unit"]) == (
        True,
        100,
        "lifetime_requests",
    )
    assert "does not renew monthly" in searchapi["notice"]["en"]

    ignav = providers["ignav_quarantine"]
    assert ignav["status"] == "quarantined"
    assert ignav["can_supply_strict_offers"] is False
    assert (ignav["quota_limit"], ignav["quota_unit"]) == (
        1_000,
        "lifetime_requests",
    )

    scrapedo = providers["scrape_do_google_flights_reference"]
    assert scrapedo["role"] == scrapedo["status"] == "reference_only"
    assert scrapedo["can_supply_strict_offers"] is False
    assert (scrapedo["quota_limit"], scrapedo["quota_unit"]) == (
        1_000,
        "monthly_credits",
    )
    assert scrapedo["quota_cost_per_call"] == 10

    airlabs = providers["airlabs_reference"]
    assert airlabs["configured"] is True
    assert airlabs["active"] is True
    assert airlabs["role"] == airlabs["status"] == "reference_only"
    assert airlabs["can_supply_strict_offers"] is False
    assert (airlabs["quota_limit"], airlabs["quota_unit"]) == (
        1_000,
        "billing_period_requests",
    )
    assert "without network I/O" in airlabs["notice"]["en"]

    aerodatabox = providers["aerodatabox_reference"]
    assert aerodatabox["active"] is True
    assert (aerodatabox["quota_limit"], aerodatabox["quota_unit"]) == (
        600,
        "provider_managed",
    )
    assert "installation-lifetime" in aerodatabox["notice"]["en"]
    assert "per month" not in aerodatabox["notice"]["en"]

    opensky = providers["opensky_reference"]
    assert opensky["configured"] is False
    assert opensky["active"] is True
    assert opensky["role"] == "reference_only"
    assert (opensky["quota_limit"], opensky["quota_unit"]) == (
        400,
        "daily_credits",
    )
    assert "without credentials" in opensky["notice"]["en"]

    serialized = json.dumps(payload, ensure_ascii=False)
    for secret in secrets.values():
        assert secret not in serialized
    assert "exception_type" not in serialized
    assert "raw_error" not in serialized


def test_airlabs_provider_status_fails_closed_without_explicit_limit(monkeypatch) -> None:
    monkeypatch.setenv("AIRLABS_API_KEY", "configured-but-never-returned")
    monkeypatch.delenv("AIRLABS_MONTHLY_CALL_LIMIT", raising=False)

    response = TestClient(app).get("/v1/provider-status")

    assert response.status_code == 200
    airlabs = _providers_by_code(response.json())["airlabs_reference"]
    assert airlabs["configured"] is True
    assert airlabs["active"] is False
    assert airlabs["status"] == "reference_only"
    assert airlabs["quota_status"] == "not_applicable"
    assert airlabs["quota_limit"] is None
    assert airlabs["quota_unit"] is None
    assert "fails closed" in airlabs["notice"]["en"]


def test_airlabs_provider_status_reads_sanitized_existing_ledger(
    monkeypatch, tmp_path
) -> None:
    usage_path = tmp_path / "airlabs-status.sqlite3"
    monkeypatch.setenv("AIRLABS_API_KEY", "configured-but-never-returned")
    monkeypatch.setenv("AIRLABS_MONTHLY_CALL_LIMIT", "900")
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    monkeypatch.setenv("AIRLABS_USAGE_DB", str(usage_path))
    ledger = AirLabsQuotaLedger(usage_path)
    now = datetime.now(UTC)
    ledger.reserve(hard_limit=900, now=now)
    snapshot = ledger.observe_payload(
        {"request": {"key": {"limits_by_month": 1_000, "limits_total": 64}}},
        observed_at=now,
    )
    assert snapshot is not None

    payload = TestClient(app).get("/v1/provider-status").json()
    airlabs = _providers_by_code(payload)["airlabs_reference"]

    assert airlabs["active"] is True
    assert airlabs["quota_status"] == "available"
    assert airlabs["quota_used"] == 64
    assert airlabs["quota_limit"] == 900
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "configured-but-never-returned" not in serialized
    assert str(usage_path) not in serialized


def test_airlabs_status_uses_service_default_ledger_path(monkeypatch, tmp_path) -> None:
    configured_model_dir = tmp_path / "artifacts" / "demo"
    usage_path = configured_model_dir.parent / "runtime" / "airlabs-usage.sqlite3"
    monkeypatch.setenv("MODEL_DIR", str(configured_model_dir))
    monkeypatch.delenv("AIRLABS_USAGE_DB", raising=False)
    monkeypatch.setenv("AIRLABS_API_KEY", "configured-but-never-returned")
    monkeypatch.setenv("AIRLABS_MONTHLY_CALL_LIMIT", "700")
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    ledger = AirLabsQuotaLedger(usage_path)
    ledger.reserve(hard_limit=700, now=datetime.now(UTC))

    providers = {item.code: item for item in _runtime_provider_status().providers}
    airlabs = providers["airlabs_reference"]

    assert airlabs.quota_used == 1
    assert airlabs.quota_limit == 700
    assert airlabs.quota_status == "available"


def test_airlabs_status_respects_global_external_context_switch(monkeypatch) -> None:
    monkeypatch.setenv("AIRLABS_API_KEY", "configured-but-never-returned")
    monkeypatch.setenv("AIRLABS_MONTHLY_CALL_LIMIT", "1000")
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")

    providers = {item.code: item for item in _runtime_provider_status().providers}
    airlabs = providers["airlabs_reference"]

    assert airlabs.configured is True
    assert airlabs.active is False
    assert airlabs.quota_limit == 1_000
    assert "fails closed" not in airlabs.notice.en


def test_supplemental_reference_statuses_respect_global_external_context_switch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AERODATABOX_API_KEY", "configured-but-never-returned")
    monkeypatch.setenv("OPENSKY_ENABLED", "1")
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")

    providers = {item.code: item for item in _runtime_provider_status().providers}
    aerodatabox = providers["aerodatabox_reference"]
    opensky = providers["opensky_reference"]

    assert aerodatabox.configured is True
    assert aerodatabox.active is False
    assert aerodatabox.quota_status == "not_applicable"
    assert opensky.active is False
    assert opensky.quota_status == "not_applicable"


def test_rate_limited_runtime_metadata_is_reported_as_quota_exhausted(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "configured-but-never-returned")
    metadata = SimpleNamespace(
        provider_code="serpapi_google_flights",
        status="rate_limited",
        coverage_status="provider_incomplete",
        monthly_calls_used=12,
        monthly_call_limit=250,
        quota_unit="billing_period_requests",
    )

    providers = {item.code: item for item in _runtime_provider_status(metadata).providers}
    serpapi = providers["serpapi_google_flights"]

    assert (serpapi.active, serpapi.status, serpapi.quota_status) == (
        True,
        "quota_exhausted",
        "exhausted",
    )
    assert (serpapi.quota_used, serpapi.quota_limit) == (12, 250)


def test_searchapi_runtime_usage_keeps_lifetime_quota_unit(monkeypatch) -> None:
    monkeypatch.setenv("SEARCHAPI_API_KEY", "configured-but-never-returned")
    metadata = SimpleNamespace(
        provider_code="searchapi_google_flights",
        status="confirmed_offers",
        coverage_status="complete",
        monthly_calls_used=7,
        monthly_call_limit=100,
    )

    providers = {item.code: item for item in _runtime_provider_status(metadata).providers}
    searchapi = providers["searchapi_google_flights"]

    assert searchapi.status == "quota_available"
    assert searchapi.quota_status == "available"
    assert searchapi.quota_used == 7
    assert searchapi.quota_limit == 100
    assert searchapi.quota_unit == "lifetime_requests"


def test_aggregate_runtime_usage_updates_each_strict_provider_run(monkeypatch) -> None:
    secrets = {
        "SERPAPI_API_KEY": "serp-aggregate-secret-not-for-response",
        "SEARCHAPI_API_KEY": "search-aggregate-secret-not-for-response",
        "IGNAV_API_KEY": "ignav-aggregate-secret-not-for-response",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "auto")
    monkeypatch.setenv("IGNAV_STRICT_RELEASE", "1")
    monkeypatch.setenv("IGNAV_FREE_ACCOUNT_ATTESTED", "1")
    metadata = SimpleNamespace(
        provider_code="strict_fare_aggregate",
        # Top-level aggregate counters must not be copied to every provider card.
        monthly_calls_used=999,
        monthly_call_limit=999,
        status="confirmed_offers",
        coverage_status="quota_and_provider_incomplete",
        provider_runs=[
            SimpleNamespace(
                provider_code="serpapi_google_flights",
                status="confirmed_offers",
                coverage_status="complete",
                monthly_calls_used=31,
                monthly_call_limit=250,
                quota_unit="billing_period_requests",
            ),
            SimpleNamespace(
                provider_code="searchapi_google_flights",
                status="rate_limited",
                coverage_status="provider_incomplete",
                monthly_calls_used=100,
                monthly_call_limit=100,
                quota_unit="lifetime_requests",
            ),
            SimpleNamespace(
                provider_code="ignav_verified_fares",
                status="no_results",
                coverage_status="complete",
                monthly_calls_used=17,
                monthly_call_limit=1_000,
                quota_unit="lifetime_requests",
            ),
        ],
    )

    response = _runtime_provider_status(metadata)
    providers = {item.code: item for item in response.providers}

    serpapi = providers["serpapi_google_flights"]
    assert (serpapi.active, serpapi.status, serpapi.quota_status) == (
        True,
        "quota_available",
        "available",
    )
    assert (serpapi.quota_used, serpapi.quota_limit, serpapi.quota_unit) == (
        31,
        250,
        "billing_period_requests",
    )

    searchapi = providers["searchapi_google_flights"]
    assert (searchapi.active, searchapi.status, searchapi.quota_status) == (
        True,
        "quota_exhausted",
        "exhausted",
    )
    assert (searchapi.quota_used, searchapi.quota_limit, searchapi.quota_unit) == (
        100,
        100,
        "lifetime_requests",
    )

    ignav = providers["ignav_verified_fares"]
    assert (ignav.active, ignav.status, ignav.quota_status) == (
        True,
        "quota_available",
        "available",
    )
    assert (ignav.quota_used, ignav.quota_limit, ignav.quota_unit) == (
        17,
        1_000,
        "lifetime_requests",
    )

    serialized = response.model_dump_json()
    assert "strict_fare_aggregate" not in {
        provider.code for provider in response.providers
    }
    for secret in secrets.values():
        assert secret not in serialized


def test_explicitly_released_ignav_has_a_distinct_strict_identity(monkeypatch) -> None:
    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "ignav")
    monkeypatch.setenv("IGNAV_API_KEY", "configured-but-never-returned")
    monkeypatch.setenv("IGNAV_STRICT_RELEASE", "1")
    monkeypatch.setenv("IGNAV_FREE_ACCOUNT_ATTESTED", "1")

    response = TestClient(app).get("/v1/provider-status")
    providers = _providers_by_code(response.json())

    assert "ignav_quarantine" not in providers
    ignav = providers["ignav_verified_fares"]
    assert ignav["display_name"] == "Ignav Verified Fares"
    assert ignav["active"] is True
    assert ignav["status"] == "configured"
    assert ignav["can_supply_strict_offers"] is True
    assert ignav["quota_unit"] == "lifetime_requests"


def test_dashboard_renders_bilingual_provider_status_and_searchapi_attribution() -> None:
    dashboard = TestClient(app).get("/").text

    for fragment in (
        'fetch("/v1/provider-status"',
        "renderProviderStatus(providerStatusData)",
        "账户终身请求数",
        "lifetime account requests",
        "每月点数",
        "credits / month",
        "每结算周期请求数",
        "requests / billing period",
        "供应商周期 / 生命周期保护",
        "provider cycle / lifetime fail-safe",
        "日期级时刻参考（仅参考）",
        "Dated schedule reference (reference only)",
        "AeroDataBox 日期级时刻参考",
        "AeroDataBox dated schedule reference",
        'status !== "future_schedule_reference"',
        'aerodatabox_schedule: "aeroDataBoxSchedule"',
        "SearchAPI.io · Google Flights",
        'provider === "searchapi_google_flights"',
        '"searchapi_google_flights_booking"',
        "ignav_verified_fares",
        "ignav_verified_booking",
        "fareProviderLabel(fare)",
    ):
        assert fragment in dashboard


def test_offer_detail_frontend_accepts_only_consistent_strict_provider_mappings() -> None:
    detail = TestClient(app).get("/details/offer").text

    for fragment in (
        (
            'serpapi_google_flights:{name:"SerpApi Google Flights",'
            'scheduleSource:"serpapi_google_flights_booking",'
            'dataBasis:"serpapi_booking_confirmed"}'
        ),
        (
            'searchapi_google_flights:{name:"SearchAPI.io Google Flights",'
            'scheduleSource:"searchapi_google_flights_booking",'
            'dataBasis:"searchapi_booking_confirmed"}'
        ),
        (
            'ignav_verified_fares:{name:"Ignav Verified Fares",'
            'scheduleSource:"ignav_verified_booking",'
            'dataBasis:"ignav_verified_booking_confirmed"}'
        ),
        'key(metadata.provider_code)!=="strict_fare_aggregate"',
        "metadataConfirmsFare(metadata,fare)",
        "Array.isArray(metadata.provider_runs)",
        'marker===provider.dataBasis',
        'key(segment.data_basis)===provider.dataBasis',
        "严格模式只展示通过完整行程、正价格与安全购票路径验证的供应商航班",
        "Strict mode shows only provider flights whose complete itinerary",
    ):
        assert fragment in detail
    assert "!===" not in detail
    assert 'key(fare.provider_code)==="serpapi_google_flights"' not in detail
