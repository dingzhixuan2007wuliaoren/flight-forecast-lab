from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from flight_forecaster.airlabs_quota import (
    AirLabsQuotaLedger,
    read_airlabs_quota_snapshot,
)
from flight_forecaster.alternate_fare_providers import (
    read_alternate_provider_quota_snapshot,
)
from flight_forecaster.api import _runtime_provider_status, app
from flight_forecaster.availability import read_serpapi_quota_snapshot
from flight_forecaster.scrapedo_reference import read_scrapedo_quota_snapshot
from flight_forecaster.supplemental_aviation import (
    read_aerodatabox_quota_snapshot,
    read_opensky_quota_snapshot,
)


@pytest.fixture(autouse=True)
def _isolated_provider_status_runtime(monkeypatch, tmp_path) -> None:
    runtime_dir = tmp_path / "artifacts" / "runtime"
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "artifacts" / "demo"))
    monkeypatch.setenv("AIRLABS_USAGE_DB", str(runtime_dir / "airlabs-usage.sqlite3"))
    monkeypatch.setenv(
        "SCRAPE_DO_USAGE_DB", str(runtime_dir / "scrapedo-reference-usage.sqlite3")
    )
    monkeypatch.setenv(
        "SUPPLEMENTAL_AVIATION_USAGE_DB",
        str(runtime_dir / "supplemental-aviation-usage.sqlite3"),
    )


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


def test_hourly_rate_limit_preserves_real_billing_balance(
    monkeypatch, tmp_path
) -> None:
    now = datetime.now(UTC)
    runtime_dir = tmp_path / "artifacts" / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "artifacts" / "demo"))
    with sqlite3.connect(runtime_dir / "serpapi-usage.sqlite3") as connection:
        connection.execute(
            """
            CREATE TABLE serpapi_quota_usage (
                scope TEXT NOT NULL, period_key TEXT NOT NULL, calls INTEGER NOT NULL,
                PRIMARY KEY(scope, period_key)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            "INSERT INTO serpapi_quota_usage VALUES ('billing_cycle', ?, 170)",
            (f"renewal:{(now + timedelta(days=30)).date().isoformat()}",),
        )
    monkeypatch.setenv("SERPAPI_API_KEY", "configured-but-never-returned")
    metadata = SimpleNamespace(
        provider_code="serpapi_google_flights",
        status="rate_limited",
        coverage_status="provider_incomplete",
        quota_limit="hourly",
        monthly_calls_used=12,
        monthly_call_limit=250,
        quota_unit="billing_period_requests",
    )

    providers = {item.code: item for item in _runtime_provider_status(metadata).providers}
    serpapi = providers["serpapi_google_flights"]

    assert (serpapi.active, serpapi.status, serpapi.quota_status) == (
        True,
        "quota_available",
        "available",
    )
    assert (serpapi.quota_used, serpapi.quota_limit, serpapi.quota_remaining) == (
        170,
        250,
        80,
    )
    assert serpapi.temporarily_rate_limited is True


def test_billing_period_rate_limit_can_mark_account_quota_exhausted(monkeypatch) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "configured-but-never-returned")
    metadata = SimpleNamespace(
        provider_code="serpapi_google_flights",
        status="rate_limited",
        coverage_status="provider_incomplete",
        quota_limit="monthly",
        monthly_calls_used=12,
        monthly_call_limit=250,
        quota_unit="billing_period_requests",
    )

    providers = {item.code: item for item in _runtime_provider_status(metadata).providers}
    serpapi = providers["serpapi_google_flights"]

    assert (serpapi.status, serpapi.quota_status, serpapi.quota_remaining) == (
        "quota_exhausted",
        "exhausted",
        0,
    )
    assert serpapi.temporarily_rate_limited is False


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


def test_dashboard_only_shows_per_provider_quota_summary_and_links_to_detail() -> None:
    client = TestClient(app)
    dashboard = client.get("/").text
    details = client.get("/details/providers").text

    for fragment in (
        'id="provider-quota-summary"',
        'href="/details/providers?lang=zh"',
        "总剩余查询额度",
        "Total remaining query allowance",
        'provider.active !== true',
        "hasValue(provider.quota_remaining) ? asNumber(provider.quota_remaining) : null",
        'measurableBases = ["provider_reported", "local_ledger", "provider_and_local_ledger"]',
        "measurableBases.indexOf(basis) < 0",
        "balances.push({",
        "balance.name",
        "providerQuotaUnitLabel(balance.unit)",
        "flight-forecast-provider-status-v1",
    ):
        assert fragment in dashboard
    assert 'id="provider-status-grid"' not in dashboard
    assert 'class="provider-status-card"' not in dashboard
    assert "provider-card-notice" not in dashboard

    for fragment in (
        'id="grid"',
        'fetch("/v1/provider-status"',
        "数据提供商状态",
        "Data-provider status",
        "quota_used",
        "quota_limit",
        "quota_remaining",
        "quota_data_basis",
        "quota_observed_at",
        "quota_reset_at",
        "quota_cost_per_call",
        "provider.can_supply_strict_offers",
        "quotaState(provider.quota_status)",
        "provider.temporarily_rate_limited===true",
        "供应商暂时限流",
        "Provider temporarily rate-limited",
        "可提供严格报价",
        "Can supply strict fares",
        "额度不适用",
        "Quota not applicable",
        "供应商返回的额度快照",
        "Provider-reported quota snapshot",
        "本地安全账本（不是供应商账户余额）",
        "Local safety ledger (not the provider account balance)",
        "仅有已配置的本地硬上限；实际已用与剩余未公布",
        "Configured local hard ceiling only; actual used and remaining are unpublished",
        "供应商未向此接口公开可验证的用量",
        "The provider does not publish verifiable usage to this interface",
        "额度账本暂时不可读取",
        "Quota ledger is temporarily unavailable",
        'if(remaining!==null)quotaCell(quota,t(basis.remaining)',
        'basisKey==="configured_limit_only"',
        'if(cached){render(cached);setState("loading",t("cached"),"cached")}load()',
        "if(!present(value))return null",
        "flight-forecast-provider-status-v1",
        'href="/#provider-status-title"',
    ):
        assert fragment in details
    assert "innerHTML" not in details
    assert "待确认" not in details
    assert 'unknown:"Pending"' not in details
    assert 'else{load()}' not in details
    assert "Math.max(0,limit-used)" not in details
    assert "Math.round(limit) - Math.max" not in dashboard
    assert "groups[unit].remaining" not in dashboard
    assert ".remaining +=" not in dashboard


def test_dashboard_bounds_comparison_wait_and_preserves_request_errors() -> None:
    dashboard = TestClient(app).get("/").text

    for fragment in (
        "var activeComparison = null;",
        "var comparisonTimeoutMs = 150000;",
        "var slowComparisonThresholdMs = 30000;",
        "state.elapsedTimer = window.setInterval(updateLoadingMessage, 1000);",
        "state.controller.abort();",
        "if (activeComparison) {",
        "window.clearTimeout(state.timeoutTimer)",
        "window.clearInterval(state.elapsedTimer)",
        "state.controller = null;",
        "仍在验证报价供应商返回的航班与购票选项",
        "Still verifying provider flights and booking options",
        "比较已超过 150 秒",
        "The comparison exceeded 150 seconds",
    ):
        assert fragment in dashboard
    assert 'setMessage(loading ? "loading"' not in dashboard
    assert "360000" not in dashboard


def test_dashboard_distinguishes_quota_limited_candidate_coverage() -> None:
    dashboard = TestClient(app).get("/").text

    for fragment in (
        (
            'fare_provider_coverage_limited: ["fareProviderCoverageLimitedTitle", '
            '"fareProviderCoverageLimitedBody"]'
        ),
        "本次已找到候选，但每个严格来源最多只验证 6 个",
        "不能据此断言其余候选不可购买",
        "Candidates were found, but each strict source verifies at most six per comparison",
        "this does not prove the remaining candidates are unbookable",
    ):
        assert fragment in dashboard


def test_provider_status_reads_exact_local_quota_ledgers(monkeypatch, tmp_path) -> None:
    now = datetime.now(UTC)
    runtime_dir = tmp_path / "artifacts" / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "artifacts" / "demo"))

    serp_path = runtime_dir / "serpapi-usage.sqlite3"
    with sqlite3.connect(serp_path) as connection:
        connection.execute(
            """
            CREATE TABLE serpapi_quota_usage (
                scope TEXT NOT NULL, period_key TEXT NOT NULL, calls INTEGER NOT NULL,
                PRIMARY KEY(scope, period_key)
            ) WITHOUT ROWID
            """
        )
        renewal = (now + timedelta(days=30)).date().isoformat()
        expired = (now - timedelta(days=1)).date().isoformat()
        connection.executemany(
            "INSERT INTO serpapi_quota_usage VALUES ('billing_cycle', ?, ?)",
            [(f"renewal:{renewal}", 170), (f"renewal:{expired}", 249)],
        )

    alternate_path = runtime_dir / "alternate-provider-usage.sqlite3"
    with sqlite3.connect(alternate_path) as connection:
        connection.execute(
            """
            CREATE TABLE alternate_provider_free_usage (
                provider_code TEXT PRIMARY KEY,
                reserved_calls INTEGER NOT NULL,
                sent_calls INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.executemany(
            "INSERT INTO alternate_provider_free_usage VALUES (?, ?, ?)",
            [
                ("searchapi_google_flights", 100, 82),
                ("ignav_quarantine", 0, 0),
            ],
        )

    scrape_path = runtime_dir / "scrapedo-reference-usage.sqlite3"
    with sqlite3.connect(scrape_path) as connection:
        connection.execute(
            """
            CREATE TABLE scrapedo_reference_usage (
                period_key TEXT PRIMARY KEY, reserved_credits INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO scrapedo_reference_usage VALUES (?, ?)",
            (now.strftime("%Y-%m"), 40),
        )

    airlabs_path = runtime_dir / "airlabs-usage.sqlite3"
    with sqlite3.connect(airlabs_path) as connection:
        connection.execute(
            """
            CREATE TABLE airlabs_monthly_usage (
                period_key TEXT PRIMARY KEY, calls INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE airlabs_account_snapshot (
                period_key TEXT PRIMARY KEY,
                limits_by_month INTEGER NOT NULL,
                limits_total INTEGER NOT NULL,
                remaining INTEGER NOT NULL,
                observed_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO airlabs_monthly_usage VALUES (?, ?)",
            (now.strftime("%Y-%m"), 1),
        )
        connection.execute(
            "INSERT INTO airlabs_account_snapshot VALUES (?, ?, ?, ?, ?)",
            (now.strftime("%Y-%m"), 1_000, 936, 64, now.isoformat()),
        )

    supplemental_path = runtime_dir / "supplemental-aviation-usage.sqlite3"
    with sqlite3.connect(supplemental_path) as connection:
        connection.execute(
            """
            CREATE TABLE supplemental_aviation_usage (
                provider TEXT NOT NULL, period_key TEXT NOT NULL, units INTEGER NOT NULL,
                PRIMARY KEY(provider, period_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE supplemental_provider_quota_windows (
                provider TEXT PRIMARY KEY, period_key TEXT NOT NULL,
                hard_limit INTEGER NOT NULL, remaining INTEGER NOT NULL,
                reset_at TEXT, evidence TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO supplemental_aviation_usage VALUES (?, ?, ?)",
            [
                ("aerodatabox", "lifetime", 20),
                ("opensky", now.strftime("%Y-%m-%d"), 5),
            ],
        )

    for name in (
        "SERPAPI_API_KEY",
        "SEARCHAPI_API_KEY",
        "IGNAV_API_KEY",
        "SCRAPE_DO_API_TOKEN",
        "AIRLABS_API_KEY",
        "AERODATABOX_API_KEY",
    ):
        monkeypatch.setenv(name, f"{name.lower()}-not-returned")
    monkeypatch.setenv("AIRLABS_MONTHLY_CALL_LIMIT", "900")
    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "auto")
    monkeypatch.setenv("IGNAV_STRICT_RELEASE", "1")
    monkeypatch.setenv("IGNAV_FREE_ACCOUNT_ATTESTED", "1")
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    monkeypatch.setenv("OPENSKY_ENABLED", "1")

    providers = {item.code: item for item in _runtime_provider_status().providers}
    expected = {
        "serpapi_google_flights": (170, 250, 80, "provider_and_local_ledger"),
        "searchapi_google_flights": (100, 100, 0, "local_ledger"),
        "ignav_verified_fares": (0, 1_000, 1_000, "local_ledger"),
        "scrape_do_google_flights_reference": (40, 1_000, 960, "local_ledger"),
        "airlabs_reference": (900, 900, 0, "provider_and_local_ledger"),
        "aerodatabox_reference": (20, 600, 580, "local_ledger"),
        "opensky_reference": (5, 400, 395, "local_ledger"),
    }
    for code, values in expected.items():
        provider = providers[code]
        assert (
            provider.quota_used,
            provider.quota_limit,
            provider.quota_remaining,
            provider.quota_data_basis,
        ) == values
        assert provider.quota_observed_at is not None
    # SerpApi exposes a renewal date, not an exact timestamp.  Do not invent a
    # UTC instant that would render as the previous calendar day in some zones.
    assert providers["serpapi_google_flights"].quota_reset_at is None


def test_quota_snapshot_readers_fail_closed_without_creating_files(tmp_path) -> None:
    now = datetime.now(UTC)

    def readers(path):
        return (
            read_serpapi_quota_snapshot(path, hard_limit=250, now=now),
            read_alternate_provider_quota_snapshot(
                path,
                provider_code="searchapi_google_flights",
                hard_limit=100,
                now=now,
            ),
            read_scrapedo_quota_snapshot(path, hard_limit=1_000, now=now),
            read_airlabs_quota_snapshot(path, hard_limit=900, now=now),
            read_aerodatabox_quota_snapshot(path, hard_limit=600, now=now),
            read_opensky_quota_snapshot(path, hard_limit=400, now=now),
        )

    missing_path = tmp_path / "missing.sqlite3"
    assert all(not snapshot.available for snapshot in readers(missing_path))
    assert not missing_path.exists()

    corrupt_path = tmp_path / "corrupt.sqlite3"
    corrupt_path.write_text("not a sqlite database", encoding="utf-8")
    assert all(not snapshot.available for snapshot in readers(corrupt_path))


def test_aerodatabox_reader_prefers_trusted_unexpired_provider_window(tmp_path) -> None:
    now = datetime.now(UTC)
    observed_at = now - timedelta(minutes=5)
    reset_at = now + timedelta(days=7)
    usage_path = tmp_path / "supplemental.sqlite3"
    with sqlite3.connect(usage_path) as connection:
        connection.execute(
            """
            CREATE TABLE supplemental_aviation_usage (
                provider TEXT NOT NULL, period_key TEXT NOT NULL, units INTEGER NOT NULL,
                PRIMARY KEY(provider, period_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE supplemental_provider_quota_windows (
                provider TEXT PRIMARY KEY, period_key TEXT NOT NULL,
                hard_limit INTEGER NOT NULL, remaining INTEGER NOT NULL,
                reset_at TEXT, evidence TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO supplemental_aviation_usage VALUES ('aerodatabox', 'lifetime', 20)"
        )
        connection.execute(
            "INSERT INTO supplemental_provider_quota_windows VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "aerodatabox",
                "provider-window",
                500,
                123,
                reset_at.isoformat(),
                "trusted_headers",
                observed_at.isoformat(),
            ),
        )

    snapshot = read_aerodatabox_quota_snapshot(
        usage_path,
        hard_limit=600,
        now=now,
    )

    assert snapshot.available is True
    assert (snapshot.used, snapshot.limit, snapshot.remaining) == (377, 500, 123)
    assert snapshot.data_basis == "conservative_minimum"
    assert snapshot.observed_at == observed_at
    assert snapshot.reset_at == reset_at


def test_candidate_coverage_limit_does_not_claim_account_quota_exhaustion(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "artifacts" / "demo"))
    monkeypatch.setenv("SERPAPI_API_KEY", "configured-but-never-returned")
    metadata = SimpleNamespace(
        provider_code="serpapi_google_flights",
        status="confirmed_offers",
        coverage_status="quota_limited",
        monthly_calls_used=2,
        monthly_call_limit=250,
        quota_unit="billing_period_requests",
    )

    providers = {item.code: item for item in _runtime_provider_status(metadata).providers}
    serpapi = providers["serpapi_google_flights"]

    assert (serpapi.status, serpapi.quota_status) == ("quota_available", "available")
    assert (serpapi.quota_used, serpapi.quota_remaining) == (2, 248)
    assert serpapi.quota_data_basis == "local_ledger"
    assert serpapi.quota_observed_at is not None


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
