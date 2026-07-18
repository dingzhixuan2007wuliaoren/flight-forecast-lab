from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from flight_forecaster.airlabs_quota import (
    airlabs_usage_path,
    configured_airlabs_monthly_limit,
    read_airlabs_quota_snapshot,
)
from flight_forecaster.alternate_fare_providers import (
    read_alternate_provider_quota_snapshot,
)
from flight_forecaster.availability import read_serpapi_quota_snapshot
from flight_forecaster.quota_status import QuotaLedgerSnapshot
from flight_forecaster.route_info import RouteLookupError
from flight_forecaster.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    ContextDetailRequest,
    NewsDetailResponse,
    OfferDetailRequest,
    OfferDetailResponse,
    OnTimePrediction,
    OnTimeRequest,
    PricePrediction,
    PriceRequest,
    RuntimeProviderStatusItem,
    RuntimeProviderStatusResponse,
    WeatherDetailResponse,
)
from flight_forecaster.scrapedo_reference import read_scrapedo_quota_snapshot
from flight_forecaster.service import OfferNotFoundError, PredictionService
from flight_forecaster.supplemental_aviation import (
    read_aerodatabox_quota_snapshot,
    read_opensky_quota_snapshot,
    supplemental_usage_path,
)
from flight_forecaster.training import ARTIFACT_FILENAME

app = FastAPI(
    title="Flight Forecast Lab",
    version="0.2.0",
    description=(
        "Bilingual global airline/cabin model comparisons with automatic weather, "
        "airport-operations, and current-news context."
    ),
)


def model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", "artifacts/demo"))


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService(model_dir())


def _service_or_503() -> PredictionService:
    try:
        return get_service()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _credential_present(*names: str) -> bool:
    return any(bool(os.getenv(name, "").strip()) for name in names)


def _ignav_strict_release_active() -> bool:
    return bool(
        os.getenv("FLIGHT_OFFER_PROVIDER", "none").strip().lower() != "ignav_quarantine"
        and _credential_present("IGNAV_API_KEY", "IGNAV_TOKEN")
        and _environment_enabled("IGNAV_STRICT_RELEASE", default=False)
        and _environment_enabled("IGNAV_FREE_ACCOUNT_ATTESTED", default=False)
    )


def _selected_provider_codes() -> set[str]:
    ignav_code = (
        "ignav_verified_fares"
        if _ignav_strict_release_active()
        else "ignav_quarantine"
    )
    aliases = {
        "serpapi": {"serpapi_google_flights"},
        "serpapi_google_flights": {"serpapi_google_flights"},
        "searchapi": {"searchapi_google_flights"},
        "searchapi_io": {"searchapi_google_flights"},
        "searchapi_google_flights": {"searchapi_google_flights"},
        "serpapi_searchapi": {
            "serpapi_google_flights",
            "searchapi_google_flights",
        },
        "auto": {
            "serpapi_google_flights",
            "searchapi_google_flights",
            ignav_code,
        },
        "ignav": {ignav_code},
        "ignav_quarantine": {"ignav_quarantine"},
        "ignav_verified_fares": {ignav_code},
    }
    raw = os.getenv("FLIGHT_OFFER_PROVIDER", "none")
    return aliases.get(raw.strip().lower(), set())


def _serpapi_quota_limit(configured: bool) -> int | None:
    if not configured:
        return None
    raw = os.getenv("SERPAPI_MONTHLY_LIMIT", "250").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 250
    return min(max(value, 1), 250)


def _bounded_quota_limit(
    names: tuple[str, ...],
    *,
    default: int,
    maximum: int,
) -> int:
    raw = next((os.getenv(name, "").strip() for name in names if os.getenv(name, "").strip()), "")
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return min(value, maximum) if value > 0 else default


def _searchapi_quota_limit() -> int:
    # SearchAPI's 100-request signup allocation is a one-time account allowance,
    # not a monthly or billing-period renewal.
    return _bounded_quota_limit(
        ("SEARCHAPI_LIFETIME_LIMIT", "SEARCHAPI_MONTHLY_LIMIT"),
        default=100,
        maximum=100,
    )


def _ignav_quota_limit() -> int:
    return _bounded_quota_limit(
        ("IGNAV_LIFETIME_LIMIT",),
        default=1_000,
        maximum=1_000,
    )


def _scrape_do_quota_limit() -> int:
    return _bounded_quota_limit(
        ("SCRAPE_DO_MONTHLY_CREDIT_LIMIT", "SCRAPEDO_MONTHLY_CREDIT_LIMIT"),
        default=1_000,
        maximum=1_000,
    )


def _environment_enabled(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _airlabs_quota_status(
    *, configured: bool
) -> tuple[bool, QuotaLedgerSnapshot, int | None]:
    """Return credential-safe AirLabs state without making a provider request."""

    if not configured:
        return False, QuotaLedgerSnapshot.unavailable(), None
    limit = configured_airlabs_monthly_limit()
    if limit is None:
        # Credentials alone never enable a potentially overage-bearing call.
        return False, QuotaLedgerSnapshot.unavailable(), None

    usage_path = airlabs_usage_path(
        model_dir().parent / "runtime" / "airlabs-usage.sqlite3"
    )
    snapshot = read_airlabs_quota_snapshot(
        usage_path,
        hard_limit=limit,
        now=datetime.now(UTC),
    )
    return True, snapshot, limit


def _runtime_provider_status(
    fare_metadata: object | None = None,
) -> RuntimeProviderStatusResponse:
    external_context_enabled = _environment_enabled(
        "EXTERNAL_CONTEXT_ENABLED", default=True
    )
    selected = _selected_provider_codes()
    serpapi_configured = _credential_present("SERPAPI_API_KEY")
    searchapi_configured = _credential_present("SEARCHAPI_API_KEY")
    ignav_configured = _credential_present("IGNAV_API_KEY", "IGNAV_TOKEN")
    ignav_released = _ignav_strict_release_active()
    scrape_do_configured = _credential_present(
        "SCRAPE_DO_API_TOKEN",
        "SCRAPEDO_API_TOKEN",
        "SCRAPE_DO_TOKEN",
        "SCRAPE_DO_API_KEY",
        "SCRAPEDO_API_KEY",
    )
    airlabs_configured = _credential_present("AIRLABS_API_KEY")
    (
        airlabs_active,
        airlabs_snapshot,
        airlabs_quota_limit,
    ) = _airlabs_quota_status(configured=airlabs_configured)
    airlabs_active = airlabs_active and external_context_enabled
    aerodatabox_configured = _credential_present("AERODATABOX_API_KEY")
    aerodatabox_active = aerodatabox_configured and external_context_enabled
    opensky_registered = _credential_present("OPENSKY_CLIENT_ID") and _credential_present(
        "OPENSKY_CLIENT_SECRET"
    )
    opensky_active = (
        _environment_enabled("OPENSKY_ENABLED") and external_context_enabled
    )
    opensky_limit = _bounded_quota_limit(
        ("OPENSKY_DAILY_CREDIT_LIMIT",),
        default=4_000 if opensky_registered else 400,
        maximum=4_000 if opensky_registered else 400,
    )
    now = datetime.now(UTC)
    runtime_dir = model_dir().parent / "runtime"
    serpapi_limit = _serpapi_quota_limit(serpapi_configured)
    serpapi_snapshot = (
        read_serpapi_quota_snapshot(
            runtime_dir / "serpapi-usage.sqlite3",
            hard_limit=serpapi_limit,
            now=now,
        )
        if serpapi_limit is not None
        else QuotaLedgerSnapshot.unavailable()
    )
    alternate_path = runtime_dir / "alternate-provider-usage.sqlite3"
    searchapi_limit = _searchapi_quota_limit()
    searchapi_snapshot = (
        read_alternate_provider_quota_snapshot(
            alternate_path,
            provider_code="searchapi_google_flights",
            hard_limit=searchapi_limit,
            now=now,
        )
        if searchapi_configured
        else QuotaLedgerSnapshot.unavailable()
    )
    ignav_limit = _ignav_quota_limit()
    ignav_snapshot = (
        read_alternate_provider_quota_snapshot(
            alternate_path,
            provider_code="ignav_quarantine",
            hard_limit=ignav_limit,
            now=now,
        )
        if ignav_configured and ignav_released
        else QuotaLedgerSnapshot.unavailable()
    )
    scrape_do_limit = _scrape_do_quota_limit()
    scrape_do_path = Path(
        os.getenv(
            "SCRAPE_DO_USAGE_DB",
            str(runtime_dir / "scrapedo-reference-usage.sqlite3"),
        )
    )
    scrape_do_snapshot = (
        read_scrapedo_quota_snapshot(
            scrape_do_path,
            hard_limit=scrape_do_limit,
            now=now,
        )
        if scrape_do_configured
        else QuotaLedgerSnapshot.unavailable()
    )
    supplemental_path = supplemental_usage_path(
        runtime_dir / "supplemental-aviation-usage.sqlite3"
    )
    aerodatabox_limit = _bounded_quota_limit(
        ("AERODATABOX_MONTHLY_UNIT_LIMIT",),
        default=600,
        maximum=600,
    )
    aerodatabox_snapshot = (
        read_aerodatabox_quota_snapshot(
            supplemental_path,
            hard_limit=aerodatabox_limit,
            now=now,
        )
        if aerodatabox_active
        else QuotaLedgerSnapshot.unavailable()
    )
    opensky_snapshot = (
        read_opensky_quota_snapshot(
            supplemental_path,
            hard_limit=opensky_limit,
            now=now,
        )
        if opensky_active
        else QuotaLedgerSnapshot.unavailable()
    )

    strict_notice = {
        "zh": (
            "只有具有正价格、完整连续航段、真实航班号与时刻，并通过安全购票路径验证的报价"
            "才能进入严格列表。额度耗尽的提供商会自动停止；隔离或仅参考来源绝不用于补造航班。"
        ),
        "en": (
            "Only offers with a positive fare, complete continuous segments, real flight numbers "
            "and times, and a verified safe booking path may enter the strict list. Providers stop "
            "when quota is exhausted; quarantined and reference-only sources never fabricate "
            "fallback flights."
        ),
    }

    def strict_item(
        *,
        code: str,
        name: str,
        configured: bool,
        eligible: bool,
        quarantined: bool = False,
        quota_limit: int | None = None,
        quota_unit: str | None = None,
        quota_snapshot: QuotaLedgerSnapshot | None = None,
        notice: dict[str, str] | None = None,
    ) -> RuntimeProviderStatusItem:
        if quarantined:
            status = "quarantined"
            quota_status = "not_applicable"
        else:
            status = "configured" if configured else "not_configured"
            quota_status = "unknown" if configured else "not_applicable"
        snapshot = quota_snapshot or QuotaLedgerSnapshot.unavailable()
        quota_data_basis = "not_applicable" if (not configured or quarantined) else (
            {
                "local_hard_limit": "local_ledger",
                "provider_snapshot": "provider_reported",
                "conservative_minimum": "provider_and_local_ledger",
            }.get(snapshot.data_basis, "unavailable")
            if snapshot.available
            else "unavailable"
        )
        if configured and not quarantined and snapshot.available:
            exhausted = snapshot.remaining == 0
            quota_status = "exhausted" if exhausted else "available"
            status = "quota_exhausted" if exhausted else "quota_available"
        if notice is None and quarantined:
            notice = {
                "zh": "该来源处于隔离状态；即使已配置，也不能向严格航班列表提供报价。",
                "en": (
                    "This source is quarantined and cannot supply the strict flight list even "
                    "when configured."
                ),
            }
        elif notice is None:
            notice = {
                "zh": "配置只表示凭据存在；每个报价仍须独立通过严格行程、价格与购票路径验证。",
                "en": (
                    "Configured means credentials are present; every offer must still pass "
                    "independent itinerary, fare, and booking-path verification."
                ),
            }
        return RuntimeProviderStatusItem(
            code=code,
            display_name=name,
            role="strict_fare" if eligible else "strict_fare_candidate",
            configured=configured,
            active=code in selected,
            status=status,
            quota_status=quota_status,
            quota_used=(snapshot.used if configured and not quarantined else None),
            quota_limit=quota_limit,
            quota_remaining=(snapshot.remaining if configured and not quarantined else None),
            quota_data_basis=quota_data_basis,
            quota_observed_at=(
                snapshot.observed_at if configured and not quarantined else None
            ),
            quota_reset_at=(snapshot.reset_at if configured and not quarantined else None),
            quota_unit=quota_unit if quota_limit is not None else None,
            can_supply_strict_offers=eligible,
            notice=notice,
        )

    def reference_quota_fields(
        snapshot: QuotaLedgerSnapshot,
        *,
        applicable: bool,
        configured_limit: int | None,
    ) -> dict[str, object]:
        if not applicable:
            return {
                "quota_status": "not_applicable",
                "quota_limit": configured_limit,
                "quota_data_basis": "not_applicable",
            }
        if not snapshot.available:
            return {
                "quota_status": "unknown",
                "quota_limit": configured_limit,
                "quota_data_basis": "unavailable",
            }
        basis = {
            "local_hard_limit": "local_ledger",
            "provider_snapshot": "provider_reported",
            "conservative_minimum": "provider_and_local_ledger",
        }.get(snapshot.data_basis, "unavailable")
        return {
            "quota_status": "exhausted" if snapshot.remaining == 0 else "available",
            "quota_used": snapshot.used,
            "quota_limit": snapshot.limit,
            "quota_remaining": snapshot.remaining,
            "quota_data_basis": basis,
            "quota_observed_at": snapshot.observed_at,
            "quota_reset_at": snapshot.reset_at,
        }

    providers = [
        strict_item(
            code="serpapi_google_flights",
            name="SerpApi · Google Flights",
            configured=serpapi_configured,
            eligible=True,
            quota_limit=serpapi_limit,
            quota_unit="billing_period_requests",
            quota_snapshot=serpapi_snapshot,
            notice={
                "zh": (
                    "严格报价源；本地硬上限按提供商账户结算周期执行，不按自然月重置。"
                    "每个报价仍须通过完整行程、价格和购票路径验证。"
                ),
                "en": (
                    "Strict fare source. Its local hard stop follows the provider account billing "
                    "period, not a calendar month; every offer still requires full itinerary, "
                    "fare, and booking-path verification."
                ),
            },
        ),
        strict_item(
            code="searchapi_google_flights",
            name="SearchAPI.io · Google Flights",
            configured=searchapi_configured,
            eligible=True,
            quota_limit=searchapi_limit,
            quota_unit="lifetime_requests",
            quota_snapshot=searchapi_snapshot,
            notice={
                "zh": (
                    "严格后备报价源；免费注册额度为账户一次性 100 次请求，不会按月恢复。"
                    "每个报价仍须通过完整行程、价格和购票路径验证。"
                ),
                "en": (
                    "Strict fallback fare source. The free signup allocation is 100 lifetime "
                    "account requests and does not renew monthly; every offer still requires full "
                    "itinerary, fare, and booking-path verification."
                ),
            },
        ),
        strict_item(
            code=("ignav_verified_fares" if ignav_released else "ignav_quarantine"),
            name=("Ignav Verified Fares" if ignav_released else "Ignav (strict quarantine)"),
            configured=ignav_configured,
            eligible=ignav_released,
            quarantined=not ignav_released,
            quota_limit=ignav_limit,
            quota_unit="lifetime_requests",
            quota_snapshot=ignav_snapshot,
            notice={
                "zh": (
                    "已完成显式严格发布与无付费方式确认；该独立已验证身份可参与严格链路，"
                    "但账户一次性免费额度仍最多为 1,000 次请求。"
                    if ignav_released
                    else (
                        "实验性来源保持隔离；账户一次性免费额度最多 1,000 次请求，"
                        "只有完成受控实时验证、无付费方式确认并明确解除隔离后才可进入严格链路。"
                    )
                ),
                "en": (
                    "Explicit strict release and no-payment-method attestation are complete. "
                    "This separately identified verified source may participate in the strict "
                    "chain, while its one-time free allowance remains capped at 1,000 requests."
                    if ignav_released
                    else (
                        "This experimental source remains quarantined. Its one-time free "
                        "allowance is at most 1,000 requests; it cannot enter the strict chain "
                        "until controlled validation, no-payment-method attestation, and explicit "
                        "release are all complete."
                    )
                ),
            },
        ),
        RuntimeProviderStatusItem(
            code="scrape_do_google_flights_reference",
            display_name="Scrape.do · Google Flights",
            role="reference_only",
            configured=scrape_do_configured,
            active=scrape_do_configured,
            status="reference_only",
            **reference_quota_fields(
                scrape_do_snapshot,
                applicable=scrape_do_configured,
                configured_limit=scrape_do_limit,
            ),
            quota_unit="monthly_credits",
            quota_cost_per_call=10,
            can_supply_strict_offers=False,
            notice={
                "zh": (
                    "仅提供聚合覆盖参考，绝不单独生成严格航班。免费硬上限为每月 1,000 点数；"
                    "每次参考查询预留 10 点数，瞬时失败最多受控重试一次。"
                ),
                "en": (
                    "Aggregate coverage reference only; it never creates strict flights. The free "
                    "hard stop is 1,000 credits per month, with 10 credits reserved per reference "
                    "call and at most one controlled retry after a transient failure."
                ),
            },
        ),
        RuntimeProviderStatusItem(
            code="airlabs_reference",
            display_name="AirLabs",
            role="reference_only",
            configured=airlabs_configured,
            active=airlabs_active,
            status="reference_only",
            **reference_quota_fields(
                airlabs_snapshot,
                applicable=airlabs_active,
                configured_limit=airlabs_quota_limit,
            ),
            quota_unit=(
                "billing_period_requests" if airlabs_quota_limit is not None else None
            ),
            can_supply_strict_offers=False,
            notice=(
                {
                    "zh": (
                        "已配置凭据，但缺少 1–1000 范围内有效的 "
                        "AIRLABS_MONTHLY_CALL_LIMIT；所有 AirLabs 网络调用均已关闭并拒绝执行。"
                    ),
                    "en": (
                        "Credentials are configured, but AirLabs fails closed until "
                        "AIRLABS_MONTHLY_CALL_LIMIT is a valid value from 1 to 1,000; all "
                        "AirLabs network calls remain disabled."
                    ),
                }
                if airlabs_configured and airlabs_quota_limit is None
                else {
                    "zh": (
                        "仅提供机场运行与时刻参考，不能证明票价、库存或可购买。所有调用共用"
                        "本地硬上限；状态页只读取已有账本中的脱敏计数，不会发起网络请求。"
                    ),
                    "en": (
                        "Airport-operations and timetable reference only; it cannot prove fare, "
                        "inventory, or bookability. All calls share a local hard stop, and this "
                        "status reads only sanitized existing-ledger counts without network I/O."
                    ),
                }
            ),
        ),
        RuntimeProviderStatusItem(
            code="aerodatabox_reference",
            display_name="AeroDataBox",
            role="reference_only",
            configured=aerodatabox_configured,
            active=aerodatabox_active,
            status="reference_only",
            **reference_quota_fields(
                aerodatabox_snapshot,
                applicable=aerodatabox_active,
                configured_limit=(
                    aerodatabox_limit if aerodatabox_configured else None
                ),
            ),
            quota_unit="provider_managed" if aerodatabox_configured else None,
            can_supply_strict_offers=False,
            notice={
                "zh": (
                    "仅提供日期级时刻参考，不提供已验证的当前票价或购票路径。可信的 "
                    "RapidAPI 免费计划 reset 信息定义供应商周期；没有可信重置信号时，"
                    "执行同一安装生命周期 600 API 单位硬墙。"
                ),
                "en": (
                    "Dated schedule reference only; it does not provide a verified current fare "
                    "or booking path. Trusted RapidAPI free-plan reset evidence defines a provider "
                    "cycle; without a trustworthy reset signal, a 600-unit installation-lifetime "
                    "hard wall applies."
                ),
            },
        ),
        RuntimeProviderStatusItem(
            code="opensky_reference",
            display_name="OpenSky Network",
            role="reference_only",
            configured=opensky_registered,
            active=opensky_active,
            status="reference_only",
            **reference_quota_fields(
                opensky_snapshot,
                applicable=opensky_active,
                configured_limit=opensky_limit,
            ),
            quota_unit="daily_credits",
            can_supply_strict_offers=False,
            notice={
                "zh": (
                    "仅提供当前航迹密度运行参考，不能证明未来机票可购。即使没有凭据，匿名模式也默认启用，"
                    "本地硬上限为每日 400 API 点数；OAuth 凭据完整时最多为每日 4,000 点数。"
                ),
                "en": (
                    "Current trajectory-density reference only; it cannot prove a future ticket "
                    "is bookable. Anonymous mode is active by default without credentials with a "
                    "400-API-credit daily hard stop; complete OAuth credentials raise the ceiling "
                    "to at most 4,000 per day."
                ),
            },
        ),
    ]

    metadata = fare_metadata
    top_level_provider_code = str(
        getattr(metadata, "provider_code", "") or ""
    ).strip().lower()
    if top_level_provider_code == "strict_fare_aggregate":
        raw_provider_runs = getattr(metadata, "provider_runs", ()) or ()
        metadata_runs = (
            tuple(raw_provider_runs)
            if isinstance(raw_provider_runs, (list, tuple))
            else ()
        )
    else:
        metadata_runs = (metadata,)

    provider_indexes = {
        provider.code: index
        for index, provider in enumerate(providers)
        if provider.role != "reference_only"
    }
    for run in metadata_runs:
        provider_code = str(getattr(run, "provider_code", "") or "").strip().lower()
        provider_code = {
            "serpapi": "serpapi_google_flights",
            "searchapi": "searchapi_google_flights",
            "ignav": "ignav_verified_fares" if ignav_released else "ignav_quarantine",
        }.get(provider_code, provider_code)
        if provider_code == "none" or not provider_code:
            provider_code = next(iter(selected), "") if len(selected) == 1 else ""
        provider_index = provider_indexes.get(provider_code)
        if provider_index is None:
            continue

        provider = providers[provider_index]
        fare_status = str(getattr(run, "status", "") or "").strip().lower()
        rate_limit_scope = str(getattr(run, "quota_limit", "") or "").strip().lower()
        temporarily_rate_limited = fare_status == "rate_limited" and rate_limit_scope not in {
            "monthly",
            "lifetime",
        }
        raw_used = getattr(run, "monthly_calls_used", None)
        raw_limit = getattr(run, "monthly_call_limit", None)
        run_used = (
            raw_used
            if isinstance(raw_used, int) and not isinstance(raw_used, bool) and raw_used >= 0
            else None
        )
        run_limit = (
            raw_limit
            if isinstance(raw_limit, int)
            and not isinstance(raw_limit, bool)
            and raw_limit > 0
            else None
        )
        effective_limit = provider.quota_limit
        if run_limit is not None:
            effective_limit = (
                min(effective_limit, run_limit)
                if effective_limit is not None
                else run_limit
            )
        effective_used = provider.quota_used
        if run_used is not None:
            effective_used = (
                max(effective_used, run_used)
                if effective_used is not None
                else run_used
            )
        if effective_used is not None and effective_limit is not None:
            effective_used = min(effective_used, effective_limit)

        effective_remaining = provider.quota_remaining
        if effective_limit is not None and effective_used is not None:
            computed_remaining = max(0, effective_limit - effective_used)
            effective_remaining = (
                min(effective_remaining, computed_remaining)
                if effective_remaining is not None
                else computed_remaining
            )
        explicitly_exhausted = fare_status == "budget_exhausted" or (
            fare_status == "rate_limited"
            and rate_limit_scope in {"monthly", "lifetime"}
        )
        if explicitly_exhausted and effective_limit is not None:
            effective_remaining = 0
        has_measurement = (
            effective_used is not None
            and effective_limit is not None
            and effective_remaining is not None
            and (provider.quota_observed_at is not None or run_used is not None)
        )
        measurably_exhausted = has_measurement and effective_remaining == 0
        exhausted = has_measurement and (explicitly_exhausted or measurably_exhausted)
        quota_available = (
            has_measurement and not exhausted and effective_remaining is not None
            and effective_remaining > 0
        )
        update: dict[str, object] = {
            "active": True,
            "temporarily_rate_limited": temporarily_rate_limited,
        }
        if exhausted:
            update.update(status="quota_exhausted", quota_status="exhausted")
        elif quota_available and provider.status != "quarantined":
            update.update(status="quota_available", quota_status="available")
        if effective_used is not None:
            update["quota_used"] = effective_used
        if effective_limit is not None:
            update["quota_limit"] = effective_limit
        if effective_remaining is not None:
            update["quota_remaining"] = effective_remaining
        if run_used is not None:
            update["quota_data_basis"] = (
                "provider_and_local_ledger"
                if provider.quota_data_basis
                in {"provider_reported", "provider_and_local_ledger"}
                else "local_ledger"
            )
            update["quota_observed_at"] = now
        if run_limit is not None:
            metadata_quota_unit = getattr(run, "quota_unit", None)
            update["quota_unit"] = metadata_quota_unit or (
                "lifetime_requests"
                if provider.code
                in {
                    "searchapi_google_flights",
                    "ignav_quarantine",
                    "ignav_verified_fares",
                }
                else "billing_period_requests"
            )
        providers[provider_index] = RuntimeProviderStatusItem.model_validate(
            {**provider.model_dump(), **update}
        )

    return RuntimeProviderStatusResponse(
        generated_at=now,
        strict_policy=strict_notice,
        providers=providers,
    )


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/details/weather", include_in_schema=False)
def weather_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "weather.html")


@app.get("/details/news", include_in_schema=False)
def news_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "news.html")


@app.get("/details/offer", include_in_schema=False)
def offer_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "offer.html")


@app.get("/details/providers", include_in_schema=False)
def provider_details_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "providers.html")


@app.get("/health")
def health() -> dict[str, str | bool]:
    artifact_exists = (model_dir() / ARTIFACT_FILENAME).exists()
    service: PredictionService | None = None
    if artifact_exists:
        try:
            service = get_service()
        except (FileNotFoundError, ValueError):
            return {
                "status": "model_not_ready",
                "model_ready": False,
                "fare_provider_configured": False,
                "fare_provider_environment": "disabled",
            }
    return {
        "status": "ok" if artifact_exists else "model_not_trained",
        "model_ready": artifact_exists,
        "fare_provider_configured": bool(
            service is not None and service.flight_offer_provider.configured
        ),
        "fare_provider_environment": (
            service.flight_offer_provider.environment if service is not None else "disabled"
        ),
    }


@app.get("/v1/model-info")
def model_info() -> dict:
    return _service_or_503().model_info()


@app.get("/v1/provider-status", response_model=RuntimeProviderStatusResponse)
def provider_status() -> RuntimeProviderStatusResponse:
    """Return credential-safe provider roles and runtime availability metadata."""

    return _runtime_provider_status()


@app.post("/v1/predict/price", response_model=PricePrediction)
def predict_price(request: PriceRequest) -> PricePrediction:
    try:
        return _service_or_503().predict_price(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/predict/on-time", response_model=OnTimePrediction)
def predict_ontime(request: OnTimeRequest) -> OnTimePrediction:
    try:
        return _service_or_503().predict_ontime(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/compare", response_model=ComparisonResponse)
def compare_flights(request: ComparisonRequest) -> ComparisonResponse:
    try:
        response = _service_or_503().compare(request)
        runtime = _runtime_provider_status(response.fare_search_metadata)
        return response.model_copy(update={"provider_statuses": runtime.providers})
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/offer-detail", response_model=OfferDetailResponse)
def offer_detail(request: OfferDetailRequest) -> OfferDetailResponse:
    try:
        return _service_or_503().offer_detail(request)
    except OfferNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/context/weather-detail", response_model=WeatherDetailResponse)
def weather_detail(request: ContextDetailRequest) -> WeatherDetailResponse:
    try:
        return _service_or_503().weather_detail(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/context/news-detail", response_model=NewsDetailResponse)
def news_detail(request: ContextDetailRequest) -> NewsDetailResponse:
    try:
        return _service_or_503().news_detail(request)
    except RouteLookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
