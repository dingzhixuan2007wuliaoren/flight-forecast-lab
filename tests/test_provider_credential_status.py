from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flight_forecaster.schemas import BilingualText, RuntimeProviderStatusItem

CHECKED_AT = datetime(2026, 8, 8, 15, tzinfo=UTC)


def _provider(**overrides: Any) -> RuntimeProviderStatusItem:
    values: dict[str, Any] = {
        "code": "serpapi_google_flights",
        "display_name": "SerpApi Google Flights",
        "role": "strict_fare",
        "configured": True,
        "active": True,
        "status": "configured",
        "quota_status": "unknown",
        "quota_data_basis": "unpublished",
        "can_supply_strict_offers": True,
        "notice": BilingualText(zh="脱敏状态", en="Sanitized status"),
    }
    values.update(overrides)
    return RuntimeProviderStatusItem(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "configured": False,
            "active": False,
            "status": "not_configured",
            "can_supply_strict_offers": False,
            "credential_state": "missing",
            "transient": False,
        },
        {"credential_state": "plausible", "transient": False},
        {
            "credential_state": "verified",
            "checked_at": CHECKED_AT,
            "http_status": 200,
            "transient": False,
        },
        {
            "credential_state": "invalid",
            "status": "authentication_failed",
            "can_supply_strict_offers": False,
            "checked_at": CHECKED_AT,
            "http_status": 401,
            "exception_type": "CredentialInvalid",
            "transient": False,
        },
        {
            "credential_state": "forbidden",
            "status": "authentication_failed",
            "can_supply_strict_offers": False,
            "checked_at": CHECKED_AT,
            "http_status": 403,
            "exception_type": "AccountForbidden",
            "transient": False,
        },
        {
            "credential_state": "inactive",
            "status": "authentication_failed",
            "can_supply_strict_offers": False,
            "checked_at": CHECKED_AT,
            "http_status": 200,
            "exception_type": "AccountInactive",
            "transient": False,
        },
        {
            "credential_state": "unknown",
            "checked_at": CHECKED_AT,
            "http_status": 503,
            "exception_type": "TransientProviderHttpError",
            "transient": True,
        },
        {
            "credential_state": "unknown",
            "checked_at": CHECKED_AT,
            "http_status": None,
            "exception_type": "TransportError",
            "transient": True,
        },
        {
            "credential_state": "unknown",
            "checked_at": CHECKED_AT,
            "http_status": 200,
            "exception_type": "PayloadError",
            "transient": False,
        },
    ],
)
def test_runtime_provider_accepts_consistent_sanitized_credential_state(
    overrides: dict[str, Any],
) -> None:
    provider = _provider(**overrides)

    dumped = provider.model_dump(mode="json")
    serialized = provider.model_dump_json()
    assert dumped.get("credential_state") == overrides.get("credential_state")
    for forbidden in ("api_key", "token", "fingerprint", "account_id", "account_email"):
        assert forbidden not in serialized.lower()


@pytest.mark.parametrize(
    "overrides",
    [
        {"checked_at": CHECKED_AT},
        {"credential_state": "missing", "transient": False},
        {
            "credential_state": "plausible",
            "checked_at": CHECKED_AT,
            "transient": False,
        },
        {
            "credential_state": "verified",
            "checked_at": CHECKED_AT,
            "http_status": 401,
            "transient": False,
        },
        {
            "credential_state": "invalid",
            "checked_at": CHECKED_AT,
            "http_status": 403,
            "exception_type": "CredentialInvalid",
            "transient": False,
        },
        {
            "credential_state": "invalid",
            "checked_at": CHECKED_AT,
            "http_status": 401,
            "exception_type": "CredentialInvalid",
            "transient": False,
            "status": "configured",
            "can_supply_strict_offers": True,
        },
        {
            "credential_state": "unknown",
            "checked_at": CHECKED_AT,
            "http_status": 503,
            "exception_type": "TransientProviderHttpError",
            "transient": False,
        },
        {
            "credential_state": "unknown",
            "checked_at": CHECKED_AT,
            "http_status": 500,
            "exception_type": "TransportError",
            "transient": True,
        },
        {
            "credential_state": "unknown",
            "checked_at": CHECKED_AT,
            "http_status": 200,
            "exception_type": "NotAllowlisted",
            "transient": False,
        },
    ],
)
def test_runtime_provider_rejects_inconsistent_credential_state(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _provider(**overrides)


def test_runtime_provider_requires_timezone_for_credential_observation() -> None:
    with pytest.raises(ValidationError):
        _provider(
            credential_state="verified",
            checked_at=datetime(2026, 8, 8, 15),
            http_status=200,
            transient=False,
        )


def test_provider_page_renders_only_sanitized_bilingual_credential_status() -> None:
    page = (
        Path(__file__).parents[1]
        / "src"
        / "flight_forecaster"
        / "static"
        / "providers.html"
    ).read_text(encoding="utf-8")

    for fragment in (
        'snapshotKey="flight-forecast-provider-status-v2"',
        'legacySnapshotKey="flight-forecast-provider-status-v1"',
        "sessionStorage.removeItem(legacySnapshotKey)",
        "credentialState(provider.credential_state)",
        "provider.checked_at",
        "provider.http_status",
        "provider.exception_type",
        "provider.transient===true",
        "凭据已免费验证",
        "Credential verified for free",
        "暂时性预检故障；不会单独使服务就绪检查失败",
        "Transient preflight failure; this alone does not fail service readiness",
    ):
        assert fragment in page
    for forbidden in (
        "provider.api_key",
        "provider.token",
        "provider.credential_fingerprint",
        "provider.account_id",
        "provider.account_email",
    ):
        assert forbidden not in page
    assert "innerHTML" not in page
