from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier, Event

import pytest

from flight_forecaster import api as api_module
from flight_forecaster.availability import CredentialPreflightStatus


class _FakePredictionService:
    instances: list[_FakePredictionService] = []
    preflight_completed = Event()

    def __init__(self, model_dir) -> None:
        self.model_dir = model_dir
        self.preflight_calls = 0
        self.credential_status = CredentialPreflightStatus(
            state="plausible",
            checked_at=None,
            http_status=None,
            exception_type=None,
            transient=False,
        )
        self.__class__.instances.append(self)

    def preflight_strict_provider_credentials(self) -> CredentialPreflightStatus:
        self.preflight_calls += 1
        self.credential_status = CredentialPreflightStatus(
            state="invalid",
            checked_at=datetime(2026, 8, 8, 21, 0, tzinfo=UTC),
            http_status=401,
            exception_type="CredentialInvalid",
            transient=False,
        )
        self.__class__.preflight_completed.set()
        return self.credential_status

    def serpapi_credential_preflight_status(self) -> CredentialPreflightStatus:
        return self.credential_status


@pytest.fixture(autouse=True)
def _clear_cached_service() -> None:
    api_module.get_service.cache_clear()
    _FakePredictionService.instances.clear()
    _FakePredictionService.preflight_completed.clear()
    yield
    api_module.get_service.cache_clear()


def test_enabled_startup_preflight_is_free_sanitized_and_runs_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    preflight_thread_completed = Event()
    run_preflight = api_module._run_strict_provider_credential_preflight

    def run_preflight_and_signal(service: _FakePredictionService) -> None:
        try:
            run_preflight(service)
        finally:
            preflight_thread_completed.set()

    monkeypatch.setenv("SERPAPI_CREDENTIAL_PREFLIGHT_ENABLED", "1")
    monkeypatch.setenv("SERPAPI_API_KEY", "must-never-appear-in-logs")
    monkeypatch.setattr(api_module, "PredictionService", _FakePredictionService)
    monkeypatch.setattr(
        api_module,
        "_run_strict_provider_credential_preflight",
        run_preflight_and_signal,
    )

    with caplog.at_level(logging.INFO, logger="flight_forecaster.api"):
        first = api_module.get_service()
        second = api_module.get_service()
        assert preflight_thread_completed.wait(timeout=1)

    assert first is second
    assert first.preflight_calls == 1
    assert "state=invalid" in caplog.text
    assert "http_status=401" in caplog.text
    assert "exception_type=CredentialInvalid" in caplog.text
    assert "must-never-appear-in-logs" not in caplog.text


def test_startup_preflight_is_opt_in_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERPAPI_CREDENTIAL_PREFLIGHT_ENABLED", raising=False)
    monkeypatch.setattr(api_module, "PredictionService", _FakePredictionService)

    service = api_module.get_service()

    assert service.preflight_calls == 0


def test_runtime_provider_status_exposes_only_the_sanitized_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "must-never-appear-in-provider-status"
    monkeypatch.setenv("SERPAPI_API_KEY", secret)
    monkeypatch.setenv("FLIGHT_OFFER_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_CREDENTIAL_PREFLIGHT_ENABLED", "1")
    monkeypatch.setattr(api_module, "PredictionService", _FakePredictionService)

    api_module.get_service()
    assert _FakePredictionService.preflight_completed.wait(timeout=1)
    payload = api_module._runtime_provider_status().model_dump(mode="json")
    serpapi = next(
        provider
        for provider in payload["providers"]
        if provider["code"] == "serpapi_google_flights"
    )

    assert serpapi["credential_state"] == "invalid"
    assert serpapi["status"] == "authentication_failed"
    assert serpapi["can_supply_strict_offers"] is False
    assert serpapi["http_status"] == 401
    assert serpapi["exception_type"] == "CredentialInvalid"
    assert serpapi["transient"] is False
    assert serpapi["checked_at"] == "2026-08-08T21:00:00Z"
    assert secret not in str(payload)


def test_startup_preflight_never_blocks_service_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    class _BlockingPredictionService(_FakePredictionService):
        def preflight_strict_provider_credentials(self) -> CredentialPreflightStatus:
            started.set()
            assert release.wait(timeout=2)
            return super().preflight_strict_provider_credentials()

    monkeypatch.setenv("SERPAPI_CREDENTIAL_PREFLIGHT_ENABLED", "1")
    monkeypatch.setattr(api_module, "PredictionService", _BlockingPredictionService)

    service = api_module.get_service()

    assert isinstance(service, _BlockingPredictionService)
    assert started.wait(timeout=1)
    assert service.preflight_calls == 0
    release.set()
    assert _BlockingPredictionService.preflight_completed.wait(timeout=1)
    assert service.preflight_calls == 1


def test_concurrent_cold_start_builds_and_preflights_exactly_one_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callers = 8
    barrier = Barrier(callers)
    monkeypatch.setenv("SERPAPI_CREDENTIAL_PREFLIGHT_ENABLED", "1")
    monkeypatch.setattr(api_module, "PredictionService", _FakePredictionService)

    def load_service() -> object:
        barrier.wait(timeout=2)
        return api_module.get_service()

    with ThreadPoolExecutor(max_workers=callers) as pool:
        services = tuple(pool.map(lambda _index: load_service(), range(callers)))

    assert _FakePredictionService.preflight_completed.wait(timeout=1)
    assert len({id(service) for service in services}) == 1
    assert len(_FakePredictionService.instances) == 1
    assert _FakePredictionService.instances[0].preflight_calls == 1
