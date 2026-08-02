from pathlib import Path

import pytest

from flight_forecaster.data import generate_demo_ontime_data, generate_demo_price_data
from flight_forecaster.training import train_models


@pytest.fixture(autouse=True)
def isolate_external_api_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unit tests must never consume credentials configured on the developer machine."""

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
    monkeypatch.delenv("SCRAPPA_API_KEY", raising=False)
    monkeypatch.delenv("SCRAPPA_MONTHLY_LIMIT", raising=False)
    monkeypatch.delenv("SEARCHAPI_LIFETIME_LIMIT", raising=False)
    monkeypatch.delenv("SEARCHAPI_MONTHLY_LIMIT", raising=False)
    monkeypatch.delenv("IGNAV_API_KEY", raising=False)
    monkeypatch.delenv("IGNAV_RELEASE_VERIFIED", raising=False)
    monkeypatch.delenv("IGNAV_FREE_ACCOUNT_ATTESTED", raising=False)
    monkeypatch.delenv("IGNAV_LIFETIME_LIMIT", raising=False)
    monkeypatch.delenv("AIRLABS_API_KEY", raising=False)
    monkeypatch.setenv("AIRLABS_MONTHLY_CALL_LIMIT", "1000")
    monkeypatch.setenv("AIRLABS_USAGE_DB", str(tmp_path / "airlabs-usage.sqlite3"))
    monkeypatch.delenv("AERODATABOX_API_KEY", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OPENSKY_DAILY_CREDIT_LIMIT", raising=False)
    monkeypatch.delenv("AERODATABOX_MONTHLY_UNIT_LIMIT", raising=False)
    monkeypatch.delenv("AERODATABOX_SCHEDULE_REQUEST_UNITS", raising=False)
    monkeypatch.delenv("SUPPLEMENTAL_AVIATION_USAGE_DB", raising=False)
    monkeypatch.delenv("FLIGHT_OFFER_PROVIDER", raising=False)
    monkeypatch.delenv("FLIGHT_OFFER_PROVIDERS", raising=False)
    for name in (
        "SCRAPE_DO_API_TOKEN",
        "SCRAPEDO_API_TOKEN",
        "SCRAPE_DO_TOKEN",
        "SCRAPE_DO_API_KEY",
        "SCRAPEDO_API_KEY",
        "SCRAPE_DO_MONTHLY_CREDIT_LIMIT",
        "SCRAPEDO_MONTHLY_CREDIT_LIMIT",
        "SCRAPE_DO_USAGE_DB",
    ):
        monkeypatch.delenv(name, raising=False)
    # Default application behaviour uses anonymous OpenSky. Unit tests opt out
    # explicitly so no service construction can consume a real public quota.
    monkeypatch.setenv("OPENSKY_ENABLED", "0")


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("models")
    train_models(
        generate_demo_price_data(rows=1_500, seed=42),
        generate_demo_ontime_data(rows=2_000, seed=43),
        output,
        data_mode="pytest_synthetic",
        random_state=42,
    )
    return output
