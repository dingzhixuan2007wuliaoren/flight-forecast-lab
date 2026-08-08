from pathlib import Path

INDEX_PAGE = (
    Path(__file__).parents[1]
    / "src"
    / "flight_forecaster"
    / "static"
    / "index.html"
).read_text(encoding="utf-8")
PROVIDERS_PAGE = (
    Path(__file__).parents[1]
    / "src"
    / "flight_forecaster"
    / "static"
    / "providers.html"
).read_text(encoding="utf-8")


def test_dashboard_discards_legacy_comparison_state_after_refresh_or_deploy() -> None:
    assert 'var sessionKey = "flight-forecast-session-v2";' in INDEX_PAGE
    assert 'var legacySessionKey = "flight-forecast-session-v1";' in INDEX_PAGE
    assert "window.sessionStorage.removeItem(legacySessionKey)" in INDEX_PAGE
    assert "comparison: isProcessingComparison(lastData) ? null : lastData" not in INDEX_PAGE
    assert "state.comparison" not in INDEX_PAGE
    assert "isProcessingComparison" not in INDEX_PAGE


def test_dashboard_session_v2_only_keeps_safe_form_and_language_state() -> None:
    save_start = INDEX_PAGE.index("function saveSessionState()")
    save_end = INDEX_PAGE.index("function restoreSessionState()", save_start)
    save_session = INDEX_PAGE[save_start:save_end]

    assert "form: formState()," in save_session
    assert "language: currentLanguage" in save_session
    assert "comparison" not in save_session
    assert "ranking" not in save_session
    assert "cabin" not in save_session


def test_dashboard_compare_request_bypasses_browser_http_cache() -> None:
    compare_start = INDEX_PAGE.index('var response = await fetch("/v1/compare"')
    compare_end = INDEX_PAGE.index("var data = {};", compare_start)
    compare_request = INDEX_PAGE[compare_start:compare_end]

    assert 'method: "POST"' in compare_request
    assert 'cache: "no-store"' in compare_request


def test_provider_status_snapshot_writer_and_reader_use_the_same_version() -> None:
    snapshot_key = "flight-forecast-provider-status-v2"

    assert f'window.sessionStorage.setItem("{snapshot_key}"' in INDEX_PAGE
    assert f'snapshotKey="{snapshot_key}"' in PROVIDERS_PAGE
    assert 'legacySnapshotKey="flight-forecast-provider-status-v1"' in PROVIDERS_PAGE
    assert "sessionStorage.removeItem(legacySnapshotKey)" in PROVIDERS_PAGE
