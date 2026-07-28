from pathlib import Path

OFFER_PAGE = (
    Path(__file__).parents[1] / "src" / "flight_forecaster" / "static" / "offer.html"
)


def test_offer_curve_is_visibly_anchored_to_the_verified_fare() -> None:
    page = OFFER_PAGE.read_text(encoding="utf-8")

    assert 'id="curve-live-badge"' in page
    assert 'id="curve-calibration"' in page
    assert "verified_fare_anchored_synthetic_trajectory" in page
    assert "log1p_offset_to_verified_fare" in page
    assert "Math.abs(anchorPrice-liveFare)<=.01" in page
    assert "Math.abs(points[0].estimated_price_usd-anchorPrice)<=.01" in page
    assert 'class:isAnchor?"curve-anchor-point":"curve-point"' in page
    assert "当前报价锚定预测" in page
    assert "Current-fare-anchored forecast" in page


def test_offer_curve_keeps_raw_model_start_as_transparent_metadata_only() -> None:
    page = OFFER_PAGE.read_text(encoding="utf-8")

    assert "curve.raw_model_start_price_usd" in page
    assert 'data-raw-model-start-price-usd' in page
    assert "原始值仅作为透明的模型元数据" in page
    assert "transparent model metadata only" in page
    assert 'id="curve-gap"' not in page
    assert "data-absolute-gap-usd" not in page
    assert "data-percentage-gap" not in page
    assert "curve-live-line" not in page
    assert "curve-live-label" not in page


def test_offer_curve_renders_distinct_route_date_cabin_market_history() -> None:
    page = OFFER_PAGE.read_text(encoding="utf-8")

    assert "data.historical_market_context" in page
    assert "route_departure_date_cabin_market" in page
    assert "market_context_not_selected_offer_history" in page
    assert 'class:"curve-history-line"' in page
    assert 'class:"curve-history-point"' in page
    assert 'id="historical-data"' in page
    assert 'id="historical-data-rows"' in page
    assert "并非此航班或购票渠道的历史" in page
    assert "not history for this flight or booking channel" in page
    assert "系统不会把模型值冒充历史报价" in page
    assert "Model values are never presented as historical quotes" in page


def test_offer_curve_keeps_every_daily_forecast_row() -> None:
    page = OFFER_PAGE.read_text(encoding="utf-8")

    assert "points.forEach(function(point){var row=append(rows" in page
    assert "sampleCurve(points,120)" in page
    assert 'data-i18n="curveDataTitle"' in page
