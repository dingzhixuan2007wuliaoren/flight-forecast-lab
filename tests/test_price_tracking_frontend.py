from __future__ import annotations

from pathlib import Path

INDEX_PAGE = (
    Path(__file__).parents[1]
    / "src"
    / "flight_forecaster"
    / "static"
    / "index.html"
).read_text(encoding="utf-8")
RENDER_BLUEPRINT = (
    Path(__file__).parents[1] / "render.yaml"
).read_text(encoding="utf-8")


def test_main_page_exposes_bilingual_price_tracking_with_truthful_boundaries() -> None:
    assert 'id="price-tracking"' in INDEX_PAGE
    assert 'id="tracking-chart"' in INDEX_PAGE
    assert 'id="tracking-about"' in INDEX_PAGE
    assert "关于价格追踪" in INDEX_PAGE
    assert "About price tracking" in INDEX_PAGE
    assert "同航线、同出发日期及同舱位" in INDEX_PAGE
    assert "same route, departure date, and cabin" in INDEX_PAGE
    assert "模型值不会冒充历史" in INDEX_PAGE
    assert "model values are never presented as history" in INDEX_PAGE


def test_main_page_loads_tracking_from_the_lowest_strictly_verified_offer() -> None:
    assert 'fetch("/v1/offer-detail"' in INDEX_PAGE
    assert 'rankedOffers(data, "lowest_price").filter(hasConfirmedFareOffer)[0]' in INDEX_PAGE
    assert "force_refresh: false" in INDEX_PAGE
    assert "renderPriceTrackingChart" in INDEX_PAGE
    assert "historical_market_context" in INDEX_PAGE
    assert "price_curve" in INDEX_PAGE


def test_main_page_keeps_provider_chain_diagnostics_in_empty_state() -> None:
    assert "data.fare_search_metadata.notice" in INDEX_PAGE
    assert "body += \" \" + metadataNotice" in INDEX_PAGE


def test_render_uses_auto_strict_provider_failover_without_committed_secrets() -> None:
    assert "key: FLIGHT_OFFER_PROVIDER\n        value: auto" in RENDER_BLUEPRINT
    assert "key: SERPAPI_API_KEY\n        sync: false" in RENDER_BLUEPRINT
    assert "key: SEARCHAPI_API_KEY\n        sync: false" in RENDER_BLUEPRINT
    assert "key: IGNAV_API_KEY\n        sync: false" in RENDER_BLUEPRINT
    assert "SERPAPI_API_KEY=" not in RENDER_BLUEPRINT
    assert "SEARCHAPI_API_KEY=" not in RENDER_BLUEPRINT
    assert "IGNAV_API_KEY=" not in RENDER_BLUEPRINT
