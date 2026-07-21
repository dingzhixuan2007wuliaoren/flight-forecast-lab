from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).parents[1] / "src" / "flight_forecaster" / "static"


def _page(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_homepage_destination_guides_are_disabled_until_comparison_succeeds() -> None:
    index = _page("index.html")

    attractions = 'id="destination-attractions-link"'
    hotels = 'id="destination-hotels-link"'
    assert attractions in index
    assert hotels in index
    assert index.index(attractions) < index.index('class="rank-toolbar"')
    assert index.index(hotels) < index.index('class="rank-toolbar"')
    assert 'href="/details/attractions?lang=zh" aria-disabled="true" tabindex="-1"' in index
    assert 'href="/details/hotels?lang=zh" aria-disabled="true" tabindex="-1"' in index
    assert 'link.setAttribute("aria-disabled", String(!valid))' in index
    assert 'loadDestinationGuideSummary(data);' in index
    assert "Destination guide enrichment must never interrupt flight results" in index


def test_destination_list_pages_have_bilingual_category_controls() -> None:
    attractions = _page("attractions.html")
    hotels = _page("hotels.html")

    for page in (attractions, hotels):
        assert "中文" in page
        assert "English" in page
        assert 'data-lang="zh"' in page
        assert 'data-lang="en"' in page
        assert 'fetchJson("/v1/destination/places"' in page
        assert 'method:"POST"' in page
        assert 'document.createElement("a")' not in page  # cards use the safe element helper
        assert "innerHTML" not in page
        assert "sessionStorage" not in page
        assert 'id="coverage-note"' in page
        assert "coverage_radius_km" in page
        assert "coverage_status" in page
        assert "coverage_notice" in page
        assert "实际成功查询覆盖半径" in page
        assert "Actual successfully queried coverage radius" in page

    assert '["all","landmark","museum","nature","entertainment","shopping"]' in attractions
    assert '["all","hotel","hostel","guest_house","motel","apartment"]' in hotels
    assert 'kind:"attraction"' in attractions
    assert 'kind:"hotel"' in hotels
    assert '"/details/place?"' in attractions
    assert '"/details/place?"' in hotels


def test_hotel_prices_are_only_requested_after_explicit_submit() -> None:
    hotels = _page("hotels.html")

    assert 'id="price-button"' in hotels
    assert "查询真实价格" in hotels
    assert "Check real prices" in hotels
    assert 'document.getElementById("price-form").addEventListener("submit",queryPrices)' in hotels
    assert 'fetchJson("/v1/destination/hotel-prices"' in hotels
    assert "queryPrices();" not in hotels
    assert 'loadPlaces();' in hotels
    assert 'quota_warning' in hotels
    assert 'next.set("hotel_id",hid)' in hotels
    assert 'kind:priced?"priced_hotel":"hotel"' in hotels
    assert 'next.set("check_in"' in hotels
    assert 'next.set("check_out"' in hotels
    assert 'next.set("adults"' in hotels
    assert 'Number.isFinite(starNumber)&&starNumber>=1&&starNumber<=5' in hotels
    assert 'stars!==undefined?" ★":""' not in hotels


def test_place_detail_is_source_safe_and_renders_airport_routes() -> None:
    detail = _page("place.html")

    assert 'fetchJson("/v1/destination/place-detail"' in detail
    assert 'fetchJson("/v1/destination/hotel-price-detail"' in detail
    assert '["car","bicycle","foot"]' in detail
    assert 'mode==="bike"||mode==="cycling"' in detail
    assert "transport.options" in detail
    assert 'kind==="priced_hotel"' in detail
    assert "nightly_price" in detail
    assert "website_url" in detail
    assert "该城市暂无可验证的公共交通路线数据" in detail
    assert "No verifiable public-transit route data" in detail
    assert 'return url.protocol==="https:"?url.href:""' in detail
    assert 'link.rel="noopener noreferrer"' in detail
    assert "innerHTML" not in detail
    assert "sessionStorage" not in detail
    assert "数据源未提供" in detail
    assert "Not provided by the data source" in detail


def test_destination_fetches_are_bounded_and_language_switches_do_not_refetch() -> None:
    attractions = _page("attractions.html")
    hotels = _page("hotels.html")
    detail = _page("place.html")

    for page in (attractions, hotels, detail):
        assert "new AbortController()" in page
        assert "controller.abort()" in page
        assert "window.clearTimeout(timeout)" in page
        assert "signal:controller.signal" in page

    assert 'if(button.dataset.lang!==language)applyLanguage(button.dataset.lang)' in attractions
    assert 'if(button.dataset.lang!==language)applyLanguage(button.dataset.lang)' in hotels
    assert 'if(button.dataset.lang!==language)applyLanguage(button.dataset.lang)' in detail
    assert "applyLanguage(button.dataset.lang);load" not in attractions
    assert "applyLanguage(button.dataset.lang);load" not in hotels
    assert "applyLanguage(button.dataset.lang);if" not in detail


def test_place_detail_localizes_categories_and_renders_complete_hotel_evidence() -> None:
    attractions = _page("attractions.html")
    hotels = _page("hotels.html")
    detail = _page("place.html")

    assert 'language==="en"&&place&&place.name_en' in attractions
    assert 'language==="en"&&place&&place.name_en' in hotels
    assert 'language==="en"&&place&&place.name_en' in detail
    for key in ("landmark", "museum", "nature", "entertainment", "shopping"):
        assert f'{key}:"' in detail
    for key in ("hotel", "hostel", "guest_house", "motel", "apartment"):
        assert f'{key}:"' in detail
    assert 'language==="zh"?tr("transitUnavailable")' in detail
    assert 'id="hotel-class"' in detail
    assert 'id="review-count"' in detail
    assert 'id="amenities"' in detail
    assert 'id="free-cancellation"' in detail
    assert 'id="nightly-price"' in detail
    assert 'id="total-price"' in detail
    for field in (
        "hotel_class",
        "stars",
        "review_count",
        "amenities",
        "free_cancellation",
        "nightly_price",
        "total_price",
    ):
        assert field in detail


def test_place_detail_never_coerces_missing_numeric_evidence_to_zero() -> None:
    detail = _page("place.html")

    assert 'input===null||input===undefined' in detail
    assert 'typeof input==="string"&&input.trim()===""' in detail
    assert 'typeof input!=="number"&&typeof input!=="string"' in detail
    assert 'minutes===null?tr("missing")' in detail
    assert 'reviewCount===null?tr("missing")' in detail
    assert 'number===null||!/^[A-Z]{3}$/.test(code)' in detail


def test_city_objects_are_rendered_without_object_stringification() -> None:
    for name in ("index.html", "attractions.html", "hotels.html", "place.html"):
        page = _page(name)
        assert "display_name" in page

    index = _page("index.html")
    assert "destinationCityName(payload.city" in index
    assert "String(payload.city)" not in index
