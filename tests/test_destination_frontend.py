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
    assert "limit:300" in attractions
    assert "description_en" in attractions
    assert "description_zh" in attractions
    assert "ratingsUnavailable" in attractions
    assert "rating.platform_en" in attractions
    assert "ratings.forEach" in attractions
    assert "ratings.slice(0,3)" not in attractions
    assert "query_parts_succeeded" in attractions
    assert "query_parts_total" in attractions
    assert "provider_truncated" in attractions
    assert "Completed all " in attractions
    assert "not a complete city inventory" in attractions
    assert "gaps in geographic coverage" in attractions
    assert "requestTimeoutMs=65000" in attractions
    assert "requestTimeoutMs=65000" in hotels
    assert 'kind:"hotel"' in hotels
    assert '"/details/place?"' in attractions
    assert '"/details/place?"' in hotels
    assert "航班结果不受影响" not in hotels
    assert "Flight results are unaffected" not in hotels
    assert "航班结果不受影响" not in attractions
    assert "Flight results are unaffected" not in attractions
    assert "destination_quota_exhausted" in attractions
    assert "destination_rate_limited" in attractions
    assert "destination_timeout" in attractions
    assert "detail.provider_attempts" in attractions
    assert "destination_quota_exhausted" in hotels
    assert "destination_rate_limited" in hotels
    assert "destination_timeout" in hotels
    assert "placeFailureText(error)" in hotels


def test_destination_back_controls_start_in_the_page_top_left() -> None:
    for name in ("attractions.html", "hotels.html", "place.html"):
        page = _page(name)
        top_left = page.index('class="top-left"')
        back = page.index('class="back"', top_left)
        brand = page.index('class="brand"', top_left)
        language = page.index('class="language"', top_left)
        assert top_left < back < brand < language


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
    assert "priceFailureText(error)" in hotels
    for code in (
        "not_configured",
        "authentication_failed",
        "quota_exhausted",
        "rate_limited",
        "provider_processing",
        "provider_error",
        "provider_unavailable",
        "response_invalid",
    ):
        assert code in hotels
    assert "providerRunSummary" in hotels
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
    assert "transit.legs" in detail
    assert "line_name" in detail
    assert "intermediate_stops" in detail
    assert "coverage_status" in detail
    assert "source_url" in detail
    assert "transit_departure_at" in detail
    assert (
        'transit.departure_time_basis==="user_supplied"&&transitDepartureAt'
        ")return transitDepartureAt"
    ) in detail
    assert (
        "transitClock(leg.departure_at,leg.departure_time_label,leg.from_timezone)"
        in detail
    )
    assert "serpapi_google_maps_directions" in detail
    assert "googleMapsSource" in detail
    assert "openstreetmap_overpass_transit_reference" in detail
    assert "route_reference_only" in detail
    assert "route_references" in detail
    assert "renderTransitReferences" in detail
    assert "reference.source_url" in detail
    assert "reference.stops" in detail
    assert "reference.operator" in detail
    assert "reference.network" in detail
    assert "附近真实公共交通线路参考" in detail
    assert "route references only" in detail
    assert (
        "transitClock(leg.arrival_at,leg.arrival_time_label,leg.to_timezone)"
        in detail
    )
    assert 'kind==="priced_hotel"' in detail
    assert "nightly_price" in detail
    assert "website_url" in detail
    assert "该城市暂无可验证的公共交通路线数据" in detail
    assert "No verifiable public-transit route data" in detail
    assert 'url.protocol==="https:"' in detail
    assert "!privateHostname(url.hostname)" in detail
    assert 'link.rel="noopener noreferrer"' in detail
    assert "innerHTML" not in detail
    assert "sessionStorage" not in detail
    assert "数据源未提供" in detail
    assert "Not provided by the data source" in detail
    assert 'id="attraction-review-section"' in detail
    assert "place.ratings_status" in detail
    assert "rating.source_url" in detail
    assert "publicRatingUnavailable" in detail
    assert "ratingProviderUnavailable" in detail
    assert "place.description_en" in detail
    assert "place.description_zh" in detail
    assert 'if(includeLiveTransit===true)payload.include_live_transit=true' in detail
    assert "await loadPrice(true)" in detail
    assert "await loadDetail(true)" in detail
    assert "loadPrice(false)" in detail
    assert "loadDetail(false)" in detail
    assert "查询实时公交（使用额度）" in detail
    assert "Check live transit (uses quota)" in detail


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
    assert 'transitTr("noItinerary")' in detail
    assert 'transitTr("providerUnavailable")' in detail
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


def test_hotel_detail_supports_exact_stay_room_and_cross_platform_evidence() -> None:
    detail = _page("place.html")

    assert 'id="stay-form"' in detail
    assert 'id="hotel-check-in"' in detail
    assert 'id="hotel-check-out"' in detail
    assert 'id="hotel-adults"' in detail
    assert 'id="room-grid"' in detail
    assert 'id="review-grid"' in detail
    assert "queryHotelEvidence" in detail
    assert "room_rates" in detail
    assert "room_rates_status" in detail
    assert "review_sources" in detail
    assert "review_sources_status" in detail
    assert "booking_url" in detail
    assert "review_url" in detail
    assert "observed_at" in detail
    assert "酒店名称、坐标和提供商房源标识" in detail
    assert "hotel name, coordinates, and provider property identity" in detail
    assert "系统没有用估算数据替代" in detail
    assert "estimated data was not substituted" in detail
    assert "if(hotelId)payload.hotel_id=hotelId" in detail
    assert "if(placeId)payload.place_id=placeId" in detail
    assert "detailData=priceData;renderDetail(priceData)" in detail
    assert "innerHTML" not in detail


def test_place_detail_renders_sourced_media_and_sparse_rating_sources() -> None:
    attractions = _page("attractions.html")
    detail = _page("place.html")

    assert "place.photos" in attractions
    assert "photo.attribution" in attractions
    assert "coordinatePreview" in attractions
    assert "精确坐标占位预览（不是地图或照片）" in attractions
    assert "Exact-coordinate placeholder (not a map or photo)" in attractions
    assert 'id="place-media-section"' in detail
    assert "place.photos" in detail
    assert "photo.source_page_url" in detail
    assert "photo.license_url" in detail
    assert "openstreetmap.org/export/embed.html" in detail
    assert "精确坐标地图预览（不是景点照片）" in detail
    assert "Exact-coordinate map preview (not an attraction photo)" in detail
    assert "attraction_rating_source_capabilities" in detail
    assert "capabilities.length" in detail
    assert 'item.adapter_status==="active"' in detail
    assert 'item.adapter_status==="catalogued"' in detail
    assert "rating.platform_id" in detail
    assert "ratings.forEach" in detail
    assert "只显示命中项" in detail
    assert "catalogue-only" in detail
    assert "place.check_in_time" in detail
    assert "place.check_out_time" in detail
    assert "place.thumbnail" in detail
    assert "place.images" in detail
    assert "rate.total_before_taxes" in detail
    assert "nightlyBeforeTaxes" in detail
    assert "totalBeforeTaxes" in detail
    assert "!url.username&&!url.password" in detail
    assert 'url.port==="443"' in detail
    assert "privateHostname(url.hostname)" in detail
    assert "innerHTML" not in detail


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
