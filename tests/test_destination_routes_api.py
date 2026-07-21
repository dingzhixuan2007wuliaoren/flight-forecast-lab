from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from fastapi.testclient import TestClient

from flight_forecaster import destination_routes
from flight_forecaster.api import app
from flight_forecaster.destination_guide import (
    DestinationCity,
    DestinationPlace,
    DestinationPlaceDetail,
    DestinationPlaceList,
    DestinationTransport,
    DestinationTransportOption,
)
from flight_forecaster.hotel_prices import HotelPriceOffer, HotelPriceSearchResult

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)


def _city() -> DestinationCity:
    return DestinationCity(
        destination_airport="YYZ",
        airport_name="Toronto Pearson International",
        served_city="Toronto",
        city_query="Toronto, CA",
        name="Toronto",
        country_code="CA",
        latitude=43.6532,
        longitude=-79.3832,
        scope="served_city",
        scope_notice="Verified served-city centre.",
        source="ourairports_municipality+nominatim",
    )


def _place() -> DestinationPlace:
    return DestinationPlace(
        place_id="osm_attraction_node_123",
        kind="attraction",
        category="landmark",
        name="Example Tower",
        name_en="Example Tower",
        address="1 Example Street, Toronto",
        description="A source-backed landmark.",
        website="https://example.org/tower",
        phone=None,
        opening_hours="10:00-18:00",
        stars=None,
        latitude=43.6426,
        longitude=-79.3871,
        distance_from_city_center_km=1.3,
        source_url="https://www.openstreetmap.org/node/123",
    )


def _transport(latitude: float = 43.6426, longitude: float = -79.3871) -> DestinationTransport:
    available = []
    for mode, minutes, distance in (
        ("car", 28, 27.1),
        ("bike", 75, 28.4),
        ("foot", 320, 26.9),
    ):
        available.append(
            DestinationTransportOption(
                mode=mode,
                status="available",
                distance_km=distance,
                duration_minutes=minutes,
                duration_basis="estimated_route_no_live_traffic",
                notice="Static OSRM estimate without live traffic.",
                data_source="routing_openstreetmap_de_osrm",
                observed_at=NOW,
                expires_at=NOW + timedelta(hours=24),
            )
        )
    available.append(
        DestinationTransportOption(
            mode="public_transit",
            status="unavailable",
            notice="No verified regional timetable is configured.",
            data_source="open_transit_coverage_unavailable",
            observed_at=NOW,
            expires_at=NOW + timedelta(hours=24),
        )
    )
    return DestinationTransport(
        destination_airport="YYZ",
        airport_name="Toronto Pearson International",
        origin_latitude=43.6777,
        origin_longitude=-79.6248,
        destination_latitude=latitude,
        destination_longitude=longitude,
        options=tuple(available),
    )


class _Guide:
    def __init__(self) -> None:
        self.route_coordinates: tuple[float, float] | None = None

    def resolve_city(self, destination: str) -> DestinationCity:
        assert destination.upper() == "YYZ"
        return _city()

    def list_places(
        self,
        destination: str,
        kind: str,
        category: str = "all",
        limit: int = 30,
    ) -> DestinationPlaceList:
        assert destination.upper() == "YYZ"
        assert kind == "attraction"
        assert category == "all"
        assert limit == 30
        return DestinationPlaceList(
            city=_city(),
            kind="attraction",
            category="all",
            places=(_place(),),
            result_count=1,
            fetched_at=NOW,
            expires_at=NOW + timedelta(hours=24),
            coverage_radius_km=30,
            coverage_status="complete",
            coverage_reason="full_radius_queried",
            partial=False,
            coverage_notice={
                "zh": "已成功查询完整 30 公里范围。",
                "en": "The full 30 km radius was queried successfully.",
            },
        )

    def get_place_detail(self, destination: str, place_id: str) -> DestinationPlaceDetail:
        assert destination.upper() == "YYZ"
        assert place_id == _place().place_id
        return DestinationPlaceDetail(
            city=_city(),
            place=_place(),
            transport=_transport(),
        )

    def get_routes(
        self,
        destination: str,
        latitude: float,
        longitude: float,
    ) -> DestinationTransport:
        assert destination == "YYZ"
        self.route_coordinates = (latitude, longitude)
        return _transport(latitude, longitude)


def _offer() -> HotelPriceOffer:
    return HotelPriceOffer(
        hotel_id="gh_0123456789abcdef0123456789abcdef",
        name="Verified Hotel",
        property_type="Hotel",
        latitude=43.6501,
        longitude=-79.3801,
        description="Provider-returned hotel description.",
        hotel_class=4,
        rating=4.4,
        review_count=321,
        nightly_price=210.0,
        total_price=237.3,
        currency="USD",
        price_source="Example booking source",
        free_cancellation=True,
        amenities=("Wi-Fi",),
        website_url="https://example.org/book-hotel",
        observed_at=NOW,
    )


class _HotelProvider:
    def __init__(self) -> None:
        self.explicit: bool | None = None
        self.detail_calls = 0

    def search(
        self,
        *args: object,
        explicit: bool = False,
        **kwargs: object,
    ) -> HotelPriceSearchResult:
        self.explicit = explicit
        assert args[:2] == ("Toronto, CA", "YYZ")
        assert kwargs["language"] == "zh-cn"
        return HotelPriceSearchResult(
            offers=(_offer(),),
            status="available",
            observed_at=NOW,
            cache_hit=False,
            calls_reserved=1,
            quota_monthly_used=171,
            quota_monthly_limit=250,
            quota_hourly_used=1,
            quota_hourly_limit=50,
        )

    def detail(self, *args: object, **kwargs: object) -> HotelPriceOffer:
        self.detail_calls += 1
        assert args[0] == _offer().hotel_id
        return _offer()


def _install_fakes(monkeypatch):
    guide = _Guide()
    hotel = _HotelProvider()
    monkeypatch.setattr(destination_routes, "get_destination_guide_service", lambda: guide)
    monkeypatch.setattr(destination_routes, "get_hotel_price_provider", lambda: hotel)
    return guide, hotel


def test_destination_pages_are_served() -> None:
    client = TestClient(app)
    for path in ("/details/attractions", "/details/hotels", "/details/place"):
        response = client.get(path)
        assert response.status_code == 200
        assert "Flight Forecast Lab" in response.text


def test_places_and_detail_expose_source_backed_routes(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    client = TestClient(app)
    places = client.post(
        "/v1/destination/places",
        json={"destination": "YYZ", "kind": "attraction", "language": "zh"},
    )
    assert places.status_code == 200
    assert places.json()["places"][0]["place_id"] == _place().place_id
    assert places.json()["city"]["name"] == "Toronto"
    assert places.json()["coverage_radius_km"] == 30
    assert places.json()["coverage_status"] == "complete"
    assert places.json()["partial"] is False
    assert places.json()["coverage_notice"]["en"].startswith("The full 30 km")

    detail = client.post(
        "/v1/destination/place-detail",
        json={
            "destination": "YYZ",
            "kind": "attraction",
            "place_id": _place().place_id,
            "language": "en",
        },
    )
    assert detail.status_code == 200
    body = detail.json()
    assert [route["mode"] for route in body["routes"]] == [
        "car",
        "bike",
        "foot",
        "public_transit",
    ]
    assert body["routes"][3]["duration_minutes"] is None
    assert body["transit_notice"]


def test_hotel_price_search_is_explicit_and_exposes_safe_quote(monkeypatch) -> None:
    _, hotel = _install_fakes(monkeypatch)
    response = TestClient(app).post(
        "/v1/destination/hotel-prices",
        json={
            "destination": "YYZ",
            "check_in": "2026-07-25",
            "check_out": "2026-07-26",
            "adults": 1,
            "language": "zh-cn",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert hotel.explicit is True
    assert body["calls_reserved"] == 1
    assert body["offers"][0]["formatted_price"] == "US$210.00 / 晚"
    assert body["offers"][0]["hotel_id"] == _offer().hotel_id
    assert "property_token" not in response.text
    assert "api_key" not in response.text
    assert "serpapi.com" not in response.text.lower()


def test_hotel_price_detail_uses_cached_offer_coordinates_for_routes(monkeypatch) -> None:
    guide, hotel = _install_fakes(monkeypatch)
    response = TestClient(app).post(
        "/v1/destination/hotel-price-detail",
        json={
            "destination": "YYZ",
            "hotel_id": _offer().hotel_id,
            "place_id": None,
            "check_in": "2026-07-25",
            "check_out": "2026-07-26",
            "adults": 1,
            "language": "en",
        },
    )
    assert response.status_code == 200
    assert hotel.detail_calls == 1
    assert guide.route_coordinates == (_offer().latitude, _offer().longitude)
    body = response.json()
    assert body["place"]["place_id"] is None
    assert body["offer"]["hotel_id"] == _offer().hotel_id
    assert len(body["routes"]) == 4


def test_destination_requests_fail_closed_on_invalid_cross_kind_id(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    response = TestClient(app).post(
        "/v1/destination/place-detail",
        json={
            "destination": "YYZ",
            "kind": "hotel",
            "place_id": _place().place_id,
            "language": "en",
        },
    )
    assert response.status_code == 422


def test_hotel_dates_and_adults_are_validated_before_provider(monkeypatch) -> None:
    _, hotel = _install_fakes(monkeypatch)
    response = TestClient(app).post(
        "/v1/destination/hotel-prices",
        json={
            "destination": "YYZ",
            "check_in": date(2026, 7, 25).isoformat(),
            "check_out": date(2026, 7, 25).isoformat(),
            "adults": 9,
            "language": "en",
        },
    )
    assert response.status_code == 422
    assert hotel.explicit is None
