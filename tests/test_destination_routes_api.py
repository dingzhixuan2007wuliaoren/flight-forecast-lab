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


def _valid_transit_departure() -> datetime:
    return (datetime.now(UTC) + timedelta(days=1)).replace(second=0, microsecond=0)


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


def _hotel_place() -> DestinationPlace:
    return DestinationPlace(
        place_id="osm_hotel_node_456",
        kind="hotel",
        category="hotel",
        name="Verified Hotel",
        name_en="Verified Hotel",
        address="2 Verified Street, Toronto",
        description="An OpenStreetMap hotel record.",
        website="https://example.org/verified-hotel",
        phone="+1 416 555 0100",
        opening_hours=None,
        stars=4,
        latitude=43.6501,
        longitude=-79.3801,
        distance_from_city_center_km=0.8,
        source_url="https://www.openstreetmap.org/node/456",
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
        self.transit_departure_at: datetime | None = None
        self.include_live_transit: bool | None = None

    def resolve_city(self, destination: str) -> DestinationCity:
        assert destination.upper() == "YYZ"
        return _city()

    def list_places(
        self,
        destination: str,
        kind: str,
        category: str = "all",
        limit: int = 300,
    ) -> DestinationPlaceList:
        assert destination.upper() == "YYZ"
        assert kind == "attraction"
        assert category == "all"
        assert limit == 300
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

    def get_place_detail(
        self,
        destination: str,
        place_id: str,
        *,
        transit_departure_at: datetime | None = None,
        include_live_transit: bool = False,
    ) -> DestinationPlaceDetail:
        assert destination.upper() == "YYZ"
        place = _hotel_place() if place_id == _hotel_place().place_id else _place()
        assert place_id == place.place_id
        self.transit_departure_at = transit_departure_at
        self.include_live_transit = include_live_transit
        return DestinationPlaceDetail(
            city=_city(),
            place=place,
            transport=_transport(place.latitude, place.longitude),
        )

    def get_routes(
        self,
        destination: str,
        latitude: float,
        longitude: float,
        *,
        transit_departure_at: datetime | None = None,
        include_live_transit: bool = False,
    ) -> DestinationTransport:
        assert destination == "YYZ"
        self.route_coordinates = (latitude, longitude)
        self.transit_departure_at = transit_departure_at
        self.include_live_transit = include_live_transit
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
        address="100 Provider Street, Toronto, ON",
        phone="+1 416 555 0199",
        check_in_time="3:00 PM",
        check_out_time="11:00 AM",
        thumbnail="https://images.example.org/hotel-thumb.jpg",
        images=(
            "https://images.example.org/hotel-1.jpg",
            "https://images.example.org/hotel-2.jpg",
        ),
    )


class _HotelProvider:
    def __init__(self) -> None:
        self.explicit: bool | None = None
        self.detail_calls = 0
        self.exact_detail_calls = 0

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
        assert kwargs["explicit"] is True
        return _offer()

    def exact_property_detail(
        self,
        hotel_names: tuple[str, ...],
        latitude: float,
        longitude: float,
        *args: object,
        **kwargs: object,
    ) -> HotelPriceOffer:
        self.exact_detail_calls += 1
        assert hotel_names == ("Verified Hotel",)
        assert (latitude, longitude) == (
            _hotel_place().latitude,
            _hotel_place().longitude,
        )
        assert args[:2] == ("Toronto, CA", "YYZ")
        assert kwargs["explicit"] is True
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
    guide, _ = _install_fakes(monkeypatch)
    client = TestClient(app)
    transit_departure = _valid_transit_departure()
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
    assert places.json()["source"] == "openstreetmap_overpass+wikimedia"
    assert places.json()["available_result_count"] == 1
    assert places.json()["result_limit"] == 300

    detail = client.post(
        "/v1/destination/place-detail",
        json={
            "destination": "YYZ",
            "kind": "attraction",
            "place_id": _place().place_id,
            "language": "en",
            "transit_departure_at": transit_departure.isoformat(),
            "include_live_transit": True,
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
    assert guide.transit_departure_at == transit_departure
    assert guide.include_live_transit is True
    assert "open_transit_coverage_unavailable" in body["source"]
    assert "transitous_motis" not in body["source"]


def test_destination_detail_rejects_naive_transit_departure_time(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    response = TestClient(app).post(
        "/v1/destination/place-detail",
        json={
            "destination": "YYZ",
            "kind": "attraction",
            "place_id": _place().place_id,
            "language": "en",
            "transit_departure_at": "2026-07-21T13:00:00",
        },
    )

    assert response.status_code == 422


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
    assert guide.include_live_transit is False
    body = response.json()
    assert body["place"]["place_id"] is None
    assert body["place"]["address"] == _offer().address
    assert body["place"]["phone"] == _offer().phone
    assert body["place"]["hours"] == "Check-in 3:00 PM · Check-out 11:00 AM"
    assert body["place"]["check_in_time"] == "3:00 PM"
    assert body["place"]["check_out_time"] == "11:00 AM"
    assert body["place"]["thumbnail"] == _offer().thumbnail
    assert body["place"]["images"] == list(_offer().images)
    assert body["offer"]["hotel_id"] == _offer().hotel_id
    assert body["offer"]["address"] == _offer().address
    assert len(body["routes"]) == 4


def test_hotel_price_detail_resolves_exact_osm_hotel_and_reuses_routes(monkeypatch) -> None:
    guide, hotel = _install_fakes(monkeypatch)
    transit_departure = _valid_transit_departure()
    response = TestClient(app).post(
        "/v1/destination/hotel-price-detail",
        json={
            "destination": "YYZ",
            "place_id": _hotel_place().place_id,
            "check_in": "2026-07-25",
            "check_out": "2026-07-26",
            "adults": 1,
            "language": "en",
            "transit_departure_at": transit_departure.isoformat(),
            "include_live_transit": True,
        },
    )

    assert response.status_code == 200
    assert hotel.exact_detail_calls == 1
    assert hotel.detail_calls == 0
    assert guide.route_coordinates is None
    assert guide.transit_departure_at == transit_departure
    assert guide.include_live_transit is True
    body = response.json()
    assert body["place"]["place_id"] == _hotel_place().place_id
    assert body["place"]["address"] == _hotel_place().address
    assert body["place"]["phone"] == _hotel_place().phone
    assert body["offer"]["hotel_id"] == _offer().hotel_id
    assert "openstreetmap_overpass" in body["source"]


def test_hotel_detail_rejects_out_of_range_transit_time_before_provider_calls(
    monkeypatch,
) -> None:
    _, hotel = _install_fakes(monkeypatch)
    invalid_departure = (datetime.now(UTC) + timedelta(days=371)).replace(
        microsecond=0
    )

    response = TestClient(app).post(
        "/v1/destination/hotel-price-detail",
        json={
            "destination": "YYZ",
            "hotel_id": _offer().hotel_id,
            "check_in": "2026-07-25",
            "check_out": "2026-07-26",
            "adults": 1,
            "language": "en",
            "transit_departure_at": invalid_departure.isoformat(),
        },
    )

    assert response.status_code == 422
    assert hotel.detail_calls == 0
    assert hotel.exact_detail_calls == 0


def test_hotel_price_detail_requires_exactly_one_identity(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    client = TestClient(app)
    common = {
        "destination": "YYZ",
        "check_in": "2026-07-25",
        "check_out": "2026-07-26",
        "adults": 1,
        "language": "en",
    }

    assert client.post("/v1/destination/hotel-price-detail", json=common).status_code == 422
    assert (
        client.post(
            "/v1/destination/hotel-price-detail",
            json={
                **common,
                "hotel_id": _offer().hotel_id,
                "place_id": _hotel_place().place_id,
            },
        ).status_code
        == 422
    )


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
