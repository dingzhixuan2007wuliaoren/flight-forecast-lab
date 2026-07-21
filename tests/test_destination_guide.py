from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

import flight_forecaster.destination_guide as destination_guide
from flight_forecaster.destination_guide import (
    OVERPASS_ENDPOINTS,
    OVERPASS_FALLBACK_URL,
    OVERPASS_OPERATION_BUDGET_SECONDS,
    OVERPASS_RADIUS_METERS,
    OVERPASS_REQUEST_TIMEOUT_SECONDS,
    OVERPASS_URL,
    ROUTING_REQUEST_TIMEOUT_SECONDS,
    BoundedJsonHttpClient,
    DestinationAirportNotFound,
    DestinationDataUnavailable,
    DestinationGuideService,
    DestinationPlace,
    DestinationPlaceNotFound,
    DestinationValidationError,
    OurAirportsMunicipalityResolver,
    _assert_allowed_outbound_url,
)
from flight_forecaster.route_info import Airport


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_modes: set[str] = set()
        self.fail_search = False
        self.search_results: list[dict[str, Any]] = [
            {
                "lat": "43.6532",
                "lon": "-79.3832",
                "name": "Toronto",
                "display_name": "Toronto, Ontario, Canada",
                "addresstype": "city",
                "address": {"city": "Toronto", "country_code": "ca"},
            }
        ]
        self.reverse_result: dict[str, Any] = {
            "address": {"city": "Mississauga", "country_code": "ca"}
        }
        self.attractions = _attraction_elements()
        self.hotels = _hotel_elements()
        self.overpass_payloads: list[Any] | None = None
        self._overpass_response_index = 0
        self.overpass_call_hook: Callable[[], None] | None = None

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: Any = None,
        data: Any = None,
        headers: Any = None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "data": data,
                "headers": headers,
                "timeout": timeout_seconds,
                "max_bytes": max_response_bytes,
            }
        )
        if url.endswith("/search"):
            if self.fail_search:
                raise DestinationDataUnavailable("offline")
            return self.search_results
        if url.endswith("/reverse"):
            return self.reverse_result
        if url in OVERPASS_ENDPOINTS:
            if self.overpass_call_hook is not None:
                self.overpass_call_hook()
            if self.overpass_payloads is not None:
                index = min(
                    self._overpass_response_index,
                    len(self.overpass_payloads) - 1,
                )
                self._overpass_response_index += 1
                payload = self.overpass_payloads[index]
                if isinstance(payload, DestinationDataUnavailable):
                    raise payload
                return payload
            query = str((data or {}).get("data") or "")
            elements = self.hotels if "guest_house" in query else self.attractions
            return {"elements": elements}
        if "routing.openstreetmap.de" in url:
            mode = next(
                value for value in ("car", "bike", "foot") if f"routed-{value}" in url
            )
            if mode in self.fail_modes:
                raise DestinationDataUnavailable("route unavailable")
            distance, duration = {
                "car": (25_500.0, 1_800.0),
                "bike": (26_200.0, 5_400.0),
                "foot": (24_800.0, 18_000.0),
            }[mode]
            return {"code": "Ok", "routes": [{"distance": distance, "duration": duration}]}
        raise AssertionError(f"unexpected fake URL: {url}")


def _attraction_elements() -> list[dict[str, Any]]:
    return [
        {
            "type": "node",
            "id": 100,
            "lat": 43.6677,
            "lon": -79.3948,
            "tags": {
                "name": "Verified Museum",
                "name:en": "Verified Museum",
                "tourism": "museum",
                "addr:housenumber": "100",
                "addr:street": "Museum Road",
                "addr:city": "Toronto",
                "website": "https://museum.example.org/visit",
                "phone": "+1 416 555 0100",
                "opening_hours": "Tu-Su 10:00-17:00",
                "description": "Description supplied by OpenStreetMap.",
            },
        },
        {
            "type": "way",
            "id": 101,
            "center": {"lat": 43.6465, "lon": -79.4637},
            "tags": {
                "name": "Green Park",
                "leisure": "park",
                "website": "http://unsafe-insecure.example.org",
            },
        },
        {
            "type": "relation",
            "id": 102,
            "center": {"lat": 43.6501, "lon": -79.3818},
            "tags": {"name": "City Theatre", "amenity": "theatre"},
        },
        {
            "type": "node",
            "id": 103,
            "lat": 43.6544,
            "lon": -79.3807,
            "tags": {"name": "City Mall", "shop": "mall"},
        },
        {
            "type": "node",
            "id": 104,
            "lat": 43.6426,
            "lon": -79.3871,
            "tags": {"name": "Historic Tower", "historic": "monument"},
        },
        {
            "type": "node",
            "id": 105,
            "lat": 43.65,
            "lon": -79.38,
            "tags": {"tourism": "museum"},
        },
    ]


def _hotel_elements() -> list[dict[str, Any]]:
    categories = ("hotel", "hostel", "guest_house", "motel", "apartment")
    return [
        {
            "type": "node",
            "id": 200 + index,
            "lat": 43.65 + index / 1_000,
            "lon": -79.38 - index / 1_000,
            "tags": {
                "name": f"Real {category}",
                "tourism": category,
                **({"stars": "4.5"} if category == "hotel" else {}),
            },
        }
        for index, category in enumerate(categories)
    ]


def _balanced_attraction_elements() -> list[dict[str, Any]]:
    category_tags = (
        {"historic": "monument"},
        {"tourism": "museum"},
        {"leisure": "park"},
        {"amenity": "theatre"},
        {"shop": "mall"},
    )
    return [
        {
            "type": "node",
            "id": 40_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {
                "name": f"Balanced Place {index:03}",
                **category_tags[index % len(category_tags)],
            },
        }
        for index in range(100)
    ]


def _service(
    transport: FakeTransport | None = None,
    clock: FakeClock | None = None,
    **kwargs: Any,
) -> tuple[DestinationGuideService, FakeTransport, FakeClock]:
    actual_transport = transport or FakeTransport()
    actual_clock = clock or FakeClock()
    service = DestinationGuideService(
        client=actual_transport,
        municipality_resolver=lambda code: "Toronto" if code == "YYZ" else None,
        monotonic_clock=actual_clock,
        sleeper=actual_clock.sleep,
        wall_clock=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        **kwargs,
    )
    return service, actual_transport, actual_clock


def _calls(transport: FakeTransport, fragment: str) -> list[dict[str, Any]]:
    return [call for call in transport.calls if fragment in call["url"]]


def test_served_city_centre_drives_real_attraction_query() -> None:
    service, transport, _ = _service()

    result = service.list_places("yyz", "attraction")

    assert result.city.destination_airport == "YYZ"
    assert result.city.served_city == "Toronto"
    assert result.city.city_query == "Toronto, CA"
    assert result.city.scope == "served_city"
    assert result.city.latitude == 43.6532
    assert result.result_count == 5
    assert result.coverage_radius_km == 30
    assert result.coverage_status == "complete"
    assert result.coverage_reason == "full_radius_queried"
    assert result.partial is False
    assert "30 公里" in result.coverage_notice.zh
    assert "full 30 km" in result.coverage_notice.en
    assert {place.category for place in result.places} == {
        "landmark",
        "museum",
        "nature",
        "entertainment",
        "shopping",
    }
    museum = next(place for place in result.places if place.category == "museum")
    assert museum.place_id == "osm_attraction_node_100"
    assert museum.address == "100 Museum Road, Toronto"
    assert museum.website == "https://museum.example.org/visit"
    assert museum.description == "Description supplied by OpenStreetMap."
    park = next(place for place in result.places if place.category == "nature")
    assert park.address is None
    assert park.website is None
    overpass_calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert len(overpass_calls) == 3
    assert OVERPASS_RADIUS_METERS == (5_000, 15_000, 30_000)
    latitude_spans: list[float] = []
    for call in overpass_calls:
        query = call["data"]["data"]
        assert "around:" not in query
        bounds_text = query.split("node(", 1)[1].split(")", 1)[0]
        south, west, north, east = (float(value) for value in bounds_text.split(","))
        assert south < 43.6532 < north
        assert west < -79.3832 < east
        assert query.count(f"node({bounds_text})") == 7
        assert "nwr(" not in query
        assert query.endswith("out body 100;")
        latitude_spans.append(north - south)
    assert latitude_spans[0] < latitude_spans[1] < latitude_spans[2]
    assert all('["name"]' in call["data"]["data"] for call in overpass_calls)
    overpass_call = overpass_calls[0]
    assert overpass_call["method"] == "POST"
    assert overpass_call["max_bytes"] == 5_000_000


def test_category_buttons_can_filter_cached_results_without_new_network_calls() -> None:
    service, transport, _ = _service()

    service.list_places("YYZ", "attraction", "all")
    museums = service.list_places("YYZ", "attraction", "museum")

    assert museums.category == "museum"
    assert museums.result_count == 1
    assert museums.places[0].name == "Verified Museum"
    assert len(_calls(transport, "nominatim")) == 1
    assert len(_calls(transport, "overpass-api.de")) == 3


def test_all_supported_hotel_types_and_source_provided_stars_are_preserved() -> None:
    service, transport, _ = _service()

    result = service.list_places("YYZ", "hotel")

    assert {place.category for place in result.places} == {
        "hotel",
        "hostel",
        "guest_house",
        "motel",
        "apartment",
    }
    hotel = next(place for place in result.places if place.category == "hotel")
    assert hotel.stars == 4.5
    assert all(place.stars is None for place in result.places if place.category != "hotel")
    assert all(place.data_source == "openstreetmap_overpass" for place in result.places)
    query = _calls(transport, "overpass-api.de")[0]["data"]["data"]
    assert '["tourism"~' not in query
    for category in ("hotel", "hostel", "guest_house", "motel", "apartment"):
        assert f'["tourism"="{category}"]["name"]' in query
    assert query.count("[\"tourism\"=") == 5


def test_place_detail_has_three_estimated_routes_and_explicit_transit_gap() -> None:
    service, transport, clock = _service()
    museum = service.list_places("YYZ", "attraction", "museum").places[0]

    detail = service.get_place_detail("YYZ", museum.place_id)

    assert detail.place == museum
    car, bike, foot, transit = detail.transport.options
    assert (car.mode, car.distance_km, car.duration_minutes) == ("car", 25.5, 30)
    assert (bike.mode, bike.distance_km, bike.duration_minutes) == ("bike", 26.2, 90)
    assert (foot.mode, foot.distance_km, foot.duration_minutes) == ("foot", 24.8, 300)
    assert car.duration_basis == "estimated_route_no_live_traffic"
    assert transit.mode == "public_transit"
    assert transit.status == "unavailable"
    assert transit.distance_km is None
    assert "not inferred" in transit.notice
    assert len(_calls(transport, "routing.openstreetmap.de")) == 3
    assert clock.sleeps == [pytest.approx(1.0)] * 4

    service.get_place_detail("YYZ", museum.place_id)
    assert len(_calls(transport, "routing.openstreetmap.de")) == 3


def test_route_provider_failure_does_not_invent_distance_or_time() -> None:
    transport = FakeTransport()
    transport.fail_modes.add("bike")
    service, _, _ = _service(transport=transport)

    routes = service.get_routes("YYZ", 43.6677, -79.3948)

    bike = routes.options[1]
    assert bike.mode == "bike"
    assert bike.status == "unavailable"
    assert bike.distance_km is None
    assert bike.duration_minutes is None
    assert bike.duration_basis is None


def test_transient_route_failure_is_retried_while_successes_remain_cached() -> None:
    transport = FakeTransport()
    transport.fail_modes.add("bike")
    service, _, _ = _service(transport=transport)

    first = service.get_routes("YYZ", 43.6677, -79.3948)
    transport.fail_modes.remove("bike")
    second = service.get_routes("YYZ", 43.6677, -79.3948)

    assert first.options[1].status == "unavailable"
    assert second.options[1].status == "available"
    route_calls = _calls(transport, "routing.openstreetmap.de")
    assert all(
        call["timeout"] == ROUTING_REQUEST_TIMEOUT_SECONDS == 5.0
        for call in route_calls
    )
    assert sum("routed-car" in call["url"] for call in route_calls) == 1
    assert sum("routed-bike" in call["url"] for call in route_calls) == 2
    assert sum("routed-foot" in call["url"] for call in route_calls) == 1


def test_failed_served_city_resolution_is_clearly_airport_scoped() -> None:
    transport = FakeTransport()
    transport.fail_search = True
    service, _, _ = _service(transport=transport)

    city = service.resolve_city("YYZ")

    assert city.served_city == "Toronto"
    assert city.name == "Toronto"
    assert city.scope == "airport_surroundings"
    assert city.latitude == pytest.approx(43.6777)
    assert "Actual queried coverage is reported separately" in city.scope_notice
    assert city.source == "nominatim_reverse_airport"
    assert len(_calls(transport, "nominatim")) == 2


def test_injected_ourairports_fallback_extends_global_airport_coverage() -> None:
    airport = Airport(
        iata="YQB",
        icao="CYQB",
        name="Quebec City Jean Lesage International",
        type="large_airport",
        country="CA",
        latitude=46.7911,
        longitude=-71.3933,
        source="ourairports",
    )
    transport = FakeTransport()
    transport.search_results = [
        {
            "lat": "46.8139",
            "lon": "-71.2080",
            "name": "Quebec City",
            "display_name": "Quebec City, Quebec, Canada",
            "addresstype": "city",
            "address": {"city": "Quebec City", "country_code": "ca"},
        }
    ]
    clock = FakeClock()
    service = DestinationGuideService(
        client=transport,
        airport_resolver=lambda code: airport if code == "YQB" else None,
        municipality_resolver=lambda code: "Quebec City" if code == "YQB" else None,
        monotonic_clock=clock,
        sleeper=clock.sleep,
        wall_clock=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )

    city = service.resolve_city("YQB")

    assert city.destination_airport == "YQB"
    assert city.airport_name == airport.name
    assert city.served_city == "Quebec City"
    assert city.latitude == 46.8139


def test_ourairports_municipality_reader_is_lazy_and_has_24_hour_cache() -> None:
    clock = FakeClock()
    payloads = [
        (
            b"iata_code,municipality\n"
            b"YYZ,Toronto\n"
            b"YQB,Quebec City\n"
        ),
        b"iata_code,municipality\nYYZ,New Toronto\n",
    ]
    calls = 0

    def downloader(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        payload = payloads[min(calls, len(payloads) - 1)]
        calls += 1
        return payload

    resolver = OurAirportsMunicipalityResolver(
        downloader=downloader,
        monotonic_clock=clock,
    )

    assert resolver("YYZ") == "Toronto"
    assert resolver("YQB") == "Quebec City"
    assert calls == 1
    clock.advance(86_399)
    assert resolver("YYZ") == "Toronto"
    assert calls == 1
    clock.advance(1)
    assert resolver("YYZ") == "New Toronto"
    assert calls == 2


def test_places_and_city_are_refetched_after_exact_24_hour_expiry() -> None:
    service, transport, clock = _service()

    service.list_places("YYZ", "hotel")
    clock.advance(86_399)
    service.list_places("YYZ", "hotel")
    assert len(_calls(transport, "overpass-api.de")) == 3

    clock.advance(1)
    service.list_places("YYZ", "hotel")
    assert len(_calls(transport, "overpass-api.de")) == 6
    assert len(_calls(transport, "nominatim")) == 2


def test_overpass_results_are_bounded_to_30_nearest_named_places() -> None:
    transport = FakeTransport()
    transport.attractions = [
        {
            "type": "node",
            "id": 10_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Museum {index:02}", "tourism": "museum"},
        }
        for index in range(35)
    ]
    service, _, _ = _service(transport=transport)

    result = service.list_places("YYZ", "attraction")

    assert result.result_count == 30
    assert result.places[0].name == "Museum 00"
    assert result.places[-1].name == "Museum 29"


def test_internal_100_supports_balanced_all_categories_filters_and_hidden_detail() -> None:
    transport = FakeTransport()
    transport.attractions = _balanced_attraction_elements()
    service, _, _ = _service(transport=transport)

    all_places = service.list_places("YYZ", "attraction", "all")
    museums = service.list_places("YYZ", "attraction", "museum")
    hidden_id = "osm_attraction_node_40099"
    detail = service.get_place_detail("YYZ", hidden_id)

    assert all_places.result_count == 30
    assert Counter(place.category for place in all_places.places) == {
        "landmark": 6,
        "museum": 6,
        "nature": 6,
        "entertainment": 6,
        "shopping": 6,
    }
    assert museums.result_count == 20
    assert all(place.category == "museum" for place in museums.places)
    assert hidden_id not in {place.place_id for place in all_places.places}
    assert detail.place.place_id == hidden_id
    assert len(_calls(transport, "overpass-api.de")) == 1


def test_first_5km_response_over_30_is_fully_cached_without_radius_expansion() -> None:
    transport = FakeTransport()
    hotel_categories = ("hotel", "hostel", "guest_house", "motel", "apartment")
    transport.hotels = [
        {
            "type": "node",
            "id": 50_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {
                "name": f"City Stay {index:02}",
                "tourism": hotel_categories[index % len(hotel_categories)],
            },
        }
        for index in range(49)
    ]
    service, _, _ = _service(transport=transport)

    all_hotels = service.list_places("YYZ", "hotel", "all")
    hostels = service.list_places("YYZ", "hotel", "hostel")
    hidden_detail = service.get_place_detail("YYZ", "osm_hotel_node_50048")

    assert all_hotels.result_count == 30
    assert hostels.result_count == 10
    assert all_hotels.coverage_radius_km == 5
    assert all_hotels.coverage_status == "partial"
    assert all_hotels.coverage_reason == "result_target_reached"
    assert all_hotels.partial is True
    assert "不是完整 30 公里覆盖" in all_hotels.coverage_notice.zh
    assert "not full 30 km coverage" in all_hotels.coverage_notice.en
    assert hidden_detail.place.place_id == "osm_hotel_node_50048"
    assert len(_calls(transport, "overpass-api.de")) == 1


def test_progressive_overpass_stops_when_100_unique_named_places_are_available() -> None:
    transport = FakeTransport()
    first_twenty = [
        {
            "type": "node",
            "id": 20_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Museum {index:02}", "tourism": "museum"},
        }
        for index in range(20)
    ]
    first_hundred = first_twenty + [
        {
            "type": "node",
            "id": 20_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Museum {index:02}", "tourism": "museum"},
        }
        for index in range(20, 100)
    ]
    transport.overpass_payloads = [
        {"elements": first_twenty},
        {"elements": first_hundred},
    ]
    service, _, clock = _service(transport=transport)

    result = service.list_places("YYZ", "attraction")

    assert result.result_count == 30
    calls = _calls(transport, "overpass-api.de")
    assert len(calls) == 2
    first_bounds = calls[0]["data"]["data"].split("node(", 1)[1].split(")", 1)[0]
    second_bounds = calls[1]["data"]["data"].split("node(", 1)[1].split(")", 1)[0]
    assert first_bounds != second_bounds
    assert all("around:" not in call["data"]["data"] for call in calls)
    assert clock.sleeps == [pytest.approx(1.0)]


def test_primary_overpass_success_never_calls_fallback_and_uses_short_timeout() -> None:
    transport = FakeTransport()
    transport.attractions = [
        {
            "type": "node",
            "id": 25_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Primary Museum {index:02}", "tourism": "museum"},
        }
        for index in range(100)
    ]
    service, _, _ = _service(transport=transport, timeout_seconds=15.0)

    result = service.list_places("YYZ", "attraction")

    assert result.result_count == 30
    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert [call["url"] for call in calls] == [OVERPASS_URL]
    assert calls[0]["timeout"] == OVERPASS_REQUEST_TIMEOUT_SECONDS == 6.0
    assert "[timeout:5]" in calls[0]["data"]["data"]


def test_primary_failure_retries_fallback_once_then_caches_success() -> None:
    transport = FakeTransport()
    fallback_elements = [
        {
            "type": "node",
            "id": 26_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Fallback Museum {index:02}", "tourism": "museum"},
        }
        for index in range(100)
    ]
    transport.overpass_payloads = [
        DestinationDataUnavailable("primary unavailable"),
        {"elements": fallback_elements},
    ]
    service, _, clock = _service(transport=transport)

    first = service.list_places("YYZ", "attraction")
    cached = service.list_places("YYZ", "attraction", "museum")

    assert first.result_count == 30
    assert cached.result_count == 30
    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert [call["url"] for call in calls] == [OVERPASS_URL, OVERPASS_FALLBACK_URL]
    assert clock.sleeps == [pytest.approx(1.0)]


def test_one_list_operation_uses_at_most_one_fallback_and_stays_under_budget() -> None:
    transport = FakeTransport()
    small_success = {"elements": _attraction_elements()}
    transport.overpass_payloads = [
        DestinationDataUnavailable("primary unavailable"),
        small_success,
        small_success,
        small_success,
        small_success,
    ]
    service, _, _ = _service(transport=transport)

    result = service.list_places("YYZ", "attraction")

    assert result.result_count == 5
    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert len(calls) == 4
    assert [call["url"] for call in calls].count(OVERPASS_FALLBACK_URL) == 1
    assert all(call["timeout"] <= OVERPASS_REQUEST_TIMEOUT_SECONDS for call in calls)
    assert OVERPASS_OPERATION_BUDGET_SECONDS == 24.0 < 30


def test_operation_budget_prevents_an_additional_live_request() -> None:
    clock = FakeClock()
    transport = FakeTransport()
    transport.overpass_call_hook = lambda: clock.advance(8.0)
    transport.overpass_payloads = [
        DestinationDataUnavailable("primary unavailable"),
        {"elements": _attraction_elements()},
        {"elements": _attraction_elements()},
        {"elements": _attraction_elements()},
    ]
    service, _, _ = _service(transport=transport, clock=clock)

    result = service.list_places("YYZ", "attraction")

    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert result.result_count == 5
    assert [call["url"] for call in calls] == [
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_URL,
    ]
    assert clock.value == 1_000.0 + OVERPASS_OPERATION_BUDGET_SECONDS


def test_primary_remark_is_retried_once_on_fallback() -> None:
    transport = FakeTransport()
    fallback_elements = [
        {
            "type": "node",
            "id": 27_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Remark Recovery {index:02}", "tourism": "museum"},
        }
        for index in range(100)
    ]
    transport.overpass_payloads = [
        {"elements": [], "remark": "runtime error: Query timed out"},
        {"elements": fallback_elements},
    ]
    service, _, _ = _service(transport=transport)

    result = service.list_places("YYZ", "attraction")

    assert result.result_count == 30
    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert [call["url"] for call in calls] == [OVERPASS_URL, OVERPASS_FALLBACK_URL]


def test_bbox_corner_places_beyond_each_circle_radius_are_clipped() -> None:
    transport = FakeTransport()
    inside = [
        {
            "type": "node",
            "id": 30_000 + index,
            "lat": 43.6532 + index / 100_000,
            "lon": -79.3832,
            "tags": {"name": f"Inside Museum {index:02}", "tourism": "museum"},
        }
        for index in range(29)
    ]
    corner_offsets = ((0.04, 0.055), (0.12, 0.165), (0.24, 0.33))
    transport.overpass_payloads = [
        {
            "elements": [
                *inside,
                {
                    "type": "node",
                    "id": 31_000 + index,
                    "lat": 43.6532 + latitude_offset,
                    "lon": -79.3832 + longitude_offset,
                    "tags": {
                        "name": f"Square Corner {index}",
                        "tourism": "museum",
                    },
                },
            ]
        }
        for index, (latitude_offset, longitude_offset) in enumerate(corner_offsets)
    ]
    service, _, _ = _service(transport=transport)

    result = service.list_places("YYZ", "attraction")

    assert result.result_count == 29
    assert all(not place.name.startswith("Square Corner") for place in result.places)
    assert len(_calls(transport, "overpass-api.de")) == 3


def test_overpass_runtime_remark_is_provider_failure_and_is_not_cached() -> None:
    transport = FakeTransport()
    transport.overpass_payloads = [
        {
            "elements": [],
            "remark": "runtime error: Query timed out in query after 13 seconds",
        }
    ]
    service, _, _ = _service(transport=transport)

    with pytest.raises(DestinationDataUnavailable):
        service.list_places("YYZ", "attraction")
    with pytest.raises(DestinationDataUnavailable):
        service.list_places("YYZ", "attraction")

    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert [call["url"] for call in calls] == [
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
    ]


def test_wider_radius_failure_keeps_and_caches_smaller_source_backed_results() -> None:
    transport = FakeTransport()
    transport.overpass_payloads = [
        {"elements": _attraction_elements()},
        {
            "elements": [],
            "remark": "runtime error: Query timed out in query after 13 seconds",
        },
    ]
    service, _, _ = _service(transport=transport)

    first = service.list_places("YYZ", "attraction")
    second = service.list_places("YYZ", "attraction", "museum")

    assert first.result_count == 4
    assert second.result_count == 1
    assert first.coverage_radius_km == 5
    assert first.coverage_status == "partial"
    assert first.coverage_reason == "provider_failure"
    assert first.partial is True
    assert "较大范围查询暂时失败" in first.coverage_notice.zh
    assert "wider-radius query failed" in first.coverage_notice.en
    assert second.coverage_notice == first.coverage_notice
    assert len([call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]) == 3


@pytest.mark.parametrize(
    ("kind", "category"),
    [("attractions", "all"), ("hotel", "museum"), ("attraction", "hostel")],
)
def test_invalid_kind_or_cross_kind_category_is_rejected(kind: str, category: str) -> None:
    service, transport, _ = _service()

    with pytest.raises(DestinationValidationError):
        service.list_places("YYZ", kind, category)  # type: ignore[arg-type]

    assert transport.calls == []


def test_unknown_airport_and_bad_place_id_are_rejected_without_provider_calls() -> None:
    service, transport, _ = _service(airport_resolver=lambda _code: None)

    with pytest.raises(DestinationAirportNotFound):
        service.resolve_city("ZZZ")
    with pytest.raises(DestinationValidationError):
        service.get_place_detail("YYZ", "https://evil.example/place")

    assert transport.calls == []


def test_valid_but_absent_place_id_returns_not_found() -> None:
    service, transport, _ = _service()

    with pytest.raises(DestinationPlaceNotFound):
        service.get_place_detail("YYZ", "osm_hotel_node_999999")

    assert len(_calls(transport, "overpass-api.de")) == 3
    assert len(_calls(transport, "routing.openstreetmap.de")) == 0


def test_route_coordinates_are_strict_and_never_interpolated_from_strings() -> None:
    service, transport, _ = _service()

    with pytest.raises(DestinationValidationError):
        service.get_routes("YYZ", "43.65", -79.38)  # type: ignore[arg-type]
    with pytest.raises(DestinationValidationError):
        service.get_routes("YYZ", 100.0, -79.38)

    assert transport.calls == []


def test_pydantic_output_models_reject_coercion_and_unsafe_websites() -> None:
    with pytest.raises(ValidationError):
        DestinationPlace(
            place_id="osm_hotel_node_1",
            kind="hotel",
            category="hotel",
            name="Hotel",
            latitude="43.6",
            longitude=-79.3,
            distance_from_city_center_km=1.0,
            source_url="https://www.openstreetmap.org/node/1",
        )
    with pytest.raises(ValidationError, match="safe public HTTPS"):
        DestinationPlace(
            place_id="osm_hotel_node_1",
            kind="hotel",
            category="hotel",
            name="Hotel",
            website="https://127.0.0.1/internal",
            latitude=43.6,
            longitude=-79.3,
            distance_from_city_center_km=1.0,
            source_url="https://www.openstreetmap.org/node/1",
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://nominatim.openstreetmap.org/search",
        "https://evil.example/search",
        "https://nominatim.openstreetmap.org/other",
    ],
)
def test_default_client_rejects_non_https_or_non_allowlisted_endpoints_before_io(
    url: str,
) -> None:
    client = BoundedJsonHttpClient()

    with pytest.raises(DestinationValidationError, match="allowlist"):
        client.request_json(
            "GET",
            url,
            timeout_seconds=1.0,
            max_response_bytes=1_000,
        )


def test_ourairports_downloader_refuses_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        @staticmethod
        def geturl() -> str:
            return destination_guide.OURAIRPORTS_CSV_URL

        @staticmethod
        def read(_limit: int) -> bytes:
            return b"iata_code,municipality\nYYZ,Toronto\n"

    class Opener:
        @staticmethod
        def open(_outbound: Any, *, timeout: float) -> Response:
            assert timeout == 1.0
            return Response()

    def build_opener(handler: Any) -> Opener:
        assert isinstance(handler, destination_guide._NoRedirectHandler)
        return Opener()

    monkeypatch.setattr(destination_guide.request, "build_opener", build_opener)

    payload = destination_guide._download_ourairports_municipalities(
        destination_guide.OURAIRPORTS_CSV_URL,
        1.0,
    )

    assert payload.startswith(b"iata_code")


def test_overpass_allowlist_contains_exactly_primary_and_one_verified_fallback() -> None:
    assert OVERPASS_ENDPOINTS == (OVERPASS_URL, OVERPASS_FALLBACK_URL)
    assert len(OVERPASS_ENDPOINTS) == 2
    for endpoint in OVERPASS_ENDPOINTS:
        _assert_allowed_outbound_url(endpoint)

    with pytest.raises(DestinationValidationError, match="allowlist"):
        _assert_allowed_outbound_url("https://overpass.private.coffee/api/interpreter")
    with pytest.raises(DestinationValidationError, match="allowlist"):
        _assert_allowed_outbound_url(
            "https://attacker.maps.mail.ru/osm/tools/overpass/api/interpreter"
        )
