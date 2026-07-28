from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

import flight_forecaster.destination_guide as destination_guide
from flight_forecaster.destination_guide import (
    ATTRACTION_OVERPASS_OPERATION_BUDGET_SECONDS,
    ATTRACTION_OVERPASS_QUERY_TIMEOUT_SECONDS,
    ATTRACTION_OVERPASS_REQUEST_TIMEOUT_SECONDS,
    OVERPASS_ENDPOINTS,
    OVERPASS_FALLBACK_URL,
    OVERPASS_RADIUS_METERS,
    OVERPASS_URL,
    PARTIAL_DESTINATION_CACHE_TTL,
    ROUTING_REQUEST_TIMEOUT_SECONDS,
    TRANSITOUS_PLAN_URL,
    TRANSITOUS_REQUEST_TIMEOUT_SECONDS,
    WIKIDATA_API_URL,
    WIKIPEDIA_API_URLS,
    BoundedJsonHttpClient,
    DestinationAirportNotFound,
    DestinationDataUnavailable,
    DestinationGuideService,
    DestinationPlace,
    DestinationPlaceNotFound,
    DestinationValidationError,
    OurAirportsMunicipalityResolver,
    _assert_allowed_outbound_url,
    _parse_osm_transit_reference_payload,
    _TtlCache,
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
        self.transit_payload: Any = {"itineraries": []}
        self.fail_transit = False
        self.wikidata_entities: dict[str, dict[str, Any]] = {}
        self.wikipedia_pages: dict[str, list[dict[str, Any]]] = {"zh": [], "en": []}

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
        if url == WIKIDATA_API_URL:
            requested = str((params or {}).get("ids") or "").split("|")
            return {
                "entities": {
                    entity_id: self.wikidata_entities[entity_id]
                    for entity_id in requested
                    if entity_id in self.wikidata_entities
                }
            }
        if url in WIKIPEDIA_API_URLS.values():
            language = "zh" if url == WIKIPEDIA_API_URLS["zh"] else "en"
            return {"query": {"pages": self.wikipedia_pages[language]}}
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
        if url == TRANSITOUS_PLAN_URL:
            if self.fail_transit:
                raise DestinationDataUnavailable("transit provider unavailable")
            return self.transit_payload
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


def _transit_itinerary() -> dict[str, Any]:
    return {
        "itineraries": [
            {
                "duration": 2_700,
                "startTime": "2026-07-21T13:00:00Z",
                "endTime": "2026-07-21T13:45:00Z",
                "transfers": 0,
                "legs": [
                    {
                        "mode": "WALK",
                        "from": {"name": "START"},
                        "to": {"name": "Airport Station"},
                        "duration": 300,
                        "distance": 350,
                        "startTime": "2026-07-21T13:00:00Z",
                        "endTime": "2026-07-21T13:05:00Z",
                        "scheduledStartTime": "2026-07-21T13:00:00Z",
                        "scheduledEndTime": "2026-07-21T13:05:00Z",
                        "realTime": False,
                        "scheduled": False,
                    },
                    {
                        "mode": "REGIONAL_RAIL",
                        "from": {"name": "Airport Station", "tz": "America/Toronto"},
                        "to": {"name": "Central Station", "tz": "America/Toronto"},
                        "duration": 1_800,
                        "startTime": "2026-07-21T13:05:00Z",
                        "endTime": "2026-07-21T13:35:00Z",
                        "scheduledStartTime": "2026-07-21T13:04:00Z",
                        "scheduledEndTime": "2026-07-21T13:34:00Z",
                        "realTime": True,
                        "scheduled": True,
                        "displayName": "Airport Express",
                        "headsign": "Central Station",
                        "agencyName": "Example Transit",
                        "intermediateStops": [
                            {"name": "Junction"},
                            {"name": "Museum Stop"},
                        ],
                    },
                    {
                        "mode": "WALK",
                        "from": {"name": "Central Station"},
                        "to": {"name": "END"},
                        "duration": 600,
                        "distance": 720,
                        "startTime": "2026-07-21T13:35:00Z",
                        "endTime": "2026-07-21T13:45:00Z",
                        "scheduledStartTime": "2026-07-21T13:35:00Z",
                        "scheduledEndTime": "2026-07-21T13:45:00Z",
                        "realTime": False,
                        "scheduled": False,
                    },
                ],
            }
        ]
    }


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
    assert len(overpass_calls) == 4
    assert OVERPASS_RADIUS_METERS == (5_000, 15_000, 30_000)
    tile_bounds: set[tuple[float, float, float, float]] = set()
    for call in overpass_calls:
        query = call["data"]["data"]
        assert "around:" not in query
        bounds_text = query.split("nwr(", 1)[1].split(")", 1)[0]
        south, west, north, east = (float(value) for value in bounds_text.split(","))
        assert south <= 43.6532 <= north
        assert west <= -79.3832 <= east
        assert query.count(f"nwr({bounds_text})") == 7
        assert query.endswith("out center 350;")
        tile_bounds.add((south, west, north, east))
    assert len(tile_bounds) == 4
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
    assert len(_calls(transport, "overpass-api.de")) == 4


def test_exact_wikimedia_identity_adds_introductions_and_platform_scores() -> None:
    transport = FakeTransport()
    transport.attractions = [
        {
            "type": "relation",
            "id": 88_001,
            "center": {"lat": 43.654, "lon": -79.382},
            "tags": {
                "name": "Verified Gallery",
                "name:en": "Verified Gallery",
                "tourism": "gallery",
                "wikidata": "Q1001",
                "wikipedia": "en:Verified Gallery",
            },
        }
    ]
    transport.wikidata_entities = {
        "Q1001": {
            "labels": {
                "en": {"language": "en", "value": "Verified Gallery"},
                "zh": {"language": "zh", "value": "已验证美术馆"},
            },
            "descriptions": {
                "en": {"language": "en", "value": "A gallery in Toronto."},
                "zh": {"language": "zh", "value": "位于多伦多的美术馆。"},
            },
            "sitelinks": {
                "enwiki": {"site": "enwiki", "title": "Verified Gallery"},
                "zhwiki": {"site": "zhwiki", "title": "已验证美术馆"},
            },
            "claims": {
                "P444": [
                    {
                        "rank": "normal",
                        "mainsnak": {"datavalue": {"value": "4.7/5"}},
                        "qualifiers": {
                            "P447": [
                                {"datavalue": {"value": {"id": "Q2001"}}}
                            ],
                            "P7887": [
                                {"datavalue": {"value": {"amount": "+1234"}}}
                            ],
                            "P585": [
                                {
                                    "datavalue": {
                                        "value": {"time": "+2026-01-02T00:00:00Z"}
                                    }
                                }
                            ],
                            "P2699": [
                                {
                                    "datavalue": {
                                        "value": "https://ratings.example.org/gallery"
                                    }
                                }
                            ],
                        },
                    }
                ]
            },
        },
        "Q2001": {
            "labels": {
                "en": {"language": "en", "value": "Example Reviews"},
                "zh": {"language": "zh", "value": "示例评价平台"},
            }
        },
    }
    transport.wikipedia_pages = {
        "zh": [
            {
                "title": "已验证美术馆",
                "extract": "这是由中文维基百科返回的真实简介。",
                "canonicalurl": (
                    "https://zh.wikipedia.org/wiki/"
                    "%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%BE%8E%E6%9C%AF%E9%A6%86"
                ),
            }
        ],
        "en": [
            {
                "title": "Verified Gallery",
                "extract": "This is a real introduction returned by English Wikipedia.",
                "canonicalurl": "https://en.wikipedia.org/wiki/Verified_Gallery",
            }
        ],
    }
    service, _, _ = _service(transport=transport)

    place = service.list_places("YYZ", "attraction", limit=300).places[0]

    assert place.description_zh == "这是由中文维基百科返回的真实简介。"
    assert place.description_en == (
        "This is a real introduction returned by English Wikipedia."
    )
    assert place.description_basis == "wikipedia_extract"
    assert place.description_source == "wikipedia"
    assert place.ratings_status == "available"
    assert len(place.ratings) == 1
    rating = place.ratings[0]
    assert (rating.platform_en, rating.platform_zh) == (
        "Example Reviews",
        "示例评价平台",
    )
    assert (rating.score_text, rating.score, rating.max_score) == ("4.7/5", 4.7, 5.0)
    assert rating.review_count == 1234
    assert rating.point_in_time == "2026-01-02"
    assert rating.source_url == "https://ratings.example.org/gallery"


def test_attraction_without_free_text_gets_only_an_osm_fact_summary() -> None:
    service, _, _ = _service()

    park = service.list_places("YYZ", "attraction", "nature").places[0]

    assert park.description_basis == "osm_tag_summary"
    assert park.description_source == "openstreetmap"
    assert "OpenStreetMap" in (park.description_zh or "")
    assert "Toronto" in (park.description_en or "")
    assert park.ratings == ()
    assert park.ratings_status == "source_not_provided"


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
    assert transit.coverage_status == "no_itinerary"
    assert transit.departure_time_basis == "request_time"
    assert transit.requested_departure_at == datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    assert len(_calls(transport, "routing.openstreetmap.de")) == 3
    assert len(_calls(transport, "api.transitous.org")) == 1
    assert clock.sleeps == [pytest.approx(1.0)] * 5

    service.get_place_detail("YYZ", museum.place_id)
    assert len(_calls(transport, "routing.openstreetmap.de")) == 3
    assert len(_calls(transport, "api.transitous.org")) == 1


def test_transitous_itinerary_exposes_real_legs_lines_stops_and_requested_time() -> None:
    transport = FakeTransport()
    transport.transit_payload = _transit_itinerary()
    service, _, _ = _service(transport=transport)
    requested = datetime(2026, 7, 21, 13, 0, 45, tzinfo=UTC)
    normalized = requested.replace(second=0)

    routes = service.get_routes(
        "YYZ",
        43.6677,
        -79.3948,
        transit_departure_at=requested,
    )

    transit = routes.options[3]
    assert transit.status == "available"
    assert transit.duration_minutes == 45
    assert transit.duration_basis == "transit_schedule_or_realtime"
    assert transit.departure_time_basis == "user_supplied"
    assert transit.requested_departure_at == normalized
    assert transit.departure_at == normalized
    assert transit.arrival_at == datetime(2026, 7, 21, 13, 45, tzinfo=UTC)
    assert transit.transfers == 0
    assert transit.realtime is True
    assert transit.coverage_status == "covered"
    assert transit.source_url == "https://transitous.org/sources/"
    assert [leg.mode for leg in transit.legs] == [
        "WALK",
        "REGIONAL_RAIL",
        "WALK",
    ]
    rail = transit.legs[1]
    assert rail.line_name == "Airport Express"
    assert rail.from_timezone == rail.to_timezone == "America/Toronto"
    assert rail.agency_name == "Example Transit"
    assert rail.headsign == "Central Station"
    assert rail.intermediate_stops == ("Junction", "Museum Stop")
    assert rail.realtime is True
    calls = _calls(transport, "api.transitous.org")
    assert len(calls) == 1
    assert calls[0]["params"]["time"] == "2026-07-21T13:00:00+00:00"
    assert calls[0]["params"]["directModes"] == ""
    assert "AIRPLANE" not in calls[0]["params"]["transitModes"]
    assert calls[0]["timeout"] == TRANSITOUS_REQUEST_TIMEOUT_SECONDS
    assert "flight-forecast-lab/0.2" in calls[0]["headers"]["User-Agent"]

    service.get_routes(
        "YYZ",
        43.6677,
        -79.3948,
        transit_departure_at=requested,
    )
    assert len(_calls(transport, "api.transitous.org")) == 1


def test_transitous_failure_is_unavailable_without_invented_route_and_is_retried() -> None:
    transport = FakeTransport()
    transport.fail_transit = True
    service, _, _ = _service(transport=transport)

    first = service.get_routes("YYZ", 43.6677, -79.3948).options[3]
    second = service.get_routes("YYZ", 43.6677, -79.3948).options[3]

    assert first.status == second.status == "unavailable"
    assert first.coverage_status == second.coverage_status == "provider_unavailable"
    assert first.duration_minutes is None
    assert first.legs == ()
    assert first.source_url == "https://transitous.org/sources/"
    assert len(_calls(transport, "api.transitous.org")) == 2


def test_nonempty_invalid_transitous_itineraries_are_provider_failure_and_not_cached() -> None:
    transport = FakeTransport()
    transport.transit_payload = {"itineraries": [{"duration": 600, "legs": "invalid"}]}
    service, _, _ = _service(transport=transport)

    first = service.get_routes("YYZ", 43.6677, -79.3948).options[3]
    second = service.get_routes("YYZ", 43.6677, -79.3948).options[3]

    assert first.coverage_status == second.coverage_status == "provider_unavailable"
    assert first.duration_minutes is None
    assert first.legs == ()
    assert len(_calls(transport, "api.transitous.org")) == 2


def test_ttl_cache_evicts_oldest_entry_at_capacity() -> None:
    clock = FakeClock()
    cache = _TtlCache(clock, ttl=timedelta(minutes=30), max_entries=2)

    cache.set(("first",), 1)
    cache.set(("second",), 2)
    cache.set(("third",), 3)

    assert cache.get(("first",)) is None
    assert cache.get(("second",)) == 2
    assert cache.get(("third",)) == 3


def test_ttl_cache_supports_a_short_per_entry_expiry() -> None:
    clock = FakeClock()
    cache = _TtlCache(clock, ttl=timedelta(hours=24))

    cache.set(("partial",), 1, ttl=PARTIAL_DESTINATION_CACHE_TTL)
    clock.advance(PARTIAL_DESTINATION_CACHE_TTL.total_seconds() - 1)
    assert cache.get(("partial",)) == 1

    clock.advance(1)
    assert cache.get(("partial",)) is None


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
    assert len(_calls(transport, "overpass-api.de")) == 4


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
    assert len(calls) == 4
    first_bounds = calls[0]["data"]["data"].split("nwr(", 1)[1].split(")", 1)[0]
    second_bounds = calls[1]["data"]["data"].split("nwr(", 1)[1].split(")", 1)[0]
    assert first_bounds != second_bounds
    assert all("around:" not in call["data"]["data"] for call in calls)
    assert clock.sleeps == [pytest.approx(1.0)] * 3


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
    assert [call["url"] for call in calls] == [OVERPASS_URL] * 4
    assert (
        calls[0]["timeout"]
        == ATTRACTION_OVERPASS_REQUEST_TIMEOUT_SECONDS
        == 9.0
    )
    assert f"[timeout:{ATTRACTION_OVERPASS_QUERY_TIMEOUT_SECONDS}]" in calls[0]["data"][
        "data"
    ]


def test_primary_failure_switches_to_fallback_then_caches_success() -> None:
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
    assert [call["url"] for call in calls] == [
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
    ]
    assert clock.sleeps == [pytest.approx(1.0)] * 4


def test_one_list_operation_reuses_fallback_and_stays_under_budget() -> None:
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
    assert len(calls) == 5
    assert [call["url"] for call in calls].count(OVERPASS_FALLBACK_URL) == 4
    assert all(
        call["timeout"] <= ATTRACTION_OVERPASS_REQUEST_TIMEOUT_SECONDS
        for call in calls
    )
    assert ATTRACTION_OVERPASS_OPERATION_BUDGET_SECONDS == 42.0 < 60


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
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
    ]
    assert clock.value == 1_040.0


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
    assert [call["url"] for call in calls] == [
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
    ]


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

    assert result.result_count == 30
    assert all(place.name != "Square Corner 2" for place in result.places)
    assert len(_calls(transport, "overpass-api.de")) == 4


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
        endpoint
        for _ in range(2)
        for _ in range(4)
        for endpoint in OVERPASS_ENDPOINTS
    ]


def test_attraction_tiles_reuse_the_last_successful_overpass_endpoint() -> None:
    transport = FakeTransport()
    transport.overpass_payloads = [
        DestinationDataUnavailable("primary unavailable"),
        {"elements": _attraction_elements()},
        {"elements": _attraction_elements()},
        {"elements": _attraction_elements()},
        {"elements": _attraction_elements()},
    ]
    service, _, _ = _service(transport=transport)

    result = service.list_places("YYZ", "attraction")

    assert result.coverage_status == "complete"
    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert [call["url"] for call in calls] == [
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
        OVERPASS_FALLBACK_URL,
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
    service, _, clock = _service(transport=transport)

    first = service.list_places("YYZ", "attraction")
    second = service.list_places("YYZ", "attraction", "museum")

    assert first.result_count == 5
    assert second.result_count == 1
    assert first.coverage_radius_km == 30
    assert first.coverage_status == "partial"
    assert first.coverage_reason == "provider_failure"
    assert first.partial is True
    assert first.query_parts_succeeded == 1
    assert first.query_parts_total == 4
    assert "1/4" in first.coverage_notice.zh
    assert "1/4" in first.coverage_notice.en
    assert second.coverage_notice == first.coverage_notice
    calls = [call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]
    assert len(calls) == 7
    clock.advance(PARTIAL_DESTINATION_CACHE_TTL.total_seconds() - 1)
    service.list_places("YYZ", "attraction")
    assert len([call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]) == 7

    transport.overpass_payloads = [{"elements": _attraction_elements()}]
    clock.advance(1)
    refreshed = service.list_places("YYZ", "attraction")
    assert refreshed.coverage_status == "complete"
    assert len([call for call in transport.calls if call["url"] in OVERPASS_ENDPOINTS]) == 11


def test_osm_transit_references_include_only_stop_and_platform_members() -> None:
    payload = {
        "elements": [
            {"type": "node", "id": 1, "tags": {"name": "Actual Stop"}},
            {"type": "node", "id": 2, "tags": {"name": "Geometry Node"}},
            {"type": "node", "id": 3, "tags": {"name": "Route Landmark"}},
            {
                "type": "relation",
                "id": 88,
                "tags": {
                    "type": "route",
                    "route": "subway",
                    "name": "Line 2",
                    "ref": "2",
                },
                "members": [
                    {"type": "node", "ref": 1, "role": "platform"},
                    {"type": "node", "ref": 2, "role": ""},
                    {"type": "node", "ref": 3, "role": "via"},
                ],
            },
        ]
    }

    routes = _parse_osm_transit_reference_payload(payload)

    assert routes[88]["stops"] == ("Actual Stop",)


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


def test_transitous_allowlist_accepts_only_the_stable_plan_endpoint() -> None:
    _assert_allowed_outbound_url(TRANSITOUS_PLAN_URL)

    with pytest.raises(DestinationValidationError, match="allowlist"):
        _assert_allowed_outbound_url("https://staging.api.transitous.org/api/v5/plan")
    with pytest.raises(DestinationValidationError, match="allowlist"):
        _assert_allowed_outbound_url("https://api.transitous.org/api/v5/trip")
