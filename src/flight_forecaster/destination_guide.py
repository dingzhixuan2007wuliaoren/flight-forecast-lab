"""Real, source-labelled destination places and airport-to-place routing.

The module deliberately keeps destination discovery independent from airfare and
hotel-price providers.  It uses OurAirports only to resolve the destination airport
and its served municipality, Nominatim to resolve the served-city centre, Overpass
for OpenStreetMap places, and the public routing.openstreetmap.de OSRM instances for
estimated car, bicycle, and walking routes.  Public-transport itineraries come from
Transitous' public MOTIS endpoint and are shown only when its open timetable data
returns a complete, parseable itinerary.

All network dependencies are injectable.  Provider payloads are bounded, cached for
the source-specific TTL, and converted into strict Pydantic models; missing OSM tags remain ``None``
instead of being inferred.
"""

from __future__ import annotations

import csv
import ipaddress
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from io import StringIO
from typing import Any, Literal, Protocol
from urllib import error, parse, request

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flight_forecaster.route_info import (
    OURAIRPORTS_CSV_URL,
    Airport,
    AirportResolver,
    OurAirportsResolver,
    lookup_airport,
)

PlaceKind = Literal["attraction", "hotel"]
AttractionCategory = Literal[
    "landmark",
    "museum",
    "nature",
    "entertainment",
    "shopping",
]
HotelCategory = Literal["hotel", "hostel", "guest_house", "motel", "apartment"]
PlaceCategory = AttractionCategory | HotelCategory
ListCategory = Literal[
    "all",
    "landmark",
    "museum",
    "nature",
    "entertainment",
    "shopping",
    "hotel",
    "hostel",
    "guest_house",
    "motel",
    "apartment",
]
TransportMode = Literal["car", "bike", "foot", "public_transit"]
TransitLegMode = Literal[
    "WALK",
    "TRAM",
    "SUBWAY",
    "FERRY",
    "SUBURBAN",
    "BUS",
    "COACH",
    "RAIL",
    "HIGHSPEED_RAIL",
    "LONG_DISTANCE",
    "NIGHT_RAIL",
    "REGIONAL_FAST_RAIL",
    "REGIONAL_RAIL",
    "CABLE_CAR",
    "FUNICULAR",
    "AERIAL_LIFT",
    "OTHER",
]

ATTRACTION_CATEGORIES = frozenset(
    {"landmark", "museum", "nature", "entertainment", "shopping"}
)
HOTEL_CATEGORIES = frozenset({"hotel", "hostel", "guest_house", "motel", "apartment"})
ATTRACTION_CATEGORY_ORDER: tuple[AttractionCategory, ...] = (
    "landmark",
    "museum",
    "nature",
    "entertainment",
    "shopping",
)
HOTEL_CATEGORY_ORDER: tuple[HotelCategory, ...] = (
    "hotel",
    "hostel",
    "guest_house",
    "motel",
    "apartment",
)
PLACE_ID_PATTERN = re.compile(
    r"^osm_(attraction|hotel)_(node|way|relation)_([1-9][0-9]{0,18})$"
)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
OVERPASS_ENDPOINTS = (OVERPASS_URL, OVERPASS_FALLBACK_URL)
ROUTING_BASE_URL = "https://routing.openstreetmap.de"
TRANSITOUS_PLAN_URL = "https://api.transitous.org/api/v5/plan"
TRANSITOUS_SOURCES_URL = "https://transitous.org/sources/"
OPENSTREETMAP_BASE_URL = "https://www.openstreetmap.org"
DESTINATION_CACHE_TTL = timedelta(hours=24)
TRANSIT_CACHE_TTL = timedelta(minutes=30)
TRANSIT_CACHE_MAX_ENTRIES = 512
DESTINATION_RADIUS_METERS = 30_000
OVERPASS_RADIUS_METERS = (5_000, 15_000, DESTINATION_RADIUS_METERS)
MAX_PLACES_PER_KIND = 30
MAX_INTERNAL_PLACES_PER_KIND = 100
MAX_OVERPASS_ELEMENTS = 100
MAX_NOMINATIM_BYTES = 1_000_000
MAX_OVERPASS_BYTES = 5_000_000
MAX_ROUTING_BYTES = 1_000_000
MAX_TRANSITOUS_BYTES = 2_000_000
MAX_OURAIRPORTS_BYTES = 30_000_000
DEFAULT_TIMEOUT_SECONDS = 10.0
OVERPASS_REQUEST_TIMEOUT_SECONDS = 6.0
OVERPASS_QUERY_TIMEOUT_SECONDS = 5
OVERPASS_OPERATION_BUDGET_SECONDS = 24.0
ROUTING_REQUEST_TIMEOUT_SECONDS = 5.0
TRANSITOUS_REQUEST_TIMEOUT_SECONDS = 8.0
TRANSITOUS_TRANSIT_MODES = (
    "TRAM,SUBWAY,FERRY,SUBURBAN,BUS,COACH,REGIONAL_RAIL,"
    "REGIONAL_FAST_RAIL,HIGHSPEED_RAIL,LONG_DISTANCE,NIGHT_RAIL,"
    "CABLE_CAR,FUNICULAR,AERIAL_LIFT,OTHER"
)
USER_AGENT = (
    "flight-forecast-lab/0.2 destination-guide "
    "(https://github.com/dingzhixuan2007wuliaoren/flight-forecast-lab)"
)


class DestinationGuideError(RuntimeError):
    """Base class for safe destination-guide failures."""


class DestinationAirportNotFound(DestinationGuideError):
    """Raised when a three-letter destination airport cannot be resolved."""


class DestinationDataUnavailable(DestinationGuideError):
    """Raised when a required public-data response is unavailable or invalid."""


class DestinationPlaceNotFound(DestinationGuideError):
    """Raised when a safe place identifier is absent from the destination result."""


class DestinationValidationError(DestinationGuideError, ValueError):
    """Raised for invalid destination-guide inputs."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DestinationCity(_StrictModel):
    destination_airport: str = Field(pattern=r"^[A-Z]{3}$")
    airport_name: str = Field(min_length=1, max_length=300)
    served_city: str | None = Field(default=None, min_length=1, max_length=200)
    city_query: str = Field(min_length=1, max_length=260)
    name: str = Field(min_length=1, max_length=200)
    country_code: str = Field(pattern=r"^[A-Z]{2}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    scope: Literal["served_city", "airport_surroundings"]
    scope_notice: str = Field(min_length=1, max_length=500)
    source: Literal[
        "ourairports_municipality+nominatim",
        "nominatim_reverse_airport",
        "airport_coordinate_fallback",
    ]


class DestinationCoverageNotice(_StrictModel):
    zh: str = Field(min_length=1, max_length=600)
    en: str = Field(min_length=1, max_length=600)


class DestinationCoverage(_StrictModel):
    coverage_radius_km: Literal[5, 15, 30]
    coverage_status: Literal["complete", "partial"]
    coverage_reason: Literal[
        "full_radius_queried",
        "result_target_reached",
        "provider_failure",
    ]
    partial: bool
    coverage_notice: DestinationCoverageNotice

    @model_validator(mode="after")
    def coverage_is_consistent(self) -> DestinationCoverage:
        if self.coverage_status == "complete":
            if self.partial or self.coverage_radius_km != 30:
                raise ValueError("complete coverage requires a successful 30 km query")
            if self.coverage_reason != "full_radius_queried":
                raise ValueError("complete coverage reason must report the full radius")
        else:
            if not self.partial or self.coverage_radius_km >= 30:
                raise ValueError("partial coverage requires an actual radius below 30 km")
            if self.coverage_reason == "full_radius_queried":
                raise ValueError("partial coverage cannot report a full-radius reason")
        return self


class DestinationPlace(_StrictModel):
    place_id: str = Field(pattern=PLACE_ID_PATTERN.pattern)
    kind: PlaceKind
    category: PlaceCategory
    name: str = Field(min_length=1, max_length=300)
    name_en: str | None = Field(default=None, min_length=1, max_length=300)
    address: str | None = Field(default=None, min_length=1, max_length=700)
    description: str | None = Field(default=None, min_length=1, max_length=2_000)
    website: str | None = Field(default=None, min_length=9, max_length=2_048)
    phone: str | None = Field(default=None, min_length=1, max_length=100)
    opening_hours: str | None = Field(default=None, min_length=1, max_length=500)
    stars: float | None = Field(default=None, ge=1, le=5)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    distance_from_city_center_km: float = Field(ge=0, le=2_000)
    data_source: Literal["openstreetmap_overpass"] = "openstreetmap_overpass"
    source_url: str = Field(min_length=20, max_length=300)

    @model_validator(mode="after")
    def category_matches_kind(self) -> DestinationPlace:
        allowed = ATTRACTION_CATEGORIES if self.kind == "attraction" else HOTEL_CATEGORIES
        if self.category not in allowed:
            raise ValueError("place category does not match place kind")
        if self.kind != "hotel" and self.stars is not None:
            raise ValueError("stars are only valid for hotel places")
        if self.website is not None and _safe_public_https_url(self.website) is None:
            raise ValueError("website must be a safe public HTTPS URL")
        if not _is_osm_element_url(self.source_url):
            raise ValueError("source_url must identify the corresponding OpenStreetMap element")
        return self


class DestinationPlaceList(_StrictModel):
    city: DestinationCity
    kind: PlaceKind
    category: ListCategory
    places: tuple[DestinationPlace, ...] = Field(max_length=MAX_PLACES_PER_KIND)
    result_count: int = Field(ge=0, le=MAX_PLACES_PER_KIND)
    fetched_at: datetime
    expires_at: datetime
    coverage_radius_km: Literal[5, 15, 30]
    coverage_status: Literal["complete", "partial"]
    coverage_reason: Literal[
        "full_radius_queried",
        "result_target_reached",
        "provider_failure",
    ]
    partial: bool
    coverage_notice: DestinationCoverageNotice
    data_source: Literal["openstreetmap_overpass"] = "openstreetmap_overpass"

    @model_validator(mode="after")
    def list_is_consistent(self) -> DestinationPlaceList:
        allowed = ATTRACTION_CATEGORIES if self.kind == "attraction" else HOTEL_CATEGORIES
        if self.category != "all" and self.category not in allowed:
            raise ValueError("list category does not match place kind")
        if self.result_count != len(self.places):
            raise ValueError("result_count must equal the number of places")
        if any(place.kind != self.kind for place in self.places):
            raise ValueError("all places must match the list kind")
        if self.category != "all" and any(
            place.category != self.category for place in self.places
        ):
            raise ValueError("all places must match the selected category")
        if self.fetched_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("cache timestamps must include timezone information")
        if self.expires_at <= self.fetched_at:
            raise ValueError("expires_at must be later than fetched_at")
        DestinationCoverage(
            coverage_radius_km=self.coverage_radius_km,
            coverage_status=self.coverage_status,
            coverage_reason=self.coverage_reason,
            partial=self.partial,
            coverage_notice=self.coverage_notice,
        )
        return self


class DestinationTransitLeg(_StrictModel):
    """One provider-returned leg in a public-transport itinerary."""

    mode: TransitLegMode
    from_name: str = Field(min_length=1, max_length=300)
    to_name: str = Field(min_length=1, max_length=300)
    from_timezone: str | None = Field(default=None, min_length=1, max_length=100)
    to_timezone: str | None = Field(default=None, min_length=1, max_length=100)
    departure_at: datetime
    arrival_at: datetime
    scheduled_departure_at: datetime
    scheduled_arrival_at: datetime
    duration_minutes: int = Field(ge=1, le=10_000)
    distance_km: float | None = Field(default=None, ge=0, le=10_000)
    line_name: str | None = Field(default=None, min_length=1, max_length=300)
    headsign: str | None = Field(default=None, min_length=1, max_length=300)
    agency_name: str | None = Field(default=None, min_length=1, max_length=300)
    intermediate_stops: tuple[str, ...] = Field(default=(), max_length=100)
    realtime: bool
    scheduled: bool

    @model_validator(mode="after")
    def leg_is_chronological(self) -> DestinationTransitLeg:
        if (
            self.departure_at.tzinfo is None
            or self.arrival_at.tzinfo is None
            or self.scheduled_departure_at.tzinfo is None
            or self.scheduled_arrival_at.tzinfo is None
            or self.departure_at.utcoffset() is None
            or self.arrival_at.utcoffset() is None
            or self.scheduled_departure_at.utcoffset() is None
            or self.scheduled_arrival_at.utcoffset() is None
        ):
            raise ValueError("transit leg timestamps must include timezone information")
        if self.arrival_at <= self.departure_at:
            raise ValueError("transit leg arrival must be after departure")
        if self.scheduled_arrival_at <= self.scheduled_departure_at:
            raise ValueError("scheduled transit leg arrival must be after departure")
        return self


class DestinationTransportOption(_StrictModel):
    mode: TransportMode
    status: Literal["available", "unavailable"]
    distance_km: float | None = Field(default=None, ge=0, le=10_000)
    duration_minutes: int | None = Field(default=None, ge=1, le=100_000)
    duration_basis: Literal[
        "estimated_route_no_live_traffic",
        "transit_schedule_or_realtime",
    ] | None = None
    requested_departure_at: datetime | None = None
    departure_time_basis: Literal["user_supplied", "request_time"] | None = None
    departure_at: datetime | None = None
    arrival_at: datetime | None = None
    transfers: int | None = Field(default=None, ge=0, le=20)
    realtime: bool | None = None
    legs: tuple[DestinationTransitLeg, ...] = Field(default=(), max_length=20)
    coverage_status: Literal[
        "covered",
        "no_itinerary",
        "provider_unavailable",
    ] | None = None
    notice: str = Field(min_length=1, max_length=600)
    data_source: Literal[
        "routing_openstreetmap_de_osrm",
        "transitous_motis",
        "open_transit_coverage_unavailable",
    ]
    source_url: str | None = Field(default=None, min_length=20, max_length=300)
    observed_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def route_is_consistent(self) -> DestinationTransportOption:
        if self.observed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("route timestamps must include timezone information")
        if self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be later than observed_at")
        if self.status == "available":
            if self.mode == "public_transit":
                if self.duration_minutes is None:
                    raise ValueError("available public transit requires a duration")
                if self.duration_basis != "transit_schedule_or_realtime":
                    raise ValueError("available public transit requires a timetable duration basis")
                if (
                    self.requested_departure_at is None
                    or self.departure_time_basis is None
                    or self.departure_at is None
                    or self.arrival_at is None
                    or self.transfers is None
                    or self.realtime is None
                    or not self.legs
                ):
                    raise ValueError("available public transit requires complete itinerary data")
                if self.arrival_at <= self.departure_at:
                    raise ValueError("public-transit arrival must be after departure")
                if not any(leg.mode != "WALK" for leg in self.legs):
                    raise ValueError("public transit requires at least one transit leg")
                if self.data_source != "transitous_motis":
                    raise ValueError("available public transit must come from Transitous")
                if (
                    self.coverage_status != "covered"
                    or self.source_url != TRANSITOUS_SOURCES_URL
                ):
                    raise ValueError("available public transit requires source and coverage")
            else:
                if self.distance_km is None or self.duration_minutes is None:
                    raise ValueError("available route requires distance and duration")
                if self.duration_basis != "estimated_route_no_live_traffic":
                    raise ValueError(
                        "available route requires an explicit estimated duration basis"
                    )
                if self.data_source != "routing_openstreetmap_de_osrm":
                    raise ValueError("available route must come from the OSRM source")
                if any(
                    value is not None
                    for value in (
                        self.requested_departure_at,
                        self.departure_time_basis,
                        self.departure_at,
                        self.arrival_at,
                        self.transfers,
                        self.realtime,
                        self.coverage_status,
                        self.source_url,
                    )
                ) or self.legs:
                    raise ValueError("street routes cannot contain public-transit data")
        elif any(
            value is not None
            for value in (
                self.distance_km,
                self.duration_minutes,
                self.duration_basis,
                self.departure_at,
                self.arrival_at,
                self.transfers,
                self.realtime,
            )
        ):
            raise ValueError("unavailable routes cannot contain invented route metrics")
        elif self.legs:
            raise ValueError("unavailable routes cannot contain itinerary legs")
        if self.mode == "public_transit" and self.status == "unavailable":
            if self.data_source == "transitous_motis":
                if (
                    self.requested_departure_at is None
                    or self.departure_time_basis is None
                    or self.coverage_status not in {"no_itinerary", "provider_unavailable"}
                    or self.source_url != TRANSITOUS_SOURCES_URL
                ):
                    raise ValueError("Transitous unavailability requires query coverage")
            elif (
                self.data_source != "open_transit_coverage_unavailable"
                or self.requested_departure_at is not None
                or self.departure_time_basis is not None
                or self.coverage_status is not None
                or self.source_url is not None
            ):
                raise ValueError("public-transit unavailability source is invalid")
        return self


class DestinationTransport(_StrictModel):
    destination_airport: str = Field(pattern=r"^[A-Z]{3}$")
    airport_name: str = Field(min_length=1, max_length=300)
    origin_latitude: float = Field(ge=-90, le=90)
    origin_longitude: float = Field(ge=-180, le=180)
    destination_latitude: float = Field(ge=-90, le=90)
    destination_longitude: float = Field(ge=-180, le=180)
    options: tuple[DestinationTransportOption, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def modes_are_complete(self) -> DestinationTransport:
        if tuple(option.mode for option in self.options) != (
            "car",
            "bike",
            "foot",
            "public_transit",
        ):
            raise ValueError("transport options must contain car, bike, foot, and public transit")
        return self


class DestinationPlaceDetail(_StrictModel):
    city: DestinationCity
    place: DestinationPlace
    transport: DestinationTransport


class JsonHttpTransport(Protocol):
    """Injectable, bounded JSON transport used by all destination HTTP providers."""

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> Any: ...


class _NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class BoundedJsonHttpClient:
    """Small urllib client with an exact HTTPS host/path allowlist and byte limits."""

    def __init__(self, opener: Any | None = None) -> None:
        self._opener = opener or request.build_opener(_NoRedirectHandler())

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> Any:
        normalized_method = method.strip().upper()
        if normalized_method not in {"GET", "POST"}:
            raise DestinationValidationError("destination HTTP method must be GET or POST")
        _assert_allowed_outbound_url(url)
        if not 0 < timeout_seconds <= 15:
            raise DestinationValidationError("destination HTTP timeout must be at most 15 seconds")
        if not 1 <= max_response_bytes <= MAX_OURAIRPORTS_BYTES:
            raise DestinationValidationError("destination response limit is outside safe bounds")
        if normalized_method == "GET" and data:
            raise DestinationValidationError("GET destination requests cannot contain form data")

        query = parse.urlencode(dict(params or {}), doseq=False)
        request_url = f"{url}?{query}" if query else url
        if len(request_url) > 8_192:
            raise DestinationValidationError("destination request URL exceeds the safety limit")
        body = None
        outbound_headers = dict(headers or {})
        if data is not None:
            body = parse.urlencode(dict(data), doseq=False).encode("utf-8")
            if len(body) > 100_000:
                raise DestinationValidationError(
                    "destination request body exceeds the safety limit"
                )
            outbound_headers.setdefault(
                "Content-Type", "application/x-www-form-urlencoded; charset=utf-8"
            )
        outbound = request.Request(
            request_url,
            data=body,
            headers=outbound_headers,
            method=normalized_method,
        )
        try:
            with self._opener.open(outbound, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                _assert_allowed_outbound_url(final_url.split("?", 1)[0])
                status = int(response.getcode())
                if status < 200 or status >= 300:
                    raise DestinationDataUnavailable(
                        "destination provider returned a non-success status"
                    )
                payload = response.read(max_response_bytes + 1)
        except (error.HTTPError, error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise DestinationDataUnavailable("destination provider request failed") from exc
        if len(payload) > max_response_bytes:
            raise DestinationDataUnavailable(
                "destination provider response exceeded the byte limit"
            )
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DestinationDataUnavailable("destination provider returned invalid JSON") from exc


MunicipalityResolver = Callable[[str], str | None]
BinaryDownloader = Callable[[str, float], bytes]


def _download_ourairports_municipalities(url: str, timeout_seconds: float) -> bytes:
    _assert_allowed_outbound_url(url)
    outbound = request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(outbound, timeout=timeout_seconds) as response:
            _assert_allowed_outbound_url(response.geturl().split("?", 1)[0])
            payload = response.read(MAX_OURAIRPORTS_BYTES + 1)
    except (error.HTTPError, error.URLError, OSError, TimeoutError) as exc:
        raise DestinationDataUnavailable("OurAirports municipality download failed") from exc
    if len(payload) > MAX_OURAIRPORTS_BYTES:
        raise DestinationDataUnavailable("OurAirports municipality response exceeded 30 MB")
    return payload


class OurAirportsMunicipalityResolver:
    """Lazy IATA-to-served-municipality cache for the OurAirports CSV."""

    def __init__(
        self,
        *,
        url: str = OURAIRPORTS_CSV_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        downloader: BinaryDownloader = _download_ourairports_municipalities,
        cache_ttl_seconds: float = DESTINATION_CACHE_TTL.total_seconds(),
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _assert_allowed_outbound_url(url)
        if not 0 < timeout_seconds <= 15:
            raise ValueError("timeout_seconds must be greater than 0 and at most 15")
        if cache_ttl_seconds != DESTINATION_CACHE_TTL.total_seconds():
            raise ValueError("municipality cache TTL must be exactly 24 hours")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._downloader = downloader
        self._clock = monotonic_clock
        self._municipalities: dict[str, str] = {}
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def __call__(self, code: str) -> str | None:
        normalized = _normalize_iata(code)
        now = self._clock()
        if now >= self._expires_at:
            self._reload(now)
        return self._municipalities.get(normalized)

    def _reload(self, observed_at: float) -> None:
        with self._lock:
            if observed_at < self._expires_at:
                return
            municipalities: dict[str, str] = {}
            try:
                payload = self._downloader(self.url, self.timeout_seconds)
                if len(payload) > MAX_OURAIRPORTS_BYTES:
                    raise DestinationDataUnavailable(
                        "OurAirports municipality response exceeded 30 MB"
                    )
                text = payload.decode("utf-8-sig")
                rows = csv.DictReader(StringIO(text))
                required = {"iata_code", "municipality"}
                if rows.fieldnames is None or not required.issubset(rows.fieldnames):
                    raise DestinationDataUnavailable(
                        "OurAirports municipality CSV is missing required columns"
                    )
                for row in rows:
                    code = str(row.get("iata_code") or "").strip().upper()
                    municipality = _clean_text(row.get("municipality"), 200)
                    if re.fullmatch(r"[A-Z]{3}", code) and municipality:
                        municipalities[code] = municipality
            except (DestinationGuideError, UnicodeError, csv.Error, TypeError, ValueError):
                municipalities = {}
            self._municipalities = municipalities
            self._expires_at = observed_at + DESTINATION_CACHE_TTL.total_seconds()


class _OneRequestPerSecond:
    def __init__(
        self,
        *,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self._clock = clock
        self._sleeper = sleeper
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last_request_at is not None:
                delay = 1.0 - (now - self._last_request_at)
                if delay > 0:
                    self._sleeper(delay)
            self._last_request_at = self._clock()


class _TtlCache:
    def __init__(
        self,
        clock: Callable[[], float],
        *,
        ttl: timedelta = DESTINATION_CACHE_TTL,
        max_entries: int | None = None,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("cache TTL must be positive")
        if max_entries is not None and max_entries < 1:
            raise ValueError("cache max_entries must be positive")
        self._clock = clock
        self._ttl_seconds = ttl.total_seconds()
        self._max_entries = max_entries
        self._values: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> Any | None:
        now = self._clock()
        with self._lock:
            cached = self._values.get(key)
            if cached is None:
                return None
            expires_at, value = cached
            if now >= expires_at:
                self._values.pop(key, None)
                return None
            return value

    def set(self, key: tuple[Any, ...], value: Any) -> None:
        with self._lock:
            now = self._clock()
            expired = [
                existing_key
                for existing_key, (expires_at, _) in self._values.items()
                if now >= expires_at
            ]
            for existing_key in expired:
                self._values.pop(existing_key, None)
            self._values.pop(key, None)
            while self._max_entries is not None and len(self._values) >= self._max_entries:
                self._values.pop(next(iter(self._values)))
            self._values[key] = (
                now + self._ttl_seconds,
                value,
            )


class DestinationGuideService:
    """Resolve destination cities, real OSM places, and estimated access routes."""

    def __init__(
        self,
        *,
        client: JsonHttpTransport | None = None,
        airport_resolver: AirportResolver | None = None,
        municipality_resolver: MunicipalityResolver | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < timeout_seconds <= 15:
            raise ValueError("timeout_seconds must be greater than 0 and at most 15")
        self.client = client or BoundedJsonHttpClient()
        self.airport_resolver = airport_resolver or OurAirportsResolver(
            timeout_seconds=min(timeout_seconds, 10)
        )
        self.municipality_resolver = (
            municipality_resolver or OurAirportsMunicipalityResolver(
                timeout_seconds=timeout_seconds,
                monotonic_clock=monotonic_clock,
            )
        )
        self.timeout_seconds = timeout_seconds
        self._clock = monotonic_clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._nominatim_limiter = _OneRequestPerSecond(clock=monotonic_clock, sleeper=sleeper)
        self._overpass_limiter = _OneRequestPerSecond(clock=monotonic_clock, sleeper=sleeper)
        self._routing_limiter = _OneRequestPerSecond(clock=monotonic_clock, sleeper=sleeper)
        self._transitous_limiter = _OneRequestPerSecond(
            clock=monotonic_clock,
            sleeper=sleeper,
        )
        self._city_cache = _TtlCache(monotonic_clock)
        self._place_cache = _TtlCache(monotonic_clock)
        self._route_cache = _TtlCache(monotonic_clock)
        self._transit_cache = _TtlCache(
            monotonic_clock,
            ttl=TRANSIT_CACHE_TTL,
            max_entries=TRANSIT_CACHE_MAX_ENTRIES,
        )

    def resolve_city(self, destination_airport: str) -> DestinationCity:
        code = _normalize_iata(destination_airport)
        cached = self._city_cache.get((code,))
        if isinstance(cached, DestinationCity):
            return cached
        airport = self._resolve_airport(code)
        served_city = self._safe_municipality(code)
        city_query = _city_query(served_city, airport)

        city = None
        if served_city:
            city = self._search_served_city(airport, served_city, city_query)
        if city is None:
            city = self._reverse_airport_city(airport, served_city, city_query)
        self._city_cache.set((code,), city)
        return city

    def list_places(
        self,
        destination_airport: str,
        kind: PlaceKind,
        category: ListCategory = "all",
        limit: int = MAX_PLACES_PER_KIND,
    ) -> DestinationPlaceList:
        code = _normalize_iata(destination_airport)
        normalized_kind = _validate_kind(kind)
        normalized_category = _validate_category(normalized_kind, category)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 30:
            raise DestinationValidationError("limit must be an integer from 1 through 30")
        city = self.resolve_city(code)
        all_places, fetched_at, expires_at, coverage = self._load_places(
            code,
            normalized_kind,
            city,
        )
        if normalized_category == "all":
            selected = _balanced_places(all_places, normalized_kind, limit)
        else:
            filtered = tuple(
                place for place in all_places if place.category == normalized_category
            )
            selected = filtered[:limit]
        return DestinationPlaceList(
            city=city,
            kind=normalized_kind,
            category=normalized_category,
            places=selected,
            result_count=len(selected),
            fetched_at=fetched_at,
            expires_at=expires_at,
            coverage_radius_km=coverage.coverage_radius_km,
            coverage_status=coverage.coverage_status,
            coverage_reason=coverage.coverage_reason,
            partial=coverage.partial,
            coverage_notice=coverage.coverage_notice,
        )

    def get_place_detail(
        self,
        destination_airport: str,
        place_id: str,
        transit_departure_at: datetime | None = None,
    ) -> DestinationPlaceDetail:
        code = _normalize_iata(destination_airport)
        match = PLACE_ID_PATTERN.fullmatch(str(place_id).strip())
        if match is None:
            raise DestinationValidationError("place_id is not a valid OSM destination identifier")
        kind = match.group(1)
        city = self.resolve_city(code)
        places, _, _, _ = self._load_places(code, kind, city)  # type: ignore[arg-type]
        place = next((item for item in places if item.place_id == place_id), None)
        if place is None:
            raise DestinationPlaceNotFound("place is not present in the destination result")
        return DestinationPlaceDetail(
            city=city,
            place=place,
            transport=self.get_routes(
                code,
                place.latitude,
                place.longitude,
                transit_departure_at=transit_departure_at,
            ),
        )

    def get_routes(
        self,
        destination_airport: str,
        latitude: float,
        longitude: float,
        *,
        transit_departure_at: datetime | None = None,
    ) -> DestinationTransport:
        code = _normalize_iata(destination_airport)
        airport = self._resolve_airport(code)
        destination_latitude = _coordinate(latitude, "latitude", -90, 90)
        destination_longitude = _coordinate(longitude, "longitude", -180, 180)
        routes = tuple(
            self._route_option(
                mode,
                airport,
                destination_latitude,
                destination_longitude,
            )
            for mode in ("car", "bike", "foot")
        )
        transit = self._transit_option(
            airport,
            destination_latitude,
            destination_longitude,
            transit_departure_at=transit_departure_at,
        )
        return DestinationTransport(
            destination_airport=airport.iata,
            airport_name=airport.name,
            origin_latitude=airport.latitude,
            origin_longitude=airport.longitude,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
            options=(*routes, transit),
        )

    def _resolve_airport(self, code: str) -> Airport:
        airport = lookup_airport(code, self.airport_resolver)
        if airport is None:
            raise DestinationAirportNotFound(
                "destination airport is not available in built-in or OurAirports data"
            )
        return airport

    def _safe_municipality(self, code: str) -> str | None:
        try:
            return _clean_text(self.municipality_resolver(code), 200)
        except (DestinationGuideError, OSError, TimeoutError, TypeError, ValueError):
            return None

    def _search_served_city(
        self,
        airport: Airport,
        served_city: str,
        city_query: str,
    ) -> DestinationCity | None:
        try:
            self._nominatim_limiter.wait()
            payload = self.client.request_json(
                "GET",
                NOMINATIM_SEARCH_URL,
                params={
                    "q": city_query,
                    "countrycodes": airport.country.lower(),
                    "format": "jsonv2",
                    "addressdetails": "1",
                    "limit": "5",
                },
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                timeout_seconds=self.timeout_seconds,
                max_response_bytes=MAX_NOMINATIM_BYTES,
            )
        except DestinationGuideError:
            return None
        candidate = _select_city_candidate(payload, served_city, airport.country)
        if candidate is None:
            return None
        latitude, longitude, resolved_name = candidate
        return DestinationCity(
            destination_airport=airport.iata,
            airport_name=airport.name,
            served_city=served_city,
            city_query=city_query,
            name=resolved_name,
            country_code=airport.country,
            latitude=latitude,
            longitude=longitude,
            scope="served_city",
            scope_notice=(
                "The served-city centre is verified. Actual queried coverage is reported "
                "separately for every place-list response and never exceeds 30 km."
            ),
            source="ourairports_municipality+nominatim",
        )

    def _reverse_airport_city(
        self,
        airport: Airport,
        served_city: str | None,
        city_query: str,
    ) -> DestinationCity:
        payload: Any = None
        try:
            self._nominatim_limiter.wait()
            payload = self.client.request_json(
                "GET",
                NOMINATIM_REVERSE_URL,
                params={
                    "lat": f"{airport.latitude:.7f}",
                    "lon": f"{airport.longitude:.7f}",
                    "format": "jsonv2",
                    "zoom": "10",
                    "addressdetails": "1",
                },
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                timeout_seconds=self.timeout_seconds,
                max_response_bytes=MAX_NOMINATIM_BYTES,
            )
        except DestinationGuideError:
            payload = None
        reverse_name = _reverse_city_name(payload)
        name = served_city or reverse_name or airport.name
        source: Literal["nominatim_reverse_airport", "airport_coordinate_fallback"] = (
            "nominatim_reverse_airport" if reverse_name else "airport_coordinate_fallback"
        )
        return DestinationCity(
            destination_airport=airport.iata,
            airport_name=airport.name,
            served_city=served_city,
            city_query=city_query,
            name=name,
            country_code=airport.country,
            latitude=airport.latitude,
            longitude=airport.longitude,
            scope="airport_surroundings",
            scope_notice=(
                "The served-city centre could not be verified, so places are explicitly limited "
                "to airport surroundings. Actual queried coverage is reported separately for "
                "every place-list response and never exceeds 30 km; this is not labelled as "
                "full city coverage."
            ),
            source=source,
        )

    def _load_places(
        self,
        code: str,
        kind: PlaceKind,
        city: DestinationCity,
    ) -> tuple[
        tuple[DestinationPlace, ...],
        datetime,
        datetime,
        DestinationCoverage,
    ]:
        key = (code, kind, round(city.latitude, 6), round(city.longitude, 6), city.scope)
        cached = self._place_cache.get(key)
        if isinstance(cached, tuple) and len(cached) == 4:
            return cached
        by_id: dict[str, DestinationPlace] = {}
        operation_started = self._clock()
        fallback_used = False
        last_successful_radius_meters: int | None = None
        expansion_failed = False
        for radius_meters in OVERPASS_RADIUS_METERS:
            query = _overpass_query(
                kind,
                city.latitude,
                city.longitude,
                radius_meters=radius_meters,
            )
            radius_places: tuple[DestinationPlace, ...] | None = None
            last_error: DestinationGuideError | None = None
            endpoints = (
                (OVERPASS_URL,)
                if fallback_used
                else (OVERPASS_URL, OVERPASS_FALLBACK_URL)
            )
            for endpoint in endpoints:
                if endpoint == OVERPASS_FALLBACK_URL:
                    fallback_used = True
                try:
                    self._overpass_limiter.wait()
                    remaining_budget = OVERPASS_OPERATION_BUDGET_SECONDS - (
                        self._clock() - operation_started
                    )
                    if remaining_budget <= 0:
                        raise DestinationDataUnavailable(
                            "OpenStreetMap request budget was exhausted"
                        )
                    payload = self.client.request_json(
                        "POST",
                        endpoint,
                        data={"data": query},
                        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                        timeout_seconds=min(
                            self.timeout_seconds,
                            OVERPASS_REQUEST_TIMEOUT_SECONDS,
                            remaining_budget,
                        ),
                        max_response_bytes=MAX_OVERPASS_BYTES,
                    )
                    radius_places = _parse_overpass_places(
                        payload,
                        kind,
                        city.latitude,
                        city.longitude,
                        radius_meters=radius_meters,
                    )
                except DestinationGuideError as exc:
                    last_error = exc
                    continue
                break
            if radius_places is None:
                # A smaller-radius response is still real, source-backed data.  Keep it
                # instead of discarding useful places when only a wider expansion fails.
                if by_id:
                    expansion_failed = True
                    break
                raise DestinationDataUnavailable(
                    "OpenStreetMap places are temporarily unavailable"
                ) from last_error
            last_successful_radius_meters = radius_meters
            for place in radius_places:
                existing = by_id.get(place.place_id)
                if existing is None or _place_completeness(place) > _place_completeness(
                    existing
                ):
                    by_id[place.place_id] = place
            if len(by_id) >= MAX_PLACES_PER_KIND:
                break
        places = tuple(
            sorted(
                by_id.values(),
                key=lambda item: (
                    item.distance_from_city_center_km,
                    item.name.casefold(),
                    item.place_id,
                ),
            )[:MAX_INTERNAL_PLACES_PER_KIND]
        )
        fetched_at = _aware_utc(self._wall_clock())
        if last_successful_radius_meters is None:
            raise DestinationDataUnavailable("OpenStreetMap places returned no usable coverage")
        coverage = _destination_coverage(
            last_successful_radius_meters,
            expansion_failed=expansion_failed,
        )
        value = (places, fetched_at, fetched_at + DESTINATION_CACHE_TTL, coverage)
        self._place_cache.set(key, value)
        return value

    def _route_option(
        self,
        mode: Literal["car", "bike", "foot"],
        airport: Airport,
        latitude: float,
        longitude: float,
    ) -> DestinationTransportOption:
        key = (
            airport.iata,
            mode,
            round(latitude, 6),
            round(longitude, 6),
        )
        cached = self._route_cache.get(key)
        if isinstance(cached, DestinationTransportOption):
            return cached
        observed_at = _aware_utc(self._wall_clock())
        expires_at = observed_at + DESTINATION_CACHE_TTL
        server_profile = {"car": "routed-car", "bike": "routed-bike", "foot": "routed-foot"}[
            mode
        ]
        coordinates = (
            f"{airport.longitude:.6f},{airport.latitude:.6f};"
            f"{longitude:.6f},{latitude:.6f}"
        )
        url = f"{ROUTING_BASE_URL}/{server_profile}/route/v1/driving/{coordinates}"
        try:
            self._routing_limiter.wait()
            payload = self.client.request_json(
                "GET",
                url,
                params={"overview": "false", "steps": "false", "alternatives": "false"},
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                timeout_seconds=min(self.timeout_seconds, ROUTING_REQUEST_TIMEOUT_SECONDS),
                max_response_bytes=MAX_ROUTING_BYTES,
            )
            distance_km, duration_minutes = _parse_osrm_route(payload)
            result = DestinationTransportOption(
                mode=mode,
                status="available",
                distance_km=distance_km,
                duration_minutes=duration_minutes,
                duration_basis="estimated_route_no_live_traffic",
                notice=(
                    "Estimated route duration from routing.openstreetmap.de; it does not include "
                    "live traffic, transit schedules, queues, or airport walking time."
                ),
                data_source="routing_openstreetmap_de_osrm",
                observed_at=observed_at,
                expires_at=expires_at,
            )
        except DestinationGuideError:
            result = DestinationTransportOption(
                mode=mode,
                status="unavailable",
                notice=(
                    "routing.openstreetmap.de did not return a valid route; no travel time is "
                    "inferred."
                ),
                data_source="routing_openstreetmap_de_osrm",
                observed_at=observed_at,
                expires_at=expires_at,
            )
        if result.status == "available":
            self._route_cache.set(key, result)
        return result

    def _transit_option(
        self,
        airport: Airport,
        latitude: float,
        longitude: float,
        *,
        transit_departure_at: datetime | None,
    ) -> DestinationTransportOption:
        observed_at = _aware_utc(self._wall_clock())
        requested_departure_at, departure_time_basis = _transit_departure(
            transit_departure_at,
            observed_at,
        )
        key = (
            airport.iata,
            "public_transit",
            round(latitude, 5),
            round(longitude, 5),
            requested_departure_at.isoformat(),
        )
        cached = self._transit_cache.get(key)
        if isinstance(cached, DestinationTransportOption):
            return cached
        expires_at = observed_at + TRANSIT_CACHE_TTL
        params = {
            "fromPlace": f"{airport.latitude:.6f},{airport.longitude:.6f}",
            "toPlace": f"{latitude:.6f},{longitude:.6f}",
            "time": requested_departure_at.isoformat(),
            "transitModes": TRANSITOUS_TRANSIT_MODES,
            # Disable the default direct WALK result. MOTIS can otherwise prune
            # slower transit itineraries in favour of walking, creating a false
            # "no public-transit itinerary" result.
            "directModes": "",
            "preTransitModes": "WALK",
            "postTransitModes": "WALK",
            "maxTransfers": "4",
            "maxTravelTime": "360",
            "searchWindow": "3600",
            "language": "en",
        }
        try:
            self._transitous_limiter.wait()
            payload = self.client.request_json(
                "GET",
                TRANSITOUS_PLAN_URL,
                params=params,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                timeout_seconds=min(
                    self.timeout_seconds,
                    TRANSITOUS_REQUEST_TIMEOUT_SECONDS,
                ),
                max_response_bytes=MAX_TRANSITOUS_BYTES,
            )
            result = _parse_transitous_option(
                payload,
                requested_departure_at=requested_departure_at,
                departure_time_basis=departure_time_basis,
                observed_at=observed_at,
                expires_at=expires_at,
            )
        except DestinationGuideError:
            result = DestinationTransportOption(
                mode="public_transit",
                status="unavailable",
                requested_departure_at=requested_departure_at,
                departure_time_basis=departure_time_basis,
                coverage_status="provider_unavailable",
                notice=(
                    "Transitous could not return a valid itinerary for the requested departure "
                    "time. A public-transport route or duration is not inferred."
                ),
                data_source="transitous_motis",
                source_url=TRANSITOUS_SOURCES_URL,
                observed_at=observed_at,
                expires_at=expires_at,
            )
        if result.coverage_status != "provider_unavailable":
            self._transit_cache.set(key, result)
        return result


def create_destination_guide_service(
    *,
    client: JsonHttpTransport | None = None,
    airport_resolver: AirportResolver | None = None,
    municipality_resolver: MunicipalityResolver | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> DestinationGuideService:
    """Create the production service; dependencies remain injectable for tests."""

    return DestinationGuideService(
        client=client,
        airport_resolver=airport_resolver,
        municipality_resolver=municipality_resolver,
        timeout_seconds=timeout_seconds,
    )


def build_destination_guide_service(
    *,
    client: JsonHttpTransport | None = None,
    airport_resolver: AirportResolver | None = None,
    municipality_resolver: MunicipalityResolver | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> DestinationGuideService:
    """Alias with the project's existing ``build_*`` factory naming convention."""

    return create_destination_guide_service(
        client=client,
        airport_resolver=airport_resolver,
        municipality_resolver=municipality_resolver,
        timeout_seconds=timeout_seconds,
    )


def _normalize_iata(value: object) -> str:
    if not isinstance(value, str):
        raise DestinationValidationError("destination airport must be a three-letter IATA code")
    code = value.strip().upper()
    if re.fullmatch(r"[A-Z]{3}", code) is None:
        raise DestinationValidationError("destination airport must be a three-letter IATA code")
    return code


def _validate_kind(value: object) -> PlaceKind:
    if value not in {"attraction", "hotel"}:
        raise DestinationValidationError("kind must be attraction or hotel")
    return value  # type: ignore[return-value]


def _validate_category(kind: PlaceKind, value: object) -> ListCategory:
    if not isinstance(value, str):
        raise DestinationValidationError("category must be a string")
    category = value.strip().lower()
    allowed = ATTRACTION_CATEGORIES if kind == "attraction" else HOTEL_CATEGORIES
    if category != "all" and category not in allowed:
        raise DestinationValidationError(f"category is not valid for {kind}")
    return category  # type: ignore[return-value]


def _balanced_places(
    places: tuple[DestinationPlace, ...],
    kind: PlaceKind,
    limit: int,
) -> tuple[DestinationPlace, ...]:
    category_order: tuple[PlaceCategory, ...] = (
        ATTRACTION_CATEGORY_ORDER if kind == "attraction" else HOTEL_CATEGORY_ORDER
    )
    buckets = {
        category: [place for place in places if place.category == category]
        for category in category_order
    }
    offsets = {category: 0 for category in category_order}
    selected: list[DestinationPlace] = []
    while len(selected) < limit:
        added = False
        for category in category_order:
            offset = offsets[category]
            bucket = buckets[category]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[category] = offset + 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.distance_from_city_center_km,
                item.name.casefold(),
                item.place_id,
            ),
        )
    )


def _destination_coverage(
    radius_meters: int,
    *,
    expansion_failed: bool,
) -> DestinationCoverage:
    radius_km = radius_meters // 1_000
    if radius_meters == DESTINATION_RADIUS_METERS and not expansion_failed:
        return DestinationCoverage(
            coverage_radius_km=30,
            coverage_status="complete",
            coverage_reason="full_radius_queried",
            partial=False,
            coverage_notice=DestinationCoverageNotice(
                zh=(
                    "已成功查询目的地中心周边 30 公里范围；结果仍受 OpenStreetMap "
                    "具名节点数据和内部最多 100 条记录限制。"
                ),
                en=(
                    "The full 30 km radius around the destination centre was queried "
                    "successfully; results remain limited to named OpenStreetMap nodes and "
                    "at most 100 internal records."
                ),
            ),
        )
    if expansion_failed:
        reason: Literal["result_target_reached", "provider_failure"] = "provider_failure"
        zh = (
            f"较大范围查询暂时失败；当前仅覆盖已成功查询的 {radius_km} 公里半径，"
            "不代表完整 30 公里覆盖。"
        )
        en = (
            f"A wider-radius query failed temporarily. Current results cover only the "
            f"successfully queried {radius_km} km radius, not the full 30 km radius."
        )
    else:
        reason = "result_target_reached"
        zh = (
            f"已在 {radius_km} 公里半径内取得至少 30 条真实来源记录；为减少公共 API "
            "负载未继续扩展，因此这不是完整 30 公里覆盖。"
        )
        en = (
            f"At least 30 source-backed records were found within {radius_km} km. The "
            "search was not expanded further to reduce public-API load, so this is not "
            "full 30 km coverage."
        )
    return DestinationCoverage(
        coverage_radius_km=radius_km,  # type: ignore[arg-type]
        coverage_status="partial",
        coverage_reason=reason,
        partial=True,
        coverage_notice=DestinationCoverageNotice(zh=zh, en=en),
    )


def _coordinate(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DestinationValidationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise DestinationValidationError(f"{label} is outside its valid range")
    return number


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DestinationValidationError("wall_clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _clean_text(value: object, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.replace("\x00", " ").split()).strip()
    if not cleaned:
        return None
    return cleaned[:maximum]


def _city_query(served_city: str | None, airport: Airport) -> str:
    if served_city:
        return f"{served_city}, {airport.country}"
    return f"{airport.name}, {airport.country}"


def _normalize_name(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


def _select_city_candidate(
    payload: Any,
    served_city: str,
    expected_country: str,
) -> tuple[float, float, str] | None:
    if not isinstance(payload, list):
        return None
    expected = _normalize_name(served_city)
    ranked: list[tuple[int, float, float, str]] = []
    for raw in payload[:5]:
        if not isinstance(raw, dict):
            continue
        address = raw.get("address") if isinstance(raw.get("address"), dict) else {}
        country_code = str(address.get("country_code") or "").strip().upper()
        if country_code and country_code != expected_country:
            continue
        latitude = _safe_float(raw.get("lat"), -90, 90)
        longitude = _safe_float(raw.get("lon"), -180, 180)
        if latitude is None or longitude is None:
            continue
        candidates = [
            raw.get("name"),
            address.get("city"),
            address.get("town"),
            address.get("municipality"),
            address.get("village"),
        ]
        names = [_clean_text(candidate, 200) for candidate in candidates]
        names = [name for name in names if name]
        display_name = _clean_text(raw.get("display_name"), 500)
        resolved_name = next(
            (name for name in names if _normalize_name(name) == expected),
            names[0] if names else served_city,
        )
        exact = any(_normalize_name(name) == expected for name in names)
        display_match = bool(display_name and expected in _normalize_name(display_name))
        place_type = str(raw.get("addresstype") or raw.get("type") or "").lower()
        type_score = place_type in {"city", "town", "municipality", "village", "administrative"}
        score = int(exact) * 4 + int(display_match) * 2 + int(type_score)
        ranked.append((score, latitude, longitude, resolved_name))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0]
    if best[0] < 2:
        return None
    return best[1], best[2], best[3]


def _reverse_city_name(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    address = payload.get("address")
    if not isinstance(address, dict):
        return None
    for key in ("city", "town", "municipality", "village", "county", "state"):
        name = _clean_text(address.get(key), 200)
        if name:
            return name
    return None


def _overpass_query(
    kind: PlaceKind,
    latitude: float,
    longitude: float,
    *,
    radius_meters: int,
) -> str:
    if radius_meters not in OVERPASS_RADIUS_METERS:
        raise DestinationValidationError("Overpass radius is outside the bounded query plan")
    south, west, north, east = _bounding_box(latitude, longitude, radius_meters)
    bounds = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    if kind == "hotel":
        selectors = [
            f'node({bounds})["tourism"="{category}"]["name"]'
            for category in HOTEL_CATEGORY_ORDER
        ]
    else:
        selectors = [
            f'node({bounds})["tourism"~"^(attraction|museum|gallery|theme_park|zoo|aquarium|viewpoint|artwork)$"]["name"]',
            f'node({bounds})["historic"]["name"]',
            f'node({bounds})["natural"~"^(beach|peak|waterfall|cave_entrance|spring)$"]["name"]',
            f'node({bounds})["leisure"~"^(park|garden|nature_reserve|water_park|amusement_arcade)$"]["name"]',
            f'node({bounds})["shop"~"^(mall|department_store)$"]["name"]',
            f'node({bounds})["amenity"~"^(marketplace|theatre|cinema)$"]["name"]',
            f'node({bounds})["man_made"~"^(lighthouse|tower)$"]["name"]',
        ]
    return (
        f"[out:json][timeout:{OVERPASS_QUERY_TIMEOUT_SECONDS}];("
        + ";".join(selectors)
        + f";);out body {MAX_OVERPASS_ELEMENTS};"
    )


def _bounding_box(
    latitude: float,
    longitude: float,
    radius_meters: int,
) -> tuple[float, float, float, float]:
    radius_km = radius_meters / 1_000
    earth_radius_km = 6_371.0088
    latitude_delta = math.degrees(radius_km / earth_radius_km)
    latitude_radians = math.radians(latitude)
    longitude_scale = max(abs(math.cos(latitude_radians)), 1e-6)
    longitude_delta = min(
        180.0,
        math.degrees(radius_km / (earth_radius_km * longitude_scale)),
    )
    return (
        max(-90.0, latitude - latitude_delta),
        max(-180.0, longitude - longitude_delta),
        min(90.0, latitude + latitude_delta),
        min(180.0, longitude + longitude_delta),
    )


def _parse_overpass_places(
    payload: Any,
    kind: PlaceKind,
    centre_latitude: float,
    centre_longitude: float,
    *,
    radius_meters: int,
) -> tuple[DestinationPlace, ...]:
    if not isinstance(payload, dict):
        raise DestinationDataUnavailable("Overpass response is not an object")
    if _clean_text(payload.get("remark"), 1_000) is not None:
        raise DestinationDataUnavailable("Overpass reported a provider runtime failure")
    if not isinstance(payload.get("elements"), list):
        raise DestinationDataUnavailable("Overpass response is missing an elements list")
    by_id: dict[str, DestinationPlace] = {}
    for raw in payload["elements"][:MAX_OVERPASS_ELEMENTS]:
        place = _parse_osm_element(raw, kind, centre_latitude, centre_longitude)
        if place is None:
            continue
        if (
            _great_circle_km(
                centre_latitude,
                centre_longitude,
                place.latitude,
                place.longitude,
            )
            > radius_meters / 1_000
        ):
            continue
        existing = by_id.get(place.place_id)
        if existing is None or _place_completeness(place) > _place_completeness(existing):
            by_id[place.place_id] = place
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (
                item.distance_from_city_center_km,
                item.name.casefold(),
                item.place_id,
            ),
        )[:MAX_INTERNAL_PLACES_PER_KIND]
    )


def _parse_osm_element(
    raw: Any,
    kind: PlaceKind,
    centre_latitude: float,
    centre_longitude: float,
) -> DestinationPlace | None:
    if not isinstance(raw, dict):
        return None
    element_type = str(raw.get("type") or "").lower()
    element_id = raw.get("id")
    if element_type not in {"node", "way", "relation"}:
        return None
    if (
        isinstance(element_id, bool)
        or not isinstance(element_id, int)
        or not 1 <= element_id <= 10**19 - 1
    ):
        return None
    tags = raw.get("tags")
    if not isinstance(tags, dict):
        return None
    name = _clean_text(tags.get("name"), 300) or _clean_text(tags.get("brand"), 300)
    if name is None:
        return None
    category = _classify_place(kind, tags)
    if category is None:
        return None
    coordinates = _element_coordinates(raw)
    if coordinates is None:
        return None
    latitude, longitude = coordinates
    place_id = _stable_place_id(kind, element_type, element_id)
    website = _safe_public_https_url(tags.get("contact:website") or tags.get("website"))
    stars = _parse_stars(tags.get("stars")) if kind == "hotel" else None
    source_url = f"{OPENSTREETMAP_BASE_URL}/{element_type}/{element_id}"
    try:
        return DestinationPlace(
            place_id=place_id,
            kind=kind,
            category=category,
            name=name,
            name_en=_clean_text(tags.get("name:en"), 300),
            address=_address_from_tags(tags),
            description=(
                _clean_text(tags.get("description"), 2_000)
                or _clean_text(tags.get("description:en"), 2_000)
            ),
            website=website,
            phone=(
                _clean_text(tags.get("contact:phone"), 100)
                or _clean_text(tags.get("phone"), 100)
            ),
            opening_hours=_clean_text(tags.get("opening_hours"), 500),
            stars=stars,
            latitude=latitude,
            longitude=longitude,
            distance_from_city_center_km=round(
                _great_circle_km(
                    centre_latitude,
                    centre_longitude,
                    latitude,
                    longitude,
                ),
                2,
            ),
            source_url=source_url,
        )
    except ValueError:
        return None


def _stable_place_id(kind: PlaceKind, element_type: str, element_id: int) -> str:
    # Only validated fixed labels and a positive OSM integer enter this identifier, so it
    # stays stable across cache refreshes and is safe to use as a URL/query parameter.
    return f"osm_{kind}_{element_type}_{element_id}"


def _classify_place(kind: PlaceKind, tags: Mapping[str, Any]) -> PlaceCategory | None:
    tourism = str(tags.get("tourism") or "").lower()
    if kind == "hotel":
        return tourism if tourism in HOTEL_CATEGORIES else None  # type: ignore[return-value]
    leisure = str(tags.get("leisure") or "").lower()
    amenity = str(tags.get("amenity") or "").lower()
    shop = str(tags.get("shop") or "").lower()
    natural = str(tags.get("natural") or "").lower()
    man_made = str(tags.get("man_made") or "").lower()
    if tourism in {"museum", "gallery"}:
        return "museum"
    if (
        tourism in {"theme_park", "zoo", "aquarium"}
        or leisure in {"water_park", "amusement_arcade"}
        or amenity in {"theatre", "cinema"}
    ):
        return "entertainment"
    if natural or leisure in {"park", "garden", "nature_reserve"}:
        return "nature"
    if shop in {"mall", "department_store"} or amenity == "marketplace":
        return "shopping"
    if (
        tags.get("historic") is not None
        or tourism in {"attraction", "viewpoint", "artwork"}
        or man_made in {"lighthouse", "tower"}
    ):
        return "landmark"
    return None


def _element_coordinates(raw: Mapping[str, Any]) -> tuple[float, float] | None:
    latitude = _safe_float(raw.get("lat"), -90, 90)
    longitude = _safe_float(raw.get("lon"), -180, 180)
    if latitude is not None and longitude is not None:
        return latitude, longitude
    centre = raw.get("center")
    if not isinstance(centre, dict):
        return None
    latitude = _safe_float(centre.get("lat"), -90, 90)
    longitude = _safe_float(centre.get("lon"), -180, 180)
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


def _safe_float(value: object, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        return None
    return result


def _address_from_tags(tags: Mapping[str, Any]) -> str | None:
    full = _clean_text(tags.get("addr:full"), 700)
    if full:
        return full
    house_number = _clean_text(tags.get("addr:housenumber"), 50)
    street = _clean_text(tags.get("addr:street"), 200)
    first_line = " ".join(part for part in (house_number, street) if part)
    city = _clean_text(tags.get("addr:city"), 150)
    postcode = _clean_text(tags.get("addr:postcode"), 50)
    state = _clean_text(tags.get("addr:state"), 150)
    country = _clean_text(tags.get("addr:country"), 100)
    parts = [part for part in (first_line, city, state, postcode, country) if part]
    return ", ".join(parts)[:700] or None


def _parse_stars(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        stars = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stars) or not 1 <= stars <= 5:
        return None
    return round(stars, 1)


def _safe_public_https_url(value: object) -> str | None:
    cleaned = _clean_text(value, 2_048)
    if cleaned is None:
        return None
    try:
        parsed = parse.urlsplit(cleaned)
        hostname = parsed.hostname
        if (
            parsed.scheme.lower() != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
        ):
            return None
    except ValueError:
        return None
    normalized_host = hostname.rstrip(".").lower()
    if normalized_host in {"localhost", "localhost.localdomain"} or normalized_host.endswith(
        ".local"
    ):
        return None
    try:
        address = ipaddress.ip_address(normalized_host.strip("[]"))
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return None
    return parse.urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def _is_osm_element_url(value: str) -> bool:
    try:
        parsed = parse.urlsplit(value)
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "www.openstreetmap.org"
        and parsed.port in {None, 443}
        and parsed.query == ""
        and parsed.fragment == ""
        and re.fullmatch(r"/(node|way|relation)/[1-9][0-9]{0,18}", parsed.path)
    )


def _place_completeness(place: DestinationPlace) -> int:
    return sum(
        value is not None
        for value in (
            place.name_en,
            place.address,
            place.description,
            place.website,
            place.phone,
            place.opening_hours,
            place.stars,
        )
    )


def _great_circle_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    radius_km = 6_371.0088
    first_latitude_radians = math.radians(first_latitude)
    second_latitude_radians = math.radians(second_latitude)
    latitude_delta = math.radians(second_latitude - first_latitude)
    longitude_delta = math.radians(second_longitude - first_longitude)
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude_radians)
        * math.cos(second_latitude_radians)
        * math.sin(longitude_delta / 2) ** 2
    )
    return radius_km * 2 * math.asin(math.sqrt(value))


def _parse_osrm_route(payload: Any) -> tuple[float, int]:
    if not isinstance(payload, dict) or payload.get("code") != "Ok":
        raise DestinationDataUnavailable("OSRM response did not report an available route")
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes or not isinstance(routes[0], dict):
        raise DestinationDataUnavailable("OSRM response is missing route metrics")
    distance_meters = _safe_float(routes[0].get("distance"), 0, 10_000_000)
    duration_seconds = _safe_float(routes[0].get("duration"), 1, 6_000_000)
    if distance_meters is None or duration_seconds is None:
        raise DestinationDataUnavailable("OSRM route metrics are invalid")
    return round(distance_meters / 1_000, 1), max(1, math.ceil(duration_seconds / 60))


def validate_transit_departure_at(
    requested: datetime | None,
    *,
    observed_at: datetime,
) -> datetime | None:
    """Validate and normalize a requested transit instant to UTC minute precision."""

    reference = _aware_utc(observed_at)
    if requested is None:
        return None
    if (
        not isinstance(requested, datetime)
        or requested.tzinfo is None
        or requested.utcoffset() is None
    ):
        raise DestinationValidationError(
            "transit_departure_at must be a timezone-aware datetime"
        )
    normalized = requested.astimezone(UTC).replace(second=0, microsecond=0)
    if normalized < reference - timedelta(days=1):
        raise DestinationValidationError(
            "transit_departure_at cannot be more than one day in the past"
        )
    if normalized > reference + timedelta(days=370):
        raise DestinationValidationError(
            "transit_departure_at must be within 370 days"
        )
    return normalized


def _transit_departure(
    requested: datetime | None,
    observed_at: datetime,
) -> tuple[datetime, Literal["user_supplied", "request_time"]]:
    normalized = validate_transit_departure_at(requested, observed_at=observed_at)
    if normalized is None:
        return observed_at.replace(second=0, microsecond=0), "request_time"
    return normalized, "user_supplied"


def _parse_transit_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 80:
        return None
    try:
        parsed_value = value.strip().replace("Z", "+00:00")
        result = datetime.fromisoformat(parsed_value)
    except ValueError:
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    return result.astimezone(UTC)


def _bounded_int(value: object, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number < minimum or number > maximum:
        return None
    return number


def _transit_place_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return _clean_text(value.get("name"), 300)


def _transit_place_timezone(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return _clean_text(value.get("tz"), 100)


def _transit_intermediate_stops(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for raw in value[:100]:
        name = _transit_place_name(raw)
        if name is not None:
            names.append(name)
    return tuple(names)


def _parse_transit_leg(value: object) -> DestinationTransitLeg | None:
    if not isinstance(value, Mapping):
        return None
    mode = str(value.get("mode") or "").strip().upper()
    allowed_modes = {
        "WALK",
        "TRAM",
        "SUBWAY",
        "FERRY",
        "SUBURBAN",
        "BUS",
        "COACH",
        "RAIL",
        "HIGHSPEED_RAIL",
        "LONG_DISTANCE",
        "NIGHT_RAIL",
        "REGIONAL_FAST_RAIL",
        "REGIONAL_RAIL",
        "CABLE_CAR",
        "FUNICULAR",
        "AERIAL_LIFT",
        "OTHER",
    }
    if mode not in allowed_modes or value.get("cancelled") is True:
        return None
    from_name = _transit_place_name(value.get("from"))
    to_name = _transit_place_name(value.get("to"))
    departure_at = _parse_transit_datetime(value.get("startTime"))
    arrival_at = _parse_transit_datetime(value.get("endTime"))
    scheduled_departure_at = _parse_transit_datetime(value.get("scheduledStartTime"))
    scheduled_arrival_at = _parse_transit_datetime(value.get("scheduledEndTime"))
    duration_seconds = _safe_float(value.get("duration"), 1, 600_000)
    if (
        from_name is None
        or to_name is None
        or departure_at is None
        or arrival_at is None
        or scheduled_departure_at is None
        or scheduled_arrival_at is None
        or duration_seconds is None
        or not isinstance(value.get("realTime"), bool)
        or not isinstance(value.get("scheduled"), bool)
    ):
        return None
    distance_meters = _safe_float(value.get("distance"), 0, 10_000_000)
    line_name = (
        _clean_text(value.get("displayName"), 300)
        or _clean_text(value.get("routeShortName"), 300)
        or _clean_text(value.get("routeLongName"), 300)
        or _clean_text(value.get("tripShortName"), 300)
    )
    try:
        return DestinationTransitLeg(
            mode=mode,
            from_name=from_name,
            to_name=to_name,
            from_timezone=_transit_place_timezone(value.get("from")),
            to_timezone=_transit_place_timezone(value.get("to")),
            departure_at=departure_at,
            arrival_at=arrival_at,
            scheduled_departure_at=scheduled_departure_at,
            scheduled_arrival_at=scheduled_arrival_at,
            duration_minutes=max(1, math.ceil(duration_seconds / 60)),
            distance_km=(
                round(distance_meters / 1_000, 1)
                if distance_meters is not None
                else None
            ),
            line_name=line_name,
            headsign=_clean_text(value.get("headsign"), 300),
            agency_name=_clean_text(value.get("agencyName"), 300),
            intermediate_stops=_transit_intermediate_stops(
                value.get("intermediateStops")
            ),
            realtime=value["realTime"],
            scheduled=value["scheduled"],
        )
    except (TypeError, ValueError):
        return None


def _parse_transitous_option(
    payload: Any,
    *,
    requested_departure_at: datetime,
    departure_time_basis: Literal["user_supplied", "request_time"],
    observed_at: datetime,
    expires_at: datetime,
) -> DestinationTransportOption:
    if not isinstance(payload, Mapping):
        raise DestinationDataUnavailable("Transitous response is not an object")
    raw_itineraries = payload.get("itineraries")
    if not isinstance(raw_itineraries, list):
        raise DestinationDataUnavailable("Transitous response is missing itineraries")
    candidates: list[
        tuple[int, int, datetime, datetime, tuple[DestinationTransitLeg, ...], bool]
    ] = []
    for raw_itinerary in raw_itineraries[:20]:
        if not isinstance(raw_itinerary, Mapping):
            continue
        duration_seconds = _safe_float(raw_itinerary.get("duration"), 1, 6_000_000)
        start_time = _parse_transit_datetime(raw_itinerary.get("startTime"))
        end_time = _parse_transit_datetime(raw_itinerary.get("endTime"))
        transfers = _bounded_int(raw_itinerary.get("transfers"), 0, 20)
        raw_legs = raw_itinerary.get("legs")
        if (
            duration_seconds is None
            or start_time is None
            or end_time is None
            or end_time <= start_time
            or transfers is None
            or not isinstance(raw_legs, list)
            or not 1 <= len(raw_legs) <= 20
        ):
            continue
        parsed_legs = tuple(_parse_transit_leg(raw) for raw in raw_legs)
        if any(leg is None for leg in parsed_legs):
            continue
        legs = tuple(leg for leg in parsed_legs if leg is not None)
        transit_legs = tuple(leg for leg in legs if leg.mode != "WALK")
        if not transit_legs:
            continue
        duration_minutes = max(1, math.ceil(duration_seconds / 60))
        candidates.append(
            (
                duration_minutes,
                transfers,
                start_time,
                end_time,
                legs,
                any(leg.realtime for leg in transit_legs),
            )
        )
    if not raw_itineraries:
        return DestinationTransportOption(
            mode="public_transit",
            status="unavailable",
            requested_departure_at=requested_departure_at,
            departure_time_basis=departure_time_basis,
            coverage_status="no_itinerary",
            notice=(
                "Transitous returned no complete public-transport itinerary for the requested "
                "departure time and coordinates. A route or duration is not inferred."
            ),
            data_source="transitous_motis",
            source_url=TRANSITOUS_SOURCES_URL,
            observed_at=observed_at,
            expires_at=expires_at,
        )
    if not candidates:
        raise DestinationDataUnavailable(
            "Transitous returned itineraries but none passed strict validation"
        )
    (
        duration_minutes,
        transfers,
        departure_at,
        arrival_at,
        legs,
        realtime,
    ) = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return DestinationTransportOption(
        mode="public_transit",
        status="available",
        duration_minutes=duration_minutes,
        duration_basis="transit_schedule_or_realtime",
        requested_departure_at=requested_departure_at,
        departure_time_basis=departure_time_basis,
        departure_at=departure_at,
        arrival_at=arrival_at,
        transfers=transfers,
        realtime=realtime,
        legs=legs,
        coverage_status="covered",
        notice=(
            "Transitous MOTIS returned this itinerary for the requested departure time. "
            "Times follow published open timetables and provider realtime flags where present; "
            "coverage depends on the underlying local feeds."
        ),
        data_source="transitous_motis",
        source_url=TRANSITOUS_SOURCES_URL,
        observed_at=observed_at,
        expires_at=expires_at,
    )


def _assert_allowed_outbound_url(url: str) -> None:
    try:
        parsed = parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise DestinationValidationError("destination provider URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise DestinationValidationError("destination provider URL is outside the HTTPS allowlist")
    host = (parsed.hostname or "").lower()
    exact_paths = {
        ("nominatim.openstreetmap.org", "/search"),
        ("nominatim.openstreetmap.org", "/reverse"),
        ("overpass-api.de", "/api/interpreter"),
        ("maps.mail.ru", "/osm/tools/overpass/api/interpreter"),
        ("davidmegginson.github.io", "/ourairports-data/airports.csv"),
        ("api.transitous.org", "/api/v5/plan"),
    }
    if (host, parsed.path) in exact_paths:
        return
    if host == "routing.openstreetmap.de" and re.fullmatch(
        r"/routed-(car|bike|foot)/route/v1/driving/"
        r"-?[0-9]{1,3}\.[0-9]{6},-?[0-9]{1,2}\.[0-9]{6};"
        r"-?[0-9]{1,3}\.[0-9]{6},-?[0-9]{1,2}\.[0-9]{6}",
        parsed.path,
    ):
        return
    raise DestinationValidationError("destination provider URL is outside the HTTPS allowlist")
