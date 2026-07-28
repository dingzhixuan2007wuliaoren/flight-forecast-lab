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
import html
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
from flight_forecaster.serpapi_transit import (
    SERPAPI_TRANSIT_SOURCE_URL,
    SerpApiTransitDirectionsProvider,
    SerpApiTransitResult,
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
AttractionRatingStatus = Literal[
    "available",
    "source_not_provided",
    "provider_unavailable",
]
AttractionPhotoStatus = Literal[
    "available",
    "source_not_provided",
    "provider_unavailable",
    "source_rejected",
]
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

ATTRACTION_CATEGORIES = frozenset({"landmark", "museum", "nature", "entertainment", "shopping"})
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
PLACE_ID_PATTERN = re.compile(r"^osm_(attraction|hotel)_(node|way|relation)_([1-9][0-9]{0,18})$")

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
OVERPASS_ENDPOINTS = (OVERPASS_URL, OVERPASS_FALLBACK_URL)
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIMEDIA_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API_URLS = {
    "zh": "https://zh.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}
ROUTING_BASE_URL = "https://routing.openstreetmap.de"
TRANSITOUS_PLAN_URL = "https://api.transitous.org/api/v5/plan"
TRANSITOUS_SOURCES_URL = "https://transitous.org/sources/"
OPENSTREETMAP_BASE_URL = "https://www.openstreetmap.org"
OSM_TRANSIT_REFERENCE_SOURCE_URL = f"{OPENSTREETMAP_BASE_URL}/copyright"
DESTINATION_CACHE_TTL = timedelta(hours=24)
PARTIAL_DESTINATION_CACHE_TTL = timedelta(minutes=10)
TRANSIT_CACHE_TTL = timedelta(minutes=30)
TRANSIT_CACHE_MAX_ENTRIES = 512
DESTINATION_RADIUS_METERS = 30_000
OVERPASS_RADIUS_METERS = (5_000, 15_000, DESTINATION_RADIUS_METERS)
MAX_PLACES_PER_KIND = 300
MAX_INTERNAL_PLACES_PER_KIND = 300
DEFAULT_PLACES_PER_KIND = 30
HOTEL_RESULT_TARGET = 30
MAX_OVERPASS_ELEMENTS = 350
MAX_WIKIMEDIA_BATCH_SIZE = 50
MAX_WIKIPEDIA_TITLES_PER_LANGUAGE = 40
MAX_WIKIDATA_RATING_ISSUERS = 100
MAX_ATTRACTION_PHOTOS = 3
COMMONS_THUMBNAIL_WIDTH = 960
MAX_NOMINATIM_BYTES = 1_000_000
MAX_OVERPASS_BYTES = 5_000_000
MAX_WIKIMEDIA_BYTES = 3_000_000
MAX_ROUTING_BYTES = 1_000_000
MAX_TRANSITOUS_BYTES = 2_000_000
MAX_OURAIRPORTS_BYTES = 30_000_000
DEFAULT_TIMEOUT_SECONDS = 10.0
OVERPASS_REQUEST_TIMEOUT_SECONDS = 6.0
OVERPASS_QUERY_TIMEOUT_SECONDS = 5
OVERPASS_OPERATION_BUDGET_SECONDS = 24.0
ATTRACTION_OVERPASS_REQUEST_TIMEOUT_SECONDS = 9.0
ATTRACTION_OVERPASS_QUERY_TIMEOUT_SECONDS = 8
ATTRACTION_OVERPASS_OPERATION_BUDGET_SECONDS = 42.0
WIKIMEDIA_REQUEST_TIMEOUT_SECONDS = 8.0
WIKIMEDIA_OPERATION_BUDGET_SECONDS = 18.0
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


class AttractionRatingSourceCapability(_StrictModel):
    """A known rating-source capability, not evidence that a score was returned."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    display_name: str = Field(min_length=1, max_length=100)
    capability: Literal["ratings", "ratings_and_reviews"]
    adapter_status: Literal["active", "catalogued"]
    exact_match_requirement: Literal[
        "wikidata_subject_and_issuer",
        "provider_place_id",
    ]
    evidence_policy: Literal["capability_only_not_query_evidence"] = (
        "capability_only_not_query_evidence"
    )


ATTRACTION_RATING_SOURCE_REGISTRY: tuple[
    AttractionRatingSourceCapability,
    ...,
] = (
    AttractionRatingSourceCapability(
        key="wikidata_p444",
        display_name="Wikidata review score statements",
        capability="ratings",
        adapter_status="active",
        exact_match_requirement="wikidata_subject_and_issuer",
    ),
    AttractionRatingSourceCapability(
        key="google_places",
        display_name="Google Places",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="tripadvisor",
        display_name="Tripadvisor",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="yelp",
        display_name="Yelp",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="foursquare",
        display_name="Foursquare",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="viator",
        display_name="Viator",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="tiqets",
        display_name="Tiqets",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="klook",
        display_name="Klook",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="getyourguide",
        display_name="GetYourGuide",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="trip_com",
        display_name="Trip.com",
        capability="ratings_and_reviews",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="apple_maps",
        display_name="Apple Maps",
        capability="ratings",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
    AttractionRatingSourceCapability(
        key="bing_maps",
        display_name="Bing Maps",
        capability="ratings",
        adapter_status="catalogued",
        exact_match_requirement="provider_place_id",
    ),
)


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
        "result_limit_reached",
        "provider_failure",
    ]
    partial: bool
    coverage_notice: DestinationCoverageNotice
    query_parts_succeeded: int = Field(default=1, ge=0, le=16)
    query_parts_total: int = Field(default=1, ge=1, le=16)
    provider_truncated: bool = False

    @model_validator(mode="after")
    def coverage_is_consistent(self) -> DestinationCoverage:
        if self.query_parts_succeeded > self.query_parts_total:
            raise ValueError("successful query parts cannot exceed total query parts")
        if self.coverage_status == "complete":
            if self.partial or self.coverage_radius_km != 30:
                raise ValueError("complete coverage requires a successful 30 km query")
            if self.coverage_reason != "full_radius_queried":
                raise ValueError("complete coverage reason must report the full radius")
            if self.query_parts_succeeded != self.query_parts_total:
                raise ValueError("complete coverage requires every query part")
            if self.provider_truncated:
                raise ValueError("complete coverage cannot contain provider-truncated parts")
        else:
            if not self.partial:
                raise ValueError("partial coverage requires partial=true")
            if self.coverage_reason == "full_radius_queried":
                raise ValueError("partial coverage cannot report a full-radius reason")
        return self


class DestinationAttractionRating(_StrictModel):
    """A real review score attached to the same Wikidata entity as an OSM place."""

    platform: str = Field(min_length=1, max_length=200)
    platform_zh: str | None = Field(default=None, min_length=1, max_length=200)
    platform_en: str | None = Field(default=None, min_length=1, max_length=200)
    platform_id: str = Field(pattern=r"^Q[1-9][0-9]{0,18}$")
    score_text: str = Field(min_length=1, max_length=80)
    score: float | None = Field(default=None, ge=0, le=100_000)
    max_score: float | None = Field(default=None, gt=0, le=100_000)
    review_count: int | None = Field(default=None, ge=0, le=2_000_000_000)
    point_in_time: str | None = Field(default=None, min_length=4, max_length=40)
    source_url: str = Field(min_length=20, max_length=2_048)
    subject_wikidata_id: str = Field(pattern=r"^Q[1-9][0-9]{0,18}$")
    source_registry_key: Literal["wikidata_p444"] = "wikidata_p444"
    identity_match_basis: Literal["osm_wikidata_subject_and_p447_issuer",] = (
        "osm_wikidata_subject_and_p447_issuer"
    )
    source_status: Literal["source_returned_exact_match"] = "source_returned_exact_match"
    data_source: Literal["wikidata"] = "wikidata"

    @model_validator(mode="after")
    def score_is_consistent(self) -> DestinationAttractionRating:
        if (self.score is None) != (self.max_score is None):
            raise ValueError("numeric score and scale must be supplied together")
        if self.score is not None and self.max_score is not None and self.score > self.max_score:
            raise ValueError("numeric score must not exceed its scale")
        if _safe_public_https_url(self.source_url) is None:
            raise ValueError("rating source URL must be a safe public HTTPS URL")
        return self


class DestinationAttractionPhoto(_StrictModel):
    """A Commons photo with exact place linkage and reusable-license evidence."""

    file_title: str = Field(
        min_length=6,
        max_length=500,
        pattern=r"^File:.+",
    )
    image_url: str = Field(min_length=20, max_length=2_048)
    thumbnail_url: str = Field(min_length=20, max_length=2_048)
    source_page_url: str = Field(min_length=20, max_length=2_048)
    author: str = Field(min_length=1, max_length=500)
    license_name: str = Field(min_length=1, max_length=200)
    license_url: str = Field(min_length=20, max_length=2_048)
    attribution: str = Field(min_length=1, max_length=1_000)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    match_basis: Literal[
        "osm_wikimedia_commons",
        "wikidata_p18",
        "wikipedia_pageimage",
    ]
    source_status: Literal["verified_reusable_photo"] = "verified_reusable_photo"
    data_source: Literal["wikimedia_commons_imageinfo"] = "wikimedia_commons_imageinfo"

    @model_validator(mode="after")
    def photo_evidence_is_safe(self) -> DestinationAttractionPhoto:
        for value in (
            self.image_url,
            self.thumbnail_url,
            self.source_page_url,
            self.license_url,
        ):
            if _safe_public_https_url(value) is None:
                raise ValueError("photo evidence URLs must be safe public HTTPS URLs")
        if not _is_wikimedia_upload_url(self.image_url):
            raise ValueError("photo URL must identify a Wikimedia upload")
        if not _is_wikimedia_upload_url(self.thumbnail_url):
            raise ValueError("photo thumbnail must identify a Wikimedia upload")
        if not _is_wikimedia_commons_page(self.source_page_url):
            raise ValueError("photo source page must identify the Commons file")
        if (
            _commons_page_file_title(self.source_page_url) or ""
        ).casefold() != self.file_title.casefold():
            raise ValueError("photo source page must match the Commons file title")
        return self


class DestinationAttractionMapPreview(_StrictModel):
    """An exact-coordinate map fallback; deliberately not represented as a photo."""

    preview_kind: Literal["exact_coordinate_map"] = "exact_coordinate_map"
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    zoom: int = Field(default=15, ge=1, le=19)
    source_url: str = Field(min_length=20, max_length=2_048)
    data_source: Literal["openstreetmap_exact_coordinate_preview"] = (
        "openstreetmap_exact_coordinate_preview"
    )

    @model_validator(mode="after")
    def preview_matches_coordinates(self) -> DestinationAttractionMapPreview:
        expected = _osm_map_preview_url(self.latitude, self.longitude, self.zoom)
        if self.source_url != expected:
            raise ValueError("map preview URL must match the exact place coordinates")
        return self


class DestinationPlace(_StrictModel):
    place_id: str = Field(pattern=PLACE_ID_PATTERN.pattern)
    kind: PlaceKind
    category: PlaceCategory
    name: str = Field(min_length=1, max_length=300)
    name_en: str | None = Field(default=None, min_length=1, max_length=300)
    address: str | None = Field(default=None, min_length=1, max_length=700)
    description: str | None = Field(default=None, min_length=1, max_length=2_000)
    description_zh: str | None = Field(default=None, min_length=1, max_length=2_000)
    description_en: str | None = Field(default=None, min_length=1, max_length=2_000)
    description_source: Literal["openstreetmap", "wikidata", "wikipedia"] | None = None
    description_basis: (
        Literal[
            "osm_description",
            "wikidata_description",
            "wikipedia_extract",
            "osm_tag_summary",
        ]
        | None
    ) = None
    description_url: str | None = Field(default=None, min_length=20, max_length=2_048)
    wikidata_id: str | None = Field(
        default=None,
        pattern=r"^Q[1-9][0-9]{0,18}$",
    )
    wikipedia_url: str | None = Field(default=None, min_length=20, max_length=2_048)
    ratings: tuple[DestinationAttractionRating, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    ratings_status: AttractionRatingStatus = "source_not_provided"
    osm_wikimedia_commons_file: str | None = Field(
        default=None,
        min_length=6,
        max_length=500,
        pattern=r"^File:.+",
    )
    photos: tuple[DestinationAttractionPhoto, ...] = Field(
        default_factory=tuple,
        max_length=MAX_ATTRACTION_PHOTOS,
    )
    photo_status: AttractionPhotoStatus = "source_not_provided"
    map_preview: DestinationAttractionMapPreview | None = None
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
        if self.kind == "hotel" and (
            self.ratings
            or self.ratings_status != "source_not_provided"
            or self.wikidata_id is not None
            or self.wikipedia_url is not None
            or self.osm_wikimedia_commons_file is not None
            or self.photos
            or self.photo_status != "source_not_provided"
            or self.map_preview is not None
        ):
            raise ValueError("attraction enrichment fields are only valid for attractions")
        if self.ratings_status == "available" and not self.ratings:
            raise ValueError("available attraction ratings require at least one score")
        if self.ratings and self.ratings_status != "available":
            raise ValueError("attraction ratings require available status")
        if self.photo_status == "available" and not self.photos:
            raise ValueError("available attraction photos require photo evidence")
        if self.photos and self.photo_status != "available":
            raise ValueError("attraction photos require available status")
        if self.photos and self.map_preview is not None:
            raise ValueError("a verified photo and map fallback cannot be active together")
        if self.website is not None and _safe_public_https_url(self.website) is None:
            raise ValueError("website must be a safe public HTTPS URL")
        for url in (self.description_url, self.wikipedia_url):
            if url is not None and _safe_public_https_url(url) is None:
                raise ValueError("description links must be safe public HTTPS URLs")
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
        "result_limit_reached",
        "provider_failure",
    ]
    partial: bool
    coverage_notice: DestinationCoverageNotice
    query_parts_succeeded: int = Field(default=1, ge=0, le=16)
    query_parts_total: int = Field(default=1, ge=1, le=16)
    provider_truncated: bool = False
    attraction_rating_source_capabilities: tuple[
        AttractionRatingSourceCapability,
        ...,
    ] = Field(
        default=ATTRACTION_RATING_SOURCE_REGISTRY,
        min_length=10,
        max_length=20,
    )
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
        if self.category != "all" and any(place.category != self.category for place in self.places):
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
            query_parts_succeeded=self.query_parts_succeeded,
            query_parts_total=self.query_parts_total,
            provider_truncated=self.provider_truncated,
        )
        return self


class DestinationTransitLeg(_StrictModel):
    """One provider-returned leg in a public-transport itinerary."""

    mode: TransitLegMode
    from_name: str = Field(min_length=1, max_length=300)
    to_name: str = Field(min_length=1, max_length=300)
    from_timezone: str | None = Field(default=None, min_length=1, max_length=100)
    to_timezone: str | None = Field(default=None, min_length=1, max_length=100)
    departure_at: datetime | None = None
    arrival_at: datetime | None = None
    scheduled_departure_at: datetime | None = None
    scheduled_arrival_at: datetime | None = None
    departure_time_label: str | None = Field(default=None, min_length=1, max_length=80)
    arrival_time_label: str | None = Field(default=None, min_length=1, max_length=80)
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
        timestamps = (
            self.departure_at,
            self.arrival_at,
            self.scheduled_departure_at,
            self.scheduled_arrival_at,
        )
        if any(value is not None for value in timestamps):
            if any(value is None for value in timestamps):
                raise ValueError("transit leg timestamps must be complete")
            if any(
                value.tzinfo is None or value.utcoffset() is None
                for value in timestamps
                if value is not None
            ):
                raise ValueError("transit leg timestamps must include timezone information")
            assert self.departure_at is not None
            assert self.arrival_at is not None
            assert self.scheduled_departure_at is not None
            assert self.scheduled_arrival_at is not None
            if self.arrival_at <= self.departure_at:
                raise ValueError("transit leg arrival must be after departure")
            if self.scheduled_arrival_at <= self.scheduled_departure_at:
                raise ValueError("scheduled transit leg arrival must be after departure")
        elif (self.departure_time_label is None) != (self.arrival_time_label is None):
            raise ValueError("provider clock labels must be complete")
        return self


class DestinationTransitRouteReference(_StrictModel):
    """One real OSM route relation, explicitly not a timed itinerary."""

    relation_id: int = Field(ge=1, le=9_223_372_036_854_775_807)
    route_mode: Literal["bus", "train", "subway", "light_rail", "tram"]
    name: str | None = Field(default=None, min_length=1, max_length=300)
    ref: str | None = Field(default=None, min_length=1, max_length=100)
    operator: str | None = Field(default=None, min_length=1, max_length=300)
    network: str | None = Field(default=None, min_length=1, max_length=300)
    duration_label: str | None = Field(default=None, min_length=1, max_length=80)
    near_airport: bool
    near_destination: bool
    stops: tuple[str, ...] = Field(min_length=1, max_length=100)
    source_url: str = Field(min_length=20, max_length=300)
    data_source: Literal["openstreetmap_route_relation"] = "openstreetmap_route_relation"

    @model_validator(mode="after")
    def reference_is_source_backed(self) -> DestinationTransitRouteReference:
        if self.name is None and self.ref is None:
            raise ValueError("transit route reference requires a source name or ref")
        if not self.near_airport and not self.near_destination:
            raise ValueError("transit route reference must be related to an endpoint")
        if self.source_url != f"{OPENSTREETMAP_BASE_URL}/relation/{self.relation_id}":
            raise ValueError("transit route reference URL must match its OSM relation")
        return self


class DestinationTransportOption(_StrictModel):
    mode: TransportMode
    status: Literal["available", "unavailable"]
    distance_km: float | None = Field(default=None, ge=0, le=10_000)
    duration_minutes: int | None = Field(default=None, ge=1, le=100_000)
    duration_basis: (
        Literal[
            "estimated_route_no_live_traffic",
            "transit_schedule_or_realtime",
        ]
        | None
    ) = None
    requested_departure_at: datetime | None = None
    departure_time_basis: Literal["user_supplied", "request_time"] | None = None
    departure_at: datetime | None = None
    arrival_at: datetime | None = None
    departure_time_label: str | None = Field(default=None, min_length=1, max_length=80)
    arrival_time_label: str | None = Field(default=None, min_length=1, max_length=80)
    transfers: int | None = Field(default=None, ge=0, le=20)
    realtime: bool | None = None
    legs: tuple[DestinationTransitLeg, ...] = Field(default=(), max_length=20)
    route_references: tuple[DestinationTransitRouteReference, ...] = Field(
        default=(),
        max_length=12,
    )
    coverage_status: (
        Literal[
            "covered",
            "no_itinerary",
            "provider_unavailable",
            "not_configured",
            "authentication_failed",
            "quota_exhausted",
            "rate_limited",
            "provider_processing",
            "provider_error",
            "response_invalid",
            "quota_ledger_unavailable",
            "route_reference_only",
        ]
        | None
    ) = None
    notice: str = Field(min_length=1, max_length=600)
    data_source: Literal[
        "routing_openstreetmap_de_osrm",
        "transitous_motis",
        "serpapi_google_maps_directions",
        "openstreetmap_overpass_transit_reference",
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
                    or self.transfers is None
                    or self.realtime is None
                    or not self.legs
                ):
                    raise ValueError("available public transit requires complete itinerary data")
                has_exact_times = self.departure_at is not None and self.arrival_at is not None
                has_provider_labels = (
                    self.departure_time_label is not None and self.arrival_time_label is not None
                )
                if not has_exact_times and not has_provider_labels:
                    raise ValueError(
                        "available public transit requires exact times or provider clock labels"
                    )
                if (self.departure_at is None) != (self.arrival_at is None):
                    raise ValueError("public-transit timestamps must be complete")
                if (self.departure_time_label is None) != (self.arrival_time_label is None):
                    raise ValueError("public-transit clock labels must be complete")
                if (
                    self.departure_at is not None
                    and self.arrival_at is not None
                    and self.arrival_at <= self.departure_at
                ):
                    raise ValueError("public-transit arrival must be after departure")
                if not any(leg.mode != "WALK" for leg in self.legs):
                    raise ValueError("public transit requires at least one transit leg")
                if self.route_references:
                    raise ValueError("timed itineraries cannot contain route references")
                valid_sources = {
                    "transitous_motis": TRANSITOUS_SOURCES_URL,
                    "serpapi_google_maps_directions": SERPAPI_TRANSIT_SOURCE_URL,
                }
                if self.source_url != valid_sources.get(self.data_source):
                    raise ValueError("available public transit requires source and coverage")
                if self.coverage_status != "covered":
                    raise ValueError("available public transit requires covered status")
            else:
                if self.distance_km is None or self.duration_minutes is None:
                    raise ValueError("available route requires distance and duration")
                if self.duration_basis != "estimated_route_no_live_traffic":
                    raise ValueError(
                        "available route requires an explicit estimated duration basis"
                    )
                if self.data_source != "routing_openstreetmap_de_osrm":
                    raise ValueError("available route must come from the OSRM source")
                if (
                    any(
                        value is not None
                        for value in (
                            self.requested_departure_at,
                            self.departure_time_basis,
                            self.departure_at,
                            self.arrival_at,
                            self.departure_time_label,
                            self.arrival_time_label,
                            self.transfers,
                            self.realtime,
                            self.coverage_status,
                            self.source_url,
                        )
                    )
                    or self.legs
                    or self.route_references
                ):
                    raise ValueError("street routes cannot contain public-transit data")
        elif any(
            value is not None
            for value in (
                self.distance_km,
                self.duration_minutes,
                self.duration_basis,
                self.departure_at,
                self.arrival_at,
                self.departure_time_label,
                self.arrival_time_label,
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
            elif self.data_source == "serpapi_google_maps_directions":
                if (
                    self.requested_departure_at is None
                    or self.departure_time_basis is None
                    or self.coverage_status
                    not in {
                        "no_itinerary",
                        "not_configured",
                        "authentication_failed",
                        "quota_exhausted",
                        "rate_limited",
                        "provider_processing",
                        "provider_error",
                        "provider_unavailable",
                        "response_invalid",
                        "quota_ledger_unavailable",
                    }
                    or self.source_url != SERPAPI_TRANSIT_SOURCE_URL
                ):
                    raise ValueError("SerpApi transit unavailability requires query coverage")
            elif self.data_source == "openstreetmap_overpass_transit_reference":
                if (
                    self.requested_departure_at is None
                    or self.departure_time_basis is None
                    or self.coverage_status != "route_reference_only"
                    or self.source_url != OSM_TRANSIT_REFERENCE_SOURCE_URL
                    or not self.route_references
                ):
                    raise ValueError("OSM transit references require real route relations")
            elif (
                self.data_source != "open_transit_coverage_unavailable"
                or self.requested_departure_at is not None
                or self.departure_time_basis is not None
                or self.coverage_status is not None
                or self.source_url is not None
            ):
                raise ValueError("public-transit unavailability source is invalid")
            if (
                self.data_source != "openstreetmap_overpass_transit_reference"
                and self.route_references
            ):
                raise ValueError("only OSM reference responses can contain route relations")
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
    attraction_rating_source_capabilities: tuple[
        AttractionRatingSourceCapability,
        ...,
    ] = Field(
        default=ATTRACTION_RATING_SOURCE_REGISTRY,
        min_length=10,
        max_length=20,
    )


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

    def set(
        self,
        key: tuple[Any, ...],
        value: Any,
        *,
        ttl: timedelta | None = None,
    ) -> None:
        ttl_seconds = self._ttl_seconds if ttl is None else ttl.total_seconds()
        if ttl_seconds <= 0:
            raise ValueError("cache TTL must be positive")
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
                now + ttl_seconds,
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
        serpapi_transit_provider: SerpApiTransitDirectionsProvider | None = None,
        enable_transit_route_references: bool = False,
    ) -> None:
        if not 0 < timeout_seconds <= 15:
            raise ValueError("timeout_seconds must be greater than 0 and at most 15")
        self.client = client or BoundedJsonHttpClient()
        self.airport_resolver = airport_resolver or OurAirportsResolver(
            timeout_seconds=min(timeout_seconds, 10)
        )
        self.municipality_resolver = municipality_resolver or OurAirportsMunicipalityResolver(
            timeout_seconds=timeout_seconds,
            monotonic_clock=monotonic_clock,
        )
        self.timeout_seconds = timeout_seconds
        self._clock = monotonic_clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._serpapi_transit_provider = serpapi_transit_provider
        self._enable_transit_route_references = bool(enable_transit_route_references)
        self._nominatim_limiter = _OneRequestPerSecond(clock=monotonic_clock, sleeper=sleeper)
        self._overpass_limiter = _OneRequestPerSecond(clock=monotonic_clock, sleeper=sleeper)
        self._wikimedia_limiter = _OneRequestPerSecond(
            clock=monotonic_clock,
            sleeper=sleeper,
        )
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
        limit: int = DEFAULT_PLACES_PER_KIND,
    ) -> DestinationPlaceList:
        code = _normalize_iata(destination_airport)
        normalized_kind = _validate_kind(kind)
        normalized_category = _validate_category(normalized_kind, category)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_PLACES_PER_KIND
        ):
            raise DestinationValidationError(
                f"limit must be an integer from 1 through {MAX_PLACES_PER_KIND}"
            )
        city = self.resolve_city(code)
        all_places, fetched_at, expires_at, coverage = self._load_places(
            code,
            normalized_kind,
            city,
        )
        if normalized_category == "all":
            selected = _balanced_places(all_places, normalized_kind, limit)
        else:
            filtered = tuple(place for place in all_places if place.category == normalized_category)
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
            query_parts_succeeded=coverage.query_parts_succeeded,
            query_parts_total=coverage.query_parts_total,
            provider_truncated=coverage.provider_truncated,
        )

    def get_place_detail(
        self,
        destination_airport: str,
        place_id: str,
        transit_departure_at: datetime | None = None,
        *,
        include_live_transit: bool = False,
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
                include_live_transit=include_live_transit,
            ),
        )

    def get_routes(
        self,
        destination_airport: str,
        latitude: float,
        longitude: float,
        *,
        transit_departure_at: datetime | None = None,
        include_live_transit: bool = False,
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
            include_live_transit=include_live_transit,
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

    def _load_attraction_places(
        self,
        code: str,
        city: DestinationCity,
    ) -> tuple[
        tuple[DestinationPlace, ...],
        datetime,
        datetime,
        DestinationCoverage,
    ]:
        """Query a bounded city-centred area in independent parts.

        A single 60 km-wide Overpass request is fragile in large cities.  Four
        independently retryable quadrants preserve real results from successful
        parts and make incomplete coverage explicit instead of collapsing the
        complete attraction page.
        """

        key = (
            code,
            "attraction-city-tiles-v2",
            round(city.latitude, 6),
            round(city.longitude, 6),
            city.scope,
        )
        cached = self._place_cache.get(key)
        if isinstance(cached, tuple) and len(cached) == 4:
            return cached

        query_bounds = _attraction_query_bounds(city.latitude, city.longitude)
        operation_started = self._clock()
        preferred_endpoint = OVERPASS_URL
        successful_parts = 0
        provider_truncated = False
        by_id: dict[str, DestinationPlace] = {}

        for bounds in query_bounds:
            remaining_budget = ATTRACTION_OVERPASS_OPERATION_BUDGET_SECONDS - (
                self._clock() - operation_started
            )
            if remaining_budget <= 0:
                break
            query = _attraction_overpass_query(bounds)
            tile_places: tuple[DestinationPlace, ...] | None = None
            tile_payload: Any = None
            endpoints = (
                preferred_endpoint,
                next(endpoint for endpoint in OVERPASS_ENDPOINTS if endpoint != preferred_endpoint),
            )
            for endpoint in endpoints:
                try:
                    self._overpass_limiter.wait()
                    remaining_budget = ATTRACTION_OVERPASS_OPERATION_BUDGET_SECONDS - (
                        self._clock() - operation_started
                    )
                    if remaining_budget <= 0:
                        break
                    tile_payload = self.client.request_json(
                        "POST",
                        endpoint,
                        data={"data": query},
                        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                        timeout_seconds=min(
                            self.timeout_seconds,
                            ATTRACTION_OVERPASS_REQUEST_TIMEOUT_SECONDS,
                            remaining_budget,
                        ),
                        max_response_bytes=MAX_OVERPASS_BYTES,
                    )
                    tile_places = _parse_overpass_places(
                        tile_payload,
                        "attraction",
                        city.latitude,
                        city.longitude,
                        radius_meters=DESTINATION_RADIUS_METERS,
                    )
                except DestinationGuideError:
                    continue
                preferred_endpoint = endpoint
                break
            if tile_places is None:
                continue
            successful_parts += 1
            provider_truncated = provider_truncated or _overpass_payload_hit_limit(tile_payload)
            for place in tile_places:
                existing = by_id.get(place.place_id)
                if existing is None or _place_completeness(place) > _place_completeness(existing):
                    by_id[place.place_id] = place

        if successful_parts == 0:
            raise DestinationDataUnavailable(
                "OpenStreetMap attraction coverage is temporarily unavailable"
            )

        ordered = tuple(
            sorted(
                by_id.values(),
                key=lambda item: (
                    item.distance_from_city_center_km,
                    item.name.casefold(),
                    item.place_id,
                ),
            )
        )
        internal_truncated = len(ordered) > MAX_INTERNAL_PLACES_PER_KIND
        places = ordered[:MAX_INTERNAL_PLACES_PER_KIND]
        places = self._enrich_attractions(places, city)
        fetched_at = _aware_utc(self._wall_clock())
        coverage = _attraction_coverage(
            successful_parts=successful_parts,
            total_parts=len(query_bounds),
            provider_truncated=provider_truncated or internal_truncated,
            result_count=len(places),
        )
        cache_ttl = PARTIAL_DESTINATION_CACHE_TTL if coverage.partial else DESTINATION_CACHE_TTL
        value = (places, fetched_at, fetched_at + cache_ttl, coverage)
        self._place_cache.set(key, value, ttl=cache_ttl)
        return value

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
        if kind == "attraction":
            return self._load_attraction_places(code, city)
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
            endpoints = (OVERPASS_URL,) if fallback_used else (OVERPASS_URL, OVERPASS_FALLBACK_URL)
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
                if existing is None or _place_completeness(place) > _place_completeness(existing):
                    by_id[place.place_id] = place
            if len(by_id) >= HOTEL_RESULT_TARGET:
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
        cache_ttl = PARTIAL_DESTINATION_CACHE_TTL if coverage.partial else DESTINATION_CACHE_TTL
        value = (places, fetched_at, fetched_at + cache_ttl, coverage)
        self._place_cache.set(key, value, ttl=cache_ttl)
        return value

    def _enrich_attractions(
        self,
        places: tuple[DestinationPlace, ...],
        city: DestinationCity,
    ) -> tuple[DestinationPlace, ...]:
        """Add only exact-identity Wikimedia evidence, then factual OSM summaries."""

        wikidata_ids = tuple(
            dict.fromkeys(place.wikidata_id for place in places if place.wikidata_id is not None)
        )
        enrichment_started = self._clock()

        def remaining_budget() -> float:
            return WIKIMEDIA_OPERATION_BUDGET_SECONDS - (self._clock() - enrichment_started)

        entities: dict[str, Mapping[str, Any]] = {}
        failed_wikidata_ids: set[str] = set()
        for batch in _chunks(wikidata_ids, MAX_WIKIMEDIA_BATCH_SIZE):
            remaining = remaining_budget()
            if remaining <= 0.25:
                break
            try:
                self._wikimedia_limiter.wait()
                remaining = remaining_budget()
                if remaining <= 0.25:
                    break
                payload = self.client.request_json(
                    "GET",
                    WIKIDATA_API_URL,
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(batch),
                        "props": "labels|descriptions|sitelinks|claims",
                        "languages": "zh|en",
                        "languagefallback": "1",
                        "sitefilter": "zhwiki|enwiki",
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                    timeout_seconds=min(
                        self.timeout_seconds,
                        WIKIMEDIA_REQUEST_TIMEOUT_SECONDS,
                        remaining,
                    ),
                    max_response_bytes=MAX_WIKIMEDIA_BYTES,
                )
                entities.update(_parse_wikidata_entities(payload, allowed_ids=set(batch)))
            except DestinationGuideError:
                failed_wikidata_ids.update(batch)
        failed_wikidata_ids.update(set(wikidata_ids) - entities.keys())

        issuer_ids = tuple(sorted(_wikidata_rating_issuer_ids(entities.values())))[
            :MAX_WIKIDATA_RATING_ISSUERS
        ]
        issuer_labels: dict[str, Mapping[str, str]] = {}
        for batch in _chunks(issuer_ids, MAX_WIKIMEDIA_BATCH_SIZE):
            remaining = remaining_budget()
            if remaining <= 0.25:
                break
            try:
                self._wikimedia_limiter.wait()
                remaining = remaining_budget()
                if remaining <= 0.25:
                    break
                payload = self.client.request_json(
                    "GET",
                    WIKIDATA_API_URL,
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(batch),
                        "props": "labels",
                        "languages": "zh|en",
                        "languagefallback": "1",
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                    timeout_seconds=min(
                        self.timeout_seconds,
                        WIKIMEDIA_REQUEST_TIMEOUT_SECONDS,
                        remaining,
                    ),
                    max_response_bytes=MAX_WIKIMEDIA_BYTES,
                )
                parsed = _parse_wikidata_entities(payload, allowed_ids=set(batch))
                issuer_labels.update(
                    {
                        entity_id: _wikidata_language_values(
                            entity.get("labels"),
                            maximum=200,
                        )
                        for entity_id, entity in parsed.items()
                    }
                )
            except DestinationGuideError:
                continue

        wikipedia_titles = _wikipedia_titles_by_language(places, entities)
        wikipedia_extracts: dict[
            tuple[str, str],
            tuple[str, str],
        ] = {}
        wikipedia_pageimages: dict[tuple[str, str], str] = {}
        failed_wikipedia_photo_keys: set[tuple[str, str]] = set()
        completed_wikipedia_photo_keys: set[tuple[str, str]] = set()
        for language, titles in wikipedia_titles.items():
            endpoint = WIKIPEDIA_API_URLS[language]
            for batch in _chunks(tuple(sorted(titles)), 20):
                requested_photo_keys = {(language, title.casefold()) for title in batch}
                remaining = remaining_budget()
                if remaining <= 0.25:
                    failed_wikipedia_photo_keys.update(requested_photo_keys)
                    break
                try:
                    self._wikimedia_limiter.wait()
                    remaining = remaining_budget()
                    if remaining <= 0.25:
                        failed_wikipedia_photo_keys.update(requested_photo_keys)
                        break
                    payload = self.client.request_json(
                        "GET",
                        endpoint,
                        params={
                            "action": "query",
                            "prop": "extracts|info|pageimages",
                            "inprop": "url",
                            "exintro": "1",
                            "explaintext": "1",
                            "piprop": "name",
                            "pilicense": "free",
                            "redirects": "1",
                            "titles": "|".join(batch),
                            "format": "json",
                            "formatversion": "2",
                        },
                        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                        timeout_seconds=min(
                            self.timeout_seconds,
                            WIKIMEDIA_REQUEST_TIMEOUT_SECONDS,
                            remaining,
                        ),
                        max_response_bytes=MAX_WIKIMEDIA_BYTES,
                    )
                    wikipedia_extracts.update(
                        _parse_wikipedia_extracts(
                            payload,
                            language=language,
                            requested_titles=set(batch),
                        )
                    )
                    wikipedia_pageimages.update(
                        _parse_wikipedia_pageimages(
                            payload,
                            language=language,
                            requested_titles=set(batch),
                        )
                    )
                    completed_wikipedia_photo_keys.update(requested_photo_keys)
                except DestinationGuideError:
                    failed_wikipedia_photo_keys.update(requested_photo_keys)
                    continue
        all_wikipedia_photo_keys = {
            (language, title.casefold())
            for language, titles in wikipedia_titles.items()
            for title in titles
        }
        failed_wikipedia_photo_keys.update(
            all_wikipedia_photo_keys - completed_wikipedia_photo_keys
        )

        photo_candidates_by_place = {
            place.place_id: _attraction_photo_candidates(
                place,
                entities.get(place.wikidata_id or ""),
                wikipedia_pageimages,
            )
            for place in places
        }
        commons_titles = tuple(
            dict.fromkeys(
                file_title
                for candidates in photo_candidates_by_place.values()
                for file_title, _match_basis in candidates
            )
        )
        commons_assets: dict[str, Mapping[str, Any]] = {}
        failed_commons_titles: set[str] = set()
        completed_commons_titles: set[str] = set()
        for batch in _chunks(commons_titles, MAX_WIKIMEDIA_BATCH_SIZE):
            remaining = remaining_budget()
            if remaining <= 0.25:
                failed_commons_titles.update(title.casefold() for title in batch)
                break
            try:
                self._wikimedia_limiter.wait()
                remaining = remaining_budget()
                if remaining <= 0.25:
                    failed_commons_titles.update(title.casefold() for title in batch)
                    break
                payload = self.client.request_json(
                    "GET",
                    WIKIMEDIA_COMMONS_API_URL,
                    params={
                        "action": "query",
                        "prop": "imageinfo",
                        "iiprop": "url|size|extmetadata",
                        "iiurlwidth": str(COMMONS_THUMBNAIL_WIDTH),
                        "titles": "|".join(batch),
                        "redirects": "1",
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                    timeout_seconds=min(
                        self.timeout_seconds,
                        WIKIMEDIA_REQUEST_TIMEOUT_SECONDS,
                        remaining,
                    ),
                    max_response_bytes=MAX_WIKIMEDIA_BYTES,
                )
                commons_assets.update(
                    _parse_commons_photo_assets(
                        payload,
                        requested_titles=set(batch),
                    )
                )
                completed_commons_titles.update(title.casefold() for title in batch)
            except DestinationGuideError:
                failed_commons_titles.update(title.casefold() for title in batch)
                continue
        failed_commons_titles.update(
            {title.casefold() for title in commons_titles} - completed_commons_titles
        )

        enriched: list[DestinationPlace] = []
        for place in places:
            entity = entities.get(place.wikidata_id) if place.wikidata_id is not None else None
            updates = _attraction_enrichment_updates(
                place,
                city,
                entity=entity,
                issuer_labels=issuer_labels,
                wikipedia_extracts=wikipedia_extracts,
                wikidata_failed=place.wikidata_id in failed_wikidata_ids,
                photo_candidates=photo_candidates_by_place.get(place.place_id, ()),
                commons_assets=commons_assets,
                failed_commons_titles=failed_commons_titles,
                wikipedia_photo_failed=any(
                    key in failed_wikipedia_photo_keys
                    for key in (
                        (language, title.casefold())
                        for language, title in _wikipedia_titles_for_place(
                            place,
                            entity,
                        ).items()
                    )
                ),
            )
            enriched.append(
                DestinationPlace.model_validate(
                    {**place.model_dump(mode="python"), **updates},
                )
            )
        return tuple(enriched)

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
        server_profile = {"car": "routed-car", "bike": "routed-bike", "foot": "routed-foot"}[mode]
        coordinates = (
            f"{airport.longitude:.6f},{airport.latitude:.6f};{longitude:.6f},{latitude:.6f}"
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
        include_live_transit: bool = False,
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
            bool(include_live_transit),
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
            "maxPreTransitTime": "3600",
            "maxPostTransitTime": "3600",
            "maxMatchingDistance": "100",
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
        if (
            result.status == "unavailable"
            and result.coverage_status in {"no_itinerary", "provider_unavailable"}
            and include_live_transit
            and self._serpapi_transit_provider is not None
            and self._serpapi_transit_provider.configured
        ):
            fallback = self._serpapi_transit_provider.search(
                origin_latitude=airport.latitude,
                origin_longitude=airport.longitude,
                destination_latitude=latitude,
                destination_longitude=longitude,
                departure_at=requested_departure_at,
                country_code=airport.country,
            )
            result = _serpapi_transit_option(
                fallback,
                requested_departure_at=requested_departure_at,
                departure_time_basis=departure_time_basis,
                expires_at=expires_at,
            )
        upstream_coverage_status = result.coverage_status
        if result.status == "unavailable" and self._enable_transit_route_references:
            reference_result = self._transit_route_reference_option(
                airport,
                latitude,
                longitude,
                requested_departure_at=requested_departure_at,
                departure_time_basis=departure_time_basis,
                observed_at=observed_at,
                expires_at=expires_at,
            )
            if reference_result is not None:
                result = reference_result
        reference_masks_transient_failure = (
            result.data_source == "openstreetmap_overpass_transit_reference"
            and upstream_coverage_status != "no_itinerary"
        )
        if not reference_masks_transient_failure and result.coverage_status not in {
            "provider_unavailable",
            "rate_limited",
            "provider_processing",
            "provider_error",
            "response_invalid",
            "quota_ledger_unavailable",
        }:
            self._transit_cache.set(key, result)
        return result

    def _transit_route_reference_option(
        self,
        airport: Airport,
        latitude: float,
        longitude: float,
        *,
        requested_departure_at: datetime,
        departure_time_basis: Literal["user_supplied", "request_time"],
        observed_at: datetime,
        expires_at: datetime,
    ) -> DestinationTransportOption | None:
        airport_routes = self._nearby_osm_transit_routes(
            airport.latitude,
            airport.longitude,
        )
        destination_routes = self._nearby_osm_transit_routes(latitude, longitude)
        references = _merge_osm_transit_references(
            airport_routes,
            destination_routes,
        )
        if not references:
            return None
        return DestinationTransportOption(
            mode="public_transit",
            status="unavailable",
            requested_departure_at=requested_departure_at,
            departure_time_basis=departure_time_basis,
            coverage_status="route_reference_only",
            route_references=references,
            notice=(
                "OpenStreetMap contains real public-transport route relations near one or "
                "both endpoints. They are route references only: no precise departure, "
                "arrival, transfer itinerary, or travel time is asserted."
            ),
            data_source="openstreetmap_overpass_transit_reference",
            source_url=OSM_TRANSIT_REFERENCE_SOURCE_URL,
            observed_at=observed_at,
            expires_at=expires_at,
        )

    def _nearby_osm_transit_routes(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[int, dict[str, Any]]:
        query = _osm_transit_reference_query(latitude, longitude)
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                self._overpass_limiter.wait()
                payload = self.client.request_json(
                    "POST",
                    endpoint,
                    data={"data": query},
                    headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                    timeout_seconds=min(
                        self.timeout_seconds,
                        OVERPASS_REQUEST_TIMEOUT_SECONDS,
                    ),
                    max_response_bytes=MAX_OVERPASS_BYTES,
                )
                return _parse_osm_transit_reference_payload(payload)
            except DestinationGuideError:
                continue
        return {}


def create_destination_guide_service(
    *,
    client: JsonHttpTransport | None = None,
    airport_resolver: AirportResolver | None = None,
    municipality_resolver: MunicipalityResolver | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    serpapi_transit_provider: SerpApiTransitDirectionsProvider | None = None,
    enable_transit_route_references: bool = True,
) -> DestinationGuideService:
    """Create the production service; dependencies remain injectable for tests."""

    return DestinationGuideService(
        client=client,
        airport_resolver=airport_resolver,
        municipality_resolver=municipality_resolver,
        timeout_seconds=timeout_seconds,
        serpapi_transit_provider=serpapi_transit_provider,
        enable_transit_route_references=enable_transit_route_references,
    )


def build_destination_guide_service(
    *,
    client: JsonHttpTransport | None = None,
    airport_resolver: AirportResolver | None = None,
    municipality_resolver: MunicipalityResolver | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    serpapi_transit_provider: SerpApiTransitDirectionsProvider | None = None,
    enable_transit_route_references: bool = True,
) -> DestinationGuideService:
    """Alias with the project's existing ``build_*`` factory naming convention."""

    return create_destination_guide_service(
        client=client,
        airport_resolver=airport_resolver,
        municipality_resolver=municipality_resolver,
        timeout_seconds=timeout_seconds,
        serpapi_transit_provider=serpapi_transit_provider,
        enable_transit_route_references=enable_transit_route_references,
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
                    f"具名地点数据和内部最多 {MAX_INTERNAL_PLACES_PER_KIND} 条记录限制。"
                ),
                en=(
                    "The full 30 km radius around the destination centre was queried "
                    "successfully; results remain limited to named OpenStreetMap features and "
                    f"at most {MAX_INTERNAL_PLACES_PER_KIND} internal records."
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


def _attraction_coverage(
    *,
    successful_parts: int,
    total_parts: int,
    provider_truncated: bool,
    result_count: int,
) -> DestinationCoverage:
    if successful_parts == total_parts and not provider_truncated:
        return DestinationCoverage(
            coverage_radius_km=30,
            coverage_status="complete",
            coverage_reason="full_radius_queried",
            partial=False,
            coverage_notice=DestinationCoverageNotice(
                zh=(
                    f"已完成目的地城市中心周边 30 公里的 {total_parts} 个分块查询，"
                    f"返回 {result_count} 个符合当前 OpenStreetMap 景点标签的具名地点。"
                    "“全部可用”仅指这些已成功查询的公开来源记录，不代表任何商业平台的全球全量。"
                ),
                en=(
                    f"The full 30 km city-centred area completed all {total_parts} bounded "
                    f"query parts, returning {result_count} named places matching "
                    "the current OpenStreetMap attraction tags. “All available” means these "
                    "successfully queried public-source records, not a global commercial inventory."
                ),
            ),
            query_parts_succeeded=successful_parts,
            query_parts_total=total_parts,
            provider_truncated=False,
        )
    if provider_truncated:
        reason: Literal["result_limit_reached", "provider_failure"] = "result_limit_reached"
        zh = (
            f"已完成 {successful_parts}/{total_parts} 个城市分块，但至少一个公开数据响应或"
            f"本地安全上限达到 {MAX_INTERNAL_PLACES_PER_KIND} 条；当前显示所有已保留的"
            "真实记录，不声称完整城市全量。"
        )
        en = (
            f"{successful_parts}/{total_parts} city query parts completed, but at least one "
            f"public response or the local safety cap reached {MAX_INTERNAL_PLACES_PER_KIND} "
            "records. Every retained real record is shown, without claiming complete city "
            "inventory."
        )
    else:
        reason = "provider_failure"
        zh = (
            f"已成功完成 {successful_parts}/{total_parts} 个城市分块；失败分块不会清空其他"
            f"真实结果。当前共返回 {result_count} 个景点，并明确存在空间覆盖缺口。"
        )
        en = (
            f"{successful_parts}/{total_parts} city query parts completed. Failed parts did "
            f"not erase successful real results; {result_count} attractions are returned with "
            "explicit spatial coverage gaps."
        )
    return DestinationCoverage(
        coverage_radius_km=30,
        coverage_status="partial",
        coverage_reason=reason,
        partial=True,
        coverage_notice=DestinationCoverageNotice(zh=zh, en=en),
        query_parts_succeeded=successful_parts,
        query_parts_total=total_parts,
        provider_truncated=provider_truncated,
    )


def _overpass_payload_hit_limit(payload: Any) -> bool:
    return bool(
        isinstance(payload, dict)
        and isinstance(payload.get("elements"), list)
        and len(payload["elements"]) >= MAX_OVERPASS_ELEMENTS
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


def _chunks(values: tuple[str, ...], size: int) -> tuple[tuple[str, ...], ...]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _wikidata_id(value: object) -> str | None:
    cleaned = _clean_text(value, 24)
    if cleaned is None or re.fullmatch(r"Q[1-9][0-9]{0,18}", cleaned) is None:
        return None
    return cleaned


def _wikipedia_url(language: str, title: str) -> str | None:
    if language not in WIKIPEDIA_API_URLS:
        return None
    cleaned = _clean_text(title, 500)
    if cleaned is None:
        return None
    encoded = parse.quote(cleaned.replace(" ", "_"), safe="()_,-")
    return f"https://{language}.wikipedia.org/wiki/{encoded}"


def _wikipedia_tag_url(value: object) -> str | None:
    cleaned = _clean_text(value, 600)
    if cleaned is None:
        return None
    match = re.fullmatch(r"(zh|en):(.+)", cleaned)
    if match is None:
        return None
    return _wikipedia_url(match.group(1), match.group(2))


def _wikipedia_title_from_url(value: str | None) -> tuple[str, str] | None:
    if value is None:
        return None
    try:
        parsed = parse.urlsplit(value)
    except ValueError:
        return None
    match = re.fullmatch(r"(zh|en)\.wikipedia\.org", (parsed.hostname or "").lower())
    if (
        match is None
        or parsed.scheme != "https"
        or parsed.port not in {None, 443}
        or not parsed.path.startswith("/wiki/")
    ):
        return None
    title = _clean_text(parse.unquote(parsed.path[6:]).replace("_", " "), 500)
    if title is None:
        return None
    return match.group(1), title


def _parse_wikidata_entities(
    payload: Any,
    *,
    allowed_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), dict):
        raise DestinationDataUnavailable("Wikidata response is missing entities")
    parsed: dict[str, Mapping[str, Any]] = {}
    for entity_id, entity in payload["entities"].items():
        if entity_id in allowed_ids and isinstance(entity, dict) and entity.get("missing") is None:
            parsed[entity_id] = entity
    return parsed


def _wikidata_language_values(
    value: object,
    *,
    maximum: int,
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for language in ("zh", "en"):
        raw = value.get(language)
        text = _clean_text(raw.get("value"), maximum) if isinstance(raw, dict) else None
        if text:
            result[language] = text
    return result


def _wikidata_rating_issuer_ids(
    entities: Any,
) -> set[str]:
    issuer_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        claims = entity.get("claims")
        ratings = claims.get("P444") if isinstance(claims, dict) else None
        if not isinstance(ratings, list):
            continue
        for statement in ratings:
            if not isinstance(statement, dict):
                continue
            qualifiers = statement.get("qualifiers")
            issuers = qualifiers.get("P447") if isinstance(qualifiers, dict) else None
            if not isinstance(issuers, list):
                continue
            for snak in issuers:
                issuer_id = _wikidata_item_snak(snak)
                if issuer_id:
                    issuer_ids.add(issuer_id)
    return issuer_ids


def _wikidata_sitelink_titles(entity: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(entity, Mapping):
        return {}
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, dict):
        return {}
    result: dict[str, str] = {}
    for language in ("zh", "en"):
        raw = sitelinks.get(f"{language}wiki")
        title = _clean_text(raw.get("title"), 500) if isinstance(raw, dict) else None
        if title:
            result[language] = title
    return result


def _wikipedia_titles_for_place(
    place: DestinationPlace,
    entity: Mapping[str, Any] | None,
) -> dict[str, str]:
    titles = _wikidata_sitelink_titles(entity)
    tagged = _wikipedia_title_from_url(place.wikipedia_url)
    if tagged is not None:
        titles.setdefault(*tagged)
    return titles


def _wikipedia_titles_by_language(
    places: tuple[DestinationPlace, ...],
    entities: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    result = {"zh": set(), "en": set()}
    for place in places:
        entity = entities.get(place.wikidata_id or "")
        for language, title in _wikipedia_titles_for_place(place, entity).items():
            result[language].add(title)
    return {
        language: set(sorted(titles)[:MAX_WIKIPEDIA_TITLES_PER_LANGUAGE])
        for language, titles in result.items()
    }


def _parse_wikipedia_extracts(
    payload: Any,
    *,
    language: str,
    requested_titles: set[str],
) -> dict[tuple[str, str], tuple[str, str]]:
    if not isinstance(payload, dict):
        raise DestinationDataUnavailable("Wikipedia response is not an object")
    query = payload.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, (list, dict)):
        raise DestinationDataUnavailable("Wikipedia response is missing pages")
    raw_pages = pages if isinstance(pages, list) else list(pages.values())
    page_by_title: dict[str, tuple[str, str]] = {}
    for page in raw_pages:
        if not isinstance(page, dict) or page.get("missing") is not None:
            continue
        title = _clean_text(page.get("title"), 500)
        extract = _clean_text(page.get("extract"), 2_000)
        url = _safe_public_https_url(
            page.get("canonicalurl") or page.get("fullurl") or _wikipedia_url(language, title or "")
        )
        if title and extract and url:
            page_by_title[title.casefold()] = (extract, url)

    aliases: dict[str, str] = {}
    for field in ("normalized", "redirects"):
        values = query.get(field) if isinstance(query, dict) else None
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            before = _clean_text(item.get("from"), 500)
            after = _clean_text(item.get("to"), 500)
            if before and after:
                aliases[before.casefold()] = after.casefold()

    result: dict[tuple[str, str], tuple[str, str]] = {}
    for requested in requested_titles:
        key = requested.casefold()
        resolved = key
        for _ in range(3):
            next_value = aliases.get(resolved)
            if next_value is None or next_value == resolved:
                break
            resolved = next_value
        item = page_by_title.get(resolved) or page_by_title.get(key)
        if item:
            result[(language, key)] = item
    return result


def _parse_wikipedia_pageimages(
    payload: Any,
    *,
    language: str,
    requested_titles: set[str],
) -> dict[tuple[str, str], str]:
    """Return only free page-image filenames attached to requested exact pages."""

    if not isinstance(payload, dict):
        raise DestinationDataUnavailable("Wikipedia response is not an object")
    query = payload.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, (list, dict)):
        raise DestinationDataUnavailable("Wikipedia response is missing pages")
    raw_pages = pages if isinstance(pages, list) else list(pages.values())
    page_by_title: dict[str, str] = {}
    for page in raw_pages:
        if not isinstance(page, dict) or page.get("missing") is not None:
            continue
        title = _clean_text(page.get("title"), 500)
        file_title = _commons_file_title(page.get("pageimage"))
        if title and file_title:
            page_by_title[title.casefold()] = file_title

    aliases: dict[str, str] = {}
    for field in ("normalized", "redirects"):
        values = query.get(field) if isinstance(query, dict) else None
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            before = _clean_text(item.get("from"), 500)
            after = _clean_text(item.get("to"), 500)
            if before and after:
                aliases[before.casefold()] = after.casefold()

    result: dict[tuple[str, str], str] = {}
    for requested in requested_titles:
        key = requested.casefold()
        resolved = key
        for _ in range(3):
            next_value = aliases.get(resolved)
            if next_value is None or next_value == resolved:
                break
            resolved = next_value
        item = page_by_title.get(resolved) or page_by_title.get(key)
        if item:
            result[(language, key)] = item
    return result


def _commons_file_title(value: object) -> str | None:
    cleaned = _clean_text(value, 500)
    if cleaned is None:
        return None
    cleaned = cleaned.replace("_", " ")
    if cleaned.casefold().startswith("category:"):
        return None
    if cleaned.casefold().startswith("file:"):
        filename = cleaned[5:].strip()
    else:
        filename = cleaned
    if (
        not filename
        or any(character in filename for character in "\r\n|[]{}<>")
        or "://" in filename
    ):
        return None
    return f"File:{filename}"[:500]


def _osm_commons_file(value: object) -> str | None:
    cleaned = _clean_text(value, 500)
    if cleaned is None or not cleaned.casefold().startswith("file:"):
        return None
    return _commons_file_title(cleaned)


def _wikidata_image_filename(entity: Mapping[str, Any] | None) -> str | None:
    if not isinstance(entity, Mapping):
        return None
    claims = entity.get("claims")
    statements = claims.get("P18") if isinstance(claims, dict) else None
    if not isinstance(statements, list):
        return None
    for statement in statements:
        if not isinstance(statement, dict) or statement.get("rank") == "deprecated":
            continue
        file_title = _commons_file_title(_wikidata_text_snak(statement.get("mainsnak"), 500))
        if file_title:
            return file_title
    return None


def _attraction_photo_candidates(
    place: DestinationPlace,
    entity: Mapping[str, Any] | None,
    wikipedia_pageimages: Mapping[tuple[str, str], str],
) -> tuple[
    tuple[
        str,
        Literal[
            "osm_wikimedia_commons",
            "wikidata_p18",
            "wikipedia_pageimage",
        ],
    ],
    ...,
]:
    candidates: list[
        tuple[
            str,
            Literal[
                "osm_wikimedia_commons",
                "wikidata_p18",
                "wikipedia_pageimage",
            ],
        ]
    ] = []
    seen: set[str] = set()

    def add(
        file_title: str | None,
        basis: Literal[
            "osm_wikimedia_commons",
            "wikidata_p18",
            "wikipedia_pageimage",
        ],
    ) -> None:
        normalized = _commons_file_title(file_title)
        if normalized is None or normalized.casefold() in seen:
            return
        seen.add(normalized.casefold())
        candidates.append((normalized, basis))

    add(place.osm_wikimedia_commons_file, "osm_wikimedia_commons")
    add(_wikidata_image_filename(entity), "wikidata_p18")
    titles = _wikipedia_titles_for_place(place, entity)
    for language in ("zh", "en"):
        title = titles.get(language)
        if title:
            add(
                wikipedia_pageimages.get((language, title.casefold())),
                "wikipedia_pageimage",
            )
    return tuple(candidates[:MAX_ATTRACTION_PHOTOS])


def _clean_commons_metadata(value: object, maximum: int) -> str | None:
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return None
    without_tags = re.sub(r"<[^>]{0,1000}>", " ", value)
    return _clean_text(html.unescape(without_tags), maximum)


def _parse_commons_photo_assets(
    payload: Any,
    *,
    requested_titles: set[str],
) -> dict[str, Mapping[str, Any]]:
    """Resolve requested Commons files to attribution-complete image evidence."""

    if not isinstance(payload, dict):
        raise DestinationDataUnavailable("Commons response is not an object")
    query = payload.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, (list, dict)):
        raise DestinationDataUnavailable("Commons response is missing pages")
    requested = {
        title.casefold()
        for title in (_commons_file_title(value) for value in requested_titles)
        if title is not None
    }
    raw_pages = pages if isinstance(pages, list) else list(pages.values())
    assets_by_title: dict[str, Mapping[str, Any]] = {}
    for page in raw_pages:
        if not isinstance(page, dict) or page.get("missing") is not None:
            continue
        file_title = _commons_file_title(page.get("title"))
        if file_title is None:
            continue
        imageinfo = page.get("imageinfo")
        if not isinstance(imageinfo, list) or not imageinfo:
            continue
        info = imageinfo[0]
        if not isinstance(info, dict):
            continue
        image_url = _safe_public_https_url(info.get("url"))
        thumbnail_url = _safe_public_https_url(info.get("thumburl") or info.get("url"))
        source_page_url = _safe_public_https_url(info.get("descriptionurl"))
        width = _bounded_int(info.get("width"), 1, 100_000)
        height = _bounded_int(info.get("height"), 1, 100_000)
        metadata = info.get("extmetadata")
        if not isinstance(metadata, dict):
            continue
        author = _clean_commons_metadata(
            metadata.get("Artist") or metadata.get("Credit"),
            500,
        )
        credit = _clean_commons_metadata(metadata.get("Credit"), 500)
        license_name = _clean_commons_metadata(
            metadata.get("LicenseShortName"),
            200,
        )
        license_url = _safe_public_https_url(
            _clean_commons_metadata(metadata.get("LicenseUrl"), 2_048)
        )
        if (
            image_url is None
            or thumbnail_url is None
            or source_page_url is None
            or width is None
            or height is None
            or author is None
            or license_name is None
            or license_url is None
            or not _is_wikimedia_upload_url(image_url)
            or not _is_wikimedia_upload_url(thumbnail_url)
            or not _is_wikimedia_commons_page(source_page_url)
        ):
            continue
        credit_text = credit if credit and credit.casefold() != author.casefold() else author
        attribution = _clean_text(
            f"{credit_text} — {author} — {license_name}"
            if credit_text != author
            else f"{author} — {license_name}",
            1_000,
        )
        if attribution is None:
            continue
        assets_by_title[file_title.casefold()] = {
            "file_title": file_title,
            "image_url": image_url,
            "thumbnail_url": thumbnail_url,
            "source_page_url": source_page_url,
            "author": author,
            "license_name": license_name,
            "license_url": license_url,
            "attribution": attribution,
            "width": width,
            "height": height,
        }
    aliases: dict[str, str] = {}
    for field in ("normalized", "redirects"):
        values = query.get(field) if isinstance(query, dict) else None
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            before = _commons_file_title(item.get("from"))
            after = _commons_file_title(item.get("to"))
            if before and after:
                aliases[before.casefold()] = after.casefold()

    result: dict[str, Mapping[str, Any]] = {}
    for requested_title in requested:
        resolved = requested_title
        for _ in range(3):
            next_value = aliases.get(resolved)
            if next_value is None or next_value == resolved:
                break
            resolved = next_value
        asset = assets_by_title.get(resolved) or assets_by_title.get(requested_title)
        if asset:
            result[requested_title] = asset
    return result


def _wikidata_item_snak(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    datavalue = value.get("datavalue")
    raw = datavalue.get("value") if isinstance(datavalue, dict) else None
    return _wikidata_id(raw.get("id")) if isinstance(raw, dict) else None


def _wikidata_text_snak(value: object, maximum: int) -> str | None:
    if not isinstance(value, dict):
        return None
    datavalue = value.get("datavalue")
    raw = datavalue.get("value") if isinstance(datavalue, dict) else None
    return _clean_text(raw, maximum)


def _first_qualifier(
    qualifiers: Mapping[str, Any],
    property_id: str,
) -> object | None:
    values = qualifiers.get(property_id)
    if not isinstance(values, list) or not values:
        return None
    return values[0]


def _wikidata_quantity_qualifier(
    qualifiers: Mapping[str, Any],
    property_id: str,
) -> int | None:
    raw = _first_qualifier(qualifiers, property_id)
    if not isinstance(raw, dict):
        return None
    datavalue = raw.get("datavalue")
    value = datavalue.get("value") if isinstance(datavalue, dict) else None
    amount = value.get("amount") if isinstance(value, dict) else None
    try:
        number = int(float(str(amount)))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if 0 <= number <= 2_000_000_000 else None


def _wikidata_time_qualifier(
    qualifiers: Mapping[str, Any],
    property_id: str,
) -> str | None:
    raw = _first_qualifier(qualifiers, property_id)
    if not isinstance(raw, dict):
        return None
    datavalue = raw.get("datavalue")
    value = datavalue.get("value") if isinstance(datavalue, dict) else None
    raw_time = value.get("time") if isinstance(value, dict) else None
    cleaned = _clean_text(raw_time, 40)
    if cleaned is None:
        return None
    match = re.match(r"^[+-]?([0-9]{4,})-([0-9]{2})-([0-9]{2})", cleaned)
    if match is None:
        return None
    year, month, day = match.groups()
    if month == "00":
        return year
    if day == "00":
        return f"{year}-{month}"
    return f"{year}-{month}-{day}"


def _numeric_review_score(score_text: str) -> tuple[float | None, float | None]:
    ratio = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|out\s+of)\s*([0-9]+(?:\.[0-9]+)?)\s*",
        score_text,
        flags=re.IGNORECASE,
    )
    if ratio is not None:
        score, maximum = (float(ratio.group(1)), float(ratio.group(2)))
        if 0 <= score <= maximum <= 100_000:
            return score, maximum
    percent = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*", score_text)
    if percent is not None:
        score = float(percent.group(1))
        if 0 <= score <= 100:
            return score, 100.0
    return None, None


def _parse_wikidata_ratings(
    entity: Mapping[str, Any] | None,
    *,
    entity_id: str | None,
    issuer_labels: Mapping[str, Mapping[str, str]],
) -> tuple[DestinationAttractionRating, ...]:
    if not isinstance(entity, Mapping) or entity_id is None:
        return ()
    claims = entity.get("claims")
    statements = claims.get("P444") if isinstance(claims, dict) else None
    if not isinstance(statements, list):
        return ()
    latest_by_issuer: dict[
        str,
        tuple[tuple[int, int, int, int, int, int], DestinationAttractionRating],
    ] = {}
    for statement in statements:
        if not isinstance(statement, dict) or statement.get("rank") == "deprecated":
            continue
        mainsnak = statement.get("mainsnak")
        score_text = _wikidata_text_snak(mainsnak, 80)
        qualifiers = statement.get("qualifiers")
        if score_text is None or not isinstance(qualifiers, dict):
            continue
        issuers = qualifiers.get("P447")
        if not isinstance(issuers, list):
            continue
        review_count = _wikidata_quantity_qualifier(qualifiers, "P7887")
        point_in_time = _wikidata_time_qualifier(qualifiers, "P585")
        score, maximum = _numeric_review_score(score_text)
        qualifier_url = _safe_public_https_url(
            _wikidata_text_snak(_first_qualifier(qualifiers, "P2699"), 2_048)
        )
        source_url = qualifier_url or f"https://www.wikidata.org/wiki/{entity_id}#P444"
        for issuer_snak in issuers:
            issuer_id = _wikidata_item_snak(issuer_snak)
            if issuer_id is None:
                continue
            labels = issuer_labels.get(issuer_id, {})
            platform_zh = labels.get("zh")
            platform_en = labels.get("en")
            platform = platform_en or platform_zh or issuer_id
            try:
                rating = DestinationAttractionRating(
                    platform=platform,
                    platform_zh=platform_zh,
                    platform_en=platform_en,
                    platform_id=issuer_id,
                    score_text=score_text,
                    score=score,
                    max_score=maximum,
                    review_count=review_count,
                    point_in_time=point_in_time,
                    source_url=source_url,
                    subject_wikidata_id=entity_id,
                )
            except ValueError:
                continue
            recency_key = _rating_snapshot_recency_key(
                point_in_time,
                rank=_clean_text(statement.get("rank"), 20),
                review_count=review_count,
                has_numeric_score=score is not None,
            )
            previous = latest_by_issuer.get(issuer_id)
            if previous is None or recency_key > previous[0]:
                latest_by_issuer[issuer_id] = (recency_key, rating)
    return tuple(
        item[1]
        for item in sorted(
            latest_by_issuer.values(),
            key=lambda item: (
                item[1].platform.casefold(),
                item[1].platform_id,
            ),
        )[:20]
    )


def _rating_snapshot_recency_key(
    point_in_time: str | None,
    *,
    rank: str | None,
    review_count: int | None,
    has_numeric_score: bool,
) -> tuple[int, int, int, int, int, int]:
    year = month = day = 0
    if point_in_time:
        parts = point_in_time.split("-")
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 0
            day = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            year = month = day = 0
    return (
        int(point_in_time is not None),
        year,
        month,
        day,
        int(rank == "preferred"),
        int(review_count is not None) + int(has_numeric_score),
    )


def _osm_tag_summary(
    place: DestinationPlace,
    city: DestinationCity,
    language: Literal["zh", "en"],
) -> str:
    category_zh = {
        "landmark": "地标",
        "museum": "博物馆或展馆",
        "nature": "自然或休闲景点",
        "entertainment": "娱乐景点",
        "shopping": "购物景点",
    }
    category_en = {
        "landmark": "landmark",
        "museum": "museum or gallery",
        "nature": "nature or leisure attraction",
        "entertainment": "entertainment attraction",
        "shopping": "shopping attraction",
    }
    if language == "zh":
        summary = (
            f"{place.name} 是 OpenStreetMap 标记为{category_zh[place.category]}的地点，"
            f"位于{city.name}目的地城市范围内。"
        )
        if place.address:
            summary += f"公开地址为：{place.address}。"
        return summary[:2_000]
    name = place.name_en or place.name
    summary = (
        f"{name} is tagged in OpenStreetMap as a {category_en[place.category]} "
        f"within the destination-city area of {city.name}."
    )
    if place.address:
        summary += f" Its published address is {place.address}."
    return summary[:2_000]


def _attraction_enrichment_updates(
    place: DestinationPlace,
    city: DestinationCity,
    *,
    entity: Mapping[str, Any] | None,
    issuer_labels: Mapping[str, Mapping[str, str]],
    wikipedia_extracts: Mapping[tuple[str, str], tuple[str, str]],
    wikidata_failed: bool,
    photo_candidates: tuple[tuple[str, str], ...],
    commons_assets: Mapping[str, Mapping[str, Any]],
    failed_commons_titles: set[str],
    wikipedia_photo_failed: bool,
) -> dict[str, Any]:
    descriptions = _wikidata_language_values(
        entity.get("descriptions") if isinstance(entity, Mapping) else None,
        maximum=2_000,
    )
    labels = _wikidata_language_values(
        entity.get("labels") if isinstance(entity, Mapping) else None,
        maximum=300,
    )
    titles = _wikipedia_titles_for_place(place, entity)
    extracts: dict[str, tuple[str, str]] = {}
    for language, title in titles.items():
        item = wikipedia_extracts.get((language, title.casefold()))
        if item:
            extracts[language] = item

    default_osm = place.description
    description_zh = (
        (extracts.get("zh") or (None, None))[0]
        or descriptions.get("zh")
        or place.description_zh
        or default_osm
        or _osm_tag_summary(place, city, "zh")
    )
    description_en = (
        (extracts.get("en") or (None, None))[0]
        or descriptions.get("en")
        or place.description_en
        or default_osm
        or _osm_tag_summary(place, city, "en")
    )
    if extracts:
        description_source = "wikipedia"
        description_basis = "wikipedia_extract"
        description_url = (extracts.get("zh") or extracts.get("en") or (None, None))[1]
    elif descriptions:
        description_source = "wikidata"
        description_basis = "wikidata_description"
        description_url = (
            f"https://www.wikidata.org/wiki/{place.wikidata_id}"
            if place.wikidata_id
            else place.source_url
        )
    elif place.description or place.description_zh or place.description_en:
        description_source = "openstreetmap"
        description_basis = "osm_description"
        description_url = place.source_url
    else:
        description_source = "openstreetmap"
        description_basis = "osm_tag_summary"
        description_url = place.source_url

    ratings = _parse_wikidata_ratings(
        entity,
        entity_id=place.wikidata_id,
        issuer_labels=issuer_labels,
    )
    ratings_status: AttractionRatingStatus
    if ratings:
        ratings_status = "available"
    elif wikidata_failed:
        ratings_status = "provider_unavailable"
    else:
        ratings_status = "source_not_provided"

    photos: list[DestinationAttractionPhoto] = []
    for file_title, match_basis in photo_candidates:
        asset = commons_assets.get(file_title.casefold())
        if asset is None:
            continue
        try:
            photos.append(
                DestinationAttractionPhoto(
                    **asset,
                    match_basis=match_basis,
                )
            )
        except ValueError:
            continue
    photo_evidence = tuple(photos[:MAX_ATTRACTION_PHOTOS])
    photo_status: AttractionPhotoStatus
    if photo_evidence:
        photo_status = "available"
    elif photo_candidates and any(
        file_title.casefold() in failed_commons_titles
        for file_title, _match_basis in photo_candidates
    ):
        photo_status = "provider_unavailable"
    elif photo_candidates:
        photo_status = "source_rejected"
    elif wikidata_failed or wikipedia_photo_failed:
        photo_status = "provider_unavailable"
    else:
        photo_status = "source_not_provided"

    wikipedia_url = place.wikipedia_url
    if wikipedia_url is None:
        for language in ("zh", "en"):
            title = titles.get(language)
            if title:
                wikipedia_url = _wikipedia_url(language, title)
                break
    return {
        "name_en": place.name_en or labels.get("en"),
        "description": description_zh or description_en,
        "description_zh": description_zh,
        "description_en": description_en,
        "description_source": description_source,
        "description_basis": description_basis,
        "description_url": description_url,
        "wikipedia_url": wikipedia_url,
        "ratings": ratings,
        "ratings_status": ratings_status,
        "photos": photo_evidence,
        "photo_status": photo_status,
        "map_preview": (
            None if photo_evidence else _attraction_map_preview(place.latitude, place.longitude)
        ),
    }


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
    return _overpass_bounded_query(
        kind,
        (south, west, north, east),
        query_timeout_seconds=OVERPASS_QUERY_TIMEOUT_SECONDS,
        element_selector="nwr" if kind == "attraction" else "node",
    )


def _attraction_query_bounds(
    latitude: float,
    longitude: float,
) -> tuple[tuple[float, float, float, float], ...]:
    south, west, north, east = _bounding_box(
        latitude,
        longitude,
        DESTINATION_RADIUS_METERS,
    )
    return (
        (south, west, latitude, longitude),
        (south, longitude, latitude, east),
        (latitude, west, north, longitude),
        (latitude, longitude, north, east),
    )


def _attraction_overpass_query(
    bounds: tuple[float, float, float, float],
) -> str:
    return _overpass_bounded_query(
        "attraction",
        bounds,
        query_timeout_seconds=ATTRACTION_OVERPASS_QUERY_TIMEOUT_SECONDS,
        element_selector="nwr",
    )


def _overpass_bounded_query(
    kind: PlaceKind,
    bounds_tuple: tuple[float, float, float, float],
    *,
    query_timeout_seconds: int,
    element_selector: Literal["node", "nwr"],
) -> str:
    south, west, north, east = bounds_tuple
    bounds = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    if kind == "hotel":
        selectors = [
            f'{element_selector}({bounds})["tourism"="{category}"]["name"]'
            for category in HOTEL_CATEGORY_ORDER
        ]
    else:
        selectors = [
            f'{element_selector}({bounds})["tourism"~"^(attraction|museum|gallery|theme_park|zoo|aquarium|viewpoint|artwork)$"]["name"]',
            f'{element_selector}({bounds})["historic"]["name"]',
            f'{element_selector}({bounds})["natural"~"^(beach|peak|waterfall|cave_entrance|spring)$"]["name"]',
            f'{element_selector}({bounds})["leisure"~"^(park|garden|nature_reserve|water_park|amusement_arcade)$"]["name"]',
            f'{element_selector}({bounds})["shop"~"^(mall|department_store)$"]["name"]',
            f'{element_selector}({bounds})["amenity"~"^(marketplace|theatre|cinema)$"]["name"]',
            f'{element_selector}({bounds})["man_made"~"^(lighthouse|tower)$"]["name"]',
        ]
    return (
        f"[out:json][timeout:{query_timeout_seconds}];("
        + ";".join(selectors)
        + f";);out center {MAX_OVERPASS_ELEMENTS};"
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
    default_description = _clean_text(tags.get("description"), 2_000)
    description_zh = _clean_text(tags.get("description:zh"), 2_000)
    description_en = _clean_text(tags.get("description:en"), 2_000)
    wikidata_id = _wikidata_id(tags.get("wikidata")) if kind == "attraction" else None
    wikipedia_url = _wikipedia_tag_url(tags.get("wikipedia")) if kind == "attraction" else None
    osm_wikimedia_commons_file = (
        _osm_commons_file(tags.get("wikimedia_commons")) if kind == "attraction" else None
    )
    try:
        return DestinationPlace(
            place_id=place_id,
            kind=kind,
            category=category,
            name=name,
            name_en=_clean_text(tags.get("name:en"), 300),
            address=_address_from_tags(tags),
            description=default_description or description_zh or description_en,
            description_zh=description_zh,
            description_en=description_en,
            description_source="openstreetmap"
            if default_description or description_zh or description_en
            else None,
            description_basis="osm_description"
            if default_description or description_zh or description_en
            else None,
            description_url=source_url
            if default_description or description_zh or description_en
            else None,
            wikidata_id=wikidata_id,
            wikipedia_url=wikipedia_url,
            osm_wikimedia_commons_file=osm_wikimedia_commons_file,
            map_preview=(
                _attraction_map_preview(latitude, longitude) if kind == "attraction" else None
            ),
            website=website,
            phone=_contact_phone_from_tags(tags),
            opening_hours=_opening_hours_from_tags(tags),
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
    full = (
        _clean_text(tags.get("addr:full"), 700)
        or _clean_text(tags.get("contact:address"), 700)
        or _clean_text(tags.get("address"), 700)
    )
    if full:
        return full
    house_number = _clean_text(tags.get("addr:housenumber"), 50)
    street = _clean_text(tags.get("addr:street"), 200)
    place_name = _clean_text(tags.get("addr:place"), 200)
    first_line = " ".join(part for part in (house_number, street or place_name) if part)
    neighbourhood = _clean_text(tags.get("addr:neighbourhood"), 150) or _clean_text(
        tags.get("addr:quarter"), 150
    )
    suburb = _clean_text(tags.get("addr:suburb"), 150)
    district = _clean_text(tags.get("addr:city_district"), 150) or _clean_text(
        tags.get("addr:district"), 150
    )
    city = _clean_text(tags.get("addr:city"), 150)
    town = _clean_text(tags.get("addr:town"), 150)
    village = _clean_text(tags.get("addr:village"), 150)
    municipality = _clean_text(tags.get("addr:municipality"), 150)
    county = _clean_text(tags.get("addr:county"), 150)
    postcode = _clean_text(tags.get("addr:postcode"), 50)
    state = _clean_text(tags.get("addr:state"), 150)
    country = _clean_text(tags.get("addr:country"), 100)
    parts = [
        part
        for part in (
            first_line,
            neighbourhood,
            suburb,
            district,
            city or town or village or municipality,
            county,
            state,
            postcode,
            country,
        )
        if part
    ]
    parts = list(dict.fromkeys(parts))
    return ", ".join(parts)[:700] or None


def _contact_phone_from_tags(tags: Mapping[str, Any]) -> str | None:
    for key in (
        "contact:phone",
        "phone",
        "contact:mobile",
        "mobile",
        "operator:phone",
    ):
        value = _clean_text(tags.get(key), 100)
        if value:
            return value
    return None


def _opening_hours_from_tags(tags: Mapping[str, Any]) -> str | None:
    for key in ("opening_hours", "contact:opening_hours"):
        value = _clean_text(tags.get(key), 500)
        if value:
            return value
    return None


def _osm_map_preview_url(latitude: float, longitude: float, zoom: int = 15) -> str:
    return (
        f"{OPENSTREETMAP_BASE_URL}/?mlat={latitude:.7f}&mlon={longitude:.7f}"
        f"#map={zoom}/{latitude:.7f}/{longitude:.7f}"
    )


def _attraction_map_preview(
    latitude: float,
    longitude: float,
) -> DestinationAttractionMapPreview:
    return DestinationAttractionMapPreview(
        latitude=latitude,
        longitude=longitude,
        source_url=_osm_map_preview_url(latitude, longitude),
    )


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


def _is_wikimedia_upload_url(value: str) -> bool:
    try:
        parsed = parse.urlsplit(value)
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "upload.wikimedia.org"
        and parsed.port in {None, 443}
        and parsed.username is None
        and parsed.password is None
        and parsed.path.startswith("/wikipedia/commons/")
    )


def _is_wikimedia_commons_page(value: str) -> bool:
    try:
        parsed = parse.urlsplit(value)
    except ValueError:
        return False
    decoded_path = parse.unquote(parsed.path)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "commons.wikimedia.org"
        and parsed.port in {None, 443}
        and parsed.username is None
        and parsed.password is None
        and decoded_path.startswith("/wiki/File:")
    )


def _commons_page_file_title(value: str) -> str | None:
    if not _is_wikimedia_commons_page(value):
        return None
    parsed = parse.urlsplit(value)
    return _commons_file_title(parse.unquote(parsed.path[len("/wiki/") :]))


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
            place.description_zh,
            place.description_en,
            place.wikidata_id,
            place.wikipedia_url,
            place.ratings if place.ratings else None,
            place.photos if place.photos else None,
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
        raise DestinationValidationError("transit_departure_at must be a timezone-aware datetime")
    normalized = requested.astimezone(UTC).replace(second=0, microsecond=0)
    if normalized < reference - timedelta(days=1):
        raise DestinationValidationError(
            "transit_departure_at cannot be more than one day in the past"
        )
    if normalized > reference + timedelta(days=370):
        raise DestinationValidationError("transit_departure_at must be within 370 days")
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


def _osm_transit_reference_query(latitude: float, longitude: float) -> str:
    lat = _safe_float(latitude, -90, 90)
    lon = _safe_float(longitude, -180, 180)
    if lat is None or lon is None:
        raise DestinationValidationError("transit reference coordinates are invalid")
    return (
        f"[out:json][timeout:{OVERPASS_QUERY_TIMEOUT_SECONDS}];"
        f'rel(around:5000,{lat:.7f},{lon:.7f})["type"="route"]'
        '["route"~"^(bus|train|subway|light_rail|tram)$"]->.routes;'
        ".routes out body 60;"
        'node(r.routes)["name"];'
        "out tags 500;"
    )


def _parse_osm_transit_reference_payload(
    payload: Any,
) -> dict[int, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise DestinationDataUnavailable("OSM transit reference response is invalid")
    elements = payload.get("elements")
    if not isinstance(elements, list) or len(elements) > 5_000:
        raise DestinationDataUnavailable("OSM transit reference elements are invalid")
    if _clean_text(payload.get("remark"), 1_000) is not None:
        raise DestinationDataUnavailable("OSM transit reference query was incomplete")

    node_names: dict[int, str] = {}
    for element in elements:
        if not isinstance(element, Mapping) or element.get("type") != "node":
            continue
        node_id = _bounded_int(element.get("id"), 1, 9_223_372_036_854_775_807)
        tags = element.get("tags")
        name = _clean_text(tags.get("name"), 300) if isinstance(tags, Mapping) else None
        if node_id is not None and name is not None:
            node_names[node_id] = name

    allowed_modes = {"bus", "train", "subway", "light_rail", "tram"}
    result: dict[int, dict[str, Any]] = {}
    for element in elements:
        if not isinstance(element, Mapping) or element.get("type") != "relation":
            continue
        relation_id = _bounded_int(
            element.get("id"),
            1,
            9_223_372_036_854_775_807,
        )
        tags = element.get("tags")
        if relation_id is None or not isinstance(tags, Mapping):
            continue
        route_mode = str(tags.get("route") or "").strip().casefold()
        if route_mode not in allowed_modes:
            continue
        name = _clean_text(tags.get("name"), 300)
        ref = _clean_text(tags.get("ref"), 100)
        if name is None and ref is None:
            continue
        stops: list[str] = []
        seen_stops: set[str] = set()
        members = element.get("members")
        if isinstance(members, list):
            for member in members[:1_000]:
                if not isinstance(member, Mapping) or member.get("type") != "node":
                    continue
                role = str(member.get("role") or "").strip().casefold()
                if role not in {
                    "stop",
                    "platform",
                    "stop_entry_only",
                    "stop_exit_only",
                    "platform_entry_only",
                    "platform_exit_only",
                    "forward_stop",
                    "backward_stop",
                    "forward_platform",
                    "backward_platform",
                }:
                    continue
                node_id = _bounded_int(
                    member.get("ref"),
                    1,
                    9_223_372_036_854_775_807,
                )
                stop_name = node_names.get(node_id) if node_id is not None else None
                if stop_name is not None and stop_name.casefold() not in seen_stops:
                    stops.append(stop_name)
                    seen_stops.add(stop_name.casefold())
                if len(stops) >= 100:
                    break
        if not stops:
            continue
        result[relation_id] = {
            "relation_id": relation_id,
            "route_mode": route_mode,
            "name": name,
            "ref": ref,
            "operator": _clean_text(tags.get("operator"), 300),
            "network": _clean_text(tags.get("network"), 300),
            # This is the exact OSM tag text. It is not inferred from geometry.
            "duration_label": _clean_text(tags.get("duration"), 80),
            "stops": tuple(stops),
        }
    return result


def _merge_osm_transit_references(
    airport_routes: Mapping[int, Mapping[str, Any]],
    destination_routes: Mapping[int, Mapping[str, Any]],
) -> tuple[DestinationTransitRouteReference, ...]:
    mode_order = {
        "subway": 0,
        "train": 1,
        "light_rail": 2,
        "tram": 3,
        "bus": 4,
    }
    candidates: list[DestinationTransitRouteReference] = []
    for relation_id in set(airport_routes) | set(destination_routes):
        airport_value = airport_routes.get(relation_id)
        destination_value = destination_routes.get(relation_id)
        values = tuple(
            value for value in (airport_value, destination_value) if isinstance(value, Mapping)
        )
        if not values:
            continue
        most_complete = max(
            values,
            key=lambda value: len(value.get("stops", ())),
        )
        stops: list[str] = []
        seen: set[str] = set()
        for value in values:
            raw_stops = value.get("stops")
            if not isinstance(raw_stops, (tuple, list)):
                continue
            for raw_stop in raw_stops:
                stop = _clean_text(raw_stop, 300)
                if stop is not None and stop.casefold() not in seen:
                    stops.append(stop)
                    seen.add(stop.casefold())
                if len(stops) >= 100:
                    break
        if not stops:
            continue
        try:
            candidates.append(
                DestinationTransitRouteReference(
                    relation_id=relation_id,
                    route_mode=most_complete.get("route_mode"),
                    name=most_complete.get("name"),
                    ref=most_complete.get("ref"),
                    operator=most_complete.get("operator"),
                    network=most_complete.get("network"),
                    duration_label=most_complete.get("duration_label"),
                    near_airport=airport_value is not None,
                    near_destination=destination_value is not None,
                    stops=tuple(stops),
                    source_url=f"{OPENSTREETMAP_BASE_URL}/relation/{relation_id}",
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                0 if item.near_airport and item.near_destination else 1,
                mode_order[item.route_mode],
                (item.ref or "").casefold(),
                (item.name or "").casefold(),
                item.relation_id,
            ),
        )[:12]
    )


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
                round(distance_meters / 1_000, 1) if distance_meters is not None else None
            ),
            line_name=line_name,
            headsign=_clean_text(value.get("headsign"), 300),
            agency_name=_clean_text(value.get("agencyName"), 300),
            intermediate_stops=_transit_intermediate_stops(value.get("intermediateStops")),
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


def _serpapi_transit_option(
    result: SerpApiTransitResult,
    *,
    requested_departure_at: datetime,
    departure_time_basis: Literal["user_supplied", "request_time"],
    expires_at: datetime,
) -> DestinationTransportOption:
    """Convert sanitized Google Maps transit evidence without inventing times."""

    observed_at = _aware_utc(result.observed_at)
    safe_expires_at = max(expires_at, observed_at + timedelta(minutes=1))
    if result.status != "available":
        coverage_status = "no_itinerary" if result.status == "no_results" else result.status
        return DestinationTransportOption(
            mode="public_transit",
            status="unavailable",
            requested_departure_at=requested_departure_at,
            departure_time_basis=departure_time_basis,
            coverage_status=coverage_status,
            notice=result.message,
            data_source="serpapi_google_maps_directions",
            source_url=SERPAPI_TRANSIT_SOURCE_URL,
            observed_at=observed_at,
            expires_at=safe_expires_at,
        )

    legs = tuple(
        DestinationTransitLeg(
            mode=leg.mode,
            from_name=leg.from_name,
            to_name=leg.to_name,
            departure_time_label=leg.departure_label,
            arrival_time_label=leg.arrival_label,
            duration_minutes=leg.duration_minutes,
            distance_km=leg.distance_km,
            line_name=leg.line_name,
            headsign=leg.headsign,
            agency_name=leg.agency_name,
            intermediate_stops=leg.intermediate_stops,
            realtime=False,
            scheduled=True,
        )
        for leg in result.legs
    )
    return DestinationTransportOption(
        mode="public_transit",
        status="available",
        distance_km=result.distance_km,
        duration_minutes=result.duration_minutes,
        duration_basis="transit_schedule_or_realtime",
        requested_departure_at=requested_departure_at,
        departure_time_basis=departure_time_basis,
        departure_at=result.departure_at,
        arrival_at=result.arrival_at,
        departure_time_label=result.departure_label,
        arrival_time_label=result.arrival_label,
        transfers=result.transfers,
        realtime=False,
        legs=legs,
        coverage_status="covered",
        notice=result.message,
        data_source="serpapi_google_maps_directions",
        source_url=SERPAPI_TRANSIT_SOURCE_URL,
        observed_at=observed_at,
        expires_at=safe_expires_at,
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
        ("www.wikidata.org", "/w/api.php"),
        ("commons.wikimedia.org", "/w/api.php"),
        ("zh.wikipedia.org", "/w/api.php"),
        ("en.wikipedia.org", "/w/api.php"),
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
