"""HTTP routes for source-backed destination guides and explicit hotel prices."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from flight_forecaster.destination_guide import (
    DestinationAirportNotFound,
    DestinationDataUnavailable,
    DestinationGuideService,
    DestinationPlaceNotFound,
    DestinationValidationError,
    build_destination_guide_service,
    validate_transit_departure_at,
)
from flight_forecaster.hotel_prices import (
    HotelPriceError,
    HotelPriceOffer,
    HotelPriceValidationError,
    SerpApiHotelPriceProvider,
    hotel_price_provider_from_env,
)
from flight_forecaster.serpapi_transit import serpapi_transit_provider_from_env

router = APIRouter()

Language = Literal["zh", "zh-cn", "en"]
PlaceKind = Literal["attraction", "hotel"]
PlaceCategory = Literal[
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


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DestinationPlacesRequest(_RequestModel):
    destination: str = Field(pattern=r"^[A-Za-z]{3}$")
    kind: PlaceKind
    category: PlaceCategory = "all"
    limit: int = Field(default=300, ge=1, le=300)
    language: Language = "zh"


class DestinationPlaceDetailRequest(_RequestModel):
    destination: str = Field(pattern=r"^[A-Za-z]{3}$")
    kind: PlaceKind
    place_id: str = Field(
        min_length=20,
        max_length=80,
        pattern=r"^osm_(?:attraction|hotel)_(?:node|way|relation)_[1-9][0-9]{0,18}$",
    )
    language: Language = "zh"
    transit_departure_at: AwareDatetime | None = None
    include_live_transit: bool = False


class HotelPricesRequest(_RequestModel):
    destination: str = Field(pattern=r"^[A-Za-z]{3}$")
    check_in: date
    check_out: date
    adults: int = Field(default=1, ge=1, le=8)
    language: Literal["zh-cn", "en"] = "zh-cn"


class HotelPriceDetailRequest(HotelPricesRequest):
    hotel_id: str | None = Field(
        default=None,
        pattern=r"^gh_[a-f0-9]{32}$",
    )
    place_id: str | None = Field(
        default=None,
        max_length=80,
        pattern=r"^osm_hotel_(?:node|way|relation)_[1-9][0-9]{0,18}$",
    )
    transit_departure_at: AwareDatetime | None = None
    include_live_transit: bool = False

    @model_validator(mode="after")
    def requires_one_hotel_identity(self) -> HotelPriceDetailRequest:
        if (self.hotel_id is None) == (self.place_id is None):
            raise ValueError("provide exactly one of hotel_id or place_id")
        return self


def _runtime_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", "artifacts/demo")).parent / "runtime"


@lru_cache(maxsize=1)
def get_destination_guide_service() -> DestinationGuideService:
    return build_destination_guide_service(
        serpapi_transit_provider=serpapi_transit_provider_from_env(
            _runtime_dir() / "serpapi-usage.sqlite3"
        )
    )


@lru_cache(maxsize=1)
def get_hotel_price_provider() -> SerpApiHotelPriceProvider:
    return hotel_price_provider_from_env(_runtime_dir() / "serpapi-usage.sqlite3")


def _language(value: Language) -> Literal["zh-cn", "en"]:
    return "en" if value == "en" else "zh-cn"


def _transport_payload(
    transport: object,
) -> tuple[dict[str, object], list[dict[str, object]], str, str]:
    payload = transport.model_dump(mode="json")  # type: ignore[attr-defined]
    options = [
        item
        for item in payload["options"]
        if isinstance(item, dict)
    ]
    transit = next(
        (item for item in options if item.get("mode") == "public_transit"),
        None,
    )
    notice = str(transit.get("notice", "")) if transit else ""
    source = "+".join(
        dict.fromkeys(
            value
            for item in options
            if (value := str(item.get("data_source") or "").strip())
        )
    )
    return payload, options, notice, source


def _source_chain(*values: str) -> str:
    return "+".join(
        dict.fromkeys(
            component
            for value in values
            for component in value.split("+")
            if component
        )
    )


def _formatted_usd(value: float | None, *, suffix: str = "") -> str | None:
    if value is None:
        return None
    return f"US${value:,.2f}{suffix}"


def _safe_offer(
    offer: HotelPriceOffer,
    *,
    language: Literal["zh-cn", "en"],
) -> dict[str, object]:
    nightly_suffix = " / 晚" if language == "zh-cn" else " / night"
    item = offer.as_safe_dict()
    item.update(
        {
            "provider_hotel_id": offer.hotel_id,
            "type": offer.property_type,
            "category": _hotel_category(offer.property_type),
            "website": offer.website_url,
            "url": offer.website_url,
            "provider": offer.price_source or "Google Hotels",
            "formatted_nightly_price": _formatted_usd(
                offer.nightly_price,
                suffix=nightly_suffix,
            ),
            "formatted_total_price": _formatted_usd(offer.total_price),
            "formatted_price": (
                _formatted_usd(offer.nightly_price, suffix=nightly_suffix)
                or _formatted_usd(offer.total_price)
            ),
            "taxes_included": None,
        }
    )
    return item


def _hotel_category(property_type: str) -> str:
    normalized = property_type.casefold().replace("-", " ").replace("_", " ")
    if "hostel" in normalized:
        return "hostel"
    if "guest" in normalized or "bed and breakfast" in normalized:
        return "guest_house"
    if "motel" in normalized:
        return "motel"
    if "apartment" in normalized or "aparthotel" in normalized:
        return "apartment"
    return "hotel"


def _destination_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DestinationValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (DestinationAirportNotFound, DestinationPlaceNotFound)):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(
        status_code=503,
        detail="Destination place data is temporarily unavailable.",
    )


def _validate_transit_time_before_external_calls(value: datetime | None) -> None:
    """Reject semantically invalid transit times before any provider call."""

    try:
        validate_transit_departure_at(value, observed_at=datetime.now(UTC))
    except DestinationValidationError as exc:
        raise _destination_error(exc) from exc


def _hotel_error(exc: HotelPriceError) -> HTTPException:
    status = {
        "validation_error": 422,
        "not_configured": 503,
        "authentication_failed": 502,
        "quota_exhausted": 429,
        "rate_limited": 429,
        "provider_processing": 503,
        "provider_error": 502,
        "provider_unavailable": 503,
        "response_invalid": 502,
        "quota_ledger_unavailable": 503,
    }.get(exc.code, 503)
    return HTTPException(
        status_code=status,
        detail={
            "code": exc.code,
            "message": str(exc),
            "quota_scope": exc.quota_scope,
        },
    )


@router.get("/details/attractions", include_in_schema=False)
def attractions_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "attractions.html")


@router.get("/details/hotels", include_in_schema=False)
def hotels_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "hotels.html")


@router.get("/details/place", include_in_schema=False)
def destination_place_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "place.html")


@router.post("/v1/destination/places")
def destination_places(request: DestinationPlacesRequest) -> dict[str, object]:
    try:
        listing = get_destination_guide_service().list_places(
            request.destination,
            request.kind,
            category=request.category,
            limit=request.limit,
        )
    except (
        DestinationValidationError,
        DestinationAirportNotFound,
        DestinationDataUnavailable,
    ) as exc:
        raise _destination_error(exc) from exc
    payload = listing.model_dump(mode="json")
    payload.update(
        {
            "destination_airport": listing.city.destination_airport,
            "status": "available" if listing.places else "no_results",
            "source": (
                "openstreetmap_overpass+wikimedia"
                if listing.kind == "attraction"
                else listing.data_source
            ),
            "observed_at": listing.fetched_at.isoformat(),
            "available_result_count": listing.result_count,
            "result_limit": request.limit,
        }
    )
    return payload


@router.post("/v1/destination/place-detail")
def destination_place_detail(
    request: DestinationPlaceDetailRequest,
) -> dict[str, object]:
    _validate_transit_time_before_external_calls(request.transit_departure_at)
    expected_prefix = f"osm_{request.kind}_"
    if not request.place_id.startswith(expected_prefix):
        raise HTTPException(status_code=422, detail="place_id does not match kind")
    try:
        detail = get_destination_guide_service().get_place_detail(
            request.destination,
            request.place_id,
            transit_departure_at=request.transit_departure_at,
            include_live_transit=request.include_live_transit,
        )
    except (
        DestinationValidationError,
        DestinationAirportNotFound,
        DestinationDataUnavailable,
        DestinationPlaceNotFound,
    ) as exc:
        raise _destination_error(exc) from exc
    payload = detail.model_dump(mode="json")
    transport, routes, transit_notice, transport_source = _transport_payload(
        detail.transport
    )
    observed_at = max(option.observed_at for option in detail.transport.options)
    payload.update(
        {
            "destination_airport": detail.city.destination_airport,
            "transport": transport,
            "routes": routes,
            "transit_notice": transit_notice,
            "source": _source_chain(
                (
                    "openstreetmap_overpass+wikimedia"
                    if request.kind == "attraction"
                    else "openstreetmap_overpass"
                ),
                transport_source,
            ),
            "observed_at": observed_at.isoformat(),
        }
    )
    return payload


@router.post("/v1/destination/hotel-prices")
def destination_hotel_prices(request: HotelPricesRequest) -> dict[str, object]:
    guide = get_destination_guide_service()
    try:
        city = guide.resolve_city(request.destination)
    except (
        DestinationValidationError,
        DestinationAirportNotFound,
        DestinationDataUnavailable,
    ) as exc:
        raise _destination_error(exc) from exc
    try:
        result = get_hotel_price_provider().search(
            city.city_query,
            city.destination_airport,
            request.check_in,
            request.check_out,
            adults=request.adults,
            language=_language(request.language),
            explicit=True,
        )
    except (HotelPriceValidationError, HotelPriceError) as exc:
        raise _hotel_error(exc) from exc
    offers = [
        _safe_offer(item, language=_language(request.language))
        for item in result.offers
    ]
    warning = (
        "结果来自一小时脱敏缓存，本次未消耗 SerpApi 查询额度。"
        if request.language != "en" and result.cache_hit
        else "Results came from the one-hour sanitized cache; no SerpApi query was used."
        if result.cache_hit
        else "本次实时酒店价格查询与严格机票查询共用 SerpApi 免费额度。"
        if request.language != "en"
        else (
            "This live hotel-price search shares the SerpApi free quota "
            "with strict flight searches."
        )
    )
    return {
        "city": city.model_dump(mode="json"),
        "destination_airport": city.destination_airport,
        "status": result.status,
        "offers": offers,
        "provider_code": result.provider_code,
        "provider_name": result.provider_name,
        "source": result.provider_code,
        "observed_at": result.observed_at.isoformat(),
        "cache_hit": result.cache_hit,
        "calls_reserved": result.calls_reserved,
        "quota_monthly_used": result.quota_monthly_used,
        "quota_monthly_limit": result.quota_monthly_limit,
        "quota_hourly_used": result.quota_hourly_used,
        "quota_hourly_limit": result.quota_hourly_limit,
        "quota_warning": warning,
    }


@router.post("/v1/destination/hotel-price-detail")
def destination_hotel_price_detail(
    request: HotelPriceDetailRequest,
) -> dict[str, object]:
    _validate_transit_time_before_external_calls(request.transit_departure_at)
    guide = get_destination_guide_service()
    try:
        city = guide.resolve_city(request.destination)
    except (
        DestinationValidationError,
        DestinationAirportNotFound,
        DestinationDataUnavailable,
    ) as exc:
        raise _destination_error(exc) from exc
    osm_place = None
    route_result = None
    if request.place_id is not None:
        try:
            osm_detail = guide.get_place_detail(
                city.destination_airport,
                request.place_id,
                transit_departure_at=request.transit_departure_at,
                include_live_transit=request.include_live_transit,
            )
        except (
            DestinationValidationError,
            DestinationAirportNotFound,
            DestinationPlaceNotFound,
            DestinationDataUnavailable,
        ) as exc:
            raise _destination_error(exc) from exc
        osm_place = osm_detail.place
        route_result = osm_detail.transport
        hotel_names = tuple(
            dict.fromkeys(
                name
                for name in (osm_place.name, osm_place.name_en)
                if isinstance(name, str) and name.strip()
            )
        )
        try:
            offer = get_hotel_price_provider().exact_property_detail(
                hotel_names,
                osm_place.latitude,
                osm_place.longitude,
                city.city_query,
                city.destination_airport,
                request.check_in,
                request.check_out,
                adults=request.adults,
                language=_language(request.language),
                explicit=True,
            )
        except (HotelPriceValidationError, HotelPriceError) as exc:
            raise _hotel_error(exc) from exc
        if offer is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The provider did not return exactly one hotel whose name and "
                    "coordinates match this OpenStreetMap property."
                ),
            )
    else:
        assert request.hotel_id is not None
        try:
            offer = get_hotel_price_provider().detail(
                request.hotel_id,
                city.city_query,
                city.destination_airport,
                request.check_in,
                request.check_out,
                adults=request.adults,
                language=_language(request.language),
                explicit=True,
            )
        except (HotelPriceValidationError, HotelPriceError) as exc:
            raise _hotel_error(exc) from exc
        if offer is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The requested hotel quote is not present in the one-hour "
                    "sanitized cache."
                ),
            )
    if route_result is None:
        try:
            route_result = guide.get_routes(
                city.destination_airport,
                offer.latitude,
                offer.longitude,
                transit_departure_at=request.transit_departure_at,
                include_live_transit=request.include_live_transit,
            )
        except (
            DestinationValidationError,
            DestinationAirportNotFound,
            DestinationDataUnavailable,
        ) as exc:
            raise _destination_error(exc) from exc
    transport, routes, transit_notice, transport_source = _transport_payload(
        route_result
    )
    price = _safe_offer(offer, language=_language(request.language))
    if osm_place is None:
        hotel = {
            **price,
            "place_id": None,
            "kind": "hotel",
            "address": None,
            "opening_hours": None,
        }
        source = _source_chain("serpapi_google_hotels", transport_source)
    else:
        hotel = {
            **price,
            "place_id": osm_place.place_id,
            "kind": "hotel",
            "name_en": osm_place.name_en,
            "address": osm_place.address,
            "opening_hours": osm_place.opening_hours,
            "phone": osm_place.phone,
            "source_url": osm_place.source_url,
        }
        source = _source_chain(
            "openstreetmap_overpass",
            "serpapi_google_hotels",
            transport_source,
        )
    return {
        "city": city.model_dump(mode="json"),
        "destination_airport": city.destination_airport,
        "hotel": hotel,
        "place": hotel,
        "offer": price,
        "offers": [price],
        "transport": transport,
        "routes": routes,
        "transit_notice": transit_notice,
        "provider": "SerpApi Google Hotels",
        "provider_name": "SerpApi Google Hotels",
        "source": source,
        "observed_at": (
            offer.detail_observed_at or offer.observed_at
        ).isoformat(),
        "warning": (
            "价格和空房仅代表查询时刻；税费、最终金额与退改规则以提供商结账页为准。"
            if request.language != "en"
            else (
                "Price and availability reflect query time; confirm taxes, final total, "
                "and change rules at checkout."
            )
        ),
    }
