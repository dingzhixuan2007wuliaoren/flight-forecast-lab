from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from flight_forecaster.airlabs_quota import AirLabsQuotaGate
from flight_forecaster.availability import (
    ConfirmedFlightOffer,
    FlightOfferSearchResult,
    FlightOfferSegment,
    RouteCabinMarketHistory,
    flight_offer_provider_from_env,
)
from flight_forecaster.catalog import AirlineProfile, get_airline_profile
from flight_forecaster.context import (
    AIRLABS_FREE_SAMPLE_LIMIT,
    ContextProvider,
    PredictionContext,
)
from flight_forecaster.details import DetailProvider
from flight_forecaster.features import (
    build_ontime_features,
    build_ontime_features_without_weather,
    build_price_features,
)
from flight_forecaster.route_info import (
    Airport,
    AirportResolver,
    OurAirportsResolver,
    RouteEstimate,
    RouteLookupError,
    estimate_route,
)
from flight_forecaster.schedules import ScheduleProvider
from flight_forecaster.schemas import (
    MAX_STRICT_ITINERARY_SEGMENTS,
    BilingualText,
    BilingualWarning,
    ComparisonOffer,
    ComparisonRankings,
    ComparisonRequest,
    ComparisonResponse,
    ContextDetailRequest,
    ContextSignal,
    FareCoverageReference,
    FareSearchMetadata,
    HistoricalMarketContext,
    HistoricalMarketPricePoint,
    ItineraryLayover,
    ItineraryLeg,
    LiveFare,
    NewsArticle,
    NewsDetailResponse,
    NewsSignal,
    OfferDetailRequest,
    OfferDetailResponse,
    OfferItinerary,
    OnTimePrediction,
    OnTimeRequest,
    OperationsEvent,
    OperationsMetric,
    OperationsSignal,
    OperationsSnapshot,
    PredictionContextResponse,
    PriceForecastCurve,
    PriceForecastPoint,
    PricePrediction,
    PriceRequest,
    ProviderOfferSegment,
    TimetableReference,
    WeatherDetailResponse,
)
from flight_forecaster.scrapedo_reference import (
    SCRAPE_DO_PROVIDER_CODE,
    SCRAPE_DO_PROVIDER_NAME,
    ScrapeDoReferenceResult,
    scrapedo_reference_provider_from_env,
)
from flight_forecaster.supplemental_aviation import (
    aerodatabox_provider_from_env,
    opensky_provider_from_env,
    supplemental_usage_path,
)
from flight_forecaster.training import ARTIFACT_FILENAME, SCHEMA_VERSION

_FARE_PROVIDER_SOURCE_URLS = {
    "serpapi_google_flights": "https://serpapi.com/google-flights-api",
    "searchapi_google_flights": "https://www.searchapi.io/google-flights",
    "scrappa_google_flights": "https://scrappa.co/services/google-flights-api",
    "ignav_verified_fares": "https://ignav.com/",
}
_FARE_PROVIDER_SCHEDULE_SOURCES = {
    "serpapi_google_flights": "serpapi_google_flights_booking",
    "searchapi_google_flights": "searchapi_google_flights_booking",
    "scrappa_google_flights": "scrappa_google_flights_booking",
    "ignav_verified_fares": "ignav_verified_booking",
}
_FARE_PROVIDER_DETAIL_BASES = {
    "serpapi_google_flights": "serpapi_booking_confirmed",
    "searchapi_google_flights": "searchapi_booking_confirmed",
    "scrappa_google_flights": "scrappa_booking_confirmed",
    "ignav_verified_fares": "ignav_verified_booking_confirmed",
}


@lru_cache(maxsize=1)
def _timezone_finder() -> TimezoneFinder:
    return TimezoneFinder(in_memory=True)


def _local_time_features(value: datetime) -> dict[str, int]:
    return {
        "departure_local_month": value.month,
        "departure_local_weekday": value.weekday(),
        "departure_local_hour": value.hour,
    }


@dataclass(frozen=True)
class _GenericAirlineProfile:
    code: str
    name: str
    # Unknown airlines must never be expanded into invented premium cabins.
    # Strict comparison excludes them until the curated cabin catalogue knows them.
    supported_cabins: tuple[str, ...] = ("economy",)
    baggage_status: str = "unknown"
    student_status: str = "unknown"
    change_status: str = "unknown"
    refund_status: str = "unknown"
    student_age_limit_zh: str = "未知；请向航司核实。"
    student_age_limit_en: str = "Unknown; verify with the airline."
    student_verification_zh: str = "未知；请向航司核实。"
    student_verification_en: str = "Unknown; verify with the airline."
    student_program_url: str | None = None


class OfferNotFoundError(ValueError):
    """Raised when a share-safe offer id cannot be recomputed for a request."""


@dataclass(frozen=True)
class _OfferScenario:
    profile: AirlineProfile | _GenericAirlineProfile
    cabin: str
    route_status: str
    stops: int | None
    model_stops: int
    routing_status: str
    departure_time: datetime
    duration_minutes: int
    distance_km: float
    confirmed_offer: ConfirmedFlightOffer
    provider_segments: tuple[ProviderOfferSegment, ...]


# Deterministic primary hubs are used only to explain a modelled one-stop scenario.
# They are not presented as a published itinerary or a claim that the carrier flies
# either leg. Alternatives avoid choosing an endpoint as its own connection airport.
_AIRLINE_HUBS: dict[str, tuple[str, ...]] = {
    "AA": ("DFW", "CLT", "MIA"),
    "AC": ("YYZ", "YUL", "YVR"),
    "AM": ("MEX",),
    "AS": ("SEA", "PDX"),
    "B6": ("JFK", "BOS"),
    "DL": ("ATL", "DTW", "SLC"),
    "F9": ("DEN",),
    "NK": ("FLL",),
    "UA": ("ORD", "DEN", "SFO"),
    "WN": ("DEN", "BWI"),
    "WS": ("YYC", "YVR", "YYZ"),
    "AD": ("GRU",),
    "AR": ("EZE",),
    "AV": ("BOG",),
    "CM": ("PTY",),
    "G3": ("GRU",),
    "LA": ("SCL", "LIM", "GRU"),
    "AF": ("CDG",),
    "AY": ("HEL",),
    "BA": ("LHR", "LGW"),
    "EI": ("DUB",),
    "FR": ("DUB",),
    "IB": ("MAD",),
    "KL": ("AMS",),
    "LH": ("FRA", "MUC"),
    "LO": ("WAW",),
    "LX": ("ZRH",),
    "OS": ("VIE",),
    "SK": ("CPH", "OSL"),
    "TK": ("IST",),
    "TP": ("LIS",),
    "U2": ("LGW",),
    "VS": ("LHR",),
    "AT": ("CMN",),
    "EK": ("DXB",),
    "ET": ("ADD",),
    "EY": ("AUH",),
    "KQ": ("NBO",),
    "MS": ("CAI",),
    "QR": ("DOH",),
    "SA": ("JNB",),
    "SV": ("JED", "RUH"),
    "6E": ("DEL", "BOM"),
    "AI": ("DEL", "BOM"),
    "BR": ("TPE",),
    "CA": ("PEK",),
    "CX": ("HKG",),
    "CZ": ("CAN",),
    "GA": ("CGK",),
    "JL": ("HND", "NRT"),
    "KE": ("ICN",),
    "MH": ("KUL",),
    "MU": ("PVG",),
    "NH": ("HND", "NRT"),
    "NZ": ("AKL",),
    "PR": ("MNL",),
    "QF": ("SYD", "MEL"),
    "SQ": ("SIN",),
    "TG": ("BKK",),
    "VN": ("SGN", "HAN"),
}


class PredictionService:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        context_provider: ContextProvider | None = None,
        airport_resolver: AirportResolver | None = None,
        detail_provider: DetailProvider | None = None,
        schedule_provider: ScheduleProvider | None = None,
        flight_offer_provider: Any | None = None,
        fare_reference_provider: Any | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        artifact_path = self.model_dir / ARTIFACT_FILENAME
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"model artifact not found at {artifact_path}; run train-demo first"
            )
        # joblib/pickle can execute code while loading. Only load locally produced artifacts.
        self.bundle: dict[str, Any] = joblib.load(artifact_path)
        if self.bundle.get("artifact_schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported model artifact schema; run train-demo again")

        supplemental_usage = supplemental_usage_path(
            self.model_dir.parent / "runtime" / "supplemental-aviation-usage.sqlite3"
        )
        airlabs_quota_gate = AirLabsQuotaGate.from_env(
            default_usage_path=self.model_dir.parent / "runtime" / "airlabs-usage.sqlite3"
        )
        self.context_provider = context_provider or ContextProvider(
            airlabs_api_key=os.getenv("AIRLABS_API_KEY"),
            context_priors=self.bundle.get("context_priors"),
            opensky_provider=opensky_provider_from_env(supplemental_usage),
            airlabs_quota_gate=airlabs_quota_gate,
        )
        self.detail_provider = detail_provider or DetailProvider(self.context_provider)
        self.schedule_provider = schedule_provider or ScheduleProvider(
            api_key=self.context_provider.airlabs_api_key,
            client=self.context_provider.client,
            enabled=self.context_provider.external_context_enabled,
            aerodatabox_provider=aerodatabox_provider_from_env(supplemental_usage),
            airlabs_quota_gate=(
                self.context_provider.airlabs_quota_gate
                if context_provider is not None
                else airlabs_quota_gate
            ),
        )
        self.flight_offer_provider = (
            flight_offer_provider
            if flight_offer_provider is not None
            else flight_offer_provider_from_env(
                self.model_dir.parent / "runtime" / "serpapi-usage.sqlite3"
            )
        )
        self.fare_reference_provider = (
            fare_reference_provider
            if fare_reference_provider is not None
            else scrapedo_reference_provider_from_env(
                self.model_dir.parent / "runtime" / "scrapedo-reference-usage.sqlite3"
            )
        )
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        if airport_resolver is not None:
            self.airport_resolver = airport_resolver
        elif (
            self.context_provider.external_context_enabled or self.flight_offer_provider.configured
        ):
            self.airport_resolver = OurAirportsResolver()
        else:
            self.airport_resolver = None

    @property
    def model_version(self) -> str:
        return str(self.bundle["model_version"])

    def _route(self, origin: str, destination: str, stops: int = 0) -> RouteEstimate:
        return estimate_route(
            origin,
            destination,
            stops,
            resolver=self.airport_resolver,
        )

    def _context(
        self,
        route: RouteEstimate,
        departure_time: datetime,
    ) -> PredictionContext:
        return self.context_provider.resolve(
            route.origin.iata,
            route.destination.iata,
            departure_time,
            route.origin.latitude,
            route.origin.longitude,
            airport_type=route.origin.type,
            icao_code=route.origin.icao or None,
            origin_name=route.origin.name,
            destination_name=route.destination.name,
            origin_country=route.origin.country,
        )

    @staticmethod
    def _airport_timezone(airport: Airport) -> tuple[ZoneInfo, str]:
        timezone_name = _timezone_finder().timezone_at(
            lng=airport.longitude,
            lat=airport.latitude,
        )
        if not timezone_name:
            raise RouteLookupError(
                f"无法确定 {airport.iata} 的时区 / unable to resolve airport timezone"
            )
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise RouteLookupError(
                f"无法加载 {airport.iata} 的时区 / unable to load airport timezone"
            ) from exc
        return timezone, timezone_name

    @staticmethod
    def _departure_at_origin(value: datetime, airport: Airport) -> tuple[datetime, str]:
        timezone, timezone_name = PredictionService._airport_timezone(airport)

        if value.tzinfo is None or value.utcoffset() is None:
            departure = value.replace(tzinfo=timezone, fold=0)
            round_trip = departure.astimezone(UTC).astimezone(timezone).replace(tzinfo=None)
            if round_trip != value:
                raise RouteLookupError(
                    "计划时间落在夏令时不存在的时段 / departure falls in a DST gap"
                )
            alternate = value.replace(tzinfo=timezone, fold=1)
            if alternate.utcoffset() != departure.utcoffset():
                raise RouteLookupError(
                    "计划时间在夏令时切换中有歧义，请改选其他时间 / "
                    "departure is ambiguous during a DST transition"
                )
        else:
            departure = value.astimezone(timezone)

        now = datetime.now(UTC)
        departure_utc = departure.astimezone(UTC)
        if departure_utc <= now:
            raise RouteLookupError("计划出发时间必须晚于当前时间 / departure must be in the future")
        if departure_utc > now + timedelta(days=370):
            raise RouteLookupError(
                "计划出发时间不得超过 370 天 / departure must be within 370 days"
            )
        return departure, timezone_name

    @staticmethod
    def _same_day_safe_reference(
        generated_at: datetime,
        timezone: ZoneInfo,
    ) -> datetime:
        """Return a real instant 30 minutes later in the origin timezone."""

        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise RouteLookupError("generated_at must include a timezone offset")
        # Move along the UTC timeline before converting back to local time. Direct
        # local wall-clock arithmetic can land in a DST gap or choose the wrong fold.
        return (generated_at.astimezone(UTC) + timedelta(minutes=30)).astimezone(timezone)

    @staticmethod
    def _departure_date_at_origin(
        value: date,
        airport: Airport,
        generated_at: datetime,
    ) -> tuple[datetime, str, str]:
        timezone, timezone_name = PredictionService._airport_timezone(airport)
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise RouteLookupError("generated_at must include a timezone offset")
        local_now = generated_at.astimezone(timezone)
        today = local_now.date()
        if value < today:
            raise RouteLookupError(
                "计划出发日期不能早于出发机场当地今天 / "
                "departure_date cannot be before today at the origin"
            )
        if value > today + timedelta(days=370):
            raise RouteLookupError(
                "计划出发日期不得超过 370 天 / departure_date must be within 370 days"
            )
        noon = datetime.combine(value, datetime.min.time()).replace(
            hour=12,
            tzinfo=timezone,
        )
        if value > today:
            return noon, timezone_name, "origin_local_noon_model_reference"

        safe_minimum = PredictionService._same_day_safe_reference(
            generated_at,
            timezone,
        )
        if noon > safe_minimum:
            return noon, timezone_name, "origin_local_noon_model_reference"
        if safe_minimum.date() != today:
            raise RouteLookupError(
                "出发机场当地今天已没有安全的同日模型参考时间 / "
                "no safe same-day model reference remains"
            )
        return (
            safe_minimum,
            timezone_name,
            "origin_local_remaining_day_model_reference",
        )

    @staticmethod
    def _model_hub(airline: str, origin: str, destination: str) -> str | None:
        for hub in _AIRLINE_HUBS.get(airline, ()):
            if hub not in {origin, destination}:
                return hub
        return None

    @staticmethod
    def _offer_id(
        *,
        origin: str,
        destination: str,
        departure_date: date,
        airline: str,
        cabin: str,
        stops: int | None,
        fare_identity: str,
    ) -> str:
        canonical = "|".join(
            (
                origin,
                destination,
                departure_date.isoformat(),
                airline,
                cabin,
                str(stops),
                fare_identity,
            )
        )
        return f"off_{sha256(canonical.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _provider_local_time(value: datetime, airport: Airport) -> datetime:
        """Attach an airport timezone to a provider-local wall-clock time.

        Strict fare sources may return airport-local segment clocks without UTC
        offsets. Strict mode resolves them from the airport and rejects DST gaps or
        ambiguous folds instead of guessing an instant.
        """

        if value.tzinfo is not None or value.utcoffset() is not None:
            raise RouteLookupError("provider local time must not include an offset")
        timezone, _ = PredictionService._airport_timezone(airport)
        localized = value.replace(tzinfo=timezone, fold=0)
        round_trip = localized.astimezone(UTC).astimezone(timezone).replace(tzinfo=None)
        if round_trip != value:
            raise RouteLookupError("provider local time falls in a DST gap")
        alternate = value.replace(tzinfo=timezone, fold=1)
        if alternate.utcoffset() != localized.utcoffset():
            raise RouteLookupError("provider local time is ambiguous during a DST fold")
        return localized

    @staticmethod
    def _full_flight_number(segment: FlightOfferSegment) -> str:
        raw_number = segment.flight_number.upper()
        carrier = segment.marketing_airline_code.upper()
        value = raw_number if raw_number.startswith(carrier) else f"{carrier}{raw_number}"
        if not (3 <= len(value) <= 12) or not value.isalnum():
            raise RouteLookupError("provider flight number is invalid")
        return value

    def _strict_provider_segments(
        self,
        offer: ConfirmedFlightOffer,
        *,
        origin: str,
        destination: str,
        departure_date: date,
        generated_at: datetime,
    ) -> tuple[tuple[ProviderOfferSegment, ...], float, int]:
        if not 1 <= len(offer.segments) <= MAX_STRICT_ITINERARY_SEGMENTS:
            raise RouteLookupError("strict offers require one to eight segments")

        segments: list[ProviderOfferSegment] = []
        total_distance_km = 0.0
        for sequence, raw in enumerate(offer.segments, start=1):
            leg_route = self._route(raw.origin, raw.destination)
            departure_local = self._provider_local_time(raw.departure_at, leg_route.origin)
            arrival_local = self._provider_local_time(raw.arrival_at, leg_route.destination)
            departure_utc = departure_local.astimezone(UTC)
            arrival_utc = arrival_local.astimezone(UTC)
            if arrival_utc <= departure_utc:
                raise RouteLookupError("provider segment arrival is not after departure")
            duration_minutes = round((arrival_utc - departure_utc).total_seconds() / 60)
            if duration_minutes <= 0 or duration_minutes > 2_160:
                raise RouteLookupError("provider segment duration is outside strict limits")

            bag_quantity = raw.checked_bags_quantity
            bag_weight = raw.checked_bags_weight
            bag_unit = (
                raw.checked_bags_weight_unit.upper()
                if raw.checked_bags_weight_unit is not None
                else None
            )
            if bag_quantity is not None and bag_weight is not None:
                raise RouteLookupError("provider returned conflicting baggage units")
            if bag_weight is not None and bag_weight <= 0:
                bag_quantity, bag_weight, bag_unit = 0, None, None
            if bag_weight is not None and bag_unit not in {"KG", "LB"}:
                raise RouteLookupError("provider returned an unsupported baggage unit")

            aircraft = raw.aircraft_icao
            if aircraft is not None and len(aircraft) > 12:
                aircraft = None
            booking_class = raw.booking_class
            if booking_class is not None and len(booking_class) > 8:
                raise RouteLookupError("provider booking class is too long")
            segment = ProviderOfferSegment(
                sequence=sequence,
                origin=raw.origin,
                destination=raw.destination,
                flight_number=self._full_flight_number(raw),
                marketing_airline_code=raw.marketing_airline_code,
                operating_airline_code=raw.operating_airline_code,
                departure_local=departure_local,
                arrival_local=arrival_local,
                departure_utc=departure_utc,
                arrival_utc=arrival_utc,
                duration_minutes=duration_minutes,
                departure_terminal=raw.departure_terminal,
                arrival_terminal=raw.arrival_terminal,
                aircraft_icao=aircraft,
                cabin=raw.cabin,
                booking_class=booking_class,
                fare_basis=raw.fare_basis,
                fare_brand=raw.fare_brand,
                included_checked_bag_quantity=bag_quantity,
                included_checked_bag_weight=bag_weight,
                included_checked_bag_weight_unit=bag_unit,
            )
            if segments:
                previous = segments[-1]
                if previous.destination != segment.origin:
                    raise RouteLookupError("provider itinerary is not continuous")
                if segment.departure_utc <= previous.arrival_utc:
                    raise RouteLookupError("provider connection is not chronological")
            segments.append(segment)
            total_distance_km += leg_route.distance_km

        first = segments[0]
        last = segments[-1]
        if (
            first.origin != origin
            or last.destination != destination
            or first.departure_local.date() != departure_date
            or first.departure_utc <= generated_at
        ):
            raise RouteLookupError("provider itinerary does not match the future request")
        total_duration_minutes = round(
            (last.arrival_utc - first.departure_utc).total_seconds() / 60
        )
        if total_duration_minutes <= 0:
            raise RouteLookupError("provider itinerary duration is invalid")
        if offer.cabin != first.cabin or any(segment.cabin != offer.cabin for segment in segments):
            raise RouteLookupError("provider itinerary mixes cabin classes")
        return tuple(segments), round(total_distance_km, 1), total_duration_minutes

    @staticmethod
    def _baggage_status(segments: tuple[ProviderOfferSegment, ...]) -> str:
        allowances: list[bool | None] = []
        for segment in segments:
            if segment.included_checked_bag_quantity is not None:
                allowances.append(segment.included_checked_bag_quantity > 0)
            elif segment.included_checked_bag_weight is not None:
                allowances.append(segment.included_checked_bag_weight > 0)
            else:
                allowances.append(None)
        if any(value is False for value in allowances):
            return "not_included"
        if all(value is True for value in allowances):
            return "confirmed_included"
        return "unknown"

    @staticmethod
    def _change_status(offer: ConfirmedFlightOffer) -> str:
        if offer.no_restriction_fare is True or offer.no_penalty_fare is True:
            return "confirmed_free"
        if offer.no_penalty_fare is False:
            return "confirmed_paid"
        return "unknown"

    @staticmethod
    def _refund_status(offer: ConfirmedFlightOffer) -> str:
        if offer.refundable_fare is False:
            return "not_included"
        if offer.refundable_fare is True and (
            offer.no_restriction_fare is True or offer.no_penalty_fare is True
        ):
            return "confirmed_free"
        return "unknown"

    @staticmethod
    def _fare_metadata(result: FlightOfferSearchResult) -> FareSearchMetadata:
        provider_label = result.provider_name
        notices = {
            "confirmed_offers": BilingualText(
                zh=f"{provider_label} 返回的报价已通过完整行程、正价格与安全购票路径验证。",
                en=(
                    f"Offers returned by {provider_label} passed complete-itinerary, "
                    "positive-fare, and safe-booking-path verification."
                ),
            ),
            "no_results": BilingualText(
                zh="生产报价源没有返回通过严格验证的可售方案。",
                en="The production fare source returned no strictly verified offer.",
            ),
            "not_configured": BilingualText(
                zh="尚未配置当前严格报价源的凭据；严格模式不会用时刻表补造结果。",
                en=(
                    "The active strict fare provider is not configured; strict mode "
                    "does not substitute timetable projections."
                ),
            ),
            "test_environment_rejected": BilingualText(
                zh="测试或样例报价已被严格模式拒绝。",
                en=("Test or illustrative fare data is rejected by strict mode."),
            ),
            "authentication_failed": BilingualText(
                zh=f"{provider_label} 认证失败；未返回未验证航班。",
                en=(
                    f"{provider_label} authentication failed; no unverified flights "
                    "were returned."
                ),
            ),
            "rate_limited": BilingualText(
                zh="生产报价源当前限流；严格模式返回空结果。",
                en="The production fare source is rate-limited; strict mode returns no offers.",
            ),
            "budget_not_configured": BilingualText(
                zh="尚未设置适用的本地免费额度硬上限；为防止超额计费，本次未调用生产接口。",
                en=(
                    "The applicable local free-quota hard limit is not configured, so the "
                    "production API was not called to prevent overage charges."
                ),
            ),
            "budget_exhausted": BilingualText(
                zh="本地免费额度硬保护已触发；本次不再调用生产接口。",
                en=(
                    "The local free-quota hard guard is exhausted; no further "
                    "production calls were made."
                ),
            ),
            "provider_processing": BilingualText(
                zh=(
                    f"{provider_label} 报价任务仍在队列中或处理中；系统已完成有界轮询，"
                    "并最多进行一次额度受控的重提。"
                ),
                en=(
                    f"The fare search is still queued or processing at {provider_label} after "
                    "bounded polling and at most one quota-reserved controlled retry."
                ),
            ),
            "provider_error": BilingualText(
                zh=("生产报价源返回终态错误、HTTP 错误或网络错误；严格模式未显示未验证航班。"),
                en=(
                    "The production fare source returned a terminal, HTTP, or "
                    "transport error; strict mode showed no unverified flights."
                ),
            ),
            "provider_unavailable": BilingualText(
                zh="生产报价源暂不可用或返回了无效数据；严格模式未降级为虚构结果。",
                en=(
                    "The production fare source is unavailable or returned invalid "
                    "data; strict mode did not fall back to invented offers."
                ),
            ),
        }
        limits = (
            result.search_monthly_limit,
            result.pricing_monthly_limit,
        )
        usage = (
            result.search_monthly_used,
            result.pricing_monthly_used,
        )
        # Every strict provider exposes its single shared hard limit in exactly
        # one legacy quota slot. Never add the slots and double-count it.
        monthly_limit = next((value for value in limits if value is not None), None)
        monthly_used = next((value for value in usage if value is not None), None)
        if result.status == "not_configured":
            monthly_limit = None
            monthly_used = None
        notice = notices[result.status]
        if result.provider_runs:
            status_labels_zh = {
                "confirmed_offers": "已验证报价",
                "no_results": "确认无结果",
                "not_configured": "未配置",
                "test_environment_rejected": "测试环境已拒绝",
                "authentication_failed": "认证失败",
                "rate_limited": "暂时限流",
                "budget_not_configured": "免费额度保护未配置",
                "budget_exhausted": "免费额度已耗尽",
                "provider_processing": "暂时处理中",
                "provider_error": "供应商错误",
                "provider_unavailable": "供应商暂不可用",
            }
            status_labels_en = {
                "confirmed_offers": "verified offers",
                "no_results": "confirmed no results",
                "not_configured": "not configured",
                "test_environment_rejected": "test environment rejected",
                "authentication_failed": "authentication failed",
                "rate_limited": "temporarily rate limited",
                "budget_not_configured": "free-quota guard not configured",
                "budget_exhausted": "free quota exhausted",
                "provider_processing": "temporarily processing",
                "provider_error": "provider error",
                "provider_unavailable": "provider temporarily unavailable",
            }
            run_summary_zh = "；".join(
                f"{run.provider_name}：{status_labels_zh[run.status]}"
                for run in result.provider_runs
            )
            run_summary_en = "; ".join(
                f"{run.provider_name}: {status_labels_en[run.status]}"
                for run in result.provider_runs
            )
            transient_statuses = {
                "provider_processing",
                "provider_error",
                "provider_unavailable",
            }
            transient_recovery_zh = (
                " 暂时性失败来源已最多完成一次额度受控重试；"
                "后续新查询仍会自动切换可用来源。"
                if any(run.status in transient_statuses for run in result.provider_runs)
                else ""
            )
            transient_recovery_en = (
                " Transiently failing sources received at most one quota-controlled "
                "retry; a later new search will still fail over across eligible sources."
                if any(run.status in transient_statuses for run in result.provider_runs)
                else ""
            )
            credential_recovery_zh = (
                " 认证失败来源需要修复凭据，系统不会对其盲目重复请求。"
                if any(
                    run.status == "authentication_failed"
                    for run in result.provider_runs
                )
                else ""
            )
            credential_recovery_en = (
                " A source with failed authentication requires a credential repair and "
                "is not retried blindly."
                if any(
                    run.status == "authentication_failed"
                    for run in result.provider_runs
                )
                else ""
            )
            notice = BilingualText(
                zh=(
                    f"系统已执行 {len(result.provider_runs)} 个可用严格报价源，"
                    "并仅聚合各来源独立完成二次购票验证的结果。"
                    "同一完整航段与舱位只保留最低最终确认价；"
                    f"逐源状态：{run_summary_zh}。"
                    f"{transient_recovery_zh}{credential_recovery_zh}"
                ),
                en=(
                    f"The system ran {len(result.provider_runs)} available strict fare "
                    "sources and aggregated only offers that independently passed each "
                    "source's second-stage booking verification. Equivalent complete "
                    "itineraries and cabins retain only the lowest final confirmed fare. "
                    f"Per-source status: {run_summary_en}."
                    f"{transient_recovery_en}{credential_recovery_en}"
                ),
            )
        elif result.coverage_status in {"quota_limited", "quota_and_provider_incomplete"}:
            quota_scope_zh = {
                "hourly": "小时",
                "monthly": "月度",
                "lifetime": "账户终身",
                "provider_specific": "各供应商独立",
            }.get(result.quota_limit, "供应商")
            provider_failure_zh = (
                f"另有 {result.provider_failed_candidate_count} 个候选因供应商错误未完成验证；"
                if result.provider_failed_candidate_count
                else ""
            )
            provider_failure_en = (
                f" A further {result.provider_failed_candidate_count} candidate(s) failed "
                "because of provider errors."
                if result.provider_failed_candidate_count
                else ""
            )
            cabin_failure_zh = (
                f"另有 {result.search_failed_cabin_count} 个舱位搜索因供应商错误未完成；"
                if result.search_failed_cabin_count
                else ""
            )
            cabin_failure_en = (
                f" A further {result.search_failed_cabin_count} cabin search(es) failed "
                "because of provider errors."
                if result.search_failed_cabin_count
                else ""
            )
            notice = BilingualText(
                zh=(
                    f"{provider_label} 四舱搜索返回 "
                    f"{result.eligible_candidate_count} 个可验证候选；"
                    f"本次受免费{quota_scope_zh}额度限制，"
                    f"仅尝试 {result.verification_attempted_count} 个，跳过 "
                    f"{result.quota_skipped_candidate_count} 个。{provider_failure_zh}"
                    f"{cabin_failure_zh}"
                    "列表只包含成功严格验证的航班，"
                    "最低价仅指已验证子集。"
                ),
                en=(
                    f"{provider_label}'s four-cabin search returned "
                    f"{result.eligible_candidate_count} "
                    f"verifiable candidate(s). The free "
                    f"{result.quota_limit or 'provider'} quota allowed "
                    f"{result.verification_attempted_count} attempt(s), leaving "
                    f"{result.quota_skipped_candidate_count} unverified."
                    f"{provider_failure_en}{cabin_failure_en} Only strictly "
                    "verified flights are listed; the lowest price applies to the verified "
                    "subset only."
                ),
            )
        elif result.coverage_status == "provider_incomplete":
            candidate_failure_zh = (
                f"其中 {result.provider_failed_candidate_count} 个候选因供应商错误"
                "未能完成验证。"
                if result.provider_failed_candidate_count
                else ""
            )
            candidate_failure_en = (
                f" {result.provider_failed_candidate_count} candidate(s) could not be "
                "completed because of provider errors."
                if result.provider_failed_candidate_count
                else ""
            )
            cabin_failure_zh = (
                f"另有 {result.search_failed_cabin_count} 个舱位搜索因供应商错误未完成。"
                if result.search_failed_cabin_count
                else ""
            )
            cabin_failure_en = (
                f" {result.search_failed_cabin_count} cabin search(es) could not be "
                "completed because of provider errors."
                if result.search_failed_cabin_count
                else ""
            )
            notice = BilingualText(
                zh=(
                    f"已尝试 {provider_label} 成功舱位返回的全部 "
                    f"{result.eligible_candidate_count} 个候选。"
                    f"{candidate_failure_zh}{cabin_failure_zh}"
                    "列表只包含成功严格验证的航班。"
                ),
                en=(
                    f"All {result.eligible_candidate_count} candidate(s) returned by "
                    f"successful {provider_label} cabin searches were attempted."
                    f"{candidate_failure_en}{cabin_failure_en} Only strictly verified "
                    "flights are listed."
                ),
            )
        if result.cache_hit and result.coverage_status != "not_evaluated":
            notice = BilingualText(
                zh="本次复用 5 分钟严格缓存；以下候选统计来自原查询。" + notice.zh,
                en=(
                    "This response reused the five-minute strict cache; the candidate "
                    "coverage counts describe the original search. " + notice.en
                ),
            )
        return FareSearchMetadata(
            status=result.status,
            provider_code=(
                "none" if result.status == "not_configured" else result.provider_code
            ),
            provider_name=(
                "No strict fare provider"
                if result.status == "not_configured"
                else result.provider_name
            ),
            provider_runs=[
                PredictionService._fare_metadata(provider_run)
                for provider_run in result.provider_runs
            ],
            environment=(
                "disabled"
                if result.status == "not_configured"
                else (
                    result.environment
                    if result.environment in {"production", "test", "disabled"}
                    else "disabled"
                )
            ),
            observed_at=result.observed_at,
            searched_cabins=list(result.searched_cabins),
            call_count=result.calls_used,
            search_call_count=result.search_calls_used,
            pricing_call_count=result.pricing_calls_used,
            cache_hit=result.cache_hit,
            monthly_call_limit=monthly_limit,
            monthly_calls_used=monthly_used,
            quota_unit=(
                None
                if monthly_limit is None
                else (
                    "lifetime_requests"
                    if result.provider_code
                    in {
                        "searchapi_google_flights",
                        "ignav_quarantine",
                        "ignav_verified_fares",
                    }
                    else "billing_period_requests"
                )
            ),
            search_monthly_limit=(
                None if result.status == "not_configured" else result.search_monthly_limit
            ),
            search_monthly_used=(
                None if result.status == "not_configured" else result.search_monthly_used
            ),
            pricing_monthly_limit=(
                None if result.status == "not_configured" else result.pricing_monthly_limit
            ),
            pricing_monthly_used=(
                None if result.status == "not_configured" else result.pricing_monthly_used
            ),
            archive_poll_count=result.archive_poll_count,
            diagnostics=[
                {
                    "observed_at": diagnostic.observed_at,
                    "stage": diagnostic.stage,
                    "http_status": diagnostic.http_status,
                    "exception_type": diagnostic.exception_type,
                    "search_id": diagnostic.search_id,
                }
                for diagnostic in result.diagnostics
            ],
            coverage_scope=result.coverage_scope,
            eligible_candidate_count=result.eligible_candidate_count,
            verification_attempted_count=result.verification_attempted_count,
            verified_candidate_count=result.verified_candidate_count,
            strictly_rejected_candidate_count=result.strictly_rejected_candidate_count,
            provider_failed_candidate_count=result.provider_failed_candidate_count,
            search_failed_cabin_count=result.search_failed_cabin_count,
            quota_skipped_candidate_count=result.quota_skipped_candidate_count,
            deduplicated_verified_count=result.deduplicated_verified_count,
            coverage_status=result.coverage_status,
            quota_limit=result.quota_limit,
            retry_quota_limited=result.retry_quota_limited,
            notice=notice,
        )

    @staticmethod
    def _historical_market_context(
        history: RouteCabinMarketHistory,
    ) -> HistoricalMarketContext:
        """Expose sanitized route/date/cabin history without offer attribution."""

        return HistoricalMarketContext(
            provider_code=history.provider_code,
            provider_name=history.provider_name,
            scope=history.scope,
            origin=history.origin,
            destination=history.destination,
            departure_date=history.departure_date,
            cabin=history.cabin,
            currency=history.currency,
            provider_observed_at=history.provider_observed_at,
            points=[
                HistoricalMarketPricePoint(
                    observed_at=point.observed_at,
                    price_usd=point.price_usd,
                )
                for point in history.points
            ],
            notice=BilingualText(
                zh=(
                    "这些价格是 SerpApi Google Flights 对相同航线、出发日期和舱位"
                    "查询返回的市场历史；它们不是所选航班、销售方或当前可购买报价"
                    "自身的历史。"
                ),
                en=(
                    "These prices are query-level market history returned by SerpApi "
                    "Google Flights for the same route, departure date, and cabin. "
                    "They are not price history for the selected flight, seller, or "
                    "currently bookable offer."
                ),
            ),
        )

    @staticmethod
    def _historical_market_contexts(
        result: FlightOfferSearchResult,
    ) -> list[HistoricalMarketContext]:
        """Retain SerpApi histories from attributed runs in an aggregate result."""

        source_results = result.provider_runs or (result,)
        histories_by_key: dict[
            tuple[str, str, str, date, str],
            RouteCabinMarketHistory,
        ] = {}
        for source_result in source_results:
            for history in source_result.historical_market_contexts:
                key = (
                    history.provider_code,
                    history.origin,
                    history.destination,
                    history.departure_date,
                    history.cabin,
                )
                histories_by_key[key] = history
        cabin_order = {
            "economy": 0,
            "premium_economy": 1,
            "business": 2,
            "first": 3,
        }
        return [
            PredictionService._historical_market_context(history)
            for history in sorted(
                histories_by_key.values(),
                key=lambda item: (
                    cabin_order[item.cabin],
                    item.provider_code,
                ),
            )
        ]

    @staticmethod
    def _fare_reference_snapshot(
        result: ScrapeDoReferenceResult,
    ) -> FareCoverageReference:
        notices = {
            "available": BilingualText(
                zh=(
                    "Scrape.do 仅返回一次经济舱列表覆盖快照。候选数量和最低参考价未经购票展开验证，"
                    "不能证明库存、最终价格或可购买性，也永远不会进入严格航班列表。"
                ),
                en=(
                    "Scrape.do supplied one economy listing-coverage snapshot only. "
                    "Candidate counts and the lowest reference price lack booking expansion, "
                    "do not prove inventory, final price, or bookability, and can never enter "
                    "the strict flight list."
                ),
            ),
            "no_results": BilingualText(
                zh=(
                    "Scrape.do 的单次经济舱参考查询成功完成，但没有返回通过基础结构检查的列表候选；"
                    "这不证明该航线没有可购买机票。"
                ),
                en=(
                    "The single Scrape.do economy reference query completed but returned no "
                    "listing candidates that passed basic structural checks; this does not "
                    "prove that no bookable ticket exists."
                ),
            ),
            "not_configured": BilingualText(
                zh="未配置 Scrape.do 免费令牌，因此没有执行候选覆盖参考查询。",
                en=(
                    "No Scrape.do free-tier token is configured, so no candidate-coverage "
                    "reference query was made."
                ),
            ),
            "quota_exhausted": BilingualText(
                zh=(
                    "Scrape.do 本地免费额度硬上限或提供方免费余额已耗尽；"
                    "系统已在下一次航班请求前停止，未使用付费额度。"
                ),
                en=(
                    "The local Scrape.do free-credit hard stop or provider-reported free "
                    "balance is exhausted; the system stopped before another flight request "
                    "and did not use paid quota."
                ),
            ),
            "authentication_failed": BilingualText(
                zh="Scrape.do 拒绝了凭据；错误信息已脱敏，未显示或保存令牌。",
                en=(
                    "Scrape.do rejected the credential; the error is sanitized and the token "
                    "is neither displayed nor retained in diagnostics."
                ),
            ),
            "rate_limited": BilingualText(
                zh=(
                    "Scrape.do 暂时限流；系统最多在重新确认免费余额后受控重试一次，"
                    "且不会用参考数据补造可购航班。"
                ),
                en=(
                    "Scrape.do temporarily rate-limited the request; the system may retry "
                    "once only after re-confirming free quota, and this reference can never "
                    "fabricate a bookable flight."
                ),
            ),
            "provider_error": BilingualText(
                zh=(
                    "Scrape.do 返回 HTTP 或传输错误；仅保留脱敏状态与固定异常类型，未保留原始响应。"
                ),
                en=(
                    "Scrape.do returned an HTTP or transport error; only a sanitized status "
                    "and fixed exception type are retained, not the raw response."
                ),
            ),
            "provider_unavailable": BilingualText(
                zh=(
                    "Scrape.do 响应缺失或未通过结构校验；没有候选详情、令牌或价格被采用。"
                ),
                en=(
                    "The Scrape.do response was missing or failed structural validation; no "
                    "candidate detail, token, or price was accepted."
                ),
            ),
        }
        return FareCoverageReference(
            provider_code=SCRAPE_DO_PROVIDER_CODE,
            provider_name=SCRAPE_DO_PROVIDER_NAME,
            status=result.status,
            observed_at=result.observed_at,
            candidate_count=result.candidate_count,
            direct_candidate_count=result.direct_candidate_count,
            lowest_price_usd=result.lowest_price_usd,
            price_level=result.price_level,
            typical_price_low_usd=result.typical_price_low_usd,
            typical_price_high_usd=result.typical_price_high_usd,
            cache_hit=result.cache_hit,
            credits_reserved=result.credits_reserved,
            monthly_credits_used=result.monthly_credits_used,
            monthly_credit_limit=result.monthly_credit_limit,
            http_status=result.http_status,
            exception_type=result.exception_type,
            provider_reported_request_cost=result.provider_reported_request_cost,
            provider_reported_remaining_credits=(
                result.provider_reported_remaining_credits
            ),
            notice=notices[result.status],
        )

    @staticmethod
    def _fallback_reason(code: str | None) -> BilingualText | None:
        reasons = {
            "airlabs_api_key_not_configured": BilingualText(
                zh="未配置免费的 AirLabs API Key；未显示航班号或精确时刻。",
                en=(
                    "A free AirLabs API key is not configured; no flight number or "
                    "exact clock time is shown."
                ),
            ),
            "external_context_disabled": BilingualText(
                zh="外部数据已关闭；当前选项仅为模型场景。",
                en="External data is disabled; this option is a model scenario only.",
            ),
            "airlabs_schedules_unavailable": BilingualText(
                zh="AirLabs 实时时刻当前不可用；未编造航班号或精确时刻。",
                en=(
                    "AirLabs live schedules are unavailable; no flight number or exact "
                    "clock time was invented."
                ),
            ),
            "airlabs_routes_unavailable": BilingualText(
                zh="AirLabs 航线时刻数据库当前不可用；未编造航班号或精确时刻。",
                en=(
                    "The AirLabs routes timetable is unavailable; no flight number or "
                    "exact clock time was invented."
                ),
            ),
            "no_complete_provider_schedule": BilingualText(
                zh="免费提供商没有返回该日期可验证的完整时刻；未编造航班号或精确钟点。",
                en=(
                    "The free provider returned no verifiable complete timetable for "
                    "this date; no flight number or exact clock time was invented."
                ),
            ),
        }
        return reasons.get(code)

    def _price_values(
        self,
        *,
        origin: str,
        destination: str,
        airline: str,
        cabin: str,
        stops: int,
        quote_time: datetime,
        departure_time: datetime,
        route: RouteEstimate,
        news_disruption_index: float,
    ) -> tuple[float, float, float, float]:
        row = pd.DataFrame(
            [
                {
                    "origin": origin,
                    "destination": destination,
                    "airline": airline,
                    "cabin": cabin,
                    "stops": stops,
                    "quote_time": quote_time,
                    "departure_time": departure_time,
                    "distance_km": route.distance_km,
                    "duration_minutes": route.duration_minutes,
                    "news_disruption_index": news_disruption_index,
                    **_local_time_features(departure_time),
                }
            ]
        )
        features = build_price_features(row)
        estimate = max(0.0, float(np.expm1(self.bundle["price_model"].predict(features)[0])))
        half_width = float(self.bundle["price_interval_half_width_usd"])
        lead_days = (departure_time - quote_time).total_seconds() / 86_400.0
        return (
            round(estimate, 2),
            round(max(0.0, estimate - half_width), 2),
            round(estimate + half_width, 2),
            round(lead_days, 1),
        )

    def _ontime_value(
        self,
        *,
        origin: str,
        destination: str,
        airline: str,
        departure_time: datetime,
        route: RouteEstimate,
        context: PredictionContext,
    ) -> float:
        row_data = {
            "origin": origin,
            "destination": destination,
            "airline": airline,
            "scheduled_departure": departure_time,
            "distance_km": route.distance_km,
            "origin_congestion_index": context.operations.value,
            "news_disruption_index": context.news.value,
            **_local_time_features(departure_time),
        }
        weather_feature_status = self._weather_feature_status(context)
        if weather_feature_status == "used":
            row_data["weather_severity_forecast"] = context.weather.value
            features = build_ontime_features(pd.DataFrame([row_data]))
            model = self.bundle["ontime_model"]
        else:
            features = build_ontime_features_without_weather(pd.DataFrame([row_data]))
            model = self.bundle["ontime_model_without_weather"]
        probability = float(model.predict_proba(features)[0, 1])
        return round(float(np.clip(probability, 0.0, 1.0)), 4)

    @staticmethod
    def _weather_feature_status(
        context: PredictionContext,
        *,
        target_departure: datetime | None = None,
        weather_reference_departure: datetime | None = None,
    ) -> str:
        if context.weather.status not in {"live", "forecast"}:
            return "ignored"
        if target_departure is None and weather_reference_departure is None:
            return "used"
        if target_departure is None or weather_reference_departure is None:
            return "ignored"
        if (
            target_departure.tzinfo is None
            or target_departure.utcoffset() is None
            or weather_reference_departure.tzinfo is None
            or weather_reference_departure.utcoffset() is None
        ):
            return "ignored"
        separation = abs(
            (
                target_departure.astimezone(UTC) - weather_reference_departure.astimezone(UTC)
            ).total_seconds()
        )
        return "used" if separation <= 2 * 60 * 60 else "ignored"

    @staticmethod
    def _weather_feature_notice(status: str) -> tuple[str, str]:
        if status == "used":
            return (
                "本次准点预测使用了适用于出发时刻的实时或预报天气。",
                "This on-time prediction uses applicable live or forecast weather.",
            )
        return (
            "天气不可用、超出预报范围或仅有历史代理值；本次准点预测已忽略天气变量。",
            "Weather is unavailable, outside the forecast horizon, or proxy-only; "
            "this on-time prediction excludes the weather feature.",
        )

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.80:
            return "low"
        if probability >= 0.60:
            return "medium"
        return "high"

    def predict_price(self, request: PriceRequest) -> PricePrediction:
        quote_time = datetime.now(UTC)
        route = self._route(request.origin, request.destination, request.stops)
        departure_time, _ = self._departure_at_origin(request.departure_time, route.origin)
        context = self._context(route, departure_time)
        estimate, low, high, lead_days = self._price_values(
            origin=request.origin,
            destination=request.destination,
            airline=request.airline,
            cabin=request.cabin,
            stops=request.stops,
            quote_time=quote_time,
            departure_time=departure_time,
            route=route,
            news_disruption_index=context.news.value,
        )
        return PricePrediction(
            estimated_price_usd=estimate,
            interval_80_low_usd=low,
            interval_80_high_usd=high,
            days_until_departure=lead_days,
            distance_km=route.distance_km,
            duration_minutes=route.duration_minutes,
            model_version=self.model_version,
            warning=("模型估价已纳入带来源的新闻风险信号，但并非实时可购买报价，也不保证最低价。"),
            warning_en=(
                "The model estimate includes a sourced news-risk signal, but it is not a "
                "live bookable fare or a lowest-price guarantee."
            ),
        )

    def predict_ontime(self, request: OnTimeRequest) -> OnTimePrediction:
        route = self._route(request.origin, request.destination)
        departure_time, _ = self._departure_at_origin(
            request.scheduled_departure,
            route.origin,
        )
        context = self._context(route, departure_time)
        probability = self._ontime_value(
            origin=request.origin,
            destination=request.destination,
            airline=request.airline,
            departure_time=departure_time,
            route=route,
            context=context,
        )
        weather_feature_status = self._weather_feature_status(context)
        weather_notice_zh, weather_notice_en = self._weather_feature_notice(weather_feature_status)
        return OnTimePrediction(
            on_time_probability=probability,
            disruption_probability=round(1.0 - probability, 4),
            distance_km=route.distance_km,
            risk_level=self._risk_level(probability),
            definition="未取消且到达延误少于 15 分钟",
            definition_en="Not cancelled and arrival delay under 15 minutes",
            model_version=self.model_version,
            weather_feature_status=weather_feature_status,
            weather_feature_notice_zh=weather_notice_zh,
            weather_feature_notice_en=weather_notice_en,
        )

    @staticmethod
    def _context_response(context: PredictionContext) -> PredictionContextResponse:
        def signal(value: Any) -> ContextSignal:
            return ContextSignal(
                value=value.value,
                status=value.status,
                source=value.source,
                observed_at=value.observed_at,
                summary_zh=value.summary_zh,
                summary_en=value.summary_en,
            )

        def operations_snapshot(value: Any) -> OperationsSnapshot:
            return OperationsSnapshot(
                value=value.value,
                status=value.status,
                source=value.source,
                observed_at=value.observed_at,
                summary_zh=value.summary_zh,
                summary_en=value.summary_en,
                method=value.method,
                data_tier=value.data_tier,
                applicability=value.applicability,
                metrics=[
                    OperationsMetric(key=metric.key, value=metric.value, unit=metric.unit)
                    for metric in value.metrics
                ],
                events=[
                    OperationsEvent(
                        event_type=event.event_type,
                        severity=event.severity,
                        reason=event.reason,
                        start_at=event.start_at,
                        end_at=event.end_at,
                        scope=event.scope,
                    )
                    for event in value.events
                ],
                fallback_reason=value.fallback_reason,
                window_start=value.window_start,
                window_end=value.window_end,
                sample_size=value.sample_size,
                sample_limit=value.sample_limit,
                sample_truncated=value.sample_truncated,
            )

        articles = [
            NewsArticle(
                title=article.title,
                url=article.url,
                source=article.source,
                published_at=(
                    article.published_at.isoformat() if article.published_at is not None else None
                ),
                language=article.language,
            )
            for article in context.news.articles
        ]
        weather_feature_status = PredictionService._weather_feature_status(context)
        weather_notice_zh, weather_notice_en = PredictionService._weather_feature_notice(
            weather_feature_status
        )
        return PredictionContextResponse(
            weather=signal(context.weather),
            operations=OperationsSignal(
                **operations_snapshot(context.operations).model_dump(),
                current_snapshot=(
                    operations_snapshot(context.operations.current_snapshot)
                    if context.operations.current_snapshot is not None
                    else None
                ),
            ),
            news=NewsSignal(
                value=context.news.value,
                status=context.news.status,
                source=context.news.source,
                observed_at=context.news.observed_at,
                summary_zh=context.news.summary_zh,
                summary_en=context.news.summary_en,
                articles=articles,
            ),
            weather_feature_status=weather_feature_status,
            weather_feature_notice_zh=weather_notice_zh,
            weather_feature_notice_en=weather_notice_en,
        )

    @staticmethod
    def _student_sort_key(offer: ComparisonOffer) -> tuple[Any, ...]:
        baggage_rank = int(offer.baggage_status not in {"confirmed_free", "confirmed_included"})

        student_discount_rank = int(offer.student_status != "confirmed_discount")
        flexibility_rank = sum(
            status != "confirmed_free" for status in (offer.change_status, offer.refund_status)
        )
        profile = get_airline_profile(offer.airline_code)
        program = profile.student_program if profile is not None else None
        if program is None:
            age_rank = (99, 0)
            verification_rank = 99
        else:
            # Lower minimums and higher (or unpublished) maximums mean broader
            # eligibility. Localized policy text remains the user-facing truth.
            age_rank = (program.minimum_age, -(program.maximum_age or 999))
            verification_rank = program.verification_steps
        return (
            (
                offer.live_fare.total_amount
                if offer.live_fare is not None
                else offer.estimated_price_usd
            ),
            baggage_rank,
            student_discount_rank,
            flexibility_rank,
            *age_rank,
            verification_rank,
            offer.airline_code,
            offer.cabin,
        )

    @staticmethod
    def _routing_rank(offer: ComparisonOffer) -> int:
        if offer.routing_status == "provider_direct":
            return 0
        if offer.routing_status == "provider_itinerary":
            return offer.stops or 1
        return {"model_one_stop": 4, "model_route_unresolved": 5}[offer.routing_status]

    def compare(
        self,
        request: ComparisonRequest,
        *,
        force_fare_refresh: bool = False,
    ) -> ComparisonResponse:
        generated_at = self._now_provider()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise RouteLookupError("service clock must include a timezone offset")
        generated_at = generated_at.astimezone(UTC)
        route = self._route(request.origin, request.destination)
        if request.departure_time is not None:
            departure_time, departure_timezone = self._departure_at_origin(
                request.departure_time,
                route.origin,
            )
            departure_date = departure_time.date()
            departure_time_basis = "legacy_input"
        else:
            departure_date = request.departure_date
            (
                departure_time,
                departure_timezone,
                departure_time_basis,
            ) = self._departure_date_at_origin(
                departure_date,
                route.origin,
                generated_at,
            )
        destination_zone, _ = self._airport_timezone(route.destination)
        origin_zone, _ = self._airport_timezone(route.origin)

        # These sources are independent. Fetch them together so strict fare
        # verification does not unnecessarily serialize weather/news/timetable I/O.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="comparison") as pool:
            schedule_future = pool.submit(
                self.schedule_provider.search,
                request.origin,
                request.destination,
                departure_date,
                origin_timezone=origin_zone,
                destination_timezone=destination_zone,
                fetched_at=generated_at,
            )
            context_future = pool.submit(self._context, route, departure_time)
            fare_future = pool.submit(
                self.flight_offer_provider.search,
                request.origin,
                request.destination,
                departure_date,
                fetched_at=generated_at,
                force_refresh=force_fare_refresh,
            )
            schedule_result = schedule_future.result()
            context = context_future.result()
            fare_result = fare_future.result()

        historical_market_contexts = self._historical_market_contexts(fare_result)
        timetable_references: list[TimetableReference] = []
        for schedule in schedule_result.schedules:
            profile = get_airline_profile(schedule.airline_code)
            if schedule.schedule_status == "recurring_timetable_projection":
                reason = BilingualText(
                    zh=("周期时刻表只作为参考，不证明所选日期实际运行、存在座位或可购买。"),
                    en=(
                        "This recurring timetable is reference-only and does not prove "
                        "operation, seat inventory, or bookability on the selected date."
                    ),
                )
            elif schedule.schedule_status == "future_schedule_reference":
                reason = BilingualText(
                    zh=(
                        "AeroDataBox 返回了所选日期的计划航班号与完整时刻，但该来源不证明"
                        "座位库存、实时价格或可购买性，因此只显示在参考区。"
                    ),
                    en=(
                        "AeroDataBox returned a flight number and complete scheduled times "
                        "for the selected date, but it does not prove seat inventory, a live "
                        "fare, or bookability, so this row remains reference-only."
                    ),
                )
            else:
                reason = BilingualText(
                    zh=(
                        "日期级时刻只证明提供商返回了航班计划；没有由已启用的严格报价来源及其"
                        "二次购票验证标识确认价格和可购路径，因此不进入严格可售列表。"
                    ),
                    en=(
                        "This dated timetable only shows a provider schedule. Without "
                        "confirmation from an enabled strict fare source and its secondary "
                        "booking-verification identifier, it cannot enter the strictly "
                        "bookable list."
                    ),
                )
            timetable_references.append(
                TimetableReference(
                    airline_code=schedule.airline_code,
                    airline_name=(profile.name if profile is not None else schedule.airline_code),
                    flight_number=schedule.flight_number,
                    duration_minutes=schedule.duration_minutes,
                    schedule_status=schedule.schedule_status,
                    schedule_source=schedule.source,
                    scheduled_departure_local=schedule.departure_local,
                    scheduled_arrival_local=schedule.arrival_local,
                    scheduled_departure_utc=schedule.departure_utc,
                    scheduled_arrival_utc=schedule.arrival_utc,
                    schedule_observed_at=schedule.observed_at,
                    provider_flight_status=schedule.provider_flight_status,
                    reference_reason=reason,
                )
            )

        scenarios: list[_OfferScenario] = []
        rejected_priced_offers = 0
        for confirmed in fare_result.offers:
            try:
                provider_segments, total_distance, total_duration = self._strict_provider_segments(
                    confirmed,
                    origin=request.origin,
                    destination=request.destination,
                    departure_date=departure_date,
                    generated_at=generated_at,
                )
            except (RouteLookupError, ValueError):
                rejected_priced_offers += 1
                continue
            profile = get_airline_profile(confirmed.validating_airline_code)
            if profile is None:
                profile = _GenericAirlineProfile(
                    code=confirmed.validating_airline_code,
                    name=confirmed.airline_name,
                    supported_cabins=(confirmed.cabin,),
                )
            scenarios.append(
                _OfferScenario(
                    profile=profile,
                    cabin=confirmed.cabin,
                    route_status="provider_confirmed",
                    stops=len(provider_segments) - 1,
                    model_stops=len(provider_segments) - 1,
                    routing_status=(
                        "provider_direct" if len(provider_segments) == 1 else "provider_itinerary"
                    ),
                    departure_time=provider_segments[0].departure_local,
                    duration_minutes=total_duration,
                    distance_km=total_distance,
                    confirmed_offer=confirmed,
                    provider_segments=provider_segments,
                )
            )

        fare_reference_snapshots: list[FareCoverageReference] = []
        reference_has_credential = bool(
            getattr(self.fare_reference_provider, "credential_present", False)
            or getattr(self.fare_reference_provider, "configured", False)
        )
        if (
            not scenarios
            and fare_result.status != "provider_processing"
            and reference_has_credential
        ):
            try:
                reference_result = self.fare_reference_provider.snapshot(
                    request.origin,
                    request.destination,
                    departure_date,
                    fetched_at=generated_at,
                )
            except Exception:
                # A reference-only source must never break or weaken the strict
                # comparison.  The fixed error type intentionally discards raw
                # exception text, URLs, request parameters, and credentials.
                reference_result = ScrapeDoReferenceResult(
                    status="provider_unavailable",
                    observed_at=generated_at,
                    monthly_credit_limit=min(
                        max(
                            int(
                                getattr(
                                    self.fare_reference_provider,
                                    "monthly_credit_limit",
                                    0,
                                )
                                or 0
                            ),
                            0,
                        ),
                        1_000,
                    ),
                    exception_type="ReferenceBoundaryError",
                )
            fare_reference_snapshots.append(
                self._fare_reference_snapshot(reference_result)
            )

        scenario_rows = [
            {
                "origin": request.origin,
                "destination": request.destination,
                "airline": scenario.profile.code,
                "cabin": scenario.cabin,
                "stops": scenario.model_stops,
                "quote_time": generated_at,
                "departure_time": scenario.departure_time,
                "distance_km": scenario.distance_km,
                "duration_minutes": scenario.duration_minutes,
                "news_disruption_index": context.news.value,
                **_local_time_features(scenario.departure_time),
            }
            for scenario in scenarios
        ]

        if scenario_rows:
            price_features = build_price_features(pd.DataFrame(scenario_rows))
            estimates = np.maximum(
                0.0,
                np.expm1(self.bundle["price_model"].predict(price_features)),
            )
        else:
            estimates = np.asarray([], dtype=float)
        half_width = float(self.bundle["price_interval_half_width_usd"])

        weather_feature_statuses = [
            self._weather_feature_status(
                context,
                target_departure=scenario.departure_time,
                weather_reference_departure=departure_time,
            )
            for scenario in scenarios
        ]
        ontime_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            ontime_rows.append(
                {
                    "origin": request.origin,
                    "destination": request.destination,
                    "airline": scenario.profile.code,
                    "scheduled_departure": scenario.departure_time,
                    "distance_km": scenario.distance_km,
                    "origin_congestion_index": context.operations.value,
                    "news_disruption_index": context.news.value,
                    **_local_time_features(scenario.departure_time),
                }
            )
        probabilities = np.zeros(len(ontime_rows), dtype=float)
        for weather_feature_status in ("used", "ignored"):
            indices = [
                index
                for index, status in enumerate(weather_feature_statuses)
                if status == weather_feature_status
            ]
            if not indices:
                continue
            selected_rows = [dict(ontime_rows[index]) for index in indices]
            if weather_feature_status == "used":
                for row in selected_rows:
                    row["weather_severity_forecast"] = context.weather.value
                ontime_features = build_ontime_features(pd.DataFrame(selected_rows))
                ontime_model = self.bundle["ontime_model"]
            else:
                ontime_features = build_ontime_features_without_weather(pd.DataFrame(selected_rows))
                ontime_model = self.bundle["ontime_model_without_weather"]
            probabilities[indices] = np.clip(
                ontime_model.predict_proba(ontime_features)[:, 1],
                0.0,
                1.0,
            )
        offers: list[ComparisonOffer] = []
        for scenario, raw_estimate, raw_probability, weather_feature_status in zip(
            scenarios,
            estimates,
            probabilities,
            weather_feature_statuses,
            strict=True,
        ):
            profile = scenario.profile
            estimate = round(float(raw_estimate), 2)
            probability = round(float(raw_probability) ** len(scenario.provider_segments), 4)
            confirmed = scenario.confirmed_offer
            first_segment = scenario.provider_segments[0]
            last_segment = scenario.provider_segments[-1]
            seats = confirmed.number_of_bookable_seats
            live_fare = LiveFare(
                provider_code=confirmed.provider_code,
                provider_name=confirmed.provider_name,
                provider_offer_id=confirmed.provider_offer_id,
                verified_at=confirmed.verified_at,
                total_amount=confirmed.total_amount_usd,
                cabin_summary=confirmed.cabin,
                provider_cache_hit=confirmed.provider_cache_hit,
                provider_cache_age_seconds=confirmed.provider_cache_age_seconds,
                booking_verified=confirmed.booking_verified,
                booking_provider=confirmed.booking_provider,
                booking_url=confirmed.booking_url,
                booking_url_kind=confirmed.booking_url_kind,
                seats_remaining=(min(seats, 9) if seats is not None else None),
                seat_count_capped=bool(seats is not None and seats >= 9),
                last_ticketing_date=confirmed.last_ticketing_date,
                source_url=_FARE_PROVIDER_SOURCE_URLS[confirmed.provider_code],
            )
            offers.append(
                ComparisonOffer(
                    id=self._offer_id(
                        origin=request.origin,
                        destination=request.destination,
                        departure_date=departure_date,
                        airline=profile.code,
                        cabin=scenario.cabin,
                        stops=scenario.stops,
                        fare_identity=confirmed.fingerprint,
                    ),
                    airline_code=profile.code,
                    airline_name=profile.name,
                    cabin=scenario.cabin,
                    stops=scenario.stops,
                    duration_minutes=scenario.duration_minutes,
                    estimated_price_usd=estimate,
                    interval_80_low_usd=round(max(0.0, raw_estimate - half_width), 2),
                    interval_80_high_usd=round(float(raw_estimate + half_width), 2),
                    on_time_probability=probability,
                    risk_level=self._risk_level(probability),
                    weather_feature_status=weather_feature_status,
                    baggage_status=self._baggage_status(scenario.provider_segments),
                    student_status=profile.student_status,
                    change_status=self._change_status(confirmed),
                    refund_status=self._refund_status(confirmed),
                    student_age_limit_zh=profile.student_age_limit_zh,
                    student_age_limit_en=profile.student_age_limit_en,
                    student_verification_zh=profile.student_verification_zh,
                    student_verification_en=profile.student_verification_en,
                    student_program_url=profile.student_program_url,
                    route_status=scenario.route_status,
                    routing_status=scenario.routing_status,
                    cabin_status="provider_confirmed",
                    punctuality_basis=(
                        "direct_leg_model"
                        if scenario.routing_status == "provider_direct"
                        else "multi_leg_independence_model"
                    ),
                    schedule_status="priced_offer",
                    schedule_source=_FARE_PROVIDER_SCHEDULE_SOURCES[
                        confirmed.provider_code
                    ],
                    flight_number=first_segment.flight_number,
                    scheduled_departure_local=first_segment.departure_local,
                    scheduled_arrival_local=last_segment.arrival_local,
                    scheduled_departure_utc=first_segment.departure_utc,
                    scheduled_arrival_utc=last_segment.arrival_utc,
                    provider_flight_status="booking_option_verified",
                    schedule_observed_at=confirmed.verified_at,
                    departure_terminal=first_segment.departure_terminal,
                    arrival_terminal=last_segment.arrival_terminal,
                    aircraft_icao=(
                        first_segment.aircraft_icao
                        if scenario.routing_status == "provider_direct"
                        else None
                    ),
                    bookability_status="booking_option_verified",
                    live_fare=live_fare,
                    segments=list(scenario.provider_segments),
                )
            )

        direct = sorted(
            offers,
            key=lambda offer: (
                self._routing_rank(offer),
                offer.live_fare.total_amount,
                -offer.on_time_probability,
                offer.airline_code,
                offer.cabin,
            ),
        )
        cheapest = sorted(
            offers,
            key=lambda offer: (
                offer.live_fare.total_amount,
                self._routing_rank(offer),
                -offer.on_time_probability,
                offer.airline_code,
                offer.cabin,
            ),
        )
        student = sorted(offers, key=self._student_sort_key)

        reference_warning = {
            "origin_local_noon_model_reference": (
                "出发时间是所选日期的出发机场当地正午，仅用于模型和上下文参考，不是航班钟点。",
                "Departure time is origin-local noon on the selected date for model and "
                "context reference only; it is not a flight time.",
            ),
            "origin_local_remaining_day_model_reference": (
                "出发时间是生成时刻后 30 分钟的同日参考，仅用于模型和上下文，不是航班钟点。",
                "Departure time is a same-day reference 30 minutes after generation for "
                "model and context use only; it is not a flight time.",
            ),
        }.get(departure_time_basis, ("", ""))
        truncation_warning = (
            (
                "AirLabs 参考区每次最多返回 50 行，因此参考时刻可能不完整。",
                "The AirLabs reference section returns at most 50 rows per query and "
                "may be incomplete.",
            )
            if schedule_result.sample_truncated
            else ("", "")
        )

        fare_metadata = self._fare_metadata(fare_result)
        result_status = {
            "confirmed_offers": ("verified_offers_found" if offers else "no_verified_offer"),
            "no_results": "no_verified_offer",
            "not_configured": "fare_provider_not_configured",
            "test_environment_rejected": "fare_provider_test_rejected",
            "authentication_failed": "fare_provider_authentication_failed",
            "rate_limited": "fare_provider_rate_limited",
            "budget_not_configured": "fare_provider_budget_not_configured",
            "budget_exhausted": "fare_provider_budget_exhausted",
            "provider_processing": "fare_provider_processing",
            "provider_error": "fare_provider_error",
            "provider_unavailable": "fare_provider_unavailable",
        }[fare_result.status]
        if (
            not offers
            and fare_result.status == "no_results"
            and fare_result.quota_skipped_candidate_count > 0
            and fare_result.coverage_status
            in {"quota_limited", "quota_and_provider_incomplete"}
        ):
            # Actual provider/account quota did not permit verification of every
            # returned candidate, so this is not evidence that no bookable flight
            # exists.
            result_status = "fare_provider_coverage_limited"
        rejected_warning = (
            (
                f" 另有 {rejected_priced_offers} 个生产报价因机场时区、连续航段或字段"
                "完整性未通过本地严格验证而被排除。",
                f" {rejected_priced_offers} additional production-priced offer(s) failed "
                "local strict validation for airport timezones, continuity, or completeness.",
            )
            if rejected_priced_offers
            else ("", "")
        )
        fare_warning = (
            (
                "主价格是查询时经已启用的严格报价来源及其二次购票验证标识确认的"
                "一位成人单程 USD 报价；",
                "The primary price is a one-way, one-adult USD offer confirmed at query "
                "time by an enabled strict fare source and that source's secondary "
                "booking-verification identifier. ",
            )
            if offers and fare_result.status == "confirmed_offers"
            else (
                "本次没有价格通过严格购票选项验证；",
                "No price passed strict booking-option verification in this response. ",
            )
        )
        coverage_warning = (
            (f" {fare_metadata.notice.zh}", f" {fare_metadata.notice.en}")
            if fare_metadata.coverage_status
            in {
                "quota_limited",
                "provider_incomplete",
                "quota_and_provider_incomplete",
            }
            else ("", "")
        )

        return ComparisonResponse(
            origin=request.origin,
            destination=request.destination,
            departure_date=departure_date,
            departure_time=departure_time,
            departure_time_basis=departure_time_basis,
            departure_timezone=departure_timezone,
            distance_km=route.distance_km,
            duration_minutes=route.duration_minutes,
            generated_at=generated_at,
            context=self._context_response(context),
            offers=offers,
            rankings=ComparisonRankings(
                direct_first=[offer.id for offer in direct],
                lowest_price=[offer.id for offer in cheapest],
                student_first=[offer.id for offer in student],
            ),
            availability_mode="strict_bookable_only",
            result_status=result_status,
            strict_mode_notice=BilingualText(
                zh=(
                    "严格可售模式仅显示经已启用的严格报价来源搜索，并由该来源的二次购票"
                    "验证标识确认的报价。系统会尝试验证来源返回的全部合格候选；只有实际免费"
                    "额度不足或供应商错误才会使覆盖不完整，未验证候选绝不会进入主列表。"
                    "测试数据、AirLabs 时刻、周期投影和纯模型航班都不能进入主列表。"
                ),
                en=(
                    "Strict bookable mode shows only offers confirmed by both an enabled "
                    "strict fare source search and that source's secondary booking-verification "
                    "identifier. The system attempts every eligible candidate returned by each "
                    "source; only actual free-quota exhaustion or provider errors can leave "
                    "coverage incomplete. "
                    "Unverified candidates "
                    "never enter the main list. Test data, AirLabs "
                    "timetables, recurring projections, and model-only flights cannot enter "
                    "the main list."
                ),
            ),
            fare_search_metadata=fare_metadata,
            fare_reference_snapshots=fare_reference_snapshots,
            historical_market_contexts=historical_market_contexts,
            timetable_references=timetable_references,
            schedule_sample_truncated=schedule_result.sample_truncated,
            schedule_sample_limit=AIRLABS_FREE_SAMPLE_LIMIT,
            warnings=BilingualWarning(
                zh=(
                    f"{fare_warning[0]}模型估价、80% 区间、价格曲线和准点率仍是"
                    "合成演示模型结果。"
                    "搜索结果和购票价格可能随时变化，航空公司最终结账页才是最终价格；"
                    "系统不会为覆盖缺口补造航班。"
                    f" {reference_warning[0]}"
                    f" {truncation_warning[0]}"
                    f"{rejected_warning[0]}"
                    f"{coverage_warning[0]}"
                ),
                en=(
                    f"{fare_warning[1]}The model estimate, 80% "
                    "interval, price curve, and on-time probability remain synthetic-demo model "
                    "outputs. Search results and booking prices can change at any time; the "
                    "airline checkout is authoritative. The system does not invent flights to "
                    "fill coverage gaps."
                    f" {reference_warning[1]}"
                    f" {truncation_warning[1]}"
                    f"{rejected_warning[1]}"
                    f"{coverage_warning[1]}"
                ),
            ),
            model_version=self.model_version,
        )

    def _price_curve_for_offer(
        self,
        *,
        comparison: ComparisonResponse,
        offer: ComparisonOffer,
        route: RouteEstimate,
    ) -> PriceForecastCurve:
        """Project the model once per origin-local day until scheduled departure.

        These points are generated together from the demo model by varying only
        quote time. They are deliberately not described as observed fare history.
        """

        if offer.scheduled_departure_local is None or offer.live_fare is None:
            raise OfferNotFoundError(
                "strict offer no longer has a complete departure time and live fare"
            )
        departure = offer.scheduled_departure_local
        departure_utc = departure.astimezone(UTC)
        generated_at = comparison.generated_at.astimezone(UTC)
        if generated_at >= departure_utc:
            raise OfferNotFoundError("strict offer has already departed")

        origin_zone, _ = self._airport_timezone(route.origin)
        start_date = generated_at.astimezone(origin_zone).date()
        requested_end_date = departure.astimezone(origin_zone).date()
        quote_times: list[datetime] = []
        quote_dates: list[date] = []
        day_count = (requested_end_date - start_date).days
        for offset in range(day_count + 1):
            target_date = start_date + timedelta(days=offset)
            if offset == 0:
                candidate = generated_at
            else:
                candidate = datetime.combine(target_date, time(12), tzinfo=origin_zone)
                latest = departure_utc - timedelta(minutes=1)
                if candidate.astimezone(UTC) > latest:
                    candidate = latest
            candidate_utc = candidate.astimezone(UTC)
            candidate_date = candidate_utc.astimezone(origin_zone).date()
            # A departure exactly at local midnight has no pre-departure instant on
            # that local date. End at the latest honest date instead of relabelling it.
            if candidate_date != target_date or candidate_utc >= departure_utc:
                continue
            if quote_times and candidate_utc <= quote_times[-1].astimezone(UTC):
                continue
            quote_dates.append(candidate_date)
            quote_times.append(candidate)

        if not quote_times:
            raise OfferNotFoundError("no pre-departure quote instant is available")

        curve_distance_km = route.distance_km
        if offer.segments:
            curve_distance_km = round(
                sum(
                    self._route(segment.origin, segment.destination).distance_km
                    for segment in offer.segments
                ),
                1,
            )

        rows = pd.DataFrame(
            [
                {
                    "origin": comparison.origin,
                    "destination": comparison.destination,
                    "airline": offer.airline_code,
                    "cabin": offer.cabin,
                    "stops": offer.stops or 0,
                    "quote_time": quote_time,
                    "departure_time": departure,
                    "distance_km": curve_distance_km,
                    "duration_minutes": offer.duration_minutes,
                    "news_disruption_index": comparison.context.news.value,
                    **_local_time_features(departure),
                }
                for quote_time in quote_times
            ]
        )
        features = build_price_features(rows)
        estimates = np.maximum(
            0.0,
            np.expm1(self.bundle["price_model"].predict(features)),
        )
        raw_estimates = [round(float(estimate), 2) for estimate in estimates]
        # The comparison already exposes this exact first model prediction. Reuse
        # that public value as the retained raw baseline so explanation metadata
        # and the selected comparison row cannot drift by floating-point noise.
        raw_estimates[0] = offer.estimated_price_usd
        anchor_price_usd = offer.live_fare.total_amount
        calibration_log1p_offset = float(
            np.log1p(anchor_price_usd) - np.log1p(raw_estimates[0])
        )

        def calibrate(raw_amount: float) -> float:
            return max(
                0.0,
                float(
                    np.expm1(
                        np.log1p(max(0.0, raw_amount))
                        + calibration_log1p_offset
                    )
                ),
            )

        half_width = float(self.bundle["price_interval_half_width_usd"])
        points: list[PriceForecastPoint] = []
        for index, (quote_date, quote_time, raw_estimate) in enumerate(
            zip(
                quote_dates,
                quote_times,
                raw_estimates,
                strict=True,
            )
        ):
            calibrated_estimate = (
                anchor_price_usd
                if index == 0
                else round(calibrate(raw_estimate), 2)
            )
            calibrated_low = round(
                calibrate(max(0.0, raw_estimate - half_width)),
                2,
            )
            calibrated_high = round(calibrate(raw_estimate + half_width), 2)
            points.append(
                PriceForecastPoint(
                    quote_date=quote_date,
                    quote_time=quote_time,
                    days_until_departure=round(
                        (
                            departure_utc - quote_time.astimezone(UTC)
                        ).total_seconds()
                        / 86_400.0,
                        4,
                    ),
                    estimated_price_usd=calibrated_estimate,
                    interval_80_low_usd=min(
                        calibrated_low,
                        calibrated_estimate,
                    ),
                    interval_80_high_usd=max(
                        calibrated_high,
                        calibrated_estimate,
                    ),
                )
            )
        extrapolated = any(point.days_until_departure > 180.0 for point in points)
        return PriceForecastCurve(
            anchor_price_usd=anchor_price_usd,
            anchor_verified_at=offer.live_fare.verified_at,
            anchor_provider_code=offer.live_fare.provider_code,
            raw_model_start_price_usd=raw_estimates[0],
            calibration_log1p_offset=calibration_log1p_offset,
            start_date=points[0].quote_date,
            end_date=points[-1].quote_date,
            generated_at=generated_at,
            extrapolated_beyond_training_horizon=extrapolated,
            points=points,
            notice=BilingualText(
                zh=(
                    "曲线先由同一合成演示模型仅改变模拟查询日期生成，再对所有点"
                    "施加同一个 log1p 偏移，使第一个点精确等于本次已验证实时报价。"
                    "趋势和区间仍是演示模型输出，不是已采集的历史票价、未来实时"
                    "票价或可购买报价。"
                    + (
                        " 部分点超过模型主要的 180 天训练提前期，属于外推。"
                        if extrapolated
                        else ""
                    )
                ),
                en=(
                    "Every point is first generated by the same synthetic-demo model while "
                    "varying only the simulated quote date. One log1p offset is then applied "
                    "to every point so the first point exactly equals this request's verified "
                    "live fare. The trajectory and interval remain demo-model outputs, not "
                    "collected fare history, future live prices, or bookable quotes."
                    + (
                        " Some points extend beyond the model's main 180-day training horizon."
                        if extrapolated
                        else ""
                    )
                ),
            ),
        )

    def offer_detail(self, request: OfferDetailRequest) -> OfferDetailResponse:
        comparison = self.compare(
            ComparisonRequest(
                origin=request.origin,
                destination=request.destination,
                departure_date=request.departure_date,
            ),
            force_fare_refresh=request.force_refresh,
        )
        offer = next(
            (candidate for candidate in comparison.offers if candidate.id == request.offer_id),
            None,
        )
        if offer is None:
            raise OfferNotFoundError(
                "该选项已不存在或与路线/日期不匹配 / "
                "offer does not exist or does not match the route and date"
            )

        route = self._route(request.origin, request.destination)
        if (
            offer.schedule_status != "priced_offer"
            or offer.bookability_status != "booking_option_verified"
            or offer.live_fare is None
            or not offer.live_fare.booking_verified
            or not offer.segments
        ):
            raise OfferNotFoundError("strict offer no longer has a verified booking option")

        legs: list[ItineraryLeg] = []
        total_distance_km = 0.0
        for segment in offer.segments:
            leg_route = self._route(segment.origin, segment.destination)
            total_distance_km += leg_route.distance_km
            legs.append(
                ItineraryLeg(
                    sequence=segment.sequence,
                    origin=segment.origin,
                    destination=segment.destination,
                    date_context=segment.departure_local.date(),
                    flight_number=segment.flight_number,
                    marketing_airline_code=segment.marketing_airline_code,
                    operating_airline_code=segment.operating_airline_code,
                    departure_local=segment.departure_local,
                    arrival_local=segment.arrival_local,
                    departure_utc=segment.departure_utc,
                    arrival_utc=segment.arrival_utc,
                    duration_minutes=segment.duration_minutes,
                    distance_km=leg_route.distance_km,
                    departure_terminal=segment.departure_terminal,
                    arrival_terminal=segment.arrival_terminal,
                    aircraft_icao=segment.aircraft_icao,
                    cabin=segment.cabin,
                    booking_class=segment.booking_class,
                    fare_basis=segment.fare_basis,
                    fare_brand=segment.fare_brand,
                    included_checked_bag_quantity=(segment.included_checked_bag_quantity),
                    included_checked_bag_weight=segment.included_checked_bag_weight,
                    included_checked_bag_weight_unit=(segment.included_checked_bag_weight_unit),
                    data_basis=_FARE_PROVIDER_DETAIL_BASES[offer.live_fare.provider_code],
                )
            )

        layovers: list[ItineraryLayover] = []
        for sequence, (previous, current) in enumerate(
            zip(offer.segments, offer.segments[1:], strict=False),
            start=1,
        ):
            duration_minutes = round(
                (current.departure_utc - previous.arrival_utc).total_seconds() / 60
            )
            if not 0 <= duration_minutes <= 1_440:
                raise OfferNotFoundError("provider layover is outside strict limits")
            layovers.append(
                ItineraryLayover(
                    sequence=sequence,
                    airport=previous.destination,
                    duration_minutes=duration_minutes,
                )
            )

        if len(legs) == 1:
            itinerary_kind = "direct"
            layover_status = "not_applicable"
        elif len(legs) == 2:
            itinerary_kind = "one_stop"
            layover_status = "provider_confirmed"
        else:
            itinerary_kind = "multi_stop"
            layover_status = "provider_confirmed"
        itinerary = OfferItinerary(
            kind=itinerary_kind,
            time_basis="provider_schedule",
            total_duration_minutes=offer.duration_minutes,
            total_distance_km=round(total_distance_km, 1),
            layover_status=layover_status,
            legs=legs,
            layovers=layovers,
        )

        historical_market_context = next(
            (
                context
                for context in comparison.historical_market_contexts
                if context.cabin == offer.live_fare.cabin_summary
            ),
            None,
        )
        return OfferDetailResponse(
            origin=request.origin,
            destination=request.destination,
            departure_date=request.departure_date,
            generated_at=comparison.generated_at,
            schedule_status=offer.schedule_status,
            schedule_source=offer.schedule_source,
            schedule_observed_at=offer.schedule_observed_at,
            schedule_sample_truncated=comparison.schedule_sample_truncated,
            schedule_sample_limit=comparison.schedule_sample_limit,
            fallback_reason=None,
            fare_search_metadata=comparison.fare_search_metadata,
            offer=offer,
            itinerary=itinerary,
            historical_market_context=historical_market_context,
            price_curve=self._price_curve_for_offer(
                comparison=comparison,
                offer=offer,
                route=route,
            ),
            notice=BilingualText(
                zh=(
                    "航班号、完整当地/UTC 时刻、舱位和主价格已通过 "
                    f"{offer.live_fare.provider_name} 的购票路径验证。"
                    "页面提供当次验证得到的 HTTPS 预订链接，"
                    "但搜索结果及最终结账价格仍可能随时变化；"
                    "模型估价、价格曲线和准点率仍是演示模型输出，价格曲线不是历史票价。"
                    + (
                        " AirLabs 参考时刻区最多返回 50 行，可能不完整。"
                        if comparison.schedule_sample_truncated
                        else ""
                    )
                ),
                en=(
                    "Flight numbers, complete local/UTC times, cabin, and the primary fare "
                    f"passed {offer.live_fare.provider_name} booking-path verification. "
                    "The page exposes "
                    "the HTTPS booking link returned by that verification, but search results "
                    "and the final checkout price can still change at any time. The model "
                    "estimate, price curve, and on-time probability remain demo-model outputs; "
                    "the curve is not observed fare history."
                    + (
                        " The AirLabs timetable reference section is capped at 50 rows and "
                        "may be incomplete."
                        if comparison.schedule_sample_truncated
                        else ""
                    )
                ),
            ),
        )

    def _model_offer_fallback(
        self,
        request: OfferDetailRequest,
        airline_code: str,
    ) -> BilingualText:
        route = self._route(request.origin, request.destination)
        origin_zone, _ = self._airport_timezone(route.origin)
        destination_zone, _ = self._airport_timezone(route.destination)
        result = self.schedule_provider.search(
            request.origin,
            request.destination,
            request.departure_date,
            origin_timezone=origin_zone,
            destination_timezone=destination_zone,
        )
        known = self._fallback_reason(result.fallback_code)
        if known is not None:
            return known
        return BilingualText(
            zh=(
                f"免费提供商没有为 {airline_code} 返回该日期可验证的完整时刻；"
                "因此不显示航班号或精确钟点。"
            ),
            en=(
                f"The free provider returned no verifiable complete timetable for "
                f"{airline_code} on this date, so no flight number or exact clock time "
                "is shown."
            ),
        )

    def weather_detail(self, request: ContextDetailRequest) -> WeatherDetailResponse:
        route = self._route(request.origin, request.destination)
        generated_at = self._now_provider()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise RouteLookupError("service clock must include a timezone offset")
        generated_at = generated_at.astimezone(UTC)
        if request.departure_date is not None:
            departure, origin_timezone, departure_time_basis = self._departure_date_at_origin(
                request.departure_date,
                route.origin,
                generated_at,
            )
        else:
            assert request.departure_time is not None
            departure, origin_timezone = self._departure_at_origin(
                request.departure_time,
                route.origin,
            )
            departure_time_basis = "legacy_input"
        destination_zone, destination_timezone = self._airport_timezone(route.destination)
        arrival = (
            departure.astimezone(UTC) + timedelta(minutes=route.duration_minutes)
        ).astimezone(destination_zone)
        with ThreadPoolExecutor(max_workers=2) as executor:
            origin_future = executor.submit(
                self.detail_provider.airport_weather,
                route.origin,
                origin_timezone,
                departure,
                generated_at=generated_at,
            )
            destination_future = executor.submit(
                self.detail_provider.airport_weather,
                route.destination,
                destination_timezone,
                arrival,
                generated_at=generated_at,
            )
            origin_weather = origin_future.result()
            destination_weather = destination_future.result()
        return WeatherDetailResponse(
            origin=route.origin.iata,
            destination=route.destination.iata,
            departure_time=departure,
            departure_time_basis=departure_time_basis,
            estimated_arrival_time=arrival,
            duration_minutes=route.duration_minutes,
            generated_at=generated_at,
            origin_weather=origin_weather,
            destination_weather=destination_weather,
            notice=BilingualText(
                zh=(
                    "到达时刻由航线模型估算；自动 METAR/TAF 解读仅供参考，"
                    "不能替代官方航空气象简报。"
                ),
                en=(
                    "Arrival time is route-model estimated. Automated METAR/TAF "
                    "interpretation is informational and does not replace an official "
                    "aviation-weather briefing."
                ),
            ),
        )

    def news_detail(self, request: ContextDetailRequest) -> NewsDetailResponse:
        route = self._route(request.origin, request.destination)
        generated_at = self._now_provider()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise RouteLookupError("service clock must include a timezone offset")
        generated_at = generated_at.astimezone(UTC)
        if request.departure_date is not None:
            departure, _, departure_time_basis = self._departure_date_at_origin(
                request.departure_date,
                route.origin,
                generated_at,
            )
        else:
            assert request.departure_time is not None
            departure, _ = self._departure_at_origin(request.departure_time, route.origin)
            departure_time_basis = "legacy_input"
        with ThreadPoolExecutor(max_workers=2) as executor:
            detail_future = executor.submit(
                self.detail_provider.news,
                route.origin.iata,
                route.destination.iata,
                origin_name=route.origin.name,
                destination_name=route.destination.name,
                generated_at=generated_at,
            )
            model_future = executor.submit(
                self.context_provider._news,  # noqa: SLF001
                route.origin.iata,
                route.destination.iata,
                departure,
                generated_at,
                origin_name=route.origin.name,
                destination_name=route.destination.name,
            )
            snapshot = detail_future.result()
            model_signal = model_future.result()
        attenuation = self.detail_provider.departure_attenuation(departure, generated_at)
        return NewsDetailResponse(
            origin=route.origin.iata,
            destination=route.destination.iata,
            departure_time=departure,
            departure_time_basis=departure_time_basis,
            generated_at=generated_at,
            article_count=len(snapshot.articles),
            articles=list(snapshot.articles),
            route_raw_risk=snapshot.route_raw_risk,
            departure_attenuation_factor=attenuation,
            model_effect=model_signal.value,
            model_signal=ContextSignal(
                value=model_signal.value,
                status=model_signal.status,
                source=model_signal.source,
                observed_at=model_signal.observed_at,
                summary_zh=model_signal.summary_zh,
                summary_en=model_signal.summary_en,
            ),
            metadata=snapshot.metadata,
            summary=snapshot.summary,
            indexed_time_notice=BilingualText(
                zh=("文章时间是 GDELT 观察/索引时间，不保证是媒体发布时间；标题保留来源语言。"),
                en=(
                    "Article times are GDELT observed/indexed times, not guaranteed "
                    "publisher timestamps; titles remain in their source language."
                ),
            ),
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "available_tasks": [
                "fare_estimation",
                "on_time_probability",
                "global_airline_cabin_comparison",
            ],
            "runtime_context": ["weather", "airport_operations", "current_news"],
            "metadata": self.bundle["metadata"],
            "metrics": self.bundle["metrics"],
        }
