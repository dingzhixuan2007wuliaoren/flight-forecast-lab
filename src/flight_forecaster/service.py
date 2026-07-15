from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from flight_forecaster.catalog import AirlineProfile, comparison_airlines, get_airline_profile
from flight_forecaster.context import (
    AIRLABS_FREE_SAMPLE_LIMIT,
    ContextProvider,
    PredictionContext,
)
from flight_forecaster.details import DetailProvider
from flight_forecaster.features import build_ontime_features, build_price_features
from flight_forecaster.route_info import (
    Airport,
    AirportResolver,
    OurAirportsResolver,
    RouteEstimate,
    RouteLookupError,
    estimate_route,
)
from flight_forecaster.schedules import FlightSchedule, ScheduleProvider
from flight_forecaster.schemas import (
    BilingualText,
    BilingualWarning,
    ComparisonOffer,
    ComparisonRankings,
    ComparisonRequest,
    ComparisonResponse,
    ContextDetailRequest,
    ContextSignal,
    ItineraryLeg,
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
    PricePrediction,
    PriceRequest,
    WeatherDetailResponse,
)
from flight_forecaster.training import ARTIFACT_FILENAME, SCHEMA_VERSION


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
    supported_cabins: tuple[str, ...] = (
        "economy",
        "premium_economy",
        "business",
        "first",
    )
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
    schedule: FlightSchedule | None = None
    hub: str | None = None


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

        self.context_provider = context_provider or ContextProvider(
            airlabs_api_key=os.getenv("AIRLABS_API_KEY"),
            context_priors=self.bundle.get("context_priors"),
        )
        self.detail_provider = detail_provider or DetailProvider(self.context_provider)
        self.schedule_provider = schedule_provider or ScheduleProvider(
            api_key=self.context_provider.airlabs_api_key,
            client=self.context_provider.client,
            enabled=self.context_provider.external_context_enabled,
        )
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        if airport_resolver is not None:
            self.airport_resolver = airport_resolver
        elif self.context_provider.external_context_enabled:
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
            round_trip = (
                departure.astimezone(UTC)
                .astimezone(timezone)
                .replace(tzinfo=None)
            )
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
            raise RouteLookupError(
                "计划出发时间必须晚于当前时间 / departure must be in the future"
            )
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
        schedule: FlightSchedule | None,
    ) -> str:
        schedule_identity = (
            f"{schedule.flight_number}|{schedule.departure_utc.isoformat()}"
            if schedule is not None
            else "model"
        )
        canonical = "|".join(
            (
                origin,
                destination,
                departure_date.isoformat(),
                airline,
                cabin,
                str(stops),
                schedule_identity,
            )
        )
        return f"off_{sha256(canonical.encode('utf-8')).hexdigest()[:24]}"

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
        row = pd.DataFrame(
            [
                {
                    "origin": origin,
                    "destination": destination,
                    "airline": airline,
                    "scheduled_departure": departure_time,
                    "distance_km": route.distance_km,
                    "weather_severity_forecast": context.weather.value,
                    "origin_congestion_index": context.operations.value,
                    "news_disruption_index": context.news.value,
                    **_local_time_features(departure_time),
                }
            ]
        )
        features = build_ontime_features(row)
        probability = float(self.bundle["ontime_model"].predict_proba(features)[0, 1])
        return round(float(np.clip(probability, 0.0, 1.0)), 4)

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
            warning=(
                "模型估价已纳入带来源的新闻风险信号，但并非实时可购买报价，也不保证最低价。"
            ),
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
        return OnTimePrediction(
            on_time_probability=probability,
            disruption_probability=round(1.0 - probability, 4),
            distance_km=route.distance_km,
            risk_level=self._risk_level(probability),
            definition="未取消且到达延误少于 15 分钟",
            definition_en="Not cancelled and arrival delay under 15 minutes",
            model_version=self.model_version,
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
        )

    @staticmethod
    def _student_sort_key(offer: ComparisonOffer) -> tuple[Any, ...]:
        baggage_rank = int(
            offer.baggage_status not in {"confirmed_free", "confirmed_included"}
        )

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
            offer.estimated_price_usd,
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
        return {
            "provider_direct": 0,
            "model_one_stop": 1,
            "model_route_unresolved": 2,
        }[offer.routing_status]

    def compare(self, request: ComparisonRequest) -> ComparisonResponse:
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
        schedule_result = self.schedule_provider.search(
            request.origin,
            request.destination,
            departure_date,
            origin_timezone=origin_zone,
            destination_timezone=destination_zone,
            fetched_at=generated_at,
        )
        confirmed_airlines = set(schedule_result.route_airlines)
        if not confirmed_airlines and (
            not self.context_provider.airlabs_api_key
            or not self.context_provider.external_context_enabled
        ):
            legacy_confirmed = self.context_provider.route_airlines(
                request.origin,
                request.destination,
            )
            if legacy_confirmed:
                confirmed_airlines.update(legacy_confirmed)
        context = self._context(route, departure_time)

        profile_codes = {profile.code for profile in comparison_airlines()}
        if confirmed_airlines:
            profile_codes.update(confirmed_airlines)
        profile_codes.update(schedule.airline_code for schedule in schedule_result.schedules)

        profiles: list[AirlineProfile | _GenericAirlineProfile] = []
        for code in sorted(profile_codes):
            profiles.append(
                get_airline_profile(code) or _GenericAirlineProfile(code=code, name=code)
            )

        schedules_by_airline: dict[str, list[FlightSchedule]] = {}
        for schedule in schedule_result.schedules:
            schedules_by_airline.setdefault(schedule.airline_code, []).append(schedule)

        scenarios: list[_OfferScenario] = []
        for profile in profiles:
            provider_schedules = schedules_by_airline.get(profile.code, [])
            if provider_schedules:
                for schedule in provider_schedules:
                    for cabin in profile.supported_cabins:
                        scenarios.append(
                            _OfferScenario(
                                profile=profile,
                                cabin=cabin,
                                route_status="provider_confirmed",
                                stops=0,
                                model_stops=0,
                                routing_status="provider_direct",
                                departure_time=schedule.departure_local,
                                duration_minutes=schedule.duration_minutes,
                                distance_km=route.distance_km,
                                schedule=schedule,
                            )
                        )
                continue

            route_confirmed = profile.code in confirmed_airlines
            hub = (
                None
                if route_confirmed
                else self._model_hub(profile.code, request.origin, request.destination)
            )
            if route_confirmed:
                scenario_stops: int | None = 0
                model_stops = 0
                routing_status = "provider_direct"
            elif hub is not None:
                scenario_stops = 1
                model_stops = 1
                routing_status = "model_one_stop"
            else:
                scenario_stops = None
                model_stops = 0
                routing_status = "model_route_unresolved"
            if hub is None:
                scenario_duration = route.duration_minutes
                scenario_distance = route.distance_km
            else:
                first_leg = self._route(request.origin, hub)
                second_leg = self._route(hub, request.destination)
                scenario_duration = (
                    first_leg.duration_minutes + 90 + second_leg.duration_minutes
                )
                scenario_distance = round(
                    first_leg.distance_km + second_leg.distance_km,
                    1,
                )
            for cabin in profile.supported_cabins:
                scenarios.append(
                    _OfferScenario(
                        profile=profile,
                        cabin=cabin,
                        route_status=(
                            "provider_confirmed" if route_confirmed else "model_scenario"
                        ),
                        stops=scenario_stops,
                        model_stops=model_stops,
                        routing_status=routing_status,
                        departure_time=departure_time,
                        duration_minutes=scenario_duration,
                        distance_km=scenario_distance,
                        hub=hub,
                    )
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

        price_features = build_price_features(pd.DataFrame(scenario_rows))
        estimates = np.maximum(
            0.0,
            np.expm1(self.bundle["price_model"].predict(price_features)),
        )
        half_width = float(self.bundle["price_interval_half_width_usd"])

        ontime_rows = pd.DataFrame(
            [
                {
                    "origin": request.origin,
                    "destination": request.destination,
                    "airline": scenario.profile.code,
                    "scheduled_departure": scenario.departure_time,
                    "distance_km": scenario.distance_km,
                    "weather_severity_forecast": context.weather.value,
                    "origin_congestion_index": context.operations.value,
                    "news_disruption_index": context.news.value,
                    **_local_time_features(scenario.departure_time),
                }
                for scenario in scenarios
            ]
        )
        ontime_features = build_ontime_features(ontime_rows)
        probabilities = np.clip(
            self.bundle["ontime_model"].predict_proba(ontime_features)[:, 1],
            0.0,
            1.0,
        )
        offers: list[ComparisonOffer] = []
        for scenario, raw_estimate, raw_probability in zip(
            scenarios,
            estimates,
            probabilities,
            strict=True,
        ):
            profile = scenario.profile
            estimate = round(float(raw_estimate), 2)
            probability = round(float(raw_probability) ** (scenario.model_stops + 1), 4)
            schedule = scenario.schedule
            offers.append(
                ComparisonOffer(
                    id=self._offer_id(
                        origin=request.origin,
                        destination=request.destination,
                        departure_date=departure_date,
                        airline=profile.code,
                        cabin=scenario.cabin,
                        stops=scenario.stops,
                        schedule=schedule,
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
                    baggage_status=profile.baggage_status,
                    student_status=profile.student_status,
                    change_status=profile.change_status,
                    refund_status=profile.refund_status,
                    student_age_limit_zh=profile.student_age_limit_zh,
                    student_age_limit_en=profile.student_age_limit_en,
                    student_verification_zh=profile.student_verification_zh,
                    student_verification_en=profile.student_verification_en,
                    student_program_url=profile.student_program_url,
                    route_status=scenario.route_status,
                    routing_status=scenario.routing_status,
                    cabin_status="catalog_scenario",
                    punctuality_basis=(
                        "direct_leg_model"
                        if scenario.routing_status == "provider_direct"
                        else (
                            "two_leg_independence_scenario"
                            if scenario.routing_status == "model_one_stop"
                            else "route_only_model"
                        )
                    ),
                    schedule_status=(schedule.schedule_status if schedule else "model_scenario"),
                    schedule_source=(schedule.source if schedule else "model_fallback"),
                    flight_number=(schedule.flight_number if schedule else None),
                    scheduled_departure_local=(
                        schedule.departure_local if schedule else None
                    ),
                    scheduled_arrival_local=(schedule.arrival_local if schedule else None),
                    scheduled_departure_utc=(schedule.departure_utc if schedule else None),
                    scheduled_arrival_utc=(schedule.arrival_utc if schedule else None),
                    provider_flight_status=(
                        schedule.provider_flight_status if schedule else None
                    ),
                    schedule_observed_at=(schedule.observed_at if schedule else None),
                    departure_terminal=(
                        schedule.departure_terminal
                        if schedule is not None
                        and schedule.schedule_status == "live_schedule"
                        else None
                    ),
                    arrival_terminal=(
                        schedule.arrival_terminal
                        if schedule is not None
                        and schedule.schedule_status == "live_schedule"
                        else None
                    ),
                    aircraft_icao=(
                        schedule.aircraft_icao
                        if schedule is not None
                        and schedule.schedule_status == "live_schedule"
                        else None
                    ),
                )
            )

        direct = sorted(
            offers,
            key=lambda offer: (
                self._routing_rank(offer),
                offer.estimated_price_usd,
                -offer.on_time_probability,
                offer.airline_code,
                offer.cabin,
            ),
        )
        cheapest = sorted(
            offers,
            key=lambda offer: (
                offer.estimated_price_usd,
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
                "免费查询最多返回 50 行，真实航班列表可能不完整。",
                "Free queries return at most 50 rows, so the actual flight list may be "
                "incomplete.",
            )
            if schedule_result.sample_truncated
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
            schedule_sample_truncated=schedule_result.sample_truncated,
            schedule_sample_limit=AIRLABS_FREE_SAMPLE_LIMIT,
            warnings=BilingualWarning(
                zh=(
                    "所有价格与准点率均来自合成演示模型，价格不是实时可售票价；未由免费数据源确认的"
                    "航司、舱位、行李、学生优惠和退改规则均明确标为场景或未知。"
                    f" {reference_warning[0]}"
                    f" {truncation_warning[0]}"
                ),
                en=(
                    "All prices and on-time probabilities come from synthetic-demo models; "
                    "prices are not live bookable fares. "
                    "Airlines, cabins, baggage, student benefits, and fare rules not confirmed "
                    "by a free source are explicitly labelled as scenarios or unknown."
                    f" {reference_warning[1]}"
                    f" {truncation_warning[1]}"
                ),
            ),
            model_version=self.model_version,
        )

    def offer_detail(self, request: OfferDetailRequest) -> OfferDetailResponse:
        comparison = self.compare(
            ComparisonRequest(
                origin=request.origin,
                destination=request.destination,
                departure_date=request.departure_date,
            )
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
        if offer.schedule_status != "model_scenario":
            required_times = (
                offer.scheduled_departure_local,
                offer.scheduled_arrival_local,
                offer.scheduled_departure_utc,
                offer.scheduled_arrival_utc,
            )
            if offer.flight_number is None or any(value is None for value in required_times):
                raise OfferNotFoundError("provider schedule is no longer complete")
            data_basis = (
                "airlabs_live_schedule"
                if offer.schedule_status == "live_schedule"
                else "airlabs_recurring_timetable_projection"
            )
            legs = [
                ItineraryLeg(
                    sequence=1,
                    origin=request.origin,
                    destination=request.destination,
                    date_context=request.departure_date,
                    flight_number=offer.flight_number,
                    departure_local=offer.scheduled_departure_local,
                    arrival_local=offer.scheduled_arrival_local,
                    departure_utc=offer.scheduled_departure_utc,
                    arrival_utc=offer.scheduled_arrival_utc,
                    duration_minutes=offer.duration_minutes,
                    distance_km=route.distance_km,
                    departure_terminal=offer.departure_terminal,
                    arrival_terminal=offer.arrival_terminal,
                    aircraft_icao=offer.aircraft_icao,
                    data_basis=data_basis,
                )
            ]
            itinerary = OfferItinerary(
                kind="direct",
                time_basis="provider_schedule",
                total_duration_minutes=offer.duration_minutes,
                total_distance_km=route.distance_km,
                layover_status="not_applicable",
                legs=legs,
            )
            fallback_reason = None
        elif offer.routing_status == "provider_direct":
            itinerary = OfferItinerary(
                kind="direct",
                time_basis="model_duration_only",
                total_duration_minutes=offer.duration_minutes,
                total_distance_km=route.distance_km,
                layover_status="not_applicable",
                legs=[
                    ItineraryLeg(
                        sequence=1,
                        origin=request.origin,
                        destination=request.destination,
                        date_context=request.departure_date,
                        duration_minutes=offer.duration_minutes,
                        distance_km=route.distance_km,
                        data_basis="model_duration_only",
                    )
                ],
            )
            fallback_reason = self._model_offer_fallback(request, offer.airline_code)
        elif offer.routing_status == "model_route_unresolved":
            itinerary = OfferItinerary(
                kind="route_unresolved",
                time_basis="model_duration_only",
                total_duration_minutes=offer.duration_minutes,
                total_distance_km=route.distance_km,
                layover_status="not_applicable",
                legs=[],
            )
            fallback_reason = self._model_offer_fallback(request, offer.airline_code)
        else:
            hub = self._model_hub(
                offer.airline_code,
                request.origin,
                request.destination,
            )
            if hub is None:
                raise OfferNotFoundError("model connection hub is no longer available")
            first_route = self._route(request.origin, hub)
            second_route = self._route(hub, request.destination)
            total_leg_distance = first_route.distance_km + second_route.distance_km
            itinerary = OfferItinerary(
                kind="one_stop",
                time_basis="model_duration_only",
                total_duration_minutes=offer.duration_minutes,
                total_distance_km=round(total_leg_distance, 1),
                layover_airport=hub,
                layover_minutes=90,
                layover_status="model_assumption",
                legs=[
                    ItineraryLeg(
                        sequence=1,
                        origin=request.origin,
                        destination=hub,
                        date_context=request.departure_date,
                        duration_minutes=first_route.duration_minutes,
                        distance_km=first_route.distance_km,
                        data_basis="model_duration_only",
                    ),
                    ItineraryLeg(
                        sequence=2,
                        origin=hub,
                        destination=request.destination,
                        date_context=request.departure_date,
                        duration_minutes=second_route.duration_minutes,
                        distance_km=second_route.distance_km,
                        data_basis="model_duration_only",
                    ),
                ],
            )
            fallback_reason = self._model_offer_fallback(request, offer.airline_code)

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
            fallback_reason=fallback_reason,
            offer=offer,
            itinerary=itinerary,
            notice=BilingualText(
                zh=(
                    "舱位始终是目录场景，并非已确认库存；价格与准点率来自演示模型。"
                    "只有标为 AirLabs 的航班号和完整时刻来自免费提供商数据。"
                    + (
                        "免费查询最多返回 50 行，真实航班列表可能不完整。"
                        if comparison.schedule_sample_truncated
                        else ""
                    )
                ),
                en=(
                    "Cabin is always a catalog scenario, not confirmed inventory; price "
                    "and on-time probability come from demo models. Only flight numbers "
                    "and complete times labelled AirLabs come from the free provider."
                    + (
                        " Free queries return at most 50 rows, so the actual flight list "
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
            departure, origin_timezone, departure_time_basis = (
                self._departure_date_at_origin(
                    request.departure_date,
                    route.origin,
                    generated_at,
                )
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
                zh=(
                    "文章时间是 GDELT 观察/索引时间，不保证是媒体发布时间；"
                    "标题保留来源语言。"
                ),
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
