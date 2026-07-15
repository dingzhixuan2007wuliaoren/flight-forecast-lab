from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from flight_forecaster.catalog import AirlineProfile, comparison_airlines, get_airline_profile
from flight_forecaster.context import ContextProvider, PredictionContext
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
from flight_forecaster.schemas import (
    BilingualText,
    BilingualWarning,
    ComparisonOffer,
    ComparisonRankings,
    ComparisonRequest,
    ComparisonResponse,
    ContextDetailRequest,
    ContextSignal,
    NewsArticle,
    NewsDetailResponse,
    NewsSignal,
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


class PredictionService:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        context_provider: ContextProvider | None = None,
        airport_resolver: AirportResolver | None = None,
        detail_provider: DetailProvider | None = None,
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

    def compare(self, request: ComparisonRequest) -> ComparisonResponse:
        generated_at = datetime.now(UTC)
        route = self._route(request.origin, request.destination)
        departure_time, departure_timezone = self._departure_at_origin(
            request.departure_time,
            route.origin,
        )
        context = self._context(route, departure_time)
        confirmed_airlines = self.context_provider.route_airlines(
            request.origin, request.destination
        )

        profile_codes = {profile.code for profile in comparison_airlines()}
        if confirmed_airlines:
            profile_codes.update(confirmed_airlines)

        profiles: list[AirlineProfile | _GenericAirlineProfile] = []
        for code in sorted(profile_codes):
            profiles.append(
                get_airline_profile(code) or _GenericAirlineProfile(code=code, name=code)
            )

        scenario_rows: list[dict[str, Any]] = []
        scenario_profiles: list[
            tuple[AirlineProfile | _GenericAirlineProfile, str, str]
        ] = []
        for profile in profiles:
            route_status = (
                "provider_confirmed"
                if confirmed_airlines is not None and profile.code in confirmed_airlines
                else "model_scenario"
            )
            scenario_stops = 0 if route_status == "provider_confirmed" else 1
            for cabin in profile.supported_cabins:
                scenario_rows.append(
                    {
                        "origin": request.origin,
                        "destination": request.destination,
                        "airline": profile.code,
                        "cabin": cabin,
                        "stops": scenario_stops,
                        "quote_time": generated_at,
                        "departure_time": departure_time,
                        "distance_km": route.distance_km,
                        "duration_minutes": route.duration_minutes + 90 * scenario_stops,
                        "news_disruption_index": context.news.value,
                        **_local_time_features(departure_time),
                    }
                )
                scenario_profiles.append((profile, cabin, route_status))

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
                    "airline": profile.code,
                    "scheduled_departure": departure_time,
                    "distance_km": route.distance_km,
                    "weather_severity_forecast": context.weather.value,
                    "origin_congestion_index": context.operations.value,
                    "news_disruption_index": context.news.value,
                    **_local_time_features(departure_time),
                }
                for profile in profiles
            ]
        )
        ontime_features = build_ontime_features(ontime_rows)
        probabilities = np.clip(
            self.bundle["ontime_model"].predict_proba(ontime_features)[:, 1],
            0.0,
            1.0,
        )
        probability_by_airline = {
            profile.code: round(float(probability), 4)
            for profile, probability in zip(profiles, probabilities, strict=True)
        }

        offers: list[ComparisonOffer] = []
        for (profile, cabin, route_status), raw_estimate in zip(
            scenario_profiles, estimates, strict=True
        ):
            estimate = round(float(raw_estimate), 2)
            scenario_stops = 0 if route_status == "provider_confirmed" else 1
            base_leg_probability = probability_by_airline[profile.code]
            probability = round(base_leg_probability ** (scenario_stops + 1), 4)
            duration_minutes = route.duration_minutes + 90 * scenario_stops
            offers.append(
                ComparisonOffer(
                    id=f"{profile.code}-{cabin}-{scenario_stops}",
                    airline_code=profile.code,
                    airline_name=profile.name,
                    cabin=cabin,
                    stops=scenario_stops,
                    duration_minutes=duration_minutes,
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
                    route_status=route_status,
                    cabin_status="catalog_scenario",
                    punctuality_basis=(
                        "direct_leg_model"
                        if scenario_stops == 0
                        else "two_leg_independence_scenario"
                    ),
                )
            )

        direct = sorted(
            offers,
            key=lambda offer: (
                offer.stops,
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
                offer.stops,
                -offer.on_time_probability,
                offer.airline_code,
                offer.cabin,
            ),
        )
        student = sorted(offers, key=self._student_sort_key)

        return ComparisonResponse(
            origin=request.origin,
            destination=request.destination,
            departure_time=departure_time,
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
            warnings=BilingualWarning(
                zh=(
                    "所有价格与准点率均来自合成演示模型，价格不是实时可售票价；未由免费数据源确认的"
                    "航司、舱位、行李、学生优惠和退改规则均明确标为场景或未知。"
                ),
                en=(
                    "All prices and on-time probabilities come from synthetic-demo models; "
                    "prices are not live bookable fares. "
                    "Airlines, cabins, baggage, student benefits, and fare rules not confirmed "
                    "by a free source are explicitly labelled as scenarios or unknown."
                ),
            ),
            model_version=self.model_version,
        )

    def weather_detail(self, request: ContextDetailRequest) -> WeatherDetailResponse:
        route = self._route(request.origin, request.destination)
        departure, origin_timezone = self._departure_at_origin(
            request.departure_time,
            route.origin,
        )
        destination_zone, destination_timezone = self._airport_timezone(route.destination)
        arrival = (
            departure.astimezone(UTC) + timedelta(minutes=route.duration_minutes)
        ).astimezone(destination_zone)
        generated_at = datetime.now(UTC)
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
        departure, _ = self._departure_at_origin(request.departure_time, route.origin)
        generated_at = datetime.now(UTC)
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
