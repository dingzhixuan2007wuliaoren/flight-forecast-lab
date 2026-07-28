"""Detailed, failure-safe weather and news data for the second-level pages.

The prediction context remains intentionally compact.  This module keeps the richer
provider payloads separate so displaying charts and source reports cannot silently
change the model inputs.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any
from xml.etree import ElementTree

from flight_forecaster.context import (
    GDELT_DOC_URL,
    GDELT_GAL_RSS_URL,
    GDELT_REQUEST_TIMEOUT_SECONDS,
    GDELT_RSS_REQUEST_TIMEOUT_SECONDS,
    NOAA_METAR_URL,
    NOAA_TAF_URL,
    OPEN_METEO_URL,
    ContextProvider,
    _bounded,
    _parse_datetime,
    _utc,
)
from flight_forecaster.route_info import Airport
from flight_forecaster.schemas import (
    AirportWeatherDetail,
    AviationWeatherReport,
    BilingualText,
    HourlyWeatherPoint,
    NewsDetailArticle,
    ProviderMetadata,
    WeatherObservation,
    WeatherRiskComponent,
)

WEATHER_CACHE_TTL_SECONDS = 600.0
NEWS_CACHE_TTL_SECONDS = 900.0
STALE_CACHE_TTL_SECONDS = 21_600.0
FAILURE_CACHE_TTL_SECONDS = 60.0
MAX_CACHE_ENTRIES = 256


@dataclass(frozen=True, slots=True)
class NewsDetailSnapshot:
    articles: tuple[NewsDetailArticle, ...]
    route_raw_risk: float
    metadata: ProviderMetadata
    summary: BilingualText


_WMO_DESCRIPTIONS: dict[int, tuple[str, str]] = {
    0: ("晴朗", "Clear sky"),
    1: ("大致晴朗", "Mainly clear"),
    2: ("局部多云", "Partly cloudy"),
    3: ("阴天", "Overcast"),
    45: ("雾", "Fog"),
    48: ("雾凇", "Depositing rime fog"),
    51: ("小毛毛雨", "Light drizzle"),
    53: ("中等毛毛雨", "Moderate drizzle"),
    55: ("强毛毛雨", "Dense drizzle"),
    56: ("轻微冻毛毛雨", "Light freezing drizzle"),
    57: ("强冻毛毛雨", "Dense freezing drizzle"),
    61: ("小雨", "Slight rain"),
    63: ("中雨", "Moderate rain"),
    65: ("大雨", "Heavy rain"),
    66: ("轻微冻雨", "Light freezing rain"),
    67: ("强冻雨", "Heavy freezing rain"),
    71: ("小雪", "Slight snowfall"),
    73: ("中雪", "Moderate snowfall"),
    75: ("大雪", "Heavy snowfall"),
    77: ("米雪", "Snow grains"),
    80: ("小阵雨", "Slight rain showers"),
    81: ("中等阵雨", "Moderate rain showers"),
    82: ("强阵雨", "Violent rain showers"),
    85: ("小阵雪", "Slight snow showers"),
    86: ("强阵雪", "Heavy snow showers"),
    95: ("雷暴", "Thunderstorm"),
    96: ("雷暴伴小冰雹", "Thunderstorm with slight hail"),
    99: ("雷暴伴强冰雹", "Thunderstorm with heavy hail"),
}


_NEWS_GROUPS: tuple[tuple[str, float, tuple[tuple[str, str], ...]], ...] = (
    (
        "airport_closure",
        0.95,
        (
            (r"\b(?:airport|runway)\s+(?:closure|closed|shut(?:down)?)\b", "airport closure"),
            (r"\bclosed\s+(?:airport|runway)\b", "airport closure"),
            (r"\bground stop\b", "ground stop"),
            (r"\bairport\s+evacuat(?:ed|ion)\b", "airport evacuation"),
        ),
    ),
    (
        "airspace_conflict",
        0.90,
        (
            (r"\bairspace\s+(?:closure|closed|restriction(?:s)?)\b", "airspace closure"),
            (r"\bclosed\s+airspace\b", "airspace closure"),
            (r"\b(?:armed conflict|war zone|hostilities)\b", "armed conflict"),
            (r"\b(?:missile|drone)\s+(?:strike|attack)\b", "missile or drone attack"),
        ),
    ),
    (
        "labor_strike",
        0.82,
        ((r"\b(?:pilot|crew|airport|airline|controller|workers?)?\s*strikes?\b", "strike"),),
    ),
    (
        "extreme_weather",
        0.82,
        (
            (r"\b(?:extreme weather|hurricane|typhoon|cyclone|blizzard)\b", "extreme weather"),
            (r"\b(?:wildfires?|flooding|earthquake|volcanic ash)\b", "natural hazard"),
        ),
    ),
    (
        "security_cyber",
        0.72,
        (
            (r"\b(?:cyberattack|cyber attack|security incident)\b", "security or cyber incident"),
        ),
    ),
    (
        "cancellation_delay",
        0.58,
        (
            (r"\b(?:flights?\s+)?(?:cancelled|canceled|cancellations?)\b", "flight cancellation"),
            (r"\b(?:flights?\s+)?(?:delayed|delays?)\b", "flight delay"),
            (r"\bgrounded\s+flights?\b", "grounded flights"),
        ),
    ),
    (
        "other_disruption",
        0.42,
        ((r"\bdisrupt(?:ion|ions|ed)\b", "disruption"),),
    ),
)


class DetailProvider:
    """Fetch rich provider data with short fresh caches and safe stale fallbacks."""

    def __init__(self, context_provider: ContextProvider) -> None:
        self.context_provider = context_provider
        self._weather_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
        self._aviation_cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._news_cache: dict[tuple[str, ...], tuple[float, NewsDetailSnapshot]] = {}
        self._news_failures: dict[tuple[str, ...], tuple[float, NewsDetailSnapshot]] = {}
        self._lock = threading.Lock()
        self._news_locks = tuple(threading.Lock() for _ in range(16))

    def airport_weather(
        self,
        airport: Airport,
        timezone_name: str,
        target_time: datetime,
        *,
        generated_at: datetime | None = None,
    ) -> AirportWeatherDetail:
        generated_at = _utc(generated_at or datetime.now(UTC))
        target_local = target_time
        target_time = _utc(target_time)
        fallback_reason: BilingualText | None = None
        payload: dict[str, Any] | None = None
        stale = False

        if self.context_provider.external_context_enabled:
            key = (round(airport.latitude, 4), round(airport.longitude, 4))
            payload, stale = self._weather_payload(key, airport)
            if payload is None:
                fallback_reason = BilingualText(
                    zh="实时天气服务暂时不可用；仅显示训练数据的历史风险基线。",
                    en=(
                        "The live weather service is temporarily unavailable; only the "
                        "historical training-risk baseline is shown."
                    ),
                )
        else:
            fallback_reason = BilingualText(
                zh="外部实时数据已停用；仅显示训练数据的历史风险基线。",
                en=(
                    "External live data is disabled; only the historical training-risk "
                    "baseline is shown."
                ),
            )

        current = self._current_observation(payload) if payload else None
        target = self._target_observation(payload, target_time) if payload else None
        hourly = self._hourly_window(payload, target_time) if payload else []
        if self.context_provider.external_context_enabled and airport.icao:
            aviation_reports, aviation_metadata = self._aviation_reports(
                airport.icao,
                target_time,
                generated_at,
            )
        else:
            reason = BilingualText(
                zh=(
                    "该机场缺少可用的 ICAO 代码，无法查询 METAR/TAF。"
                    if not airport.icao
                    else "外部实时数据已停用，未查询 METAR/TAF。"
                ),
                en=(
                    "This airport has no usable ICAO code for a METAR/TAF lookup."
                    if not airport.icao
                    else "External live data is disabled, so METAR/TAF was not queried."
                ),
            )
            aviation_reports = []
            aviation_metadata = ProviderMetadata(
                status="unavailable",
                source="noaa_aviation_weather",
                observed_at=generated_at,
                fallback_reason=reason,
            )

        prior = self.context_provider._weather_prior(  # noqa: SLF001
            target_local,
            airport.latitude,
            generated_at,
        )
        overall_risk = target.risk if target is not None else prior.value
        if target is None and fallback_reason is None:
            fallback_reason = BilingualText(
                zh="目标时刻超出免费预报覆盖范围；风险值采用训练数据历史基线。",
                en=(
                    "The target time is outside the free forecast horizon; the risk value "
                    "uses the historical training baseline."
                ),
            )

        if target is not None:
            status = "historical" if stale else "forecast"
            source = "open_meteo_forecast_stale_cache" if stale else "open_meteo_forecast"
        elif current is not None and abs(target_time - generated_at) <= timedelta(hours=2):
            overall_risk = current.risk
            status = "historical" if stale else "live"
            source = "open_meteo_current_stale_cache" if stale else "open_meteo_current"
        else:
            status = "proxy"
            source = "synthetic_demo_training_average"

        winning_report: AviationWeatherReport | None = None
        applicable_reports = [
            report
            for report in aviation_reports
            if report.product == "TAF"
            or (
                report.product == "METAR"
                and abs(target_time - generated_at) <= timedelta(hours=2)
            )
        ]
        if applicable_reports:
            highest_report = max(applicable_reports, key=lambda report: report.risk)
            if highest_report.risk > overall_risk:
                winning_report = highest_report
                overall_risk = highest_report.risk
                status = "forecast" if highest_report.product == "TAF" else "live"
                source = f"noaa_{highest_report.product.casefold()}"

        hourly_times = [point.time for point in hourly]
        observed_at = current.time if current else generated_at
        valid_from = min(hourly_times) if hourly_times else None
        valid_to = max(hourly_times) if hourly_times else None
        if winning_report is not None:
            observed_at = winning_report.issued_at or generated_at
            valid_from = winning_report.valid_from
            valid_to = winning_report.valid_to
            if target is None:
                fallback_reason = BilingualText(
                    zh=(
                        "Open-Meteo 目标时刻预报不可用；总体风险改用适用于目标时刻的 "
                        f"NOAA {winning_report.product}。"
                    ),
                    en=(
                        "The Open-Meteo target forecast is unavailable; overall risk uses "
                        f"the applicable NOAA {winning_report.product}."
                    ),
                )
        metadata = ProviderMetadata(
            status=status,
            source=source,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            fallback_reason=fallback_reason,
        )
        return AirportWeatherDetail(
            airport_code=airport.iata,
            airport_name=airport.name,
            icao_code=airport.icao or None,
            timezone=timezone_name,
            target_time=target_time,
            current=current,
            target=target,
            hourly=hourly,
            aviation_reports=aviation_reports,
            aviation_metadata=aviation_metadata,
            overall_risk=overall_risk,
            metadata=metadata,
        )

    def news(
        self,
        origin: str,
        destination: str,
        *,
        origin_name: str | None,
        destination_name: str | None,
        generated_at: datetime | None = None,
    ) -> NewsDetailSnapshot:
        generated_at = _utc(generated_at or datetime.now(UTC))
        key = (
            origin,
            destination,
            self.context_provider._news_cache_name(origin_name),  # noqa: SLF001
            self.context_provider._news_cache_name(destination_name),  # noqa: SLF001
        )
        fresh = self._cache_get(
            self._news_cache,
            key,
            NEWS_CACHE_TTL_SECONDS,
            evict=False,
        )
        if fresh is not None:
            return fresh
        recent_failure = self._cache_get(self._news_failures, key, FAILURE_CACHE_TTL_SECONDS)
        if recent_failure is not None:
            return recent_failure

        lock = self._news_locks[hash(key) % len(self._news_locks)]
        with lock:
            fresh = self._cache_get(
                self._news_cache,
                key,
                NEWS_CACHE_TTL_SECONDS,
                evict=False,
            )
            if fresh is not None:
                return fresh
            stale = self._cache_get(
                self._news_cache,
                key,
                STALE_CACHE_TTL_SECONDS,
                evict=False,
            )
            try:
                if not self.context_provider.external_context_enabled:
                    raise LookupError("external context disabled")
                snapshot = self._fetch_news(
                    origin,
                    destination,
                    origin_name,
                    destination_name,
                    generated_at,
                )
            except Exception:
                snapshot = self._stale_news(stale) if stale is not None else self._empty_news(
                    generated_at,
                    disabled=not self.context_provider.external_context_enabled,
                )
                self._cache_set(self._news_failures, key, snapshot)
            else:
                self._cache_set(self._news_cache, key, snapshot)
                with self._lock:
                    self._news_failures.pop(key, None)
            return snapshot

    def _weather_payload(
        self,
        key: tuple[float, float],
        airport: Airport,
    ) -> tuple[dict[str, Any] | None, bool]:
        fresh = self._cache_get(
            self._weather_cache,
            key,
            WEATHER_CACHE_TTL_SECONDS,
            evict=False,
        )
        if fresh is not None:
            return fresh, False
        stale = self._cache_get(
            self._weather_cache,
            key,
            STALE_CACHE_TTL_SECONDS,
            evict=False,
        )
        try:
            payload = self.context_provider._get_json(  # noqa: SLF001
                OPEN_METEO_URL,
                {
                    "latitude": airport.latitude,
                    "longitude": airport.longitude,
                    "current": (
                        "temperature_2m,weather_code,wind_speed_10m,wind_gusts_10m,"
                        "precipitation,visibility"
                    ),
                    "hourly": (
                        "temperature_2m,weather_code,wind_speed_10m,wind_gusts_10m,"
                        "precipitation,precipitation_probability,visibility"
                    ),
                    "timezone": "UTC",
                    "forecast_days": 16,
                },
            )
            if (
                not isinstance(payload, dict)
                or payload.get("error") is True
                or not any(isinstance(payload.get(key), dict) for key in ("current", "hourly"))
            ):
                raise LookupError("Open-Meteo response has no usable weather data")
        except Exception:
            return stale, stale is not None
        self._cache_set(self._weather_cache, key, payload)
        return payload, False

    def _current_observation(self, payload: dict[str, Any]) -> WeatherObservation | None:
        current = payload.get("current")
        if not isinstance(current, dict):
            return None
        time = _parse_datetime(current.get("time"))
        if time is None:
            return None
        return self._observation(
            time=time,
            temperature=current.get("temperature_2m"),
            weather_code=current.get("weather_code"),
            wind=current.get("wind_speed_10m"),
            gust=current.get("wind_gusts_10m"),
            precipitation=current.get("precipitation"),
            precipitation_probability=None,
            visibility=current.get("visibility"),
        )

    def _target_observation(
        self,
        payload: dict[str, Any],
        target_time: datetime,
    ) -> WeatherObservation | None:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            return None
        index_time = self._nearest_hourly(hourly, target_time)
        if index_time is None:
            return None
        index, time = index_time
        return self._observation_from_hourly(hourly, index, time)

    def _hourly_window(
        self,
        payload: dict[str, Any],
        target_time: datetime,
    ) -> list[HourlyWeatherPoint]:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            return []
        times = hourly.get("time")
        if not isinstance(times, list):
            return []
        points: list[HourlyWeatherPoint] = []
        window_start = target_time - timedelta(hours=12)
        window_end = target_time + timedelta(hours=12)
        for index, value in enumerate(times):
            time = _parse_datetime(value)
            if time is None or not window_start <= time <= window_end:
                continue
            observation = self._observation_from_hourly(hourly, index, time)
            points.append(
                HourlyWeatherPoint(
                    time=observation.time,
                    temperature_c=observation.temperature_c,
                    weather_code=observation.weather_code,
                    wind_speed_kmh=observation.wind_speed_kmh,
                    wind_gust_kmh=observation.wind_gust_kmh,
                    precipitation_mm=observation.precipitation_mm,
                    precipitation_probability_pct=(
                        observation.precipitation_probability_pct
                    ),
                    visibility_m=observation.visibility_m,
                    risk=observation.risk,
                )
            )
            if len(points) == 25:
                break
        return points

    @staticmethod
    def _nearest_hourly(
        hourly: dict[str, Any],
        target_time: datetime,
    ) -> tuple[int, datetime] | None:
        times = hourly.get("time")
        if not isinstance(times, list):
            return None
        candidates = [
            (index, parsed)
            for index, value in enumerate(times)
            if (parsed := _parse_datetime(value)) is not None
        ]
        if not candidates:
            return None
        selected = min(candidates, key=lambda item: abs(item[1] - target_time))
        return selected if abs(selected[1] - target_time) <= timedelta(hours=2) else None

    def _observation_from_hourly(
        self,
        hourly: dict[str, Any],
        index: int,
        time: datetime,
    ) -> WeatherObservation:
        return self._observation(
            time=time,
            temperature=self._list_value(hourly, "temperature_2m", index),
            weather_code=self._list_value(hourly, "weather_code", index),
            wind=self._list_value(hourly, "wind_speed_10m", index),
            gust=self._list_value(hourly, "wind_gusts_10m", index),
            precipitation=self._list_value(hourly, "precipitation", index),
            precipitation_probability=self._list_value(
                hourly,
                "precipitation_probability",
                index,
            ),
            visibility=self._list_value(hourly, "visibility", index),
        )

    def _observation(
        self,
        *,
        time: datetime,
        temperature: Any,
        weather_code: Any,
        wind: Any,
        gust: Any,
        precipitation: Any,
        precipitation_probability: Any,
        visibility: Any,
    ) -> WeatherObservation:
        temperature_number = self._number_or_none(temperature)
        code_number = self._integer_or_none(weather_code)
        wind_number = self._nonnegative_number(wind)
        gust_number = self._nonnegative_number(gust)
        precipitation_number = self._nonnegative_number(precipitation)
        probability_number = self._bounded_percent(precipitation_probability)
        visibility_number = self._nonnegative_number(visibility)
        components = self._risk_components(
            code_number,
            wind_number,
            gust_number,
            precipitation_number,
            probability_number,
            visibility_number,
        )
        description = _WMO_DESCRIPTIONS.get(
            code_number if code_number is not None else -1,
            ("未知天气代码", "Unknown weather code"),
        )
        return WeatherObservation(
            time=time,
            temperature_c=temperature_number,
            weather_code=code_number,
            weather_description=BilingualText(zh=description[0], en=description[1]),
            wind_speed_kmh=wind_number,
            wind_gust_kmh=gust_number,
            precipitation_mm=precipitation_number,
            precipitation_probability_pct=probability_number,
            visibility_m=visibility_number,
            risk=max(component.risk for component in components),
            risk_components=components,
        )

    def _risk_components(
        self,
        weather_code: int | None,
        wind: float | None,
        gust: float | None,
        precipitation: float | None,
        probability: float | None,
        visibility: float | None,
    ) -> list[WeatherRiskComponent]:
        precipitation_risk = max(
            _bounded((precipitation or 0) / 10 * 0.7),
            _bounded((probability or 0) / 100 * 0.7),
        )
        return [
            WeatherRiskComponent(
                key="weather_code",
                label=BilingualText(zh="天气现象", en="Weather condition"),
                input_value=float(weather_code) if weather_code is not None else None,
                unit="WMO code",
                risk=self.context_provider._weather_code_risk(weather_code),  # noqa: SLF001
            ),
            WeatherRiskComponent(
                key="wind",
                label=BilingualText(zh="持续风速", en="Sustained wind"),
                input_value=wind,
                unit="km/h",
                risk=_bounded((wind or 0) / 100),
            ),
            WeatherRiskComponent(
                key="gust",
                label=BilingualText(zh="阵风", en="Wind gust"),
                input_value=gust,
                unit="km/h",
                risk=_bounded((gust or 0) / 130),
            ),
            WeatherRiskComponent(
                key="precipitation",
                label=BilingualText(zh="降水", en="Precipitation"),
                input_value=probability if probability is not None else precipitation,
                unit="%" if probability is not None else "mm",
                risk=precipitation_risk,
            ),
            WeatherRiskComponent(
                key="visibility",
                label=BilingualText(zh="能见度", en="Visibility"),
                input_value=visibility,
                unit="m",
                risk=_bounded(
                    (10_000 - (visibility if visibility is not None else 10_000))
                    / 10_000
                ),
            ),
        ]

    def _aviation_reports(
        self,
        icao: str,
        target_time: datetime,
        generated_at: datetime,
    ) -> tuple[list[AviationWeatherReport], ProviderMetadata]:
        reports: list[AviationWeatherReport] = []
        successful_products = 0
        failed_products = 0
        for product, url in (("METAR", NOAA_METAR_URL), ("TAF", NOAA_TAF_URL)):
            key = (url, icao)
            payload = self._cache_get(self._aviation_cache, key, WEATHER_CACHE_TTL_SECONDS)
            fetched = False
            if payload is None:
                try:
                    payload = self.context_provider._get_json(  # noqa: SLF001
                        url,
                        {"ids": icao, "format": "json"},
                    )
                    fetched = True
                except Exception:
                    payload = None
            if payload is None:
                failed_products += 1
                continue
            if isinstance(payload, list):
                rows = payload
            elif (
                isinstance(payload, dict)
                and payload.get("error") is not True
                and isinstance(payload.get("data"), list)
            ):
                rows = payload["data"]
            else:
                failed_products += 1
                continue
            if fetched:
                self._cache_set(self._aviation_cache, key, payload)
            successful_products += 1
            report = self._best_aviation_report(product, rows, target_time, generated_at)
            if report is not None:
                reports.append(report)
        if reports:
            status = "live" if any(report.product == "METAR" for report in reports) else "forecast"
            observed_at = max(
                (report.issued_at for report in reports if report.issued_at is not None),
                default=generated_at,
            )
            valid_from = min(
                (report.valid_from for report in reports if report.valid_from is not None),
                default=None,
            )
            valid_to = max(
                (report.valid_to for report in reports if report.valid_to is not None),
                default=None,
            )
            fallback_reason = (
                BilingualText(
                    zh="部分 NOAA 航空气象产品暂时不可用；仅显示成功返回的报告。",
                    en=(
                        "Some NOAA aviation-weather products are temporarily unavailable; "
                        "only successful reports are shown."
                    ),
                )
                if failed_products
                else None
            )
        elif successful_products:
            status = "neutral"
            observed_at = generated_at
            valid_from = None
            valid_to = None
            fallback_reason = BilingualText(
                zh="NOAA 已响应，但没有适用于该目标时刻的当前 METAR/TAF。",
                en="NOAA responded, but no current METAR/TAF applies to the target time.",
            )
        else:
            status = "unavailable"
            observed_at = generated_at
            valid_from = None
            valid_to = None
            fallback_reason = BilingualText(
                zh="NOAA METAR/TAF 服务暂时不可用。",
                en="The NOAA METAR/TAF service is temporarily unavailable.",
            )
        return reports, ProviderMetadata(
            status=status,
            source="noaa_aviation_weather",
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            fallback_reason=fallback_reason,
        )

    def _best_aviation_report(
        self,
        product: str,
        rows: list[Any],
        target_time: datetime,
        generated_at: datetime,
    ) -> AviationWeatherReport | None:
        candidates: list[tuple[datetime, AviationWeatherReport]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if product == "METAR":
                raw = row.get("rawOb") or row.get("raw_text") or row.get("raw")
                issued = self._first_time(row, "obsTime", "reportTime", "receiptTime")
                valid_from = issued
                valid_to = issued + timedelta(hours=2) if issued else None
            else:
                raw = row.get("rawTAF") or row.get("raw_text") or row.get("raw")
                issued = self._first_time(row, "issueTime", "bulletinTime", "dbPopTime")
                valid_from = self._first_time(row, "validTimeFrom")
                valid_to = self._first_time(row, "validTimeTo")
            raw_text = re.sub(r"\s+", " ", str(raw or "")).strip()[:10_000]
            if not raw_text or issued is None:
                continue
            max_age = timedelta(hours=2 if product == "METAR" else 36)
            if issued > generated_at + timedelta(hours=1) or generated_at - issued > max_age:
                continue
            if product == "TAF" and (
                (valid_from is not None and target_time < valid_from)
                or (valid_to is not None and target_time > valid_to)
            ):
                continue
            structured_rows = [row]
            if product == "TAF" and isinstance(row.get("fcsts"), list):
                overlapping = [
                    forecast
                    for forecast in row["fcsts"]
                    if isinstance(forecast, dict)
                    and self.context_provider._forecast_covers(  # noqa: SLF001
                        forecast,
                        target_time,
                    )
                ]
                if not overlapping:
                    continue
                structured_rows = overlapping
            risk = max(
                self.context_provider._aviation_structured_risk(  # noqa: SLF001
                    structured,
                    weather_text=raw_text,
                )
                for structured in structured_rows
            )
            explanation = self._aviation_explanation(product, risk)
            report = AviationWeatherReport(
                product=product,
                raw_report=raw_text,
                issued_at=issued,
                valid_from=valid_from,
                valid_to=valid_to,
                explanation=explanation,
                risk=risk,
            )
            candidates.append((issued, report))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _aviation_explanation(product: str, risk: float) -> BilingualText:
        if risk >= 0.8:
            zh_level = "检测到严重航空天气信号"
            en_level = "severe aviation-weather signals detected"
        elif risk >= 0.45:
            zh_level = "检测到需要关注的航空天气信号"
            en_level = "elevated aviation-weather signals detected"
        else:
            zh_level = "未从结构化字段中识别出严重信号"
            en_level = "no severe signal identified in structured fields"
        return BilingualText(
            zh=f"{product} 自动保守解读：{zh_level}；不能替代官方飞行气象简报。",
            en=(
                f"Conservative automated {product} interpretation: {en_level}; this does "
                "not replace an official aviation-weather briefing."
            ),
        )

    def _fetch_news(
        self,
        origin: str,
        destination: str,
        origin_name: str | None,
        destination_name: str | None,
        generated_at: datetime,
    ) -> NewsDetailSnapshot:
        route_terms = self.context_provider._gdelt_route_terms(  # noqa: SLF001
            origin,
            destination,
            origin_name,
            destination_name,
        )
        if not route_terms:
            raise LookupError("route terms unavailable")
        query = (
            f"({' OR '.join(route_terms)}) (airport OR airline OR flight) "
            "(strike OR closure OR closed OR conflict OR disruption OR cancellation OR "
            'cancelled OR canceled OR "extreme weather" OR "ground stop" OR hurricane OR '
            "typhoon OR wildfire OR flooding OR earthquake OR cyberattack)"
        )
        try:
            payload = self.context_provider._get_json(  # noqa: SLF001
                GDELT_DOC_URL,
                {
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": 75,
                    "format": "json",
                    "sort": "DateDesc",
                    "timespan": "7d",
                },
                timeout=GDELT_REQUEST_TIMEOUT_SECONDS,
            )
            articles = self._articles_from_doc(payload, generated_at)
            source = "gdelt_doc_2_near_realtime"
            valid_from = generated_at - timedelta(days=7)
        except Exception:
            rss = self.context_provider._get_text(  # noqa: SLF001
                GDELT_GAL_RSS_URL,
                timeout=GDELT_RSS_REQUEST_TIMEOUT_SECONDS,
            )
            articles = self._articles_from_rss(
                rss,
                origin,
                destination,
                origin_name,
                destination_name,
                generated_at,
            )
            source = "gdelt_gal_rss"
            valid_from = min(
                (article.indexed_at for article in articles),
                default=generated_at - timedelta(days=1),
            )
        route_risk = self._aggregate_news_risk(articles)
        return NewsDetailSnapshot(
            articles=tuple(articles),
            route_raw_risk=route_risk,
            metadata=ProviderMetadata(
                status="live",
                source=source,
                observed_at=generated_at,
                valid_from=valid_from,
                valid_to=generated_at,
            ),
            summary=BilingualText(
                zh=f"GDELT 找到 {len(articles)} 篇经验证的航线中断相关新闻。",
                en=f"GDELT returned {len(articles)} validated route-disruption articles.",
            ),
        )

    def _articles_from_doc(
        self,
        payload: Any,
        generated_at: datetime,
    ) -> list[NewsDetailArticle]:
        rows = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise LookupError("GDELT article list missing")
        candidates = [
            self._article_from_values(
                title=row.get("title"),
                url=row.get("url"),
                language=row.get("language"),
                indexed_at=_parse_datetime(row.get("seendate") or row.get("published_at")),
                generated_at=generated_at,
                provider_matched=True,
            )
            for row in rows
            if isinstance(row, dict)
        ]
        return self._deduplicate_articles([article for article in candidates if article])

    def _articles_from_rss(
        self,
        rss: str,
        origin: str,
        destination: str,
        origin_name: str | None,
        destination_name: str | None,
        generated_at: datetime,
    ) -> list[NewsDetailArticle]:
        root = ElementTree.fromstring(rss)
        candidates: list[NewsDetailArticle] = []
        for item in root.findall(".//item"):
            title = item.findtext("title")
            if not self.context_provider._title_mentions_route(  # noqa: SLF001
                str(title or ""),
                origin,
                destination,
                origin_name,
                destination_name,
            ):
                continue
            article = self._article_from_values(
                title=title,
                url=item.findtext("link"),
                language=None,
                indexed_at=self.context_provider._parse_rss_datetime(  # noqa: SLF001
                    item.findtext("pubDate")
                ),
                generated_at=generated_at,
                provider_matched=False,
                max_age=timedelta(days=1),
            )
            if article is not None and article.matched_risk_terms:
                candidates.append(article)
        return self._deduplicate_articles(candidates)

    def _article_from_values(
        self,
        *,
        title: Any,
        url: Any,
        language: Any,
        indexed_at: datetime | None,
        generated_at: datetime,
        provider_matched: bool,
        max_age: timedelta = timedelta(days=7, hours=1),
    ) -> NewsDetailArticle | None:
        clean_title = self.context_provider._clean_news_title(title)  # noqa: SLF001
        canonical = self.context_provider._canonical_news_url(url)  # noqa: SLF001
        if clean_title is None or canonical is None or indexed_at is None:
            return None
        if not generated_at - max_age <= indexed_at <= generated_at + timedelta(minutes=15):
            return None
        _, clean_url, domain = canonical
        category, raw_score, matched_terms = self._classify_news(clean_title)
        if provider_matched and raw_score == 0:
            category = "other_disruption"
            raw_score = 0.18
        recency = self.context_provider._news_recency_factor(  # noqa: SLF001
            indexed_at,
            generated_at,
        )
        clean_language = re.sub(r"[^a-z0-9-]", "", str(language or "").lower())[:16]
        return NewsDetailArticle(
            title=clean_title,
            url=clean_url,
            source=domain,
            language=clean_language or None,
            indexed_at=indexed_at,
            category=category,
            matched_risk_terms=matched_terms,
            raw_score=raw_score,
            recency_factor=recency,
            weighted_score=_bounded(raw_score * recency),
        )

    @staticmethod
    def _classify_news(title: str) -> tuple[str, float, list[str]]:
        normalized = re.sub(r"\s+", " ", title.casefold())
        normalized = re.sub(
            r"\bstrikes?\s+(?:a|the)\s+(?:deal|agreement)\b|"
            r"\b(?:deal|agreement)\s+(?:averts?|ends?)\s+(?:a\s+)?strike\b|"
            r"\bstrike\s+(?:averted|called off|ends?)\b",
            " ",
            normalized,
        )
        matches: list[tuple[str, float, list[str]]] = []
        for category, score, patterns in _NEWS_GROUPS:
            terms = sorted({term for pattern, term in patterns if re.search(pattern, normalized)})
            if terms:
                matches.append((category, score, terms))
        if not matches:
            return "other_disruption", 0.0, []
        category, score, _ = max(matches, key=lambda item: item[1])
        all_terms = sorted({term for _, _, terms in matches for term in terms})[:12]
        return category, score, all_terms

    def _deduplicate_articles(
        self,
        articles: list[NewsDetailArticle],
    ) -> list[NewsDetailArticle]:
        articles.sort(key=lambda article: article.indexed_at, reverse=True)
        result: list[NewsDetailArticle] = []
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        for article in articles:
            canonical = self.context_provider._canonical_news_url(article.url)  # noqa: SLF001
            if canonical is None:
                continue
            url_key = canonical[0]
            title_key = self.context_provider._news_title_key(article.title)  # noqa: SLF001
            if url_key in seen_urls or title_key in seen_titles:
                continue
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            result.append(article)
            if len(result) == 20:
                break
        return result

    @staticmethod
    def _aggregate_news_risk(articles: list[NewsDetailArticle]) -> float:
        high_risk_domains = {
            article.source for article in articles if article.weighted_score >= 0.4
        }
        return _bounded(
            max((article.weighted_score for article in articles), default=0.0)
            + 0.03 * max(0, len(high_risk_domains) - 1)
        )

    @staticmethod
    def departure_attenuation(departure: datetime, generated_at: datetime) -> float:
        lead_hours = max(0.0, (_utc(departure) - _utc(generated_at)).total_seconds() / 3_600)
        if lead_hours <= 72:
            return 1.0
        if lead_hours <= 7 * 24:
            return 0.75
        if lead_hours <= 14 * 24:
            return 0.45
        if lead_hours <= 30 * 24:
            return 0.2
        return 0.1

    @staticmethod
    def _stale_news(snapshot: NewsDetailSnapshot) -> NewsDetailSnapshot:
        return replace(
            snapshot,
            route_raw_risk=_bounded(snapshot.route_raw_risk * 0.5),
            metadata=snapshot.metadata.model_copy(
                update={
                    "status": "historical",
                    "source": f"{snapshot.metadata.source}_stale_cache",
                    "fallback_reason": BilingualText(
                        zh="实时新闻刷新失败；使用最近一次成功缓存，并将风险降低 50%。",
                        en=(
                            "The live news refresh failed; the latest successful cache is "
                            "used with its risk reduced by 50%."
                        ),
                    ),
                }
            ),
            summary=BilingualText(
                zh="实时新闻刷新失败，当前显示降权后的最近成功缓存。",
                en="The live news refresh failed; showing a down-weighted successful cache.",
            ),
        )

    @staticmethod
    def _empty_news(generated_at: datetime, *, disabled: bool) -> NewsDetailSnapshot:
        reason = BilingualText(
            zh=("外部实时数据已停用。" if disabled else "实时新闻服务暂时不可用。"),
            en=(
                "External live data is disabled."
                if disabled
                else "The live news service is temporarily unavailable."
            ),
        )
        return NewsDetailSnapshot(
            articles=(),
            route_raw_risk=0.0,
            metadata=ProviderMetadata(
                status="unavailable",
                source="neutral_fallback",
                observed_at=generated_at,
                fallback_reason=reason,
            ),
            summary=BilingualText(
                zh="未生成或伪造新闻；模型新闻影响保持中性。",
                en="No news was generated or fabricated; the model news effect stays neutral.",
            ),
        )

    def _cache_get(
        self,
        cache: dict[Any, tuple[float, Any]],
        key: Any,
        ttl_seconds: float,
        *,
        evict: bool = True,
    ) -> Any:
        with self._lock:
            entry = cache.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if monotonic() - stored_at <= ttl_seconds:
                return value
            if evict:
                cache.pop(key, None)
            return None

    def _cache_set(self, cache: dict[Any, tuple[float, Any]], key: Any, value: Any) -> None:
        with self._lock:
            if len(cache) >= MAX_CACHE_ENTRIES:
                oldest_key = min(cache, key=lambda item: cache[item][0])
                cache.pop(oldest_key, None)
            cache[key] = (monotonic(), value)

    @staticmethod
    def _list_value(values: dict[str, Any], key: str, index: int) -> Any:
        sequence = values.get(key)
        return sequence[index] if isinstance(sequence, list) and index < len(sequence) else None

    @staticmethod
    def _number_or_none(value: Any) -> float | None:
        return ContextProvider._optional_number(value)  # noqa: SLF001

    @classmethod
    def _nonnegative_number(cls, value: Any) -> float | None:
        number = cls._number_or_none(value)
        return max(0.0, number) if number is not None else None

    @classmethod
    def _bounded_percent(cls, value: Any) -> float | None:
        number = cls._number_or_none(value)
        return max(0.0, min(100.0, number)) if number is not None else None

    @staticmethod
    def _integer_or_none(value: Any) -> int | None:
        number = ContextProvider._optional_number(value)  # noqa: SLF001
        return int(number) if number is not None else None

    @staticmethod
    def _first_time(row: dict[str, Any], *keys: str) -> datetime | None:
        for key in keys:
            parsed = _parse_datetime(row.get(key))
            if parsed is not None:
                return parsed
        return None
