"""Failure-safe, free data providers for flight prediction context.

The providers in this module deliberately treat third-party data as optional.
Every network path has a deterministic model-prior fallback, so a prediction is
still possible when a free service is unavailable or has exhausted its quota.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any
from urllib import parse, request

REQUEST_TIMEOUT_SECONDS = 3.0
CACHE_TTL_SECONDS = 300.0
MAX_CACHE_ENTRIES = 512
MAX_JSON_RESPONSE_BYTES = 5_000_000
NOAA_MAX_LEAD_HOURS = 30
CURRENT_WEATHER_MAX_LEAD_HOURS = 2
CURRENT_WEATHER_MAX_AGE_MINUTES = 90
METAR_MAX_AGE_HOURS = 2
CURRENT_OPERATIONS_MAX_LEAD_HOURS = 6

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NOAA_TAF_URL = "https://aviationweather.gov/api/data/taf"
NOAA_METAR_URL = "https://aviationweather.gov/api/data/metar"
AIRLABS_SCHEDULES_URL = "https://airlabs.co/api/v9/schedules"
AIRLABS_ROUTES_URL = "https://airlabs.co/api/v9/routes"
ADSB_LOL_URL = "https://api.adsb.lol/v2/lat/{latitude}/lon/{longitude}/dist/100"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _bounded(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if re.fullmatch(r"\d{9,13}(?:\.\d+)?", text):
        timestamp = float(text)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    formats = ("%Y%m%dT%H%M%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ")
    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).replace(tzinfo=UTC)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(parsed)


@dataclass(frozen=True, slots=True)
class ContextSignal:
    """A normalized contextual risk signal."""

    value: float
    status: str
    source: str
    observed_at: datetime
    summary_zh: str
    summary_en: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _bounded(self.value))
        object.__setattr__(self, "observed_at", _utc(self.observed_at))


@dataclass(frozen=True, slots=True)
class WeatherSignal(ContextSignal):
    """Weather severity, where 0 is benign and 1 is severe."""


@dataclass(frozen=True, slots=True)
class OperationsSignal(ContextSignal):
    """Origin-airport operating pressure, where 1 is highly congested."""


@dataclass(frozen=True, slots=True)
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.published_at is not None:
            object.__setattr__(self, "published_at", _utc(self.published_at))


@dataclass(frozen=True, slots=True)
class NewsSignal(ContextSignal):
    """Route disruption news risk with source articles, never synthetic text."""

    articles: tuple[NewsArticle, ...] = ()

    def __post_init__(self) -> None:
        ContextSignal.__post_init__(self)
        object.__setattr__(self, "articles", tuple(self.articles[:5]))


@dataclass(frozen=True, slots=True)
class PredictionContext:
    weather: WeatherSignal
    operations: OperationsSignal
    news: NewsSignal
    resolved_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_at", _utc(self.resolved_at))


class _UrllibResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status_code = status
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if not self._body:
            return []
        return json.loads(self._body.decode("utf-8"))


class _UrllibClient:
    """Small sync client with the subset of ``httpx.Client`` used here."""

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> _UrllibResponse:
        query = parse.urlencode(params or {})
        request_url = f"{url}?{query}" if query else url
        http_request = request.Request(request_url, headers=headers or {})
        with request.urlopen(http_request, timeout=timeout) as response:  # noqa: S310
            body = response.read(MAX_JSON_RESPONSE_BYTES + 1)
            if len(body) > MAX_JSON_RESPONSE_BYTES:
                raise RuntimeError("provider response exceeded the 5 MB safety limit")
            return _UrllibResponse(response.status, body)


class ContextProvider:
    """Resolve free weather, operating, and news context without hard failures."""

    def __init__(
        self,
        airlabs_api_key: str | None = None,
        client: Any = None,
        context_priors: dict[str, Any] | None = None,
    ) -> None:
        self.airlabs_api_key = (airlabs_api_key or "").strip() or None
        setting = os.getenv("EXTERNAL_CONTEXT_ENABLED", "1").strip().lower()
        self.external_context_enabled = setting not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.client = client or _UrllibClient()
        self.context_priors = context_priors or {}
        self._cache: dict[tuple[Any, ...], tuple[float, PredictionContext]] = {}
        self._route_cache: dict[tuple[str, str], tuple[float, set[str] | None]] = {}
        self._noaa_cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

    def resolve(
        self,
        origin: str,
        destination: str,
        departure_time: datetime,
        latitude: float,
        longitude: float,
        airport_type: str = "large_airport",
        icao_code: str | None = None,
        origin_name: str | None = None,
        destination_name: str | None = None,
    ) -> PredictionContext:
        """Return a complete context; provider outages never escape this method."""

        resolved_at = datetime.now(UTC)
        try:
            departure_local = departure_time
            departure = _utc(departure_time)
        except (AttributeError, TypeError, ValueError):
            departure_local = resolved_at
            departure = resolved_at
        origin_code = self._airport_code(origin)
        destination_code = self._airport_code(destination)
        icao = self._airport_code(icao_code) if icao_code else None
        try:
            latitude_value = float(latitude)
            longitude_value = float(longitude)
        except (TypeError, ValueError):
            latitude_value, longitude_value = 0.0, 0.0

        key = (
            origin_code,
            destination_code,
            departure.replace(minute=0, second=0, microsecond=0),
            round(latitude_value, 4),
            round(longitude_value, 4),
            airport_type,
            icao,
            origin_name,
            destination_name,
        )
        found, cached = self._cache_get(self._cache, key)
        if found:
            return cached

        if self.external_context_enabled:
            weather = self._weather(
                departure, latitude_value, longitude_value, icao, resolved_at
            )
            operations = self._operations(
                origin_code,
                departure_local,
                latitude_value,
                longitude_value,
                airport_type,
                resolved_at,
            )
            news = self._news(
                origin_code,
                destination_code,
                resolved_at,
                origin_name=origin_name,
                destination_name=destination_name,
            )
        else:
            weather = self._weather_prior(departure_local, latitude_value, resolved_at)
            operations = self._operations_prior(
                origin_code,
                departure_local,
                airport_type,
                resolved_at,
            )
            news = self._neutral_news(resolved_at, "offline_fallback")
        context = PredictionContext(weather, operations, news, resolved_at)
        self._cache_set(self._cache, key, context)
        return context

    def route_airlines(self, origin: str, destination: str) -> set[str] | None:
        """Return AirLabs-confirmed route airlines, or ``None`` without confirmation."""

        if not self.external_context_enabled or not self.airlabs_api_key:
            return None
        origin_code = self._airport_code(origin)
        destination_code = self._airport_code(destination)
        key = (origin_code, destination_code)
        found, cached = self._cache_get(self._route_cache, key)
        if found:
            return set(cached) if cached is not None else None
        try:
            payload = self._get_json(
                AIRLABS_ROUTES_URL,
                {
                    "api_key": self.airlabs_api_key,
                    "dep_iata": origin_code,
                    "arr_iata": destination_code,
                },
            )
            rows = payload.get("response") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                return self._cache_route_failure(key)
            airlines = {
                code
                for row in rows
                if isinstance(row, dict)
                for code in [self._airport_code(row.get("airline_iata"))]
                if 2 <= len(code) <= 3
            }
            self._cache_set(self._route_cache, key, airlines)
            return airlines
        except Exception:
            return self._cache_route_failure(key)

    def _weather(
        self,
        departure: datetime,
        latitude: float,
        longitude: float,
        icao: str | None,
        fetched_at: datetime,
    ) -> WeatherSignal:
        open_meteo: WeatherSignal | None = None
        try:
            open_meteo = self._open_meteo_weather(
                departure,
                latitude,
                longitude,
                fetched_at,
            )
        except Exception:
            pass

        lead = departure - fetched_at
        aviation: WeatherSignal | None = None
        if icao and timedelta(hours=-2) <= lead <= timedelta(hours=NOAA_MAX_LEAD_HOURS):
            aviation = self._noaa_weather(
                icao,
                departure,
                fetched_at,
                include_metar=lead <= timedelta(hours=CURRENT_WEATHER_MAX_LEAD_HOURS),
            )

        if open_meteo and aviation:
            severity = max(open_meteo.value, aviation.value)
            status = "live" if "live" in {open_meteo.status, aviation.status} else "forecast"
            return WeatherSignal(
                severity,
                status,
                f"{open_meteo.source}+{aviation.source}",
                max(open_meteo.observed_at, aviation.observed_at),
                (
                    f"{open_meteo.summary_zh.rstrip('。')}；已结合 NOAA 航空气象，"
                    f"综合风险 {severity:.0%}。"
                ),
                (
                    f"{open_meteo.summary_en.rstrip('.')}; combined with NOAA aviation "
                    f"weather, the overall risk is {severity:.0%}."
                ),
            )
        if open_meteo:
            return open_meteo
        if aviation:
            return aviation
        return self._weather_prior(departure, latitude, fetched_at)

    def _open_meteo_weather(
        self,
        departure: datetime,
        latitude: float,
        longitude: float,
        fetched_at: datetime,
    ) -> WeatherSignal:
        payload = self._get_json(
            OPEN_METEO_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,weather_code,wind_speed_10m,"
                    "wind_gusts_10m,precipitation,visibility"
                ),
                "hourly": (
                    "temperature_2m,weather_code,wind_speed_10m,"
                    "wind_gusts_10m,precipitation_probability,visibility"
                ),
                "timezone": "UTC",
                "forecast_days": 16,
            },
        )
        if not isinstance(payload, dict):
            raise LookupError("Open-Meteo response is not an object")

        lead = departure - fetched_at
        if timedelta(hours=-2) <= lead <= timedelta(
            hours=CURRENT_WEATHER_MAX_LEAD_HOURS
        ):
            current_signal = self._open_meteo_current(payload, fetched_at)
            if current_signal is not None:
                return current_signal

        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise LookupError("Open-Meteo hourly data missing")
        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise LookupError("Open-Meteo forecast times missing")
        parsed_times = [_parse_datetime(value) for value in times]
        choices = [(index, value) for index, value in enumerate(parsed_times) if value]
        if not choices:
            raise LookupError("Open-Meteo forecast times invalid")
        index, forecast_time = min(choices, key=lambda item: abs(item[1] - departure))
        if abs(forecast_time - departure) > timedelta(hours=2):
            raise LookupError("Departure is outside the available forecast horizon")

        weather_code = self._hourly_value(hourly, "weather_code", index, 0)
        wind = self._hourly_value(hourly, "wind_speed_10m", index, 0)
        gust = self._hourly_value(hourly, "wind_gusts_10m", index, wind)
        precipitation = self._hourly_value(
            hourly,
            "precipitation_probability",
            index,
            0,
        )
        visibility = self._hourly_value(hourly, "visibility", index, 10_000)
        temperature = self._hourly_value(hourly, "temperature_2m", index, None)
        severity = self._weather_severity(
            weather_code,
            wind,
            gust,
            precipitation,
            visibility,
            precipitation_is_probability=True,
        )
        temperature_zh, temperature_en = self._temperature_fragments(temperature)
        return WeatherSignal(
            severity,
            "forecast",
            "open_meteo_forecast",
            forecast_time,
            (
                f"出发时段 Open-Meteo 预报风险 {severity:.0%}{temperature_zh}；"
                f"阵风约 {self._number(gust):.0f} km/h。"
            ),
            (
                f"Open-Meteo departure-time forecast risk {severity:.0%}{temperature_en}; "
                f"gusts about {self._number(gust):.0f} km/h."
            ),
        )

    def _open_meteo_current(
        self,
        payload: dict[str, Any],
        fetched_at: datetime,
    ) -> WeatherSignal | None:
        current = payload.get("current")
        if not isinstance(current, dict):
            return None
        observed_at = _parse_datetime(current.get("time"))
        if observed_at is None:
            return None
        age = fetched_at - observed_at
        if not timedelta(minutes=-15) <= age <= timedelta(
            minutes=CURRENT_WEATHER_MAX_AGE_MINUTES
        ):
            return None

        weather_code = self._optional_number(current.get("weather_code"))
        wind = self._optional_number(current.get("wind_speed_10m"))
        gust = self._optional_number(current.get("wind_gusts_10m"))
        if weather_code is None or wind is None or gust is None:
            return None
        precipitation = current.get("precipitation", 0)
        visibility = current.get("visibility", 10_000)
        if visibility is None:
            visibility = 10_000
        temperature = current.get("temperature_2m")
        severity = self._weather_severity(
            weather_code,
            wind,
            gust,
            precipitation,
            visibility,
            precipitation_is_probability=False,
        )
        temperature_zh, temperature_en = self._temperature_fragments(temperature)
        return WeatherSignal(
            severity,
            "live",
            "open_meteo_current_model",
            observed_at,
            (
                f"Open-Meteo 15 分钟模型当前天气风险 {severity:.0%}{temperature_zh}；"
                f"阵风约 {self._number(gust):.0f} km/h。"
            ),
            (
                f"Open-Meteo 15-minute model current-weather risk {severity:.0%}"
                f"{temperature_en}; gusts about {self._number(gust):.0f} km/h."
            ),
        )

    def _noaa_weather(
        self,
        icao: str,
        departure: datetime,
        fetched_at: datetime,
        *,
        include_metar: bool,
    ) -> WeatherSignal | None:
        signals: list[WeatherSignal] = []
        taf = self._noaa_product_signal(NOAA_TAF_URL, icao, departure, fetched_at)
        if taf is not None:
            signals.append(taf)
        if include_metar:
            metar = self._noaa_product_signal(NOAA_METAR_URL, icao, departure, fetched_at)
            if metar is not None:
                signals.append(metar)
        if not signals:
            return None

        severity = max(signal.value for signal in signals)
        status = "live" if any(signal.status == "live" for signal in signals) else "forecast"
        source = "+".join(signal.source for signal in signals)
        observed_at = max(signal.observed_at for signal in signals)
        products_zh = "METAR 机场实况与 TAF 航站预报" if len(signals) > 1 else (
            "METAR 机场实况" if signals[0].source == "noaa_metar" else "TAF 航站预报"
        )
        products_en = "METAR observation and TAF forecast" if len(signals) > 1 else (
            "METAR airport observation"
            if signals[0].source == "noaa_metar"
            else "TAF terminal forecast"
        )
        return WeatherSignal(
            severity,
            status,
            source,
            observed_at,
            f"NOAA {products_zh}风险 {severity:.0%}。",
            f"NOAA {products_en} risk {severity:.0%}.",
        )

    def _noaa_product_signal(
        self,
        url: str,
        icao: str,
        departure: datetime,
        fetched_at: datetime,
    ) -> WeatherSignal | None:
        cache_key = (url, icao)
        found, payload = self._cache_get(self._noaa_cache, cache_key)
        if not found:
            try:
                payload = self._get_json(url, {"ids": icao, "format": "json"})
            except Exception:
                payload = None
            self._cache_set(self._noaa_cache, cache_key, payload)
        rows = payload if isinstance(payload, list) else (
            payload.get("data", []) if isinstance(payload, dict) else []
        )
        if not isinstance(rows, list):
            return None

        candidates: list[tuple[float, datetime]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if url == NOAA_METAR_URL:
                observed_at = self._first_datetime(
                    row,
                    "obsTime",
                    "reportTime",
                    "receiptTime",
                )
                if observed_at is None:
                    continue
                age = fetched_at - observed_at
                if not timedelta(minutes=-15) <= age <= timedelta(hours=METAR_MAX_AGE_HOURS):
                    continue
                raw = row.get("rawOb") or row.get("raw_text") or row.get("raw") or ""
                risk = self._aviation_structured_risk(row, weather_text=str(raw))
            else:
                observed_at = self._first_datetime(
                    row,
                    "issueTime",
                    "bulletinTime",
                    "dbPopTime",
                )
                if observed_at is None or fetched_at - observed_at > timedelta(hours=24):
                    continue
                if observed_at - fetched_at > timedelta(hours=1):
                    continue
                valid_from = self._first_datetime(row, "validTimeFrom")
                valid_to = self._first_datetime(row, "validTimeTo")
                if valid_from is not None and departure < valid_from:
                    continue
                if valid_to is not None and departure > valid_to:
                    continue
                forecasts = row.get("fcsts")
                if not isinstance(forecasts, list):
                    continue
                overlapping = [
                    forecast
                    for forecast in forecasts
                    if isinstance(forecast, dict)
                    and self._forecast_covers(forecast, departure)
                ]
                if not overlapping:
                    continue
                risk = max(self._aviation_structured_risk(forecast) for forecast in overlapping)
            candidates.append((risk, observed_at))
        if not candidates:
            return None

        risk, observed_at = max(candidates, key=lambda candidate: candidate[0])
        source = "noaa_metar" if url == NOAA_METAR_URL else "noaa_taf"
        status = "live" if source == "noaa_metar" else "forecast"
        return WeatherSignal(risk, status, source, observed_at, "", "")

    def _operations(
        self,
        origin: str,
        departure: datetime,
        latitude: float,
        longitude: float,
        airport_type: str,
        fetched_at: datetime,
    ) -> OperationsSignal:
        lead = departure - fetched_at
        if not timedelta(hours=-2) <= lead <= timedelta(
            hours=CURRENT_OPERATIONS_MAX_LEAD_HOURS
        ):
            return self._operations_prior(origin, departure, airport_type, fetched_at)

        if self.airlabs_api_key:
            try:
                payload = self._get_json(
                    AIRLABS_SCHEDULES_URL,
                    {"api_key": self.airlabs_api_key, "dep_iata": origin},
                )
                rows = payload.get("response") if isinstance(payload, dict) else None
                if isinstance(rows, list) and rows:
                    delayed = 0
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        status = str(row.get("status", "")).lower()
                        delays = (row.get("dep_delayed", 0), row.get("arr_delayed", 0))
                        has_delay = any(self._number(value) >= 15 for value in delays)
                        delayed += int(has_delay or status in {"delayed", "cancelled"})
                    delayed_fraction = delayed / len(rows)
                    pressure = _bounded(0.55 * min(len(rows) / 60, 1) + 0.45 * delayed_fraction)
                    return OperationsSignal(
                        pressure,
                        "live",
                        "airlabs_schedules",
                        fetched_at,
                        f"AirLabs 返回 {len(rows)} 个出港班次；运行压力 {pressure:.0%}。",
                        (
                            f"AirLabs returned {len(rows)} departures; "
                            f"operating pressure {pressure:.0%}."
                        ),
                    )
            except Exception:
                pass

        try:
            url = ADSB_LOL_URL.format(
                latitude=round(latitude, 4), longitude=round(longitude, 4)
            )
            payload = self._get_json(url, {})
            aircraft = payload.get("ac") if isinstance(payload, dict) else None
            if not isinstance(aircraft, list):
                raise LookupError("ADSB.lol aircraft list missing")
            denominator = {
                "large_airport": 75,
                "medium_airport": 35,
                "small_airport": 15,
            }.get(airport_type, 35)
            pressure = _bounded(0.05 + 0.95 * min(len(aircraft) / denominator, 1))
            return OperationsSignal(
                pressure,
                "proxy",
                "adsb_lol",
                fetched_at,
                f"机场周边发现 {len(aircraft)} 架飞机；以航班密度估算拥堵 {pressure:.0%}。",
                (
                    f"{len(aircraft)} aircraft detected near the airport; "
                    f"traffic-density congestion proxy {pressure:.0%}."
                ),
            )
        except Exception:
            return self._operations_prior(origin, departure, airport_type, fetched_at)

    def _news(
        self,
        origin: str,
        destination: str,
        fetched_at: datetime,
        *,
        origin_name: str | None = None,
        destination_name: str | None = None,
    ) -> NewsSignal:
        route_terms = [origin, destination]
        route_terms.extend(
            f'"{name}"'
            for name in (origin_name, destination_name)
            if name and len(name.strip()) >= 4
        )
        query = (
            f"({' OR '.join(route_terms)}) (airport OR airline OR flight) "
            '(strike OR closure OR conflict OR "extreme weather" OR disruption OR cancellation)'
        )
        try:
            payload = self._get_json(
                GDELT_DOC_URL,
                {
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": 25,
                    "format": "json",
                    "sort": "HybridRel",
                    "timespan": "7d",
                },
            )
            rows = payload.get("articles") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise LookupError("GDELT article list missing")
            scored: list[tuple[float, NewsArticle]] = []
            seen_urls: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "").strip()
                url = str(row.get("url") or "").strip()
                if not title or not url:
                    continue
                normalized_url = url.split("#", 1)[0]
                if normalized_url in seen_urls:
                    continue
                score = self._news_text_risk(title)
                if score <= 0:
                    continue
                published_at = _parse_datetime(row.get("seendate") or row.get("published_at"))
                if published_at and published_at < fetched_at - timedelta(days=7, hours=1):
                    continue
                article = NewsArticle(
                    title,
                    url,
                    str(row.get("domain") or row.get("source") or "GDELT"),
                    published_at,
                )
                scored.append((score, article))
                seen_urls.add(normalized_url)
            scored.sort(key=lambda item: item[0], reverse=True)
            articles = tuple(article for _, article in scored[:5])
            risk = _bounded(
                max((score for score, _ in scored), default=0)
                + 0.03 * max(0, len(scored) - 1)
            )
            if articles:
                summary_zh = f"近 7 天发现 {len(articles)} 条相关中断新闻；风险 {risk:.0%}。"
                summary_en = (
                    f"Found {len(articles)} relevant disruption articles in 7 days; "
                    f"risk {risk:.0%}."
                )
            else:
                summary_zh = "近 7 天未发现可确认的航线中断新闻。"
                summary_en = "No confirmed route-disruption articles found in the last 7 days."
            return NewsSignal(
                risk,
                "live",
                "gdelt_doc_2",
                fetched_at,
                summary_zh,
                summary_en,
                articles,
            )
        except Exception:
            return self._neutral_news(fetched_at, "neutral_fallback")

    @staticmethod
    def _neutral_news(fetched_at: datetime, source: str) -> NewsSignal:
        return NewsSignal(
            0,
            "neutral",
            source,
            fetched_at,
            "新闻服务不可用；使用中性新闻影响，不生成新闻。",
            "News service unavailable; using a neutral news effect with no generated articles.",
            (),
        )

    def _weather_prior(
        self, departure: datetime, latitude: float, fetched_at: datetime
    ) -> WeatherSignal:
        weather_by_month = self.context_priors.get("weather_by_month")
        if isinstance(weather_by_month, dict):
            candidate = weather_by_month.get(str(departure.month))
            if candidate is None:
                candidate = self.context_priors.get("weather_global")
            if candidate is not None:
                risk = _bounded(candidate)
                source = str(self.context_priors.get("source") or "training_average")
                return WeatherSignal(
                    risk,
                    "proxy",
                    source,
                    fetched_at,
                    f"实时天气不可用；采用训练集同月平均值 {risk:.0%}（来源：{source}）。",
                    (
                        "Live weather unavailable; using the same-month training average "
                        f"{risk:.0%} (source: {source})."
                    ),
                )

        latitude = abs(latitude)
        winter = departure.month in ({12, 1, 2} if departure.month else set())
        tropical_storm_season = departure.month in {6, 7, 8, 9, 10, 11}
        risk = 0.18
        if latitude >= 40 and winter:
            risk = 0.42
        elif 10 <= latitude < 35 and tropical_storm_season:
            risk = 0.3
        elif latitude >= 55:
            risk = 0.28
        return WeatherSignal(
            risk,
            "proxy",
            "synthetic_model_prior",
            fetched_at,
            f"实时天气不可用；采用合成演示模型的季节先验 {risk:.0%}。",
            f"Live weather unavailable; synthetic-demo seasonal prior {risk:.0%}.",
        )

    def _operations_prior(
        self,
        origin: str,
        departure: datetime,
        airport_type: str,
        fetched_at: datetime,
    ) -> OperationsSignal:
        operations_by_origin = self.context_priors.get("operations_by_origin")
        if isinstance(operations_by_origin, dict):
            candidate = operations_by_origin.get(origin)
            if candidate is None:
                candidate = self.context_priors.get("operations_global")
            if candidate is not None:
                risk = _bounded(candidate)
                source = str(self.context_priors.get("source") or "training_average")
                return OperationsSignal(
                    risk,
                    "proxy",
                    source,
                    fetched_at,
                    f"当前运行数据不适用于该时刻；采用出发机场训练集平均值 {risk:.0%}。",
                    (
                        "Current operations do not apply to this departure; using the "
                        f"origin-airport training average {risk:.0%}."
                    ),
                )

        risk = {
            "large_airport": 0.52,
            "medium_airport": 0.36,
            "small_airport": 0.2,
            "heliport": 0.12,
        }.get(airport_type, 0.36)
        if departure.hour in {*range(7, 11), *range(16, 21)}:
            risk += 0.12
        elif departure.hour in {*range(0, 6)}:
            risk -= 0.08
        risk = _bounded(risk)
        return OperationsSignal(
            risk,
            "proxy",
            "synthetic_model_prior",
            fetched_at,
            f"当前运行数据不适用于该出发时刻；采用合成演示模型先验 {risk:.0%}。",
            (
                "Current airport operations do not apply to this departure; using the "
                f"synthetic-demo model prior {risk:.0%}."
            ),
        )

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        response = self.client.get(
            url,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "flight-forecast-lab/0.1 (context data client)",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        return response.json()

    @staticmethod
    def _hourly_value(hourly: dict[str, Any], field: str, index: int, default: Any) -> Any:
        values = hourly.get(field)
        if not isinstance(values, list) or index >= len(values) or values[index] is None:
            return default
        return values[index]

    @classmethod
    def _weather_severity(
        cls,
        weather_code: Any,
        wind: Any,
        gust: Any,
        precipitation: Any,
        visibility: Any,
        *,
        precipitation_is_probability: bool,
    ) -> float:
        precipitation_value = cls._number(precipitation)
        if precipitation_is_probability:
            precipitation_risk = _bounded(precipitation_value / 100 * 0.7)
        else:
            precipitation_risk = _bounded(precipitation_value / 10 * 0.7)
        return max(
            cls._weather_code_risk(weather_code),
            _bounded(cls._number(wind) / 100),
            _bounded(cls._number(gust) / 130),
            precipitation_risk,
            _bounded((10_000 - cls._number(visibility)) / 10_000),
        )

    @classmethod
    def _temperature_fragments(cls, value: Any) -> tuple[str, str]:
        if value is None:
            return "", ""
        try:
            temperature = float(value)
        except (TypeError, ValueError):
            return "", ""
        return f"；气温约 {temperature:.0f}°C", f"; temperature about {temperature:.0f}°C"

    @staticmethod
    def _first_datetime(row: dict[str, Any], *fields: str) -> datetime | None:
        for field in fields:
            parsed = _parse_datetime(row.get(field))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _weather_code_risk(code: Any) -> float:
        code_number = int(ContextProvider._number(code))
        if code_number in {95, 96, 99}:
            return 0.92
        if code_number in {71, 73, 75, 77, 85, 86}:
            return 0.72
        if code_number in {65, 67, 82}:
            return 0.68
        if code_number in {45, 48, 55, 57, 63, 66, 81}:
            return 0.5
        if code_number in {51, 53, 56, 61, 80}:
            return 0.32
        return 0.08 if code_number in {1, 2, 3} else 0.03

    @staticmethod
    def _forecast_covers(forecast: dict[str, Any], departure: datetime) -> bool:
        time_from = _parse_datetime(forecast.get("timeFrom"))
        time_to = _parse_datetime(forecast.get("timeTo"))
        return bool(time_from and time_to and time_from <= departure < time_to)

    @classmethod
    def _aviation_structured_risk(
        cls,
        row: dict[str, Any],
        *,
        weather_text: str = "",
    ) -> float:
        text = " ".join(
            value
            for value in (
                weather_text,
                str(row.get("wxString") or ""),
                str(row.get("notDecoded") or ""),
            )
            if value
        )
        risks = [cls._aviation_weather_text_risk(text.upper())]

        wind = cls._optional_number(row.get("wspd"))
        gust = cls._optional_number(row.get("wgst"))
        if wind is not None:
            risks.append(_bounded(wind * 1.852 / 100))
        if gust is not None:
            risks.append(_bounded(gust * 1.852 / 130))

        visibility = cls._visibility_meters(row.get("visib"))
        if visibility is not None:
            risks.append(_bounded((10_000 - visibility) / 10_000))

        flight_category = str(row.get("fltCat") or "").upper()
        risks.append({"LIFR": 0.95, "IFR": 0.75, "MVFR": 0.45}.get(flight_category, 0.0))

        ceiling_values: list[float] = []
        vertical_visibility = cls._optional_number(row.get("vertVis"))
        if vertical_visibility is not None:
            ceiling_values.append(vertical_visibility)
        clouds = row.get("clouds")
        if isinstance(clouds, list):
            for cloud in clouds:
                if not isinstance(cloud, dict):
                    continue
                if str(cloud.get("cover") or "").upper() not in {"BKN", "OVC", "VV"}:
                    continue
                base = cls._optional_number(cloud.get("base"))
                if base is not None:
                    ceiling_values.append(base)
        if ceiling_values:
            ceiling = min(ceiling_values)
            if ceiling <= 200:
                risks.append(0.95)
            elif ceiling <= 500:
                risks.append(0.8)
            elif ceiling <= 1_000:
                risks.append(0.6)
            elif ceiling <= 3_000:
                risks.append(0.3)
        return max(risks)

    @staticmethod
    def _visibility_meters(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip().upper().removesuffix("SM")
        text = text.replace("+", "").lstrip("PM")
        if not text:
            return None
        miles = 0.0
        try:
            for part in text.split():
                if "/" in part:
                    numerator, denominator = part.split("/", maxsplit=1)
                    miles += float(numerator) / float(denominator)
                else:
                    miles += float(part)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return max(0.0, miles * 1_609.344)

    @staticmethod
    def _aviation_weather_text_risk(text: str) -> float:
        if any(term in text for term in ("+TS", "TSRA", "FZRA", "+SN", "VA ")):
            return 0.95
        if any(term in text for term in (" TS", "SN", "FG", "+RA", "SQ")):
            return 0.76
        if any(term in text for term in ("-SN", "RA", "BR", "FZFG")):
            return 0.48
        return 0.08

    @staticmethod
    def _news_text_risk(text: str) -> float:
        normalized = re.sub(r"\s+", " ", text.lower())
        groups = (
            (
                0.95,
                (
                    "airspace closure",
                    "airport closure",
                    "airport closed",
                    "closed airport",
                    "volcanic ash",
                    "missile strike",
                ),
            ),
            (0.9, ("armed conflict", "war zone", "hostilities", "hurricane", "typhoon")),
            (0.82, ("strike", "cyclone", "blizzard", "extreme weather", "grounded")),
            (0.58, ("mass cancellation", "flight cancellation", "cancelled flights")),
            (0.42, ("disruption", "delays", "flight delay", "cancellation")),
        )
        return max(
            (weight for weight, terms in groups if any(term in normalized for term in terms)),
            default=0.0,
        )

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _airport_code(value: Any) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())[:4]

    def _cache_get(self, cache: dict[Any, tuple[float, Any]], key: Any) -> tuple[bool, Any]:
        with self._cache_lock:
            entry = cache.get(key)
            if not entry:
                return False, None
            stored_at, value = entry
            if monotonic() - stored_at > CACHE_TTL_SECONDS:
                cache.pop(key, None)
                return False, None
            return True, value

    def _cache_set(self, cache: dict[Any, tuple[float, Any]], key: Any, value: Any) -> None:
        with self._cache_lock:
            if key not in cache and len(cache) >= MAX_CACHE_ENTRIES:
                oldest_key = min(cache, key=lambda item: cache[item][0])
                cache.pop(oldest_key, None)
            cache[key] = (monotonic(), value)

    def _cache_route_failure(self, key: tuple[str, str]) -> None:
        self._cache_set(self._route_cache, key, None)
        return None
