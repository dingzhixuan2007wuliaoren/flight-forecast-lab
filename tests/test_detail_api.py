from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

import flight_forecaster.api as api_module
from flight_forecaster.api import app
from flight_forecaster.context import (
    GDELT_DOC_URL,
    GDELT_GAL_RSS_URL,
    GDELT_RSS_REQUEST_TIMEOUT_SECONDS,
    NOAA_METAR_URL,
    NOAA_TAF_URL,
    OPEN_METEO_URL,
    ContextProvider,
)
from flight_forecaster.details import (
    NEWS_CACHE_TTL_SECONDS,
    WEATHER_CACHE_TTL_SECONDS,
    DetailProvider,
)
from flight_forecaster.route_info import Airport
from flight_forecaster.service import PredictionService


class FakeResponse:
    def __init__(self, payload: Any = None, *, text: str = "") -> None:
        self.payload = payload
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class FakeClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        del headers
        self.calls.append((url, params, timeout))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def _hourly_payload(start: datetime, hours: int = 48) -> dict[str, Any]:
    times = [(start + timedelta(hours=index)).isoformat() for index in range(hours)]
    return {
        "current": {
            "time": start.isoformat(),
            "temperature_2m": 21,
            "weather_code": 2,
            "wind_speed_10m": 18,
            "wind_gusts_10m": 30,
            "precipitation": 0,
            "visibility": 10_000,
        },
        "hourly": {
            "time": times,
            "temperature_2m": [20 + index / 10 for index in range(hours)],
            "weather_code": [95 if index == 20 else 3 for index in range(hours)],
            "wind_speed_10m": [25] * hours,
            "wind_gusts_10m": [45] * hours,
            "precipitation": [0.4] * hours,
            "precipitation_probability": [60] * hours,
            "visibility": [8_000] * hours,
        },
    }


def test_weather_detail_has_current_target_hourly_and_raw_noaa(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    target = generated_at + timedelta(hours=20)
    fake_client = FakeClient(
        {
            OPEN_METEO_URL: FakeResponse(_hourly_payload(generated_at)),
            NOAA_METAR_URL: FakeResponse(
                [
                    {
                        "rawOb": "KJFK 141151Z 18015G25KT 10SM SCT020 24/18 A2992",
                        "obsTime": (generated_at - timedelta(minutes=9)).timestamp(),
                        "wspd": 15,
                        "wgst": 25,
                        "visib": "10",
                        "fltCat": "VFR",
                    }
                ]
            ),
            NOAA_TAF_URL: FakeResponse(
                [
                    {
                        "rawTAF": "TAF KJFK 141100Z 1412/1518 18015G25KT P6SM SCT020",
                        "issueTime": (generated_at - timedelta(hours=1)).timestamp(),
                        "validTimeFrom": generated_at.timestamp(),
                        "validTimeTo": (generated_at + timedelta(hours=30)).timestamp(),
                        "fcsts": [
                            {
                                "timeFrom": generated_at.timestamp(),
                                "timeTo": (generated_at + timedelta(hours=30)).timestamp(),
                                "wspd": 15,
                                "wgst": 25,
                                "visib": "6+",
                            }
                        ],
                    }
                ]
            ),
        }
    )
    detail_provider = DetailProvider(ContextProvider(client=fake_client))
    airport = Airport(
        "JFK",
        "KJFK",
        "John F. Kennedy International",
        "large_airport",
        "US",
        40.6413,
        -73.7781,
    )

    detail = detail_provider.airport_weather(
        airport,
        "America/New_York",
        target,
        generated_at=generated_at,
    )

    assert detail.current is not None
    assert detail.target is not None
    assert detail.target.weather_code == 95
    assert detail.target.risk == 0.92
    assert len(detail.hourly) == 25
    assert {report.product for report in detail.aviation_reports} == {"METAR", "TAF"}
    assert all(report.raw_report for report in detail.aviation_reports)
    assert detail.metadata.source == "open_meteo_forecast"
    assert [call[0] for call in fake_client.calls].count(OPEN_METEO_URL) == 1


def test_news_detail_validates_deduplicates_scores_and_caches(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    payload = {
        "articles": [
            {
                "title": "JFK airport closed after extreme weather",
                "url": "https://example.com/story?utm_source=test",
                "seendate": (generated_at - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ"),
                "language": "English",
            },
            {
                "title": "JFK airport closed after extreme weather",
                "url": "https://example.com/story",
                "seendate": (generated_at - timedelta(hours=2)).strftime("%Y%m%dT%H%M%SZ"),
            },
            {
                "title": "Flights delayed at LAX",
                "url": "https://news.example.net/delay",
                "seendate": (generated_at - timedelta(days=2)).strftime("%Y%m%dT%H%M%SZ"),
                "language": "Spanish",
            },
            {
                "title": "Unsafe URL is rejected",
                "url": "javascript:alert(1)",
                "seendate": generated_at.strftime("%Y%m%dT%H%M%SZ"),
            },
            {
                "title": "Old article is rejected",
                "url": "https://old.example.org/story",
                "seendate": (generated_at - timedelta(days=9)).strftime("%Y%m%dT%H%M%SZ"),
            },
        ]
    }
    fake_client = FakeClient({GDELT_DOC_URL: FakeResponse(payload)})
    detail_provider = DetailProvider(ContextProvider(client=fake_client))

    first = detail_provider.news(
        "JFK",
        "LAX",
        origin_name="John F. Kennedy International",
        destination_name="Los Angeles International",
        generated_at=generated_at,
    )
    second = detail_provider.news(
        "JFK",
        "LAX",
        origin_name="John F. Kennedy International",
        destination_name="Los Angeles International",
        generated_at=generated_at + timedelta(minutes=1),
    )

    assert len(first.articles) == 2
    assert first.articles[0].category == "airport_closure"
    assert first.articles[0].raw_score == 0.95
    assert first.articles[1].category == "cancellation_delay"
    assert first.route_raw_risk == 0.98
    assert second == first
    assert len(fake_client.calls) == 1
    url, params, timeout = fake_client.calls[0]
    assert url == GDELT_DOC_URL
    assert params["sort"] == "DateDesc"
    assert params["timespan"] == "7d"
    assert timeout == 15.0


def test_news_detail_falls_back_to_official_rss(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    pub_date = (generated_at - timedelta(minutes=5)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    rss = f"""
        <rss><channel><item>
          <title>JFK flight cancellations disrupt passengers</title>
          <link>https://rss.example.com/jfk</link>
          <pubDate>{pub_date}</pubDate>
        </item></channel></rss>
    """
    fake_client = FakeClient(
        {
            GDELT_DOC_URL: RuntimeError("rate limited"),
            GDELT_GAL_RSS_URL: FakeResponse(text=rss),
        }
    )
    detail_provider = DetailProvider(ContextProvider(client=fake_client))

    result = detail_provider.news(
        "JFK",
        "LAX",
        origin_name="John F. Kennedy International",
        destination_name="Los Angeles International",
        generated_at=generated_at,
    )

    assert result.metadata.source == "gdelt_gal_rss"
    assert len(result.articles) == 1
    assert result.articles[0].category == "cancellation_delay"
    assert fake_client.calls[-1][2] == GDELT_RSS_REQUEST_TIMEOUT_SECONDS


def test_detail_provider_uses_stale_weather_and_news_after_refresh_failure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    airport = Airport(
        "JFK",
        "",
        "John F. Kennedy International",
        "large_airport",
        "US",
        40.6413,
        -73.7781,
    )
    news_payload = {
        "articles": [
            {
                "title": "JFK airport closed after extreme weather",
                "url": "https://example.com/stale-story",
                "seendate": generated_at.strftime("%Y%m%dT%H%M%SZ"),
            }
        ]
    }
    fake_client = FakeClient(
        {
            OPEN_METEO_URL: FakeResponse(_hourly_payload(generated_at)),
            GDELT_DOC_URL: FakeResponse(news_payload),
        }
    )
    provider = DetailProvider(ContextProvider(client=fake_client))
    target = generated_at + timedelta(hours=20)

    live_weather = provider.airport_weather(
        airport,
        "America/New_York",
        target,
        generated_at=generated_at,
    )
    live_news = provider.news(
        "JFK",
        "LAX",
        origin_name=airport.name,
        destination_name="Los Angeles International",
        generated_at=generated_at,
    )
    weather_key = next(iter(provider._weather_cache))  # noqa: SLF001
    _, weather_payload = provider._weather_cache[weather_key]  # noqa: SLF001
    provider._weather_cache[weather_key] = (  # noqa: SLF001
        monotonic() - WEATHER_CACHE_TTL_SECONDS - 1,
        weather_payload,
    )
    news_key = next(iter(provider._news_cache))  # noqa: SLF001
    _, news_snapshot = provider._news_cache[news_key]  # noqa: SLF001
    provider._news_cache[news_key] = (  # noqa: SLF001
        monotonic() - NEWS_CACHE_TTL_SECONDS - 1,
        news_snapshot,
    )
    fake_client.responses[OPEN_METEO_URL] = RuntimeError("weather unavailable")
    fake_client.responses[GDELT_DOC_URL] = RuntimeError("news unavailable")
    fake_client.responses[GDELT_GAL_RSS_URL] = RuntimeError("rss unavailable")

    stale_weather = provider.airport_weather(
        airport,
        "America/New_York",
        target,
        generated_at=generated_at + timedelta(minutes=16),
    )
    stale_news = provider.news(
        "JFK",
        "LAX",
        origin_name=airport.name,
        destination_name="Los Angeles International",
        generated_at=generated_at + timedelta(minutes=16),
    )

    assert live_weather.metadata.status == "forecast"
    assert stale_weather.metadata.status == "historical"
    assert stale_weather.metadata.source == "open_meteo_forecast_stale_cache"
    assert live_news.metadata.status == "live"
    assert stale_news.metadata.status == "historical"
    assert stale_news.route_raw_risk == pytest.approx(live_news.route_raw_risk * 0.5)


def test_weather_detail_uses_applicable_taf_and_hides_expired_taf(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    target = generated_at + timedelta(hours=4)
    hourly = _hourly_payload(generated_at, hours=24 * 7)
    hourly["hourly"].update(
        {
            "weather_code": [0] * (24 * 7),
            "wind_speed_10m": [5] * (24 * 7),
            "wind_gusts_10m": [8] * (24 * 7),
            "precipitation": [0] * (24 * 7),
            "precipitation_probability": [0] * (24 * 7),
            "visibility": [10_000] * (24 * 7),
        }
    )
    taf_row = {
        "rawTAF": "TAF KJFK 141100Z 1412/1518 18015G25KT P6SM +TSRA",
        "issueTime": generated_at.timestamp(),
        "validTimeFrom": generated_at.timestamp(),
        "validTimeTo": (generated_at + timedelta(hours=30)).timestamp(),
        "fcsts": [
            {
                "timeFrom": generated_at.timestamp(),
                "timeTo": (generated_at + timedelta(hours=30)).timestamp(),
                "wxString": "+TSRA",
            }
        ],
    }
    fake_client = FakeClient(
        {
            OPEN_METEO_URL: FakeResponse(hourly),
            NOAA_METAR_URL: FakeResponse([]),
            NOAA_TAF_URL: FakeResponse([taf_row]),
        }
    )
    provider = DetailProvider(ContextProvider(client=fake_client))
    airport = Airport(
        "JFK",
        "KJFK",
        "John F. Kennedy International",
        "large_airport",
        "US",
        40.6413,
        -73.7781,
    )

    applicable = provider.airport_weather(
        airport,
        "America/New_York",
        target,
        generated_at=generated_at,
    )
    expired = provider.airport_weather(
        airport,
        "America/New_York",
        generated_at + timedelta(days=5),
        generated_at=generated_at,
    )

    assert applicable.overall_risk == pytest.approx(0.95)
    assert applicable.metadata.source == "noaa_taf"
    assert applicable.metadata.valid_to == datetime(2026, 7, 15, 18, tzinfo=UTC)
    assert [report.product for report in applicable.aviation_reports] == ["TAF"]
    assert all(report.product != "TAF" for report in expired.aviation_reports)


def test_weather_detail_uses_noaa_metadata_when_open_meteo_fails(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    target = generated_at + timedelta(hours=4)
    taf_row = {
        "rawTAF": "TAF KJFK 141100Z 1412/1518 P6SM +TSRA",
        "issueTime": generated_at.timestamp(),
        "validTimeFrom": generated_at.timestamp(),
        "validTimeTo": (generated_at + timedelta(hours=30)).timestamp(),
        "fcsts": [
            {
                "timeFrom": generated_at.timestamp(),
                "timeTo": (generated_at + timedelta(hours=30)).timestamp(),
                "wxString": "+TSRA",
            }
        ],
    }
    fake_client = FakeClient(
        {
            OPEN_METEO_URL: RuntimeError("weather unavailable"),
            NOAA_METAR_URL: FakeResponse([]),
            NOAA_TAF_URL: FakeResponse([taf_row]),
        }
    )
    provider = DetailProvider(ContextProvider(client=fake_client))
    airport = Airport(
        "JFK",
        "KJFK",
        "John F. Kennedy International",
        "large_airport",
        "US",
        40.6413,
        -73.7781,
    )

    detail = provider.airport_weather(
        airport,
        "America/New_York",
        target,
        generated_at=generated_at,
    )

    assert detail.overall_risk == pytest.approx(0.95)
    assert detail.metadata.source == "noaa_taf"
    assert detail.metadata.observed_at == generated_at
    assert detail.metadata.valid_to == generated_at + timedelta(hours=30)
    assert detail.metadata.fallback_reason is not None
    assert "overall risk uses" in detail.metadata.fallback_reason.en


def test_zero_visibility_is_maximum_visibility_risk() -> None:
    provider = DetailProvider(ContextProvider())

    observation = provider._observation(  # noqa: SLF001
        time=datetime(2026, 7, 14, 12, tzinfo=UTC),
        temperature=20,
        weather_code=0,
        wind=0,
        gust=0,
        precipitation=0,
        precipitation_probability=0,
        visibility=0,
    )

    visibility = next(
        component for component in observation.risk_components if component.key == "visibility"
    )
    assert visibility.risk == 1
    assert observation.risk == 1


def test_weather_prior_uses_airport_local_month_at_utc_boundary(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")

    class CapturingProvider(ContextProvider):
        prior_departure: datetime | None = None

        def _weather_prior(
            self,
            departure: datetime,
            latitude: float,
            fetched_at: datetime,
        ):
            self.prior_departure = departure
            return super()._weather_prior(departure, latitude, fetched_at)

    context_provider = CapturingProvider()
    provider = DetailProvider(context_provider)
    airport = Airport(
        "LAX",
        "KLAX",
        "Los Angeles International",
        "large_airport",
        "US",
        33.9416,
        -118.4085,
    )
    local_target = datetime(2026, 9, 30, 20, tzinfo=ZoneInfo("America/Los_Angeles"))

    detail = provider.airport_weather(
        airport,
        "America/Los_Angeles",
        local_target,
        generated_at=datetime(2026, 7, 14, 12, tzinfo=UTC),
    )

    assert context_provider.prior_departure == local_target
    assert context_provider.prior_departure.month == 9
    assert detail.target_time.month == 10


def test_open_meteo_error_object_is_not_cached_as_weather(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    fake_client = FakeClient({OPEN_METEO_URL: FakeResponse({"error": True})})
    provider = DetailProvider(ContextProvider(client=fake_client))
    airport = Airport("YYZ", "", "Toronto Pearson", "large_airport", "CA", 43.6777, -79.6248)

    detail = provider.airport_weather(
        airport,
        "America/Toronto",
        generated_at + timedelta(days=30),
        generated_at=generated_at,
    )

    assert detail.metadata.status == "proxy"
    assert detail.metadata.source == "synthetic_demo_training_average"
    assert provider._weather_cache == {}  # noqa: SLF001


def test_malformed_noaa_success_payload_is_reported_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")
    generated_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    fake_client = FakeClient(
        {
            OPEN_METEO_URL: FakeResponse(_hourly_payload(generated_at)),
            NOAA_METAR_URL: FakeResponse({"error": True, "message": "bad request"}),
            NOAA_TAF_URL: FakeResponse({"unexpected": "shape"}),
        }
    )
    provider = DetailProvider(ContextProvider(client=fake_client))
    airport = Airport(
        "JFK",
        "KJFK",
        "John F. Kennedy International",
        "large_airport",
        "US",
        40.6413,
        -73.7781,
    )

    detail = provider.airport_weather(
        airport,
        "America/New_York",
        generated_at + timedelta(hours=4),
        generated_at=generated_at,
    )

    assert detail.aviation_reports == []
    assert detail.aviation_metadata.status == "unavailable"
    assert provider._aviation_cache == {}  # noqa: SLF001


def test_detail_api_returns_typed_fallbacks(
    monkeypatch,
    trained_model_dir: Path,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    service = PredictionService(trained_model_dir)
    monkeypatch.setattr(api_module, "get_service", lambda: service)
    client = TestClient(app)
    departure = (datetime.now(UTC) + timedelta(days=45)).isoformat()
    request = {"origin": "JFK", "destination": "LAX", "departure_time": departure}

    weather = client.post("/v1/context/weather-detail", json=request)
    news = client.post("/v1/context/news-detail", json=request)

    assert weather.status_code == 200
    weather_payload = weather.json()
    assert weather_payload["origin_weather"]["metadata"]["status"] == "proxy"
    assert weather_payload["origin_weather"]["current"] is None
    assert weather_payload["destination_weather"]["timezone"] == "America/Los_Angeles"
    assert weather_payload["estimated_arrival_time"]
    assert weather_payload["departure_time_basis"] == "legacy_input"

    assert news.status_code == 200
    news_payload = news.json()
    assert news_payload["metadata"]["status"] == "unavailable"
    assert news_payload["articles"] == []
    assert news_payload["model_effect"] == 0
    assert news_payload["departure_attenuation_factor"] == 0.1
    assert news_payload["departure_time_basis"] == "legacy_input"


def test_date_only_detail_refresh_recomputes_expired_same_day_reference(
    monkeypatch,
    trained_model_dir: Path,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    origin_zone = ZoneInfo("America/Toronto")
    clock = {
        "now": datetime(2026, 7, 14, 20, 0, tzinfo=origin_zone).astimezone(UTC)
    }
    service = PredictionService(
        trained_model_dir,
        now_provider=lambda: clock["now"],
    )
    monkeypatch.setattr(api_module, "get_service", lambda: service)
    client = TestClient(app)
    request = {
        "origin": "YYZ",
        "destination": "LHR",
        "departure_date": "2026-07-14",
    }

    first_weather = client.post("/v1/context/weather-detail", json=request)
    first_news = client.post("/v1/context/news-detail", json=request)
    assert first_weather.status_code == 200
    assert first_news.status_code == 200
    first_weather_payload = first_weather.json()
    first_news_payload = first_news.json()
    assert (
        first_weather_payload["departure_time_basis"]
        == "origin_local_remaining_day_model_reference"
    )
    assert (
        first_news_payload["departure_time_basis"]
        == "origin_local_remaining_day_model_reference"
    )
    first_reference = datetime.fromisoformat(first_weather_payload["departure_time"])
    assert first_reference.astimezone(UTC) == clock["now"] + timedelta(minutes=30)

    clock["now"] += timedelta(hours=1)
    refreshed_weather = client.post("/v1/context/weather-detail", json=request)
    refreshed_news = client.post("/v1/context/news-detail", json=request)

    assert refreshed_weather.status_code == 200
    assert refreshed_news.status_code == 200
    refreshed_reference = datetime.fromisoformat(
        refreshed_weather.json()["departure_time"]
    )
    assert refreshed_reference.astimezone(UTC) == clock["now"] + timedelta(minutes=30)
    assert refreshed_reference > first_reference


def test_second_level_pages_are_served() -> None:
    client = TestClient(app)

    weather_page = client.get("/details/weather")
    news_page = client.get("/details/news")

    assert weather_page.status_code == 200
    assert news_page.status_code == 200
    assert "text/html" in weather_page.headers["content-type"]
    assert "text/html" in news_page.headers["content-type"]
    assert "departure_date" in weather_page.text
    assert "departure_time_basis" in weather_page.text
    assert "不是航班计划" in weather_page.text
    assert "departure_date" in news_page.text
    assert "departure_time_basis" in news_page.text
    assert "不是航班计划" in news_page.text

    dashboard = client.get("/").text
    assert "departure_date: departureDate" in dashboard
    assert "departure_time_basis: departureTimeBasis" in dashboard
