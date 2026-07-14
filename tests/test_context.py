from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any

import pytest

from flight_forecaster.context import (
    ADSB_LOL_URL,
    AIRLABS_ROUTES_URL,
    AIRLABS_SCHEDULES_URL,
    GDELT_DOC_URL,
    GDELT_GAL_RSS_URL,
    GDELT_REQUEST_TIMEOUT_SECONDS,
    NEWS_CACHE_TTL_SECONDS,
    NOAA_METAR_URL,
    NOAA_TAF_URL,
    OPEN_METEO_URL,
    ContextProvider,
)


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.payload

    @property
    def text(self) -> str:
        if isinstance(self.payload, bytes):
            return self.payload.decode("utf-8")
        if isinstance(self.payload, str):
            return self.payload
        raise TypeError("FakeResponse text payload must be str or bytes")


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
        response = self.responses.get(url)
        if isinstance(response, Exception):
            raise response
        if response is None:
            raise AssertionError(f"Unexpected HTTP call: {url}")
        return FakeResponse(response)


class FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
        self.calls += 1
        raise TimeoutError("provider unavailable")


def _open_meteo_payload(
    departure: datetime,
    weather_code: int = 1,
    *,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hourly": {
            "time": [departure.replace(tzinfo=None).isoformat(timespec="minutes")],
            "temperature_2m": [21],
            "weather_code": [weather_code],
            "wind_speed_10m": [18],
            "wind_gusts_10m": [30],
            "precipitation_probability": [20],
            "visibility": [10_000],
        }
    }
    if current_time is not None:
        payload["current"] = {
            "time": current_time.replace(tzinfo=None).isoformat(timespec="minutes"),
            "temperature_2m": 19,
            "weather_code": weather_code,
            "wind_speed_10m": 18,
            "wind_gusts_10m": 30,
            "precipitation": 0.2,
            "visibility": 10_000,
        }
    return payload


@pytest.fixture(autouse=True)
def enable_external_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "1")


def test_live_weather_proxy_operations_and_news_scoring() -> None:
    departure = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
    seen = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    adsb_url = ADSB_LOL_URL.format(latitude=40.6413, longitude=-73.7781)
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(departure, weather_code=95),
            adsb_url: {"ac": [{"hex": str(index)} for index in range(30)]},
            GDELT_DOC_URL: {
                "articles": [
                    {
                        "title": "Airport closure follows conflict near route",
                        "url": "https://example.test/closure",
                        "domain": "example.test",
                        "seendate": seen,
                    },
                    {
                        "title": "Airline strike causes widespread disruption",
                        "url": "https://news.test/strike",
                        "domain": "news.test",
                        "seendate": seen,
                    },
                    {
                        "title": "Airport opens a new terminal cafe",
                        "url": "https://example.test/cafe",
                        "domain": "example.test",
                        "seendate": seen,
                    },
                ]
            },
        }
    )

    context = ContextProvider(client=client).resolve(
        "JFK", "LAX", departure, 40.6413, -73.7781
    )

    assert context.weather.status == "forecast"
    assert context.weather.source == "open_meteo_forecast"
    assert context.weather.value == pytest.approx(0.92)
    assert context.operations.status == "proxy"
    assert context.operations.source == "adsb_lol"
    assert 0.4 < context.operations.value < 0.5
    assert context.news.status == "live"
    assert context.news.source == "gdelt_doc_2_near_realtime"
    assert context.news.value == pytest.approx(0.98)
    assert [article.url for article in context.news.articles] == [
        "https://example.test/closure",
        "https://news.test/strike",
    ]
    assert context.weather.summary_zh and context.weather.summary_en
    signals = (context.weather, context.operations, context.news)
    assert all(0 <= signal.value <= 1 for signal in signals)
    gdelt_call = next(call for call in client.calls if call[0] == GDELT_DOC_URL)
    assert gdelt_call[1]["sort"] == "DateDesc"
    assert gdelt_call[2] == GDELT_REQUEST_TIMEOUT_SECONDS
    assert all(
        timeout == 3.0
        for url, _, timeout in client.calls
        if url != GDELT_DOC_URL
    )


def test_news_cache_is_route_scoped_across_departure_dates_and_attenuates_far_news() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    client = FakeClient(
        {
            GDELT_DOC_URL: {
                "articles": [
                    {
                        "title": "JFK airport closure disrupts flights to LAX",
                        "url": "https://news.test/jfk-closure",
                        "domain": "news.test",
                        "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                    }
                ]
            }
        }
    )
    provider = ContextProvider(client=client)

    near = provider._news("JFK", "LAX", now + timedelta(hours=2), now)
    far = provider._news("JFK", "LAX", now + timedelta(days=45), now)

    assert sum(url == GDELT_DOC_URL for url, _, _ in client.calls) == 1
    assert near.value == pytest.approx(0.95)
    assert far.value == pytest.approx(near.value * 0.1)
    assert far.articles == near.articles
    assert "attenuated" in far.summary_en


def test_gdelt_keeps_provider_matched_non_english_titles() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    signal = ContextProvider()._news_from_doc(
        {
            "articles": [
                {
                    "title": "机场运营受到影响，部分航班调整",
                    "url": "https://example.cn/aviation/update",
                    "domain": "example.cn",
                    "language": "zh",
                    "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                }
            ]
        },
        now,
    )

    assert signal.value == pytest.approx(0.18)
    assert len(signal.articles) == 1
    assert signal.articles[0].title == "机场运营受到影响，部分航班调整"
    assert signal.articles[0].language == "zh"


def test_gdelt_recency_decay_reduces_older_article_risk() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    provider = ContextProvider()

    fresh = provider._news_from_doc(
        {
            "articles": [
                {
                    "title": "Flight disruption at JFK airport",
                    "url": "https://fresh.test/disruption",
                    "seendate": (now - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ"),
                }
            ]
        },
        now,
    )
    older = provider._news_from_doc(
        {
            "articles": [
                {
                    "title": "Flight disruption at JFK airport",
                    "url": "https://older.test/disruption",
                    "seendate": (now - timedelta(days=4)).strftime("%Y%m%dT%H%M%SZ"),
                }
            ]
        },
        now,
    )

    assert fresh.value == pytest.approx(0.42)
    assert older.value == pytest.approx(0.189)
    assert older.value < fresh.value


def test_gdelt_filters_invalid_dates_and_urls_and_deduplicates_tracking_variants() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    provider = ContextProvider()
    canonical_a = provider._canonical_news_url(
        "https://Example.test/story?utm_source=feed#section"
    )
    canonical_b = provider._canonical_news_url(
        "https://example.test/story?utm_medium=email&fbclid=123"
    )

    assert canonical_a is not None
    assert canonical_b is not None
    assert canonical_a[0] == canonical_b[0]
    assert canonical_a[1] == "https://example.test/story"
    assert provider._news_title_key("JFK Airport Closure!") == provider._news_title_key(
        " jfk airport closure "
    )

    signal = provider._news_from_doc(
        {
            "articles": [
                {
                    "title": "JFK airport closure disrupts flights",
                    "url": "https://Example.test/story?utm_source=feed#section",
                    "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                },
                {
                    "title": "Airline strike grounds JFK flights",
                    "url": "https://example.test/story?utm_medium=email&fbclid=123",
                    "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                },
                {
                    "title": "jfk airport closure disrupts flights!",
                    "url": "https://copy.test/reposted-story",
                    "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                },
                {
                    "title": "Airport closure on unsafe page",
                    "url": "javascript:alert(1)",
                    "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                },
                {
                    "title": "Airport closure reported in the future",
                    "url": "https://future.test/story",
                    "seendate": (now + timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ"),
                },
                {
                    "title": "Stale airport closure report",
                    "url": "https://stale.test/story",
                    "seendate": (now - timedelta(days=8)).strftime("%Y%m%dT%H%M%SZ"),
                },
                {
                    "title": "x" * 301,
                    "url": "https://oversized.test/story",
                    "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                },
            ]
        },
        now,
    )

    assert [article.url for article in signal.articles] == [
        "https://example.test/story"
    ]


def test_news_text_risk_does_not_treat_strike_a_deal_as_labor_action() -> None:
    provider = ContextProvider()

    assert provider._news_text_risk("Airline and pilots strike a deal on pay") == 0
    assert provider._news_text_risk("Pilot strike closes the airport") == pytest.approx(0.82)


def test_gdelt_doc_failure_uses_free_gal_rss_fallback() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    pub_date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>JFK airport closure disrupts flights to LAX</title>
        <link>https://rss-news.test/jfk-closure?utm_source=rss</link>
        <pubDate>{pub_date}</pubDate>
      </item>
      <item>
        <title>Unrelated airport closure</title>
        <link>https://rss-news.test/unrelated</link>
        <pubDate>{pub_date}</pubDate>
      </item>
    </channel></rss>"""
    client = FakeClient(
        {
            GDELT_DOC_URL: TimeoutError("DOC API unavailable"),
            GDELT_GAL_RSS_URL: rss,
        }
    )

    signal = ContextProvider(client=client)._fetch_news(
        "JFK",
        "LAX",
        now,
        origin_name="John F. Kennedy International Airport",
        destination_name="Los Angeles International Airport",
    )

    assert signal.source == "gdelt_gal_rss"
    assert signal.value == pytest.approx(0.95)
    assert [article.url for article in signal.articles] == [
        "https://rss-news.test/jfk-closure"
    ]
    assert [(url, timeout) for url, _, timeout in client.calls] == [
        (GDELT_DOC_URL, GDELT_REQUEST_TIMEOUT_SECONDS),
        (GDELT_GAL_RSS_URL, 3.0),
    ]


def test_rss_route_matching_does_not_treat_lowercase_iata_word_as_airport_code() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    pub_date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Travelers can avoid disruption after schedule changes</title>
        <link>https://rss-news.test/lowercase-can</link>
        <pubDate>{pub_date}</pubDate>
      </item>
      <item>
        <title>CAN airport disruption delays flights</title>
        <link>https://rss-news.test/uppercase-can</link>
        <pubDate>{pub_date}</pubDate>
      </item>
    </channel></rss>"""

    signal = ContextProvider()._news_from_rss(
        rss,
        "CAN",
        "PVG",
        now,
        origin_name=None,
        destination_name=None,
    )

    assert [article.url for article in signal.articles] == [
        "https://rss-news.test/uppercase-can"
    ]


def test_news_refresh_failure_uses_attenuated_stale_last_good_cache() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    client = FakeClient(
        {
            GDELT_DOC_URL: {
                "articles": [
                    {
                        "title": "JFK airport closure disrupts flights to LAX",
                        "url": "https://news.test/jfk-closure",
                        "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                    }
                ]
            }
        }
    )
    provider = ContextProvider(client=client)
    departure = now + timedelta(hours=2)

    live = provider._news("JFK", "LAX", departure, now)
    cache_key = next(iter(provider._news_cache))
    _, cached_signal = provider._news_cache[cache_key]
    provider._news_cache[cache_key] = (
        monotonic() - NEWS_CACHE_TTL_SECONDS - 1,
        cached_signal,
    )
    client.responses[GDELT_DOC_URL] = TimeoutError("DOC API unavailable")
    client.responses[GDELT_GAL_RSS_URL] = TimeoutError("RSS unavailable")

    stale = provider._news("JFK", "LAX", departure, now + timedelta(minutes=16))
    provider_call_count = len(client.calls)
    stale_again = provider._news("JFK", "LAX", departure, now + timedelta(minutes=17))

    assert live.status == "live"
    assert stale.status == "historical"
    assert stale.source == "gdelt_doc_2_near_realtime_stale_cache"
    assert stale.value == pytest.approx(live.value * 0.5)
    assert stale.articles == live.articles
    assert "latest successful cache" in stale.summary_en
    assert stale_again.status == "historical"
    assert stale_again.source == "gdelt_doc_2_near_realtime_stale_cache"
    assert stale_again.value == stale.value
    assert stale_again.articles == live.articles
    assert len(client.calls) == provider_call_count


def test_airlabs_schedules_and_route_airline_cache() -> None:
    departure = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    schedules = [
        {"status": "scheduled", "dep_delayed": 20 if index < 10 else 0}
        for index in range(20)
    ]
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(departure),
            AIRLABS_SCHEDULES_URL: {"response": schedules},
            AIRLABS_ROUTES_URL: {
                "response": [
                    {"airline_iata": "AA"},
                    {"airline_iata": "DL"},
                    {"airline_iata": "AA"},
                ]
            },
            GDELT_DOC_URL: {"articles": []},
        }
    )
    provider = ContextProvider(airlabs_api_key="free-test-key", client=client)

    context = provider.resolve("JFK", "LAX", departure, 40.6413, -73.7781)
    first_airlines = provider.route_airlines("JFK", "LAX")
    second_airlines = provider.route_airlines("jfk", "lax")

    assert context.operations.status == "live"
    assert context.operations.source == "airlabs_schedules"
    assert context.operations.value == pytest.approx(0.4083)
    assert first_airlines == {"AA", "DL"}
    assert second_airlines == first_airlines
    assert sum(url == AIRLABS_ROUTES_URL for url, _, _ in client.calls) == 1
    assert not any(url.startswith("https://api.adsb.lol") for url, _, _ in client.calls)


def test_noaa_aviation_weather_supplements_open_meteo() -> None:
    now = datetime.now(UTC)
    departure = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    adsb_url = ADSB_LOL_URL.format(latitude=40.6413, longitude=-73.7781)
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(departure),
            NOAA_TAF_URL: [
                {
                    "rawTAF": "KJFK 141700Z 1418/1524 +TSRA BKN015",
                    "issueTime": now.isoformat(),
                    "validTimeFrom": int((now - timedelta(hours=1)).timestamp()),
                    "validTimeTo": int((now + timedelta(hours=30)).timestamp()),
                    "fcsts": [
                        {
                            "timeFrom": int((now - timedelta(hours=1)).timestamp()),
                            "timeTo": int((now + timedelta(hours=30)).timestamp()),
                            "wxString": "+TSRA",
                            "wspd": 10,
                            "visib": "6+",
                            "clouds": [{"cover": "BKN", "base": 1_500}],
                        }
                    ],
                }
            ],
            NOAA_METAR_URL: [
                {
                    "rawOb": "KJFK 141651Z 20010KT 10SM FEW020",
                    "obsTime": int(now.timestamp()),
                }
            ],
            adsb_url: {"ac": []},
            GDELT_DOC_URL: {"articles": []},
        }
    )

    context = ContextProvider(client=client).resolve(
        "JFK", "LAX", departure, 40.6413, -73.7781, icao_code="KJFK"
    )

    assert context.weather.status == "live"
    assert context.weather.source == "open_meteo_forecast+noaa_taf+noaa_metar"
    assert context.weather.value == pytest.approx(0.95)


def test_taf_but_not_current_metar_is_used_for_later_same_day() -> None:
    now = datetime.now(UTC)
    departure = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=12)
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(departure),
            NOAA_TAF_URL: [
                {
                    "rawTAF": "KJFK 141700Z 1418/1524 TSRA BKN015",
                    "issueTime": now.isoformat(),
                    "validTimeFrom": int((now - timedelta(hours=1)).timestamp()),
                    "validTimeTo": int((now + timedelta(hours=30)).timestamp()),
                    "fcsts": [
                        {
                            "timeFrom": int((now - timedelta(hours=1)).timestamp()),
                            "timeTo": int((now + timedelta(hours=30)).timestamp()),
                            "wxString": "TSRA",
                            "wspd": 10,
                            "visib": "6+",
                            "clouds": [{"cover": "BKN", "base": 1_500}],
                        }
                    ],
                }
            ],
            GDELT_DOC_URL: {"articles": []},
        }
    )

    context = ContextProvider(client=client).resolve(
        "JFK", "LAX", departure, 40.6413, -73.7781, icao_code="KJFK"
    )

    called_urls = {url for url, _, _ in client.calls}
    assert context.weather.source == "open_meteo_forecast+noaa_taf"
    assert NOAA_TAF_URL in called_urls
    assert NOAA_METAR_URL not in called_urls


def test_near_departure_uses_fresh_open_meteo_current_conditions() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    departure = now + timedelta(hours=1)
    adsb_url = ADSB_LOL_URL.format(latitude=40.6413, longitude=-73.7781)
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(
                departure,
                weather_code=95,
                current_time=now,
            ),
            adsb_url: {"ac": []},
            GDELT_DOC_URL: {"articles": []},
        }
    )

    context = ContextProvider(client=client).resolve(
        "JFK",
        "LAX",
        departure,
        40.6413,
        -73.7781,
    )

    weather_call = next(params for url, params, _ in client.calls if url == OPEN_METEO_URL)
    assert "current" in weather_call
    assert context.weather.status == "live"
    assert context.weather.source == "open_meteo_current_model"
    assert context.weather.value == pytest.approx(0.92)
    assert context.weather.observed_at == now
    assert "15-minute model current-weather" in context.weather.summary_en


def test_stale_current_conditions_fall_back_to_departure_forecast() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    departure = now + timedelta(hours=1)
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(
                departure,
                current_time=now - timedelta(hours=3),
            ),
        }
    )

    weather = ContextProvider(client=client)._weather(
        departure,
        40.6413,
        -73.7781,
        None,
        now,
    )

    assert weather.status == "forecast"
    assert weather.source == "open_meteo_forecast"
    assert weather.observed_at == departure


def test_incomplete_current_conditions_fall_back_to_departure_forecast() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    departure = now + timedelta(hours=1)
    payload = _open_meteo_payload(departure)
    payload["current"] = {"time": now.replace(tzinfo=None).isoformat(timespec="minutes")}
    client = FakeClient({OPEN_METEO_URL: payload})

    weather = ContextProvider(client=client)._weather(
        departure,
        40.6413,
        -73.7781,
        None,
        now,
    )

    assert weather.status == "forecast"
    assert weather.source == "open_meteo_forecast"


def test_fresh_noaa_metar_is_used_when_open_meteo_is_unavailable() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    departure = now + timedelta(hours=1)
    client = FakeClient(
        {
            OPEN_METEO_URL: TimeoutError("Open-Meteo unavailable"),
            NOAA_TAF_URL: [],
            NOAA_METAR_URL: [
                {
                    "rawOb": "KJFK 141651Z 20010KT 1SM +TSRA BKN010",
                    "obsTime": int(now.timestamp()),
                }
            ],
        }
    )

    weather = ContextProvider(client=client)._weather(
        departure,
        40.6413,
        -73.7781,
        "KJFK",
        now,
    )

    assert weather.status == "live"
    assert weather.source == "noaa_metar"
    assert weather.value == pytest.approx(0.95)


def test_metar_structured_wind_visibility_and_ceiling_raise_risk() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    client = FakeClient(
        {
            NOAA_METAR_URL: [
                {
                    "rawOb": "KJFK 141651Z 20055G75KT M1/4SM OVC001",
                    "obsTime": int(now.timestamp()),
                    "wspd": 55,
                    "wgst": 75,
                    "visib": "1/4",
                    "fltCat": "LIFR",
                    "clouds": [{"cover": "OVC", "base": 100}],
                }
            ]
        }
    )

    weather = ContextProvider(client=client)._noaa_product_signal(
        NOAA_METAR_URL,
        "KJFK",
        now + timedelta(hours=1),
        now,
    )

    assert weather is not None
    assert weather.status == "live"
    assert weather.value == 1


def test_taf_scores_only_segments_overlapping_departure() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    departure = now + timedelta(hours=1)
    client = FakeClient(
        {
            NOAA_TAF_URL: [
                {
                    "rawTAF": "KJFK CLEAR TEMPO LATER +TSRA",
                    "issueTime": now.isoformat(),
                    "validTimeFrom": int((now - timedelta(hours=1)).timestamp()),
                    "validTimeTo": int((now + timedelta(hours=8)).timestamp()),
                    "fcsts": [
                        {
                            "timeFrom": int((now - timedelta(hours=1)).timestamp()),
                            "timeTo": int((now + timedelta(hours=2)).timestamp()),
                            "wspd": 10,
                            "visib": "10+",
                            "clouds": [{"cover": "FEW", "base": 5_000}],
                        },
                        {
                            "timeFrom": int((now + timedelta(hours=2)).timestamp()),
                            "timeTo": int((now + timedelta(hours=5)).timestamp()),
                            "wxString": "+TSRA",
                            "wspd": 25,
                            "visib": "2",
                            "clouds": [{"cover": "BKN", "base": 500}],
                        },
                    ],
                }
            ]
        }
    )

    weather = ContextProvider(client=client)._noaa_product_signal(
        NOAA_TAF_URL,
        "KJFK",
        departure,
        now,
    )

    assert weather is not None
    assert weather.status == "forecast"
    assert weather.value == pytest.approx(0.1852)


def test_noaa_products_are_cached_to_respect_frequency_guidance() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    departure = now + timedelta(hours=1)
    client = FakeClient(
        {
            NOAA_TAF_URL: [
                {
                    "rawTAF": "KJFK 141700Z 1418/1524 SCT020",
                    "issueTime": now.isoformat(),
                    "validTimeFrom": int((now - timedelta(hours=1)).timestamp()),
                    "validTimeTo": int((now + timedelta(hours=30)).timestamp()),
                    "fcsts": [
                        {
                            "timeFrom": int((now - timedelta(hours=1)).timestamp()),
                            "timeTo": int((now + timedelta(hours=30)).timestamp()),
                            "wspd": 10,
                            "visib": "10+",
                            "clouds": [{"cover": "FEW", "base": 5_000}],
                        }
                    ],
                }
            ],
            NOAA_METAR_URL: [
                {
                    "rawOb": "KJFK 141651Z 20010KT 10SM FEW020",
                    "obsTime": int(now.timestamp()),
                }
            ],
        }
    )
    provider = ContextProvider(client=client)

    first = provider._noaa_weather("KJFK", departure, now, include_metar=True)
    second = provider._noaa_weather("KJFK", departure, now, include_metar=True)

    assert first == second
    assert sum(url == NOAA_TAF_URL for url, _, _ in client.calls) == 1
    assert sum(url == NOAA_METAR_URL for url, _, _ in client.calls) == 1


def test_provider_failures_return_priors_and_cache_result() -> None:
    departure = datetime(2026, 1, 14, 8, tzinfo=UTC)
    client = FailingClient()
    provider = ContextProvider(client=client)

    first = provider.resolve("YYZ", "YVR", departure, 43.6777, -79.6248)
    call_count = client.calls
    second = provider.resolve("YYZ", "YVR", departure, 43.6777, -79.6248)

    assert first is second
    assert client.calls == call_count
    assert first.weather.status == "proxy"
    assert first.weather.source == "synthetic_model_prior"
    assert first.weather.value == pytest.approx(0.42)
    assert first.operations.status == "proxy"
    assert first.operations.source == "synthetic_model_prior"
    assert first.news.status == "neutral"
    assert first.news.source == "neutral_fallback"
    assert first.news.value == 0
    assert first.news.articles == ()


def test_route_airline_provider_failure_is_cached() -> None:
    client = FailingClient()
    provider = ContextProvider(airlabs_api_key="free-test-key", client=client)

    assert provider.route_airlines("JFK", "LAX") is None
    assert provider.route_airlines("JFK", "LAX") is None
    assert client.calls == 1


def test_external_context_can_be_disabled_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_CONTEXT_ENABLED", "0")
    client = FailingClient()
    provider = ContextProvider(airlabs_api_key="unused", client=client)
    departure = datetime(2026, 7, 14, 17, tzinfo=UTC)

    context = provider.resolve("JFK", "LAX", departure, 40.6413, -73.7781)

    assert client.calls == 0
    assert context.weather.source == "synthetic_model_prior"
    assert context.operations.source == "synthetic_model_prior"
    assert context.operations.value == pytest.approx(0.64)
    assert context.news.source == "offline_fallback"
    assert context.news.value == 0
    assert provider.route_airlines("JFK", "LAX") is None


def test_future_departure_does_not_use_current_operations_or_noaa() -> None:
    departure = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(days=5)
    client = FakeClient(
        {
            OPEN_METEO_URL: _open_meteo_payload(departure),
            GDELT_DOC_URL: {"articles": []},
        }
    )

    context = ContextProvider(airlabs_api_key="unused-key", client=client).resolve(
        "JFK",
        "LAX",
        departure,
        40.6413,
        -73.7781,
        icao_code="KJFK",
        origin_name="John F. Kennedy International",
        destination_name="Los Angeles International",
    )

    called_urls = {url for url, _, _ in client.calls}
    assert context.weather.status == "forecast"
    assert context.operations.status == "proxy"
    assert context.operations.source == "synthetic_model_prior"
    assert NOAA_TAF_URL not in called_urls
    assert NOAA_METAR_URL not in called_urls
    assert AIRLABS_SCHEDULES_URL not in called_urls
    assert not any(url.startswith("https://api.adsb.lol") for url in called_urls)
    gdelt_call = next(params for url, params, _ in client.calls if url == GDELT_DOC_URL)
    assert '"John F. Kennedy International"' in gdelt_call["query"]
