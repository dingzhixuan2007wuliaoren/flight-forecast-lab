import numpy as np
import pandas as pd
import pytest

from flight_forecaster.catalog import get_airline_profile
from flight_forecaster.data import (
    AIRPORT_COUNTRY_BY_IATA,
    AIRPORT_TIMEZONE_BY_IATA,
    generate_demo_ontime_data,
    generate_demo_price_data,
)
from flight_forecaster.features import (
    LEGACY_PRICE_CATEGORICAL_FEATURES,
    LEGACY_PRICE_NUMERIC_FEATURES,
    PRICE_CATEGORICAL_FEATURES,
    PRICE_NUMERIC_FEATURES,
    build_ontime_features,
    build_ontime_features_without_weather,
    build_price_features,
)
from flight_forecaster.training import split_temporal_window, temporal_split


def test_demo_generation_is_deterministic() -> None:
    first = generate_demo_price_data(rows=500, seed=7)
    second = generate_demo_price_data(rows=500, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_temporal_split_keeps_future_rows_out_of_training() -> None:
    data = generate_demo_price_data(rows=1_000)
    split = temporal_split(data, "quote_time")
    assert split.train["quote_time"].max() <= split.calibration["quote_time"].min()
    assert split.calibration["quote_time"].max() <= split.test["quote_time"].min()


def test_temporal_window_split_keeps_equal_timestamp_snapshots_together() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
        utc=True,
    )
    first, second = split_temporal_window(
        pd.DataFrame({"quote_time": timestamps}),
        "quote_time",
    )

    assert first["quote_time"].max() < second["quote_time"].min()
    assert set(first["quote_time"]).isdisjoint(second["quote_time"])


def test_feature_builders_exclude_targets_and_post_flight_values() -> None:
    price = generate_demo_price_data(rows=500)
    on_time = generate_demo_ontime_data(rows=500)
    price_features = build_price_features(price)
    ontime_features = build_ontime_features(on_time)
    assert "news_disruption_index" in price_features
    assert "news_disruption_index" in ontime_features
    assert "price_usd" not in price_features
    assert "arrival_delay_minutes" not in ontime_features
    assert "cancelled" not in ontime_features
    assert "on_time" not in ontime_features


def test_no_weather_feature_builder_neither_requires_nor_emits_weather() -> None:
    data = generate_demo_ontime_data(rows=500).iloc[:5].drop(columns=["weather_severity_forecast"])

    features = build_ontime_features_without_weather(data)

    assert "weather_severity_forecast" not in features.columns
    assert len(features) == 5


def test_price_feature_builder_rejects_departures_in_the_past() -> None:
    data = generate_demo_price_data(rows=500).iloc[:1].copy()
    data["departure_time"] = data["quote_time"]
    with pytest.raises(ValueError, match="after quote_time"):
        build_price_features(data)


def test_local_departure_components_override_utc_clock_features() -> None:
    data = generate_demo_ontime_data(rows=500).iloc[:1].copy()
    data["scheduled_departure"] = pd.Timestamp("2026-08-10T14:00:00Z")
    data["departure_local_month"] = 8
    data["departure_local_weekday"] = 0
    data["departure_local_hour"] = 7

    features = build_ontime_features(data)

    assert features.iloc[0]["is_peak_hour"] == 1
    assert features.iloc[0]["is_weekend"] == 0
    assert features.iloc[0]["departure_hour_sin"] == pytest.approx(np.sin(2 * np.pi * 7 / 24))


@pytest.mark.parametrize(
    ("generator", "time_column"),
    [
        (generate_demo_price_data, "departure_time"),
        (generate_demo_ontime_data, "scheduled_departure"),
    ],
)
def test_demo_rows_store_origin_local_departure_components(generator, time_column: str) -> None:
    data = generator(rows=500).iloc[:25]

    for _, row in data.iterrows():
        origin = str(row["origin"])
        local = pd.Timestamp(row[time_column]).tz_convert(AIRPORT_TIMEZONE_BY_IATA[origin])
        assert row["origin_country"] == AIRPORT_COUNTRY_BY_IATA[origin]
        assert row["destination_country"] == AIRPORT_COUNTRY_BY_IATA[str(row["destination"])]
        assert row["departure_local_month"] == local.month
        assert row["departure_local_weekday"] == local.weekday()
        assert row["departure_local_hour"] == local.hour
        profile = get_airline_profile(str(row["airline"]))
        assert profile is not None
        assert row["carrier_service_model"] == profile.service_model


def test_enriched_price_feature_sets_preserve_the_legacy_sets() -> None:
    assert PRICE_CATEGORICAL_FEATURES == LEGACY_PRICE_CATEGORICAL_FEATURES + [
        "route_scope",
        "carrier_service_model",
    ]
    assert PRICE_NUMERIC_FEATURES == LEGACY_PRICE_NUMERIC_FEATURES


def test_price_factors_use_country_codes_and_explicit_unknown_fallbacks() -> None:
    quote_time = pd.Timestamp("2026-01-15T12:00:00Z")
    departure_time = pd.Timestamp("2026-03-15T12:00:00Z")
    common = {
        "cabin": "economy",
        "stops": 0,
        "duration_minutes": 300,
        "distance_km": 3_000,
        "quote_time": quote_time,
        "departure_time": departure_time,
    }
    data = pd.DataFrame(
        [
            {**common, "origin": "JFK", "destination": "LAX", "airline": "DL"},
            {**common, "origin": "JFK", "destination": "LHR", "airline": "ZZ"},
            {
                **common,
                "origin": "AAA",
                "destination": "BBB",
                "origin_country": "ca",
                "destination_country": "US",
                "airline": "XY",
                "carrier_service_model": "hybrid",
            },
            {
                **common,
                "origin": "AAA",
                "destination": "BBB",
                "airline": "XY",
                "carrier_service_model": "not-a-model",
            },
        ]
    )

    features = build_price_features(data)

    assert features["route_scope"].tolist() == [
        "domestic",
        "international",
        "international",
        "unknown",
    ]
    assert features["carrier_service_model"].tolist() == [
        "full_service",
        "unknown",
        "hybrid",
        "unknown",
    ]
