import pandas as pd
import pytest

from flight_forecaster.data import generate_demo_ontime_data, generate_demo_price_data
from flight_forecaster.features import build_ontime_features, build_price_features
from flight_forecaster.training import temporal_split


def test_demo_generation_is_deterministic() -> None:
    first = generate_demo_price_data(rows=500, seed=7)
    second = generate_demo_price_data(rows=500, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_temporal_split_keeps_future_rows_out_of_training() -> None:
    data = generate_demo_price_data(rows=1_000)
    split = temporal_split(data, "quote_time")
    assert split.train["quote_time"].max() <= split.calibration["quote_time"].min()
    assert split.calibration["quote_time"].max() <= split.test["quote_time"].min()


def test_feature_builders_exclude_targets_and_post_flight_values() -> None:
    price = generate_demo_price_data(rows=500)
    on_time = generate_demo_ontime_data(rows=500)
    price_features = build_price_features(price)
    ontime_features = build_ontime_features(on_time)
    assert "price_usd" not in price_features
    assert "arrival_delay_minutes" not in ontime_features
    assert "cancelled" not in ontime_features
    assert "on_time" not in ontime_features


def test_price_feature_builder_rejects_departures_in_the_past() -> None:
    data = generate_demo_price_data(rows=500).iloc[:1].copy()
    data["departure_time"] = data["quote_time"]
    with pytest.raises(ValueError, match="after quote_time"):
        build_price_features(data)
