from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from flight_forecaster.hotel_model import (
    HOTEL_ARTIFACT_FILENAME,
    HOTEL_CORE_FEATURES,
    HOTEL_ENRICHED_FEATURES,
    HOTEL_TARGET,
    build_hotel_price_features,
    generate_demo_hotel_price_data,
    hotel_temporal_split,
    load_hotel_price_model,
    predict_hotel_price,
    save_hotel_price_model,
    train_hotel_price_model,
)


@pytest.fixture(scope="module")
def hotel_data() -> pd.DataFrame:
    return generate_demo_hotel_price_data(rows=1_800, seed=44)


@pytest.fixture(scope="module")
def hotel_bundle(hotel_data: pd.DataFrame) -> dict:
    return train_hotel_price_model(hotel_data, random_state=42)


def test_demo_generation_is_deterministic_and_has_strict_target() -> None:
    first = generate_demo_hotel_price_data(rows=500, seed=9)
    second = generate_demo_hotel_price_data(rows=500, seed=9)

    pd.testing.assert_frame_equal(first, second)
    assert (first[HOTEL_TARGET] > 0).all()
    assert set(first["currency"]) == {"USD"}
    assert set(first["source"]) == {"synthetic_hotel_demo"}
    with pytest.raises(ValueError, match="at least 500"):
        generate_demo_hotel_price_data(rows=499)


def test_feature_whitelist_excludes_prices_identity_and_post_stay_values(
    hotel_data: pd.DataFrame,
) -> None:
    sample = hotel_data.iloc[:8].copy()
    sample["current_price_usd"] = 1.0
    sample["total_price_usd"] = 999_999.0
    sample["hotel_id"] = "must-not-enter-model"
    sample["property_token"] = "secret-provider-token"
    sample["booking_url"] = "https://example.com/book"
    sample["actual_occupancy"] = 1.0

    core = build_hotel_price_features(sample, feature_set="core")
    enriched = build_hotel_price_features(sample, feature_set="enriched")

    assert list(core.columns) == HOTEL_CORE_FEATURES
    assert list(enriched.columns) == HOTEL_ENRICHED_FEATURES
    assert not any("price" in column for column in enriched.columns)
    assert not {
        HOTEL_TARGET,
        "current_price_usd",
        "total_price_usd",
        "hotel_id",
        "property_token",
        "booking_url",
        "actual_occupancy",
    }.intersection(enriched.columns)
    assert "distance_from_airport_km" in enriched


def test_derived_stay_lead_weekend_and_cyclic_features_are_correct() -> None:
    frame = pd.DataFrame(
        [
            {
                "quote_time": "2026-08-28T15:00:00Z",
                "check_in": "2026-09-04",
                "check_out": "2026-09-07",
                "destination": "yyz",
                "property_type": "Guest House",
                "adults": 2,
            }
        ]
    )

    features = build_hotel_price_features(frame, feature_set="core").iloc[0]

    assert features["destination"] == "YYZ"
    assert features["property_type"] == "guest_house"
    assert features["days_until_check_in"] == 7
    assert features["stay_nights"] == 3
    assert features["weekend_night_share"] == pytest.approx(2 / 3)
    assert features["check_in_month_sin"] == pytest.approx(np.sin(2 * np.pi * 9 / 12))
    assert features["check_in_weekday_cos"] == pytest.approx(np.cos(2 * np.pi * 4 / 7))


def test_invalid_dates_adults_and_optional_factors_are_rejected() -> None:
    base = {
        "quote_time": "2026-08-28T15:00:00Z",
        "check_in": "2026-09-04",
        "check_out": "2026-09-07",
        "destination": "YYZ",
        "property_type": "hotel",
        "adults": 2,
    }
    invalid_rows = (
        ({**base, "check_out": "2026-09-04"}, "between 1 and 30 nights"),
        ({**base, "check_in": "2026-08-27"}, "check_in"),
        ({**base, "adults": 9}, "adults"),
        ({**base, "distance_from_airport_km": -1}, "distance_from_airport_km"),
        ({**base, "rating": 6}, "rating"),
        ({**base, "free_cancellation": "maybe"}, "free_cancellation"),
    )
    for payload, message in invalid_rows:
        with pytest.raises(ValueError, match=message):
            build_hotel_price_features(pd.DataFrame([payload]), feature_set="enriched")


def test_temporal_split_keeps_every_later_stage_out_of_training(
    hotel_data: pd.DataFrame,
) -> None:
    split = hotel_temporal_split(hotel_data)

    assert split.train["quote_time"].max() <= split.selection_calibration["quote_time"].min()
    assert (
        split.selection_calibration["quote_time"].max()
        <= split.interval_calibration["quote_time"].min()
    )
    assert set(split.selection_calibration["quote_time"]).isdisjoint(
        split.interval_calibration["quote_time"]
    )
    assert split.interval_calibration["quote_time"].max() <= split.test["quote_time"].min()
    assert sum(
        len(item)
        for item in (
            split.train,
            split.selection_calibration,
            split.interval_calibration,
            split.test,
        )
    ) == len(hotel_data)


def test_training_selects_enrichment_only_after_better_calibration_mae(
    hotel_bundle: dict,
) -> None:
    selection = hotel_bundle["metrics"]["feature_selection"]
    metadata = hotel_bundle["metadata"]

    assert selection["enriched_calibration_mae_usd"] < selection["core_calibration_mae_usd"]
    assert hotel_bundle["selected_feature_set"] == "enriched"
    assert metadata["feature_selection"]["selected"] == "enriched"
    assert "practical threshold" in metadata["feature_selection"]["rule"]
    assert metadata["interval_calibration"]["method"].endswith("log1p price space")
    assert metadata["synthetic_demo"] is True
    assert metadata["model_version"]
    assert datetime.fromisoformat(metadata["trained_at_utc"]).utcoffset() is not None
    assert len(metadata["data_fingerprint_sha256"]) == 64
    assert metadata["runtime"]["python"]
    assert metadata["runtime"]["scikit_learn"]
    assert metadata["destination_coverage"]["known_destination_codes"] == [
        "CDG",
        "DXB",
        "JFK",
        "LAX",
        "LHR",
        "SIN",
        "SYD",
        "YYZ",
    ]
    assert metadata["destination_coverage"]["destination_count"] == 8
    json.dumps(metadata, allow_nan=False)
    json.dumps(hotel_bundle["metrics"], allow_nan=False)


def test_model_known_destination_coverage_uses_training_rows_only(
    hotel_data: pd.DataFrame,
) -> None:
    frame = hotel_data.sort_values("quote_time", kind="stable").reset_index(drop=True)
    frame.loc[frame.index[-50:], "destination"] = "ZZZ"

    bundle = train_hotel_price_model(frame, random_state=42)

    assert "ZZZ" not in bundle["metadata"]["destination_coverage"][
        "known_destination_codes"
    ]


def test_selected_model_beats_baseline_and_has_a_useful_80_percent_interval(
    hotel_bundle: dict,
) -> None:
    metrics = hotel_bundle["metrics"]["test"]

    assert metrics["mae_usd"] < metrics["baseline_mae_usd"]
    assert metrics["rmse_usd"] > 0
    assert metrics["interval_80_mean_width_usd"] > 0
    assert 0.65 <= metrics["interval_80_empirical_coverage"] <= 0.95
    assert hotel_bundle["log_interval_half_width"] > 0


def test_prediction_is_bounded_handles_unseen_categories_and_ignores_current_price(
    hotel_bundle: dict,
) -> None:
    quote_time = datetime.now(UTC).replace(microsecond=0)
    request = {
        "destination": "BOS",
        "property_type": "hotel",
        "check_in": (quote_time + timedelta(days=45)).date().isoformat(),
        "check_out": (quote_time + timedelta(days=49)).date().isoformat(),
        "adults": 2,
        "hotel_class": 4,
        "rating": 4.3,
        "review_count": 850,
        "distance_from_city_center_km": 2.1,
        "distance_from_airport_km": 14.5,
        "amenity_count": 11,
        "free_cancellation": True,
    }
    first = predict_hotel_price(hotel_bundle, request, quote_time=quote_time)
    second = predict_hotel_price(
        hotel_bundle,
        {**request, "current_price_usd": 999_999.0, "nightly_price_usd": 1.0},
        quote_time=quote_time,
    )

    assert first == second
    assert 0 <= first.interval_80_low_usd <= first.estimated_nightly_price_usd
    assert first.interval_80_high_usd >= first.estimated_nightly_price_usd
    assert first.stay_nights == 4
    assert first.days_until_check_in == 45
    assert first.estimated_stay_total_usd == pytest.approx(
        first.estimated_nightly_price_usd * first.stay_nights,
        abs=0.02,
    )
    assert first.feature_set == "enriched"
    assert first.data_mode == "synthetic_demo"
    assert "not a live bookable quote" in first.warning_en
    assert "synthetic demo data" in first.warning_en


def test_nonpositive_training_target_is_rejected(hotel_data: pd.DataFrame) -> None:
    invalid = hotel_data.iloc[:500].copy()
    invalid.loc[invalid.index[0], HOTEL_TARGET] = 0

    with pytest.raises(ValueError, match="finite positive"):
        train_hotel_price_model(invalid)


def test_hotel_artifact_round_trip_preserves_prediction(
    hotel_bundle: dict,
    tmp_path,
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    request = {
        "destination": "YYZ",
        "property_type": "hotel",
        "check_in": (now + timedelta(days=30)).date().isoformat(),
        "check_out": (now + timedelta(days=33)).date().isoformat(),
        "adults": 2,
        "hotel_class": 4,
        "rating": 4.2,
        "review_count": 500,
        "distance_from_city_center_km": 2.0,
        "distance_from_airport_km": 22.0,
        "amenity_count": 10,
        "free_cancellation": True,
    }
    expected = predict_hotel_price(hotel_bundle, request, quote_time=now)

    artifact = save_hotel_price_model(hotel_bundle, tmp_path)
    loaded = load_hotel_price_model(tmp_path)

    assert artifact == tmp_path / HOTEL_ARTIFACT_FILENAME
    assert artifact.is_file()
    assert not list(tmp_path.glob(f".{HOTEL_ARTIFACT_FILENAME}.*.tmp"))
    assert predict_hotel_price(loaded, request, quote_time=now) == expected


def test_hotel_artifact_rejects_wrong_schema(hotel_bundle: dict, tmp_path) -> None:
    invalid = {**hotel_bundle, "hotel_model_schema_version": 999}

    with pytest.raises(ValueError, match="unsupported hotel model schema"):
        save_hotel_price_model(invalid, tmp_path)
    assert not (tmp_path / HOTEL_ARTIFACT_FILENAME).exists()

    joblib.dump(invalid, tmp_path / HOTEL_ARTIFACT_FILENAME)
    with pytest.raises(ValueError, match="unsupported hotel model schema"):
        load_hotel_price_model(tmp_path)


@pytest.mark.parametrize(
    "missing_key",
    ["trained_at_utc", "model_version", "runtime", "destination_coverage"],
)
def test_hotel_artifact_rejects_missing_provenance_metadata(
    hotel_bundle: dict,
    tmp_path,
    missing_key: str,
) -> None:
    invalid = deepcopy(hotel_bundle)
    invalid["metadata"].pop(missing_key)

    with pytest.raises(ValueError, match="hotel model"):
        save_hotel_price_model(invalid, tmp_path)


def test_missing_hotel_artifact_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="hotel model artifact not found"):
        load_hotel_price_model(tmp_path)
