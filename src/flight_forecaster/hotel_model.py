"""Independent, leakage-aware hotel price modelling for the portfolio demo.

The strict hotel provider module returns bookable price evidence.  This module
does something deliberately different: it trains a reproducible supervised
learning demo that estimates a provider-displayed nightly USD price.  Live or
"current" prices are targets/evaluation evidence only and are never features.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from flight_forecaster import __version__
from flight_forecaster.training import (
    TemporalSplit,
    split_temporal_window,
    temporal_split,
)

HOTEL_MODEL_SCHEMA_VERSION = 1
HOTEL_ARTIFACT_FILENAME = "hotel_model_bundle.joblib"
HOTEL_TARGET = "nightly_price_usd"
HOTEL_INTERVAL_COVERAGE = 0.80
HOTEL_MAX_FUTURE_DAYS = 370
HOTEL_MAX_STAY_NIGHTS = 30

HOTEL_PROPERTY_TYPES = (
    "hotel",
    "hostel",
    "guest_house",
    "motel",
    "apartment",
)

HOTEL_CORE_CATEGORICAL_FEATURES = ["destination", "property_type"]
HOTEL_CORE_NUMERIC_FEATURES = [
    "adults",
    "days_until_check_in",
    "stay_nights",
    "weekend_night_share",
    "check_in_month_sin",
    "check_in_month_cos",
    "check_in_weekday_sin",
    "check_in_weekday_cos",
]
HOTEL_ENRICHED_NUMERIC_FEATURES = [
    *HOTEL_CORE_NUMERIC_FEATURES,
    "hotel_class",
    "rating",
    "log_review_count",
    "distance_from_city_center_km",
    "distance_from_airport_km",
    "amenity_count",
    "free_cancellation",
    "free_cancellation_known",
]
HOTEL_CORE_FEATURES = HOTEL_CORE_CATEGORICAL_FEATURES + HOTEL_CORE_NUMERIC_FEATURES
HOTEL_ENRICHED_FEATURES = (
    HOTEL_CORE_CATEGORICAL_FEATURES + HOTEL_ENRICHED_NUMERIC_FEATURES
)

FeatureSet = Literal["core", "enriched"]


@dataclass(frozen=True)
class HotelTemporalSplit:
    """Ordered train, model-selection, interval-calibration, and test rows."""

    train: pd.DataFrame
    selection_calibration: pd.DataFrame
    interval_calibration: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class HotelPricePrediction:
    """A model estimate, not a live or bookable hotel quote."""

    estimated_nightly_price_usd: float
    interval_80_low_usd: float
    interval_80_high_usd: float
    estimated_stay_total_usd: float
    interval_80_total_low_usd: float
    interval_80_total_high_usd: float
    stay_nights: int
    days_until_check_in: int
    feature_set: FeatureSet
    data_mode: str
    model_schema_version: int
    warning: str
    warning_en: str


def _cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * values.astype(float) / period
    return np.sin(radians), np.cos(radians)


def _date_series(frame: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_datetime(frame[name], errors="raise")
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_convert(None)
    return values.dt.normalize()


def _optional_numeric(
    frame: pd.DataFrame,
    name: str,
    *,
    low: float,
    high: float,
    integer: bool = False,
) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    try:
        values = pd.to_numeric(frame[name], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric when provided") from exc
    present = values.notna()
    if not values[present].between(low, high).all():
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    if integer and not np.equal(np.mod(values[present], 1), 0).all():
        raise ValueError(f"{name} must contain whole numbers")
    return values


def _optional_boolean(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    def convert(value: Any) -> float:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if value in {0, 1}:
                return float(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "1", "yes"}:
                return 1.0
            if normalized in {"false", "0", "no"}:
                return 0.0
        raise ValueError(f"{name} must be boolean when provided")

    return frame[name].map(convert).astype(float)


def _weekend_night_share(check_in: pd.Series, nights: pd.Series) -> pd.Series:
    shares = []
    for start, count in zip(check_in, nights, strict=True):
        count_int = int(count)
        weekend_nights = sum(
            (start + pd.Timedelta(days=offset)).weekday() >= 5
            for offset in range(count_int)
        )
        shares.append(weekend_nights / count_int)
    return pd.Series(shares, index=check_in.index, dtype=float)


def build_hotel_price_features(
    frame: pd.DataFrame,
    *,
    feature_set: FeatureSet = "enriched",
) -> pd.DataFrame:
    """Build an explicit prediction-time feature whitelist.

    Extra columns are ignored.  In particular, nightly/total/current prices,
    provider tokens, hotel identity, booking URLs, and post-stay outcomes can
    never enter the returned matrix.
    """

    if feature_set not in {"core", "enriched"}:
        raise ValueError("feature_set must be core or enriched")
    required = {
        "quote_time",
        "check_in",
        "check_out",
        "destination",
        "property_type",
        "adults",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"hotel price data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("hotel price data cannot be empty")

    result = frame.copy()
    result["destination"] = result["destination"].astype(str).str.strip().str.upper()
    if not result["destination"].str.fullmatch(r"[A-Z]{3}").all():
        raise ValueError("destination must contain three uppercase letters")
    result["property_type"] = (
        result["property_type"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"[-\s]+", "_", regex=True)
    )
    if not result["property_type"].isin(HOTEL_PROPERTY_TYPES).all():
        raise ValueError("property_type is unsupported")

    quote_time = pd.to_datetime(result["quote_time"], utc=True, errors="raise")
    check_in = _date_series(result, "check_in")
    check_out = _date_series(result, "check_out")
    quote_date = quote_time.dt.tz_convert(None).dt.normalize()
    stay_nights = (check_out - check_in).dt.days
    lead_days = (check_in - quote_date).dt.days
    checkout_lead_days = (check_out - quote_date).dt.days
    if not stay_nights.between(1, HOTEL_MAX_STAY_NIGHTS).all():
        raise ValueError(
            f"hotel stays must be between 1 and {HOTEL_MAX_STAY_NIGHTS} nights"
        )
    if not lead_days.between(0, HOTEL_MAX_FUTURE_DAYS).all():
        raise ValueError(
            f"check_in must be from quote date through {HOTEL_MAX_FUTURE_DAYS} days"
        )
    if not checkout_lead_days.between(1, HOTEL_MAX_FUTURE_DAYS).all():
        raise ValueError(f"check_out must be within {HOTEL_MAX_FUTURE_DAYS} days")

    adults = _optional_numeric(result, "adults", low=1, high=8, integer=True)
    if adults.isna().any():
        raise ValueError("adults is required")
    result["adults"] = adults
    result["days_until_check_in"] = lead_days.astype(float)
    result["stay_nights"] = stay_nights.astype(float)
    result["weekend_night_share"] = _weekend_night_share(check_in, stay_nights)
    result["check_in_month_sin"], result["check_in_month_cos"] = _cyclic(
        check_in.dt.month,
        12,
    )
    result["check_in_weekday_sin"], result["check_in_weekday_cos"] = _cyclic(
        check_in.dt.weekday,
        7,
    )

    if feature_set == "core":
        return result[HOTEL_CORE_FEATURES]

    result["hotel_class"] = _optional_numeric(
        result,
        "hotel_class",
        low=1,
        high=5,
    )
    result["rating"] = _optional_numeric(result, "rating", low=0, high=5)
    review_count = _optional_numeric(
        result,
        "review_count",
        low=0,
        high=100_000_000,
        integer=True,
    )
    result["log_review_count"] = np.log1p(review_count)
    result["distance_from_city_center_km"] = _optional_numeric(
        result,
        "distance_from_city_center_km",
        low=0,
        high=2_000,
    )
    result["distance_from_airport_km"] = _optional_numeric(
        result,
        "distance_from_airport_km",
        low=0,
        high=2_000,
    )
    result["amenity_count"] = _optional_numeric(
        result,
        "amenity_count",
        low=0,
        high=200,
        integer=True,
    )
    cancellation = _optional_boolean(result, "free_cancellation")
    result["free_cancellation_known"] = cancellation.notna().astype(float)
    result["free_cancellation"] = cancellation
    return result[HOTEL_ENRICHED_FEATURES]


def _hotel_class_for_type(
    rng: np.random.Generator,
    property_types: np.ndarray,
) -> np.ndarray:
    class_options = {
        "hotel": (np.array([2, 3, 4, 5]), np.array([0.12, 0.43, 0.35, 0.10])),
        "hostel": (np.array([1, 2, 3]), np.array([0.58, 0.34, 0.08])),
        "guest_house": (np.array([2, 3, 4]), np.array([0.30, 0.52, 0.18])),
        "motel": (np.array([1, 2, 3]), np.array([0.16, 0.64, 0.20])),
        "apartment": (np.array([2, 3, 4, 5]), np.array([0.12, 0.46, 0.34, 0.08])),
    }
    return np.array(
        [
            rng.choice(
                class_options[property_type][0],
                p=class_options[property_type][1],
            )
            for property_type in property_types
        ],
        dtype=int,
    )


def generate_demo_hotel_price_data(rows: int = 6_000, seed: int = 44) -> pd.DataFrame:
    """Create deterministic synthetic hotel observations for pipeline validation."""

    if rows < 500:
        raise ValueError("hotel demo data requires at least 500 rows")
    rng = np.random.default_rng(seed)
    destinations = np.array(["YYZ", "JFK", "LAX", "LHR", "CDG", "DXB", "SYD", "SIN"])
    city_base = {
        "YYZ": 165.0,
        "JFK": 245.0,
        "LAX": 225.0,
        "LHR": 265.0,
        "CDG": 215.0,
        "DXB": 190.0,
        "SYD": 205.0,
        "SIN": 195.0,
    }
    airport_distance_base = {
        "YYZ": 24.0,
        "JFK": 21.0,
        "LAX": 19.0,
        "LHR": 24.0,
        "CDG": 25.0,
        "DXB": 14.0,
        "SYD": 12.0,
        "SIN": 18.0,
    }
    property_factor = {
        "hotel": 1.00,
        "hostel": 0.42,
        "guest_house": 0.73,
        "motel": 0.63,
        "apartment": 0.88,
    }

    observed_day = rng.integers(0, 1_000, size=rows)
    observed_hour = rng.integers(0, 24, size=rows)
    quote_time = (
        pd.Timestamp("2023-01-01", tz="UTC")
        + pd.to_timedelta(observed_day, unit="D")
        + pd.to_timedelta(observed_hour, unit="h")
    )
    lead_days = np.clip(rng.gamma(2.4, 18.0, size=rows).astype(int), 0, 180)
    stay_nights = rng.choice(
        np.arange(1, 11),
        size=rows,
        p=[0.27, 0.29, 0.18, 0.10, 0.06, 0.04, 0.025, 0.015, 0.005, 0.015],
    )
    check_in = quote_time.normalize() + pd.to_timedelta(lead_days, unit="D")
    check_out = check_in + pd.to_timedelta(stay_nights, unit="D")
    destination = rng.choice(destinations, size=rows)
    property_type = rng.choice(
        np.array(HOTEL_PROPERTY_TYPES),
        size=rows,
        p=[0.62, 0.07, 0.10, 0.07, 0.14],
    )
    hotel_class = _hotel_class_for_type(rng, property_type)
    rating = np.clip(
        2.45 + 0.39 * hotel_class + rng.normal(0, 0.34, size=rows),
        1.0,
        5.0,
    )
    review_count = np.clip(
        rng.lognormal(mean=4.3 + 0.17 * hotel_class, sigma=1.05, size=rows),
        0,
        100_000,
    ).astype(int)
    city_distance = np.clip(rng.gamma(1.8, 2.4, size=rows), 0.05, 35.0)
    airport_distance = np.clip(
        np.array([airport_distance_base[item] for item in destination])
        + rng.normal(0, 7.0, size=rows)
        + 0.18 * city_distance,
        0.5,
        80.0,
    )
    amenity_count = np.clip(
        rng.poisson(2.5 + 2.0 * hotel_class, size=rows),
        0,
        40,
    )
    free_cancellation = rng.random(rows) < np.clip(
        0.38 + 0.055 * hotel_class,
        0,
        0.85,
    )
    adults = rng.choice([1, 2, 3, 4], size=rows, p=[0.30, 0.53, 0.11, 0.06])
    weekend_share = _weekend_night_share(
        pd.Series(check_in.tz_convert(None)),
        pd.Series(stay_nights),
    ).to_numpy()
    month = check_in.month.to_numpy()

    base = np.array([city_base[item] for item in destination])
    type_effect = np.array([property_factor[item] for item in property_type])
    class_effect = 0.56 + 0.22 * hotel_class
    rating_effect = 1.0 + 0.08 * (rating - 3.5)
    review_effect = 1.0 + 0.025 * np.log1p(review_count)
    centre_effect = 0.92 + 0.22 * np.exp(-city_distance / 3.5)
    airport_effect = 0.97 + 0.09 * np.exp(-airport_distance / 11.0)
    amenities_effect = 1.0 + 0.014 * amenity_count
    cancellation_effect = np.where(free_cancellation, 1.055, 1.0)
    urgency_effect = 1.0 + 0.24 * np.exp(-lead_days / 10.0)
    season_effect = np.where(np.isin(month, [6, 7, 8, 12]), 1.16, 1.0)
    weekend_effect = 1.0 + 0.12 * weekend_share
    length_effect = 1.0 - 0.014 * np.minimum(stay_nights - 1, 7)
    occupancy_effect = 1.0 + 0.055 * (adults - 1)
    noise = rng.lognormal(mean=0.0, sigma=0.105, size=rows)
    nightly_price = np.maximum(
        25.0,
        base
        * type_effect
        * class_effect
        * rating_effect
        * review_effect
        * centre_effect
        * airport_effect
        * amenities_effect
        * cancellation_effect
        * urgency_effect
        * season_effect
        * weekend_effect
        * length_effect
        * occupancy_effect
        * noise,
    )

    return pd.DataFrame(
        {
            "quote_time": quote_time,
            "check_in": check_in.date,
            "check_out": check_out.date,
            "destination": destination,
            "property_type": property_type,
            "hotel_class": hotel_class,
            "rating": np.round(rating, 2),
            "review_count": review_count,
            "distance_from_city_center_km": np.round(city_distance, 2),
            "distance_from_airport_km": np.round(airport_distance, 2),
            "amenity_count": amenity_count,
            "free_cancellation": free_cancellation,
            "adults": adults,
            HOTEL_TARGET: np.round(nightly_price, 2),
            "currency": "USD",
            "source": "synthetic_hotel_demo",
        }
    )


def hotel_temporal_split(frame: pd.DataFrame) -> HotelTemporalSplit:
    """Preserve time order and separate model selection from interval calibration."""

    if len(frame) < 500:
        raise ValueError("hotel model training requires at least 500 rows")
    split: TemporalSplit = temporal_split(frame, "quote_time")
    selection_calibration, interval_calibration = split_temporal_window(
        split.calibration,
        "quote_time",
    )
    return HotelTemporalSplit(
        train=split.train,
        selection_calibration=selection_calibration,
        interval_calibration=interval_calibration,
        test=split.test,
    )


def _preprocessor(feature_set: FeatureSet) -> ColumnTransformer:
    categorical = HOTEL_CORE_CATEGORICAL_FEATURES
    numeric = (
        HOTEL_CORE_NUMERIC_FEATURES
        if feature_set == "core"
        else HOTEL_ENRICHED_NUMERIC_FEATURES
    )
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
        ],
        sparse_threshold=0.0,
    )


def _pipeline(feature_set: FeatureSet, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(feature_set)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.07,
                    max_iter=180,
                    max_leaf_nodes=31,
                    l2_regularization=0.25,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _positive_target(frame: pd.DataFrame) -> np.ndarray:
    if HOTEL_TARGET not in frame:
        raise ValueError(f"hotel price data must contain {HOTEL_TARGET}")
    try:
        target = pd.to_numeric(frame[HOTEL_TARGET], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{HOTEL_TARGET} must be numeric") from exc
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError(f"{HOTEL_TARGET} must contain finite positive values")
    return target


def _finite_sample_quantile(residuals: np.ndarray, coverage: float) -> float:
    if residuals.size == 0:
        raise ValueError("hotel calibration residuals cannot be empty")
    level = min(1.0, math.ceil((residuals.size + 1) * coverage) / residuals.size)
    return float(np.quantile(residuals, level, method="higher"))


def _point_predictions(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, np.expm1(model.predict(features)))


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: float,
) -> dict[str, float]:
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "baseline_mae_usd": float(
            mean_absolute_error(actual, np.full_like(actual, baseline))
        ),
    }


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def train_hotel_price_model(
    frame: pd.DataFrame,
    *,
    data_mode: str = "synthetic_demo",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train core/enriched candidates and calibrate the selected hotel model.

    Both candidates use the same training rows.  The enriched model is selected
    only when it clears the pre-registered practical MAE threshold on the next
    chronological segment.  A later, untouched segment calibrates the 80%
    log-space conformal interval.
    """

    if not isinstance(data_mode, str) or not data_mode.strip():
        raise ValueError("data_mode is required")
    _positive_target(frame)
    split = hotel_temporal_split(frame)
    y_train = _positive_target(split.train)
    y_selection = _positive_target(split.selection_calibration)
    candidate_models: dict[FeatureSet, Pipeline] = {}
    selection_mae: dict[FeatureSet, float] = {}

    for feature_set in ("core", "enriched"):
        model = _pipeline(feature_set, random_state)
        model.fit(
            build_hotel_price_features(split.train, feature_set=feature_set),
            np.log1p(y_train),
        )
        prediction = _point_predictions(
            model,
            build_hotel_price_features(
                split.selection_calibration,
                feature_set=feature_set,
            ),
        )
        candidate_models[feature_set] = model
        selection_mae[feature_set] = float(mean_absolute_error(y_selection, prediction))

    minimum_practical_improvement_usd = 0.50
    enriched_improvement_usd = selection_mae["core"] - selection_mae["enriched"]
    selected: FeatureSet = (
        "enriched"
        if enriched_improvement_usd >= minimum_practical_improvement_usd
        else "core"
    )
    model = candidate_models[selected]
    interval_features = build_hotel_price_features(
        split.interval_calibration,
        feature_set=selected,
    )
    y_interval = _positive_target(split.interval_calibration)
    interval_prediction_log = model.predict(interval_features)
    log_half_width = _finite_sample_quantile(
        np.abs(np.log1p(y_interval) - interval_prediction_log),
        HOTEL_INTERVAL_COVERAGE,
    )

    test_features = build_hotel_price_features(split.test, feature_set=selected)
    y_test = _positive_target(split.test)
    test_prediction_log = model.predict(test_features)
    test_prediction = np.maximum(0.0, np.expm1(test_prediction_log))
    test_lower = np.maximum(0.0, np.expm1(test_prediction_log - log_half_width))
    test_upper = np.maximum(0.0, np.expm1(test_prediction_log + log_half_width))
    baseline = float(np.median(y_train))
    test_metrics = _regression_metrics(y_test, test_prediction, baseline)
    test_metrics.update(
        {
            "interval_80_empirical_coverage": float(
                np.mean((y_test >= test_lower) & (y_test <= test_upper))
            ),
            "interval_80_mean_width_usd": float(np.mean(test_upper - test_lower)),
        }
    )
    selected_features = (
        HOTEL_ENRICHED_FEATURES if selected == "enriched" else HOTEL_CORE_FEATURES
    )
    normalized_coverage = build_hotel_price_features(split.train, feature_set="core")
    metadata = {
        "hotel_model_schema_version": HOTEL_MODEL_SCHEMA_VERSION,
        "model_version": __version__,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "data_mode": data_mode.strip(),
        "synthetic_demo": data_mode.strip().startswith("synthetic"),
        "data_fingerprint_sha256": _frame_fingerprint(frame),
        "target_definition": (
            "provider-displayed average nightly USD price; tax inclusion and "
            "bookability are not inferred"
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "destination_coverage": {
            "known_destination_codes": sorted(
                normalized_coverage["destination"].unique().tolist()
            ),
            "destination_count": int(normalized_coverage["destination"].nunique()),
            "known_property_types": sorted(
                normalized_coverage["property_type"].unique().tolist()
            ),
            "property_type_count": int(normalized_coverage["property_type"].nunique()),
            "unseen_categories_are_not_validated_real_world_coverage": True,
        },
        "selected_feature_set": selected,
        "selected_features": list(selected_features),
        "feature_selection": {
            "selection_period": "chronological first half of the calibration split",
            "core_calibration_mae_usd": selection_mae["core"],
            "enriched_calibration_mae_usd": selection_mae["enriched"],
            "enriched_improvement_usd": enriched_improvement_usd,
            "minimum_practical_improvement_usd": minimum_practical_improvement_usd,
            "selected": selected,
            "rule": (
                "select enriched only when its calibration MAE improves by at least "
                "the pre-registered practical threshold"
            ),
        },
        "interval_calibration": {
            "method": "absolute residual conformal calibration in log1p price space",
            "target_coverage": HOTEL_INTERVAL_COVERAGE,
            "log_half_width": log_half_width,
            "period": "chronological second half of the calibration split",
        },
        "row_counts": {
            "all": len(frame),
            "train": len(split.train),
            "selection_calibration": len(split.selection_calibration),
            "interval_calibration": len(split.interval_calibration),
            "test": len(split.test),
        },
        "time_ranges": {
            key: {
                "start": pd.to_datetime(value["quote_time"], utc=True).min().isoformat(),
                "end": pd.to_datetime(value["quote_time"], utc=True).max().isoformat(),
            }
            for key, value in {
                "train": split.train,
                "selection_calibration": split.selection_calibration,
                "interval_calibration": split.interval_calibration,
                "test": split.test,
            }.items()
        },
        "leakage_policy": (
            "current/live prices, hotel identity, provider tokens, booking URLs, and "
            "post-stay outcomes are excluded from the feature whitelist"
        ),
    }
    return {
        "hotel_model_schema_version": HOTEL_MODEL_SCHEMA_VERSION,
        "model": model,
        "selected_feature_set": selected,
        "interval_coverage": HOTEL_INTERVAL_COVERAGE,
        "log_interval_half_width": log_half_width,
        "metrics": {
            "feature_selection": {
                "core_calibration_mae_usd": selection_mae["core"],
                "enriched_calibration_mae_usd": selection_mae["enriched"],
                "enriched_improvement_usd": enriched_improvement_usd,
                "minimum_practical_improvement_usd": minimum_practical_improvement_usd,
                "selected_feature_set": selected,
            },
            "test": test_metrics,
        },
        "metadata": metadata,
    }


def _validated_hotel_model_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a deserialized bundle's fixed, non-secret public contract."""

    schema_version = bundle.get("hotel_model_schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != HOTEL_MODEL_SCHEMA_VERSION
    ):
        raise ValueError("unsupported hotel model schema")
    selected = bundle.get("selected_feature_set")
    if selected not in {"core", "enriched"}:
        raise ValueError("hotel model feature set is invalid")
    model = bundle.get("model")
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ValueError("hotel model is missing")
    try:
        interval_coverage = float(bundle["interval_coverage"])
        half_width = float(bundle["log_interval_half_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("hotel model interval is invalid") from exc
    if (
        not math.isfinite(interval_coverage)
        or not math.isclose(interval_coverage, HOTEL_INTERVAL_COVERAGE)
        or not math.isfinite(half_width)
        or half_width < 0
    ):
        raise ValueError("hotel model interval is invalid")
    metadata = bundle.get("metadata")
    metrics = bundle.get("metrics")
    if not isinstance(metadata, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("hotel model metadata is invalid")
    if metadata.get("hotel_model_schema_version") != HOTEL_MODEL_SCHEMA_VERSION:
        raise ValueError("hotel model metadata schema is invalid")
    if metadata.get("selected_feature_set") != selected:
        raise ValueError("hotel model metadata feature set is inconsistent")
    if not isinstance(metadata.get("data_mode"), str) or not metadata["data_mode"].strip():
        raise ValueError("hotel model data mode is invalid")
    if not isinstance(metadata.get("model_version"), str) or not metadata[
        "model_version"
    ].strip():
        raise ValueError("hotel model version is invalid")
    try:
        trained_at = datetime.fromisoformat(str(metadata["trained_at_utc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("hotel model training timestamp is invalid") from exc
    if trained_at.utcoffset() is None:
        raise ValueError("hotel model training timestamp is invalid")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, Mapping) or any(
        not isinstance(runtime.get(key), str) or not runtime[key].strip()
        for key in ("python", "numpy", "pandas", "scikit_learn")
    ):
        raise ValueError("hotel model runtime metadata is invalid")
    coverage = metadata.get("destination_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("hotel model destination coverage is invalid")
    destination_codes = coverage.get("known_destination_codes")
    property_types = coverage.get("known_property_types")
    if (
        not isinstance(destination_codes, list)
        or not destination_codes
        or not all(isinstance(value, str) and value for value in destination_codes)
        or coverage.get("destination_count") != len(destination_codes)
        or not isinstance(property_types, list)
        or not property_types
        or not all(isinstance(value, str) and value for value in property_types)
        or coverage.get("property_type_count") != len(property_types)
    ):
        raise ValueError("hotel model destination coverage is invalid")
    fingerprint = metadata.get("data_fingerprint_sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("hotel model data fingerprint is invalid")
    return dict(bundle)


def save_hotel_price_model(
    bundle: Mapping[str, Any],
    model_dir: str | Path,
) -> Path:
    """Atomically save a locally trained hotel bundle under its fixed filename."""

    validated = _validated_hotel_model_bundle(bundle)
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / HOTEL_ARTIFACT_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{HOTEL_ARTIFACT_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        joblib.dump(validated, temporary_path, compress=3)
        os.replace(temporary_path, artifact_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return artifact_path


def load_hotel_price_model(model_dir: str | Path) -> dict[str, Any]:
    """Load a trusted local artifact and fail closed on a mismatched contract.

    Joblib uses pickle internally and can execute code during loading.  This
    helper is only for artifacts produced by the project's trusted build; it is
    not a safe parser for user uploads or downloaded model files.
    """

    artifact_path = Path(model_dir) / HOTEL_ARTIFACT_FILENAME
    if not artifact_path.is_file():
        raise FileNotFoundError(f"hotel model artifact not found at {artifact_path}")
    loaded = joblib.load(artifact_path)
    if not isinstance(loaded, Mapping):
        raise ValueError("hotel model artifact must contain a mapping")
    return _validated_hotel_model_bundle(loaded)


def predict_hotel_price(
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    quote_time: datetime | None = None,
) -> HotelPricePrediction:
    """Predict one stay without using any live/current price in ``row``."""

    validated_bundle = _validated_hotel_model_bundle(bundle)
    selected = validated_bundle["selected_feature_set"]
    model = validated_bundle["model"]
    half_width = float(validated_bundle["log_interval_half_width"])

    payload = dict(row)
    effective_quote_time = quote_time
    if effective_quote_time is None:
        effective_quote_time = payload.get("quote_time")
    if effective_quote_time is None:
        effective_quote_time = datetime.now(UTC)
    payload["quote_time"] = effective_quote_time
    frame = pd.DataFrame([payload])
    features = build_hotel_price_features(frame, feature_set=selected)
    prediction_log = float(model.predict(features)[0])
    point = max(0.0, float(np.expm1(prediction_log)))
    low = max(0.0, float(np.expm1(prediction_log - half_width)))
    high = max(0.0, float(np.expm1(prediction_log + half_width)))
    nights = int(features.iloc[0]["stay_nights"])
    lead_days = int(features.iloc[0]["days_until_check_in"])
    data_mode = str(validated_bundle["metadata"]["data_mode"])
    return HotelPricePrediction(
        estimated_nightly_price_usd=round(point, 2),
        interval_80_low_usd=round(low, 2),
        interval_80_high_usd=round(high, 2),
        estimated_stay_total_usd=round(point * nights, 2),
        interval_80_total_low_usd=round(low * nights, 2),
        interval_80_total_high_usd=round(high * nights, 2),
        stay_nights=nights,
        days_until_check_in=lead_days,
        feature_set=selected,
        data_mode=data_mode,
        model_schema_version=HOTEL_MODEL_SCHEMA_VERSION,
        warning=(
            "这是酒店价格模型估算，不是实时可订报价；不预测税费、库存或最终结账金额。"
            + (" 当前模型仅由合成演示数据训练。" if data_mode.startswith("synthetic") else "")
        ),
        warning_en=(
            "This hotel model estimate is not a live bookable quote; taxes, inventory, "
            "and the final checkout total are not predicted."
            + (
                " The current model is trained only on synthetic demo data."
                if data_mode.startswith("synthetic")
                else ""
            )
        ),
    )
