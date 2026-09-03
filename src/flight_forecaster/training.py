from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_forecaster import __version__
from flight_forecaster.features import (
    LEGACY_PRICE_CATEGORICAL_FEATURES,
    LEGACY_PRICE_NUMERIC_FEATURES,
    ONTIME_CATEGORICAL_FEATURES,
    ONTIME_NUMERIC_FEATURES,
    ONTIME_NUMERIC_FEATURES_WITHOUT_WEATHER,
    PRICE_CATEGORICAL_FEATURES,
    PRICE_NUMERIC_FEATURES,
    build_ontime_features,
    build_ontime_features_without_weather,
    build_price_features,
)

ARTIFACT_FILENAME = "model_bundle.joblib"
SCHEMA_VERSION = 4


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def split_temporal_window(
    frame: pd.DataFrame,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one chronological window without dividing an equal-time snapshot."""

    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(
        ordered[time_column], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(time_column, kind="stable").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("temporal window is too small to split")
    cumulative_group_sizes = (
        ordered.groupby(time_column, sort=False).size().cumsum().iloc[:-1]
    )
    if cumulative_group_sizes.empty:
        raise ValueError("temporal window needs at least two distinct timestamps")
    target = len(ordered) / 2
    split_at = int(
        min(cumulative_group_sizes.to_numpy(int), key=lambda value: abs(value - target))
    )
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def temporal_split(
    frame: pd.DataFrame,
    time_column: str,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> TemporalSplit:
    if len(frame) < 100:
        raise ValueError("at least 100 rows are required for a temporal split")
    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True, errors="raise")
    ordered = ordered.sort_values(time_column, kind="stable").reset_index(drop=True)
    train_target = int(len(ordered) * train_fraction)
    calibration_target = int(len(ordered) * (train_fraction + calibration_fraction))
    train_boundary = ordered.iloc[train_target - 1][time_column]
    calibration_boundary = ordered.iloc[calibration_target - 1][time_column]
    # A provider snapshot can contain several itineraries with the same timestamp.
    # Keep that whole snapshot in one split instead of leaking adjacent rows forward.
    train_end = int(ordered[time_column].searchsorted(train_boundary, side="right"))
    calibration_end = int(
        ordered[time_column].searchsorted(calibration_boundary, side="right")
    )
    if not 0 < train_end < calibration_end < len(ordered):
        raise ValueError("invalid temporal split fractions")
    return TemporalSplit(
        train=ordered.iloc[:train_end].copy(),
        calibration=ordered.iloc[train_end:calibration_end].copy(),
        test=ordered.iloc[calibration_end:].copy(),
    )


def _preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        sparse_threshold=0.0,
    )


def _price_pipeline(
    random_state: int,
    *,
    categorical_features: list[str] = PRICE_CATEGORICAL_FEATURES,
    numeric_features: list[str] = PRICE_NUMERIC_FEATURES,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(categorical_features, numeric_features)),
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


def _paired_absolute_error_delta(
    actual: np.ndarray,
    reference_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    random_state: int,
    bootstrap_samples: int = 1_000,
    groups: pd.Series | np.ndarray | None = None,
    confidence_level: float = 0.95,
) -> dict[str, float | str | int]:
    """Summarise paired MAE improvement with an optional block bootstrap."""

    if not 0 < confidence_level < 1:
        raise ValueError("bootstrap confidence level must be between 0 and 1")
    delta = np.abs(actual - reference_prediction) - np.abs(actual - candidate_prediction)
    if delta.size == 0:
        raise ValueError("paired factor evaluation requires at least one row")
    if groups is not None:
        group_values = np.asarray(groups)
        if group_values.shape[0] != delta.size:
            raise ValueError("paired factor evaluation groups must match the rows")
        bootstrap_values = (
            pd.DataFrame({"group": group_values, "delta": delta})
            .groupby("group", sort=False)["delta"]
            .mean()
            .to_numpy(float)
        )
        bootstrap_unit = "quote_date_equal_weighted"
    else:
        bootstrap_values = delta
        bootstrap_unit = "row"
    rng = np.random.default_rng(random_state)
    block_count = bootstrap_values.size
    indices = rng.integers(0, block_count, size=(bootstrap_samples, block_count))
    sampled = bootstrap_values[indices].mean(axis=1)
    alpha = 1 - confidence_level
    return {
        "mean_mae_improvement_usd": float(bootstrap_values.mean()),
        "bootstrap_low_usd": float(np.quantile(sampled, alpha / 2)),
        "bootstrap_high_usd": float(np.quantile(sampled, 1 - alpha / 2)),
        "bootstrap_confidence_level": confidence_level,
        "bootstrap_unit": bootstrap_unit,
        "independent_block_count": block_count,
    }


def _ontime_pipeline(
    random_state: int,
    *,
    numeric_features: list[str] = ONTIME_NUMERIC_FEATURES,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(ONTIME_CATEGORICAL_FEATURES, numeric_features)),
            (
                "model",
                LogisticRegression(
                    C=0.75,
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _ontime_pipeline_without_weather(random_state: int) -> Pipeline:
    """Use a small non-linear model for the weaker no-weather signal set."""

    return Pipeline(
        [
            (
                "preprocess",
                _preprocessor(
                    ONTIME_CATEGORICAL_FEATURES,
                    ONTIME_NUMERIC_FEATURES_WITHOUT_WEATHER,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    max_leaf_nodes=7,
                    l2_regularization=0.5,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _finite_sample_quantile(residuals: np.ndarray, coverage: float) -> float:
    if residuals.size == 0:
        raise ValueError("calibration residuals cannot be empty")
    level = min(1.0, np.ceil((residuals.size + 1) * coverage) / residuals.size)
    return float(np.quantile(residuals, level, method="higher"))


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    sample = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(sample).hexdigest()


def _price_metrics(actual: np.ndarray, predicted: np.ndarray, baseline: float) -> dict[str, float]:
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "baseline_mae_usd": float(mean_absolute_error(actual, np.full_like(actual, baseline))),
    }


def _ontime_metrics(
    actual: np.ndarray, probability: np.ndarray, baseline: float
) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "brier_score": float(brier_score_loss(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(actual, predicted)),
        "baseline_brier_score": float(
            brier_score_loss(actual, np.full_like(probability, baseline))
        ),
    }


def _context_priors(frame: pd.DataFrame, *, data_mode: str) -> dict[str, Any]:
    """Build fallback averages from training rows only, never from held-out future rows."""

    working = frame.copy()
    departure = pd.to_datetime(working["scheduled_departure"], utc=True, errors="raise")
    weather = pd.to_numeric(working["weather_severity_forecast"], errors="raise").clip(0, 1)
    operations = pd.to_numeric(working["origin_congestion_index"], errors="raise").clip(0, 1)
    departure_month = (
        pd.to_numeric(working["departure_local_month"], errors="raise")
        if "departure_local_month" in working
        else departure.dt.month
    )
    weather_by_month = weather.groupby(departure_month).mean()
    operations_by_origin = operations.groupby(working["origin"].astype(str).str.upper()).mean()
    source = f"{data_mode}_training_average"
    return {
        "status": "proxy",
        "source": source,
        "weather_global": round(float(weather.mean()), 4),
        "weather_by_month": {
            str(int(month)): round(float(value), 4) for month, value in weather_by_month.items()
        },
        "operations_global": round(float(operations.mean()), 4),
        "operations_by_origin": {
            str(origin): round(float(value), 4) for origin, value in operations_by_origin.items()
        },
    }


def train_models(
    price_data: pd.DataFrame,
    ontime_data: pd.DataFrame,
    output_dir: str | Path,
    *,
    data_mode: str,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the fare model and both on-time variants, then persist a versioned bundle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if "price_usd" not in price_data:
        raise ValueError("price data must contain price_usd")
    if "on_time" not in ontime_data:
        raise ValueError("on-time data must contain on_time")

    price_data = price_data.copy()
    ontime_data = ontime_data.copy()
    price_target = pd.to_numeric(price_data["price_usd"], errors="raise").to_numpy(float)
    if not np.isfinite(price_target).all() or (price_target <= 0).any():
        raise ValueError("price_usd must contain finite positive values")
    price_data["price_usd"] = price_target
    ontime_target = pd.to_numeric(ontime_data["on_time"], errors="raise").to_numpy(float)
    if not np.isfinite(ontime_target).all() or not set(np.unique(ontime_target)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("on_time must contain only 0 and 1")
    ontime_data["on_time"] = ontime_target.astype(int)

    price_split = temporal_split(price_data, "quote_time")
    ontime_split = temporal_split(ontime_data, "scheduled_departure")

    price_selection_calibration, price_interval_calibration = split_temporal_window(
        price_split.calibration,
        "quote_time",
    )

    x_price_train = build_price_features(price_split.train)
    y_price_train = pd.to_numeric(price_split.train["price_usd"], errors="raise").to_numpy(float)
    x_price_selection = build_price_features(price_selection_calibration)
    y_price_selection = price_selection_calibration["price_usd"].to_numpy(float)
    selection_groups = pd.to_datetime(
        price_selection_calibration["quote_time"], utc=True
    ).dt.date
    candidate_categories = {
        "legacy": list(LEGACY_PRICE_CATEGORICAL_FEATURES),
        "route_scope": [*LEGACY_PRICE_CATEGORICAL_FEATURES, "route_scope"],
        "carrier_service_model": [
            *LEGACY_PRICE_CATEGORICAL_FEATURES,
            "carrier_service_model",
        ],
        "enriched": list(PRICE_CATEGORICAL_FEATURES),
    }
    candidate_models: dict[str, Pipeline] = {}
    candidate_selection_predictions: dict[str, np.ndarray] = {}
    candidate_selection_mae: dict[str, float] = {}
    for variant, categorical_features in candidate_categories.items():
        candidate = _price_pipeline(
            random_state,
            categorical_features=categorical_features,
            numeric_features=LEGACY_PRICE_NUMERIC_FEATURES,
        )
        candidate.fit(x_price_train, np.log1p(y_price_train))
        selection_prediction = np.maximum(
            0,
            np.expm1(candidate.predict(x_price_selection)),
        )
        candidate_models[variant] = candidate
        candidate_selection_predictions[variant] = selection_prediction
        candidate_selection_mae[variant] = float(
            pd.DataFrame(
                {
                    "group": selection_groups,
                    "absolute_error": np.abs(y_price_selection - selection_prediction),
                }
            )
            .groupby("group", sort=False)["absolute_error"]
            .mean()
            .mean()
        )

    nonlegacy_variants = [variant for variant in candidate_models if variant != "legacy"]
    familywise_confidence_level = 0.95
    selection_confidence_level = 1 - (
        (1 - familywise_confidence_level) / len(nonlegacy_variants)
    )
    selection_evidence = {
        variant: _paired_absolute_error_delta(
            y_price_selection,
            candidate_selection_predictions["legacy"],
            prediction,
            random_state=random_state,
            groups=selection_groups,
            confidence_level=selection_confidence_level,
        )
        for variant, prediction in candidate_selection_predictions.items()
        if variant != "legacy"
    }
    minimum_practical_improvement_usd = 0.50
    minimum_independent_blocks = 5
    candidate_eligibility = {}
    for variant, evidence in selection_evidence.items():
        improvement = candidate_selection_mae["legacy"] - candidate_selection_mae[variant]
        candidate_eligibility[variant] = {
            "improvement_over_legacy_usd": improvement,
            "meets_practical_threshold": improvement
            >= minimum_practical_improvement_usd,
            "meets_uncertainty_threshold": float(evidence["bootstrap_low_usd"]) > 0,
            "has_enough_independent_blocks": int(evidence["independent_block_count"])
            >= minimum_independent_blocks,
        }
        candidate_eligibility[variant]["eligible"] = all(
            (
                candidate_eligibility[variant]["meets_practical_threshold"],
                candidate_eligibility[variant]["meets_uncertainty_threshold"],
                candidate_eligibility[variant]["has_enough_independent_blocks"],
            )
        )
    eligible_variants = [
        variant
        for variant, eligibility in candidate_eligibility.items()
        if eligibility["eligible"]
    ]
    price_feature_variant = (
        min(eligible_variants, key=candidate_selection_mae.get)
        if eligible_variants
        else "legacy"
    )
    price_model = candidate_models[price_feature_variant]
    x_price_interval = build_price_features(price_interval_calibration)
    y_price_interval = price_interval_calibration["price_usd"].to_numpy(float)
    price_interval_prediction = np.maximum(
        0,
        np.expm1(price_model.predict(x_price_interval)),
    )
    interval_half_width = _finite_sample_quantile(
        np.abs(y_price_interval - price_interval_prediction), coverage=0.80
    )
    x_price_test = build_price_features(price_split.test)
    y_price_test = price_split.test["price_usd"].to_numpy(float)
    price_test_prediction = np.maximum(0, np.expm1(price_model.predict(x_price_test)))
    price_baseline = float(np.median(y_price_train))
    price_metrics = _price_metrics(y_price_test, price_test_prediction, price_baseline)
    lower = np.maximum(0, price_test_prediction - interval_half_width)
    upper = price_test_prediction + interval_half_width
    price_metrics["interval_80_empirical_coverage"] = float(
        np.mean((y_price_test >= lower) & (y_price_test <= upper))
    )
    price_metrics["interval_80_mean_width_usd"] = float(np.mean(upper - lower))
    candidate_test_predictions = {
        variant: np.maximum(0, np.expm1(model.predict(x_price_test)))
        for variant, model in candidate_models.items()
    }
    selected_factors = {
        "legacy": [],
        "route_scope": ["route_scope"],
        "carrier_service_model": ["carrier_service_model"],
        "enriched": ["route_scope", "carrier_service_model"],
    }[price_feature_variant]
    test_groups = pd.to_datetime(price_split.test["quote_time"], utc=True).dt.date
    price_metrics["factor_selection"] = {
        "selected_variant": price_feature_variant,
        "selected_factors": selected_factors,
        "selection_window": "chronological first half of calibration",
        "interval_window": "chronological second half of calibration",
        "legacy_calibration_mae_usd": candidate_selection_mae["legacy"],
        "enriched_calibration_mae_usd": candidate_selection_mae["enriched"],
        "candidate_selection_mae_usd": candidate_selection_mae,
        "candidate_selection_evidence": selection_evidence,
        "candidate_eligibility": candidate_eligibility,
        "pre_registered_minimum_improvement_usd": minimum_practical_improvement_usd,
        "minimum_independent_quote_date_blocks": minimum_independent_blocks,
        "multiple_comparison_method": "Bonferroni-adjusted paired block bootstrap",
        "familywise_confidence_level": familywise_confidence_level,
        "per_candidate_confidence_level": selection_confidence_level,
        "enriched_factors": ["route_scope", "carrier_service_model"],
        "paired_test_delta": _paired_absolute_error_delta(
            y_price_test,
            candidate_test_predictions["legacy"],
            candidate_test_predictions[price_feature_variant],
            random_state=random_state,
            groups=test_groups,
        ),
        "synthetic_validation_only": data_mode.startswith("synthetic")
        or "synthetic" in data_mode,
    }

    ontime_model = _ontime_pipeline(random_state)
    x_ontime_train = build_ontime_features(ontime_split.train)
    y_ontime_train = ontime_split.train["on_time"].astype(int).to_numpy()
    if set(np.unique(y_ontime_train)) != {0, 1}:
        raise ValueError("on_time must contain both 0 and 1 in the training period")
    ontime_model.fit(x_ontime_train, y_ontime_train)
    x_ontime_test = build_ontime_features(ontime_split.test)
    y_ontime_test = ontime_split.test["on_time"].astype(int).to_numpy()
    ontime_probability = ontime_model.predict_proba(x_ontime_test)[:, 1]
    ontime_baseline = float(np.mean(y_ontime_train))
    ontime_metrics = _ontime_metrics(y_ontime_test, ontime_probability, ontime_baseline)

    ontime_model_without_weather = _ontime_pipeline_without_weather(random_state)
    x_ontime_train_without_weather = build_ontime_features_without_weather(ontime_split.train)
    ontime_model_without_weather.fit(
        x_ontime_train_without_weather,
        y_ontime_train,
    )
    x_ontime_test_without_weather = build_ontime_features_without_weather(ontime_split.test)
    ontime_probability_without_weather = ontime_model_without_weather.predict_proba(
        x_ontime_test_without_weather
    )[:, 1]
    ontime_metrics_without_weather = _ontime_metrics(
        y_ontime_test,
        ontime_probability_without_weather,
        ontime_baseline,
    )
    context_priors = _context_priors(ontime_split.train, data_mode=data_mode)

    metrics = {
        "price": price_metrics,
        "on_time": ontime_metrics,
        "on_time_without_weather": ontime_metrics_without_weather,
    }
    trained_at = datetime.now(UTC).isoformat()
    metadata = {
        "artifact_schema_version": SCHEMA_VERSION,
        "model_version": __version__,
        "trained_at_utc": trained_at,
        "data_mode": data_mode,
        "random_state": random_state,
        "row_counts": {
            "price": len(price_data),
            "price_train": len(price_split.train),
            "price_selection_calibration": len(price_selection_calibration),
            "price_interval_calibration": len(price_interval_calibration),
            "on_time": len(ontime_data),
        },
        "test_row_counts": {"price": len(price_split.test), "on_time": len(ontime_split.test)},
        "data_fingerprints": {
            "price_sha256": _frame_fingerprint(price_data),
            "on_time_sha256": _frame_fingerprint(ontime_data),
        },
        "data_time_ranges": {
            "price": {
                "start": pd.to_datetime(price_data["quote_time"], utc=True).min().isoformat(),
                "end": pd.to_datetime(price_data["quote_time"], utc=True).max().isoformat(),
            },
            "price_selection_calibration": {
                "start": pd.to_datetime(
                    price_selection_calibration["quote_time"], utc=True
                ).min().isoformat(),
                "end": pd.to_datetime(
                    price_selection_calibration["quote_time"], utc=True
                ).max().isoformat(),
                "distinct_quote_date_blocks": int(selection_groups.nunique()),
            },
            "price_interval_calibration": {
                "start": pd.to_datetime(
                    price_interval_calibration["quote_time"], utc=True
                ).min().isoformat(),
                "end": pd.to_datetime(
                    price_interval_calibration["quote_time"], utc=True
                ).max().isoformat(),
            },
            "on_time": {
                "start": pd.to_datetime(ontime_data["scheduled_departure"], utc=True)
                .min()
                .isoformat(),
                "end": pd.to_datetime(ontime_data["scheduled_departure"], utc=True)
                .max()
                .isoformat(),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "target_definitions": {
            "price": "USD fare estimate conditional on itinerary and booking lead time",
            "on_time": "not cancelled and arrival delay below 15 minutes",
        },
        "runtime_context_features": {
            "route_scope": (
                "domestic/international/unknown derived from versioned airport countries"
            ),
            "carrier_service_model": (
                "full_service/hybrid/low_cost/unknown from the versioned airline catalog"
            ),
            "news_disruption_index": (
                "bounded recent-news signal; synthetic relationship in demo training"
            ),
            "weather_severity_forecast": (
                "used only when live or forecast weather is usable; otherwise the "
                "separate no-weather on-time model is selected"
            ),
            "origin_congestion_index": "resolved automatically at prediction time",
        },
        "context_prior": {
            "source": context_priors["source"],
            "status": context_priors["status"],
            "weather_month_groups": len(context_priors["weather_by_month"]),
            "operations_airport_groups": len(context_priors["operations_by_origin"]),
            "training_rows_only": True,
        },
        "fare_feature_selection": price_metrics["factor_selection"],
    }
    bundle = {
        "artifact_schema_version": SCHEMA_VERSION,
        "model_version": __version__,
        "price_model": price_model,
        "price_feature_variant": price_feature_variant,
        "price_interval_half_width_usd": interval_half_width,
        "ontime_model": ontime_model,
        "ontime_model_without_weather": ontime_model_without_weather,
        "metrics": metrics,
        "metadata": metadata,
        "context_priors": context_priors,
    }
    joblib.dump(bundle, output_path / ARTIFACT_FILENAME, compress=3)
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_path / "report.md").write_text(
        _render_report(metrics, metadata, interval_half_width), encoding="utf-8"
    )
    return bundle


def _render_report(
    metrics: dict[str, dict[str, Any]], metadata: dict[str, Any], interval: float
) -> str:
    price = metrics["price"]
    on_time = metrics["on_time"]
    on_time_without_weather = metrics["on_time_without_weather"]
    fare_factors = price["factor_selection"]
    paired_delta = fare_factors["paired_test_delta"]
    selected_variant = fare_factors["selected_variant"]
    enriched_selection = fare_factors["candidate_selection_evidence"]["enriched"]
    selection_confidence = 100 * float(
        enriched_selection["bootstrap_confidence_level"]
    )
    held_out_comparison = (
        "- Held-out selected-vs-legacy improvement: not applicable "
        "(legacy retained; zero by definition)"
        if selected_variant == "legacy"
        else (
            f"- Held-out `{selected_variant}`-vs-legacy MAE improvement: "
            f"`${paired_delta['mean_mae_improvement_usd']:.2f}` "
            f"(`{100 * float(paired_delta['bootstrap_confidence_level']):.1f}%` "
            f"block-bootstrap interval `${paired_delta['bootstrap_low_usd']:.2f}` to "
            f"`${paired_delta['bootstrap_high_usd']:.2f}`)"
        )
    )
    return f"""# Demo training report

Generated at `{metadata["trained_at_utc"]}` using `{metadata["data_mode"]}` data.

## Fare model

- Test MAE: `${price["mae_usd"]:.2f}`
- Naive median baseline MAE: `${price["baseline_mae_usd"]:.2f}`
- RMSE: `${price["rmse_usd"]:.2f}`
- R²: `{price["r2"]:.3f}`
- 80% conformal interval half-width: `${interval:.2f}`
- Empirical interval coverage: `{price["interval_80_empirical_coverage"]:.3f}`
- Factor set selected on the selection window: `{selected_variant}`
- Selected factors: `{", ".join(fare_factors["selected_factors"]) or "none"}`
- Legacy selection-window MAE: `${fare_factors["legacy_calibration_mae_usd"]:.2f}`
- Enriched selection-window MAE: `${fare_factors["enriched_calibration_mae_usd"]:.2f}`
- Enriched selection-window paired MAE improvement: \
`${enriched_selection["mean_mae_improvement_usd"]:.2f}` \
(`{selection_confidence:.1f}%` Bonferroni-adjusted block-bootstrap interval \
`${enriched_selection["bootstrap_low_usd"]:.2f}` to \
`${enriched_selection["bootstrap_high_usd"]:.2f}`)
{held_out_comparison}

## On-time model

- Brier score: `{on_time["brier_score"]:.4f}` (lower is better)
- Naive-rate baseline Brier score: `{on_time["baseline_brier_score"]:.4f}`
- ROC AUC: `{on_time["roc_auc"]:.4f}`
- Log loss: `{on_time["log_loss"]:.4f}`

## On-time model without weather

- Brier score: `{on_time_without_weather["brier_score"]:.4f}` (lower is better)
- Naive-rate baseline Brier score: `{on_time_without_weather["baseline_brier_score"]:.4f}`
- ROC AUC: `{on_time_without_weather["roc_auc"]:.4f}`
- Log loss: `{on_time_without_weather["log_loss"]:.4f}`

The weather-enhanced model is used only for usable live or forecast weather.
All other weather states select this separate no-weather model; no proxy value is
inserted into the prediction.

> These numbers describe a deterministic synthetic-data demo. They are pipeline checks,
> not evidence of production performance. Retrain and re-evaluate on representative data.
"""
