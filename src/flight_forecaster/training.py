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
from sklearn.ensemble import HistGradientBoostingRegressor
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
    ONTIME_CATEGORICAL_FEATURES,
    ONTIME_NUMERIC_FEATURES,
    PRICE_CATEGORICAL_FEATURES,
    PRICE_NUMERIC_FEATURES,
    build_ontime_features,
    build_price_features,
)

ARTIFACT_FILENAME = "model_bundle.joblib"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


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
    train_end = int(len(ordered) * train_fraction)
    calibration_end = int(len(ordered) * (train_fraction + calibration_fraction))
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


def _price_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(PRICE_CATEGORICAL_FEATURES, PRICE_NUMERIC_FEATURES)),
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


def _ontime_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(ONTIME_CATEGORICAL_FEATURES, ONTIME_NUMERIC_FEATURES)),
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


def train_models(
    price_data: pd.DataFrame,
    ontime_data: pd.DataFrame,
    output_dir: str | Path,
    *,
    data_mode: str,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train both models, evaluate on future rows, and persist a versioned bundle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if "price_usd" not in price_data:
        raise ValueError("price data must contain price_usd")
    if "on_time" not in ontime_data:
        raise ValueError("on-time data must contain on_time")

    price_split = temporal_split(price_data, "quote_time")
    ontime_split = temporal_split(ontime_data, "scheduled_departure")

    price_model = _price_pipeline(random_state)
    x_price_train = build_price_features(price_split.train)
    y_price_train = pd.to_numeric(price_split.train["price_usd"], errors="raise").to_numpy(float)
    if (y_price_train <= 0).any():
        raise ValueError("price_usd must be positive")
    price_model.fit(x_price_train, np.log1p(y_price_train))

    x_price_calibration = build_price_features(price_split.calibration)
    y_price_calibration = price_split.calibration["price_usd"].to_numpy(float)
    price_calibration_prediction = np.expm1(price_model.predict(x_price_calibration))
    interval_half_width = _finite_sample_quantile(
        np.abs(y_price_calibration - price_calibration_prediction), coverage=0.80
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

    metrics = {"price": price_metrics, "on_time": ontime_metrics}
    trained_at = datetime.now(UTC).isoformat()
    metadata = {
        "artifact_schema_version": SCHEMA_VERSION,
        "model_version": __version__,
        "trained_at_utc": trained_at,
        "data_mode": data_mode,
        "random_state": random_state,
        "row_counts": {"price": len(price_data), "on_time": len(ontime_data)},
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
    }
    bundle = {
        "artifact_schema_version": SCHEMA_VERSION,
        "model_version": __version__,
        "price_model": price_model,
        "price_interval_half_width_usd": interval_half_width,
        "ontime_model": ontime_model,
        "metrics": metrics,
        "metadata": metadata,
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
    metrics: dict[str, dict[str, float]], metadata: dict[str, Any], interval: float
) -> str:
    price = metrics["price"]
    on_time = metrics["on_time"]
    return f"""# Demo training report

Generated at `{metadata["trained_at_utc"]}` using `{metadata["data_mode"]}` data.

## Fare model

- Test MAE: `${price["mae_usd"]:.2f}`
- Naive median baseline MAE: `${price["baseline_mae_usd"]:.2f}`
- RMSE: `${price["rmse_usd"]:.2f}`
- R²: `{price["r2"]:.3f}`
- 80% conformal interval half-width: `${interval:.2f}`
- Empirical interval coverage: `{price["interval_80_empirical_coverage"]:.3f}`

## On-time model

- Brier score: `{on_time["brier_score"]:.4f}` (lower is better)
- Naive-rate baseline Brier score: `{on_time["baseline_brier_score"]:.4f}`
- ROC AUC: `{on_time["roc_auc"]:.4f}`
- Log loss: `{on_time["log_loss"]:.4f}`

> These numbers describe a deterministic synthetic-data demo. They are pipeline checks,
> not evidence of production performance. Retrain and re-evaluate on representative data.
"""
