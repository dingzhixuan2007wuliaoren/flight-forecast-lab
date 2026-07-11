from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_forecaster.features import build_ontime_features, build_price_features
from flight_forecaster.schemas import (
    OnTimePrediction,
    OnTimeRequest,
    PricePrediction,
    PriceRequest,
)
from flight_forecaster.training import ARTIFACT_FILENAME, SCHEMA_VERSION


class PredictionService:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        artifact_path = self.model_dir / ARTIFACT_FILENAME
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"model artifact not found at {artifact_path}; run train-demo first"
            )
        # joblib/pickle can execute code while loading. Only load locally produced artifacts.
        self.bundle: dict[str, Any] = joblib.load(artifact_path)
        if self.bundle.get("artifact_schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported model artifact schema")

    @property
    def model_version(self) -> str:
        return str(self.bundle["model_version"])

    def predict_price(self, request: PriceRequest) -> PricePrediction:
        row = pd.DataFrame([request.model_dump()])
        features = build_price_features(row)
        estimate = float(np.expm1(self.bundle["price_model"].predict(features)[0]))
        estimate = max(0.0, estimate)
        half_width = float(self.bundle["price_interval_half_width_usd"])
        lead_days = (request.departure_time - request.quote_time).total_seconds() / 86_400.0
        return PricePrediction(
            estimated_price_usd=round(estimate, 2),
            interval_80_low_usd=round(max(0.0, estimate - half_width), 2),
            interval_80_high_usd=round(estimate + half_width, 2),
            days_until_departure=round(lead_days, 1),
            model_version=self.model_version,
            warning="条件估价并非实时可购买报价，也不保证最低价。",
        )

    def predict_ontime(self, request: OnTimeRequest) -> OnTimePrediction:
        row = pd.DataFrame([request.model_dump()])
        features = build_ontime_features(row)
        probability = float(self.bundle["ontime_model"].predict_proba(features)[0, 1])
        probability = float(np.clip(probability, 0.0, 1.0))
        if probability >= 0.80:
            risk = "low"
        elif probability >= 0.60:
            risk = "medium"
        else:
            risk = "high"
        return OnTimePrediction(
            on_time_probability=round(probability, 4),
            disruption_probability=round(1.0 - probability, 4),
            risk_level=risk,
            definition="未取消且到达延误少于 15 分钟",
            model_version=self.model_version,
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "available_tasks": ["fare_estimation", "on_time_probability"],
            "metadata": self.bundle["metadata"],
            "metrics": self.bundle["metrics"],
        }
