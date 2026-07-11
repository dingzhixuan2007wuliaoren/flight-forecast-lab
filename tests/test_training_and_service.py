import json
from pathlib import Path

import joblib

from flight_forecaster.schemas import OnTimeRequest, PriceRequest
from flight_forecaster.service import PredictionService
from flight_forecaster.training import ARTIFACT_FILENAME


def test_training_writes_versioned_artifacts_and_beats_baselines(
    trained_model_dir: Path,
) -> None:
    bundle = joblib.load(trained_model_dir / ARTIFACT_FILENAME)
    assert bundle["artifact_schema_version"] == 1
    assert bundle["metrics"]["price"]["mae_usd"] < bundle["metrics"]["price"]["baseline_mae_usd"]
    assert (
        bundle["metrics"]["on_time"]["brier_score"]
        < bundle["metrics"]["on_time"]["baseline_brier_score"]
    )
    json.loads((trained_model_dir / "metrics.json").read_text(encoding="utf-8"))
    json.loads((trained_model_dir / "metadata.json").read_text(encoding="utf-8"))


def test_service_predictions_are_bounded(trained_model_dir: Path) -> None:
    service = PredictionService(trained_model_dir)
    price = service.predict_price(
        PriceRequest.model_validate(
            {
                "origin": "JFK",
                "destination": "LAX",
                "airline": "DL",
                "cabin": "economy",
                "stops": 0,
                "duration_minutes": 365,
                "distance_km": 3983,
                "quote_time": "2026-07-15T12:00:00-04:00",
                "departure_time": "2026-09-15T08:00:00-04:00",
            }
        )
    )
    on_time = service.predict_ontime(
        OnTimeRequest.model_validate(
            {
                "origin": "ZZZ",
                "destination": "YYY",
                "airline": "XY",
                "distance_km": 1800,
                "scheduled_departure": "2026-09-15T08:00:00-04:00",
                "weather_severity_forecast": 0.4,
                "origin_congestion_index": 0.5,
            }
        )
    )
    assert 0 <= price.interval_80_low_usd <= price.estimated_price_usd
    assert price.interval_80_high_usd >= price.estimated_price_usd
    assert 0 <= on_time.on_time_probability <= 1
    assert abs(on_time.on_time_probability + on_time.disruption_probability - 1) < 0.0011
