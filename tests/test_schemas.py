from datetime import datetime

import pytest
from pydantic import ValidationError

from flight_forecaster.schemas import OnTimeRequest, PriceRequest


def test_price_request_rejects_naive_datetimes() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        PriceRequest(
            origin="JFK",
            destination="LAX",
            airline="DL",
            cabin="economy",
            stops=0,
            duration_minutes=365,
            distance_km=3983,
            quote_time=datetime(2026, 7, 1, 12, 0),
            departure_time=datetime(2026, 8, 1, 12, 0),
        )


def test_codes_are_normalised() -> None:
    request = OnTimeRequest.model_validate(
        {
            "origin": "jfk",
            "destination": "lax",
            "airline": "dl",
            "distance_km": 3983,
            "scheduled_departure": "2026-09-15T08:00:00-04:00",
            "weather_severity_forecast": 0.2,
            "origin_congestion_index": 0.6,
        }
    )
    assert request.origin == "JFK"
    assert request.airline == "DL"
