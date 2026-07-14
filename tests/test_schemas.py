from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from flight_forecaster.schemas import ComparisonRequest, OnTimeRequest, PriceRequest


def test_price_request_rejects_naive_datetimes() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        PriceRequest(
            origin="JFK",
            destination="LAX",
            airline="DL",
            cabin="economy",
            stops=0,
            departure_time=datetime.now() + timedelta(days=30),
        )


def test_price_request_rejects_removed_quote_time() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        PriceRequest.model_validate(
            {
                "origin": "JFK",
                "destination": "LAX",
                "airline": "DL",
                "departure_time": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
                "quote_time": datetime.now(UTC).isoformat(),
            }
        )


def test_codes_are_normalised() -> None:
    request = OnTimeRequest.model_validate(
        {
            "origin": "jfk",
            "destination": "lax",
            "airline": "dl",
            "scheduled_departure": (datetime.now(UTC) + timedelta(days=60)).isoformat(),
        }
    )
    assert request.origin == "JFK"
    assert request.airline == "DL"


def test_on_time_request_rejects_past_departure() -> None:
    with pytest.raises(ValidationError, match="future"):
        OnTimeRequest.model_validate(
            {
                "origin": "JFK",
                "destination": "LAX",
                "airline": "DL",
                "scheduled_departure": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            }
        )

    with pytest.raises(ValidationError, match="370 days"):
        OnTimeRequest.model_validate(
            {
                "origin": "JFK",
                "destination": "LAX",
                "airline": "DL",
                "scheduled_departure": (datetime.now(UTC) + timedelta(days=371)).isoformat(),
            }
        )


def test_comparison_request_only_needs_route_and_departure() -> None:
    departure = (datetime.now(UTC) + timedelta(days=60)).replace(tzinfo=None)
    request = ComparisonRequest.model_validate(
        {
            "origin": "yyz",
            "destination": "lhr",
            "departure_time": departure.isoformat(timespec="minutes"),
        }
    )
    assert request.origin == "YYZ"
    assert request.destination == "LHR"
    assert request.departure_time.tzinfo is None
