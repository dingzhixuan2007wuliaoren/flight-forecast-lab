from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from flight_forecaster.schemas import (
    ComparisonRequest,
    ContextDetailRequest,
    OnTimeRequest,
    PriceRequest,
)


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


def test_comparison_request_uses_a_canonical_date_and_requires_one() -> None:
    departure = date.today() + timedelta(days=30)
    request = ComparisonRequest.model_validate(
        {
            "origin": "JFK",
            "destination": "LAX",
            "departure_date": departure.isoformat(),
        }
    )
    assert request.departure_date == departure
    assert request.departure_time is None

    with pytest.raises(ValidationError, match="departure_date"):
        ComparisonRequest.model_validate({"origin": "JFK", "destination": "LAX"})

    with pytest.raises(ValidationError, match="exact dates"):
        ComparisonRequest.model_validate(
            {
                "origin": "JFK",
                "destination": "LAX",
                "departure_date": f"{departure.isoformat()}T09:00:00",
            }
        )


def test_context_detail_request_accepts_date_or_legacy_time_but_not_both() -> None:
    selected_date = date(2026, 8, 15)
    date_request = ContextDetailRequest(
        origin="jfk",
        destination="lax",
        departure_date=selected_date,
    )
    assert date_request.departure_date == selected_date
    assert date_request.departure_time is None

    legacy_request = ContextDetailRequest(
        origin="JFK",
        destination="LAX",
        departure_time=datetime(2026, 8, 15, 9, 30),
    )
    assert legacy_request.departure_date is None
    assert legacy_request.departure_time == datetime(2026, 8, 15, 9, 30)

    with pytest.raises(ValidationError, match="exactly one"):
        ContextDetailRequest(origin="JFK", destination="LAX")
    with pytest.raises(ValidationError, match="exactly one"):
        ContextDetailRequest(
            origin="JFK",
            destination="LAX",
            departure_date=selected_date,
            departure_time=datetime(2026, 8, 15, 9, 30),
        )
