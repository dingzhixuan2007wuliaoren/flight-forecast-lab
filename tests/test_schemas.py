from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from flight_forecaster.schemas import (
    ComparisonOffer,
    ComparisonRequest,
    ContextDetailRequest,
    FareSearchMetadata,
    LiveFare,
    OfferDetailRequest,
    OnTimeRequest,
    PriceRequest,
)


def _confirmed_offer_payload() -> dict[str, object]:
    segments = [
        {
            "sequence": 1,
            "origin": "YYZ",
            "destination": "YUL",
            "flight_number": "AC400",
            "marketing_airline_code": "AC",
            "operating_airline_code": "AC",
            "departure_local": "2026-08-15T08:00:00-04:00",
            "arrival_local": "2026-08-15T09:15:00-04:00",
            "departure_utc": "2026-08-15T12:00:00Z",
            "arrival_utc": "2026-08-15T13:15:00Z",
            "duration_minutes": 75,
            "cabin": "economy",
            "booking_class": "K",
            "included_checked_bag_quantity": 1,
        },
        {
            "sequence": 2,
            "origin": "YUL",
            "destination": "LHR",
            "flight_number": "AC864",
            "marketing_airline_code": "AC",
            "operating_airline_code": "AC",
            "departure_local": "2026-08-15T11:00:00-04:00",
            "arrival_local": "2026-08-15T22:50:00+01:00",
            "departure_utc": "2026-08-15T15:00:00Z",
            "arrival_utc": "2026-08-15T21:50:00Z",
            "duration_minutes": 410,
            "cabin": "economy",
            "booking_class": "K",
            "included_checked_bag_quantity": 1,
        },
    ]
    return {
        "id": "off_0123456789abcdef01234567",
        "airline_code": "AC",
        "airline_name": "Air Canada",
        "cabin": "economy",
        "stops": 1,
        "duration_minutes": 590,
        "estimated_price_usd": 510.0,
        "interval_80_low_usd": 510.0,
        "interval_80_high_usd": 510.0,
        "on_time_probability": 0.78,
        "risk_level": "medium",
        "baggage_status": "confirmed_included",
        "student_status": "unknown",
        "change_status": "unknown",
        "refund_status": "unknown",
        "student_age_limit_zh": "未知",
        "student_age_limit_en": "Unknown",
        "student_verification_zh": "未知",
        "student_verification_en": "Unknown",
        "route_status": "provider_confirmed",
        "routing_status": "provider_itinerary",
        "cabin_status": "provider_confirmed",
        "punctuality_basis": "multi_leg_independence_model",
        "schedule_status": "priced_offer",
        "schedule_source": "serpapi_google_flights_booking",
        "flight_number": "AC400",
        "scheduled_departure_local": "2026-08-15T08:00:00-04:00",
        "scheduled_arrival_local": "2026-08-15T22:50:00+01:00",
        "scheduled_departure_utc": "2026-08-15T12:00:00Z",
        "scheduled_arrival_utc": "2026-08-15T21:50:00Z",
        "schedule_observed_at": "2026-07-15T12:00:00Z",
        "bookability_status": "booking_option_verified",
        "live_fare": {
            "provider_name": "SerpApi Google Flights",
            "provider_offer_id": "serpapi-offer-1",
            "verified_at": "2026-07-15T12:00:00Z",
            "total_amount": 510.0,
            "cabin_summary": "economy",
            "provider_cache_hit": False,
            "provider_cache_age_seconds": 0,
            "seats_remaining": 4,
            "booking_provider": "Air Canada",
            "booking_url": "https://www.google.com/travel/flights/booking/example",
            "booking_url_kind": "direct_get",
            "source_url": "https://serpapi.com/google-flights-api",
        },
        "segments": segments,
    }


def _confirmed_offer_payload_with_segment_count(count: int) -> dict[str, object]:
    payload = _confirmed_offer_payload()
    airports = ("YYZ", "YUL", "JFK", "BOS", "IAD", "ATL", "MIA", "DFW", "DEN", "LAX")
    start = datetime(2026, 8, 15, 12, tzinfo=UTC)
    segments: list[dict[str, object]] = []
    for index in range(count):
        departure = start + timedelta(hours=index * 2)
        arrival = departure + timedelta(hours=1)
        segments.append(
            {
                "sequence": index + 1,
                "origin": airports[index],
                "destination": airports[index + 1],
                "flight_number": f"AC{800 + index}",
                "marketing_airline_code": "AC",
                "operating_airline_code": "AC",
                "departure_local": departure.isoformat(),
                "arrival_local": arrival.isoformat(),
                "departure_utc": departure.isoformat(),
                "arrival_utc": arrival.isoformat(),
                "duration_minutes": 60,
                "cabin": "economy",
                "booking_class": "K",
                "included_checked_bag_quantity": 1,
            }
        )
    final_arrival = start + timedelta(hours=(count - 1) * 2 + 1)
    payload.update(
        {
            "stops": count - 1,
            "duration_minutes": round((final_arrival - start).total_seconds() / 60),
            "flight_number": "AC800",
            "scheduled_departure_local": start.isoformat(),
            "scheduled_arrival_local": final_arrival.isoformat(),
            "scheduled_departure_utc": start.isoformat(),
            "scheduled_arrival_utc": final_arrival.isoformat(),
            "segments": segments,
        }
    )
    return payload


def test_strict_offer_schema_accepts_eight_and_rejects_nine_segments() -> None:
    accepted = ComparisonOffer.model_validate(_confirmed_offer_payload_with_segment_count(8))

    assert accepted.stops == 7
    assert len(accepted.segments) == 8
    with pytest.raises(ValidationError):
        ComparisonOffer.model_validate(_confirmed_offer_payload_with_segment_count(9))


def test_price_request_accepts_seven_stops_and_rejects_eight() -> None:
    payload = {
        "origin": "JFK",
        "destination": "LAX",
        "airline": "DL",
        "cabin": "economy",
        "departure_time": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
    }

    assert PriceRequest.model_validate({**payload, "stops": 7}).stops == 7
    with pytest.raises(ValidationError):
        PriceRequest.model_validate({**payload, "stops": 8})


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


def test_confirmed_offer_requires_all_provider_evidence() -> None:
    offer = ComparisonOffer.model_validate(_confirmed_offer_payload())
    assert offer.bookability_status == "booking_option_verified"
    assert len(offer.segments) == 2
    assert offer.live_fare is not None
    assert offer.live_fare.taxes_included is None
    assert offer.live_fare.provider_cache_hit is False
    assert offer.live_fare.provider_cache_age_seconds == 0

    invalid_cache_age = _confirmed_offer_payload()
    invalid_fare = invalid_cache_age["live_fare"]
    assert isinstance(invalid_fare, dict)
    invalid_fare["provider_cache_age_seconds"] = -1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ComparisonOffer.model_validate(invalid_cache_age)

    stale_cache = _confirmed_offer_payload()
    stale_fare = stale_cache["live_fare"]
    assert isinstance(stale_fare, dict)
    stale_fare["provider_cache_age_seconds"] = 3_901
    with pytest.raises(ValidationError, match="less than or equal to 3900"):
        ComparisonOffer.model_validate(stale_cache)

    missing_fare = _confirmed_offer_payload()
    missing_fare.pop("live_fare")
    with pytest.raises(ValidationError, match="verified booking options"):
        ComparisonOffer.model_validate(missing_fare)


def test_production_live_fare_rejects_test_environment() -> None:
    payload = _confirmed_offer_payload()["live_fare"]
    assert isinstance(payload, dict)
    payload["environment"] = "test"
    with pytest.raises(ValidationError, match="production"):
        LiveFare.model_validate(payload)

    metadata = FareSearchMetadata.model_validate(
        {
            "status": "test_environment_rejected",
            "provider_code": "serpapi_google_flights",
            "environment": "test",
            "observed_at": "2026-07-15T12:00:00Z",
            "searched_cabins": ["economy"],
            "notice": {"zh": "测试环境被拒绝", "en": "Test environment rejected"},
        }
    )
    assert metadata.status == "test_environment_rejected"


def test_quarantined_ignav_identity_is_not_a_live_fare_provider() -> None:
    payload = _confirmed_offer_payload()["live_fare"]
    assert isinstance(payload, dict)
    payload["provider_code"] = "ignav_quarantine"
    payload["provider_name"] = "Ignav (strict quarantine)"

    with pytest.raises(ValidationError):
        LiveFare.model_validate(payload)


def test_verified_ignav_identity_requires_its_matching_schedule_source() -> None:
    payload = _confirmed_offer_payload()
    fare = payload["live_fare"]
    assert isinstance(fare, dict)
    fare.update(
        provider_code="ignav_verified_fares",
        provider_name="Ignav Verified Fares",
        source_url="https://ignav.com/",
    )
    payload["schedule_source"] = "ignav_verified_booking"

    offer = ComparisonOffer.model_validate(payload)
    assert offer.live_fare is not None
    assert offer.live_fare.provider_code == "ignav_verified_fares"

    payload["schedule_source"] = "searchapi_google_flights_booking"
    with pytest.raises(ValidationError, match="verified booking options"):
        ComparisonOffer.model_validate(payload)


def test_offer_detail_request_defaults_to_cached_result_and_accepts_refresh() -> None:
    base = {
        "origin": "YYZ",
        "destination": "LHR",
        "departure_date": "2026-08-15",
        "offer_id": "off_0123456789abcdef01234567",
    }
    assert OfferDetailRequest.model_validate(base).force_refresh is False
    assert OfferDetailRequest.model_validate({**base, "force_refresh": True}).force_refresh is True


def test_serpapi_metadata_requires_one_shared_quota_no_higher_than_250() -> None:
    base = {
        "status": "no_results",
        "provider_code": "serpapi_google_flights",
        "environment": "production",
        "observed_at": "2026-07-15T12:00:00Z",
        "searched_cabins": ["economy"],
        "call_count": 2,
        "search_call_count": 1,
        "pricing_call_count": 1,
        "monthly_call_limit": 250,
        "monthly_calls_used": 2,
        "search_monthly_limit": 250,
        "search_monthly_used": 2,
        "notice": {"zh": "无结果", "en": "No results"},
    }
    metadata = FareSearchMetadata.model_validate(base)
    assert metadata.monthly_call_limit == 250

    split = dict(base)
    split.update(
        {
            "pricing_monthly_limit": 250,
            "pricing_monthly_used": 1,
            "search_monthly_used": 1,
        }
    )
    with pytest.raises(ValidationError, match="one shared monthly quota"):
        FareSearchMetadata.model_validate(split)

    over_limit = dict(base)
    over_limit.update({"monthly_call_limit": 251, "search_monthly_limit": 251})
    with pytest.raises(ValidationError, match="cannot exceed 250"):
        FareSearchMetadata.model_validate(over_limit)

    inconsistent_calls = dict(base)
    inconsistent_calls["call_count"] = 3
    with pytest.raises(ValidationError, match="must equal"):
        FareSearchMetadata.model_validate(inconsistent_calls)


@pytest.mark.parametrize(
    "status",
    ["provider_processing", "provider_error", "no_results"],
)
def test_fare_metadata_distinguishes_terminal_and_pending_outcomes(
    status: str,
) -> None:
    payload = {
        "status": status,
        "provider_code": "serpapi_google_flights",
        "environment": "production",
        "observed_at": "2026-07-15T12:00:00Z",
        "searched_cabins": ["economy"],
        "call_count": 1,
        "search_call_count": 1,
        "archive_poll_count": 2,
        "diagnostics": [
            {
                "observed_at": "2026-07-15T12:00:00Z",
                "stage": "search_archive",
                "http_status": 200,
                "exception_type": "ProviderProcessingError",
                "search_id": "pendingsearch01",
            }
        ],
        "notice": {"zh": "状态说明", "en": "Status notice"},
    }

    metadata = FareSearchMetadata.model_validate(payload)

    assert metadata.status == status
    assert metadata.archive_poll_count == 2
    assert metadata.diagnostics[0].search_id == "pendingsearch01"
    serialized = metadata.model_dump_json()
    assert "api_key" not in serialized
    assert '"booking_token":' not in serialized


def test_fare_metadata_validates_partial_candidate_coverage() -> None:
    payload = {
        "status": "confirmed_offers",
        "provider_code": "serpapi_google_flights",
        "environment": "production",
        "observed_at": "2026-07-15T12:00:00Z",
        "searched_cabins": ["economy", "business"],
        "call_count": 6,
        "search_call_count": 4,
        "pricing_call_count": 2,
        "eligible_candidate_count": 9,
        "verification_attempted_count": 2,
        "verified_candidate_count": 1,
        "strictly_rejected_candidate_count": 0,
        "provider_failed_candidate_count": 1,
        "quota_skipped_candidate_count": 7,
        "deduplicated_verified_count": 0,
        "coverage_status": "quota_and_provider_incomplete",
        "quota_limit": "hourly",
        "notice": {"zh": "覆盖不完整", "en": "Partial coverage"},
    }

    metadata = FareSearchMetadata.model_validate(payload)

    assert metadata.coverage_status == "quota_and_provider_incomplete"
    assert metadata.eligible_candidate_count == 9

    inconsistent = dict(payload)
    inconsistent["verification_attempted_count"] = 3
    with pytest.raises(ValidationError, match="verification attempts"):
        FareSearchMetadata.model_validate(inconsistent)

    missing_quota = dict(payload)
    missing_quota["quota_limit"] = None
    with pytest.raises(ValidationError, match="quota limit"):
        FareSearchMetadata.model_validate(missing_quota)


def test_fare_diagnostic_rejects_unredacted_or_malformed_fields() -> None:
    with pytest.raises(ValidationError):
        FareSearchMetadata.model_validate(
            {
                "status": "provider_error",
                "provider_code": "serpapi_google_flights",
                "environment": "production",
                "observed_at": "2026-07-15T12:00:00Z",
                "diagnostics": [
                    {
                        "observed_at": "2026-07-15T12:00:00Z",
                        "stage": "search_archive",
                        "http_status": 200,
                        "exception_type": "raw error: api_key=secret",
                        "search_id": "https://unsafe.example/search",
                    }
                ],
                "notice": {"zh": "错误", "en": "Error"},
            }
        )


def test_confirmed_offer_rejects_mixed_cabins() -> None:
    payload = _confirmed_offer_payload()
    segments = payload["segments"]
    assert isinstance(segments, list)
    assert isinstance(segments[1], dict)
    segments[1]["cabin"] = "business"
    with pytest.raises(ValidationError, match="must use one cabin"):
        ComparisonOffer.model_validate(payload)


def test_confirmed_offer_rejects_discontinuous_segments() -> None:
    payload = _confirmed_offer_payload()
    segments = payload["segments"]
    assert isinstance(segments, list)
    assert isinstance(segments[1], dict)
    segments[1]["origin"] = "CDG"
    with pytest.raises(ValidationError, match="continuous route"):
        ComparisonOffer.model_validate(payload)
