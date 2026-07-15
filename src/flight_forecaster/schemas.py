from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Cabin = Literal["economy", "premium_economy", "business", "first"]
RiskLevel = Literal["low", "medium", "high"]
SignalStatus = Literal[
    "live",
    "forecast",
    "proxy",
    "historical",
    "neutral",
    "unavailable",
]
PolicyStatus = Literal[
    "confirmed_free",
    "confirmed_included",
    "confirmed_discount",
    "confirmed_paid",
    "program_available",
    "not_included",
    "not_applicable",
    "unknown",
]
NewsCategory = Literal[
    "airport_closure",
    "airspace_conflict",
    "labor_strike",
    "extreme_weather",
    "cancellation_delay",
    "security_cyber",
    "other_disruption",
]
ScheduleStatus = Literal[
    "live_schedule",
    "recurring_timetable_projection",
    "model_scenario",
]
ScheduleSource = Literal[
    "airlabs_schedules",
    "airlabs_routes",
    "model_fallback",
]
DepartureTimeBasis = Literal[
    "origin_local_noon_model_reference",
    "origin_local_remaining_day_model_reference",
    "legacy_input",
]


def _normalise_code(value: str, *, min_length: int, max_length: int) -> str:
    code = value.strip().upper()
    if not (min_length <= len(code) <= max_length) or not code.isalnum():
        raise ValueError(f"must be {min_length}-{max_length} letters or digits")
    return code


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must include a timezone offset, for example 2026-08-15T14:00:00-04:00")
    return value


class PriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    airline: str = Field(examples=["DL"])
    cabin: Cabin = "economy"
    stops: int = Field(default=0, ge=0, le=3)
    departure_time: datetime

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @field_validator("airline")
    @classmethod
    def validate_airline(cls, value: str) -> str:
        return _normalise_code(value, min_length=2, max_length=3)

    @field_validator("departure_time")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_trip(self) -> PriceRequest:
        quote_time = datetime.now(UTC)
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if self.departure_time <= quote_time:
            raise ValueError("departure_time must be in the future")
        if (self.departure_time - quote_time).days > 370:
            raise ValueError("departure_time must be within 370 days")
        return self


class OnTimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    airline: str = Field(examples=["DL"])
    scheduled_departure: datetime

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @field_validator("airline")
    @classmethod
    def validate_airline(cls, value: str) -> str:
        return _normalise_code(value, min_length=2, max_length=3)

    @field_validator("scheduled_departure")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_route(self) -> OnTimeRequest:
        now = datetime.now(UTC)
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if self.scheduled_departure <= now:
            raise ValueError("scheduled_departure must be in the future")
        if self.scheduled_departure > now + timedelta(days=370):
            raise ValueError("scheduled_departure must be within 370 days")
        return self


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LHR"])
    departure_date: date = Field(examples=["2026-08-15"])
    departure_time: datetime | None = Field(
        default=None,
        exclude=True,
        description="Legacy input; new clients should send departure_date.",
    )

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_departure_time(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("departure_date") is not None:
            return value
        raw = value.get("departure_time")
        parsed: datetime | None = raw if isinstance(raw, datetime) else None
        if parsed is None and isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        if parsed is None:
            return value
        return {**value, "departure_date": parsed.date()}

    @model_validator(mode="after")
    def validate_trip(self) -> ComparisonRequest:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if (
            self.departure_time is not None
            and self.departure_date != self.departure_time.date()
        ):
            raise ValueError("departure_date and legacy departure_time must match")
        return self


class ContextDetailRequest(BaseModel):
    """Route plus a canonical date or legacy exact time for context pages."""

    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    departure_date: date | None = Field(default=None, examples=["2026-08-15"])
    departure_time: datetime | None = Field(
        default=None,
        description="Legacy exact-time input; date-only clients should send departure_date.",
    )

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_trip(self) -> ContextDetailRequest:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if (self.departure_date is None) == (self.departure_time is None):
            raise ValueError("provide exactly one of departure_date or departure_time")
        return self


class PricePrediction(BaseModel):
    estimated_price_usd: float
    interval_80_low_usd: float
    interval_80_high_usd: float
    days_until_departure: float
    distance_km: float
    duration_minutes: int
    model_version: str
    warning: str
    warning_en: str


class OnTimePrediction(BaseModel):
    on_time_probability: float
    disruption_probability: float
    distance_km: float
    risk_level: RiskLevel
    definition: str
    definition_en: str
    model_version: str


class ContextSignal(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    status: SignalStatus
    source: str
    observed_at: datetime | None = None
    summary_zh: str
    summary_en: str


class OperationsMetric(BaseModel):
    key: str = Field(min_length=1, max_length=80)
    value: float | int | str | bool
    unit: str | None = Field(default=None, max_length=32)


class OperationsEvent(BaseModel):
    event_type: str = Field(min_length=1, max_length=80)
    severity: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=1_000)
    start_at: datetime | None = None
    end_at: datetime | None = None
    scope: str | None = Field(default=None, max_length=500)


class OperationsSnapshot(ContextSignal):
    method: str = Field(default="unknown", max_length=80)
    data_tier: str = Field(default="unknown", max_length=80)
    applicability: str = Field(default="current_only", max_length=80)
    metrics: list[OperationsMetric] = Field(default_factory=list, max_length=24)
    events: list[OperationsEvent] = Field(default_factory=list, max_length=20)
    fallback_reason: str | None = Field(default=None, max_length=1_000)
    window_start: datetime | None = None
    window_end: datetime | None = None
    sample_size: int = Field(default=0, ge=0)
    sample_limit: int | None = Field(default=None, ge=1)
    sample_truncated: bool = False


class OperationsSignal(OperationsSnapshot):
    current_snapshot: OperationsSnapshot | None = None


class NewsArticle(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2048, pattern=r"^https?://")
    source: str = Field(min_length=1, max_length=255)
    published_at: str | None = None
    language: str | None = Field(default=None, max_length=16, pattern=r"^[a-z0-9-]+$")


class NewsSignal(ContextSignal):
    articles: list[NewsArticle] = Field(default_factory=list, max_length=5)


class PredictionContextResponse(BaseModel):
    weather: ContextSignal
    operations: OperationsSignal
    news: NewsSignal


class BilingualText(BaseModel):
    zh: str
    en: str


class ProviderMetadata(BaseModel):
    status: SignalStatus
    source: str
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    fallback_reason: BilingualText | None = None


class WeatherRiskComponent(BaseModel):
    key: Literal["weather_code", "wind", "gust", "precipitation", "visibility"]
    label: BilingualText
    input_value: float | None = None
    unit: str
    risk: float = Field(ge=0.0, le=1.0)


class WeatherObservation(BaseModel):
    time: datetime
    temperature_c: float | None = None
    weather_code: int | None = None
    weather_description: BilingualText
    wind_speed_kmh: float | None = Field(default=None, ge=0.0)
    wind_gust_kmh: float | None = Field(default=None, ge=0.0)
    precipitation_mm: float | None = Field(default=None, ge=0.0)
    precipitation_probability_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    visibility_m: float | None = Field(default=None, ge=0.0)
    risk: float = Field(ge=0.0, le=1.0)
    risk_components: list[WeatherRiskComponent] = Field(default_factory=list, max_length=5)


class HourlyWeatherPoint(BaseModel):
    time: datetime
    temperature_c: float | None = None
    weather_code: int | None = None
    wind_speed_kmh: float | None = Field(default=None, ge=0.0)
    wind_gust_kmh: float | None = Field(default=None, ge=0.0)
    precipitation_mm: float | None = Field(default=None, ge=0.0)
    precipitation_probability_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    visibility_m: float | None = Field(default=None, ge=0.0)
    risk: float = Field(ge=0.0, le=1.0)


class AviationWeatherReport(BaseModel):
    product: Literal["METAR", "TAF"]
    raw_report: str = Field(min_length=1, max_length=10_000)
    issued_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    explanation: BilingualText
    risk: float = Field(ge=0.0, le=1.0)
    source: Literal["NOAA AviationWeather"] = "NOAA AviationWeather"


class AirportWeatherDetail(BaseModel):
    airport_code: str = Field(min_length=3, max_length=3)
    airport_name: str
    icao_code: str | None = None
    timezone: str
    target_time: datetime
    current: WeatherObservation | None = None
    target: WeatherObservation | None = None
    hourly: list[HourlyWeatherPoint] = Field(default_factory=list, max_length=25)
    aviation_reports: list[AviationWeatherReport] = Field(default_factory=list, max_length=4)
    aviation_metadata: ProviderMetadata
    overall_risk: float = Field(ge=0.0, le=1.0)
    metadata: ProviderMetadata


class WeatherDetailResponse(BaseModel):
    origin: str
    destination: str
    departure_time: datetime
    departure_time_basis: DepartureTimeBasis
    estimated_arrival_time: datetime
    duration_minutes: int = Field(gt=0)
    generated_at: datetime
    origin_weather: AirportWeatherDetail
    destination_weather: AirportWeatherDetail
    notice: BilingualText


class NewsDetailArticle(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2048, pattern=r"^https?://")
    source: str = Field(min_length=1, max_length=255)
    language: str | None = Field(default=None, max_length=16, pattern=r"^[a-z0-9-]+$")
    indexed_at: datetime
    category: NewsCategory
    matched_risk_terms: list[str] = Field(default_factory=list, max_length=12)
    raw_score: float = Field(ge=0.0, le=1.0)
    recency_factor: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=1.0)


class NewsDetailResponse(BaseModel):
    origin: str
    destination: str
    departure_time: datetime
    departure_time_basis: DepartureTimeBasis
    generated_at: datetime
    article_count: int = Field(ge=0, le=20)
    articles: list[NewsDetailArticle] = Field(default_factory=list, max_length=20)
    route_raw_risk: float = Field(ge=0.0, le=1.0)
    departure_attenuation_factor: float = Field(ge=0.0, le=1.0)
    model_effect: float = Field(ge=0.0, le=1.0)
    model_signal: ContextSignal
    metadata: ProviderMetadata
    summary: BilingualText
    indexed_time_notice: BilingualText


class ComparisonOffer(BaseModel):
    id: str
    airline_code: str
    airline_name: str
    cabin: Cabin
    stops: int | None = Field(default=None, ge=0, le=3)
    duration_minutes: int = Field(gt=0)
    estimated_price_usd: float = Field(ge=0.0)
    interval_80_low_usd: float = Field(ge=0.0)
    interval_80_high_usd: float = Field(ge=0.0)
    on_time_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    baggage_status: PolicyStatus
    student_status: PolicyStatus
    change_status: PolicyStatus
    refund_status: PolicyStatus
    student_age_limit_zh: str
    student_age_limit_en: str
    student_verification_zh: str
    student_verification_en: str
    student_program_url: str | None = None
    route_status: Literal["provider_confirmed", "model_scenario"]
    routing_status: Literal[
        "provider_direct",
        "model_one_stop",
        "model_route_unresolved",
    ]
    cabin_status: Literal["catalog_scenario"]
    punctuality_basis: Literal[
        "direct_leg_model",
        "two_leg_independence_scenario",
        "route_only_model",
    ]
    schedule_status: ScheduleStatus = "model_scenario"
    schedule_source: ScheduleSource = "model_fallback"
    flight_number: str | None = Field(default=None, pattern=r"^[A-Z0-9]{3,12}$")
    scheduled_departure_local: datetime | None = None
    scheduled_arrival_local: datetime | None = None
    scheduled_departure_utc: datetime | None = None
    scheduled_arrival_utc: datetime | None = None
    provider_flight_status: str | None = Field(default=None, max_length=40)
    schedule_observed_at: datetime | None = None
    departure_terminal: str | None = Field(default=None, max_length=40)
    arrival_terminal: str | None = Field(default=None, max_length=40)
    aircraft_icao: str | None = Field(default=None, max_length=12)

    @model_validator(mode="after")
    def validate_schedule_claims(self) -> ComparisonOffer:
        expected_routing = {
            "provider_direct": ("provider_confirmed", 0, "direct_leg_model"),
            "model_one_stop": (
                "model_scenario",
                1,
                "two_leg_independence_scenario",
            ),
            "model_route_unresolved": (
                "model_scenario",
                None,
                "route_only_model",
            ),
        }[self.routing_status]
        if (
            self.route_status,
            self.stops,
            self.punctuality_basis,
        ) != expected_routing:
            raise ValueError("routing status, stops, route status, and basis must agree")
        schedule_values = (
            self.flight_number,
            self.scheduled_departure_local,
            self.scheduled_arrival_local,
            self.scheduled_departure_utc,
            self.scheduled_arrival_utc,
        )
        if self.schedule_status == "model_scenario":
            if self.schedule_source != "model_fallback" or any(
                value is not None for value in schedule_values
            ):
                raise ValueError("model scenarios cannot include a flight number or clock time")
            return self
        expected_source = {
            "live_schedule": "airlabs_schedules",
            "recurring_timetable_projection": "airlabs_routes",
        }[self.schedule_status]
        if self.schedule_source != expected_source or any(
            value is None for value in schedule_values
        ):
            raise ValueError("provider schedules require a matching source and complete times")
        assert self.scheduled_departure_local is not None
        assert self.scheduled_arrival_local is not None
        assert self.scheduled_departure_utc is not None
        assert self.scheduled_arrival_utc is not None
        for value in (
            self.scheduled_departure_local,
            self.scheduled_arrival_local,
            self.scheduled_departure_utc,
            self.scheduled_arrival_utc,
        ):
            _require_timezone(value)
        if self.scheduled_departure_utc.utcoffset() != timedelta(0) or (
            self.scheduled_arrival_utc.utcoffset() != timedelta(0)
        ):
            raise ValueError("UTC schedule fields must use a zero UTC offset")
        if self.scheduled_arrival_utc <= self.scheduled_departure_utc:
            raise ValueError("scheduled arrival must be after scheduled departure")
        elapsed_minutes = round(
            (
                self.scheduled_arrival_utc - self.scheduled_departure_utc
            ).total_seconds()
            / 60
        )
        if abs(elapsed_minutes - self.duration_minutes) > 15:
            raise ValueError("schedule duration must match its UTC timestamps")
        if abs(
            (
                self.scheduled_departure_local.astimezone(UTC)
                - self.scheduled_departure_utc
            ).total_seconds()
        ) > 120 or abs(
            (
                self.scheduled_arrival_local.astimezone(UTC) - self.scheduled_arrival_utc
            ).total_seconds()
        ) > 120:
            raise ValueError("local and UTC schedule fields must describe the same instants")
        return self


class ComparisonRankings(BaseModel):
    direct_first: list[str]
    lowest_price: list[str]
    student_first: list[str]


class BilingualWarning(BaseModel):
    zh: str
    en: str


class ComparisonResponse(BaseModel):
    origin: str
    destination: str
    departure_date: date
    departure_time: datetime
    departure_time_basis: DepartureTimeBasis
    departure_timezone: str
    distance_km: float
    duration_minutes: int
    generated_at: datetime
    context: PredictionContextResponse
    offers: list[ComparisonOffer]
    rankings: ComparisonRankings
    schedule_sample_truncated: bool
    schedule_sample_limit: Literal[50] = 50
    warnings: BilingualWarning
    model_version: str


class OfferDetailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    departure_date: date = Field(examples=["2026-08-15"])
    offer_id: str = Field(pattern=r"^off_[a-f0-9]{24}$")

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_trip(self) -> OfferDetailRequest:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        return self


class ItineraryLeg(BaseModel):
    sequence: int = Field(ge=1, le=4)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    date_context: date
    flight_number: str | None = Field(default=None, pattern=r"^[A-Z0-9]{3,12}$")
    departure_local: datetime | None = None
    arrival_local: datetime | None = None
    departure_utc: datetime | None = None
    arrival_utc: datetime | None = None
    duration_minutes: int = Field(gt=0)
    distance_km: float = Field(gt=0.0)
    departure_terminal: str | None = Field(default=None, max_length=40)
    arrival_terminal: str | None = Field(default=None, max_length=40)
    aircraft_icao: str | None = Field(default=None, max_length=12)
    data_basis: Literal[
        "airlabs_live_schedule",
        "airlabs_recurring_timetable_projection",
        "model_duration_only",
    ]

    @model_validator(mode="after")
    def validate_time_basis(self) -> ItineraryLeg:
        times = (
            self.departure_local,
            self.arrival_local,
            self.departure_utc,
            self.arrival_utc,
        )
        if self.data_basis == "model_duration_only":
            if self.flight_number is not None or any(value is not None for value in times):
                raise ValueError("model legs cannot include flight numbers or exact clock times")
            return self
        if self.flight_number is None or any(value is None for value in times):
            raise ValueError("provider legs require a flight number and complete times")
        return self


class OfferItinerary(BaseModel):
    kind: Literal["direct", "one_stop", "route_unresolved"]
    time_basis: Literal["provider_schedule", "model_duration_only"]
    total_duration_minutes: int = Field(gt=0)
    total_distance_km: float = Field(gt=0.0)
    layover_airport: str | None = Field(default=None, min_length=3, max_length=3)
    layover_minutes: int | None = Field(default=None, ge=0)
    layover_status: Literal["not_applicable", "model_assumption"]
    legs: list[ItineraryLeg] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_itinerary(self) -> OfferItinerary:
        if self.kind == "direct":
            if (
                len(self.legs) != 1
                or self.layover_airport is not None
                or self.layover_minutes is not None
                or self.layover_status != "not_applicable"
            ):
                raise ValueError("single-leg itineraries cannot include a layover")
        elif self.kind == "route_unresolved":
            if (
                self.legs
                or self.layover_airport is not None
                or self.layover_minutes is not None
                or self.layover_status != "not_applicable"
                or self.time_basis != "model_duration_only"
            ):
                raise ValueError(
                    "unresolved routes must have no legs or layover and use model reference data"
                )
            return self
        elif (
            len(self.legs) != 2
            or self.layover_airport is None
            or self.layover_minutes is None
            or self.layover_status != "model_assumption"
        ):
            raise ValueError("one-stop model itineraries require two legs and a layover")
        expected_duration = sum(leg.duration_minutes for leg in self.legs) + (
            self.layover_minutes or 0
        )
        if expected_duration != self.total_duration_minutes:
            raise ValueError("itinerary duration must equal its legs plus layover")
        if abs(sum(leg.distance_km for leg in self.legs) - self.total_distance_km) > 0.2:
            raise ValueError("itinerary distance must equal its leg distances")
        provider_basis = all(leg.data_basis != "model_duration_only" for leg in self.legs)
        if (self.time_basis == "provider_schedule") != provider_basis:
            raise ValueError("itinerary time basis must match every leg")
        return self


class OfferDetailResponse(BaseModel):
    origin: str
    destination: str
    departure_date: date
    generated_at: datetime
    schedule_status: ScheduleStatus
    schedule_source: ScheduleSource
    schedule_observed_at: datetime | None = None
    schedule_sample_truncated: bool
    schedule_sample_limit: Literal[50] = 50
    fallback_reason: BilingualText | None = None
    offer: ComparisonOffer
    itinerary: OfferItinerary
    notice: BilingualText
