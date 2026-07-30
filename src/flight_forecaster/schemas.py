from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Cabin = Literal["economy", "premium_economy", "business", "first"]
RiskLevel = Literal["low", "medium", "high"]
MAX_STRICT_ITINERARY_SEGMENTS = 8
MAX_STRICT_ITINERARY_STOPS = MAX_STRICT_ITINERARY_SEGMENTS - 1
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
    "priced_offer",
]
ScheduleSource = Literal[
    "airlabs_schedules",
    "airlabs_routes",
    "model_fallback",
    "serpapi_google_flights_booking",
    "searchapi_google_flights_booking",
    "ignav_verified_booking",
]
DepartureTimeBasis = Literal[
    "origin_local_noon_model_reference",
    "origin_local_remaining_day_model_reference",
    "legacy_input",
]
WeatherFeatureStatus = Literal["used", "ignored"]


def _normalise_code(value: str, *, min_length: int, max_length: int) -> str:
    code = value.strip().upper()
    if not (min_length <= len(code) <= max_length) or not code.isalnum():
        raise ValueError(f"must be {min_length}-{max_length} letters or digits")
    return code


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must include a timezone offset, for example 2026-08-15T14:00:00-04:00")
    return value


def _validate_checked_bag_fields(
    quantity: int | None,
    weight: float | None,
    unit: str | None,
) -> None:
    if (weight is None) != (unit is None):
        raise ValueError("checked-bag weight and unit must be provided together")
    if quantity is not None and weight is not None:
        raise ValueError("checked-bag allowance must use quantity or weight, not both")


class PriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    airline: str = Field(examples=["DL"])
    cabin: Cabin = "economy"
    stops: int = Field(default=0, ge=0, le=MAX_STRICT_ITINERARY_STOPS)
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
        if self.departure_time is not None and self.departure_date != self.departure_time.date():
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
    weather_feature_status: WeatherFeatureStatus
    weather_feature_notice_zh: str
    weather_feature_notice_en: str


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
    weather_feature_status: WeatherFeatureStatus
    weather_feature_notice_zh: str
    weather_feature_notice_en: str


class BilingualText(BaseModel):
    zh: str
    en: str


ProviderRole = Literal["strict_fare", "strict_fare_candidate", "reference_only"]
ProviderRuntimeStatus = Literal[
    "not_configured",
    "configured",
    "quota_available",
    "quota_exhausted",
    "quarantined",
    "reference_only",
]
ProviderQuotaStatus = Literal["unknown", "available", "exhausted", "not_applicable"]
ProviderQuotaDataBasis = Literal[
    "provider_reported",
    "local_ledger",
    "provider_and_local_ledger",
    "configured_limit_only",
    "unpublished",
    "not_applicable",
    "unavailable",
]


class RuntimeProviderStatusItem(BaseModel):
    """Credential-safe provider state; never contains keys, tokens, or raw errors."""

    code: str = Field(pattern=r"^[a-z0-9_]{2,64}$")
    display_name: str = Field(min_length=1, max_length=80)
    role: ProviderRole
    configured: bool
    active: bool = False
    status: ProviderRuntimeStatus
    quota_status: ProviderQuotaStatus = "unknown"
    quota_used: int | None = Field(default=None, ge=0)
    quota_limit: int | None = Field(default=None, ge=1)
    quota_remaining: int | None = Field(default=None, ge=0)
    quota_data_basis: ProviderQuotaDataBasis | None = None
    quota_observed_at: datetime | None = None
    quota_reset_at: datetime | None = None
    temporarily_rate_limited: bool = Field(default=False, strict=True)
    quota_unit: (
        Literal[
            "hour",
            "billing_period",
            "credits",
            "provider_managed",
            "billing_period_requests",
            "lifetime_requests",
            "monthly_credits",
            "daily_credits",
            "monthly_units",
        ]
        | None
    ) = None
    quota_cost_per_call: int | None = Field(default=None, ge=1)
    can_supply_strict_offers: bool
    notice: BilingualText

    @model_validator(mode="after")
    def validate_provider_policy(self) -> RuntimeProviderStatusItem:
        if self.role == "reference_only":
            if self.status != "reference_only" or self.can_supply_strict_offers:
                raise ValueError("reference-only providers cannot supply strict offers")
        if self.status == "quarantined" and self.can_supply_strict_offers:
            raise ValueError("quarantined providers cannot supply strict offers")
        if self.status == "quota_exhausted" and self.quota_status != "exhausted":
            raise ValueError("quota-exhausted provider must report exhausted quota")
        if self.quota_used is not None and self.quota_limit is not None:
            if self.quota_used > self.quota_limit:
                raise ValueError("provider quota usage cannot exceed its limit")
        if self.quota_remaining is not None:
            if self.quota_limit is None or self.quota_remaining > self.quota_limit:
                raise ValueError("provider quota remaining requires a valid limit")
            if self.quota_used is not None and (
                self.quota_used + self.quota_remaining > self.quota_limit
            ):
                raise ValueError("provider quota usage plus remaining exceeds its limit")
        if self.quota_data_basis is None and any(
            value is not None
            for value in (
                self.quota_used,
                self.quota_remaining,
                self.quota_observed_at,
                self.quota_reset_at,
            )
        ):
            raise ValueError("provider quota observations require a data basis")
        measurable_bases = {
            "provider_reported",
            "local_ledger",
            "provider_and_local_ledger",
        }
        non_observation_bases = {"not_applicable", "unpublished", "unavailable"}
        if self.quota_data_basis in measurable_bases and any(
            value is None
            for value in (
                self.quota_used,
                self.quota_limit,
                self.quota_remaining,
                self.quota_observed_at,
            )
        ):
            raise ValueError("measurable provider quota requires a complete observation")
        if self.quota_data_basis == "configured_limit_only" and any(
            value is not None
            for value in (
                self.quota_used,
                self.quota_remaining,
                self.quota_observed_at,
                self.quota_reset_at,
            )
        ):
            raise ValueError("configured-limit-only quota cannot report observations")
        if self.quota_data_basis in non_observation_bases and any(
            value is not None
            for value in (
                self.quota_used,
                self.quota_remaining,
                self.quota_observed_at,
                self.quota_reset_at,
            )
        ):
            raise ValueError("non-observation quota basis cannot report observations")
        if self.quota_status in {"available", "exhausted"}:
            if self.quota_data_basis not in measurable_bases:
                raise ValueError("known provider quota status requires measured data")
            if self.quota_status == "exhausted" and self.quota_remaining != 0:
                raise ValueError("exhausted provider quota must have zero remaining")
            if self.quota_status == "available" and self.quota_remaining == 0:
                raise ValueError("available provider quota must have positive remaining")
        if self.quota_observed_at is not None:
            _require_timezone(self.quota_observed_at)
        if self.quota_reset_at is not None:
            _require_timezone(self.quota_reset_at)
        if (
            self.quota_used is not None or self.quota_limit is not None
        ) and self.quota_unit is None:
            raise ValueError("provider quota counts require a quota unit")
        if self.quota_cost_per_call is not None and self.quota_unit is None:
            raise ValueError("provider quota cost requires a quota unit")
        return self


class RuntimeDestinationCapability(BaseModel):
    """Credential-free destination API capability, not a fare-provider claim."""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,49}$")
    display_name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=80)
    active: bool
    configured: bool
    strict_evidence_requirement: str = Field(min_length=1, max_length=120)
    source_host: str = Field(
        min_length=3,
        max_length=253,
        pattern=r"^[a-z0-9.-]+$",
    )
    data_family: Literal[
        "ourairports",
        "openstreetmap",
        "wikimedia",
        "transitous",
        "optional_commercial",
    ]

    @model_validator(mode="after")
    def validate_capability_boundary(self) -> RuntimeDestinationCapability:
        if self.active and not self.configured:
            raise ValueError("active destination capabilities must be configured")
        return self


class RuntimeProviderStatusResponse(BaseModel):
    generated_at: datetime
    strict_policy: BilingualText
    providers: list[RuntimeProviderStatusItem]
    destination_capabilities: list[RuntimeDestinationCapability] = Field(
        default_factory=list,
        max_length=30,
    )

    @model_validator(mode="after")
    def validate_unique_providers(self) -> RuntimeProviderStatusResponse:
        codes = [provider.code for provider in self.providers]
        if len(codes) != len(set(codes)):
            raise ValueError("provider status codes must be unique")
        capability_codes = [
            capability.code for capability in self.destination_capabilities
        ]
        if len(capability_codes) != len(set(capability_codes)):
            raise ValueError("destination capability codes must be unique")
        return self


FareReferenceStatus = Literal[
    "available",
    "no_results",
    "not_configured",
    "quota_exhausted",
    "authentication_failed",
    "rate_limited",
    "provider_error",
    "provider_unavailable",
]


class FareCoverageReference(BaseModel):
    """Aggregate listing coverage that is permanently excluded from strict offers."""

    model_config = ConfigDict(extra="forbid")

    provider_code: Literal["scrape_do_google_flights_reference"] = (
        "scrape_do_google_flights_reference"
    )
    provider_name: Literal["Scrape.do Google Flights"] = "Scrape.do Google Flights"
    role: Literal["reference_only"] = "reference_only"
    status: FareReferenceStatus
    observed_at: datetime
    query_cabin: Literal["economy"] = "economy"
    currency: Literal["USD"] = "USD"
    candidate_count: int = Field(default=0, ge=0, le=30)
    direct_candidate_count: int = Field(default=0, ge=0, le=30)
    lowest_price_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    price_level: Literal["low", "typical", "high"] | None = None
    typical_price_low_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    typical_price_high_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cache_hit: bool = False
    credits_reserved: Literal[0, 10, 20] = 0
    monthly_credits_used: int = Field(default=0, ge=0, le=1_000)
    monthly_credit_limit: int = Field(default=1_000, ge=0, le=1_000)
    http_status: int | None = Field(default=None, ge=100, le=599)
    exception_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$",
    )
    provider_reported_request_cost: int | None = Field(default=None, ge=0, le=10)
    provider_reported_remaining_credits: int | None = Field(
        default=None,
        ge=0,
        le=1_000,
    )
    can_supply_strict_offers: Literal[False] = False
    notice: BilingualText

    @model_validator(mode="after")
    def validate_reference_boundary(self) -> FareCoverageReference:
        _require_timezone(self.observed_at)
        if self.direct_candidate_count > self.candidate_count:
            raise ValueError("direct reference candidates cannot exceed all candidates")
        if self.status == "available":
            if self.candidate_count == 0 or self.lowest_price_usd is None:
                raise ValueError("available fare reference requires candidates and a price")
        elif (
            any(
                value is not None
                for value in (
                    self.lowest_price_usd,
                    self.price_level,
                    self.typical_price_low_usd,
                    self.typical_price_high_usd,
                )
            )
            or self.candidate_count
            or self.direct_candidate_count
        ):
            raise ValueError("non-available fare reference cannot retain listing facts")
        if (self.typical_price_low_usd is None) != (self.typical_price_high_usd is None):
            raise ValueError("fare reference typical-price range must be complete")
        if (
            self.typical_price_low_usd is not None
            and self.typical_price_high_usd is not None
            and self.typical_price_high_usd < self.typical_price_low_usd
        ):
            raise ValueError("fare reference typical-price range is reversed")
        if self.monthly_credits_used > self.monthly_credit_limit:
            raise ValueError("fare reference usage cannot exceed its local free hard limit")
        return self


class HistoricalMarketPricePoint(BaseModel):
    """One SerpApi Google Flights query-level historical market observation."""

    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    price_usd: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_point(self) -> HistoricalMarketPricePoint:
        _require_timezone(self.observed_at)
        return self


class HistoricalMarketContext(BaseModel):
    """Route/date/cabin market history that is not a selected-offer history."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["available"] = "available"
    provider_code: Literal["serpapi_google_flights"] = "serpapi_google_flights"
    provider_name: Literal["SerpApi Google Flights"] = "SerpApi Google Flights"
    scope: Literal["route_departure_date_cabin_market"] = (
        "route_departure_date_cabin_market"
    )
    relation_to_offer: Literal["market_context_not_selected_offer_history"] = (
        "market_context_not_selected_offer_history"
    )
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    departure_date: date
    cabin: Cabin
    currency: Literal["USD"] = "USD"
    provider_observed_at: datetime
    points: list[HistoricalMarketPricePoint] = Field(min_length=1, max_length=400)
    notice: BilingualText

    @model_validator(mode="after")
    def validate_context(self) -> HistoricalMarketContext:
        _require_timezone(self.provider_observed_at)
        if self.origin == self.destination:
            raise ValueError("historical market origin and destination must differ")
        observed_times = [point.observed_at for point in self.points]
        if observed_times != sorted(set(observed_times)):
            raise ValueError(
                "historical market observations must be unique and increasing"
            )
        latest_allowed = self.provider_observed_at.astimezone(UTC) + timedelta(minutes=5)
        if any(point.observed_at.astimezone(UTC) > latest_allowed for point in self.points):
            raise ValueError("historical market context contains a future observation")
        return self


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


class ProviderOfferSegment(BaseModel):
    """One provider-priced segment with internally consistent local and UTC times."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1, le=MAX_STRICT_ITINERARY_SEGMENTS)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    flight_number: str = Field(pattern=r"^[A-Z0-9]{3,12}$")
    marketing_airline_code: str = Field(min_length=2, max_length=3)
    operating_airline_code: str | None = Field(default=None, min_length=2, max_length=3)
    departure_local: datetime
    arrival_local: datetime
    departure_utc: datetime
    arrival_utc: datetime
    duration_minutes: int = Field(gt=0, le=2_160)
    departure_terminal: str | None = Field(default=None, max_length=40)
    arrival_terminal: str | None = Field(default=None, max_length=40)
    aircraft_icao: str | None = Field(default=None, max_length=12)
    cabin: Cabin
    booking_class: str | None = Field(default=None, max_length=8, pattern=r"^[A-Z0-9]+$")
    fare_basis: str | None = Field(default=None, max_length=64)
    fare_brand: str | None = Field(default=None, max_length=120)
    included_checked_bag_quantity: int | None = Field(default=None, ge=0, le=9)
    included_checked_bag_weight: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    included_checked_bag_weight_unit: Literal["KG", "LB"] | None = None

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @field_validator("marketing_airline_code", "operating_airline_code")
    @classmethod
    def validate_airline_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalise_code(value, min_length=2, max_length=3)

    @model_validator(mode="after")
    def validate_segment(self) -> ProviderOfferSegment:
        if self.origin == self.destination:
            raise ValueError("segment origin and destination must differ")
        for value in (
            self.departure_local,
            self.arrival_local,
            self.departure_utc,
            self.arrival_utc,
        ):
            _require_timezone(value)
        if self.departure_utc.utcoffset() != timedelta(0) or (
            self.arrival_utc.utcoffset() != timedelta(0)
        ):
            raise ValueError("segment UTC fields must use a zero UTC offset")
        if self.arrival_utc <= self.departure_utc:
            raise ValueError("segment arrival must be after departure")
        elapsed_minutes = round((self.arrival_utc - self.departure_utc).total_seconds() / 60)
        if abs(elapsed_minutes - self.duration_minutes) > 15:
            raise ValueError("segment duration must match its UTC timestamps")
        if (
            abs((self.departure_local.astimezone(UTC) - self.departure_utc).total_seconds()) > 120
            or abs((self.arrival_local.astimezone(UTC) - self.arrival_utc).total_seconds()) > 120
        ):
            raise ValueError("segment local and UTC fields must describe the same instants")
        _validate_checked_bag_fields(
            self.included_checked_bag_quantity,
            self.included_checked_bag_weight,
            self.included_checked_bag_weight_unit,
        )
        return self


class LiveFare(BaseModel):
    """A live provider fare with a separately verified booking option."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["booking_option_confirmed"] = "booking_option_confirmed"
    provider_code: Literal[
        "serpapi_google_flights",
        "searchapi_google_flights",
        "ignav_verified_fares",
    ] = "serpapi_google_flights"
    provider_name: str = Field(min_length=1, max_length=255)
    provider_offer_id: str = Field(min_length=1, max_length=512)
    environment: Literal["production"] = "production"
    verified_at: datetime
    expires_at: datetime | None = None
    total_amount: float = Field(gt=0.0, allow_inf_nan=False)
    currency: Literal["USD"] = "USD"
    taxes_included: bool | None = None
    provider_cache_hit: bool = Field(strict=True)
    provider_cache_age_seconds: int = Field(ge=0, le=3_900, strict=True)
    price_basis: Literal["one_way_per_adult"] = "one_way_per_adult"
    traveler_count: Literal[1] = 1
    cabin_summary: Cabin
    mixed_cabin: Literal[False] = False
    availability_status: Literal["booking_option_verified"] = "booking_option_verified"
    booking_verified: Literal[True] = True
    booking_provider: str = Field(min_length=1, max_length=255)
    booking_url: str = Field(max_length=2_048, pattern=r"^https://")
    booking_url_kind: Literal["direct_get", "google_flights_itinerary"]
    seats_remaining: int | None = Field(default=None, ge=1, le=9)
    seat_count_capped: bool = False
    last_ticketing_date: date | None = None
    source_url: str = Field(max_length=2_048, pattern=r"^https://")

    @model_validator(mode="after")
    def validate_fare(self) -> LiveFare:
        _require_timezone(self.verified_at)
        provider_names = {
            "serpapi_google_flights": "SerpApi Google Flights",
            "searchapi_google_flights": "SearchAPI.io Google Flights",
            "ignav_verified_fares": "Ignav Verified Fares",
        }
        if self.provider_name != provider_names[self.provider_code]:
            raise ValueError("live-fare provider code and name must match")
        if self.expires_at is not None:
            _require_timezone(self.expires_at)
            if self.expires_at <= self.verified_at:
                raise ValueError("fare expiry must be after verification")
        if self.seat_count_capped and self.seats_remaining != 9:
            raise ValueError("a capped seat count must report exactly 9 seats")
        if (
            self.last_ticketing_date is not None
            and self.last_ticketing_date < self.verified_at.date()
        ):
            raise ValueError("last ticketing date cannot precede fare verification")
        return self


class FareProviderDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    stage: Literal[
        "account",
        "cabin_search",
        "booking_options",
        "search_archive",
        "validation",
    ]
    http_status: int | None = Field(default=None, ge=100, le=599)
    exception_type: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    search_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,128}$",
    )

    @model_validator(mode="after")
    def validate_diagnostic(self) -> FareProviderDiagnostic:
        _require_timezone(self.observed_at)
        return self


class FareSearchMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "confirmed_offers",
        "no_results",
        "not_configured",
        "test_environment_rejected",
        "authentication_failed",
        "rate_limited",
        "budget_not_configured",
        "budget_exhausted",
        "provider_processing",
        "provider_error",
        "provider_unavailable",
    ]
    provider_code: Literal[
        "serpapi_google_flights",
        "searchapi_google_flights",
        "ignav_quarantine",
        "ignav_verified_fares",
        "strict_fare_aggregate",
        "none",
    ]
    provider_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )
    provider_runs: list[FareSearchMetadata] = Field(default_factory=list, max_length=4)
    environment: Literal["production", "test", "disabled"]
    observed_at: datetime
    searched_cabins: list[Cabin] = Field(default_factory=list, max_length=4)
    call_count: int = Field(default=0, ge=0)
    search_call_count: int = Field(default=0, ge=0)
    pricing_call_count: int = Field(default=0, ge=0)
    cache_hit: bool = False
    quota_unit: Literal["billing_period_requests", "lifetime_requests"] | None = None
    monthly_call_limit: int | None = Field(
        default=None,
        ge=0,
        description="Legacy field name; interpret with quota_unit.",
    )
    monthly_calls_used: int | None = Field(
        default=None,
        ge=0,
        description="Legacy field name; interpret with quota_unit.",
    )
    search_monthly_limit: int | None = Field(
        default=None,
        ge=0,
        description="Legacy field name; interpret with quota_unit.",
    )
    search_monthly_used: int | None = Field(
        default=None,
        ge=0,
        description="Legacy field name; interpret with quota_unit.",
    )
    pricing_monthly_limit: int | None = Field(
        default=None,
        ge=0,
        description="Legacy field name; interpret with quota_unit.",
    )
    pricing_monthly_used: int | None = Field(
        default=None,
        ge=0,
        description="Legacy field name; interpret with quota_unit.",
    )
    archive_poll_count: int = Field(default=0, ge=0)
    diagnostics: list[FareProviderDiagnostic] = Field(
        default_factory=list,
        max_length=10,
    )
    coverage_scope: Literal["provider_returned_booking_verification_candidates"] = (
        "provider_returned_booking_verification_candidates"
    )
    eligible_candidate_count: int = Field(default=0, ge=0)
    verification_attempted_count: int = Field(default=0, ge=0)
    verified_candidate_count: int = Field(default=0, ge=0)
    strictly_rejected_candidate_count: int = Field(default=0, ge=0)
    provider_failed_candidate_count: int = Field(default=0, ge=0)
    search_failed_cabin_count: int = Field(default=0, ge=0)
    quota_skipped_candidate_count: int = Field(default=0, ge=0)
    deduplicated_verified_count: int = Field(default=0, ge=0)
    coverage_status: Literal[
        "not_evaluated",
        "complete",
        "quota_limited",
        "provider_incomplete",
        "quota_and_provider_incomplete",
    ] = "not_evaluated"
    quota_limit: Literal[
        "monthly",
        "hourly",
        "lifetime",
        "provider_specific",
    ] | None = None
    retry_quota_limited: bool = Field(default=False, strict=True)
    notice: BilingualText

    @model_validator(mode="after")
    def validate_metadata(self) -> FareSearchMetadata:
        _require_timezone(self.observed_at)
        if self.provider_code == "strict_fare_aggregate":
            if len(self.provider_runs) < 2:
                raise ValueError("aggregate fare metadata requires at least two provider runs")
            if any(run.provider_runs for run in self.provider_runs):
                raise ValueError("aggregate provider runs cannot be nested")
            run_codes = [run.provider_code for run in self.provider_runs]
            if (
                len(run_codes) != len(set(run_codes))
                or any(code in {"none", "strict_fare_aggregate"} for code in run_codes)
            ):
                raise ValueError("aggregate provider runs must identify unique strict providers")
        elif self.provider_runs:
            raise ValueError("single-provider fare metadata cannot contain provider runs")
        if len(self.searched_cabins) != len(set(self.searched_cabins)):
            raise ValueError("searched cabins must be unique")
        if self.call_count != self.search_call_count + self.pricing_call_count:
            raise ValueError("total call count must equal search plus booking-validation calls")
        if self.verification_attempted_count != (
            self.verified_candidate_count
            + self.strictly_rejected_candidate_count
            + self.provider_failed_candidate_count
        ):
            raise ValueError("verification attempts must equal all verification outcomes")
        if self.deduplicated_verified_count > self.verified_candidate_count:
            raise ValueError("deduplicated verified count cannot exceed verified candidates")
        if self.retry_quota_limited and not (
            self.provider_failed_candidate_count or self.search_failed_cabin_count
        ):
            raise ValueError("retry quota limitation requires a failed provider attempt")
        if self.coverage_status == "not_evaluated":
            if self.retry_quota_limited:
                raise ValueError("unevaluated candidate coverage cannot be retry-quota limited")
            if any(
                (
                    self.verification_attempted_count,
                    self.verified_candidate_count,
                    self.strictly_rejected_candidate_count,
                    self.provider_failed_candidate_count,
                    self.search_failed_cabin_count,
                    self.quota_skipped_candidate_count,
                    self.deduplicated_verified_count,
                )
            ):
                raise ValueError("unevaluated candidate coverage cannot report outcomes")
        elif self.eligible_candidate_count != (
            self.verification_attempted_count + self.quota_skipped_candidate_count
        ):
            raise ValueError("eligible candidates must equal attempted plus quota-skipped")
        provider_run_incomplete = bool(
            self.provider_runs
            and any(
                run.status not in {"confirmed_offers", "no_results"}
                or run.coverage_status
                in {"provider_incomplete", "quota_and_provider_incomplete"}
                for run in self.provider_runs
            )
        )
        provider_run_quota_limited = bool(
            self.provider_runs
            and any(
                run.status in {"rate_limited", "budget_exhausted"}
                or run.coverage_status
                in {"quota_limited", "quota_and_provider_incomplete"}
                for run in self.provider_runs
            )
        )
        quota_truncated = bool(
            self.quota_skipped_candidate_count
            or self.retry_quota_limited
            or provider_run_quota_limited
        )
        expected_coverage = (
            "quota_and_provider_incomplete"
            if (
                self.provider_failed_candidate_count
                or self.search_failed_cabin_count
                or provider_run_incomplete
            )
            and quota_truncated
            else (
                "provider_incomplete"
                if (
                    self.provider_failed_candidate_count
                    or self.search_failed_cabin_count
                    or provider_run_incomplete
                )
                else (
                    "quota_limited"
                    if quota_truncated
                    else (
                        "not_evaluated" if self.coverage_status == "not_evaluated" else "complete"
                    )
                )
            )
        )
        if self.coverage_status != expected_coverage:
            raise ValueError("candidate coverage status does not match its counts")
        quota_limited = self.coverage_status in {
            "quota_limited",
            "quota_and_provider_incomplete",
        }
        unevaluated_quota_wall = (
            self.coverage_status == "not_evaluated"
            and self.status in {"rate_limited", "budget_exhausted"}
        )
        if not unevaluated_quota_wall and quota_limited != (self.quota_limit is not None):
            raise ValueError("candidate quota limit must match quota-truncated coverage")
        if self.monthly_calls_used is not None and self.monthly_call_limit is None:
            raise ValueError("monthly usage requires a monthly call limit")
        if self.search_monthly_used is not None and self.search_monthly_limit is None:
            raise ValueError("search usage requires a search monthly limit")
        if self.pricing_monthly_used is not None and self.pricing_monthly_limit is None:
            raise ValueError("pricing usage requires a pricing monthly limit")
        individual_limits = [
            value
            for value in (self.search_monthly_limit, self.pricing_monthly_limit)
            if value is not None
        ]
        individual_usage = [
            value
            for value in (self.search_monthly_used, self.pricing_monthly_used)
            if value is not None
        ]
        if len(individual_limits) > 1 or len(individual_usage) > 1:
            raise ValueError(
                "a fare provider must expose one shared monthly quota (or lifetime quota), "
                "not split quotas"
            )
        if self.monthly_call_limit is not None:
            maximum = {
                "serpapi_google_flights": 250,
                "searchapi_google_flights": 100,
                "ignav_quarantine": 1_000,
                "ignav_verified_fares": 1_000,
                "strict_fare_aggregate": 0,
                "none": 0,
            }[self.provider_code]
            if self.monthly_call_limit > maximum:
                raise ValueError(f"local free-quota hard limit cannot exceed {maximum}")
            if not individual_limits or self.monthly_call_limit != individual_limits[0]:
                raise ValueError("monthly limit must equal the single shared quota")
        elif individual_limits:
            raise ValueError("the shared quota requires a monthly total limit")
        if self.monthly_calls_used is not None:
            if not individual_usage or self.monthly_calls_used != individual_usage[0]:
                raise ValueError("monthly usage must equal the single shared usage counter")
        elif individual_usage:
            raise ValueError("the shared usage counter requires monthly total usage")
        if self.status == "not_configured":
            if self.provider_code != "none" or self.environment != "disabled":
                raise ValueError("an unconfigured provider must be disabled and use code none")
        elif self.provider_code == "none":
            raise ValueError("configured fare-search results must identify their provider")
        provider_names = {
            "serpapi_google_flights": "SerpApi Google Flights",
            "searchapi_google_flights": "SearchAPI.io Google Flights",
            "ignav_quarantine": "Ignav (strict quarantine)",
            "ignav_verified_fares": "Ignav Verified Fares",
            "strict_fare_aggregate": "Strict Fare Provider Aggregate",
            "none": "No strict fare provider",
        }
        if self.provider_name is None:
            self.provider_name = provider_names[self.provider_code]
        elif self.provider_name != provider_names[self.provider_code]:
            raise ValueError("fare-search provider code and name must match")
        expected_quota_unit = {
            "serpapi_google_flights": "billing_period_requests",
            "searchapi_google_flights": "lifetime_requests",
            "ignav_quarantine": "lifetime_requests",
            "ignav_verified_fares": "lifetime_requests",
            "strict_fare_aggregate": None,
            "none": None,
        }[self.provider_code]
        has_quota_counts = any(
            value is not None
            for value in (
                self.monthly_call_limit,
                self.monthly_calls_used,
                self.search_monthly_limit,
                self.search_monthly_used,
                self.pricing_monthly_limit,
                self.pricing_monthly_used,
            )
        )
        if has_quota_counts and self.quota_unit is None:
            self.quota_unit = expected_quota_unit
        if self.quota_unit != (expected_quota_unit if has_quota_counts else None):
            raise ValueError("fare-search quota unit must match its provider and counters")
        if self.status == "test_environment_rejected" and self.environment not in {
            "test",
            "disabled",
        }:
            raise ValueError("test-environment rejection requires test or disabled environment")
        if self.status in {"confirmed_offers", "no_results"} and (self.environment != "production"):
            raise ValueError("live fare-search results require the production environment")
        return self

    def includes_confirmed_provider(self, provider_code: str, provider_name: str) -> bool:
        """Return whether this metadata contains the fare's confirmed source run."""

        if (self.provider_code, self.provider_name) == (provider_code, provider_name):
            return self.status == "confirmed_offers"
        return bool(
            self.provider_code == "strict_fare_aggregate"
            and any(
                run.status == "confirmed_offers"
                and (run.provider_code, run.provider_name) == (provider_code, provider_name)
                for run in self.provider_runs
            )
        )


class ComparisonOffer(BaseModel):
    id: str
    airline_code: str
    airline_name: str
    cabin: Cabin
    stops: int | None = Field(default=None, ge=0, le=MAX_STRICT_ITINERARY_STOPS)
    duration_minutes: int = Field(gt=0)
    estimated_price_usd: float = Field(ge=0.0)
    interval_80_low_usd: float = Field(ge=0.0)
    interval_80_high_usd: float = Field(ge=0.0)
    on_time_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    weather_feature_status: WeatherFeatureStatus = "ignored"
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
        "provider_itinerary",
        "model_one_stop",
        "model_route_unresolved",
    ]
    cabin_status: Literal["catalog_scenario", "provider_confirmed"]
    punctuality_basis: Literal[
        "direct_leg_model",
        "multi_leg_independence_model",
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
    bookability_status: Literal["unverified", "booking_option_verified"] = "unverified"
    live_fare: LiveFare | None = None
    segments: list[ProviderOfferSegment] = Field(
        default_factory=list,
        max_length=MAX_STRICT_ITINERARY_SEGMENTS,
    )

    @model_validator(mode="after")
    def validate_schedule_claims(self) -> ComparisonOffer:
        priced_sources = {
            "serpapi_google_flights": "serpapi_google_flights_booking",
            "searchapi_google_flights": "searchapi_google_flights_booking",
            "ignav_verified_fares": "ignav_verified_booking",
        }
        if self.routing_status == "provider_itinerary":
            if (
                self.route_status != "provider_confirmed"
                or self.stops is None
                or not 1 <= self.stops <= MAX_STRICT_ITINERARY_STOPS
                or self.punctuality_basis != "multi_leg_independence_model"
            ):
                raise ValueError(
                    "provider itineraries require one to seven stops and the multi-leg basis"
                )
        else:
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

        if self.bookability_status == "unverified" and self.live_fare is not None:
            raise ValueError("unverified offers cannot include a live fare")
        if self.bookability_status == "booking_option_verified":
            expected_priced_source = (
                priced_sources.get(self.live_fare.provider_code)
                if self.live_fare is not None
                else None
            )
            if (
                self.route_status != "provider_confirmed"
                or self.routing_status not in {"provider_direct", "provider_itinerary"}
                or self.schedule_status != "priced_offer"
                or self.schedule_source != expected_priced_source
                or self.cabin_status != "provider_confirmed"
                or self.live_fare is None
                or not self.live_fare.booking_verified
                or not self.segments
                or self.stops is None
            ):
                raise ValueError(
                    "verified booking options require a provider itinerary, provider "
                    "schedule, provider-confirmed cabin, live fare, booking evidence, and segments"
                )
            if len(self.segments) != self.stops + 1:
                raise ValueError("segment count must equal stops plus one")
            if [segment.sequence for segment in self.segments] != list(
                range(1, len(self.segments) + 1)
            ):
                raise ValueError("offer segments must have consecutive sequences")
            if self.live_fare.cabin_summary != self.cabin or any(
                segment.cabin != self.cabin for segment in self.segments
            ):
                raise ValueError("live fare, offer, and every segment must use one cabin")
            for previous, current in zip(self.segments, self.segments[1:], strict=False):
                if previous.destination != current.origin:
                    raise ValueError("offer segments must form one continuous route")
                if current.departure_utc <= previous.arrival_utc:
                    raise ValueError("each segment must depart after the prior segment arrives")
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
        expected_sources = {
            "live_schedule": {"airlabs_schedules"},
            "recurring_timetable_projection": {"airlabs_routes"},
            "priced_offer": set(priced_sources.values()),
        }[self.schedule_status]
        if self.schedule_source not in expected_sources or any(
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
            (self.scheduled_arrival_utc - self.scheduled_departure_utc).total_seconds() / 60
        )
        if abs(elapsed_minutes - self.duration_minutes) > 15:
            raise ValueError("schedule duration must match its UTC timestamps")
        if (
            abs(
                (
                    self.scheduled_departure_local.astimezone(UTC) - self.scheduled_departure_utc
                ).total_seconds()
            )
            > 120
            or abs(
                (
                    self.scheduled_arrival_local.astimezone(UTC) - self.scheduled_arrival_utc
                ).total_seconds()
            )
            > 120
        ):
            raise ValueError("local and UTC schedule fields must describe the same instants")
        if self.bookability_status == "booking_option_verified":
            first_segment = self.segments[0]
            last_segment = self.segments[-1]
            assert self.flight_number is not None
            if self.flight_number != first_segment.flight_number:
                raise ValueError("summary flight number must match the first segment")
            summary_and_segment_times = (
                (self.scheduled_departure_local, first_segment.departure_local),
                (self.scheduled_departure_utc, first_segment.departure_utc),
                (self.scheduled_arrival_local, last_segment.arrival_local),
                (self.scheduled_arrival_utc, last_segment.arrival_utc),
            )
            if any(
                abs((summary - segment).total_seconds()) > 120
                for summary, segment in summary_and_segment_times
            ):
                raise ValueError("summary times must match the first and last segments")
        return self


class ComparisonRankings(BaseModel):
    direct_first: list[str]
    lowest_price: list[str]
    student_first: list[str]


class BilingualWarning(BaseModel):
    zh: str
    en: str


class TimetableReference(BaseModel):
    """A provider timetable row excluded from strict flight comparison."""

    airline_code: str = Field(min_length=2, max_length=3)
    airline_name: str
    flight_number: str = Field(pattern=r"^[A-Z0-9]{3,12}$")
    duration_minutes: int = Field(gt=0)
    schedule_status: Literal[
        "live_schedule",
        "future_schedule_reference",
        "recurring_timetable_projection",
    ]
    schedule_source: Literal[
        "airlabs_schedules",
        "aerodatabox_schedule",
        "airlabs_routes",
    ]
    scheduled_departure_local: datetime
    scheduled_arrival_local: datetime
    scheduled_departure_utc: datetime
    scheduled_arrival_utc: datetime
    schedule_observed_at: datetime | None = None
    provider_flight_status: str | None = Field(default=None, max_length=40)
    bookability_status: Literal["unverified"] = "unverified"
    reference_reason: BilingualText

    @model_validator(mode="after")
    def validate_reference(self) -> TimetableReference:
        expected_source = {
            "live_schedule": "airlabs_schedules",
            "future_schedule_reference": "aerodatabox_schedule",
            "recurring_timetable_projection": "airlabs_routes",
        }[self.schedule_status]
        if self.schedule_source != expected_source:
            raise ValueError("timetable status and source must agree")
        for value in (
            self.scheduled_departure_local,
            self.scheduled_arrival_local,
            self.scheduled_departure_utc,
            self.scheduled_arrival_utc,
        ):
            _require_timezone(value)
        if self.scheduled_arrival_utc <= self.scheduled_departure_utc:
            raise ValueError("timetable arrival must be after departure")
        return self


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
    availability_mode: Literal[
        "strict_bookable_only",
        "strict_schedule_only",
    ] = "strict_bookable_only"
    result_status: Literal[
        "verified_offers_found",
        "no_verified_offer",
        "fare_provider_not_configured",
        "fare_provider_test_rejected",
        "fare_provider_authentication_failed",
        "fare_provider_rate_limited",
        "fare_provider_budget_not_configured",
        "fare_provider_budget_exhausted",
        "fare_provider_coverage_limited",
        "fare_provider_processing",
        "fare_provider_error",
        "fare_provider_unavailable",
        "verified_schedules_found",
        "no_verified_schedule",
    ]
    strict_mode_notice: BilingualText
    fare_search_metadata: FareSearchMetadata | None = None
    provider_statuses: list[RuntimeProviderStatusItem] = Field(default_factory=list)
    fare_reference_snapshots: list[FareCoverageReference] = Field(
        default_factory=list,
        max_length=2,
    )
    historical_market_contexts: list[HistoricalMarketContext] = Field(
        default_factory=list,
        max_length=4,
    )
    timetable_references: list[TimetableReference] = Field(default_factory=list)
    schedule_sample_truncated: bool
    schedule_sample_limit: Literal[50] = 50
    warnings: BilingualWarning
    model_version: str

    @model_validator(mode="after")
    def validate_availability_mode(self) -> ComparisonResponse:
        reference_codes = [reference.provider_code for reference in self.fare_reference_snapshots]
        if len(reference_codes) != len(set(reference_codes)):
            raise ValueError("fare reference provider codes must be unique")
        if any(reference.can_supply_strict_offers for reference in self.fare_reference_snapshots):
            raise ValueError("fare coverage references cannot supply strict offers")
        market_context_keys = [
            (context.provider_code, context.cabin)
            for context in self.historical_market_contexts
        ]
        if len(market_context_keys) != len(set(market_context_keys)):
            raise ValueError("historical market contexts must be unique by provider and cabin")
        if any(
            context.origin != self.origin
            or context.destination != self.destination
            or context.departure_date != self.departure_date
            for context in self.historical_market_contexts
        ):
            raise ValueError("historical market context must match the comparison query")
        offer_ids = [offer.id for offer in self.offers]
        if len(offer_ids) != len(set(offer_ids)):
            raise ValueError("comparison offer ids must be unique")
        expected_ranking_ids = set(offer_ids)
        for ranking_name, ranking in (
            ("direct_first", self.rankings.direct_first),
            ("lowest_price", self.rankings.lowest_price),
            ("student_first", self.rankings.student_first),
        ):
            if len(ranking) != len(offer_ids) or set(ranking) != expected_ranking_ids:
                raise ValueError(f"{ranking_name} must contain every offer id exactly once")
        if self.availability_mode != "strict_bookable_only":
            return self
        if self.fare_search_metadata is None:
            raise ValueError("strict bookable mode requires fare-search metadata")
        if any(offer.bookability_status != "booking_option_verified" for offer in self.offers):
            raise ValueError("strict bookable mode may only return confirmed offers")
        if self.offers and self.result_status != "verified_offers_found":
            raise ValueError("strict bookable offers require result status verified_offers_found")
        if not self.offers and self.result_status == "verified_offers_found":
            raise ValueError("verified_offers_found requires at least one confirmed offer")
        if self.result_status in {"verified_schedules_found", "no_verified_schedule"}:
            raise ValueError("strict bookable mode requires a fare-provider result status")
        if self.offers and (
            self.fare_search_metadata.status != "confirmed_offers"
            or self.fare_search_metadata.environment != "production"
        ):
            raise ValueError("confirmed offers require production fare-search metadata")
        if self.offers and any(
            offer.live_fare is None
            or not self.fare_search_metadata.includes_confirmed_provider(
                offer.live_fare.provider_code,
                offer.live_fare.provider_name,
            )
            for offer in self.offers
        ):
            raise ValueError(
                "comparison metadata must contain every live fare's confirmed provider run"
            )
        expected_status = {
            "confirmed_offers": ("verified_offers_found" if self.offers else "no_verified_offer"),
            "no_results": "no_verified_offer",
            "not_configured": "fare_provider_not_configured",
            "test_environment_rejected": "fare_provider_test_rejected",
            "authentication_failed": "fare_provider_authentication_failed",
            "rate_limited": "fare_provider_rate_limited",
            "budget_not_configured": "fare_provider_budget_not_configured",
            "budget_exhausted": "fare_provider_budget_exhausted",
            "provider_processing": "fare_provider_processing",
            "provider_error": "fare_provider_error",
            "provider_unavailable": "fare_provider_unavailable",
        }[self.fare_search_metadata.status]
        if (
            not self.offers
            and self.fare_search_metadata.status == "no_results"
            and self.fare_search_metadata.quota_skipped_candidate_count > 0
            and self.fare_search_metadata.coverage_status
            in {"quota_limited", "quota_and_provider_incomplete"}
        ):
            expected_status = "fare_provider_coverage_limited"
        if self.result_status != expected_status:
            raise ValueError("result status must agree with fare-search metadata and offers")
        return self


class OfferDetailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    departure_date: date = Field(examples=["2026-08-15"])
    offer_id: str = Field(pattern=r"^off_[a-f0-9]{24}$")
    force_refresh: bool = False

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
    sequence: int = Field(ge=1, le=MAX_STRICT_ITINERARY_SEGMENTS)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    date_context: date
    flight_number: str | None = Field(default=None, pattern=r"^[A-Z0-9]{3,12}$")
    marketing_airline_code: str | None = Field(default=None, min_length=2, max_length=3)
    operating_airline_code: str | None = Field(default=None, min_length=2, max_length=3)
    departure_local: datetime | None = None
    arrival_local: datetime | None = None
    departure_utc: datetime | None = None
    arrival_utc: datetime | None = None
    duration_minutes: int = Field(gt=0)
    distance_km: float = Field(gt=0.0)
    departure_terminal: str | None = Field(default=None, max_length=40)
    arrival_terminal: str | None = Field(default=None, max_length=40)
    aircraft_icao: str | None = Field(default=None, max_length=12)
    cabin: Cabin | None = None
    booking_class: str | None = Field(default=None, max_length=8, pattern=r"^[A-Z0-9]+$")
    fare_basis: str | None = Field(default=None, max_length=64)
    fare_brand: str | None = Field(default=None, max_length=120)
    included_checked_bag_quantity: int | None = Field(default=None, ge=0, le=9)
    included_checked_bag_weight: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    included_checked_bag_weight_unit: Literal["KG", "LB"] | None = None
    data_basis: Literal[
        "airlabs_live_schedule",
        "airlabs_recurring_timetable_projection",
        "serpapi_booking_confirmed",
        "searchapi_booking_confirmed",
        "ignav_verified_booking_confirmed",
        "model_duration_only",
    ]

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @field_validator("marketing_airline_code", "operating_airline_code")
    @classmethod
    def validate_airline_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalise_code(value, min_length=2, max_length=3)

    @model_validator(mode="after")
    def validate_time_basis(self) -> ItineraryLeg:
        if self.origin == self.destination:
            raise ValueError("itinerary leg origin and destination must differ")
        _validate_checked_bag_fields(
            self.included_checked_bag_quantity,
            self.included_checked_bag_weight,
            self.included_checked_bag_weight_unit,
        )
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
        assert self.departure_local is not None
        assert self.arrival_local is not None
        assert self.departure_utc is not None
        assert self.arrival_utc is not None
        for value in times:
            assert value is not None
            _require_timezone(value)
        if self.departure_utc.utcoffset() != timedelta(0) or (
            self.arrival_utc.utcoffset() != timedelta(0)
        ):
            raise ValueError("provider leg UTC fields must use a zero UTC offset")
        if self.arrival_utc <= self.departure_utc:
            raise ValueError("provider leg arrival must be after departure")
        elapsed_minutes = round((self.arrival_utc - self.departure_utc).total_seconds() / 60)
        if abs(elapsed_minutes - self.duration_minutes) > 15:
            raise ValueError("provider leg duration must match its UTC timestamps")
        if (
            abs((self.departure_local.astimezone(UTC) - self.departure_utc).total_seconds()) > 120
            or abs((self.arrival_local.astimezone(UTC) - self.arrival_utc).total_seconds()) > 120
        ):
            raise ValueError("provider leg local and UTC fields must describe the same instants")
        if self.data_basis in {
            "serpapi_booking_confirmed",
            "searchapi_booking_confirmed",
            "ignav_verified_booking_confirmed",
        } and (
            self.marketing_airline_code is None or self.cabin is None
        ):
            raise ValueError(
                "Google Flights booking-verified legs require marketing airline and cabin"
            )
        return self


class ItineraryLayover(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1, le=MAX_STRICT_ITINERARY_STOPS)
    airport: str = Field(min_length=3, max_length=3)
    duration_minutes: int = Field(ge=0, le=1_440)

    @field_validator("airport")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)


class OfferItinerary(BaseModel):
    kind: Literal["direct", "one_stop", "multi_stop", "route_unresolved"]
    time_basis: Literal["provider_schedule", "model_duration_only"]
    total_duration_minutes: int = Field(gt=0)
    total_distance_km: float = Field(gt=0.0)
    layover_airport: str | None = Field(default=None, min_length=3, max_length=3)
    layover_minutes: int | None = Field(default=None, ge=0)
    layover_status: Literal[
        "not_applicable",
        "model_assumption",
        "provider_confirmed",
    ]
    legs: list[ItineraryLeg] = Field(
        default_factory=list,
        max_length=MAX_STRICT_ITINERARY_SEGMENTS,
    )
    layovers: list[ItineraryLayover] = Field(
        default_factory=list,
        max_length=MAX_STRICT_ITINERARY_STOPS,
    )

    @field_validator("layover_airport")
    @classmethod
    def validate_layover_airport(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalise_code(value, min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_itinerary(self) -> OfferItinerary:
        if self.kind == "direct":
            if (
                len(self.legs) != 1
                or self.layover_airport is not None
                or self.layover_minutes is not None
                or self.layovers
                or self.layover_status != "not_applicable"
            ):
                raise ValueError("single-leg itineraries cannot include a layover")
        elif self.kind == "route_unresolved":
            if (
                self.legs
                or self.layover_airport is not None
                or self.layover_minutes is not None
                or self.layovers
                or self.layover_status != "not_applicable"
                or self.time_basis != "model_duration_only"
            ):
                raise ValueError(
                    "unresolved routes must have no legs or layover and use model reference data"
                )
            return self
        elif self.kind == "one_stop":
            if len(self.legs) != 2:
                raise ValueError("one-stop itineraries require exactly two legs")
            legacy_layover = self.layover_airport is not None and self.layover_minutes is not None
            provider_layover = len(self.layovers) == 1
            if legacy_layover == provider_layover:
                raise ValueError(
                    "one-stop itineraries require either one provider layover or the legacy layover"
                )
            if legacy_layover and self.layover_status != "model_assumption":
                raise ValueError("legacy one-stop layovers must be model assumptions")
            if provider_layover and self.layover_status != "provider_confirmed":
                raise ValueError("provider one-stop layovers must be provider confirmed")
        elif (
            not 3 <= len(self.legs) <= MAX_STRICT_ITINERARY_SEGMENTS
            or len(self.layovers) != len(self.legs) - 1
            or self.layover_airport is not None
            or self.layover_minutes is not None
            or self.layover_status != "provider_confirmed"
        ):
            raise ValueError(
                "multi-stop itineraries require three to eight legs and provider layovers"
            )

        if [leg.sequence for leg in self.legs] != list(range(1, len(self.legs) + 1)):
            raise ValueError("itinerary legs must have consecutive sequences")
        for previous, current in zip(self.legs, self.legs[1:], strict=False):
            if previous.destination != current.origin:
                raise ValueError("itinerary legs must form one continuous route")

        if self.layovers:
            if [layover.sequence for layover in self.layovers] != list(
                range(1, len(self.layovers) + 1)
            ):
                raise ValueError("itinerary layovers must have consecutive sequences")
            for index, layover in enumerate(self.layovers):
                previous = self.legs[index]
                current = self.legs[index + 1]
                if layover.airport != previous.destination:
                    raise ValueError("layover airport must match the connecting legs")
                assert previous.arrival_utc is not None
                assert current.departure_utc is not None
                if current.departure_utc <= previous.arrival_utc:
                    raise ValueError("each connecting leg must depart after arrival")
                elapsed_layover = round(
                    (current.departure_utc - previous.arrival_utc).total_seconds() / 60
                )
                if abs(elapsed_layover - layover.duration_minutes) > 15:
                    raise ValueError("layover duration must match provider timestamps")
        elif self.kind == "one_stop":
            assert self.layover_airport is not None
            if self.layover_airport != self.legs[0].destination:
                raise ValueError("layover airport must match the connecting legs")

        expected_duration = sum(leg.duration_minutes for leg in self.legs) + (
            sum(layover.duration_minutes for layover in self.layovers)
            if self.layovers
            else (self.layover_minutes or 0)
        )
        if expected_duration != self.total_duration_minutes:
            raise ValueError("itinerary duration must equal its legs plus layover")
        if abs(sum(leg.distance_km for leg in self.legs) - self.total_distance_km) > 0.2:
            raise ValueError("itinerary distance must equal its leg distances")
        provider_basis = all(leg.data_basis != "model_duration_only" for leg in self.legs)
        if (self.time_basis == "provider_schedule") != provider_basis:
            raise ValueError("itinerary time basis must match every leg")
        if self.layover_status == "provider_confirmed" and (
            self.time_basis != "provider_schedule"
            or any(
                leg.data_basis
                not in {
                    "serpapi_booking_confirmed",
                    "searchapi_booking_confirmed",
                    "ignav_verified_booking_confirmed",
                }
                for leg in self.legs
            )
        ):
            raise ValueError(
                "provider-confirmed connections require booking-verified provider legs"
            )
        return self


class PriceForecastPoint(BaseModel):
    quote_date: date
    quote_time: datetime
    days_until_departure: float = Field(gt=0.0)
    estimated_price_usd: float = Field(ge=0.0)
    interval_80_low_usd: float = Field(ge=0.0)
    interval_80_high_usd: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> PriceForecastPoint:
        _require_timezone(self.quote_time)
        if not (self.interval_80_low_usd <= self.estimated_price_usd <= self.interval_80_high_usd):
            raise ValueError("price estimate must be inside its 80% interval")
        return self


class PriceForecastCurve(BaseModel):
    status: Literal["model_projection"] = "model_projection"
    basis: Literal["verified_fare_anchored_synthetic_trajectory"] = (
        "verified_fare_anchored_synthetic_trajectory"
    )
    calibration_method: Literal["log1p_offset_to_verified_fare"] = (
        "log1p_offset_to_verified_fare"
    )
    interval_basis: Literal["synthetic_demo_interval_log1p_shifted"] = (
        "synthetic_demo_interval_log1p_shifted"
    )
    currency: Literal["USD"] = "USD"
    anchor_price_usd: float = Field(gt=0.0, allow_inf_nan=False)
    anchor_verified_at: datetime
    anchor_provider_code: Literal[
        "serpapi_google_flights",
        "searchapi_google_flights",
        "ignav_verified_fares",
    ]
    raw_model_start_price_usd: float = Field(ge=0.0, allow_inf_nan=False)
    calibration_log1p_offset: float = Field(allow_inf_nan=False)
    start_date: date
    end_date: date
    generated_at: datetime
    historical_prices_available: Literal[False] = False
    extrapolated_beyond_training_horizon: bool = False
    points: list[PriceForecastPoint] = Field(min_length=1, max_length=371)
    notice: BilingualText

    @model_validator(mode="after")
    def validate_curve(self) -> PriceForecastCurve:
        _require_timezone(self.generated_at)
        _require_timezone(self.anchor_verified_at)
        if self.end_date < self.start_date:
            raise ValueError("price curve end date cannot precede its start date")
        quote_dates = [point.quote_date for point in self.points]
        if quote_dates != sorted(set(quote_dates)):
            raise ValueError("price curve quote dates must be unique and increasing")
        if quote_dates[0] != self.start_date or quote_dates[-1] != self.end_date:
            raise ValueError("price curve points must include the start and end dates")
        if self.points[0].estimated_price_usd != self.anchor_price_usd:
            raise ValueError("price curve must start exactly at its verified fare anchor")
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
    fare_search_metadata: FareSearchMetadata | None = None
    offer: ComparisonOffer
    itinerary: OfferItinerary
    historical_market_context: HistoricalMarketContext | None = None
    price_curve: PriceForecastCurve
    notice: BilingualText

    @model_validator(mode="after")
    def validate_provider_attribution(self) -> OfferDetailResponse:
        if self.offer.live_fare is None or self.fare_search_metadata is None:
            raise ValueError("strict offer detail requires fare evidence and search metadata")
        fare = self.offer.live_fare
        metadata = self.fare_search_metadata
        if not metadata.includes_confirmed_provider(fare.provider_code, fare.provider_name):
            raise ValueError(
                "offer detail metadata must contain the live fare's confirmed provider run"
            )
        provider_basis = {
            "serpapi_google_flights": "serpapi_booking_confirmed",
            "searchapi_google_flights": "searchapi_booking_confirmed",
            "ignav_verified_fares": "ignav_verified_booking_confirmed",
        }[fare.provider_code]
        if any(leg.data_basis != provider_basis for leg in self.itinerary.legs):
            raise ValueError("offer detail itinerary basis must match the live-fare provider")
        if (
            self.price_curve.anchor_provider_code != fare.provider_code
            or self.price_curve.anchor_price_usd != fare.total_amount
            or self.price_curve.anchor_verified_at != fare.verified_at
            or self.price_curve.raw_model_start_price_usd
            != self.offer.estimated_price_usd
        ):
            raise ValueError("price curve anchor must match the selected live fare")
        if self.historical_market_context is not None and (
            self.historical_market_context.origin != self.origin
            or self.historical_market_context.destination != self.destination
            or self.historical_market_context.departure_date != self.departure_date
            or self.historical_market_context.cabin != fare.cabin_summary
        ):
            raise ValueError(
                "historical market context must match the selected route, date, and cabin"
            )
        return self
