from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

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
    departure_time: datetime

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_trip(self) -> ComparisonRequest:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
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
    operations: ContextSignal
    news: NewsSignal


class ComparisonOffer(BaseModel):
    id: str
    airline_code: str
    airline_name: str
    cabin: Cabin
    stops: int = Field(ge=0, le=3)
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
    cabin_status: Literal["catalog_scenario"]
    punctuality_basis: Literal["direct_leg_model", "two_leg_independence_scenario"]


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
    departure_time: datetime
    departure_timezone: str
    distance_km: float
    duration_minutes: int
    generated_at: datetime
    context: PredictionContextResponse
    offers: list[ComparisonOffer]
    rankings: ComparisonRankings
    warnings: BilingualWarning
    model_version: str
