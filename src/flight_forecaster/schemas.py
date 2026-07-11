from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    airline: str = Field(examples=["DL"])
    cabin: Literal["economy", "premium_economy", "business", "first"] = "economy"
    stops: int = Field(default=0, ge=0, le=3)
    duration_minutes: int = Field(gt=30, le=1800)
    distance_km: float = Field(gt=50, le=20_000)
    quote_time: datetime
    departure_time: datetime

    @field_validator("origin", "destination")
    @classmethod
    def validate_airport(cls, value: str) -> str:
        return _normalise_code(value, min_length=3, max_length=3)

    @field_validator("airline")
    @classmethod
    def validate_airline(cls, value: str) -> str:
        return _normalise_code(value, min_length=2, max_length=3)

    @field_validator("quote_time", "departure_time")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_trip(self) -> PriceRequest:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if self.departure_time <= self.quote_time:
            raise ValueError("departure_time must be after quote_time")
        if (self.departure_time - self.quote_time).days > 370:
            raise ValueError("departure_time must be within 370 days of quote_time")
        return self


class OnTimeRequest(BaseModel):
    origin: str = Field(examples=["JFK"])
    destination: str = Field(examples=["LAX"])
    airline: str = Field(examples=["DL"])
    distance_km: float = Field(gt=50, le=20_000)
    scheduled_departure: datetime
    weather_severity_forecast: float = Field(default=0.2, ge=0.0, le=1.0)
    origin_congestion_index: float = Field(default=0.4, ge=0.0, le=1.0)

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
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        return self


class PricePrediction(BaseModel):
    estimated_price_usd: float
    interval_80_low_usd: float
    interval_80_high_usd: float
    days_until_departure: float
    model_version: str
    warning: str


class OnTimePrediction(BaseModel):
    on_time_probability: float
    disruption_probability: float
    risk_level: Literal["low", "medium", "high"]
    definition: str
    model_version: str
