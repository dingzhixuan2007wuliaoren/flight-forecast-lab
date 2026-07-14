from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_CATEGORICAL_FEATURES = ["origin", "destination", "route", "airline", "cabin"]
PRICE_NUMERIC_FEATURES = [
    "stops",
    "duration_minutes",
    "distance_km",
    "days_until_departure",
    "news_disruption_index",
    "departure_month_sin",
    "departure_month_cos",
    "departure_weekday_sin",
    "departure_weekday_cos",
    "departure_hour_sin",
    "departure_hour_cos",
    "is_weekend",
]
PRICE_FEATURES = PRICE_CATEGORICAL_FEATURES + PRICE_NUMERIC_FEATURES

ONTIME_CATEGORICAL_FEATURES = ["origin", "destination", "route", "airline"]
ONTIME_NUMERIC_FEATURES = [
    "distance_km",
    "weather_severity_forecast",
    "origin_congestion_index",
    "news_disruption_index",
    "departure_month_sin",
    "departure_month_cos",
    "departure_weekday_sin",
    "departure_weekday_cos",
    "departure_hour_sin",
    "departure_hour_cos",
    "is_weekend",
    "is_peak_hour",
]
ONTIME_FEATURES = ONTIME_CATEGORICAL_FEATURES + ONTIME_NUMERIC_FEATURES


def _cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * values.astype(float) / period
    return np.sin(radians), np.cos(radians)


def _codes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("origin", "destination", "airline"):
        result[column] = result[column].astype(str).str.strip().str.upper()
    result["route"] = result["origin"] + "-" + result["destination"]
    return result


def _local_time_parts(
    frame: pd.DataFrame,
    fallback: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    components: list[pd.Series] = []
    bounds = (
        ("departure_local_month", 1, 12),
        ("departure_local_weekday", 0, 6),
        ("departure_local_hour", 0, 23),
    )
    defaults = (fallback.dt.month, fallback.dt.weekday, fallback.dt.hour)
    for (column, minimum, maximum), default in zip(bounds, defaults, strict=True):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="raise")
            if not values.between(minimum, maximum).all():
                raise ValueError(f"{column} must be between {minimum} and {maximum}")
            components.append(values.astype(int))
        else:
            components.append(default)
    return components[0], components[1], components[2]


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "origin",
        "destination",
        "airline",
        "cabin",
        "stops",
        "duration_minutes",
        "distance_km",
        "quote_time",
        "departure_time",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"price data is missing columns: {sorted(missing)}")

    result = _codes(frame)
    if "news_disruption_index" not in result:
        result["news_disruption_index"] = 0.0
    quote_time = pd.to_datetime(result["quote_time"], utc=True, errors="raise")
    departure_time = pd.to_datetime(result["departure_time"], utc=True, errors="raise")
    lead_days = (departure_time - quote_time).dt.total_seconds() / 86_400.0
    if (lead_days <= 0).any():
        raise ValueError("all departure_time values must be after quote_time")

    result["days_until_departure"] = lead_days
    month, weekday, hour = _local_time_parts(result, departure_time)
    result["departure_month_sin"], result["departure_month_cos"] = _cyclic(month, 12)
    result["departure_weekday_sin"], result["departure_weekday_cos"] = _cyclic(weekday, 7)
    result["departure_hour_sin"], result["departure_hour_cos"] = _cyclic(hour, 24)
    result["is_weekend"] = weekday.isin([5, 6]).astype(int)
    return result[PRICE_FEATURES]


def build_ontime_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "origin",
        "destination",
        "airline",
        "distance_km",
        "scheduled_departure",
        "weather_severity_forecast",
        "origin_congestion_index",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"on-time data is missing columns: {sorted(missing)}")

    result = _codes(frame)
    if "news_disruption_index" not in result:
        result["news_disruption_index"] = 0.0
    scheduled = pd.to_datetime(result["scheduled_departure"], utc=True, errors="raise")
    month, weekday, hour = _local_time_parts(result, scheduled)
    result["departure_month_sin"], result["departure_month_cos"] = _cyclic(month, 12)
    result["departure_weekday_sin"], result["departure_weekday_cos"] = _cyclic(weekday, 7)
    result["departure_hour_sin"], result["departure_hour_cos"] = _cyclic(hour, 24)
    result["is_weekend"] = weekday.isin([5, 6]).astype(int)
    result["is_peak_hour"] = hour.isin([6, 7, 8, 16, 17, 18, 19]).astype(int)
    return result[ONTIME_FEATURES]
