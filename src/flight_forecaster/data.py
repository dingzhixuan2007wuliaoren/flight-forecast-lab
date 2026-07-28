from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flight_forecaster.catalog import comparison_airlines

ROUTES = (
    ("JFK", "LAX", 3_983, 365, 320),
    ("LAX", "JFK", 3_983, 345, 315),
    ("SFO", "SEA", 1_093, 130, 150),
    ("SEA", "SFO", 1_093, 125, 145),
    ("ORD", "MIA", 1_933, 185, 205),
    ("MIA", "ORD", 1_933, 195, 210),
    ("ATL", "BOS", 1_520, 165, 185),
    ("BOS", "ATL", 1_520, 175, 190),
    ("DEN", "LAS", 1_010, 120, 135),
    ("LAS", "DEN", 1_010, 115, 130),
    ("DFW", "SFO", 2_353, 235, 245),
    ("SFO", "DFW", 2_353, 220, 240),
    ("JFK", "LHR", 5_554, 415, 520),
    ("LHR", "JFK", 5_554, 485, 540),
    ("CDG", "DXB", 5_248, 405, 430),
    ("DXB", "CDG", 5_248, 445, 450),
    ("DXB", "SYD", 12_040, 825, 840),
    ("SYD", "DXB", 12_040, 875, 870),
    ("SIN", "LHR", 10_875, 835, 850),
    ("LHR", "SIN", 10_875, 790, 820),
    ("HND", "LAX", 8_815, 620, 760),
    ("LAX", "HND", 8_815, 690, 790),
    ("YYZ", "YVR", 3_349, 310, 340),
    ("YVR", "YYZ", 3_349, 275, 330),
    ("GRU", "MAD", 8_378, 630, 650),
    ("MAD", "GRU", 8_378, 660, 680),
    ("JNB", "LHR", 9_080, 665, 690),
    ("LHR", "JNB", 9_080, 650, 670),
    ("DOH", "AKL", 14_535, 1_050, 1_100),
    ("AKL", "DOH", 14_535, 1_020, 1_070),
    ("SCL", "LIM", 2_462, 235, 230),
    ("LIM", "SCL", 2_462, 225, 220),
    ("NRT", "SIN", 5_350, 430, 480),
    ("SIN", "NRT", 5_350, 420, 470),
    ("DEL", "FRA", 6_130, 500, 510),
    ("FRA", "DEL", 6_130, 475, 500),
    ("CAI", "JED", 1_215, 140, 190),
    ("JED", "CAI", 1_215, 145, 185),
    ("AKL", "SYD", 2_160, 220, 260),
    ("SYD", "AKL", 2_160, 190, 250),
    ("MEX", "BOG", 3_150, 270, 290),
    ("BOG", "MEX", 3_150, 285, 300),
    ("IST", "JFK", 8_040, 660, 690),
    ("JFK", "IST", 8_040, 585, 670),
)

_AIRLINE_PROFILES = comparison_airlines()
AIRLINES = np.array([profile.code for profile in _AIRLINE_PROFILES])
_SERVICE_PRICE = {"low_cost": 0.82, "hybrid": 0.93, "full_service": 1.04}
_SERVICE_DISRUPTION = {"low_cost": 0.12, "hybrid": 0.04, "full_service": -0.05}


def _stable_carrier_offset(code: str) -> float:
    """Small deterministic variation used only by the synthetic global demo."""

    return ((sum(ord(character) for character in code) % 13) - 6) * 0.006


AIRLINE_PRICE_MULTIPLIER = {
    profile.code: _SERVICE_PRICE[profile.service_model] + _stable_carrier_offset(profile.code)
    for profile in _AIRLINE_PROFILES
}
AIRLINE_DISRUPTION_EFFECT = {
    profile.code: _SERVICE_DISRUPTION[profile.service_model]
    + _stable_carrier_offset(profile.code) * 1.8
    for profile in _AIRLINE_PROFILES
}
_AIRLINE_WEIGHTS = np.array(
    [0.75 if profile.service_model == "low_cost" else 1.0 for profile in _AIRLINE_PROFILES]
)
AIRLINE_PROBABILITIES = _AIRLINE_WEIGHTS / _AIRLINE_WEIGHTS.sum()
CABINS = np.array(["economy", "premium_economy", "business", "first"])
CABIN_MULTIPLIER = {
    "economy": 1.0,
    "premium_economy": 1.55,
    "business": 3.15,
    "first": 4.50,
}


def _sample_routes(rng: np.random.Generator, rows: int) -> list[tuple[str, str, int, int, int]]:
    indices = rng.integers(0, len(ROUTES), size=rows)
    return [ROUTES[index] for index in indices]


def generate_demo_price_data(rows: int = 6_000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic fare observations with realistic, learnable relationships."""
    if rows < 500:
        raise ValueError("price demo data requires at least 500 rows")
    rng = np.random.default_rng(seed)
    routes = _sample_routes(rng, rows)
    observed_day = rng.integers(0, 1_000, size=rows)
    observed_hour = rng.integers(0, 24, size=rows)
    quote_time = (
        pd.Timestamp("2023-01-01", tz="UTC")
        + pd.to_timedelta(observed_day, unit="D")
        + pd.to_timedelta(observed_hour, unit="h")
    )
    lead_days = np.clip(rng.gamma(shape=2.8, scale=18.0, size=rows).astype(int) + 2, 2, 180)
    departure_hour = rng.choice([6, 8, 10, 13, 16, 18, 21], size=rows)
    departure_time = (
        quote_time.normalize()
        + pd.to_timedelta(lead_days, unit="D")
        + pd.to_timedelta(departure_hour, unit="h")
    )

    airline = rng.choice(AIRLINES, size=rows, p=AIRLINE_PROBABILITIES)
    cabin = rng.choice(CABINS, size=rows, p=[0.82, 0.09, 0.075, 0.015])
    stops = rng.choice([0, 1, 2], size=rows, p=[0.72, 0.25, 0.03])
    news_disruption = np.clip(
        rng.beta(0.8, 8.0, size=rows) + (rng.random(rows) < 0.025) * rng.uniform(0.4, 0.9, rows),
        0,
        1,
    )
    month = departure_time.month.to_numpy()
    weekday = departure_time.weekday.to_numpy()

    base_fare = np.array([route[4] for route in routes], dtype=float)
    distance_km = np.array([route[2] for route in routes], dtype=float)
    duration_minutes = np.array([route[3] for route in routes], dtype=float)
    duration_minutes += stops * rng.integers(45, 130, size=rows) + rng.normal(0, 12, size=rows)

    urgency = 1.0 + 0.80 * np.exp(-lead_days / 12.0) + 0.10 * (lead_days > 120)
    season = np.where(np.isin(month, [6, 7, 12]), 1.18, 1.0)
    weekend = np.where(np.isin(weekday, [4, 5, 6]), 1.08, 1.0)
    stop_discount = np.choose(stops, [1.0, 0.84, 0.72])
    carrier = np.array([AIRLINE_PRICE_MULTIPLIER[value] for value in airline])
    cabin_factor = np.array([CABIN_MULTIPLIER[value] for value in cabin])
    market_noise = rng.lognormal(mean=0.0, sigma=0.13, size=rows)
    news_factor = 1.0 + 0.08 * news_disruption
    price = (
        base_fare
        * urgency
        * season
        * weekend
        * stop_discount
        * carrier
        * cabin_factor
        * news_factor
    )
    price = np.maximum(49.0, price * market_noise)

    return pd.DataFrame(
        {
            "quote_time": quote_time,
            "departure_time": departure_time,
            "origin": [route[0] for route in routes],
            "destination": [route[1] for route in routes],
            "airline": airline,
            "cabin": cabin,
            "stops": stops,
            "duration_minutes": np.round(duration_minutes).astype(int),
            "distance_km": distance_km,
            "news_disruption_index": np.round(news_disruption, 4),
            "price_usd": np.round(price, 2),
        }
    )


def generate_demo_ontime_data(rows: int = 8_000, seed: int = 43) -> pd.DataFrame:
    """Create deterministic operations data using only features known before departure."""
    if rows < 500:
        raise ValueError("on-time demo data requires at least 500 rows")
    rng = np.random.default_rng(seed)
    routes = _sample_routes(rng, rows)
    scheduled_day = rng.integers(0, 1_000, size=rows)
    scheduled_hour = rng.choice([5, 6, 8, 10, 13, 16, 18, 20, 22], size=rows)
    scheduled = (
        pd.Timestamp("2023-01-01", tz="UTC")
        + pd.to_timedelta(scheduled_day, unit="D")
        + pd.to_timedelta(scheduled_hour, unit="h")
    )
    airline = rng.choice(AIRLINES, size=rows, p=AIRLINE_PROBABILITIES)
    news_disruption = np.clip(
        rng.beta(0.8, 8.0, size=rows) + (rng.random(rows) < 0.025) * rng.uniform(0.4, 0.9, rows),
        0,
        1,
    )
    distance_km = np.array([route[2] for route in routes], dtype=float)
    month = scheduled.month.to_numpy()
    weekday = scheduled.weekday.to_numpy()

    winter = np.isin(month, [1, 2, 12]).astype(float)
    weather = np.clip(rng.beta(1.6, 5.5, size=rows) + winter * rng.uniform(0.0, 0.25, rows), 0, 1)
    airport_base = {
        "ATL": 0.67,
        "BOS": 0.57,
        "DEN": 0.52,
        "DFW": 0.63,
        "JFK": 0.72,
        "LAS": 0.45,
        "LAX": 0.66,
        "MIA": 0.55,
        "ORD": 0.73,
        "SEA": 0.50,
        "SFO": 0.64,
    }
    origin_congestion = np.array([airport_base.get(route[0], 0.56) for route in routes])
    origin_congestion = np.clip(origin_congestion + rng.normal(0, 0.09, rows), 0, 1)
    peak = np.isin(scheduled_hour, [6, 8, 16, 18, 20]).astype(float)
    weekend = np.isin(weekday, [5, 6]).astype(float)
    carrier_effect = np.array([AIRLINE_DISRUPTION_EFFECT[value] for value in airline])

    disruption_logit = (
        -2.25
        + 2.80 * weather
        + 1.45 * origin_congestion
        + 0.45 * peak
        - 0.16 * weekend
        + 0.16 * (distance_km > 3_000)
        + 1.10 * news_disruption
        + carrier_effect
    )
    disruption_probability = 1.0 / (1.0 + np.exp(-disruption_logit))
    disrupted = rng.random(rows) < disruption_probability
    cancellation_probability = np.clip(
        0.003 + 0.07 * weather**2 + 0.015 * origin_congestion, 0, 0.20
    )
    cancelled = disrupted & (rng.random(rows) < cancellation_probability)
    arrival_delay = np.where(
        disrupted,
        15 + rng.gamma(shape=2.2, scale=18.0, size=rows),
        rng.normal(loc=-3.0, scale=7.0, size=rows),
    )
    arrival_delay = np.where(cancelled, np.nan, arrival_delay)
    on_time = (~cancelled & (arrival_delay < 15)).astype(int)

    return pd.DataFrame(
        {
            "scheduled_departure": scheduled,
            "origin": [route[0] for route in routes],
            "destination": [route[1] for route in routes],
            "airline": airline,
            "distance_km": distance_km,
            "weather_severity_forecast": np.round(weather, 4),
            "origin_congestion_index": np.round(origin_congestion, 4),
            "news_disruption_index": np.round(news_disruption, 4),
            "cancelled": cancelled.astype(int),
            "arrival_delay_minutes": np.round(arrival_delay, 1),
            "on_time": on_time,
        }
    )


def load_price_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "price_usd" not in frame:
        raise ValueError("price CSV must contain a price_usd target column")
    return frame


def load_ontime_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "on_time" not in frame:
        if {"cancelled", "arrival_delay_minutes"}.issubset(frame.columns):
            cancelled = frame["cancelled"].fillna(0).astype(bool)
            delay = pd.to_numeric(frame["arrival_delay_minutes"], errors="coerce")
            unresolved = ~cancelled & delay.isna()
            if unresolved.any():
                raise ValueError(
                    "non-cancelled rows need arrival_delay_minutes; unresolved outcomes "
                    "must not be labelled as disruptions"
                )
            frame["on_time"] = (~cancelled & delay.lt(15)).astype(int)
        else:
            raise ValueError(
                "on-time CSV needs on_time, or both cancelled and arrival_delay_minutes"
            )
    return frame
