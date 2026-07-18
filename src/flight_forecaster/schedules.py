"""Conservative AirLabs schedule/timetable integration.

Only complete, internally consistent direct-flight rows become provider-backed
offers.  All other cases are handed back to the caller as an explicit model
fallback; this module never fabricates a flight number or clock time.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

from flight_forecaster.airlabs_quota import AirLabsQuotaGate
from flight_forecaster.context import (
    AIRLABS_FREE_SAMPLE_LIMIT,
    AIRLABS_ROUTES_URL,
    AIRLABS_SCHEDULES_URL,
    REQUEST_TIMEOUT_SECONDS,
)

SCHEDULE_CACHE_TTL_SECONDS = 300.0
MAX_SCHEDULE_CACHE_ENTRIES = 256
MAX_FLIGHT_DURATION_MINUTES = 2_160
_CODE_PATTERN = re.compile(r"^[A-Z0-9]{2,3}$")
_FLIGHT_PATTERN = re.compile(r"^[A-Z0-9]{3,12}$")
_TERMINAL_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,40}$")
_DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True, slots=True)
class FlightSchedule:
    airline_code: str
    flight_number: str
    schedule_status: str
    source: str
    departure_local: datetime
    arrival_local: datetime
    departure_utc: datetime
    arrival_utc: datetime
    duration_minutes: int
    departure_terminal: str | None = None
    arrival_terminal: str | None = None
    aircraft_icao: str | None = None
    provider_flight_status: str | None = None
    observed_at: datetime | None = None

    @property
    def identity(self) -> tuple[str, str, datetime]:
        return (self.airline_code, self.flight_number, self.departure_utc)


@dataclass(frozen=True, slots=True)
class ScheduleSearchResult:
    schedules: tuple[FlightSchedule, ...]
    route_airlines: frozenset[str]
    fallback_code: str | None = None
    sample_truncated: bool = False


class ScheduleProvider:
    """Fetch and validate free-tier AirLabs schedule and route rows."""

    def __init__(
        self,
        *,
        api_key: str | None,
        client: Any,
        enabled: bool = True,
        aerodatabox_provider: Any | None = None,
        airlabs_quota_gate: AirLabsQuotaGate | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.client = client
        self.enabled = bool(enabled)
        self.aerodatabox_provider = aerodatabox_provider
        self.airlabs_quota_gate = airlabs_quota_gate or AirLabsQuotaGate.from_env()
        self._cache: dict[tuple[str, str, date], tuple[float, ScheduleSearchResult]] = {}
        self._lock = threading.Lock()

    def search(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        *,
        origin_timezone: ZoneInfo,
        destination_timezone: ZoneInfo,
        fetched_at: datetime | None = None,
    ) -> ScheduleSearchResult:
        origin_code = self._airport_code(origin)
        destination_code = self._airport_code(destination)
        if len(origin_code) != 3 or len(destination_code) != 3 or origin_code == destination_code:
            return ScheduleSearchResult((), frozenset(), "invalid_route")
        if not isinstance(departure_date, date) or isinstance(departure_date, datetime):
            return ScheduleSearchResult((), frozenset(), "invalid_departure_date")
        if not self.enabled:
            return ScheduleSearchResult((), frozenset(), "external_context_disabled")

        key = (origin_code, destination_code, departure_date)
        now_tick = monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now_tick - cached[0] <= SCHEDULE_CACHE_TTL_SECONDS:
                return cached[1]

        observed = self._as_utc(fetched_at or datetime.now(UTC))
        reasons: list[str] = []
        schedules: list[FlightSchedule] = []
        route_airlines: set[str] = set()
        blocked_live_identities: set[tuple[str, str, datetime]] = set()
        sample_truncated = False

        if self.api_key:
            # The live schedules endpoint only covers roughly the next ten hours. Avoid
            # spending a free-tier call for dates that cannot be in that window.
            origin_today = observed.astimezone(origin_timezone).date()
            if origin_today <= departure_date <= origin_today + timedelta(days=1):
                try:
                    payload = self._get_json(
                        AIRLABS_SCHEDULES_URL,
                        {
                            "api_key": self.api_key,
                            "dep_iata": origin_code,
                            "arr_iata": destination_code,
                            "limit": AIRLABS_FREE_SAMPLE_LIMIT,
                        },
                    )
                    sample_truncated = sample_truncated or self._sample_is_truncated(payload)
                    rows = self._response_rows(payload)
                    for row in rows:
                        airline = self._matching_airline(row, origin_code, destination_code)
                        if airline is not None:
                            route_airlines.add(airline)
                        live_identity = self._live_identity(
                            row,
                            origin_code,
                            destination_code,
                            departure_date,
                            origin_timezone,
                        )
                        if live_identity is not None and (
                            live_identity[2] <= observed
                            or not self._live_status_is_eligible(self._status(row.get("status")))
                        ):
                            blocked_live_identities.add(live_identity)
                        parsed = self._parse_live_row(
                            row,
                            origin_code,
                            destination_code,
                            departure_date,
                            origin_timezone,
                            destination_timezone,
                            observed,
                        )
                        if parsed is not None:
                            schedules.append(parsed)
                except Exception:
                    reasons.append("airlabs_schedules_unavailable")

            # The routes DB can fill the remainder of a date outside the live endpoint's
            # roughly ten-hour window. Rows stay explicitly labelled as projections.
            try:
                payload = self._get_json(
                    AIRLABS_ROUTES_URL,
                    {
                        "api_key": self.api_key,
                        "dep_iata": origin_code,
                        "arr_iata": destination_code,
                        "limit": AIRLABS_FREE_SAMPLE_LIMIT,
                    },
                )
                sample_truncated = sample_truncated or self._sample_is_truncated(payload)
                rows = self._response_rows(payload)
                for row in rows:
                    airline = self._matching_airline(row, origin_code, destination_code)
                    if airline is not None:
                        route_airlines.add(airline)
                    parsed = self._parse_recurring_row(
                        row,
                        origin_code,
                        destination_code,
                        departure_date,
                        origin_timezone,
                        destination_timezone,
                    )
                    if parsed is not None:
                        schedules.append(parsed)
            except Exception:
                reasons.append("airlabs_routes_unavailable")
        else:
            reasons.append("airlabs_api_key_not_configured")

        # A dated AeroDataBox result is more specific than an AirLabs recurring
        # projection. It remains a reference-only row and never enters fare offers.
        has_dated_schedule = any(
            schedule.schedule_status == "live_schedule" for schedule in schedules
        )
        if not has_dated_schedule and self.aerodatabox_provider is not None:
            try:
                supplemental = self.aerodatabox_provider.search(
                    origin_code,
                    destination_code,
                    departure_date,
                    origin_timezone=origin_timezone,
                    destination_timezone=destination_timezone,
                    fetched_at=observed,
                )
                sample_truncated = sample_truncated or supplemental.sample_truncated
                if supplemental.fallback_code:
                    reasons.append(supplemental.fallback_code)
                for item in supplemental.schedules:
                    route_airlines.add(item.airline_code)
                    schedules.append(
                        FlightSchedule(
                            airline_code=item.airline_code,
                            flight_number=item.flight_number,
                            schedule_status="future_schedule_reference",
                            source="aerodatabox_schedule",
                            departure_local=item.departure_local,
                            arrival_local=item.arrival_local,
                            departure_utc=item.departure_utc,
                            arrival_utc=item.arrival_utc,
                            duration_minutes=item.duration_minutes,
                            departure_terminal=item.departure_terminal,
                            arrival_terminal=item.arrival_terminal,
                            aircraft_icao=item.aircraft_icao,
                            provider_flight_status=item.provider_flight_status,
                            observed_at=item.observed_at,
                        )
                    )
            except Exception:
                reasons.append("aerodatabox_provider_unavailable")

        unique: dict[tuple[str, str, datetime], FlightSchedule] = {}
        schedule_priority = {
            "live_schedule": 3,
            "future_schedule_reference": 2,
            "recurring_timetable_projection": 1,
        }
        for schedule in schedules:
            existing = unique.get(schedule.identity)
            if existing is None or schedule_priority.get(schedule.schedule_status, 0) > (
                schedule_priority.get(existing.schedule_status, 0)
            ):
                unique[schedule.identity] = schedule
        eligible = [
            schedule
            for identity, schedule in unique.items()
            if identity not in blocked_live_identities
            and schedule.departure_utc > observed
            and (
                schedule.schedule_status != "live_schedule"
                or self._live_status_is_eligible(schedule.provider_flight_status)
            )
        ]
        ordered = tuple(
            sorted(
                eligible,
                key=lambda item: (
                    item.departure_utc,
                    item.airline_code,
                    item.flight_number,
                ),
            )
        )
        result = ScheduleSearchResult(
            schedules=ordered,
            route_airlines=frozenset(route_airlines),
            fallback_code=(
                reasons[-1]
                if not ordered and reasons
                else ("no_complete_provider_schedule" if not ordered else None)
            ),
            sample_truncated=sample_truncated,
        )
        with self._lock:
            if len(self._cache) >= MAX_SCHEDULE_CACHE_ENTRIES:
                oldest = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest, None)
            self._cache[key] = (monotonic(), result)
        return result

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        return self.airlabs_quota_gate.get_json(
            self.client,
            url,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "flight-forecast-lab/0.2.0 (AirLabs schedule client)",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _response_rows(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or isinstance(payload.get("error"), dict):
            raise LookupError("AirLabs response object is invalid")
        rows = payload.get("response")
        if not isinstance(rows, list):
            raise LookupError("AirLabs response list is missing")
        return [row for row in rows[:AIRLABS_FREE_SAMPLE_LIMIT] if isinstance(row, dict)]

    @staticmethod
    def _sample_is_truncated(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        rows = payload.get("response")
        if isinstance(rows, list) and len(rows) >= AIRLABS_FREE_SAMPLE_LIMIT:
            return True
        request = payload.get("request")
        if not isinstance(request, dict):
            return False
        has_more = request.get("has_more")
        if isinstance(has_more, bool):
            return has_more
        if isinstance(has_more, str):
            return has_more.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    @classmethod
    def _matching_airline(
        cls,
        row: dict[str, Any],
        origin: str,
        destination: str,
    ) -> str | None:
        if cls._airport_code(row.get("dep_iata")) != origin:
            return None
        if cls._airport_code(row.get("arr_iata")) != destination:
            return None
        airline = cls._airport_code(row.get("airline_iata"))
        return airline if _CODE_PATTERN.fullmatch(airline) else None

    @classmethod
    def _parse_live_row(
        cls,
        row: dict[str, Any],
        origin: str,
        destination: str,
        requested_date: date,
        origin_timezone: ZoneInfo,
        destination_timezone: ZoneInfo,
        observed_at: datetime,
    ) -> FlightSchedule | None:
        airline = cls._matching_airline(row, origin, destination)
        flight = cls._flight_number(row, airline)
        if airline is None or flight is None:
            return None
        departure = cls._complete_local_datetime(
            row,
            "dep_time",
            "dep_time_utc",
            "dep_time_ts",
            origin_timezone,
        )
        arrival = cls._complete_local_datetime(
            row,
            "arr_time",
            "arr_time_utc",
            "arr_time_ts",
            destination_timezone,
        )
        if departure is None or arrival is None or departure.date() != requested_date:
            return None
        departure_utc = departure.astimezone(UTC)
        arrival_utc = arrival.astimezone(UTC)
        duration = cls._validated_duration(row.get("duration"), departure_utc, arrival_utc)
        if duration is None:
            return None
        return FlightSchedule(
            airline_code=airline,
            flight_number=flight,
            schedule_status="live_schedule",
            source="airlabs_schedules",
            departure_local=departure,
            arrival_local=arrival,
            departure_utc=departure_utc,
            arrival_utc=arrival_utc,
            duration_minutes=duration,
            departure_terminal=cls._terminal(row.get("dep_terminal")),
            arrival_terminal=cls._terminal(row.get("arr_terminal")),
            aircraft_icao=cls._aircraft(row.get("aircraft_icao")),
            provider_flight_status=cls._status(row.get("status")),
            observed_at=observed_at,
        )

    @classmethod
    def _parse_recurring_row(
        cls,
        row: dict[str, Any],
        origin: str,
        destination: str,
        requested_date: date,
        origin_timezone: ZoneInfo,
        destination_timezone: ZoneInfo,
    ) -> FlightSchedule | None:
        airline = cls._matching_airline(row, origin, destination)
        flight = cls._flight_number(row, airline)
        days = row.get("days")
        if airline is None or flight is None or not isinstance(days, list):
            return None
        valid_days = {str(value).strip().lower() for value in days if isinstance(value, str)}
        if _DAY_NAMES[requested_date.weekday()] not in valid_days:
            return None
        departure_clock = cls._clock(row.get("dep_time"))
        arrival_clock = cls._clock(row.get("arr_time"))
        duration_value = cls._positive_int(row.get("duration"))
        if (
            departure_clock is None
            or arrival_clock is None
            or duration_value is None
            or duration_value > MAX_FLIGHT_DURATION_MINUTES
        ):
            return None
        departure = cls._localize(
            datetime.combine(requested_date, departure_clock),
            origin_timezone,
        )
        if departure is None:
            return None
        departure_utc = departure.astimezone(UTC)
        arrival_utc = departure_utc + timedelta(minutes=duration_value)
        arrival = arrival_utc.astimezone(destination_timezone)
        # Both the provider arrival clock and duration must agree. This also safely
        # resolves overnight and International Date Line arrivals.
        expected = datetime.combine(arrival.date(), arrival_clock)
        clock_delta = abs((arrival.replace(tzinfo=None) - expected).total_seconds())
        if clock_delta > 5 * 60:
            return None
        observed = cls._parse_provider_datetime(row.get("updated"), UTC)
        return FlightSchedule(
            airline_code=airline,
            flight_number=flight,
            schedule_status="recurring_timetable_projection",
            source="airlabs_routes",
            departure_local=departure,
            arrival_local=arrival,
            departure_utc=departure_utc,
            arrival_utc=arrival_utc,
            duration_minutes=duration_value,
            observed_at=observed,
        )

    @classmethod
    def _live_identity(
        cls,
        row: dict[str, Any],
        origin: str,
        destination: str,
        requested_date: date,
        origin_timezone: ZoneInfo,
    ) -> tuple[str, str, datetime] | None:
        airline = cls._matching_airline(row, origin, destination)
        flight = cls._flight_number(row, airline)
        departure = cls._complete_local_datetime(
            row,
            "dep_time",
            "dep_time_utc",
            "dep_time_ts",
            origin_timezone,
        )
        if (
            airline is None
            or flight is None
            or departure is None
            or departure.date() != requested_date
        ):
            return None
        return (airline, flight, departure.astimezone(UTC))

    @staticmethod
    def _live_status_is_eligible(status: str | None) -> bool:
        if status in {None, "scheduled"}:
            return True
        known_non_scheduled = {
            "active",
            "canceled",
            "cancelled",
            "departed",
            "diverted",
            "en-route",
            "landed",
        }
        # Undocumented values are kept and surfaced verbatim instead of silently
        # guessing that they mean a cancellation or completed flight.
        return status not in known_non_scheduled

    @classmethod
    def _complete_local_datetime(
        cls,
        row: dict[str, Any],
        local_key: str,
        utc_key: str,
        timestamp_key: str,
        timezone: ZoneInfo,
    ) -> datetime | None:
        local_naive = cls._parse_naive_datetime(row.get(local_key))
        if local_naive is None:
            return None
        utc_value = cls._parse_provider_datetime(row.get(utc_key), UTC)
        timestamp = row.get(timestamp_key)
        timestamp_value: datetime | None = None
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
            try:
                timestamp_value = datetime.fromtimestamp(float(timestamp), tz=UTC)
            except (OverflowError, OSError, ValueError):
                return None
        if utc_value is not None and timestamp_value is not None:
            if abs((utc_value - timestamp_value).total_seconds()) > 120:
                return None
        authoritative_utc = utc_value or timestamp_value
        if authoritative_utc is not None:
            localized = authoritative_utc.astimezone(timezone)
            if abs((localized.replace(tzinfo=None) - local_naive).total_seconds()) > 120:
                return None
            return localized
        return cls._localize(local_naive, timezone)

    @staticmethod
    def _localize(value: datetime, timezone: ZoneInfo) -> datetime | None:
        first = value.replace(tzinfo=timezone, fold=0)
        round_trip = first.astimezone(UTC).astimezone(timezone).replace(tzinfo=None)
        if round_trip != value:
            return None
        alternate = value.replace(tzinfo=timezone, fold=1)
        if alternate.utcoffset() != first.utcoffset():
            return None
        return first

    @classmethod
    def _validated_duration(
        cls,
        raw_duration: Any,
        departure_utc: datetime,
        arrival_utc: datetime,
    ) -> int | None:
        elapsed = round((arrival_utc - departure_utc).total_seconds() / 60)
        if elapsed <= 0 or elapsed > MAX_FLIGHT_DURATION_MINUTES:
            return None
        provider_duration = cls._positive_int(raw_duration)
        if provider_duration is not None and abs(provider_duration - elapsed) > 15:
            return None
        return elapsed

    @classmethod
    def _flight_number(cls, row: dict[str, Any], airline: str | None) -> str | None:
        if airline is None:
            return None
        flight_iata = cls._airport_code(row.get("flight_iata"))
        if _FLIGHT_PATTERN.fullmatch(flight_iata) and flight_iata.startswith(airline):
            return flight_iata
        number = cls._airport_code(row.get("flight_number"))
        candidate = f"{airline}{number}"
        return candidate if number and _FLIGHT_PATTERN.fullmatch(candidate) else None

    @staticmethod
    def _airport_code(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _parse_naive_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str) or len(value) > 32:
            return None
        cleaned = value.strip().replace("T", " ")
        for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(cleaned, pattern)
            except ValueError:
                continue
        return None

    @classmethod
    def _parse_provider_datetime(cls, value: Any, timezone: ZoneInfo) -> datetime | None:
        if not isinstance(value, str) or len(value) > 40:
            return None
        cleaned = value.strip()
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            parsed = cls._parse_naive_datetime(cleaned)
        if parsed is None:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone)
        return parsed.astimezone(timezone)

    @staticmethod
    def _clock(value: Any) -> time | None:
        if not isinstance(value, str) or len(value) > 8:
            return None
        for pattern in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(value.strip(), pattern).time()
            except ValueError:
                continue
        return None

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not number.is_integer() or number <= 0:
            return None
        return int(number)

    @classmethod
    def _terminal(cls, value: Any) -> str | None:
        terminal = str(value or "").strip()
        return terminal if _TERMINAL_PATTERN.fullmatch(terminal) else None

    @classmethod
    def _first_terminal(cls, value: Any) -> str | None:
        if not isinstance(value, list):
            return None
        return next((terminal for item in value if (terminal := cls._terminal(item))), None)

    @staticmethod
    def _aircraft(value: Any) -> str | None:
        aircraft = str(value or "").strip().upper()
        return aircraft if re.fullmatch(r"[A-Z0-9-]{2,12}", aircraft) else None

    @staticmethod
    def _status(value: Any) -> str | None:
        status = str(value or "").strip().lower()
        return status if re.fullmatch(r"[a-z-]{2,40}", status) else None

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
