"""Quota-safe Google Maps transit directions through the existing SerpApi account.

The adapter is intentionally small and fail-closed.  It shares the exact SQLite
usage ledger used by flight and hotel searches, polls only an existing Search
Archive record while a search is queued, and allows at most one separately
reserved re-submission.  Raw provider payloads, API keys, archive URLs, and
opaque search identifiers are never persisted by this module.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any, Literal
from urllib import parse

from flight_forecaster.availability import (
    ACCOUNT_REQUEST_TIMEOUT_SECONDS,
    MAX_PROVIDER_RESPONSE_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    SEARCH_ARCHIVE_REQUEST_TIMEOUT_SECONDS,
    SERPAPI_ACCOUNT_URL,
    SERPAPI_DEFAULT_MONTHLY_LIMIT,
    SERPAPI_MAX_HOURLY_LIMIT,
    SERPAPI_MAX_MONTHLY_LIMIT,
    SERPAPI_SEARCH_ARCHIVE_URL,
    SERPAPI_SEARCH_URL,
    _UrllibClient,
    _UsageError,
    _UsageLedger,
)

SerpApiTransitStatus = Literal[
    "available",
    "no_results",
    "not_configured",
    "authentication_failed",
    "quota_exhausted",
    "rate_limited",
    "provider_processing",
    "provider_error",
    "provider_unavailable",
    "response_invalid",
    "quota_ledger_unavailable",
]
SerpApiTransitMode = Literal[
    "WALK",
    "TRAM",
    "SUBWAY",
    "FERRY",
    "SUBURBAN",
    "BUS",
    "COACH",
    "RAIL",
    "HIGHSPEED_RAIL",
    "LONG_DISTANCE",
    "NIGHT_RAIL",
    "REGIONAL_FAST_RAIL",
    "REGIONAL_RAIL",
    "CABLE_CAR",
    "FUNICULAR",
    "AERIAL_LIFT",
    "OTHER",
]

SERPAPI_TRANSIT_SOURCE_URL = "https://serpapi.com/google-maps-directions-api"
SERPAPI_TRANSIT_CACHE_TTL_SECONDS = 30 * 60
SERPAPI_TRANSIT_POLL_DELAYS_SECONDS = (0.5, 1.0, 2.0, 4.0, 5.0)
SERPAPI_TRANSIT_MAX_LEGS = 20
SERPAPI_TRANSIT_MAX_STOPS_PER_LEG = 100
SERPAPI_TRANSIT_MAX_DURATION_SECONDS = 7 * 24 * 60 * 60
_SEARCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class SerpApiTransitLeg:
    """One public-transport leg returned by Google Maps through SerpApi."""

    mode: SerpApiTransitMode
    from_name: str
    to_name: str
    duration_minutes: int
    distance_km: float | None
    line_name: str | None
    headsign: str | None
    agency_name: str | None
    intermediate_stops: tuple[str, ...]
    departure_label: str | None
    arrival_label: str | None


@dataclass(frozen=True, slots=True)
class SerpApiTransitResult:
    """Sanitized transit evidence or a classified, user-safe failure."""

    status: SerpApiTransitStatus
    observed_at: datetime
    message: str
    duration_minutes: int | None = None
    distance_km: float | None = None
    departure_at: datetime | None = None
    arrival_at: datetime | None = None
    departure_label: str | None = None
    arrival_label: str | None = None
    transfers: int | None = None
    legs: tuple[SerpApiTransitLeg, ...] = ()


@dataclass(frozen=True, slots=True)
class _AccountQuota:
    billing_cycle_key: str
    hour_bucket_key: str
    monthly_used: int
    hourly_used: int
    monthly_limit: int
    hourly_limit: int


class _TransitProviderError(RuntimeError):
    def __init__(self, status: SerpApiTransitStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class SerpApiTransitDirectionsProvider:
    """Optional fallback for regions missing from Transitous open timetables."""

    def __init__(
        self,
        api_key: str | None,
        *,
        usage_path: Path,
        monthly_limit: int | str | None = SERPAPI_DEFAULT_MONTHLY_LIMIT,
        client: Any = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        now_provider: Any = None,
        poll_delays_seconds: tuple[float, ...] = SERPAPI_TRANSIT_POLL_DELAYS_SECONDS,
        sleep_provider: Any = None,
    ) -> None:
        self._api_key = _safe_api_key(api_key)
        self._ledger = _UsageLedger(Path(usage_path))
        self._monthly_limit = _bounded_monthly_limit(monthly_limit)
        try:
            timeout = float(timeout_seconds)
            delays = tuple(float(value) for value in poll_delays_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("transit provider timing configuration is invalid") from exc
        if not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError("transit provider timeout is invalid")
        if (
            not delays
            or len(delays) > 5
            or any(not math.isfinite(value) or value < 0 or value > 5 for value in delays)
        ):
            raise ValueError("transit provider poll delays are invalid")
        self._client = client or _UrllibClient()
        self._timeout_seconds = timeout
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._poll_delays_seconds = delays
        self._sleep_provider = sleep_provider or sleep

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    def search(
        self,
        *,
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
        departure_at: datetime,
        country_code: str,
    ) -> SerpApiTransitResult:
        observed_at = self._now()
        try:
            origin = _coordinates(origin_latitude, origin_longitude)
            destination = _coordinates(destination_latitude, destination_longitude)
            requested = _aware_utc(departure_at)
            country = _country_code(country_code)
        except (TypeError, ValueError, OverflowError):
            return _failure(
                "response_invalid",
                observed_at,
                "Transit directions received invalid coordinates or departure time.",
            )
        if not self.configured:
            return _failure(
                "not_configured",
                observed_at,
                "The SerpApi Google Maps transit fallback is not configured.",
            )

        params: dict[str, Any] = {
            "engine": "google_maps_directions",
            "start_coords": origin,
            "end_coords": destination,
            "travel_mode": "3",
            "distance_unit": "0",
            "time": f"depart_at:{int(requested.timestamp())}",
            "hl": "en",
            "gl": country.lower(),
            "async": "true",
            "api_key": self._api_key,
        }
        try:
            account = self._account_quota()
            self._reserve_one(account)
            payload, received_at = self._request_json(
                SERPAPI_SEARCH_URL,
                params=params,
                account_request=False,
                allow_pending=True,
            )
            if _search_status(payload) in {"processing", "queued"}:
                payload, received_at = self._poll_pending_search(payload)
            if _search_status(payload) in {"processing", "queued"}:
                # One controlled retry only.  It gets its own quota reservation.
                account = self._account_quota()
                self._reserve_one(account)
                payload, received_at = self._request_json(
                    SERPAPI_SEARCH_URL,
                    params=params,
                    account_request=False,
                    allow_pending=True,
                )
                if _search_status(payload) in {"processing", "queued"}:
                    payload, received_at = self._poll_pending_search(payload)
                if _search_status(payload) in {"processing", "queued"}:
                    raise _TransitProviderError(
                        "provider_processing",
                        "Google Maps transit directions are still processing.",
                    )
            _validate_search_echo(
                payload,
                origin=origin,
                destination=destination,
                requested_timestamp=int(requested.timestamp()),
            )
            return _parse_transit_result(
                payload,
                requested_departure_at=requested,
                observed_at=received_at,
            )
        except _TransitProviderError as exc:
            return _failure(exc.status, self._now(), str(exc))

    def _account_quota(self) -> _AccountQuota:
        payload, received_at = self._request_json(
            SERPAPI_ACCOUNT_URL,
            params={"api_key": self._api_key},
            account_request=True,
        )
        if str(payload.get("account_status", "")).strip().lower() != "active":
            raise _TransitProviderError(
                "authentication_failed",
                "The SerpApi account is not active.",
            )
        monthly_used = _nonnegative_int(payload.get("this_month_usage"))
        hourly_used = _nonnegative_int(payload.get("this_hour_searches"))
        provider_monthly_limit = _positive_int(payload.get("searches_per_month"))
        provider_hourly_limit = _positive_int(payload.get("account_rate_limit_per_hour"))
        renewal = _renewal_date(payload.get("plan_renewal_date"), received_at)
        if None in (
            monthly_used,
            hourly_used,
            provider_monthly_limit,
            provider_hourly_limit,
            renewal,
        ):
            raise _TransitProviderError(
                "response_invalid",
                "The SerpApi account quota metadata is incomplete.",
            )
        assert isinstance(renewal, date)
        return _AccountQuota(
            billing_cycle_key=f"renewal:{renewal.isoformat()}",
            hour_bucket_key=received_at.strftime("%Y-%m-%dT%H"),
            monthly_used=monthly_used,
            hourly_used=hourly_used,
            monthly_limit=min(self._monthly_limit, provider_monthly_limit),
            hourly_limit=min(SERPAPI_MAX_HOURLY_LIMIT, provider_hourly_limit),
        )

    def _reserve_one(self, account: _AccountQuota) -> None:
        try:
            reservation = self._ledger.synchronize_and_reserve(
                account.billing_cycle_key,
                account.hour_bucket_key,
                calls=1,
                monthly_limit=account.monthly_limit,
                hourly_limit=account.hourly_limit,
                provider_monthly_usage=account.monthly_used,
                provider_hourly_usage=account.hourly_used,
                require_all=True,
            )
        except _UsageError as exc:
            raise _TransitProviderError(
                "quota_ledger_unavailable",
                "The shared SerpApi quota ledger is unavailable.",
            ) from exc
        if reservation.reserved_calls != 1:
            raise _TransitProviderError(
                "quota_exhausted",
                "The shared SerpApi quota is exhausted.",
            )

    def _poll_pending_search(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], datetime]:
        search_id = _payload_search_id(payload)
        if search_id is None:
            raise _TransitProviderError(
                "response_invalid",
                "The pending transit search did not contain a safe search ID.",
            )
        archive_url = SERPAPI_SEARCH_ARCHIVE_URL.format(search_id=search_id)
        if _archive_search_id(archive_url) != search_id:
            raise _TransitProviderError(
                "response_invalid",
                "The transit Search Archive URL is invalid.",
            )
        last_payload = payload
        last_observed = self._now()
        for delay_seconds in self._poll_delays_seconds:
            try:
                self._sleep_provider(delay_seconds)
            except Exception as exc:
                raise _TransitProviderError(
                    "provider_unavailable",
                    "The transit provider polling wait failed.",
                ) from exc
            archived, observed_at = self._request_json(
                archive_url,
                params={"api_key": self._api_key},
                account_request=False,
                allow_pending=True,
            )
            if _payload_search_id(archived) != search_id:
                raise _TransitProviderError(
                    "response_invalid",
                    "The transit Search Archive response did not match the request.",
                )
            last_payload = archived
            last_observed = observed_at
            if _search_status(archived) in {"success", "cached"}:
                return archived, observed_at
        return last_payload, last_observed

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        account_request: bool,
        allow_pending: bool = False,
    ) -> tuple[dict[str, Any], datetime]:
        archive_id = _archive_search_id(url)
        if account_request:
            allowed_url = url == SERPAPI_ACCOUNT_URL
        else:
            allowed_url = url == SERPAPI_SEARCH_URL or archive_id is not None
        if not allowed_url:
            raise _TransitProviderError(
                "response_invalid",
                "The transit provider URL is not allowlisted.",
            )
        try:
            response = self._client.get(
                url,
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "flight-forecast-lab/0.2.0 "
                        "(Google Maps transit fallback)"
                    ),
                },
                timeout=(
                    min(self._timeout_seconds, ACCOUNT_REQUEST_TIMEOUT_SECONDS)
                    if account_request
                    else min(
                        self._timeout_seconds,
                        SEARCH_ARCHIVE_REQUEST_TIMEOUT_SECONDS,
                    )
                    if archive_id is not None
                    else self._timeout_seconds
                ),
            )
        except Exception as exc:
            raise _TransitProviderError(
                "provider_unavailable",
                "The transit provider is temporarily unavailable.",
            ) from exc
        observed_at = self._now()
        status = _response_status(response)
        if status in {401, 403}:
            raise _TransitProviderError(
                "authentication_failed",
                "The transit provider rejected authentication.",
            )
        if status == 429:
            raise _TransitProviderError(
                "rate_limited",
                "The transit provider rate limit was reached.",
            )
        if status in {408, 425} or 500 <= status <= 599:
            raise _TransitProviderError(
                "provider_unavailable",
                "The transit provider is temporarily unavailable.",
            )
        if not 200 <= status <= 299:
            raise _TransitProviderError(
                "provider_error",
                "The transit provider rejected the request.",
            )
        payload = _response_json(response)
        if not isinstance(payload, dict):
            raise _TransitProviderError(
                "response_invalid",
                "The transit provider returned an invalid response.",
            )
        if not account_request:
            search_status = _search_status(payload)
            if search_status in {"processing", "queued"}:
                if allow_pending:
                    return payload, observed_at
                raise _TransitProviderError(
                    "provider_processing",
                    "The transit provider is still processing the request.",
                )
            if search_status not in {"success", "cached"}:
                status_code: SerpApiTransitStatus = (
                    "provider_error" if search_status == "error" else "response_invalid"
                )
                raise _TransitProviderError(
                    status_code,
                    "The transit provider search failed.",
                )
        return payload, observed_at

    def _now(self) -> datetime:
        try:
            return _aware_utc(self._now_provider())
        except (TypeError, ValueError, OverflowError) as exc:
            raise _TransitProviderError(
                "response_invalid",
                "The transit provider clock is invalid.",
            ) from exc


def serpapi_transit_provider_from_env(
    usage_path: Path,
) -> SerpApiTransitDirectionsProvider:
    """Build the optional transit fallback from the shared SerpApi settings."""

    return SerpApiTransitDirectionsProvider(
        os.getenv("SERPAPI_API_KEY"),
        usage_path=Path(usage_path),
        monthly_limit=os.getenv(
            "SERPAPI_MONTHLY_LIMIT",
            str(SERPAPI_DEFAULT_MONTHLY_LIMIT),
        ),
    )


def _failure(
    status: SerpApiTransitStatus,
    observed_at: datetime,
    message: str,
) -> SerpApiTransitResult:
    return SerpApiTransitResult(
        status=status,
        observed_at=_aware_utc(observed_at),
        message=_safe_text(message, 500) or "Transit directions are unavailable.",
    )


def _validate_search_echo(
    payload: dict[str, Any],
    *,
    origin: str,
    destination: str,
    requested_timestamp: int,
) -> None:
    parameters = payload.get("search_parameters")
    if not isinstance(parameters, dict):
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider did not confirm the search parameters.",
        )
    echoed_time = str(parameters.get("time", "")).strip()
    checks = (
        str(parameters.get("engine", "")).strip() == "google_maps_directions",
        str(parameters.get("start_coords", "")).strip() == origin,
        str(parameters.get("end_coords", "")).strip() == destination,
        str(parameters.get("travel_mode", "")).strip() == "3",
        echoed_time == f"depart_at:{requested_timestamp}",
    )
    if not all(checks):
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider returned mismatched search parameters.",
        )


def _parse_transit_result(
    payload: dict[str, Any],
    *,
    requested_departure_at: datetime,
    observed_at: datetime,
) -> SerpApiTransitResult:
    directions = payload.get("directions")
    if directions is None:
        return _failure(
            "no_results",
            observed_at,
            "Google Maps returned no public-transit itinerary for this request.",
        )
    if not isinstance(directions, list):
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider directions list is invalid.",
        )
    candidates: list[tuple[int, int, SerpApiTransitResult]] = []
    for raw in directions[:30]:
        parsed_result = _parse_direction(
            raw,
            requested_departure_at=requested_departure_at,
            observed_at=observed_at,
        )
        if parsed_result is None:
            continue
        candidates.append(
            (
                parsed_result.duration_minutes or 100_000,
                parsed_result.transfers or 0,
                parsed_result,
            )
        )
    if candidates:
        return min(candidates, key=lambda item: (item[0], item[1]))[2]
    if directions:
        raise _TransitProviderError(
            "response_invalid",
            "Google Maps returned directions but no complete transit itinerary.",
        )
    return _failure(
        "no_results",
        observed_at,
        "Google Maps returned no public-transit itinerary for this request.",
    )


def _parse_direction(
    value: Any,
    *,
    requested_departure_at: datetime,
    observed_at: datetime,
) -> SerpApiTransitResult | None:
    if not isinstance(value, dict):
        return None
    if str(value.get("travel_mode", "")).strip().casefold() != "transit":
        return None
    duration_seconds = _bounded_number(
        value.get("duration"),
        minimum=1,
        maximum=SERPAPI_TRANSIT_MAX_DURATION_SECONDS,
    )
    raw_trips = value.get("trips")
    if duration_seconds is None or not isinstance(raw_trips, list):
        return None
    legs = tuple(
        parsed
        for parsed in (
            _parse_transit_trip(raw) for raw in raw_trips[:SERPAPI_TRANSIT_MAX_LEGS * 3]
        )
        if parsed is not None
    )
    if not legs or len(legs) > SERPAPI_TRANSIT_MAX_LEGS:
        return None
    distance_meters = _bounded_number(value.get("distance"), minimum=0, maximum=10_000_000)
    departure_at = _unix_datetime(value.get("leave_around"))
    arrival_at = _unix_datetime(value.get("arrive_around"))
    if departure_at is not None and arrival_at is not None:
        if (
            arrival_at <= departure_at
            or departure_at < requested_departure_at.replace(second=0, microsecond=0)
            - timedelta(minutes=5)
            or arrival_at - departure_at > timedelta(days=7)
        ):
            departure_at = None
            arrival_at = None
    elif departure_at is not None or arrival_at is not None:
        departure_at = None
        arrival_at = None
    departure_label = _safe_text(value.get("start_time"), 80)
    arrival_label = _safe_text(value.get("end_time"), 80)
    if (
        departure_at is None
        and (departure_label is None or arrival_label is None)
    ):
        return None
    return SerpApiTransitResult(
        status="available",
        observed_at=_aware_utc(observed_at),
        message=(
            "Google Maps returned this transit itinerary through SerpApi. "
            "Displayed stop and clock labels remain provider-supplied."
        ),
        duration_minutes=max(1, math.ceil(duration_seconds / 60)),
        distance_km=(
            round(distance_meters / 1_000, 1)
            if distance_meters is not None
            else None
        ),
        departure_at=departure_at,
        arrival_at=arrival_at,
        departure_label=departure_label,
        arrival_label=arrival_label,
        transfers=max(0, len(legs) - 1),
        legs=legs,
    )


def _parse_transit_trip(value: Any) -> SerpApiTransitLeg | None:
    if not isinstance(value, dict):
        return None
    if str(value.get("travel_mode", "")).strip().casefold() != "transit":
        return None
    start = value.get("start_stop")
    end = value.get("end_stop")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    from_name = _safe_text(start.get("name"), 300)
    to_name = _safe_text(end.get("name"), 300)
    duration_seconds = _bounded_number(
        value.get("duration"),
        minimum=1,
        maximum=SERPAPI_TRANSIT_MAX_DURATION_SECONDS,
    )
    if from_name is None or to_name is None or duration_seconds is None:
        return None
    distance_meters = _bounded_number(value.get("distance"), minimum=0, maximum=10_000_000)
    service = value.get("service_run_by")
    agency_name = (
        _safe_text(service.get("name"), 300) if isinstance(service, dict) else None
    )
    line_name = _safe_text(value.get("title"), 300)
    departure_label = _safe_text(start.get("time"), 80)
    arrival_label = _safe_text(end.get("time"), 80)
    if departure_label is None or arrival_label is None:
        return None
    intermediate_stops: list[str] = []
    raw_stops = value.get("stops")
    if isinstance(raw_stops, list):
        for item in raw_stops[:SERPAPI_TRANSIT_MAX_STOPS_PER_LEG]:
            if not isinstance(item, dict):
                continue
            name = _safe_text(item.get("name"), 250)
            clock = _safe_text(item.get("time"), 80)
            if name is not None:
                intermediate_stops.append(f"{name} · {clock}" if clock else name)
    return SerpApiTransitLeg(
        mode=_transit_mode(value),
        from_name=from_name,
        to_name=to_name,
        duration_minutes=max(1, math.ceil(duration_seconds / 60)),
        distance_km=(
            round(distance_meters / 1_000, 1)
            if distance_meters is not None
            else None
        ),
        line_name=line_name,
        headsign=_safe_text(value.get("headsign"), 300),
        agency_name=agency_name,
        intermediate_stops=tuple(intermediate_stops),
        departure_label=departure_label,
        arrival_label=arrival_label,
    )


def _transit_mode(value: dict[str, Any]) -> SerpApiTransitMode:
    marker = " ".join(
        filter(
            None,
            (
                _safe_text(value.get("title"), 300),
                _safe_text(value.get("icon"), 500),
            ),
        )
    ).casefold()
    if "subway" in marker or "metro" in marker:
        return "SUBWAY"
    if "light_rail" in marker or "light rail" in marker or "tram" in marker:
        return "TRAM"
    if "highspeed" in marker or "high-speed" in marker:
        return "HIGHSPEED_RAIL"
    if "train" in marker or "rail" in marker:
        return "RAIL"
    if "ferry" in marker:
        return "FERRY"
    if "coach" in marker:
        return "COACH"
    if "bus" in marker:
        return "BUS"
    return "OTHER"


def _response_status(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", getattr(response, "status", 0)))
    except (TypeError, ValueError):
        return 0


def _response_json(response: Any) -> Any:
    try:
        content = getattr(response, "content", None)
        text = getattr(response, "text", None)
    except Exception as exc:
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider returned an unreadable response.",
        ) from exc
    if isinstance(content, bytes) and len(content) > MAX_PROVIDER_RESPONSE_BYTES:
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider response exceeded the safety limit.",
        )
    if isinstance(text, str) and len(text.encode("utf-8")) > MAX_PROVIDER_RESPONSE_BYTES:
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider response exceeded the safety limit.",
        )
    try:
        payload = response.json()
        size = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except Exception as exc:
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider returned invalid JSON.",
        ) from exc
    if size > MAX_PROVIDER_RESPONSE_BYTES:
        raise _TransitProviderError(
            "response_invalid",
            "The transit provider response exceeded the safety limit.",
        )
    return payload


def _search_status(payload: Any) -> str:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    return (
        str(metadata.get("status", "")).strip().lower()
        if isinstance(metadata, dict)
        else ""
    )


def _payload_search_id(payload: Any) -> str | None:
    metadata = payload.get("search_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("id")
    return value if isinstance(value, str) and _SEARCH_ID_PATTERN.fullmatch(value) else None


def _archive_search_id(url: Any) -> str | None:
    if not isinstance(url, str) or len(url) > 256:
        return None
    try:
        parts = parse.urlsplit(url)
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    if (
        parts.scheme != "https"
        or parts.hostname != "serpapi.com"
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or parts.query
        or parts.fragment
        or not parts.path.startswith("/searches/")
        or not parts.path.endswith(".json")
    ):
        return None
    search_id = parts.path.removeprefix("/searches/").removesuffix(".json")
    return search_id if _SEARCH_ID_PATTERN.fullmatch(search_id) else None


def _coordinates(latitude: Any, longitude: Any) -> str:
    lat = _bounded_number(latitude, minimum=-90, maximum=90)
    lon = _bounded_number(longitude, minimum=-180, maximum=180)
    if lat is None or lon is None:
        raise ValueError("coordinates are invalid")
    return f"{lat:.6f},{lon:.6f}"


def _country_code(value: Any) -> str:
    code = str(value or "").strip().upper()
    if re.fullmatch(r"[A-Z]{2}", code) is None:
        raise ValueError("country code is invalid")
    # Google localization uses ``uk`` even though airport catalogs use ISO ``GB``.
    return "UK" if code == "GB" else code


def _bounded_number(value: Any, *, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and minimum <= number <= maximum else None


def _safe_text(value: Any, max_length: int) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return None
    text = " ".join(str(value).split())
    return text if text and len(text) <= max_length and not _CONTROL_PATTERN.search(text) else None


def _safe_api_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip()
    return key if key and len(key) <= 512 and not _CONTROL_PATTERN.search(key) else None


def _bounded_monthly_limit(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = SERPAPI_DEFAULT_MONTHLY_LIMIT
    return min(max(parsed, 1), SERPAPI_MAX_MONTHLY_LIMIT)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _positive_int(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _renewal_date(value: Any, observed_at: datetime) -> date | None:
    text = _safe_text(value, 80)
    if text is None:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match is None:
        return None
    try:
        result = date.fromisoformat(match.group(0))
    except ValueError:
        return None
    if not 0 <= (result - observed_at.date()).days <= 62:
        return None
    return result


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _unix_datetime(value: Any) -> datetime | None:
    seconds = _bounded_number(value, minimum=0, maximum=32_503_680_000)
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None
