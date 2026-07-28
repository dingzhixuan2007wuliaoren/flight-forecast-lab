import pytest

from flight_forecaster.route_info import (
    Airport,
    OurAirportsResolver,
    RouteLookupError,
    estimate_route,
    lookup_airport,
)


def test_training_route_uses_exact_profile() -> None:
    route = estimate_route("jfk", "lax")
    assert route.distance_km == 3983
    assert route.duration_minutes == 365
    assert route.source == "training_route"
    assert route.origin.iata == "JFK"
    assert route.destination.icao == "KLAX"
    assert route.origin_airport == route.origin


def test_coordinate_route_is_estimated_without_user_input() -> None:
    route = estimate_route("YYZ", "JFK")
    assert 500 < route.distance_km < 700
    assert 70 < route.duration_minutes < 120
    assert route.source == "airport_coordinates"
    assert route.origin.country == "CA"


def test_built_in_catalog_supports_intercontinental_major_airports() -> None:
    route = estimate_route("SYD", "LHR")
    assert 16_000 < route.distance_km < 18_000
    assert route.duration_minutes > 1_200
    assert route.source == "airport_coordinates"
    assert route.origin.name == "Sydney Kingsford Smith"
    assert route.destination.type == "large_airport"


def test_stops_increase_route_distance_and_duration() -> None:
    direct = estimate_route("JFK", "LAX")
    one_stop = estimate_route("JFK", "LAX", stops=1)
    assert one_stop.distance_km > direct.distance_km
    assert one_stop.duration_minutes == direct.duration_minutes + 75


def test_unknown_airport_is_rejected() -> None:
    with pytest.raises(RouteLookupError, match="airport not available"):
        estimate_route("ZZZ", "JFK")


def test_injected_resolver_extends_airport_coverage_without_network() -> None:
    resolved = Airport(
        iata="YQB",
        icao="CYQB",
        name="Quebec City Jean Lesage International",
        type="large_airport",
        country="CA",
        latitude=46.7911,
        longitude=-71.3933,
        source="ourairports",
    )

    route = estimate_route(
        "YQB",
        "JFK",
        resolver=lambda code: resolved if code == "YQB" else None,
    )

    assert route.source == "ourairports"
    assert route.origin == resolved
    assert route.distance_km > 500


def test_ourairports_resolver_loads_lazily_and_caches_csv() -> None:
    calls: list[tuple[str, float]] = []
    payload = (
        b"ident,type,name,latitude_deg,longitude_deg,iso_country,gps_code,iata_code\n"
        b"CYQB,large_airport,Quebec City Jean Lesage International,46.7911,-71.3933,"
        b"CA,CYQB,YQB\n"
        b"CLOSED,closed,Closed Field,0,0,CA,,ZZZ\n"
    )

    def downloader(url: str, timeout: float) -> bytes:
        calls.append((url, timeout))
        return payload

    resolver = OurAirportsResolver(timeout_seconds=1.0, downloader=downloader)
    assert resolver.loaded is False
    assert resolver("YQB") is not None
    assert resolver("ZZZ") is None
    assert resolver.loaded is True
    assert resolver.load_error is None
    assert len(calls) == 1


def test_ourairports_failure_is_safe_and_cached() -> None:
    calls = 0

    def downloader(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        raise TimeoutError("offline")

    resolver = OurAirportsResolver(downloader=downloader)
    assert resolver("YQB") is None
    assert resolver("YQB") is None
    assert resolver.loaded is True
    assert resolver.load_error == "TimeoutError: offline"
    assert calls == 1


def test_built_in_lookup_does_not_call_fallback_resolver() -> None:
    def unexpected_resolver(_code: str) -> Airport | None:
        raise AssertionError("built-in airport should not invoke fallback")

    assert lookup_airport("LHR", unexpected_resolver) is not None
