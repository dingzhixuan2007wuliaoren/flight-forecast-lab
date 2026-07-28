from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from io import StringIO
from math import asin, cos, radians, sin, sqrt
from threading import Lock
from urllib.request import Request, urlopen

from flight_forecaster.data import ROUTES


class RouteLookupError(ValueError):
    """Raised when an airport pair cannot be converted into model inputs."""


@dataclass(frozen=True)
class Airport:
    iata: str
    icao: str
    name: str
    type: str
    country: str
    latitude: float
    longitude: float
    source: str = "built_in"

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RouteEstimate:
    distance_km: float
    duration_minutes: int
    source: str
    origin: Airport
    destination: Airport

    @property
    def origin_airport(self) -> Airport:
        return self.origin

    @property
    def destination_airport(self) -> Airport:
        return self.destination

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _airport(
    iata: str,
    icao: str,
    name: str,
    country: str,
    latitude: float,
    longitude: float,
    airport_type: str = "large_airport",
) -> Airport:
    return Airport(iata, icao, name, airport_type, country, latitude, longitude)


# Representative international and domestic hubs on every inhabited continent. The
# fallback resolver below can extend this table without making ordinary predictions
# dependent on the network.
_BUILT_IN_AIRPORTS = (
    # United States and Canada
    _airport("ATL", "KATL", "Hartsfield-Jackson Atlanta International", "US", 33.6407, -84.4277),
    _airport("BOS", "KBOS", "Boston Logan International", "US", 42.3656, -71.0096),
    _airport("BWI", "KBWI", "Baltimore/Washington International", "US", 39.1774, -76.6684),
    _airport("CLT", "KCLT", "Charlotte Douglas International", "US", 35.2140, -80.9431),
    _airport("DCA", "KDCA", "Ronald Reagan Washington National", "US", 38.8512, -77.0402),
    _airport("DEN", "KDEN", "Denver International", "US", 39.8561, -104.6737),
    _airport("DFW", "KDFW", "Dallas Fort Worth International", "US", 32.8998, -97.0403),
    _airport("DTW", "KDTW", "Detroit Metropolitan Wayne County", "US", 42.2162, -83.3554),
    _airport("EWR", "KEWR", "Newark Liberty International", "US", 40.6895, -74.1745),
    _airport("FLL", "KFLL", "Fort Lauderdale-Hollywood International", "US", 26.0726, -80.1527),
    _airport("HNL", "PHNL", "Daniel K. Inouye International", "US", 21.3187, -157.9225),
    _airport("IAD", "KIAD", "Washington Dulles International", "US", 38.9531, -77.4565),
    _airport("IAH", "KIAH", "George Bush Intercontinental", "US", 29.9902, -95.3368),
    _airport("JFK", "KJFK", "John F. Kennedy International", "US", 40.6413, -73.7781),
    _airport("LAS", "KLAS", "Harry Reid International", "US", 36.0840, -115.1537),
    _airport("LAX", "KLAX", "Los Angeles International", "US", 33.9416, -118.4085),
    _airport("LGA", "KLGA", "LaGuardia", "US", 40.7769, -73.8740),
    _airport("MCO", "KMCO", "Orlando International", "US", 28.4312, -81.3081),
    _airport("MIA", "KMIA", "Miami International", "US", 25.7959, -80.2870),
    _airport("MSP", "KMSP", "Minneapolis-Saint Paul International", "US", 44.8848, -93.2223),
    _airport("ORD", "KORD", "Chicago O'Hare International", "US", 41.9742, -87.9073),
    _airport("PDX", "KPDX", "Portland International", "US", 45.5898, -122.5951),
    _airport("PHL", "KPHL", "Philadelphia International", "US", 39.8744, -75.2424),
    _airport("PHX", "KPHX", "Phoenix Sky Harbor International", "US", 33.4342, -112.0116),
    _airport("SAN", "KSAN", "San Diego International", "US", 32.7338, -117.1933),
    _airport("SEA", "KSEA", "Seattle-Tacoma International", "US", 47.4502, -122.3088),
    _airport("SFO", "KSFO", "San Francisco International", "US", 37.6213, -122.3790),
    _airport("SLC", "KSLC", "Salt Lake City International", "US", 40.7899, -111.9791),
    _airport("YYC", "CYYC", "Calgary International", "CA", 51.1215, -114.0076),
    _airport("YUL", "CYUL", "Montreal-Trudeau International", "CA", 45.4706, -73.7408),
    _airport("YVR", "CYVR", "Vancouver International", "CA", 49.1967, -123.1815),
    _airport("YYZ", "CYYZ", "Toronto Pearson International", "CA", 43.6777, -79.6248),
    # Mexico, Central America, and the Caribbean
    _airport("CUN", "MMUN", "Cancun International", "MX", 21.0365, -86.8771),
    _airport("MEX", "MMMX", "Mexico City International", "MX", 19.4361, -99.0719),
    _airport("PTY", "MPTO", "Tocumen International", "PA", 9.0714, -79.3835),
    _airport("SJO", "MROC", "Juan Santamaria International", "CR", 9.9939, -84.2088),
    _airport("SJU", "TJSJ", "Luis Munoz Marin International", "PR", 18.4394, -66.0018),
    # South America
    _airport("BOG", "SKBO", "El Dorado International", "CO", 4.7016, -74.1469),
    _airport("EZE", "SAEZ", "Ministro Pistarini International", "AR", -34.8222, -58.5358),
    _airport("GIG", "SBGL", "Rio de Janeiro-Galeao International", "BR", -22.8090, -43.2506),
    _airport("GRU", "SBGR", "Sao Paulo-Guarulhos International", "BR", -23.4356, -46.4731),
    _airport("LIM", "SPJC", "Jorge Chavez International", "PE", -12.0219, -77.1143),
    _airport("MDE", "SKRG", "Jose Maria Cordova International", "CO", 6.1645, -75.4231),
    _airport("SCL", "SCEL", "Arturo Merino Benitez International", "CL", -33.3930, -70.7858),
    _airport("UIO", "SEQM", "Mariscal Sucre International", "EC", -0.1292, -78.3575),
    # Europe
    _airport("AMS", "EHAM", "Amsterdam Airport Schiphol", "NL", 52.3105, 4.7683),
    _airport("ATH", "LGAV", "Athens International", "GR", 37.9364, 23.9445),
    _airport("BCN", "LEBL", "Barcelona-El Prat", "ES", 41.2974, 2.0833),
    _airport("BRU", "EBBR", "Brussels Airport", "BE", 50.9010, 4.4844),
    _airport("CDG", "LFPG", "Paris Charles de Gaulle", "FR", 49.0097, 2.5479),
    _airport("CPH", "EKCH", "Copenhagen Airport", "DK", 55.6180, 12.6508),
    _airport("DUB", "EIDW", "Dublin Airport", "IE", 53.4264, -6.2499),
    _airport("FCO", "LIRF", "Rome Fiumicino", "IT", 41.8003, 12.2389),
    _airport("FRA", "EDDF", "Frankfurt Airport", "DE", 50.0379, 8.5622),
    _airport("HEL", "EFHK", "Helsinki Airport", "FI", 60.3172, 24.9633),
    _airport("IST", "LTFM", "Istanbul Airport", "TR", 41.2753, 28.7519),
    _airport("LGW", "EGKK", "London Gatwick", "GB", 51.1537, -0.1821),
    _airport("LHR", "EGLL", "London Heathrow", "GB", 51.4700, -0.4543),
    _airport("LIS", "LPPT", "Humberto Delgado Airport", "PT", 38.7742, -9.1342),
    _airport("MAD", "LEMD", "Adolfo Suarez Madrid-Barajas", "ES", 40.4983, -3.5676),
    _airport("MXP", "LIMC", "Milan Malpensa", "IT", 45.6301, 8.7231),
    _airport("MUC", "EDDM", "Munich Airport", "DE", 48.3538, 11.7861),
    _airport("OSL", "ENGM", "Oslo Gardermoen", "NO", 60.1939, 11.1004),
    _airport("SAW", "LTFJ", "Istanbul Sabiha Gokcen", "TR", 40.8986, 29.3092),
    _airport("VIE", "LOWW", "Vienna International", "AT", 48.1103, 16.5697),
    _airport("WAW", "EPWA", "Warsaw Chopin", "PL", 52.1657, 20.9671),
    _airport("ZRH", "LSZH", "Zurich Airport", "CH", 47.4581, 8.5555),
    # Middle East
    _airport("AMM", "OJAI", "Queen Alia International", "JO", 31.7226, 35.9932),
    _airport("AUH", "OMAA", "Zayed International", "AE", 24.4330, 54.6511),
    _airport("BAH", "OBBI", "Bahrain International", "BH", 26.2708, 50.6336),
    _airport("DOH", "OTHH", "Hamad International", "QA", 25.2731, 51.6081),
    _airport("DXB", "OMDB", "Dubai International", "AE", 25.2532, 55.3657),
    _airport("JED", "OEJN", "King Abdulaziz International", "SA", 21.6702, 39.1525),
    _airport("KWI", "OKKK", "Kuwait International", "KW", 29.2266, 47.9689),
    _airport("MCT", "OOMS", "Muscat International", "OM", 23.5933, 58.2844),
    _airport("RUH", "OERK", "King Khalid International", "SA", 24.9576, 46.6988),
    _airport("TLV", "LLBG", "Ben Gurion International", "IL", 32.0114, 34.8867),
    # Africa
    _airport("ACC", "DGAA", "Kotoka International", "GH", 5.6052, -0.1668),
    _airport("ADD", "HAAB", "Addis Ababa Bole International", "ET", 8.9779, 38.7993),
    _airport("ALG", "DAAG", "Houari Boumediene Airport", "DZ", 36.6910, 3.2154),
    _airport("CAI", "HECA", "Cairo International", "EG", 30.1219, 31.4056),
    _airport("CMN", "GMMN", "Mohammed V International", "MA", 33.3675, -7.58997),
    _airport("CPT", "FACT", "Cape Town International", "ZA", -33.9696, 18.5972),
    _airport("DAR", "HTDA", "Julius Nyerere International", "TZ", -6.8781, 39.2026),
    _airport("JNB", "FAOR", "O. R. Tambo International", "ZA", -26.1337, 28.2420),
    _airport("LOS", "DNMM", "Murtala Muhammed International", "NG", 6.5774, 3.3212),
    _airport("NBO", "HKJK", "Jomo Kenyatta International", "KE", -1.3192, 36.9278),
    _airport("TUN", "DTTA", "Tunis-Carthage International", "TN", 36.8510, 10.2272),
    # East and Southeast Asia
    _airport("BKK", "VTBS", "Suvarnabhumi Airport", "TH", 13.6900, 100.7501),
    _airport("CAN", "ZGGG", "Guangzhou Baiyun International", "CN", 23.3924, 113.2990),
    _airport("CGK", "WIII", "Soekarno-Hatta International", "ID", -6.1256, 106.6559),
    _airport("DMK", "VTBD", "Don Mueang International", "TH", 13.9126, 100.6068),
    _airport("HAN", "VVNB", "Noi Bai International", "VN", 21.2212, 105.8072),
    _airport("HKG", "VHHH", "Hong Kong International", "HK", 22.3080, 113.9185),
    _airport("HND", "RJTT", "Tokyo Haneda", "JP", 35.5494, 139.7798),
    _airport("ICN", "RKSI", "Incheon International", "KR", 37.4602, 126.4407),
    _airport("KIX", "RJBB", "Kansai International", "JP", 34.4347, 135.2440),
    _airport("KUL", "WMKK", "Kuala Lumpur International", "MY", 2.7456, 101.7072),
    _airport("MNL", "RPLL", "Ninoy Aquino International", "PH", 14.5086, 121.0198),
    _airport("NRT", "RJAA", "Narita International", "JP", 35.7720, 140.3929),
    _airport("PEK", "ZBAA", "Beijing Capital International", "CN", 40.0799, 116.6031),
    _airport("PKX", "ZBAD", "Beijing Daxing International", "CN", 39.5098, 116.4105),
    _airport("PVG", "ZSPD", "Shanghai Pudong International", "CN", 31.1443, 121.8083),
    _airport("SGN", "VVTS", "Tan Son Nhat International", "VN", 10.8188, 106.6520),
    _airport("SIN", "WSSS", "Singapore Changi", "SG", 1.3644, 103.9915),
    _airport("SZX", "ZGSZ", "Shenzhen Bao'an International", "CN", 22.6393, 113.8107),
    _airport("TPE", "RCTP", "Taiwan Taoyuan International", "TW", 25.0797, 121.2342),
    # South Asia
    _airport("BLR", "VOBL", "Kempegowda International", "IN", 13.1986, 77.7066),
    _airport("BOM", "VABB", "Chhatrapati Shivaji Maharaj International", "IN", 19.0896, 72.8656),
    _airport("DEL", "VIDP", "Indira Gandhi International", "IN", 28.5562, 77.1000),
    _airport("HYD", "VOHS", "Rajiv Gandhi International", "IN", 17.2403, 78.4294),
    _airport("MAA", "VOMM", "Chennai International", "IN", 12.9941, 80.1709),
    # Oceania and the Pacific
    _airport("AKL", "NZAA", "Auckland Airport", "NZ", -37.0082, 174.7850),
    _airport("BNE", "YBBN", "Brisbane Airport", "AU", -27.3842, 153.1175),
    _airport("CHC", "NZCH", "Christchurch International", "NZ", -43.4894, 172.5322),
    _airport("MEL", "YMML", "Melbourne Airport", "AU", -37.6690, 144.8410),
    _airport("NAN", "NFFN", "Nadi International", "FJ", -17.7554, 177.4434),
    _airport("PER", "YPPH", "Perth Airport", "AU", -31.9403, 115.9672),
    _airport("SYD", "YSSY", "Sydney Kingsford Smith", "AU", -33.9399, 151.1753),
)

AIRPORTS: dict[str, Airport] = {airport.iata: airport for airport in _BUILT_IN_AIRPORTS}
if len(AIRPORTS) != len(_BUILT_IN_AIRPORTS):  # pragma: no cover - import-time invariant
    raise RuntimeError("built-in airport catalog contains duplicate IATA codes")

# Backwards-compatible coordinate mapping used by older consumers and notebooks.
AIRPORT_COORDINATES: dict[str, tuple[float, float]] = {
    code: airport.coordinates for code, airport in AIRPORTS.items()
}

EXACT_ROUTE_PROFILES = {
    (origin, destination): (distance_km, duration_minutes)
    for origin, destination, distance_km, duration_minutes, _ in ROUTES
}

OURAIRPORTS_CSV_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
Downloader = Callable[[str, float], bytes]
AirportResolver = Callable[[str], Airport | None]
_IATA_PATTERN = re.compile(r"^[A-Z]{3}$")


def _download_ourairports(url: str, timeout_seconds: float) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "flight-forecast-lab/0.1 (OurAirports fallback)"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = response.read(30_000_001)
    if len(payload) > 30_000_000:
        raise ValueError("OurAirports response exceeds the 30 MB safety limit")
    return payload


class OurAirportsResolver:
    """Lazy, cacheable resolver for the public OurAirports CSV.

    Loading errors are contained and cached as an empty result. Applications opt in by
    passing an instance (or its ``resolve`` method) to :func:`estimate_route`; tests can
    inject an in-memory downloader and never contact the network.
    """

    def __init__(
        self,
        *,
        url: str = OURAIRPORTS_CSV_URL,
        timeout_seconds: float = 2.5,
        downloader: Downloader = _download_ourairports,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 10:
            raise ValueError("timeout_seconds must be greater than 0 and at most 10")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._downloader = downloader
        self._airports: dict[str, Airport] = {}
        self._loaded = False
        self._load_error: str | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def __call__(self, code: str) -> Airport | None:
        return self.resolve(code)

    def resolve(self, code: str) -> Airport | None:
        normalized = code.strip().upper()
        if not _IATA_PATTERN.fullmatch(normalized):
            return None
        self._ensure_loaded()
        return self._airports.get(normalized)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                payload = self._downloader(self.url, self.timeout_seconds)
                self._airports = self._parse(payload)
            except (OSError, TimeoutError, UnicodeError, csv.Error, ValueError) as exc:
                self._airports = {}
                self._load_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._loaded = True

    @staticmethod
    def _parse(payload: bytes) -> dict[str, Airport]:
        text = payload.decode("utf-8-sig")
        rows = csv.DictReader(StringIO(text))
        required = {
            "ident",
            "type",
            "name",
            "latitude_deg",
            "longitude_deg",
            "iso_country",
            "gps_code",
            "iata_code",
        }
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError("OurAirports CSV is missing required columns")

        airports: dict[str, Airport] = {}
        for row in rows:
            code = row["iata_code"].strip().upper()
            if not _IATA_PATTERN.fullmatch(code) or row["type"] == "closed":
                continue
            try:
                latitude = float(row["latitude_deg"])
                longitude = float(row["longitude_deg"])
            except (TypeError, ValueError):
                continue
            icao = row["gps_code"].strip().upper()
            if not icao:
                ident = row["ident"].strip().upper()
                icao = ident if len(ident) == 4 else ""
            airports[code] = Airport(
                iata=code,
                icao=icao,
                name=row["name"].strip() or code,
                type=row["type"].strip() or "airport",
                country=row["iso_country"].strip().upper(),
                latitude=latitude,
                longitude=longitude,
                source="ourairports",
            )
        return airports


def lookup_airport(code: str, resolver: AirportResolver | None = None) -> Airport | None:
    """Resolve a valid IATA code from built-ins and then an optional fallback."""

    normalized = code.strip().upper()
    if not _IATA_PATTERN.fullmatch(normalized):
        return None
    built_in = AIRPORTS.get(normalized)
    if built_in is not None or resolver is None:
        return built_in
    candidate = resolver(normalized)
    if candidate is None or candidate.iata.strip().upper() != normalized:
        return None
    return candidate


def _great_circle_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_latitude, first_longitude = map(radians, first)
    second_latitude, second_longitude = map(radians, second)
    latitude_delta = second_latitude - first_latitude
    longitude_delta = second_longitude - first_longitude
    haversine = (
        sin(latitude_delta / 2) ** 2
        + cos(first_latitude) * cos(second_latitude) * sin(longitude_delta / 2) ** 2
    )
    return 6_371.0088 * 2 * asin(sqrt(haversine))


def estimate_route(
    origin: str,
    destination: str,
    stops: int = 0,
    *,
    resolver: AirportResolver | None = None,
) -> RouteEstimate:
    origin_code = origin.strip().upper()
    destination_code = destination.strip().upper()
    if stops < 0:
        raise RouteLookupError("经停次数 / stops cannot be negative")

    origin_airport = lookup_airport(origin_code, resolver)
    destination_airport = lookup_airport(destination_code, resolver)
    missing = [
        code
        for code, airport in (
            (origin_code, origin_airport),
            (destination_code, destination_airport),
        )
        if airport is None
    ]
    if missing:
        raise RouteLookupError(
            "暂不支持机场 / airport not available: "
            f"{', '.join(missing)}. Configure the optional OurAirports resolver for "
            "additional valid IATA airports."
        )

    # The None case is excluded by the error above and these assertions aid type checkers.
    assert origin_airport is not None
    assert destination_airport is not None
    route = (origin_code, destination_code)
    if route in EXACT_ROUTE_PROFILES:
        direct_distance, direct_duration = EXACT_ROUTE_PROFILES[route]
        source = "training_route"
    else:
        direct_distance = (
            _great_circle_km(origin_airport.coordinates, destination_airport.coordinates) * 1.03
        )
        direct_duration = 45 + direct_distance / 800 * 60
        source = (
            "ourairports"
            if "ourairports" in {origin_airport.source, destination_airport.source}
            else "airport_coordinates"
        )

    distance = direct_distance * (1 + 0.08 * stops)
    duration = direct_duration + 75 * stops
    return RouteEstimate(
        distance_km=round(float(distance), 1),
        duration_minutes=round(float(duration)),
        source=source,
        origin=origin_airport,
        destination=destination_airport,
    )
