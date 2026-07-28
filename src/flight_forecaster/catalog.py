from __future__ import annotations

from dataclasses import asdict, dataclass

Cabin = str
PolicyStatus = str


@dataclass(frozen=True)
class LocalizedText:
    """A small bilingual value that is safe to return from the API."""

    zh: str
    en: str


@dataclass(frozen=True)
class StudentProgram:
    """Public student-program metadata, not proof of an available student fare."""

    status: str
    age_requirement: LocalizedText
    verification: LocalizedText
    official_url: str
    minimum_age: int
    maximum_age: int | None
    verification_steps: int
    actual_discount_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AirlineProfile:
    """Conservative metadata used to build deterministic comparison scenarios.

    Cabin entries describe scenarios supported by this application. They are not a
    promise that a cabin is sold on a particular flight. Fare-brand policies are left
    unknown because baggage, change, and refund rules normally vary by route and fare.
    """

    code: str
    name: str
    region: str
    supported_cabins: tuple[Cabin, ...]
    service_model: str
    checked_baggage_status: PolicyStatus = "unknown"
    free_change_status: PolicyStatus = "unknown"
    refund_status: PolicyStatus = "unknown"
    student_program: StudentProgram | None = None

    @property
    def cabins(self) -> tuple[Cabin, ...]:
        """Compatibility alias for consumers that use the shorter field name."""

        return self.supported_cabins

    @property
    def baggage_status(self) -> PolicyStatus:
        return self.checked_baggage_status

    @property
    def change_status(self) -> PolicyStatus:
        return self.free_change_status

    @property
    def student_status(self) -> PolicyStatus:
        return self.student_program.status if self.student_program else "unknown"

    @property
    def student_age_limit_zh(self) -> str:
        return self.student_program.age_requirement.zh if self.student_program else "未知"

    @property
    def student_age_limit_en(self) -> str:
        return self.student_program.age_requirement.en if self.student_program else "Unknown"

    @property
    def student_verification_zh(self) -> str:
        return self.student_program.verification.zh if self.student_program else "未知"

    @property
    def student_verification_en(self) -> str:
        return self.student_program.verification.en if self.student_program else "Unknown"

    @property
    def student_program_url(self) -> str | None:
        return self.student_program.official_url if self.student_program else None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ECONOMY = ("economy",)
ECONOMY_BUSINESS = ("economy", "business")
THREE_CABINS = ("economy", "premium_economy", "business")
FOUR_CABINS = ("economy", "premium_economy", "business", "first")


STUDENT_PROGRAMS: dict[str, StudentProgram] = {
    "EK": StudentProgram(
        status="program_available",
        age_requirement=LocalizedText(
            zh="公开规则通常面向 16–31 岁学生；市场、日期和航线条件可能不同。",
            en="Published eligibility is generally for students aged 16–31; market, date, "
            "and route conditions can differ.",
        ),
        verification=LocalizedText(
            zh="通常须在值机时出示有效学生证或学校录取证明；以当前官方条款为准。",
            en="A valid student ID or school acceptance document is normally required at "
            "check-in; current official terms control.",
        ),
        official_url=("https://www.emirates.com/us/english/special-offers/student-special-fares/"),
        minimum_age=16,
        maximum_age=31,
        verification_steps=1,
    ),
    "LH": StudentProgram(
        status="program_available",
        age_requirement=LocalizedText(
            zh="验证时须年满 16 岁；学生票仅适用于官方列出的出发市场和航线。",
            en="Students must be at least 16 at verification; fares are limited to the "
            "published origin markets and routes.",
        ),
        verification=LocalizedText(
            zh="须通过汉莎指定的学生身份验证流程；验证方式因市场而异。",
            en="Student status must be verified through Lufthansa's designated process, "
            "which varies by market.",
        ),
        official_url="https://www.lufthansa.com/us/en/local-page/student-fares",
        minimum_age=16,
        maximum_age=None,
        verification_steps=1,
    ),
    "QR": StudentProgram(
        status="program_available",
        age_requirement=LocalizedText(
            zh="Student Club 公布的会员年龄通常为 18–30 岁；以当前条款为准。",
            en="Published Student Club membership is generally for ages 18–30; current "
            "terms control.",
        ),
        verification=LocalizedText(
            zh="须加入 Student Club，并通过 SheerID 验证认可高校的在读身份。",
            en="Student Club enrollment and SheerID verification at an accredited college "
            "or university are required.",
        ),
        official_url="https://www.qatarairways.com/en/student-club.html",
        minimum_age=18,
        maximum_age=30,
        verification_steps=2,
    ),
    "SQ": StudentProgram(
        status="program_available",
        age_requirement=LocalizedText(
            zh="在线验证通常须年满 16 岁；未满 16 岁可凭证明联系新航，未公布统一年龄上限。",
            en="Online verification normally requires age 16 or older; younger students "
            "may contact the airline with proof, and no universal upper limit is published.",
        ),
        verification=LocalizedText(
            zh="须加入 KrisFlyer，并通过 SheerID 验证符合条件高校的在读身份。",
            en="KrisFlyer membership and SheerID verification at an eligible college or "
            "university are required.",
        ),
        official_url=(
            "https://www.singaporeair.com/en_UK/us/ppsclub-krisflyer/krisflyer/"
            "krisflyer-students/"
        ),
        minimum_age=16,
        maximum_age=None,
        verification_steps=2,
    ),
    "TK": StudentProgram(
        status="program_available",
        age_requirement=LocalizedText(
            zh="公开学生资格通常覆盖 12–34 岁；国内、国际及市场规则可能不同。",
            en="Published student eligibility generally covers ages 12–34; domestic, "
            "international, and market rules can differ.",
        ),
        verification=LocalizedText(
            zh="须加入 Miles&Smiles，并提交学生资料完成身份审核。",
            en="Miles&Smiles membership and approval of submitted student documentation "
            "are required.",
        ),
        official_url="https://www.turkishairlines.com/en-int/student/",
        minimum_age=12,
        maximum_age=34,
        verification_steps=2,
    ),
}


def _airline(
    code: str,
    name: str,
    region: str,
    cabins: tuple[Cabin, ...],
    service_model: str = "full_service",
) -> AirlineProfile:
    return AirlineProfile(
        code=code,
        name=name,
        region=region,
        supported_cabins=cabins,
        service_model=service_model,
        student_program=STUDENT_PROGRAMS.get(code),
    )


# The catalog intentionally contains major network and low-cost passenger carriers from
# every inhabited region. It is a comparison universe, not a claim of live inventory.
_AIRLINES = (
    # North America
    _airline("AA", "American Airlines", "North America", FOUR_CABINS),
    _airline("AC", "Air Canada", "North America", THREE_CABINS),
    _airline("AM", "Aeromexico", "North America", ECONOMY_BUSINESS),
    _airline("AS", "Alaska Airlines", "North America", ECONOMY_BUSINESS),
    _airline("B6", "JetBlue", "North America", ECONOMY_BUSINESS, "hybrid"),
    _airline("DL", "Delta Air Lines", "North America", FOUR_CABINS),
    _airline("F9", "Frontier Airlines", "North America", ECONOMY, "low_cost"),
    _airline("NK", "Spirit Airlines", "North America", ECONOMY, "low_cost"),
    _airline("UA", "United Airlines", "North America", FOUR_CABINS),
    _airline("WN", "Southwest Airlines", "North America", ECONOMY, "low_cost"),
    _airline("WS", "WestJet", "North America", THREE_CABINS, "hybrid"),
    # Latin America and the Caribbean
    _airline("AD", "Azul Brazilian Airlines", "Latin America", ECONOMY_BUSINESS, "hybrid"),
    _airline("AR", "Aerolineas Argentinas", "Latin America", ECONOMY_BUSINESS),
    _airline("AV", "Avianca", "Latin America", ECONOMY_BUSINESS, "hybrid"),
    _airline("CM", "Copa Airlines", "Latin America", ECONOMY_BUSINESS),
    _airline("G3", "GOL Linhas Aereas", "Latin America", ECONOMY, "low_cost"),
    _airline("LA", "LATAM Airlines", "Latin America", THREE_CABINS),
    # Europe
    _airline("AF", "Air France", "Europe", FOUR_CABINS),
    _airline("AY", "Finnair", "Europe", THREE_CABINS),
    _airline("BA", "British Airways", "Europe", FOUR_CABINS),
    _airline("EI", "Aer Lingus", "Europe", ECONOMY_BUSINESS),
    _airline("FR", "Ryanair", "Europe", ECONOMY, "low_cost"),
    _airline("IB", "Iberia", "Europe", THREE_CABINS),
    _airline("KL", "KLM", "Europe", THREE_CABINS),
    _airline("LH", "Lufthansa", "Europe", FOUR_CABINS),
    _airline("LO", "LOT Polish Airlines", "Europe", THREE_CABINS),
    _airline("LX", "SWISS", "Europe", FOUR_CABINS),
    _airline("OS", "Austrian Airlines", "Europe", THREE_CABINS),
    _airline("SK", "Scandinavian Airlines", "Europe", THREE_CABINS),
    _airline("TK", "Turkish Airlines", "Europe / West Asia", ECONOMY_BUSINESS),
    _airline("TP", "TAP Air Portugal", "Europe", THREE_CABINS),
    _airline("U2", "easyJet", "Europe", ECONOMY, "low_cost"),
    _airline("VS", "Virgin Atlantic", "Europe", THREE_CABINS),
    # Middle East and Africa
    _airline("AT", "Royal Air Maroc", "Africa", ECONOMY_BUSINESS),
    _airline("EK", "Emirates", "Middle East", FOUR_CABINS),
    _airline("ET", "Ethiopian Airlines", "Africa", ECONOMY_BUSINESS),
    _airline("EY", "Etihad Airways", "Middle East", FOUR_CABINS),
    _airline("KQ", "Kenya Airways", "Africa", ECONOMY_BUSINESS),
    _airline("MS", "Egyptair", "Africa", ECONOMY_BUSINESS),
    _airline("QR", "Qatar Airways", "Middle East", FOUR_CABINS),
    _airline("SA", "South African Airways", "Africa", ECONOMY_BUSINESS),
    _airline("SV", "Saudia", "Middle East", FOUR_CABINS),
    # Asia-Pacific
    _airline("6E", "IndiGo", "South Asia", ECONOMY, "low_cost"),
    _airline("AI", "Air India", "South Asia", FOUR_CABINS),
    _airline("BR", "EVA Air", "East Asia", THREE_CABINS),
    _airline("CA", "Air China", "East Asia", FOUR_CABINS),
    _airline("CX", "Cathay Pacific", "East Asia", FOUR_CABINS),
    _airline("CZ", "China Southern Airlines", "East Asia", FOUR_CABINS),
    _airline("GA", "Garuda Indonesia", "Southeast Asia", ECONOMY_BUSINESS),
    _airline("JL", "Japan Airlines", "East Asia", FOUR_CABINS),
    _airline("KE", "Korean Air", "East Asia", FOUR_CABINS),
    _airline("MH", "Malaysia Airlines", "Southeast Asia", ECONOMY_BUSINESS),
    _airline("MU", "China Eastern Airlines", "East Asia", FOUR_CABINS),
    _airline("NH", "ANA", "East Asia", FOUR_CABINS),
    _airline("NZ", "Air New Zealand", "Oceania", THREE_CABINS),
    _airline("PR", "Philippine Airlines", "Southeast Asia", ECONOMY_BUSINESS),
    _airline("QF", "Qantas", "Oceania", FOUR_CABINS),
    _airline("SQ", "Singapore Airlines", "Southeast Asia", FOUR_CABINS),
    _airline("TG", "Thai Airways", "Southeast Asia", ECONOMY_BUSINESS),
    _airline("VN", "Vietnam Airlines", "Southeast Asia", THREE_CABINS),
)

_AIRLINES_BY_CODE = {profile.code: profile for profile in _AIRLINES}
if len(_AIRLINES_BY_CODE) != len(_AIRLINES):  # pragma: no cover - import-time invariant
    raise RuntimeError("airline catalog contains duplicate IATA codes")


def comparison_airlines() -> tuple[AirlineProfile, ...]:
    """Return the stable comparison catalog in IATA-code order."""

    return tuple(sorted(_AIRLINES, key=lambda profile: profile.code))


def get_airline_profile(code: str) -> AirlineProfile | None:
    """Look up an airline by case-insensitive two-character IATA code."""

    return _AIRLINES_BY_CODE.get(code.strip().upper())


def airline_profile(code: str) -> AirlineProfile | None:
    """Compatibility alias for :func:`get_airline_profile`."""

    return get_airline_profile(code)
