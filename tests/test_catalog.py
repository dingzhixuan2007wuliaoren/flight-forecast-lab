from flight_forecaster.catalog import (
    STUDENT_PROGRAMS,
    airline_profile,
    comparison_airlines,
    get_airline_profile,
)


def test_comparison_catalog_is_global_unique_and_deterministic() -> None:
    first = comparison_airlines()
    second = comparison_airlines()
    codes = [profile.code for profile in first]

    assert first == second
    assert 45 <= len(first) <= 60
    assert codes == sorted(codes)
    assert len(codes) == len(set(codes))
    assert {profile.region for profile in first} >= {
        "North America",
        "Latin America",
        "Europe",
        "Africa",
        "Middle East",
        "East Asia",
        "Oceania",
    }


def test_profiles_have_supported_scenarios_and_conservative_policies() -> None:
    allowed_cabins = {"economy", "premium_economy", "business", "first"}
    allowed_service_models = {"full_service", "hybrid", "low_cost"}

    for profile in comparison_airlines():
        assert profile.supported_cabins
        assert set(profile.supported_cabins) <= allowed_cabins
        assert profile.service_model in allowed_service_models
        assert profile.checked_baggage_status == "unknown"
        assert profile.free_change_status == "unknown"
        assert profile.refund_status == "unknown"


def test_known_student_programs_are_bilingual_but_not_discount_verification() -> None:
    assert {"EK", "TK", "QR", "LH", "SQ"} <= STUDENT_PROGRAMS.keys()
    for code in ("EK", "TK", "QR", "LH", "SQ"):
        profile = get_airline_profile(code)
        assert profile is not None
        program = profile.student_program
        assert program is not None
        assert program.status == "program_available"
        assert program.age_requirement.zh
        assert program.age_requirement.en
        assert program.verification.zh
        assert program.verification.en
        assert program.official_url.startswith("https://")
        assert program.actual_discount_verified is False
        assert program.minimum_age >= 0
        assert program.maximum_age is None or program.maximum_age >= program.minimum_age
        assert program.verification_steps >= 1


def test_profile_lookup_is_case_insensitive_and_safe_for_unknown_code() -> None:
    profile = get_airline_profile(" ek ")
    assert profile is not None
    assert profile.name == "Emirates"
    assert airline_profile("EK") == profile
    assert get_airline_profile("ZZ") is None
    assert profile.student_status == "program_available"
    assert profile.student_program_url == STUDENT_PROGRAMS["EK"].official_url
    student_program = profile.to_dict()["student_program"]
    assert isinstance(student_program, dict)
    assert student_program["actual_discount_verified"] is False
