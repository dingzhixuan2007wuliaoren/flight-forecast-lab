from pathlib import Path

OFFER_PAGE = (
    Path(__file__).parents[1] / "src" / "flight_forecaster" / "static" / "offer.html"
)


def test_offer_curve_distinguishes_verified_fare_from_synthetic_model() -> None:
    page = OFFER_PAGE.read_text(encoding="utf-8")

    assert 'id="curve-live-badge"' in page
    assert 'id="curve-gap"' in page
    assert 'id="curve-gap-summary"' in page
    assert 'id="curve-gap-explanation"' in page
    assert "未校准合成模型" in page
    assert "Uncalibrated synthetic model" in page
    assert "未使用本次已验证报价校准" in page
    assert "not calibrated to this verified fare" in page
    assert "可能超出演示训练分布" in page
    assert "may be outside the demo training distribution" in page


def test_offer_curve_computes_gap_and_plots_comparable_live_fare() -> None:
    page = OFFER_PAGE.read_text(encoding="utf-8")

    assert "percentageGap=absoluteGap/liveFare" in page
    assert 'data-absolute-gap-usd' in page
    assert 'data-percentage-gap' in page
    assert "concat(comparableLiveFare?[liveFare]:[])" in page
    assert 'class:"curve-live-line"' in page
    assert 'class:"curve-live-label"' in page
    assert "No exchange-rate conversion is applied" in page
