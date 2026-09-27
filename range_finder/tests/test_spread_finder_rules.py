"""Pure Spread Finder rules lifted out of the tab's render function, where
no test could reach them (that is how the VIX breaker stayed dead)."""
import pytest

from range_finder.spread_finder_rules import check_reference, detect_regime_shift


@pytest.mark.parametrize("live,trailing,severity", [
    (22.0, 15.0, None),              # 1.47× — below the 1.5× trip
    (22.5, 15.0, "elevated"),        # exactly 1.5×
    (30.0, 15.0, "extreme"),         # 2.0×
])
def test_regime_shift_thresholds(live, trailing, severity):
    shift = detect_regime_shift(live, trailing)
    if severity is None:
        assert shift is None
    else:
        assert shift.severity == severity
        assert shift.ratio == pytest.approx(live / trailing)


@pytest.mark.parametrize("live,trailing", [
    (None, 15.0), (30.0, None), (0.0, 15.0), (30.0, 0.0), (30.0, float("nan")),
])
def test_no_live_or_trailing_vix_means_no_judgement(live, trailing):
    assert detect_regime_shift(live, trailing) is None


@pytest.mark.parametrize("ref,status", [
    (7400.0, "ok"),
    (7400.0 * 1.13, "warn"),          # >12%: extreme but conceivable
    (7400.0 * 1.30, "reset"),         # >25%: corruption, not a market move
    (100.0, "reset"),                 # the observed SPX-ref-of-100 bug
    (None, "reset"), ("junk", "reset"), (-5.0, "reset"),
])
def test_reference_check(ref, status):
    assert check_reference(ref, spot=7400.0).status == status


def test_reference_check_without_spot_cannot_judge():
    assert check_reference(100.0, spot=0.0).status == "ok"
    assert check_reference(100.0, spot=None).status == "ok"
