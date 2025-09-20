import pytest

from market_analysis.position_risk_manager import evaluate_liquidation_buffer


def test_liquidation_buffer_safe_with_deep_cushion():
    """A wide liquidation cushion relative to the stop distance is marked safe."""
    result = evaluate_liquidation_buffer(
        stop_distance=10.0,
        liquidation_distance=30.0,
        threshold_ratio=2.0,
    )

    assert result["safe"] is True
    assert pytest.approx(result["ratio"]) == 3.0
    assert pytest.approx(result["required_distance"]) == 20.0
    assert pytest.approx(result["threshold_ratio"]) == 2.0


def test_liquidation_buffer_flags_risk_when_stop_widens():
    """Wider stops (e.g. from higher volatility) can erode the liquidation cushion."""
    baseline = evaluate_liquidation_buffer(
        stop_distance=10.0,
        liquidation_distance=30.0,
        threshold_ratio=2.0,
    )
    widened = evaluate_liquidation_buffer(
        stop_distance=20.0,
        liquidation_distance=30.0,
        threshold_ratio=2.0,
    )

    assert baseline["safe"] is True
    assert widened["safe"] is False
    assert pytest.approx(widened["ratio"]) == 1.5


def test_liquidation_buffer_tracks_volatility_driven_threshold():
    """Higher buffer multiples (e.g. during extreme volatility) tighten requirements."""
    relaxed = evaluate_liquidation_buffer(
        stop_distance=12.0,
        liquidation_distance=30.0,
        threshold_ratio=2.0,
    )
    strict = evaluate_liquidation_buffer(
        stop_distance=12.0,
        liquidation_distance=30.0,
        threshold_ratio=3.0,
    )

    assert relaxed["safe"] is True
    assert strict["safe"] is False
    assert pytest.approx(strict["threshold_ratio"]) == 3.0
