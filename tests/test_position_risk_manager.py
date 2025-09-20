import pandas as pd
import pytest
from unittest.mock import Mock

from market_analysis import position_risk_manager as prm
from market_analysis.position_risk_manager import (
    PositionRiskManager,
    evaluate_liquidation_buffer,
)


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


@pytest.fixture
def configured_manager(monkeypatch):
    df = pd.DataFrame({
        "open": [100 + i * 0.1 for i in range(60)],
        "high": [101 + i * 0.1 for i in range(60)],
        "low": [99 + i * 0.1 for i in range(60)],
        "close": [100 + i * 0.1 for i in range(60)],
    })

    monkeypatch.setattr(prm, "get_bybit_market_info", lambda *args, **kwargs: {"tick_size": 0.5})
    monkeypatch.setattr(prm, "get_klines_bybit", lambda *args, **kwargs: df.copy())
    monkeypatch.setattr(prm, "compute_atr", lambda *_args, **_kwargs: pd.Series([1.0] * len(df)))
    monkeypatch.setattr(prm, "get_live_price_bybit", lambda *args, **kwargs: 100.0)

    monkeypatch.setattr(prm, "sigma_ann_and_sigma_H_from_har", lambda *args, **kwargs: (0.5, 0.02))

    def fake_garch(*_args, **_kwargs):
        return 0.5, 0.02, Mock()

    monkeypatch.setattr(prm, "garch_sigma_ann_and_sigma_H", fake_garch)
    monkeypatch.setattr(prm, "validate_garch_result", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(prm, "blended_sigma_h", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(prm, "calculate_confidence_score", lambda *_args, **_kwargs: (0, {}))
    monkeypatch.setattr(prm, "sl_tp_and_size", lambda **_kwargs: {
        'SL': 90.0,
        'TP': 120.0,
        'SL_distance': 10.0,
        'TP_distance': 20.0,
        'Q': 1.0,
    })

    def fake_dynamic_levels(**_kwargs):
        return {
            'SL': 90.0,
            'TP': 120.0,
            'p_tp': None,
            'reasons': [],
        }

    monkeypatch.setattr(prm, "dynamic_levels_from_state", fake_dynamic_levels)
    monkeypatch.setattr(prm, "compute_trailing_stop", lambda **_kwargs: 0.0)

    manager = PositionRiskManager.__new__(PositionRiskManager)
    manager.cfg = {
        "risk": {
            "base_target_pct": 0.025,
            "min_target_pct": 0.02,
            "max_target_pct": 0.03,
            "use_dynamic": False,
            "risk_deviation_warning_threshold": 1.2,
        },
        "stops": {
            "use_state_anchored": False,
        },
        "portfolio": {},
    }
    manager.exchange = Mock()
    manager.account_metrics = {
        "total_equity": 100000.0,
        "total_wallet_balance": 100000.0,
        "available_balance": 50000.0,
        "todays_realized_pnl": 0.0,
        "todays_total_pnl": 0.0,
    }
    manager.liquidation_buffer_multiple = 2.0
    manager.positions = []
    manager.risk_analysis = {}

    return manager


def base_position(size: float, pnl_pct: float = -1.0) -> dict:
    return {
        "symbol": "BTCUSDT",
        "side": "Long",
        "entryPrice": 100.0,
        "size": size,
        "notional": 1000.0,
        "leverage": 5,
        "unrealizedPnl": 0.0,
        "percentage": pnl_pct,
        "liquidationPrice": 50.0,
        "markPrice": 100.0,
    }


def test_risk_overage_triggers_warning(configured_manager):
    position = base_position(size=3.0)

    analysis = configured_manager.analyze_position_volatility(position)

    assert analysis["risk_deviation_ratio"] == pytest.approx(3.0)
    assert analysis["risk_size_overage_pct"] == pytest.approx(200.0)
    assert analysis["risk_size_overage_triggered"] is True
    assert analysis["position_health"] == "WARNING"
    assert analysis["action_required"] == "Trim position size to align with risk limits"


def test_risk_overage_within_bounds_keeps_normal_health(configured_manager):
    position = base_position(size=1.0)

    analysis = configured_manager.analyze_position_volatility(position)

    assert analysis["risk_deviation_ratio"] == pytest.approx(1.0)
    assert analysis["risk_size_overage_pct"] == 0.0
    assert analysis["risk_size_overage_triggered"] is False
    assert analysis["position_health"] == "NORMAL"
    assert analysis["action_required"] == "Set SL/TP as recommended"


def test_portfolio_throttle_scales_risk_when_cap_is_breached(configured_manager):
    manager = configured_manager
    manager.cfg["portfolio"] = {
        "max_portfolio_risk_frac": 0.04,
        "min_portfolio_risk_frac": 0.01,
        "drawdown_multipliers": [{"pnl_frac": -0.02, "multiplier": 0.5}],
    }
    manager.account_metrics["total_equity"] = 50_000.0
    manager.account_metrics["todays_realized_pnl"] = -3_000.0
    manager.account_metrics["todays_total_pnl"] = -3_000.0

    first = base_position(size=1.0)
    second = base_position(size=1.0)
    second["symbol"] = "ETHUSDT"
    manager.positions = [first, second]

    template_analysis = {
        "dollar_risk": 4_000.0,
        "dollar_reward": 8_000.0,
        "optimal_position_size": 2.0,
        "target_risk_dollars": 4_000.0,
    }

    manager.risk_analysis = {
        "BTCUSDT": template_analysis.copy(),
        "ETHUSDT": template_analysis.copy(),
    }

    manager._apply_portfolio_risk_throttle()

    scale = 0.125  # Risk budget reduced to $1,000 on $50k equity with 6% drawdown
    for symbol in ("BTCUSDT", "ETHUSDT"):
        analysis = manager.risk_analysis[symbol]
        assert analysis["dollar_risk"] == pytest.approx(500.0)
        assert analysis["dollar_reward"] == pytest.approx(1_000.0)
        assert analysis["target_risk_dollars"] == pytest.approx(500.0)
        assert analysis["optimal_position_size"] == pytest.approx(0.25)
        assert analysis["portfolio_throttle_scale"] == pytest.approx(scale)

    throttle_meta = manager.risk_analysis["portfolio_throttle"]
    assert throttle_meta["throttle_applied"] is True
    assert throttle_meta["risk_cap_dollars"] == pytest.approx(1_000.0)
    assert throttle_meta["drawdown_fraction"] == pytest.approx(-0.06)
    assert throttle_meta["post_scale_total_risk"] == pytest.approx(1_000.0)


def test_portfolio_throttle_leaves_risk_unchanged_when_within_budget(configured_manager):
    manager = configured_manager
    manager.cfg["portfolio"] = {
        "max_portfolio_risk_frac": 0.04,
        "min_portfolio_risk_frac": 0.01,
        "drawdown_multipliers": [{"pnl_frac": -0.02, "multiplier": 0.5}],
    }
    manager.account_metrics["total_equity"] = 50_000.0
    manager.account_metrics["todays_realized_pnl"] = 1_000.0
    manager.account_metrics["todays_total_pnl"] = 1_000.0

    first = base_position(size=1.0)
    second = base_position(size=1.0)
    second["symbol"] = "ETHUSDT"
    manager.positions = [first, second]

    manager.risk_analysis = {
        "BTCUSDT": {
            "dollar_risk": 400.0,
            "dollar_reward": 800.0,
            "optimal_position_size": 2.0,
            "target_risk_dollars": 400.0,
        },
        "ETHUSDT": {
            "dollar_risk": 400.0,
            "dollar_reward": 800.0,
            "optimal_position_size": 2.0,
            "target_risk_dollars": 400.0,
        },
    }

    manager._apply_portfolio_risk_throttle()

    for symbol in ("BTCUSDT", "ETHUSDT"):
        analysis = manager.risk_analysis[symbol]
        assert analysis["dollar_risk"] == pytest.approx(400.0)
        assert "portfolio_throttle_scale" not in analysis

    throttle_meta = manager.risk_analysis["portfolio_throttle"]
    assert throttle_meta["throttle_applied"] is False
    assert throttle_meta["risk_cap_dollars"] == pytest.approx(2_000.0)
    assert throttle_meta["scale"] == pytest.approx(1.0)
