import pandas as pd
import pytest
from unittest.mock import Mock

from market_analysis import position_risk_manager as prm
from market_analysis.position_risk_manager import PositionRiskManager


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
    }
    manager.exchange = Mock()

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
