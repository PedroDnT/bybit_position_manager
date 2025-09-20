import pytest
import numpy as np
from market_analysis.garch_vol_triggers import compute_atr, sl_tp_and_size
from tests.test_data import ohlcv_sample_data

def test_compute_atr():
    """
    Test the ATR calculation with a known data sample.
    The expected ATR value is calculated manually for verification.
    """
    df = ohlcv_sample_data()
    atr_period = 5 # Using a shorter period for a small sample
    df['atr'] = compute_atr(df, period=atr_period)

    # Expected values calculated manually/using a trusted library for this specific dataset
    # For this sample, the last ATR value should be around 146.8
    # This is a simple check, more rigorous checks could compare the whole series.
    assert 'atr' in df.columns
    assert not df['atr'].isnull().any()
    # A more precise check on the final value
    # Note: The exact expected value can be sensitive to the smoothing formula details.
    # Wilder's original formula uses RMA, which is equivalent to EWMA with alpha=1/N.
    # Let's verify the last value approximately.
    # TRs: 100, 90, 140, 130, 290, 150, 70, 70, 160, 80
    # Smoothed ATR (alpha=1/5=0.2):
    # 1: 100
    # 2: (100*4 + 90)/5 = 98
    # 3: (98*4 + 140)/5 = 106.4
    # ... after full calculation ...
    # The final value is approximately 118.15
    assert np.isclose(df['atr'].iloc[-1], 118.15, atol=0.1)


def test_sl_tp_and_size_long():
    """
    Test stop-loss, take-profit, and position size calculation for a long position.
    """
    params = sl_tp_and_size(
        entry_price=100.0,
        sigma_H=0.05,  # 5% volatility for the horizon
        k=2.0,  # SL multiplier
        m=3.0,  # TP multiplier
        side="long",
        R=50.0,  # Risk $50
        tick_size=0.01
    )

    # Expected SL distance: k * sigma_price = 2.0 * (100 * 0.05) = 10.0
    # Expected TP distance: m * sigma_price = 3.0 * (100 * 0.05) = 15.0
    assert np.isclose(params['SL_distance'], 10.0)
    assert np.isclose(params['TP_distance'], 15.0)

    # Expected SL price: 100 - 10 = 90
    # Expected TP price: 100 + 15 = 115
    assert np.isclose(params['SL'], 90.0)
    assert np.isclose(params['TP'], 115.0)

    # Expected Quantity: R / SL_distance = 50 / 10 = 5
    assert np.isclose(params['Q'], 5.0)

def test_sl_tp_and_size_short():
    """
    Test stop-loss, take-profit, and position size calculation for a short position.
    """
    params = sl_tp_and_size(
        entry_price=200.0,
        sigma_H=0.1,  # 10% volatility for the horizon
        k=1.5,  # SL multiplier
        m=2.5,  # TP multiplier
        side="short",
        R=150.0, # Risk $150
        tick_size=0.1
    )

    # Expected SL distance: k * sigma_price = 1.5 * (200 * 0.1) = 30.0
    # Expected TP distance: m * sigma_price = 2.5 * (200 * 0.1) = 50.0
    assert np.isclose(params['SL_distance'], 30.0)
    assert np.isclose(params['TP_distance'], 50.0)

    # Expected SL price: 200 + 30 = 230
    # Expected TP price: 200 - 50 = 150
    assert np.isclose(params['SL'], 230.0)
    assert np.isclose(params['TP'], 150.0)

    # Expected Quantity: R / SL_distance = 150 / 30 = 5
    assert np.isclose(params['Q'], 5.0)

def test_sl_tp_and_size_zero_vol():
    """
    Test edge case with zero volatility.
    The function should handle this gracefully by returning a position size of 0
    and distances of 0, not by raising an error.
    """
    params = sl_tp_and_size(
        entry_price=100.0,
        sigma_H=0.0,
        k=2.0,
        m=3.0,
        side="long",
        R=50.0
    )
    assert params['SL_distance'] == 0
    assert params['TP_distance'] == 0
    assert params['SL'] == 100.0
    assert params['TP'] == 100.0
    assert params['Q'] == 0.0 # Should not be infinite, should be 0

import math
from market_analysis.garch_vol_triggers import probability_hit_tp_before_sl, dynamic_levels_from_state
from .test_data import ohlcv_sample_data


def test_probability_hit_tp_before_sl_symmetric():
    price = 100.0
    sigma = 2.0
    # symmetric distances → ~0.5
    p_long = probability_hit_tp_before_sl(price=price, tp=price+sigma, sl=price-sigma, sigma_price=sigma, side='long')
    p_short = probability_hit_tp_before_sl(price=price, tp=price-sigma, sl=price+sigma, sigma_price=sigma, side='short')
    assert 0.45 <= p_long <= 0.55
    assert 0.45 <= p_short <= 0.55


def test_probability_hit_tp_before_sl_asymmetric_favors_tp():
    price = 100.0
    sigma = 2.0
    # TP closer than SL for long → prob > 0.5
    p = probability_hit_tp_before_sl(price=price, tp=price+1.0, sl=price-4.0, sigma_price=sigma, side='long')
    assert p > 0.5


essential_cfg = {
    'vol': {'horizon_hours': 4},
    'stops': {'atr_trail_mult_initial': 2.0, 'atr_trail_mult_late': 1.5, 'breakeven_after_R': 1.0},
    'prob': {'prob_alpha': 1.0, 'prob_target': 0.55, 'm_min': 2.0, 'm_max': 6.0, 'm_step': 0.5},
}


def test_dynamic_levels_from_state_long_tightens_after_breakeven():
    df = ohlcv_sample_data()
    entry = float(df['close'].iloc[2])
    current = entry + 50.0  # unrealized profit
    atr = 20.0
    sigma_H = 0.001  # 0.1% to ensure r_unreal >= breakeven
    out = dynamic_levels_from_state(current_price=current, entry_price=entry, side='long', sigma_H=sigma_H, atr=atr, base_k=2.0, base_m=3.0, cfg=essential_cfg)
    assert out['SL'] >= entry  # tightened to at least breakeven or trail
    assert out['TP'] > current
    assert 0.0 <= out['p_tp'] <= 1.0


def test_dynamic_levels_from_state_short_directionality():
    entry = 100.0
    current = 95.0
    atr = 2.0
    sigma_H = 0.02
    out = dynamic_levels_from_state(current_price=current, entry_price=entry, side='short', sigma_H=sigma_H, atr=atr, base_k=2.0, base_m=3.0, cfg=essential_cfg)
    assert out['TP'] < current  # TP should be below for shorts
    assert out['SL'] > current  # SL above for shorts
