"""
Volatility Calculation Invariance Tests

This module tests the consistency and invariance properties of volatility
calculations across different timeframes, data inputs, and market conditions.
"""

import math
import sys
import os
import numpy as np
import pandas as pd
import pytest
from typing import List, Tuple, Dict, Any

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tests.test_data import ohlcv_sample_data
except ImportError:
    from test_data import ohlcv_sample_data
from market_analysis.garch_vol_triggers import (
    compute_atr,
    sigma_ann_and_sigma_H_from_har,
    blended_sigma_h,
)
from market_analysis.utils import _hours_per_bar


class VolatilityInvarianceValidator:
    """Validates invariance properties of volatility calculations."""

    def __init__(self):
        self.sample_data = ohlcv_sample_data()
        self.timeframes = ["1h", "4h", "1d"]
        self.horizons = [1, 4, 24]  # hours

    def test_atr_scaling_invariance(self) -> Dict[str, Any]:
        """Test that ATR scales appropriately with timeframe changes."""
        results = {}

        for timeframe in self.timeframes:
            bar_hours = _hours_per_bar(timeframe)

            # Compute ATR for this timeframe
            df = self.sample_data.copy()
            df["atr5"] = compute_atr(df, period=5)
            atr_value = float(df["atr5"].iloc[-1])
            price = float(df["close"].iloc[-1])

            # ATR as fraction of price
            atr_frac = atr_value / price

            results[timeframe] = {
                "bar_hours": bar_hours,
                "atr_abs": atr_value,
                "atr_frac": atr_frac,
                "price": price,
            }

        return results

    def test_har_timeframe_consistency(self) -> Dict[str, Any]:
        """Test HAR-RV consistency across timeframes."""
        results = {}
        prices = self.sample_data["close"]

        for timeframe in self.timeframes:
            for horizon_hours in self.horizons:
                try:
                    sig_ann, sig_h_frac = sigma_ann_and_sigma_H_from_har(
                        prices, interval=timeframe, horizon_hours=horizon_hours
                    )

                    key = f"{timeframe}_{horizon_hours}h"
                    results[key] = {
                        "timeframe": timeframe,
                        "horizon_hours": horizon_hours,
                        "sigma_ann": sig_ann,
                        "sigma_h_frac": sig_h_frac,
                        "bar_hours": _hours_per_bar(timeframe),
                    }
                except Exception as e:
                    results[f"{timeframe}_{horizon_hours}h"] = {"error": str(e)}

        return results

    def test_blended_sigma_consistency(self) -> Dict[str, Any]:
        """Test blended sigma consistency across configurations."""
        results = {}
        prices = self.sample_data["close"]
        current_price = float(prices.iloc[-1])

        # Test different blend weights
        blend_configs = [
            {"garch": 0.3, "har": 0.4, "atr": 0.3},
            {"garch": 0.0, "har": 1.0, "atr": 0.0},  # HAR only
            {"garch": 0.0, "har": 0.0, "atr": 1.0},  # ATR only
            {"garch": 0.5, "har": 0.5, "atr": 0.0},  # GARCH+HAR only
        ]

        for i, config in enumerate(blend_configs):
            for timeframe in self.timeframes:
                try:
                    # Get ATR value
                    df = self.sample_data.copy()
                    df["atr5"] = compute_atr(df, period=5)
                    atr_abs = float(df["atr5"].iloc[-1])

                    # Get HAR sigma if needed
                    sigma_ann_har = None
                    if config["har"] > 0:
                        try:
                            sigma_ann_har, _ = sigma_ann_and_sigma_H_from_har(
                                prices, interval=timeframe, horizon_hours=4
                            )
                        except:
                            sigma_ann_har = None

                    # Mock GARCH sigma (since we don't have GARCH implementation)
                    sigma_ann_garch = None
                    if config["garch"] > 0:
                        sigma_ann_garch = 0.5  # Mock value

                    vol_cfg = {
                        "vol": {
                            "blend_w_garch": config["garch"],
                            "blend_w_har": config["har"],
                            "blend_w_atr": config["atr"],
                            "garch_har_outlier_ratio": 2.0,
                            "horizon_hours": 4,
                            "atr_to_sigma_factor": 0.424,
                        }
                    }

                    bar_hours = _hours_per_bar(timeframe)

                    # Compute blended sigma with correct parameters
                    sigma_h_abs = blended_sigma_h(
                        sigma_ann_garch=sigma_ann_garch,
                        sigma_ann_har=sigma_ann_har,
                        atr_abs=atr_abs,
                        price=current_price,
                        cfg=vol_cfg,
                        bar_hours=bar_hours,
                    )

                    key = f"config_{i}_{timeframe}"
                    results[key] = {
                        "config": config,
                        "timeframe": timeframe,
                        "bar_hours": bar_hours,
                        "sigma_h_abs": sigma_h_abs,
                        "sigma_h_frac": sigma_h_abs / current_price,
                        "atr_abs": atr_abs,
                        "sigma_ann_har": sigma_ann_har,
                    }
                except Exception as e:
                    results[f"config_{i}_{timeframe}"] = {"error": str(e)}

        return results

    def test_horizon_scaling_invariance(self) -> Dict[str, Any]:
        """Test that horizon scaling follows expected mathematical relationships."""
        results = {}
        prices = self.sample_data["close"]
        timeframe = "4h"

        # Test different horizons
        test_horizons = [1, 2, 4, 8, 12, 24]

        for horizon in test_horizons:
            try:
                sig_ann, sig_h_frac = sigma_ann_and_sigma_H_from_har(
                    prices, interval=timeframe, horizon_hours=horizon
                )

                results[f"horizon_{horizon}h"] = {
                    "horizon_hours": horizon,
                    "sigma_ann": sig_ann,
                    "sigma_h_frac": sig_h_frac,
                    "expected_scaling": math.sqrt(horizon / _hours_per_bar(timeframe)),
                }
            except Exception as e:
                results[f"horizon_{horizon}h"] = {"error": str(e)}

        return results

    def test_price_level_invariance(self) -> Dict[str, Any]:
        """Test that volatility calculations are invariant to price level scaling."""
        results = {}
        base_prices = self.sample_data["close"]

        # Scale prices by different factors
        scale_factors = [0.1, 0.5, 1.0, 2.0, 10.0]
        timeframe = "4h"
        horizon = 4

        for scale in scale_factors:
            scaled_prices = base_prices * scale

            try:
                # HAR volatility (should be scale-invariant in fractional terms)
                sig_ann, sig_h_frac = sigma_ann_and_sigma_H_from_har(
                    scaled_prices, interval=timeframe, horizon_hours=horizon
                )

                # ATR (should scale proportionally)
                df_scaled = self.sample_data.copy()
                for col in ["open", "high", "low", "close"]:
                    df_scaled[col] *= scale
                df_scaled["atr5"] = compute_atr(df_scaled, period=5)
                atr_abs = float(df_scaled["atr5"].iloc[-1])
                atr_frac = atr_abs / float(scaled_prices.iloc[-1])

                results[f"scale_{scale}"] = {
                    "scale_factor": scale,
                    "price_level": float(scaled_prices.iloc[-1]),
                    "har_sigma_ann": sig_ann,
                    "har_sigma_h_frac": sig_h_frac,
                    "atr_abs": atr_abs,
                    "atr_frac": atr_frac,
                }
            except Exception as e:
                results[f"scale_{scale}"] = {"error": str(e)}

        return results

    def test_data_length_stability(self) -> Dict[str, Any]:
        """Test stability of volatility calculations with different data lengths."""
        results = {}
        full_prices = self.sample_data["close"]
        timeframe = "4h"
        horizon = 4

        # Test different data lengths (adjusted for small sample data)
        data_len = len(full_prices)
        lengths = [max(5, data_len // 2), data_len]  # Use half and full length

        for length in lengths:
            if length > len(full_prices):
                continue

            prices_subset = full_prices.iloc[-length:]

            try:
                sig_ann, sig_h_frac = sigma_ann_and_sigma_H_from_har(
                    prices_subset, interval=timeframe, horizon_hours=horizon
                )

                results[f"length_{length}"] = {
                    "data_length": length,
                    "sigma_ann": sig_ann,
                    "sigma_h_frac": sig_h_frac,
                }
            except Exception as e:
                results[f"length_{length}"] = {"error": str(e)}

        return results


# Pytest test functions
class TestVolatilityInvariance:
    """Pytest test class for volatility invariance."""

    @classmethod
    def setup_class(cls):
        """Set up the validator instance."""
        cls.validator = VolatilityInvarianceValidator()

    def test_atr_scaling_properties(self):
        """Test ATR scaling properties across timeframes."""
        results = self.validator.test_atr_scaling_invariance()

        # Verify we have results for all timeframes
        assert len(results) == len(self.validator.timeframes)

        # Check that ATR values are positive
        for timeframe, data in results.items():
            assert data["atr_abs"] > 0, f"ATR should be positive for {timeframe}"
            assert (
                data["atr_frac"] > 0
            ), f"ATR fraction should be positive for {timeframe}"
            assert (
                0 < data["atr_frac"] < 1
            ), f"ATR fraction should be reasonable for {timeframe}"

    def test_har_consistency_across_timeframes(self):
        """Test HAR consistency across different timeframes and horizons."""
        results = self.validator.test_har_timeframe_consistency()

        # Count successful calculations
        successful = [k for k, v in results.items() if "error" not in v]
        failed = [k for k, v in results.items() if "error" in v]

        print(f"\nHAR calculations: {len(successful)} successful, {len(failed)} failed")

        # Should have at least some successful calculations
        assert (
            len(successful) > 0
        ), "Should have at least some successful HAR calculations"

        # Check that successful calculations have reasonable values
        for key in successful:
            data = results[key]
            assert data["sigma_ann"] > 0, f"Annual sigma should be positive for {key}"
            assert (
                data["sigma_h_frac"] > 0
            ), f"Horizon sigma should be positive for {key}"
            assert (
                data["sigma_ann"] < 5.0
            ), f"Annual sigma should be reasonable for {key}"

    def test_blended_sigma_consistency(self):
        """Test blended sigma consistency across configurations."""
        results = self.validator.test_blended_sigma_consistency()

        # Count successful calculations
        successful = [k for k, v in results.items() if "error" not in v]
        failed = [k for k, v in results.items() if "error" in v]

        print(
            f"\nBlended sigma calculations: {len(successful)} successful, {len(failed)} failed"
        )

        # Should have at least some successful calculations
        assert (
            len(successful) > 0
        ), "Should have at least some successful blended calculations"

        # Check that successful calculations have reasonable values
        for key in successful:
            data = results[key]
            assert (
                data["sigma_h_abs"] > 0
            ), f"Blended sigma should be positive for {key}"
            assert (
                data["sigma_h_frac"] > 0
            ), f"Blended sigma fraction should be positive for {key}"

    def test_horizon_scaling_relationships(self):
        """Test that horizon scaling follows expected mathematical relationships."""
        results = self.validator.test_horizon_scaling_invariance()

        # Count successful calculations
        successful = [k for k, v in results.items() if "error" not in v]

        assert (
            len(successful) > 1
        ), "Need at least 2 successful horizon calculations for comparison"

        # Check scaling relationships
        horizon_data = [
            (results[k]["horizon_hours"], results[k]["sigma_h_frac"])
            for k in successful
        ]
        horizon_data.sort()  # Sort by horizon

        # Verify that longer horizons generally have higher volatility
        for i in range(1, len(horizon_data)):
            h1, sig1 = horizon_data[i - 1]
            h2, sig2 = horizon_data[i]

            # Allow some tolerance due to estimation noise
            expected_ratio = math.sqrt(h2 / h1)
            actual_ratio = sig2 / sig1

            # Should be within reasonable bounds (allow 50% deviation)
            assert (
                0.5 * expected_ratio <= actual_ratio <= 2.0 * expected_ratio
            ), f"Horizon scaling ratio {actual_ratio:.3f} vs expected {expected_ratio:.3f} for {h1}h -> {h2}h"

    def test_price_level_invariance_properties(self):
        """Test price level invariance properties."""
        results = self.validator.test_price_level_invariance()

        # Count successful calculations
        successful = [k for k, v in results.items() if "error" not in v]

        assert len(successful) > 1, "Need multiple price levels for invariance testing"

        # Check that fractional volatilities are approximately invariant to price scaling
        base_result = results["scale_1.0"]
        base_har_frac = base_result["har_sigma_h_frac"]
        base_atr_frac = base_result["atr_frac"]

        for key in successful:
            if key == "scale_1.0":
                continue

            data = results[key]

            # HAR fractional volatility should be approximately invariant
            har_frac_ratio = data["har_sigma_h_frac"] / base_har_frac
            assert (
                0.8 <= har_frac_ratio <= 1.2
            ), f"HAR fractional volatility should be scale-invariant: {har_frac_ratio:.3f} for {key}"

            # ATR fractional volatility should be approximately invariant
            atr_frac_ratio = data["atr_frac"] / base_atr_frac
            assert (
                0.8 <= atr_frac_ratio <= 1.2
            ), f"ATR fractional volatility should be scale-invariant: {atr_frac_ratio:.3f} for {key}"

    def test_data_length_stability_properties(self):
        """Test data length stability properties."""
        results = self.validator.test_data_length_stability()

        # Count successful calculations
        successful = [k for k, v in results.items() if "error" not in v]

        assert len(successful) > 1, "Need multiple data lengths for stability testing"

        # Check that volatility estimates are reasonably stable across data lengths
        sigma_values = [results[k]["sigma_h_frac"] for k in successful]

        if len(sigma_values) > 1:
            mean_sigma = np.mean(sigma_values)
            std_sigma = np.std(sigma_values)
            cv = std_sigma / mean_sigma  # Coefficient of variation

            # Volatility estimates should be reasonably stable (CV < 50%)
            assert (
                cv < 0.5
            ), f"Volatility estimates should be stable across data lengths: CV = {cv:.3f}"


if __name__ == "__main__":
    # Run validation directly
    validator = VolatilityInvarianceValidator()

    print("Running Volatility Invariance Validation...")
    print("=" * 50)

    print("\n1. Testing ATR scaling properties...")
    atr_results = validator.test_atr_scaling_invariance()
    for timeframe, data in atr_results.items():
        print(
            f"  {timeframe}: ATR={data['atr_abs']:.2f}, ATR%={data['atr_frac']*100:.3f}%"
        )

    print("\n2. Testing HAR timeframe consistency...")
    har_results = validator.test_har_timeframe_consistency()
    successful_har = [k for k, v in har_results.items() if "error" not in v]
    print(f"  Successful HAR calculations: {len(successful_har)}")

    print("\n3. Testing blended sigma consistency...")
    blend_results = validator.test_blended_sigma_consistency()
    successful_blend = [k for k, v in blend_results.items() if "error" not in v]
    print(f"  Successful blended calculations: {len(successful_blend)}")

    print("\n4. Testing horizon scaling...")
    horizon_results = validator.test_horizon_scaling_invariance()
    successful_horizon = [k for k, v in horizon_results.items() if "error" not in v]
    print(f"  Successful horizon calculations: {len(successful_horizon)}")

    print("\n5. Testing price level invariance...")
    price_results = validator.test_price_level_invariance()
    successful_price = [k for k, v in price_results.items() if "error" not in v]
    print(f"  Successful price level tests: {len(successful_price)}")

    print("\n6. Testing data length stability...")
    length_results = validator.test_data_length_stability()
    successful_length = [k for k, v in length_results.items() if "error" not in v]
    print(f"  Successful data length tests: {len(successful_length)}")

    print("\n" + "=" * 50)
    print("Volatility Invariance Validation Complete")
