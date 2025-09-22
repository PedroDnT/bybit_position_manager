"""
Calibration Validation Tests

This module validates that the calibrated parameters improve model performance
and maintain stability across different market conditions and timeframes.
"""

import math
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple

# Import path handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from tests.test_data import ohlcv_sample_data
except ImportError:
    from test_data import ohlcv_sample_data

from market_analysis.garch_vol_triggers import (
    blended_sigma_h, compute_atr, sigma_ann_and_sigma_H_from_har
)
from market_analysis.utils import _hours_per_bar


class CalibrationValidator:
    """Validates calibrated parameter performance."""

    def __init__(self):
        self.sample_data = ohlcv_sample_data()
        self.original_config = {
            "atr_to_sigma_factor": 0.424,
            "blend_w_garch": 0.3,
            "blend_w_har": 0.4,
            "blend_w_atr": 0.3,
            "garch_har_outlier_ratio": 2.0
        }
        self.calibrated_config = {
            "atr_to_sigma_factor": 0.8,
            "blend_w_garch": 0.5,
            "blend_w_har": 0.5,
            "blend_w_atr": 0.0,
            "garch_har_outlier_ratio": 3.0
        }

    def test_atr_factor_improvement(self) -> Dict[str, float]:
        """
        Test that the calibrated ATR factor provides better alignment
        with empirical volatility.
        """
        print("Testing ATR factor improvement...")
        
        prices = self.sample_data["close"]
        returns = prices.pct_change().dropna()
        empirical_vol = returns.std() * math.sqrt(252)
        
        # Calculate ATR
        df = self.sample_data.copy()
        df["atr5"] = compute_atr(df, period=5)
        atr_abs = float(df["atr5"].iloc[-1])
        current_price = float(prices.iloc[-1])
        atr_frac = atr_abs / current_price
        
        # Test both configurations
        results = {}
        for config_name, config in [("original", self.original_config), ("calibrated", self.calibrated_config)]:
            factor = config["atr_to_sigma_factor"]
            
            # Calculate implied volatility from ATR
            horizon_hours = 4
            bar_hours = 4
            scaling = math.sqrt(horizon_hours / bar_hours)
            atr_implied_vol = atr_frac * factor * scaling * math.sqrt(252 * 6)
            
            # Calculate alignment with empirical volatility
            alignment_error = abs(atr_implied_vol - empirical_vol) / empirical_vol
            
            results[f"{config_name}_atr_vol"] = atr_implied_vol
            results[f"{config_name}_alignment_error"] = alignment_error
        
        results["empirical_vol"] = empirical_vol
        results["improvement"] = (results["original_alignment_error"] - results["calibrated_alignment_error"]) / results["original_alignment_error"]
        
        return results

    def test_blend_weight_stability(self) -> Dict[str, float]:
        """
        Test that calibrated blend weights provide better stability
        across different timeframes.
        """
        print("Testing blend weight stability...")
        
        timeframes = ["1h", "4h", "1d"]
        horizon_hours = 4
        
        results = {}
        for config_name, config in [("original", self.original_config), ("calibrated", self.calibrated_config)]:
            sigma_values = []
            
            for timeframe in timeframes:
                try:
                    # Get current price and ATR
                    current_price = float(self.sample_data["close"].iloc[-1])
                    df = self.sample_data.copy()
                    df["atr5"] = compute_atr(df, period=5)
                    atr_abs = float(df["atr5"].iloc[-1])
                    
                    # Get HAR sigma
                    sigma_ann_har, _ = sigma_ann_and_sigma_H_from_har(
                        self.sample_data["close"], interval=timeframe, horizon_hours=horizon_hours
                    )
                    
                    # Mock GARCH sigma (use HAR as proxy)
                    sigma_ann_garch = sigma_ann_har * 1.1 if sigma_ann_har else None
                    
                    # Create config structure
                    cfg = {
                        "vol": {
                            "blend_w_garch": config["blend_w_garch"],
                            "blend_w_har": config["blend_w_har"],
                            "blend_w_atr": config["blend_w_atr"],
                            "atr_to_sigma_factor": config["atr_to_sigma_factor"],
                            "garch_har_outlier_ratio": config["garch_har_outlier_ratio"],
                            "horizon_hours": horizon_hours
                        }
                    }
                    
                    # Calculate blended sigma
                    sigma_h_abs = blended_sigma_h(
                        sigma_ann_garch=sigma_ann_garch,
                        sigma_ann_har=sigma_ann_har,
                        atr_abs=atr_abs,
                        price=current_price,
                        cfg=cfg,
                        bar_hours=_hours_per_bar(timeframe)
                    )
                    
                    if sigma_h_abs and sigma_h_abs > 0:
                        # Convert to fractional sigma for comparison
                        sigma_h_frac = sigma_h_abs / current_price
                        sigma_values.append(sigma_h_frac)
                
                except Exception as e:
                    print(f"Error in {timeframe}: {e}")
                    continue
            
            if len(sigma_values) >= 2:
                # Calculate coefficient of variation (stability metric)
                mean_sigma = np.mean(sigma_values)
                std_sigma = np.std(sigma_values)
                cv = std_sigma / mean_sigma if mean_sigma > 0 else float('inf')
                
                results[f"{config_name}_cv"] = cv
                results[f"{config_name}_mean_sigma"] = mean_sigma
                results[f"{config_name}_std_sigma"] = std_sigma
        
        if "original_cv" in results and "calibrated_cv" in results:
            results["stability_improvement"] = (results["original_cv"] - results["calibrated_cv"]) / results["original_cv"]
        
        return results

    def test_outlier_threshold_effectiveness(self) -> Dict[str, float]:
        """
        Test that the calibrated outlier threshold provides better
        detection of unrealistic volatility estimates.
        """
        print("Testing outlier threshold effectiveness...")
        
        # Create test scenarios with varying GARCH/HAR ratios
        test_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        base_har_sigma = 0.15
        
        results = {}
        for config_name, config in [("original", self.original_config), ("calibrated", self.calibrated_config)]:
            threshold = config["garch_har_outlier_ratio"]
            
            outliers_detected = 0
            valid_estimates = 0
            
            for ratio in test_ratios:
                garch_sigma = base_har_sigma * ratio
                
                # Simulate outlier detection logic
                if garch_sigma / base_har_sigma > threshold:
                    outliers_detected += 1
                else:
                    valid_estimates += 1
            
            results[f"{config_name}_outliers"] = outliers_detected
            results[f"{config_name}_valid"] = valid_estimates
            results[f"{config_name}_threshold"] = threshold
        
        return results

    def test_overall_performance_metrics(self) -> Dict[str, float]:
        """
        Test overall performance metrics comparing original vs calibrated parameters.
        """
        print("Testing overall performance metrics...")
        
        # Combine all test results
        atr_results = self.test_atr_factor_improvement()
        stability_results = self.test_blend_weight_stability()
        outlier_results = self.test_outlier_threshold_effectiveness()
        
        # Calculate composite performance score
        performance_metrics = {}
        
        # ATR alignment improvement (lower error is better)
        if "improvement" in atr_results:
            performance_metrics["atr_alignment_improvement"] = atr_results["improvement"]
        
        # Stability improvement (lower CV is better)
        if "stability_improvement" in stability_results:
            performance_metrics["stability_improvement"] = stability_results["stability_improvement"]
        
        # Outlier detection balance
        orig_outliers = outlier_results.get("original_outliers", 0)
        calib_outliers = outlier_results.get("calibrated_outliers", 0)
        
        # Ideal outlier detection should catch extreme cases but not be too sensitive
        # Target: detect ratios > 2.5 but allow ratios < 2.5
        ideal_outliers = len([r for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] if r > 2.5])
        
        orig_outlier_accuracy = 1.0 - abs(orig_outliers - ideal_outliers) / len([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        calib_outlier_accuracy = 1.0 - abs(calib_outliers - ideal_outliers) / len([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        
        performance_metrics["outlier_detection_improvement"] = calib_outlier_accuracy - orig_outlier_accuracy
        
        # Calculate overall improvement score
        improvements = [v for k, v in performance_metrics.items() if "improvement" in k and not math.isnan(v)]
        if improvements:
            performance_metrics["overall_improvement"] = np.mean(improvements)
        
        return performance_metrics

    def run_validation_suite(self) -> Dict[str, Dict]:
        """Run complete validation suite and return results."""
        print("Running calibration validation suite...")
        print("=" * 50)
        
        results = {
            "atr_factor": self.test_atr_factor_improvement(),
            "blend_stability": self.test_blend_weight_stability(),
            "outlier_threshold": self.test_outlier_threshold_effectiveness(),
            "overall_performance": self.test_overall_performance_metrics()
        }
        
        return results

    def generate_validation_report(self, results: Dict[str, Dict]) -> str:
        """Generate formatted validation report."""
        report = []
        report.append("CALIBRATION VALIDATION REPORT")
        report.append("=" * 40)
        report.append("")
        
        # ATR Factor Results
        atr_results = results["atr_factor"]
        report.append("ATR FACTOR CALIBRATION:")
        report.append("-" * 25)
        report.append(f"Empirical Volatility: {atr_results.get('empirical_vol', 0):.3f}")
        report.append(f"Original ATR Implied Vol: {atr_results.get('original_atr_vol', 0):.3f}")
        report.append(f"Calibrated ATR Implied Vol: {atr_results.get('calibrated_atr_vol', 0):.3f}")
        report.append(f"Original Alignment Error: {atr_results.get('original_alignment_error', 0):.1%}")
        report.append(f"Calibrated Alignment Error: {atr_results.get('calibrated_alignment_error', 0):.1%}")
        report.append(f"Improvement: {atr_results.get('improvement', 0):.1%}")
        report.append("")
        
        # Stability Results
        stability_results = results["blend_stability"]
        if "original_cv" in stability_results:
            report.append("BLEND WEIGHT STABILITY:")
            report.append("-" * 23)
            report.append(f"Original CV: {stability_results.get('original_cv', 0):.3f}")
            report.append(f"Calibrated CV: {stability_results.get('calibrated_cv', 0):.3f}")
            report.append(f"Stability Improvement: {stability_results.get('stability_improvement', 0):.1%}")
            report.append("")
        
        # Overall Performance
        overall = results["overall_performance"]
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 20)
        for metric, value in overall.items():
            if not math.isnan(value):
                if "improvement" in metric:
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        report.append("")
        
        return "\n".join(report)


def test_calibration_validation():
    """Pytest test function for calibration validation."""
    validator = CalibrationValidator()
    results = validator.run_validation_suite()
    
    # Assert improvements
    overall = results["overall_performance"]
    
    # Check that overall improvement is positive
    if "overall_improvement" in overall and not math.isnan(overall["overall_improvement"]):
        assert overall["overall_improvement"] > 0, f"Overall improvement should be positive, got {overall['overall_improvement']}"
    
    # Check ATR alignment improvement
    atr_results = results["atr_factor"]
    if "improvement" in atr_results and not math.isnan(atr_results["improvement"]):
        assert atr_results["improvement"] > 0, f"ATR alignment should improve, got {atr_results['improvement']}"
    
    print("✓ Calibration validation tests passed")


if __name__ == "__main__":
    validator = CalibrationValidator()
    results = validator.run_validation_suite()
    
    print("\n" + validator.generate_validation_report(results))
    
    # Run pytest assertion
    test_calibration_validation()
    
    print("Calibration validation completed successfully!")