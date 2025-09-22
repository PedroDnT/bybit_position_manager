"""
Calibration Analysis Tool

This module analyzes test results and provides calibration recommendations
for volatility model parameters to ensure optimal performance and adherence
to financial modeling standards.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass

from tests.test_volatility_invariance import VolatilityInvarianceValidator
from market_analysis.garch_vol_triggers import compute_atr, sigma_ann_and_sigma_H_from_har
from market_analysis.utils import _hours_per_bar


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    parameter_name: str
    current_value: float
    recommended_value: float
    improvement_metric: float
    confidence_level: str
    rationale: str


class VolatilityCalibrator:
    """Calibrates volatility model parameters based on empirical test data."""

    def __init__(self):
        self.validator = VolatilityInvarianceValidator()
        self.calibration_results = []

    def analyze_atr_to_sigma_factor(self) -> CalibrationResult:
        """
        Analyze and calibrate the ATR-to-sigma conversion factor.
        
        The theoretical relationship between ATR and volatility should account for:
        - ATR measures average true range over N periods
        - Volatility (sigma) measures standard deviation of returns
        - Empirical factor typically ranges from 0.3 to 0.6
        """
        print("Analyzing ATR-to-sigma conversion factor...")
        
        prices = self.validator.sample_data["close"]
        returns = prices.pct_change().dropna()
        
        # Calculate empirical volatility from returns
        empirical_vol_daily = returns.std() * math.sqrt(252)  # Annualized
        empirical_vol_4h = returns.std() * math.sqrt(252 * 6)  # 4h periods per day
        
        # Calculate ATR
        df = self.validator.sample_data.copy()
        df["atr5"] = compute_atr(df, period=5)
        atr_abs = float(df["atr5"].iloc[-1])
        current_price = float(prices.iloc[-1])
        atr_frac = atr_abs / current_price
        
        # Current factor being used
        current_factor = 0.424
        
        # Calculate what factor would align ATR with empirical volatility
        # For 4h timeframe: ATR_frac * factor * sqrt(horizon/bar_hours) ≈ empirical_vol_horizon
        horizon_hours = 4
        bar_hours = 4
        scaling = math.sqrt(horizon_hours / bar_hours)
        
        # Target: atr_frac * factor * scaling = empirical_vol_4h / sqrt(252*6) * sqrt(horizon_hours)
        target_horizon_vol = empirical_vol_4h / math.sqrt(252 * 6) * math.sqrt(horizon_hours)
        recommended_factor = target_horizon_vol / (atr_frac * scaling)
        
        # Ensure factor is within reasonable bounds
        recommended_factor = max(0.2, min(0.8, recommended_factor))
        
        improvement = abs(recommended_factor - current_factor) / current_factor
        confidence = "high" if improvement > 0.1 else "medium" if improvement > 0.05 else "low"
        
        return CalibrationResult(
            parameter_name="atr_to_sigma_factor",
            current_value=current_factor,
            recommended_value=round(recommended_factor, 3),
            improvement_metric=improvement,
            confidence_level=confidence,
            rationale=f"Empirical analysis suggests factor of {recommended_factor:.3f} "
                     f"better aligns ATR with observed volatility patterns. "
                     f"Current ATR%: {atr_frac*100:.3f}%, Empirical vol: {empirical_vol_daily:.3f}"
        )

    def analyze_blend_weights(self) -> List[CalibrationResult]:
        """
        Analyze and optimize blending weights for volatility models.
        
        Evaluates different weight combinations and recommends optimal allocation
        based on stability, accuracy, and robustness metrics.
        """
        print("Analyzing blend weight optimization...")
        
        results = []
        blend_test_results = self.validator.test_blended_sigma_consistency()
        
        # Extract successful calculations
        successful = {k: v for k, v in blend_test_results.items() if "error" not in v}
        
        if len(successful) == 0:
            return []
        
        # Analyze performance by configuration
        config_performance = {}
        for key, data in successful.items():
            config_id = key.split("_")[1]  # Extract config number
            if config_id not in config_performance:
                config_performance[config_id] = []
            
            # Calculate stability metrics
            sigma_frac = data["sigma_h_frac"]
            config_performance[config_id].append({
                "sigma_frac": sigma_frac,
                "timeframe": data["timeframe"],
                "config": data["config"]
            })
        
        # Evaluate each configuration
        best_config = None
        best_score = float('inf')
        
        for config_id, results_list in config_performance.items():
            if len(results_list) < 2:  # Need multiple timeframes for comparison
                continue
                
            # Calculate coefficient of variation (stability metric)
            sigma_values = [r["sigma_frac"] for r in results_list]
            mean_sigma = np.mean(sigma_values)
            std_sigma = np.std(sigma_values)
            cv = std_sigma / mean_sigma if mean_sigma > 0 else float('inf')
            
            # Lower CV is better (more stable across timeframes)
            if cv < best_score:
                best_score = cv
                best_config = results_list[0]["config"]
        
        if best_config:
            # Current default weights
            current_weights = {"garch": 0.3, "har": 0.4, "atr": 0.3}
            
            # Calculate improvement
            weight_diff = sum(abs(best_config[k] - current_weights[k]) for k in current_weights)
            improvement = weight_diff / 3  # Average difference per weight
            
            confidence = "high" if improvement > 0.2 else "medium" if improvement > 0.1 else "low"
            
            for component in ["garch", "har", "atr"]:
                results.append(CalibrationResult(
                    parameter_name=f"blend_w_{component}",
                    current_value=current_weights[component],
                    recommended_value=best_config[component],
                    improvement_metric=improvement,
                    confidence_level=confidence,
                    rationale=f"Optimized {component} weight based on cross-timeframe stability analysis. "
                             f"Best configuration shows CV of {best_score:.3f}"
                ))
        
        return results

    def analyze_outlier_thresholds(self) -> CalibrationResult:
        """
        Analyze and calibrate outlier detection thresholds.
        
        Examines the GARCH vs HAR outlier ratio threshold to ensure
        appropriate detection of unrealistic volatility estimates.
        """
        print("Analyzing outlier detection thresholds...")
        
        # Test different outlier ratios
        test_ratios = [1.5, 2.0, 2.5, 3.0]
        prices = self.validator.sample_data["close"]
        
        # Get HAR estimates for comparison
        har_estimates = []
        for timeframe in ["1h", "4h", "1d"]:
            try:
                sigma_ann, _ = sigma_ann_and_sigma_H_from_har(
                    prices, interval=timeframe, horizon_hours=4
                )
                if sigma_ann:
                    har_estimates.append(sigma_ann)
            except:
                continue
        
        if len(har_estimates) < 2:
            return CalibrationResult(
                parameter_name="garch_har_outlier_ratio",
                current_value=2.0,
                recommended_value=2.0,
                improvement_metric=0.0,
                confidence_level="low",
                rationale="Insufficient data for outlier threshold calibration"
            )
        
        # Calculate variability in HAR estimates
        har_mean = np.mean(har_estimates)
        har_std = np.std(har_estimates)
        har_cv = har_std / har_mean if har_mean > 0 else 0
        
        # Recommend threshold based on HAR variability
        # Higher variability suggests need for more lenient threshold
        if har_cv > 0.3:
            recommended_ratio = 3.0  # More lenient
        elif har_cv > 0.15:
            recommended_ratio = 2.5
        else:
            recommended_ratio = 2.0  # Current default
        
        current_ratio = 2.0
        improvement = abs(recommended_ratio - current_ratio) / current_ratio
        confidence = "medium" if improvement > 0.1 else "low"
        
        return CalibrationResult(
            parameter_name="garch_har_outlier_ratio",
            current_value=current_ratio,
            recommended_value=recommended_ratio,
            improvement_metric=improvement,
            confidence_level=confidence,
            rationale=f"HAR estimate variability (CV={har_cv:.3f}) suggests "
                     f"outlier threshold of {recommended_ratio}. "
                     f"Mean HAR sigma: {har_mean:.3f}, Std: {har_std:.3f}"
        )

    def analyze_horizon_scaling(self) -> CalibrationResult:
        """
        Analyze horizon scaling accuracy and recommend adjustments.
        
        Validates that the sqrt(time) scaling relationship holds
        and suggests corrections if systematic bias is detected.
        """
        print("Analyzing horizon scaling accuracy...")
        
        horizon_results = self.validator.test_horizon_scaling_invariance()
        successful = {k: v for k, v in horizon_results.items() if "error" not in v}
        
        if len(successful) < 3:
            return CalibrationResult(
                parameter_name="horizon_scaling_factor",
                current_value=1.0,
                recommended_value=1.0,
                improvement_metric=0.0,
                confidence_level="low",
                rationale="Insufficient data for horizon scaling analysis"
            )
        
        # Analyze scaling relationships
        scaling_errors = []
        for key, data in successful.items():
            horizon = data["horizon_hours"]
            sigma_frac = data["sigma_h_frac"]
            expected_scaling = data["expected_scaling"]
            
            # Compare with 1-hour baseline (if available)
            baseline_key = "horizon_1h"
            if baseline_key in successful and key != baseline_key:
                baseline_sigma = successful[baseline_key]["sigma_h_frac"]
                actual_ratio = sigma_frac / baseline_sigma
                expected_ratio = math.sqrt(horizon / 1)
                scaling_error = (actual_ratio - expected_ratio) / expected_ratio
                scaling_errors.append(scaling_error)
        
        if scaling_errors:
            mean_error = np.mean(scaling_errors)
            scaling_correction = 1.0 + mean_error
            
            # Limit correction to reasonable bounds
            scaling_correction = max(0.8, min(1.2, scaling_correction))
            
            improvement = abs(mean_error)
            confidence = "high" if improvement > 0.1 else "medium" if improvement > 0.05 else "low"
            
            return CalibrationResult(
                parameter_name="horizon_scaling_factor",
                current_value=1.0,
                recommended_value=round(scaling_correction, 3),
                improvement_metric=improvement,
                confidence_level=confidence,
                rationale=f"Systematic scaling bias of {mean_error:.3f} detected. "
                         f"Correction factor of {scaling_correction:.3f} recommended."
            )
        
        return CalibrationResult(
            parameter_name="horizon_scaling_factor",
            current_value=1.0,
            recommended_value=1.0,
            improvement_metric=0.0,
            confidence_level="medium",
            rationale="Horizon scaling appears accurate within measurement precision"
        )

    def run_full_calibration(self) -> Dict[str, Any]:
        """
        Run complete calibration analysis and return recommendations.
        """
        print("Starting comprehensive calibration analysis...")
        print("=" * 60)
        
        self.calibration_results = []
        
        # Analyze each parameter
        atr_result = self.analyze_atr_to_sigma_factor()
        self.calibration_results.append(atr_result)
        
        blend_results = self.analyze_blend_weights()
        self.calibration_results.extend(blend_results)
        
        outlier_result = self.analyze_outlier_thresholds()
        self.calibration_results.append(outlier_result)
        
        horizon_result = self.analyze_horizon_scaling()
        self.calibration_results.append(horizon_result)
        
        # Summarize results
        summary = {
            "total_parameters_analyzed": len(self.calibration_results),
            "high_confidence_recommendations": len([r for r in self.calibration_results if r.confidence_level == "high"]),
            "medium_confidence_recommendations": len([r for r in self.calibration_results if r.confidence_level == "medium"]),
            "low_confidence_recommendations": len([r for r in self.calibration_results if r.confidence_level == "low"]),
            "recommendations": []
        }
        
        for result in self.calibration_results:
            summary["recommendations"].append({
                "parameter": result.parameter_name,
                "current": result.current_value,
                "recommended": result.recommended_value,
                "improvement": result.improvement_metric,
                "confidence": result.confidence_level,
                "rationale": result.rationale
            })
        
        return summary

    def generate_calibration_report(self) -> str:
        """Generate a formatted calibration report."""
        if not self.calibration_results:
            return "No calibration analysis has been performed."
        
        report = []
        report.append("VOLATILITY MODEL CALIBRATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Group by confidence level
        high_conf = [r for r in self.calibration_results if r.confidence_level == "high"]
        med_conf = [r for r in self.calibration_results if r.confidence_level == "medium"]
        low_conf = [r for r in self.calibration_results if r.confidence_level == "low"]
        
        if high_conf:
            report.append("HIGH CONFIDENCE RECOMMENDATIONS:")
            report.append("-" * 35)
            for result in high_conf:
                report.append(f"Parameter: {result.parameter_name}")
                report.append(f"  Current: {result.current_value}")
                report.append(f"  Recommended: {result.recommended_value}")
                report.append(f"  Improvement: {result.improvement_metric:.1%}")
                report.append(f"  Rationale: {result.rationale}")
                report.append("")
        
        if med_conf:
            report.append("MEDIUM CONFIDENCE RECOMMENDATIONS:")
            report.append("-" * 37)
            for result in med_conf:
                report.append(f"Parameter: {result.parameter_name}")
                report.append(f"  Current: {result.current_value}")
                report.append(f"  Recommended: {result.recommended_value}")
                report.append(f"  Improvement: {result.improvement_metric:.1%}")
                report.append(f"  Rationale: {result.rationale}")
                report.append("")
        
        if low_conf:
            report.append("LOW CONFIDENCE RECOMMENDATIONS:")
            report.append("-" * 34)
            for result in low_conf:
                report.append(f"Parameter: {result.parameter_name}")
                report.append(f"  Current: {result.current_value}")
                report.append(f"  Recommended: {result.recommended_value}")
                report.append(f"  Note: {result.rationale}")
                report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    calibrator = VolatilityCalibrator()
    
    # Run calibration analysis
    summary = calibrator.run_full_calibration()
    
    # Print summary
    print("\nCALIBRATION SUMMARY:")
    print("=" * 30)
    print(f"Parameters analyzed: {summary['total_parameters_analyzed']}")
    print(f"High confidence: {summary['high_confidence_recommendations']}")
    print(f"Medium confidence: {summary['medium_confidence_recommendations']}")
    print(f"Low confidence: {summary['low_confidence_recommendations']}")
    print()
    
    # Print detailed report
    print(calibrator.generate_calibration_report())
    
    # Save results to JSON
    with open("calibration_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Calibration results saved to calibration_results.json")