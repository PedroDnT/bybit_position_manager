import os
import argparse
from typing import Any
from .position_risk_manager import PositionRiskManager


def main():
    parser = argparse.ArgumentParser(description="Advanced Crypto Position Risk Manager")
    parser.add_argument("--monitor", action="store_true", help="Run real-time monitoring loop after initial analysis")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between monitoring updates")
    parser.add_argument("--iterations", type=int, default=0, help="Number of monitoring iterations (0 means unlimited)")
    # New: print full report to stdout
    parser.add_argument("--print-report", action="store_true", help="Print the full risk report to stdout")
    args = parser.parse_args()

    manager = PositionRiskManager()
    manager.fetch_positions()
    manager.analyze_all_positions()
    report = manager.generate_report()

    # Export JSON
    out_path = os.path.join(os.path.dirname(__file__), "risk_analysis.json")
    with open(out_path, "w") as f:
        f.write(report)

    # Optionally print the full report to stdout
    if args.print_report:
        print(report)

    # Pretty print summary table that reflects the GARCH-driven levels

    def _fmt_number(value: Any, width: int = 11, precision: int = 4) -> str:
        """Safely format numeric values for table output."""

        try:
            return f"{float(value):<{width}.{precision}f}"
        except (TypeError, ValueError):
            return f"{'N/A':<{width}}"

    def _fmt_price(value: Any) -> str:
        try:
            return f"${float(value):<10.4f}"
        except (TypeError, ValueError):
            return f"{'N/A':<11}"

    def _fmt_ratio(value: Any) -> str:
        try:
            return f"{float(value):<5.1f}:1"
        except (TypeError, ValueError):
            return f"{'N/A':<7}"

    def _fmt_pct(value: Any) -> str:
        try:
            return f"{float(value) * 100:<6.2f}%"
        except (TypeError, ValueError):
            return f"{'N/A':<7}"

    print(
        "Symbol            Side   Entry       Price       Size        SL          TP          Trail       R:R   Risk%   k/m       Method"
    )
    for sym, analysis in manager.risk_analysis.items():
        if not isinstance(analysis, dict) or "side" not in analysis:
            continue

        trail = analysis.get("trail_stop_suggestion")
        km = "N/A"
        k_mult = analysis.get("k_multiplier")
        m_mult = analysis.get("m_multiplier")
        if k_mult is not None and m_mult is not None:
            try:
                km = f"{float(k_mult):.1f}/{float(m_mult):.1f}"
            except (TypeError, ValueError):
                km = "N/A"

        vol_method = analysis.get("volatility_method") or "N/A"

        print(
            f"{sym:<15} "
            f"{analysis.get('side', ''):<6} "
            f"{_fmt_number(analysis.get('entry_price'))} "
            f"{_fmt_price(analysis.get('current_price'))} "
            f"{_fmt_number(analysis.get('position_size'))} "
            f"{_fmt_price(analysis.get('stop_loss'))} "
            f"{_fmt_price(analysis.get('take_profit'))} "
            f"{_fmt_price(trail)} "
            f"{_fmt_ratio(analysis.get('risk_reward_ratio'))} "
            f"{_fmt_pct(analysis.get('risk_target_pct'))} "
            f"{km:<9} "
            f"{vol_method:<12}"
        )

    if args.monitor:
        # if iterations is 0, treat as unlimited
        iterations = None if args.iterations == 0 else args.iterations
        manager.monitor_positions(interval_seconds=args.interval, iterations=iterations)


if __name__ == "__main__":
    main()
