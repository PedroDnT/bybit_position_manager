import os
import argparse
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

    # Pretty print summary table
    print("Symbol            Side   Entry       Price       Size        SL          TP          R:R   Risk%   k/m")
    for sym, analysis in manager.risk_analysis.items():
        if not isinstance(analysis, dict) or 'side' not in analysis:
            continue
        print(
            f"{sym:<15} "
            f"{analysis['side']:<6} "
            f"{analysis['entry_price']:<11.4f} "
            f"${analysis['current_price']:<10.4f} "
            f"{analysis['position_size']:<10.4f} "
            f"${analysis['stop_loss']:<9.4f} "
            f"${analysis['take_profit']:<9.4f} "
            f"{analysis['risk_reward_ratio']:<5.1f}:1 "
            f"{analysis['risk_target_pct']*100:<6.2f}% "
            f"{analysis['k_multiplier']:.1f}/{analysis['m_multiplier']:.1f}"
        )

    if args.monitor:
        # if iterations is 0, treat as unlimited
        iterations = None if args.iterations == 0 else args.iterations
        manager.monitor_positions(interval_seconds=args.interval, iterations=iterations)


if __name__ == "__main__":
    main()
