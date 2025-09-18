from .position_risk_manager import PositionRiskManager
import argparse
from .order_executor import RiskOrderExecutor


def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Crypto Position Risk Manager")
    parser.add_argument("--sandbox", action="store_true", help="Use Bybit testnet")
    parser.add_argument("--place-orders", action="store_true", help="Submit orders after analysis")
    parser.add_argument("--live", action="store_true", help="Actually place orders (default is dry-run)")
    parser.add_argument("--disable-trailing", action="store_true", help="Disable trailing stop submission")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to limit submissions")
    args = parser.parse_args()

    # Lazy import to avoid heavy imports at CLI metadata time
    from .order_executor import RiskOrderExecutor

    # Initialize risk manager
    manager = PositionRiskManager(sandbox=args.sandbox)

    # Fetch and analyze positions
    positions = manager.fetch_positions()

    if not positions:
        print("No positions to analyze. Exiting.")
        return

    # Optional symbol filtering (robust normalization)
    if args.symbols:
        def _norm(s: str) -> str:
            return "".join(ch for ch in s.upper().strip() if ch.isalnum())
        allow_norm = {_norm(s) for s in args.symbols.split(",") if s.strip()}
        filtered = []
        for p in positions:
            pos_sym = p.get("symbol", "")
            pos_norm = _norm(pos_sym)
            if any(a in pos_norm for a in allow_norm):
                filtered.append(p)
        if not filtered:
            print(f"No positions matched --symbols filter: {args.symbols}. Exiting.")
            return
        positions = filtered
        # Ensure manager analyzes only the filtered positions
        manager.positions = positions

    # Analyze all positions
    manager.analyze_all_positions()

    # Generate and print report
    report = manager.generate_report()
    print("\n" + report)

    # Export to JSON
    manager.export_to_json()

    # Print summary table
    print("\n" + "=" * 80)
    print("QUICK REFERENCE TABLE")
    print("=" * 80)

    if manager.account_metrics:
        metrics = manager.account_metrics
        print(f"Account Equity: ${metrics.get('total_equity', 0):,.2f}")
        print(f"Wallet Balance: ${metrics.get('total_wallet_balance', 0):,.2f}")
        print(f"Today's Realized PnL: ${metrics.get('todays_realized_pnl', 0):,.2f}")
        print(f"Today's Total PnL (Realized + Unrealized): ${metrics.get('todays_total_pnl', 0):,.2f}")
        print("-" * 80)
    print(f"{'Symbol':<15} {'Side':<5} {'Entry':<10} {'SL':<10} {'TP':<10} {'R:R':<6} {'Risk%':<7} {'k/m':<9}")
    print("-" * 80)

    for position in positions:
        symbol = position['symbol']
        analysis = manager.risk_analysis.get(symbol, {})

        if analysis and 'stop_loss' in analysis:
            print(f"{symbol:<15} {analysis['side'].upper():<5} "
                  f"${analysis['entry_price']:<9.4f} "
                  f"${analysis['stop_loss']:<9.4f} "
                  f"${analysis['take_profit']:<9.4f} "
                  f"{analysis['risk_reward_ratio']:.2f}:1 "
                  f"{analysis['risk_target_pct']*100:.2f}% "
                  f"{analysis['k_multiplier']:.1f}/{analysis['m_multiplier']:.1f}")

    # Optionally place orders
    if args.place_orders:
        executor = RiskOrderExecutor(manager.exchange, dry_run=(not args.live))
        enable_trailing = not args.disable_trailing
        mode = "LIVE" if args.live else "DRY-RUN (verbose)"
        print(f"\nSubmitting orders... mode={mode}, trailing={'on' if enable_trailing else 'off'}")
        for position in positions:
            sym = position.get("symbol")
            analysis = manager.risk_analysis.get(sym, {})
            if not analysis or 'stop_loss' not in analysis:
                print(f"Skipping {sym}: no valid analysis with stop-loss.")
                continue

            # Verbose summary for dry-run before submission
            if not args.live:
                side = analysis['side'].lower()
                qty = float(position.get('size', 0.0))
                sl = float(analysis['stop_loss'])
                tp1 = float(analysis.get('tp1', analysis.get('take_profit')))
                tp2 = float(analysis.get('tp2', analysis.get('take_profit')))
                frac1 = float(analysis.get('scaleout_frac1', 0.5))
                frac2 = float(analysis.get('scaleout_frac2', 0.5))
                runner_frac = float(analysis.get('leave_runner_frac', 0.0))
                tp1_qty = qty * max(min(frac1, 1.0), 0.0)
                tp2_qty = qty * max(min(frac2, 1.0), 0.0)
                runner_qty = qty * max(min(runner_frac, 1.0), 0.0)
                trail_price = analysis.get('trail_stop_suggestion')
                current_price = float(analysis.get('current_price', 0.0))
                callback_pct = 0.0
                if trail_price and current_price:
                    try:
                        callback_pct = abs(float(trail_price) - current_price) / current_price * 100.0
                    except Exception:
                        callback_pct = 0.0
                print(f"- {sym} {side} qty={qty:g}")
                print(f"  StopLoss: price={sl}")
                print(f"  TP1: qty={tp1_qty:g} price={tp1}")
                print(f"  TP2: qty={tp2_qty:g} price={tp2}")
                if enable_trailing and trail_price:
                    print(f"  Trailing: activation={float(trail_price)} approx_callback%={callback_pct:.3f} runner_qty={runner_qty:g}")

            try:
                executor.submit_orders(position, analysis, enable_trailing=enable_trailing)
                print(f"{'LIVE' if args.live else 'DRY-RUN'}: submitted for {sym}")
            except Exception as e:
                print(f"Error submitting orders for {sym}: {e}")


if __name__ == "__main__":
    main()
