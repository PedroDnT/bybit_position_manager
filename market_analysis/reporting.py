import datetime as dt
from typing import Dict, Any, List


def _calculate_portfolio_metrics(
    positions: List[Dict[str, Any]], risk_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate portfolio-wide risk metrics."""
    total_notional = sum(p["notional"] for p in positions)
    total_pnl = sum(p.get("unrealizedPnl", 0) for p in positions)

    total_dollar_risk = 0
    total_dollar_reward = 0
    positions_at_risk = []

    for symbol, analysis in risk_analysis.items():
        if isinstance(analysis, dict) and "dollar_risk" in analysis:
            total_dollar_risk += analysis["dollar_risk"]
            total_dollar_reward += analysis["dollar_reward"]

            if analysis.get("position_health") in ["CRITICAL", "WARNING"]:
                positions_at_risk.append(symbol)

    portfolio_rr = (
        total_dollar_reward / total_dollar_risk if total_dollar_risk > 0 else 0
    )

    throttle_info = risk_analysis.get("portfolio_throttle")
    if not isinstance(throttle_info, dict):
        throttle_info = {}

    risk_cap_dollars = throttle_info.get("risk_cap_dollars")
    risk_cap_fraction = throttle_info.get("risk_cap_fraction")
    risk_utilization = (
        (total_dollar_risk / risk_cap_dollars)
        if risk_cap_dollars and risk_cap_dollars > 0
        else None
    )
    drawdown_pct = (
        throttle_info.get("drawdown_fraction") * 100
        if throttle_info.get("drawdown_fraction") is not None
        else None
    )

    return {
        "total_positions": len(positions),
        "total_notional": total_notional,
        "total_unrealized_pnl": total_pnl,
        "total_risk_if_all_sl_hit": total_dollar_risk,
        "total_reward_if_all_tp_hit": total_dollar_reward,
        "portfolio_risk_reward_ratio": portfolio_rr,
        "positions_at_risk": positions_at_risk,
        "risk_pct_of_notional": (
            (total_dollar_risk / total_notional * 100) if total_notional > 0 else 0
        ),
        "portfolio_risk_cap_dollars": risk_cap_dollars,
        "portfolio_risk_cap_fraction": risk_cap_fraction,
        "portfolio_risk_utilization": risk_utilization,
        "portfolio_drawdown_pct": drawdown_pct,
        "portfolio_throttle_applied": bool(throttle_info.get("throttle_applied")),
        "portfolio_throttle_scale": throttle_info.get("scale"),
    }


def generate_report(
    risk_analysis: Dict[str, Any],
    positions: List[Dict[str, Any]],
    cfg: dict,
    account_metrics: Dict[str, Any] | None = None,
) -> str:
    """Generate comprehensive risk management report."""
    if not risk_analysis:
        return "No positions to analyze."

    report = []
    report.append("=" * 80)
    report.append("POSITION RISK MANAGEMENT REPORT")
    report.append(
        f"Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    report.append("=" * 80)

    # Portfolio summary
    portfolio = risk_analysis.get("portfolio", {})
    if "portfolio" in risk_analysis:
        report.append("\n📊 PORTFOLIO SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Positions: {portfolio.get('total_positions', 0)}")
        report.append(f"Total Notional: ${portfolio.get('total_notional', 0):,.2f}")
        report.append(
            f"Total Unrealized PnL: ${portfolio.get('total_unrealized_pnl', 0):,.2f}"
        )
        report.append(
            f"Total Risk (if all SL hit): ${portfolio.get('total_risk_if_all_sl_hit', 0):,.2f}"
        )
        report.append(
            f"Total Reward (if all TP hit): ${portfolio.get('total_reward_if_all_tp_hit', 0):,.2f}"
        )
        report.append(
            f"Portfolio Risk/Reward: {portfolio.get('portfolio_risk_reward_ratio', 0):.2f}:1"
        )

        cap_dollars = portfolio.get("portfolio_risk_cap_dollars")
        cap_fraction = portfolio.get("portfolio_risk_cap_fraction")
        utilization = portfolio.get("portfolio_risk_utilization")
        drawdown_pct = portfolio.get("portfolio_drawdown_pct")
        throttle_active = portfolio.get("portfolio_throttle_applied")

        if cap_dollars:
            fraction_pct = cap_fraction * 100 if cap_fraction else None
            if fraction_pct is not None:
                report.append(
                    f"Max Portfolio Risk Budget: ${cap_dollars:,.2f} ({fraction_pct:.2f}% of equity)"
                )
            else:
                report.append(
                    f"Max Portfolio Risk Budget: ${cap_dollars:,.2f}"
                )

        if utilization is not None:
            report.append(
                f"Portfolio Risk Utilization: {utilization * 100:.1f}% of cap"
            )

        if drawdown_pct is not None:
            report.append(
                f"Current Drawdown Anchor: {drawdown_pct:.2f}%"
            )

        if throttle_active:
            scale = portfolio.get("portfolio_throttle_scale")
            if scale is not None:
                report.append(
                    f"⚠️  Portfolio throttle active – new risk scaled by {scale:.2f}."
                )
            else:
                report.append("⚠️  Portfolio throttle active – risk reduced by drawdown guardrails.")

        if portfolio.get("positions_at_risk"):
            report.append(
                f"\n⚠️  Positions at Risk: {', '.join(portfolio['positions_at_risk'])}"
            )

    # Account overview
    if account_metrics:
        report.append("\n🏦 ACCOUNT OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Equity: ${account_metrics.get('total_equity', 0):,.2f}")
        report.append(
            f"Wallet Balance: ${account_metrics.get('total_wallet_balance', 0):,.2f}"
        )
        report.append(
            f"Available Balance: ${account_metrics.get('available_balance', 0):,.2f}"
        )
        report.append(
            f"Total Unrealized PnL: ${account_metrics.get('total_unrealized_pnl', 0):,.2f}"
        )
        report.append(
            f"Today's Realized PnL: ${account_metrics.get('todays_realized_pnl', 0):,.2f}"
        )
        report.append(
            f"Today's Combined PnL: ${account_metrics.get('todays_total_pnl', 0):,.2f}"
        )

    # Individual position analysis
    report.append("\n" + "=" * 80)
    report.append("INDIVIDUAL POSITION ANALYSIS")
    report.append("=" * 80)

    for i, position in enumerate(positions, 1):
        symbol = position["symbol"]
        analysis = risk_analysis.get(symbol, {})

        if not analysis or "error" in analysis:
            report.append(f"\nPosition {i}: {symbol}")
            report.append("  ❌ Analysis failed - manual review required")
            continue

        # Position header
        report.append(f"\nPosition {i}: {symbol}")
        report.append("-" * 40)

        # Current status
        report.append("Current Status:")
        report.append(
            f"  Entry: ${analysis['entry_price']:.6f} | "
            f"Current: ${analysis.get('current_price', 0):.6f} | "
            f"PnL: {analysis.get('current_pnl_pct', 0):.2f}%"
        )
        report.append(
            f"  Size: {analysis['position_size']} | "
            f"Notional: ${analysis['notional']:.2f} | "
            f"Leverage: {analysis['leverage']}x"
        )

        if analysis.get("liquidation_price"):
            report.append(f"  Liquidation Price: ${analysis['liquidation_price']:.6f}")

        # Volatility analysis
        report.append("\nVolatility Analysis:")
        report.append(f"  Method: {analysis['volatility_method']}")
        report.append(
            f"  ATR(20): ${analysis.get('atr', 0):.6f} ({analysis.get('atr_pct', 0):.2f}% of price)"
        )
        if analysis.get("har_sigma_ann"):
            report.append(f"  HAR-RV σ(annual): {analysis['har_sigma_ann']:.1%}")
        if analysis.get("garch_sigma_ann"):
            report.append(f"  GARCH σ(annual): {analysis['garch_sigma_ann']:.1%}")
        report.append(
            f"  Vol blend (H={int(cfg.get('vol', {}).get('horizon_hours', 4))}h): {analysis.get('sigmaH_blend_abs', 0):.6f} abs"
        )

        # Risk management recommendations
        report.append("\n🎯 Recommended Levels:")
        report.append(
            f"  STOP LOSS: ${analysis['stop_loss']:.6f} ({analysis['sl_pct']:.2f}% from entry)"
        )
        report.append(
            f"    💰 Optimal Risk: ${analysis['dollar_risk']:.2f} (for optimal size: {analysis.get('optimal_position_size', 0):.2f})"
        )
        report.append(
            f"    💰 Current Risk: ${analysis.get('current_dollar_risk', 0):.2f} (for current size: {analysis['position_size']:.2f})"
        )
        if analysis.get("liquidation_buffer_safe") is not None:
            ratio = analysis.get("liquidation_buffer_ratio")
            threshold = analysis.get("liquidation_buffer_threshold")
            detail = ""
            if ratio is not None and threshold is not None:
                detail = f" (buffer {ratio:.2f}× vs ≥ {threshold:.2f}×)"
            elif threshold is not None:
                detail = f" (requires ≥ {threshold:.2f}× cushion)"

            if analysis["liquidation_buffer_safe"]:
                report.append(f"    ✅ Safe from liquidation{detail}")
            else:
                report.append(f"    ⚠️  TOO CLOSE TO LIQUIDATION!{detail}")
        report.append(
            f"  TAKE PROFIT: ${analysis['take_profit']:.6f} ({analysis['tp_pct']:.2f}% from entry)"
        )
        report.append(f"    💰 Optimal Reward: ${analysis['dollar_reward']:.2f}")
        report.append(
            f"    💰 Current Reward: ${analysis.get('current_dollar_reward', 0):.2f}"
        )
        report.append(f"    📊 Risk/Reward: {analysis['risk_reward_ratio']:.2f}:1")

        # Scale-outs and trailing
        report.append("  Scale-outs:")
        report.append(
            f"    • Close {analysis['scaleout_frac1']*100:.0f}% @ {analysis['tp1']:.6f}"
        )
        report.append(
            f"    • Close {analysis['scaleout_frac2']*100:.0f}% @ {analysis['tp2']:.6f}"
        )
        report.append(
            f"    • Leave {analysis['leave_runner_frac']*100:.0f}% runner; Trail suggestion: {analysis['trail_stop_suggestion']:.6f}"
        )

        # Professional confidence analysis
        report.append("  Professional Analysis:")
        report.append(f"    Confidence Score: {analysis['regime_score']}/5")
        if analysis.get("confidence_factors"):
            report.append(
                f"    Confidence Factors: {', '.join(analysis['confidence_factors'])}"
            )
        report.append(
            f"    Volatility Adjustment: {analysis.get('volatility_adjustment', 1.0):.2f}x"
        )
        report.append(
            f"    Confidence Adjustment: {analysis.get('confidence_adjustment', 1.0):.2f}x"
        )

        # Risk management details
        report.append("  Risk Management:")
        report.append(
            f"    Risk Target: {analysis['risk_target_pct']*100:.2f}% of notional"
        )
        report.append(
            f"    Stop Loss: {analysis['k_multiplier']:.2f}× ATR (Professional: 2-4×)"
        )
        report.append(
            f"    Take Profit: {analysis['m_multiplier']:.2f}× ATR (Professional: 2.5-4.5×)"
        )

        # Portfolio cap note if present
        if analysis.get("portfolio_note"):
            report.append(f"  Portfolio cap: {analysis['portfolio_note']}")

        # Check if position size is significantly different from optimal
        if analysis.get("optimal_position_size"):
            size_ratio = analysis["position_size"] / analysis["optimal_position_size"]
            if size_ratio > 1.5:
                report.append(
                    f"    ⚠️  POSITION SIZE TOO LARGE: Current is {size_ratio:.1f}x optimal"
                )
            elif size_ratio < 0.5:
                report.append(
                    f"    ℹ️  POSITION SIZE SMALL: Current is {size_ratio:.1f}x optimal"
                )

        # Position assessment
        health_emoji = {
            "CRITICAL": "🔴",
            "WARNING": "🟡",
            "NORMAL": "🟢",
            "PROFITABLE": "💚",
            "UNKNOWN": "⚪",
        }

        report.append("\nRisk Assessment:")
        report.append(
            f"  Status: {health_emoji.get(analysis.get('position_health','NORMAL'), '⚪')} "
            f"{analysis.get('position_health','NORMAL')}"
        )
        report.append(
            f"  Action: {analysis.get('action_required','Set SL/TP as recommended')}"
        )

    # Final recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)

    if portfolio.get("positions_at_risk"):
        report.append("⚠️  IMMEDIATE ATTENTION REQUIRED:")
        for symbol in portfolio["positions_at_risk"]:
            analysis = risk_analysis.get(symbol, {})
            report.append(
                f"  • {symbol}: {analysis.get('action_required', 'Review position')}"
            )

    report.append("\n✅ GENERAL GUIDELINES:")
    report.append("  1. Set all stop losses immediately to protect capital")
    report.append("  2. Monitor high-leverage positions (15x+) closely")
    report.append("  3. Consider trailing stops for profitable positions")
    report.append("  4. Review positions with poor risk/reward ratios (<1.5:1)")

    return "\n".join(report)
