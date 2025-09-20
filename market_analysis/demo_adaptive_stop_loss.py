#!/usr/bin/env python3
"""
Adaptive Stop-Loss Demonstration
===============================

This script demonstrates the new adaptive stop-loss functionality that solves
the issue of stop-loss orders calculated based on entry price for existing positions.

Key Features Demonstrated:
1. Current price-based stop-loss calculations
2. Dynamic volatility adjustments  
3. Trailing stop functionality
4. Integration with existing position management
5. Order execution (dry run and live modes)

Usage:
    python demo_adaptive_stop_loss.py [--live] [--symbol SYMBOL]
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_analysis.position_risk_manager import PositionRiskManager
from market_analysis.adaptive_stop_loss import AdaptiveStopLossManager


def print_banner():
    """Print demonstration banner."""
    print("=" * 80)
    print("ADAPTIVE STOP-LOSS DEMONSTRATION")
    print("=" * 80)
    print("Solving the issue of entry-price based stop-loss for existing positions")
    print("Industry best practices implementation for dynamic stop-loss management")
    print("=" * 80)


def demonstrate_problem():
    """Demonstrate the original problem with entry-price based stop-loss."""
    print("\n🔍 PROBLEM DEMONSTRATION")
    print("-" * 40)
    print("Original Issue: Stop-loss calculated from entry price")
    print()
    
    # Simulate a position that has moved favorably
    entry_price = 45000.0
    current_price = 52000.0
    original_stop_loss = entry_price * 0.95  # 5% below entry
    
    print(f"Position Details:")
    print(f"  Entry Price:     ${entry_price:,.2f}")
    print(f"  Current Price:   ${current_price:,.2f}")
    print(f"  Unrealized P&L:  +{((current_price - entry_price) / entry_price * 100):.1f}%")
    print()
    print(f"Original Stop-Loss Calculation:")
    print(f"  Stop-Loss:       ${original_stop_loss:,.2f} (5% below entry)")
    print(f"  Risk from Current: {((current_price - original_stop_loss) / current_price * 100):.1f}%")
    print()
    print("❌ ISSUES:")
    print("  • Stop-loss may have already been triggered")
    print("  • Doesn't account for favorable price movement")
    print("  • Risk exposure increases as position becomes profitable")
    print("  • No dynamic adjustment for changing market conditions")


def demonstrate_solution(manager: AdaptiveStopLossManager):
    """Demonstrate the adaptive stop-loss solution."""
    print("\n✅ ADAPTIVE SOLUTION DEMONSTRATION")
    print("-" * 40)
    print("New Approach: Current price-based adaptive stop-loss")
    print()
    
    # Calculate adaptive levels for the same position
    try:
        levels = manager.calculate_adaptive_levels(
            symbol="BTCUSDT",
            side="Buy", 
            entry_price=45000.0,
            position_size=1.0
        )
        
        print(f"Adaptive Stop-Loss Calculation:")
        print(f"  Current Price:        ${levels.current_price:,.2f}")
        print(f"  Entry Price:          ${levels.entry_price:,.2f}")
        print(f"  Adaptive Stop-Loss:   ${levels.stop_loss:,.2f}")
        print(f"  Adaptive Take-Profit: ${levels.take_profit:,.2f}")
        print(f"  Trailing Distance:    ${levels.trailing_distance:,.2f}")
        print(f"  Volatility Multiplier: {levels.volatility_multiplier:.2f}x")
        print()
        print(f"Risk Management:")
        print(f"  Risk from Current:    {((levels.current_price - levels.stop_loss) / levels.current_price * 100):.1f}%")
        print(f"  Potential Reward:     {((levels.take_profit - levels.current_price) / levels.current_price * 100):.1f}%")
        print(f"  Risk/Reward Ratio:    1:{((levels.take_profit - levels.current_price) / (levels.current_price - levels.stop_loss)):.2f}")
        print()
        print("✅ ADVANTAGES:")
        print("  • Based on current market price")
        print("  • Accounts for realized gains")
        print("  • Dynamic volatility adjustment")
        print("  • Trailing stop functionality")
        print("  • Protects profits while allowing for continued gains")
        
        if levels.needs_adjustment:
            print()
            print(f"📊 ADJUSTMENT ANALYSIS:")
            print(f"  Needs Adjustment: {levels.needs_adjustment}")
            print(f"  Reason: {levels.adjustment_reason}")
            if levels.suggested_stop:
                print(f"  Suggested Stop: ${levels.suggested_stop:,.2f}")
            if levels.trailing_stop:
                print(f"  Trailing Stop: ${levels.trailing_stop:,.2f}")
        
        return levels
        
    except Exception as e:
        print(f"❌ Error calculating adaptive levels: {e}")
        return None


def demonstrate_trailing_stop(manager: AdaptiveStopLossManager):
    """Demonstrate trailing stop functionality."""
    print("\n📈 TRAILING STOP DEMONSTRATION")
    print("-" * 40)
    
    # Simulate price movements
    initial_price = 50000.0
    initial_stop = 48000.0
    
    price_movements = [
        (51000.0, "Price moves up 2%"),
        (52000.0, "Price moves up another 2%"),
        (51500.0, "Price retraces slightly"),
        (53000.0, "Price breaks to new high"),
        (52500.0, "Minor pullback")
    ]
    
    current_stop = initial_stop
    print(f"Initial Position:")
    print(f"  Price: ${initial_price:,.2f}")
    print(f"  Stop:  ${initial_stop:,.2f}")
    print()
    
    for price, description in price_movements:
        try:
            new_stop = manager.update_trailing_stop(
                "BTCUSDT", price, current_stop, "Buy"
            )
            
            stop_moved = new_stop != current_stop
            print(f"{description}:")
            print(f"  Price: ${price:,.2f} | Stop: ${new_stop:,.2f} {'📈' if stop_moved else '⏸️'}")
            
            current_stop = new_stop
            
        except Exception as e:
            print(f"  Error updating trailing stop: {e}")
    
    print()
    print("🎯 Trailing Stop Benefits:")
    print("  • Automatically locks in profits as price moves favorably")
    print("  • Only moves in favorable direction (up for longs, down for shorts)")
    print("  • Maintains consistent risk distance based on volatility")


def demonstrate_integration(risk_manager: PositionRiskManager):
    """Demonstrate integration with existing position risk manager."""
    print("\n🔗 INTEGRATION DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Analyze positions with adaptive stop-loss
        print("Analyzing positions with adaptive stop-loss integration...")
        analysis_result = risk_manager.analyze_all_positions()
        
        if not analysis_result or 'positions' not in analysis_result:
            print("No positions found or analysis failed")
            return
        
        print(f"\nFound {len([k for k in analysis_result['positions'].keys() if k != 'portfolio'])} position(s)")
        
        for symbol, analysis in analysis_result['positions'].items():
            if symbol == 'portfolio':
                continue
                
            print(f"\n📊 {symbol} Analysis:")
            if 'adaptive_stop_loss' in analysis:
                print(f"  ✅ Adaptive Stop-Loss: ${analysis['adaptive_stop_loss']:,.6f}")
                print(f"  ✅ Adaptive Take-Profit: ${analysis['adaptive_take_profit']:,.6f}")
                print(f"  ✅ Volatility Adjustment: {analysis['volatility_adjustment']:.2f}x")
                print(f"  ✅ Method: {analysis['adaptive_method']}")
            else:
                print(f"  ❌ Adaptive calculation failed: {analysis.get('adaptive_error', 'Unknown error')}")
        
        # Demonstrate order updates (dry run)
        print(f"\n🔄 ORDER UPDATE SIMULATION")
        print("-" * 30)
        update_results = risk_manager.update_stop_loss_orders(dry_run=True)
        
        if update_results.get('updated_orders'):
            print(f"✅ Successfully simulated {len(update_results['updated_orders'])} order updates:")
            for order in update_results['updated_orders']:
                print(f"  • {order['symbol']}: New stop-loss ${order['new_stop_loss']:,.6f}")
        
        if update_results.get('failed_orders'):
            print(f"❌ Failed to update {len(update_results['failed_orders'])} orders:")
            for failed in update_results['failed_orders']:
                print(f"  • {failed['symbol']}: {failed['error']}")
                
    except Exception as e:
        print(f"❌ Integration demonstration failed: {e}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Adaptive Stop-Loss Demonstration')
    parser.add_argument('--live', action='store_true', 
                       help='Use live trading (default: sandbox)')
    parser.add_argument('--symbol', default='BTCUSDT',
                       help='Symbol to demonstrate (default: BTCUSDT)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Demonstrate the original problem
    demonstrate_problem()
    
    try:
        # Initialize managers
        print(f"\n🔧 INITIALIZING SYSTEM")
        print("-" * 40)
        print(f"Mode: {'LIVE TRADING' if not args.live else 'SANDBOX'}")
        print(f"Symbol: {args.symbol}")
        
        risk_manager = PositionRiskManager(sandbox=not args.live)
        adaptive_manager = risk_manager.adaptive_sl_manager
        
        print("✅ System initialized successfully")
        
        # Demonstrate the solution
        levels = demonstrate_solution(adaptive_manager)
        
        # Demonstrate trailing stops
        demonstrate_trailing_stop(adaptive_manager)
        
        # Demonstrate integration
        demonstrate_integration(risk_manager)
        
        print(f"\n🎉 DEMONSTRATION COMPLETE")
        print("-" * 40)
        print("Key Takeaways:")
        print("✅ Adaptive stop-loss solves entry-price calculation issues")
        print("✅ Dynamic adjustment based on current market conditions")
        print("✅ Trailing stops protect profits while allowing for gains")
        print("✅ Seamless integration with existing position management")
        print("✅ Industry best practices implementation")
        
        if not args.live:
            print("\n💡 To test with live data, run with --live flag")
            print("⚠️  Always test thoroughly before using in live trading")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("• Ensure API credentials are properly configured")
        print("• Check network connectivity")
        print("• Verify exchange is accessible")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())