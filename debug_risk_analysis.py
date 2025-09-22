#!/usr/bin/env python3
"""Debug script to trace the risk analysis process and see where positions might be lost."""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_analysis.position_risk_manager import PositionRiskManager

def debug_risk_analysis():
    """Debug the risk analysis process step by step."""
    print("🔍 DEBUGGING RISK ANALYSIS PROCESS")
    print("=" * 60)
    
    try:
        # Initialize risk manager
        print("1. Initializing PositionRiskManager...")
        risk_manager = PositionRiskManager(sandbox=False)
        
        # Fetch positions
        print("\n2. Fetching positions...")
        positions = risk_manager.fetch_positions()
        print(f"   Raw positions fetched: {len(positions)}")
        
        for i, pos in enumerate(positions):
            print(f"   Position {i+1}: {pos.get('symbol')} - {pos.get('side')} - Size: {pos.get('size')}")
        
        # Analyze each position individually
        print(f"\n3. Analyzing each position individually...")
        risk_analysis = {}
        
        for i, position in enumerate(positions):
            symbol = position.get('symbol')
            print(f"\n   Analyzing position {i+1}/{len(positions)}: {symbol}")
            
            try:
                analysis = risk_manager.analyze_position_volatility(position)
                risk_analysis[symbol] = analysis
                print(f"   ✅ Successfully analyzed {symbol}")
                
                # Check if analysis has key data
                if 'atr' in analysis:
                    print(f"      ATR: {analysis['atr']:.6f}")
                if 'recommended_sl' in analysis:
                    print(f"      Recommended SL: {analysis['recommended_sl']}")
                    
            except Exception as e:
                print(f"   ❌ Failed to analyze {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n4. Risk analysis results:")
        print(f"   Positions analyzed: {len(risk_analysis)}")
        print(f"   Symbols in analysis: {list(risk_analysis.keys())}")
        
        # Run full analysis
        print(f"\n5. Running full analyze_all_positions()...")
        full_analysis = risk_manager.analyze_all_positions()
        
        positions_in_full = full_analysis.get('positions', {})
        print(f"   Positions in full analysis: {len(positions_in_full)}")
        print(f"   Symbols in full analysis: {list(positions_in_full.keys())}")
        
        # Compare results
        print(f"\n6. Comparison:")
        print(f"   Raw positions: {len(positions)}")
        print(f"   Individual analysis: {len(risk_analysis)}")
        print(f"   Full analysis: {len(positions_in_full)}")
        
        # Check for missing positions
        raw_symbols = {pos.get('symbol') for pos in positions}
        full_symbols = set(positions_in_full.keys())
        missing_symbols = raw_symbols - full_symbols
        
        if missing_symbols:
            print(f"\n❌ MISSING POSITIONS IN FULL ANALYSIS:")
            for symbol in missing_symbols:
                print(f"   - {symbol}")
        else:
            print(f"\n✅ All positions included in full analysis")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_risk_analysis()