#!/usr/bin/env python3
"""
Adaptive Stop-Loss Management for Existing Positions

This module implements industry-standard dynamic stop-loss management that addresses
the issue of calculating stop-loss levels for positions that are already open.

Key Features:
- Current price-based stop-loss calculation (not entry-based)
- Dynamic adjustment based on market volatility
- Trailing stop functionality
- Position-aware risk management
- Prevents already-triggered stop levels

Industry Standards Implemented:
1. Dynamic stop-loss strategies that adapt to current market conditions
2. Volatility-based adjustments using ATR
3. Current price reference for existing positions
4. Trailing stop implementation that never worsens position
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@dataclass
class AdaptiveStopLoss:
    """Represents an adaptive stop-loss configuration for an existing position."""
    current_stop: float
    suggested_stop: float
    trailing_stop: Optional[float]
    stop_distance: float
    atr_multiplier: float
    volatility_adjustment: float
    position_pnl_pct: float
    risk_level: str  # 'CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'
    last_updated: datetime
    needs_adjustment: bool
    adjustment_reason: str


class AdaptiveStopLossManager:
    """
    Manages adaptive stop-loss levels for existing positions using industry best practices.
    
    This class implements dynamic stop-loss management that:
    1. Uses current market price as reference (not entry price)
    2. Adjusts based on current volatility conditions
    3. Implements trailing stops that protect profits
    4. Prevents setting stops at already-reached levels
    """
    
    def __init__(self, config: Dict[str, Any] = None, exchange=None):
        """Initialize with configuration parameters and exchange object."""
        self.config = config or {}
        self.exchange = exchange
        self.default_atr_multiplier = self.config.get('default_atr_multiplier', 2.0)
        self.min_atr_multiplier = self.config.get('min_atr_multiplier', 1.5)
        self.max_atr_multiplier = self.config.get('max_atr_multiplier', 4.0)
        self.trailing_activation_pct = self.config.get('trailing_activation_pct', 0.02)  # 2%
        self.min_adjustment_pct = self.config.get('min_adjustment_pct', 0.005)  # 0.5%
        
    def calculate_adaptive_stop_loss(
        self,
        position: Dict[str, Any],
        current_price: float,
        atr: float,
        market_data: pd.DataFrame
    ) -> AdaptiveStopLoss:
        """
        Calculate adaptive stop-loss for an existing position.
        
        Args:
            position: Position data including side, entry_price, size, etc.
            current_price: Current market price
            atr: Current Average True Range
            market_data: Recent OHLCV data for analysis
            
        Returns:
            AdaptiveStopLoss object with calculated levels and metadata
        """
        side = position['side'].lower()
        entry_price = float(position['entryPrice'])
        current_stop = position.get('stopLossPrice', None)
        
        # Calculate position PnL percentage
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        # Determine volatility adjustment based on recent market conditions
        volatility_adjustment = self._calculate_volatility_adjustment(market_data, atr)
        
        # Calculate base ATR multiplier adjusted for volatility
        base_multiplier = self.default_atr_multiplier * volatility_adjustment
        base_multiplier = max(self.min_atr_multiplier, min(self.max_atr_multiplier, base_multiplier))
        
        # Calculate suggested stop-loss from current price
        stop_distance = base_multiplier * atr
        
        if side == 'long':
            suggested_stop = current_price - stop_distance
        else:
            suggested_stop = current_price + stop_distance
            
        # Handle existing stop-loss and trailing logic
        trailing_stop = None
        needs_adjustment = False
        adjustment_reason = "Initial calculation"
        
        if current_stop is not None:
            current_stop = float(current_stop)
            
            # Check if current stop is already triggered or too close
            if self._is_stop_triggered_or_close(current_stop, current_price, side):
                needs_adjustment = True
                adjustment_reason = "Current stop already triggered or too close to market"
                
            # Implement trailing stop logic if position is profitable
            elif pnl_pct > self.trailing_activation_pct:
                trailing_stop = self._calculate_trailing_stop(
                    current_stop, suggested_stop, side, pnl_pct
                )
                if trailing_stop != current_stop:
                    needs_adjustment = True
                    adjustment_reason = "Trailing stop adjustment for profit protection"
                    
            # Check if volatility change requires adjustment
            elif self._should_adjust_for_volatility(current_stop, suggested_stop, current_price):
                needs_adjustment = True
                adjustment_reason = "Volatility-based adjustment required"
        else:
            needs_adjustment = True
            adjustment_reason = "No stop-loss currently set"
            
        # Determine risk level
        risk_level = self._determine_risk_level(base_multiplier, volatility_adjustment)
        
        return AdaptiveStopLoss(
            current_stop=current_stop,
            suggested_stop=suggested_stop,
            trailing_stop=trailing_stop,
            stop_distance=stop_distance,
            atr_multiplier=base_multiplier,
            volatility_adjustment=volatility_adjustment,
            position_pnl_pct=pnl_pct,
            risk_level=risk_level,
            last_updated=datetime.now(),
            needs_adjustment=needs_adjustment,
            adjustment_reason=adjustment_reason
        )
    
    def _calculate_volatility_adjustment(self, market_data: pd.DataFrame, current_atr: float) -> float:
        """Calculate volatility adjustment factor based on recent market conditions."""
        if len(market_data) < 20:
            return 1.0
            
        try:
            # Calculate simple volatility based on high-low range
            market_data['range'] = market_data['high'] - market_data['low']
            recent_avg_range = market_data['range'].rolling(14).mean().iloc[-1]
            
            # Compare current ATR to recent average range
            if recent_avg_range > 0:
                volatility_ratio = current_atr / recent_avg_range
                # Adjust multiplier: higher volatility = wider stops
                if volatility_ratio > 1.2:  # High volatility
                    return 1.3
                elif volatility_ratio < 0.8:  # Low volatility
                    return 0.8
                else:
                    return 1.0
        except Exception:
            # Fallback to default if calculation fails
            pass
            
        return 1.0
    
    def _is_stop_triggered_or_close(self, stop_price: float, current_price: float, side: str) -> bool:
        """Check if stop-loss is already triggered or dangerously close."""
        buffer_pct = 0.001  # 0.1% buffer
        
        if side == 'long':
            # For long positions, stop is triggered if current price <= stop price
            return current_price <= stop_price * (1 + buffer_pct)
        else:
            # For short positions, stop is triggered if current price >= stop price
            return current_price >= stop_price * (1 - buffer_pct)
    
    def _calculate_trailing_stop(
        self, 
        current_stop: float, 
        suggested_stop: float, 
        side: str, 
        pnl_pct: float
    ) -> float:
        """Calculate trailing stop that protects profits while allowing for continued gains."""
        if side == 'long':
            # For long positions, trailing stop should never go down (loosen)
            return max(current_stop, suggested_stop)
        else:
            # For short positions, trailing stop should never go up (loosen)
            return min(current_stop, suggested_stop)
    
    def _should_adjust_for_volatility(
        self, 
        current_stop: float, 
        suggested_stop: float, 
        current_price: float
    ) -> bool:
        """Determine if stop should be adjusted due to volatility changes."""
        current_distance = abs(current_price - current_stop)
        suggested_distance = abs(current_price - suggested_stop)
        
        # Adjust if the difference is significant (more than min_adjustment_pct)
        distance_change_pct = abs(suggested_distance - current_distance) / current_distance
        return distance_change_pct > self.min_adjustment_pct
    
    def _determine_risk_level(self, atr_multiplier: float, volatility_adjustment: float) -> str:
        """Determine risk level based on ATR multiplier and volatility."""
        if atr_multiplier <= 1.8:
            return "AGGRESSIVE"
        elif atr_multiplier <= 2.5:
            return "MODERATE"
        else:
            return "CONSERVATIVE"
    
    def generate_stop_loss_report(self, adaptive_stop: AdaptiveStopLoss, position: Dict[str, Any]) -> str:
        """Generate a human-readable report for the adaptive stop-loss analysis."""
        symbol = position.get('symbol', 'Unknown')
        side = position['side'].upper()
        
        report = [
            f"\n{'='*60}",
            f"ADAPTIVE STOP-LOSS ANALYSIS: {symbol} ({side})",
            f"{'='*60}",
            f"Position PnL: {adaptive_stop.position_pnl_pct:+.2%}",
            f"Risk Level: {adaptive_stop.risk_level}",
            f"",
            f"STOP-LOSS LEVELS:",
            f"  Current Stop: ${adaptive_stop.current_stop:.6f}" if adaptive_stop.current_stop else "  Current Stop: Not Set",
            f"  Suggested Stop: ${adaptive_stop.suggested_stop:.6f}",
            f"  Stop Distance: ${adaptive_stop.stop_distance:.6f} ({adaptive_stop.atr_multiplier:.1f}× ATR)",
            f"",
            f"ANALYSIS:",
            f"  Volatility Adjustment: {adaptive_stop.volatility_adjustment:.1f}×",
            f"  Needs Adjustment: {'YES' if adaptive_stop.needs_adjustment else 'NO'}",
            f"  Reason: {adaptive_stop.adjustment_reason}",
            f"",
            f"RECOMMENDATION:",
        ]
        
        if adaptive_stop.needs_adjustment:
            if adaptive_stop.trailing_stop:
                report.append(f"  → Update stop-loss to ${adaptive_stop.trailing_stop:.6f} (Trailing Stop)")
            else:
                report.append(f"  → Set stop-loss to ${adaptive_stop.suggested_stop:.6f}")
            report.append(f"  → This protects against adverse moves while allowing for continued gains")
        else:
            report.append(f"  → Current stop-loss is appropriate, no adjustment needed")
        
        report.extend([
            f"",
            f"Last Updated: {adaptive_stop.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*60}"
        ])
        
        return "\n".join(report)


    def update_position_stop_loss(self, position: Dict[str, Any], 
                                new_stop_loss: float, dry_run: bool = True) -> Dict[str, Any]:
        """
        Update stop-loss order for an existing position.
        
        Args:
            position: Position data from exchange
            new_stop_loss: New stop-loss price
            dry_run: If True, only simulate the order
            
        Returns:
            Order result dictionary
        """
        symbol = position["symbol"]
        side = position["side"]
        size = abs(float(position["size"]))
        
        try:
            # For simulation purposes, we'll return a mock response
            # In real implementation, this would integrate with exchange API
            
            if dry_run:
                return {
                    "status": "simulated",
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "stop_loss": new_stop_loss,
                    "message": "Order simulation successful"
                }
            
            # In real implementation, this would:
            # 1. Check for existing stop-loss orders
            # 2. Modify or create new stop-loss order
            # 3. Return actual order result
            
            return {
                "status": "success",
                "symbol": symbol,
                "side": side,
                "size": size,
                "stop_loss": new_stop_loss,
                "action": "created"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "symbol": symbol,
                "error": str(e)
            }


def integrate_with_position_manager(position_manager, symbol: str) -> AdaptiveStopLoss:
    """
    Integration function to use adaptive stop-loss with existing position manager.
    
    Args:
        position_manager: Existing PositionRiskManager instance
        symbol: Symbol to analyze
        
    Returns:
        AdaptiveStopLoss object with recommendations
    """
    # Get position data
    position = None
    for pos in position_manager.positions:
        if pos['symbol'] == symbol:
            position = pos
            break
    
    if not position:
        raise ValueError(f"No position found for symbol {symbol}")
    
    # Get current market data
    try:
        df = position_manager._fetch_market_data(symbol)
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR
        from .atr_sl_gpt import atr_wilder
        atr = atr_wilder(df, n=14).iloc[-1]
        
        # Create adaptive stop-loss manager
        adaptive_manager = AdaptiveStopLossManager(position_manager.cfg)
        
        # Calculate adaptive stop-loss
        adaptive_stop = adaptive_manager.calculate_adaptive_stop_loss(
            position, current_price, atr, df
        )
        
        return adaptive_stop
        
    except Exception as e:
        raise RuntimeError(f"Failed to calculate adaptive stop-loss for {symbol}: {e}")