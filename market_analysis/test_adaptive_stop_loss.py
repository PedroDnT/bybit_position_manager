#!/usr/bin/env python3
"""
Test Suite for Adaptive Stop-Loss Management
===========================================

Comprehensive tests for the adaptive stop-loss functionality that addresses
the issue of stop-loss orders calculated based on entry price for existing positions.

This test suite validates:
1. Current price-based stop-loss calculations
2. Dynamic volatility adjustments
3. Trailing stop functionality
4. Integration with position risk manager
5. Order execution simulation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from adaptive_stop_loss import (
    AdaptiveStopLoss,
    AdaptiveStopLossManager,
    integrate_with_position_manager
)


class TestAdaptiveStopLoss(unittest.TestCase):
    """Test the AdaptiveStopLoss data class and core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_adaptive_stop = AdaptiveStopLoss(
            symbol="BTCUSDT",
            current_price=50000.0,
            entry_price=48000.0,
            stop_loss=47000.0,
            take_profit=52000.0,
            trailing_distance=1000.0,
            volatility_multiplier=1.5,
            needs_adjustment=True,
            adjustment_reason="Current price significantly above entry",
            suggested_stop=49000.0,
            trailing_stop=49500.0,
            last_updated=datetime.now()
        )
    
    def test_adaptive_stop_loss_creation(self):
        """Test AdaptiveStopLoss object creation and attributes."""
        asl = self.sample_adaptive_stop
        
        self.assertEqual(asl.symbol, "BTCUSDT")
        self.assertEqual(asl.current_price, 50000.0)
        self.assertEqual(asl.entry_price, 48000.0)
        self.assertTrue(asl.needs_adjustment)
        self.assertIsNotNone(asl.last_updated)
    
    def test_to_dict_method(self):
        """Test conversion to dictionary."""
        asl_dict = self.sample_adaptive_stop.to_dict()
        
        self.assertIn('symbol', asl_dict)
        self.assertIn('current_price', asl_dict)
        self.assertIn('needs_adjustment', asl_dict)
        self.assertEqual(asl_dict['symbol'], "BTCUSDT")


class TestAdaptiveStopLossManager(unittest.TestCase):
    """Test the AdaptiveStopLossManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_exchange = Mock()
        self.manager = AdaptiveStopLossManager(self.mock_exchange)
        
        # Mock market data
        self.mock_ohlcv = [
            [1640995200000, 47000, 48000, 46500, 47500, 1000],  # timestamp, o, h, l, c, v
            [1641081600000, 47500, 49000, 47000, 48500, 1200],
            [1641168000000, 48500, 50000, 48000, 49500, 1100],
            [1641254400000, 49500, 51000, 49000, 50000, 1300],
        ]
        
        self.mock_ticker = {
            'last': 50000.0,
            'bid': 49995.0,
            'ask': 50005.0
        }
    
    @patch('adaptive_stop_loss.compute_atr')
    def test_get_volatility_metrics(self, mock_atr):
        """Test volatility metrics calculation."""
        mock_atr.return_value = 1500.0
        self.mock_exchange.fetch_ohlcv.return_value = self.mock_ohlcv
        
        volatility = self.manager._get_volatility_metrics("BTCUSDT")
        
        self.assertIn('atr', volatility)
        self.assertIn('volatility_pct', volatility)
        self.mock_exchange.fetch_ohlcv.assert_called_once()
    
    def test_calculate_adaptive_levels_long_position(self):
        """Test adaptive level calculation for long positions."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 1500.0,
                'volatility_pct': 0.03,
                'confidence': 0.8
            }
            self.mock_exchange.fetch_ticker.return_value = self.mock_ticker
            
            levels = self.manager.calculate_adaptive_levels(
                "BTCUSDT", "Buy", 48000.0, 1.0
            )
            
            self.assertEqual(levels.symbol, "BTCUSDT")
            self.assertEqual(levels.current_price, 50000.0)
            self.assertEqual(levels.entry_price, 48000.0)
            self.assertLess(levels.stop_loss, levels.current_price)
            self.assertGreater(levels.take_profit, levels.current_price)
    
    def test_calculate_adaptive_levels_short_position(self):
        """Test adaptive level calculation for short positions."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 1500.0,
                'volatility_pct': 0.03,
                'confidence': 0.8
            }
            self.mock_exchange.fetch_ticker.return_value = self.mock_ticker
            
            levels = self.manager.calculate_adaptive_levels(
                "BTCUSDT", "Sell", 52000.0, 1.0
            )
            
            self.assertEqual(levels.symbol, "BTCUSDT")
            self.assertGreater(levels.stop_loss, levels.current_price)
            self.assertLess(levels.take_profit, levels.current_price)
    
    def test_needs_adjustment_logic(self):
        """Test the logic for determining if adjustment is needed."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 1500.0,
                'volatility_pct': 0.03,
                'confidence': 0.8
            }
            self.mock_exchange.fetch_ticker.return_value = self.mock_ticker
            
            # Test case where current price is significantly above entry
            levels = self.manager.calculate_adaptive_levels(
                "BTCUSDT", "Buy", 45000.0, 1.0  # Entry much lower than current
            )
            
            self.assertTrue(levels.needs_adjustment)
            self.assertIn("significantly", levels.adjustment_reason.lower())
    
    def test_update_trailing_stop_long_position(self):
        """Test trailing stop update for long positions."""
        with patch.object(self.manager, 'calculate_adaptive_levels') as mock_calc:
            mock_levels = Mock()
            mock_levels.trailing_distance = 1000.0
            mock_calc.return_value = mock_levels
            
            # Test moving stop-loss up
            new_stop = self.manager.update_trailing_stop(
                "BTCUSDT", 51000.0, 48000.0, "Buy"
            )
            
            # Should move stop-loss up to 50000 (51000 - 1000)
            self.assertEqual(new_stop, 50000.0)
            
            # Test not moving stop-loss down
            new_stop = self.manager.update_trailing_stop(
                "BTCUSDT", 49000.0, 48000.0, "Buy"
            )
            
            # Should keep existing stop-loss (48000) as it's higher
            self.assertEqual(new_stop, 48000.0)
    
    def test_update_position_stop_loss_dry_run(self):
        """Test stop-loss order update in dry run mode."""
        position = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "size": "1.0"
        }
        
        result = self.manager.update_position_stop_loss(
            position, 49000.0, dry_run=True
        )
        
        self.assertEqual(result["status"], "simulated")
        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["stop_loss"], 49000.0)
    
    def test_update_position_stop_loss_live_mode(self):
        """Test stop-loss order update in live mode."""
        position = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "size": "1.0"
        }
        
        result = self.manager.update_position_stop_loss(
            position, 49000.0, dry_run=False
        )
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["action"], "created")


class TestPositionRiskManagerIntegration(unittest.TestCase):
    """Test integration with PositionRiskManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_position_manager = Mock()
        self.mock_position_manager.positions = [
            {
                "symbol": "BTCUSDT",
                "side": "Buy",
                "avgPrice": "48000.0",
                "size": "1.0"
            }
        ]
    
    def test_integration_function(self):
        """Test the integration function with position manager."""
        with patch('adaptive_stop_loss.AdaptiveStopLossManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            mock_adaptive_stop = Mock()
            mock_adaptive_stop.symbol = "BTCUSDT"
            mock_manager.calculate_adaptive_levels.return_value = mock_adaptive_stop
            
            result = integrate_with_position_manager(
                self.mock_position_manager, "BTCUSDT"
            )
            
            self.assertEqual(result.symbol, "BTCUSDT")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_exchange = Mock()
        self.manager = AdaptiveStopLossManager(self.mock_exchange)
    
    def test_invalid_symbol(self):
        """Test handling of invalid symbols."""
        self.mock_exchange.fetch_ticker.side_effect = Exception("Symbol not found")
        
        with self.assertRaises(Exception):
            self.manager.calculate_adaptive_levels("INVALID", "Buy", 100.0, 1.0)
    
    def test_zero_volatility(self):
        """Test handling of zero volatility scenarios."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 0.0,
                'volatility_pct': 0.0,
                'confidence': 0.0
            }
            self.mock_exchange.fetch_ticker.return_value = {'last': 100.0}
            
            levels = self.manager.calculate_adaptive_levels(
                "STABLECOIN", "Buy", 100.0, 1.0
            )
            
            # Should use minimum volatility
            self.assertGreater(levels.trailing_distance, 0)
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 1000.0,
                'volatility_pct': 0.5,  # 50% volatility
                'confidence': 0.9
            }
            self.mock_exchange.fetch_ticker.return_value = {'last': 100000.0}
            
            # Entry at 50000, current at 100000 (100% gain)
            levels = self.manager.calculate_adaptive_levels(
                "VOLATILE", "Buy", 50000.0, 1.0
            )
            
            self.assertTrue(levels.needs_adjustment)
            self.assertGreater(levels.suggested_stop, 50000.0)  # Should be above entry


class TestPerformanceScenarios(unittest.TestCase):
    """Test various market performance scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_exchange = Mock()
        self.manager = AdaptiveStopLossManager(self.mock_exchange)
    
    def test_bull_market_scenario(self):
        """Test adaptive stop-loss in bull market conditions."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 2000.0,
                'volatility_pct': 0.04,
                'confidence': 0.85
            }
            
            # Simulate strong upward movement
            self.mock_exchange.fetch_ticker.return_value = {'last': 55000.0}
            
            levels = self.manager.calculate_adaptive_levels(
                "BTCUSDT", "Buy", 45000.0, 1.0
            )
            
            # In bull market, should allow for wider stops but still protect gains
            self.assertTrue(levels.needs_adjustment)
            self.assertGreater(levels.suggested_stop, 45000.0)
            self.assertLess(levels.suggested_stop, 55000.0)
    
    def test_bear_market_scenario(self):
        """Test adaptive stop-loss in bear market conditions."""
        with patch.object(self.manager, '_get_volatility_metrics') as mock_vol:
            mock_vol.return_value = {
                'atr': 3000.0,
                'volatility_pct': 0.06,
                'confidence': 0.7
            }
            
            # Simulate downward movement for short position
            self.mock_exchange.fetch_ticker.return_value = {'last': 40000.0}
            
            levels = self.manager.calculate_adaptive_levels(
                "BTCUSDT", "Sell", 45000.0, 1.0
            )
            
            # In bear market for short, should tighten stops to protect gains
            self.assertTrue(levels.needs_adjustment)
            self.assertLess(levels.suggested_stop, 45000.0)
            self.assertGreater(levels.suggested_stop, 40000.0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAdaptiveStopLoss,
        TestAdaptiveStopLossManager,
        TestPositionRiskManagerIntegration,
        TestEdgeCases,
        TestPerformanceScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")