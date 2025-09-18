"""Utilities to push stop-loss, take-profit, and trailing-stop orders via the exchange API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

import ccxt

logger = logging.getLogger(__name__)


@dataclass
class OrderExecutionResult:
    """Simple container for order submission responses."""

    stop_loss: Dict[str, Any] | None
    take_profit_1: Dict[str, Any] | None
    take_profit_2: Dict[str, Any] | None
    trailing_stop: Dict[str, Any] | None


class RiskOrderExecutor:
    """Submit stop, take-profit, and trailing orders for an existing position."""

    def __init__(self, exchange: ccxt.Exchange, *, dry_run: bool = False):
        self.exchange = exchange
        self.dry_run = dry_run
        # Ensure we don't hang indefinitely on network calls
        try:
            # ccxt uses milliseconds for timeout; use a conservative 15s default if unset or excessively large
            current_timeout = getattr(self.exchange, 'timeout', None)
            if current_timeout is None or (isinstance(current_timeout, (int, float)) and current_timeout > 60000):
                self.exchange.timeout = 15000
        except Exception:
            # Don't let timeout tuning break execution
            pass

    def submit_orders(
        self,
        position: Dict[str, Any],
        analysis: Dict[str, Any],
        *,
        enable_trailing: bool = True,
    ) -> OrderExecutionResult:
        """Place all risk-management orders for a single position."""

        if not analysis or 'stop_loss' not in analysis:
            raise ValueError("Risk analysis missing stop-loss data for order execution.")

        symbol = position['symbol']
        side = analysis['side'].lower()
        quantity = float(position['size'])
        if quantity <= 0:
            raise ValueError(f"Position size for {symbol} is zero; nothing to hedge.")

        stop_loss_price = float(analysis['stop_loss'])
        tp1_price = float(analysis.get('tp1', analysis.get('take_profit')))
        tp2_price = float(analysis.get('tp2', analysis.get('take_profit')))
        frac1 = float(analysis.get('scaleout_frac1', 0.5))
        frac2 = float(analysis.get('scaleout_frac2', 0.5))
        runner_frac = float(analysis.get('leave_runner_frac', 0.0))
        trail_price = analysis.get('trail_stop_suggestion')
        current_price = float(analysis.get('current_price', 0.0))

        # Compute scale-out quantities and cap total to available position size
        desired_tp1_qty = quantity * max(min(frac1, 1.0), 0.0)
        desired_tp2_qty = quantity * max(min(frac2, 1.0), 0.0)
        desired_runner_qty = quantity * max(min(runner_frac, 1.0), 0.0)
        remaining = quantity
        tp1_qty = min(desired_tp1_qty, remaining)
        remaining -= tp1_qty
        tp2_qty = min(desired_tp2_qty, remaining)
        remaining -= tp2_qty
        runner_qty = min(desired_runner_qty, remaining)

        results = OrderExecutionResult(None, None, None, None)

        results.stop_loss = self._place_stop_loss(symbol, side, quantity, stop_loss_price)
        results.take_profit_1 = self._place_take_profit(symbol, side, tp1_qty, tp1_price, current_price, label="TP1")
        results.take_profit_2 = self._place_take_profit(symbol, side, tp2_qty, tp2_price, current_price, label="TP2")
        if enable_trailing and trail_price:
            results.trailing_stop = self._place_trailing_stop(
                symbol,
                side,
                runner_qty or quantity,
                float(trail_price),
                current_price
            )
        return results

    def _position_idx_for_side(self, side: str) -> int:
        """Bybit hedge mode requires positionIdx: 1 for LONG, 2 for SHORT; 0 for one-way."""
        side_l = (side or '').lower()
        if side_l == 'long':
            return 1
        if side_l == 'short':
            return 2
        return 0

    def _place_stop_loss(self, symbol: str, side: str, qty: float, stop_price: float) -> Dict[str, Any] | None:
        if qty <= 0:
            return None
        opposite = 'sell' if side == 'long' else 'buy'
        params = {
            'reduceOnly': True,
            'stopLossPrice': float(stop_price),
            # Target the correct hedge leg on Bybit
            'positionIdx': self._position_idx_for_side(side),
            'category': 'linear',
        }
        return self._send_order(symbol, 'stopMarket', opposite, qty, None, params, order_label='Stop loss')

    def _place_take_profit(self, symbol: str, side: str, qty: float, tp_price: float, current_price: float, *, label: str = "TP") -> Dict[str, Any] | None:
        if qty <= 0:
            return None
        opposite = 'sell' if side == 'long' else 'buy'
        # If TP is invalid relative to current price, convert to immediate reduce-only market order
        immediate_reduce = False
        try:
            if side == 'long' and tp_price <= current_price:
                immediate_reduce = True
            elif side == 'short' and tp_price >= current_price:
                immediate_reduce = True
        except Exception:
            immediate_reduce = False

        if immediate_reduce:
            params = {
                'reduceOnly': True,
                'positionIdx': self._position_idx_for_side(side),
                'category': 'linear',
            }
            return self._send_order(symbol, 'market', opposite, qty, None, params, order_label=f'{label} (immediate)')

        params = {
            'reduceOnly': True,
            'takeProfitPrice': float(tp_price),
            'positionIdx': self._position_idx_for_side(side),
            'category': 'linear',
        }
        return self._send_order(symbol, 'takeProfitMarket', opposite, qty, None, params, order_label=label)

    def _place_trailing_stop(
        self,
        symbol: str,
        side: str,
        qty: float,
        activation_price: float,
        current_price: float,
    ) -> Dict[str, Any] | None:
        if qty <= 0:
            return None
        opposite = 'sell' if side == 'long' else 'buy'
        # Derive a sensible callback percent from activation distance
        callback_pct = 0.0
        if current_price and activation_price:
            try:
                callback_pct = abs(activation_price - current_price) / current_price * 100.0
            except Exception:
                callback_pct = 0.0
        # Clamp to reasonable bounds (configurable later)
        callback_pct = max(0.1, min(callback_pct, 5.0))
        params = {
            'reduceOnly': True,
            'activationPrice': activation_price,
            'callbackRate': callback_pct,
            'positionIdx': self._position_idx_for_side(side),
            'category': 'linear',
        }
        return self._send_order(symbol, 'trailingStopMarket', opposite, qty, None, params, order_label='Trailing stop')

    def _send_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        qty: float,
        price: float | None,
        params: Dict[str, Any],
        *,
        order_label: str,
    ) -> Dict[str, Any] | None:
        """Create an order or log the payload in dry-run mode."""

        payload = {
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': qty,
            'price': price,
            'params': params,
        }
        if self.dry_run:
            logger.info("[DRY RUN] %s order payload: %s", order_label, payload)
            return payload

        # Visible console output for live submissions
        try:
            print(f"[LIVE] Submitting {order_label}: {side} {qty:g} {symbol} type={order_type} price={price if price is not None else 'MKT'} params={params}")
        except Exception:
            # Never let printing break the flow
            pass

        try:
            response = self.exchange.create_order(symbol, order_type, side, qty, price, params)
            logger.info("Submitted %s for %s @ %s", order_label, symbol, price or params.get('triggerPrice'))
            print(f"[LIVE] Submitted {order_label} for {symbol}")
            return response
        except Exception as exc:
            msg = str(exc)
            # Bybit-specific: 34040 "not modified" means TP/SL already set to this value; treat as benign
            if 'retCode' in msg and ('34040' in msg or 'not modified' in msg.lower()):
                logger.info("No change for %s on %s (already set): %s", order_label, symbol, msg)
                print(f"[LIVE] No change for {order_label} on {symbol} (already set)")
                return {'status': 'not_modified', 'error': msg, 'payload': payload}
            logger.error("Failed to submit %s for %s: %s", order_label, symbol, exc)
            # Surface the error so the CLI can report it to the user
            raise
