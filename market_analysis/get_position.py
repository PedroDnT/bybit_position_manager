"""
bybit_positions.py
------------------

This script fetches open derivatives positions from Bybit using the `ccxt` library.
It requires BYBIT_API_KEY and BYBIT_API_SECRET to be set in a .env file.

Example usage::

    python main.py

The program will fetch and display all open positions with their details including
symbol, side, size, entry price, mark price, and unrealized PnL.

Note: This script only reads position data and does not execute any trades.
"""

import os
import json
from typing import Dict, List, Any
import datetime as dt

try:
    import ccxt  # type: ignore
except ImportError as exc:
    raise ImportError(
        "ccxt library is required. Install it via 'pip install ccxt'."
    ) from exc

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError as exc:
    raise ImportError(
        "python-dotenv library is required. Install it via 'pip install python-dotenv'."
    ) from exc


def fetch_bybit_positions(exchange: ccxt.Exchange) -> List[Dict[str, Any]]:
    """Fetch current derivatives positions from Bybit.

    Returns a list of position dictionaries, or an empty list if
    API credentials are missing or there's an error.
    """
    try:
        # Fetch positions
        positions = exchange.fetch_positions()

        # Filter to only open positions
        open_positions = []
        for pos in positions:
            if pos["contracts"] and pos["contracts"] > 0:
                # Calculate percentage manually if not available
                percentage = pos.get("percentage")
                if (
                    percentage is None
                    and pos.get("unrealizedPnl") is not None
                    and pos.get("notional") is not None
                ):
                    try:
                        # Calculate percentage as (unrealizedPnl / notional) * 100
                        percentage = (
                            float(pos["unrealizedPnl"]) / float(pos["notional"])
                        ) * 100
                    except (ValueError, ZeroDivisionError):
                        percentage = None

                open_positions.append(
                    {
                        "symbol": pos["symbol"],
                        "side": pos["side"],
                        "size": pos["contracts"],
                        "notional": pos["notional"],
                        "entryPrice": pos["entryPrice"],
                        "markPrice": pos["markPrice"],
                        "unrealizedPnl": pos["unrealizedPnl"],
                        "percentage": percentage,
                        "liquidationPrice": pos.get("liquidationPrice"),
                        "leverage": pos.get("leverage"),
                        "marginMode": pos.get("marginMode"),
                        "marginType": pos.get("marginType"),
                        "maintenanceMargin": pos.get("maintenanceMargin"),
                        "initialMargin": pos.get("initialMargin"),
                        "marginRatio": pos.get("marginRatio"),
                    }
                )

        return open_positions

    except Exception as exc:
        print(f"Error: Failed to fetch Bybit positions: {exc}")
        return []


def fetch_bybit_account_balance(exchange: ccxt.Exchange) -> Dict[str, Any]:
    """Fetch account balance information from Bybit.

    Returns a dictionary with account balance details.
    """
    try:
        # Fetch account balance
        balance = exchange.fetch_balance()

        # Extract relevant balance information
        account_info = {
            "total_equity": balance.get("info", {}).get("totalEquity"),
            "total_wallet_balance": balance.get("info", {}).get("totalWalletBalance"),
            "total_unrealized_pnl": balance.get("info", {}).get("totalUnrealizedPnl"),
            "total_margin_balance": balance.get("info", {}).get("totalMarginBalance"),
            "total_initial_margin": balance.get("info", {}).get("totalInitialMargin"),
            "total_maintenance_margin": balance.get("info", {}).get(
                "totalMaintenanceMargin"
            ),
            "total_position_margin": balance.get("info", {}).get("totalPositionMargin"),
            "total_order_margin": balance.get("info", {}).get("totalOrderMargin"),
            "available_balance": balance.get("info", {}).get("availableBalance"),
            "used_margin": balance.get("info", {}).get("usedMargin"),
            "free_margin": balance.get("info", {}).get("freeMargin"),
        }

        return account_info

    except Exception as exc:
        print(f"Error: Failed to fetch Bybit account balance: {exc}")
        return {}


def fetch_bybit_account_metrics(exchange: ccxt.Exchange) -> Dict[str, Any]:
    """Return enriched Bybit account metrics including today's PnL."""

    account_info = fetch_bybit_account_balance(exchange)

    # Parse basic balance fields with safe fallbacks
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    metrics: Dict[str, Any] = {
        "total_equity": _to_float(account_info.get("total_equity")),
        "total_wallet_balance": _to_float(account_info.get("total_wallet_balance")),
        "total_unrealized_pnl": _to_float(account_info.get("total_unrealized_pnl")),
        "available_balance": _to_float(account_info.get("available_balance")),
        "total_margin_balance": _to_float(account_info.get("total_margin_balance")),
        "total_initial_margin": _to_float(account_info.get("total_initial_margin")),
    }

    # Attempt to pull today's realized PnL via the ledger endpoint
    today_realized = 0.0
    today_fees = 0.0
    try:
        start_of_day = dt.datetime.now(dt.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        since = int(start_of_day.timestamp() * 1000)
        ledger_entries = exchange.fetch_ledger(since=since, limit=200)

        for entry in ledger_entries or []:
            entry_type = entry.get("type") or entry.get("info", {}).get("type")
            amount = _to_float(entry.get("amount"))

            if entry_type in {"realizedpnl", "pnl", "settlement"}:
                today_realized += amount
            elif entry_type in {"fee", "commission"}:
                today_fees += amount

    except Exception as exc:
        # Ledger access is optional – log and continue with zeros
        print(f"Warning: unable to fetch ledger entries for PnL calculation: {exc}")

    metrics["todays_realized_pnl"] = today_realized
    metrics["todays_fees"] = today_fees
    metrics["todays_total_pnl"] = today_realized + metrics["total_unrealized_pnl"]

    return metrics
