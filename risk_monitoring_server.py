#!/usr/bin/env python3
"""
Real-Time Risk Monitoring Server
--------------------------------
Enhanced FastAPI server that integrates with the position risk manager
to provide real-time risk monitoring data with WebSocket streaming.

Features:
- Real-time position risk analysis
- WebSocket streaming for live updates
- Comprehensive risk metrics and thresholds
- Security and performance optimizations
- Connection stability and error handling
"""

import json
import asyncio
import logging
import datetime as dt
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer
import uvicorn

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import risk management modules with fallback
try:
    from market_analysis.position_risk_manager import PositionRiskManager
    from market_analysis.config import settings
except ImportError as e:
    logger.warning(f"Could not import risk management modules: {e}")
    PositionRiskManager = None
    settings = {}

# Global variables for risk manager and data
risk_manager: Optional[PositionRiskManager] = None
latest_risk_data: Dict[str, Any] = {}
clients: List[WebSocket] = []
data_update_task: Optional[asyncio.Task] = None

# Security
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global risk_manager, data_update_task
    
    try:
        # Initialize risk manager
        logger.info("Initializing Position Risk Manager...")
        risk_manager = PositionRiskManager(sandbox=False)
        
        # Start background data update task
        logger.info("Starting background data update task...")
        data_update_task = asyncio.create_task(update_risk_data_periodically())
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize risk manager: {e}")
        # Continue with mock data if real system fails
        yield
    finally:
        # Cleanup
        if data_update_task:
            data_update_task.cancel()
            try:
                await data_update_task
            except asyncio.CancelledError:
                pass
        logger.info("Risk monitoring server shutdown complete")

app = FastAPI(
    title="Real-Time Risk Monitoring API",
    description="Advanced risk monitoring system with real-time WebSocket streaming",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_current_risk_data() -> Dict[str, Any]:
    """Get current risk data from the risk manager with enhanced calculations."""
    global risk_manager, latest_risk_data
    
    if not risk_manager:
        logger.warning("Risk manager not initialized, using mock data")
        return get_mock_risk_data()
    
    try:
        # Run risk analysis
        analysis_result = await asyncio.get_event_loop().run_in_executor(
            None, risk_manager.analyze_all_positions
        )
        
        # Get the analysis results - analyze_all_positions returns positions as risk analysis dict
        risk_analysis = analysis_result.get('positions', {})
        positions = risk_manager.positions  # Get raw positions from the manager
        account_metrics = analysis_result.get('account', {})
        
        # Format data for frontend with enhanced risk calculations
        formatted_positions = format_positions_for_frontend(positions, risk_analysis)
        total_unrealized_pnl = 0.0
        total_notional = 0.0
        process_time = dt.datetime.now(dt.timezone.utc)
        raw_positions = {p.get("symbol", ""): p for p in positions if p.get("symbol")}

        for symbol, details in formatted_positions.items():
            raw_position = dict(raw_positions.get(symbol, {}))

            # Normalise naming differences from upstream sources
            if "liquidationPrice" not in raw_position and raw_position.get("liqPrice") is not None:
                raw_position["liquidationPrice"] = raw_position.get("liqPrice")
            if "markPrice" not in raw_position:
                raw_position["markPrice"] = raw_position.get("mark_price") or details.get("mark_price")
            if "percentage" not in raw_position:
                raw_position["percentage"] = raw_position.get("pnlPct") or details.get("percentage") or 0
            if "leverage" not in raw_position:
                raw_position["leverage"] = details.get("leverage") or raw_position.get("lever") or 0

            risk_score = calculate_position_risk_score(raw_position)

            details["risk_score"] = risk_score
            details["timestamp"] = process_time.isoformat()

            unrealized = details.get("unrealized_pnl")
            if unrealized is None:
                unrealized = raw_position.get("unrealizedPnl")
            total_unrealized_pnl += float(unrealized or 0.0)

            notional = details.get("notional")
            if notional is None:
                notional = raw_position.get("notional")
            if notional is None:
                entry_price = details.get("entry_price") or raw_position.get("avgPrice") or 0.0
                size = details.get("size") or details.get("position_size") or raw_position.get("size") or 0.0
                try:
                    notional = float(entry_price) * float(size)
                except (TypeError, ValueError):
                    notional = 0.0
            details["notional"] = float(notional or 0.0)
            total_notional += float(details["notional"])

        # Calculate portfolio metrics
        portfolio_analysis = risk_analysis.get("portfolio", {}) if isinstance(risk_analysis, dict) else {}
        if not isinstance(portfolio_analysis, dict):
            portfolio_analysis = {}

        total_risk_if_all_sl_hit = float(portfolio_analysis.get("total_risk_if_all_sl_hit", 0.0) or 0.0)
        total_reward_if_all_tp_hit = float(portfolio_analysis.get("total_reward_if_all_tp_hit", 0.0) or 0.0)
        portfolio_pnl_pct = (total_unrealized_pnl / total_notional * 100) if total_notional > 0 else 0.0
        risk_reward_ratio = portfolio_analysis.get("portfolio_risk_reward_ratio")
        if risk_reward_ratio is None:
            risk_reward_ratio = (total_reward_if_all_tp_hit / total_risk_if_all_sl_hit) if total_risk_if_all_sl_hit else 0.0
        risk_pct_of_notional = portfolio_analysis.get("risk_pct_of_notional")
        if risk_pct_of_notional is None:
            risk_pct_of_notional = (total_risk_if_all_sl_hit / total_notional * 100) if total_notional else 0.0

        positions_at_risk = portfolio_analysis.get("positions_at_risk")
        if not positions_at_risk:
            positions_at_risk = [
                symbol
                for symbol, data in formatted_positions.items()
                if data.get("risk_level") in {"CRITICAL", "HIGH"}
            ]

        portfolio_data = {
            "total_positions": len(formatted_positions),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_unrealized_pnl_pct": round(portfolio_pnl_pct, 2),
            "total_notional": round(total_notional, 2),
            "total_risk_if_all_sl_hit": round(total_risk_if_all_sl_hit, 2),
            "total_reward_if_all_tp_hit": round(total_reward_if_all_tp_hit, 2),
            "portfolio_risk_reward_ratio": round(risk_reward_ratio, 2) if isinstance(risk_reward_ratio, (int, float)) else 0.0,
            "risk_pct_of_notional": round(risk_pct_of_notional, 2),
            "positions_at_risk": positions_at_risk,
            "portfolio_drawdown_pct": portfolio_analysis.get("portfolio_drawdown_pct"),
            "portfolio_risk_utilization": portfolio_analysis.get("portfolio_risk_utilization"),
            "timestamp": process_time.isoformat()
        }

        # Derive overall system health
        health_status = "healthy"
        if any(data.get("risk_level") == "CRITICAL" for data in formatted_positions.values()):
            health_status = "critical"
        elif any(data.get("risk_level") == "HIGH" for data in formatted_positions.values()):
            health_status = "warning"
        elif any(data.get("risk_level") == "MEDIUM" for data in formatted_positions.values()):
            health_status = "degraded"

        # Generate risk alerts
        alerts = generate_risk_alerts(formatted_positions, portfolio_data)

        # Prepare complete data structure
        formatted_data = {
            "positions": formatted_positions,
            "portfolio": portfolio_data,
            "account": account_metrics,
            "alerts": alerts,
            "risk_thresholds": get_risk_thresholds(),
            "data_freshness": {
                "last_update": process_time.isoformat(),
                "update_interval": 5,  # seconds
                "status": "fresh"
            },
            "connection_status": "connected",
            "health_status": health_status,
            "timestamp": process_time.isoformat()
        }
        
        latest_risk_data = formatted_data
        return formatted_data
        
    except Exception as e:
        logger.error(f"Error getting risk data: {e}")
        # Update connection status on error
        if latest_risk_data:
            latest_risk_data["connection_status"] = "error"
            latest_risk_data["data_freshness"]["status"] = "stale"
        return latest_risk_data if latest_risk_data else get_mock_risk_data()

def format_positions_for_frontend(positions: List[Dict], risk_analysis: Dict) -> Dict[str, Any]:
    """Format position data for frontend consumption."""
    formatted_positions = {}

    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    for position in positions:
        symbol = position.get("symbol", "")
        analysis = risk_analysis.get(symbol, {})
        
        # Calculate risk indicators
        risk_level = calculate_risk_level(position, analysis)
        
        formatted_positions[symbol] = {
            "symbol": symbol,
            "side": position.get("side", ""),
            "size": safe_float(position.get("size")),
            "position_size": safe_float(position.get("size")),
            "entry_price": safe_float(position.get("avgPrice", position.get("entryPrice"))),
            "mark_price": safe_float(position.get("markPrice", position.get("indexPrice"))),
            "current_price": safe_float(position.get("markPrice", position.get("indexPrice"))),
            "unrealized_pnl": safe_float(position.get("unrealizedPnl")),
            "current_pnl": safe_float(position.get("unrealizedPnl")),
            "percentage": safe_float(position.get("percentage")),
            "current_pnl_pct": safe_float(position.get("percentage")),
            "leverage": safe_float(position.get("leverage", 1.0), 1.0),
            "notional": safe_float(position.get("notional")),
            
            # Risk analysis data
            "stop_loss": analysis.get("recommended_sl"),
            "take_profit": analysis.get("recommended_tp"),
            "tp1": analysis.get("recommended_tp"),
            "tp2": analysis.get("secondary_tp"),
            "trail_stop_suggestion": analysis.get("trailing_stop"),
            "risk_amount": analysis.get("dollar_risk", 0),
            "reward_amount": analysis.get("dollar_reward", 0),
            "risk_reward_ratio": analysis.get("risk_reward_ratio", 0),
            "regime_score": analysis.get("regime_score"),
            "confidence_factors": analysis.get("confidence_factors", []),
            "action_required": analysis.get("action_required"),
            
            # Risk indicators
            "risk_level": risk_level,
            "position_health": analysis.get("position_health", "UNKNOWN"),
            "confidence_score": analysis.get("confidence_score", 0),
            "volatility_metrics": analysis.get("volatility_metrics", {}),
            
            # Liquidation analysis
            "liquidation_price": position.get("liqPrice") or analysis.get("liquidation_price"),
            "liquidation_buffer": analysis.get("liquidation_buffer", {}),
            
            # Timestamps
            "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    
    return formatted_positions

def calculate_risk_level(position: Dict, analysis: Dict) -> str:
    """Calculate risk level based on position and analysis data."""
    health = analysis.get("position_health", "UNKNOWN")
    confidence = analysis.get("confidence_score", 0)
    pnl_pct = float(position.get("percentage", 0))
    
    if health == "CRITICAL":
        return "CRITICAL"
    elif health == "WARNING":
        return "HIGH"
    elif confidence < 0 or pnl_pct < -5:
        return "MEDIUM"
    elif confidence >= 2 and pnl_pct > 0:
        return "LOW"
    else:
        return "MEDIUM"

def get_risk_thresholds() -> Dict[str, Any]:
    """Get risk threshold configuration."""
    risk_cfg = settings.get("risk", {})
    
    return {
        "critical_pnl_threshold": -10.0,  # -10% PnL
        "warning_pnl_threshold": -5.0,   # -5% PnL
        "high_risk_threshold": 0.8,      # 80% of risk budget
        "medium_risk_threshold": 0.6,    # 60% of risk budget
        "max_portfolio_risk": risk_cfg.get("max_equity_risk_frac", 0.01) * 100,  # 1%
        "min_confidence_score": 0,
        "max_leverage": 20,
        "liquidation_buffer_min": risk_cfg.get("liquidation_buffer_multiple", 2.0),
        "critical_loss": -5.0,  # -5% loss
        "warning_loss": -3.0,   # -3% loss
        "high_leverage": 10.0,  # 10x leverage
        "liquidation_distance": 15.0,  # 15% from liquidation
        "portfolio_risk": 20.0,  # 20% of portfolio at risk
        "position_size": 30.0,  # 30% position concentration
        "volatility_high": 0.05,  # 5% daily volatility
        "margin_ratio": 80.0,   # 80% margin usage
    }

def calculate_position_risk_score(position: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive risk score for a position (0-100 scale)."""
    risk_factors = {}
    weights = {
        "pnl": 0.3,
        "leverage": 0.25,
        "liquidation": 0.25,
        "volatility": 0.2
    }
    
    # PnL Risk (0-100)
    pnl_pct = position.get("percentage", 0)
    if pnl_pct <= -5:
        pnl_score = 100
    elif pnl_pct <= -3:
        pnl_score = 70
    elif pnl_pct <= -1:
        pnl_score = 40
    elif pnl_pct >= 5:
        pnl_score = 0
    else:
        pnl_score = max(0, 30 - (pnl_pct * 6))
    
    risk_factors["pnl"] = {
        "score": pnl_score,
        "value": pnl_pct,
        "description": f"P&L: {pnl_pct:.2f}%"
    }
    
    # Leverage Risk (0-100)
    leverage = position.get("leverage", 1)
    leverage_score = min(100, max(0, (leverage - 1) * 8))
    risk_factors["leverage"] = {
        "score": leverage_score,
        "value": leverage,
        "description": f"Leverage: {leverage}x"
    }
    
    # Liquidation Distance Risk (0-100)
    liquidation_price = position.get("liquidationPrice")
    mark_price = position.get("markPrice")
    if liquidation_price and mark_price:
        distance_pct = abs(liquidation_price - mark_price) / mark_price * 100
        if distance_pct <= 5:
            liquidation_score = 100
        elif distance_pct <= 15:
            liquidation_score = 70
        elif distance_pct <= 30:
            liquidation_score = 40
        else:
            liquidation_score = max(0, 40 - (distance_pct - 30))
        
        risk_factors["liquidation"] = {
            "score": liquidation_score,
            "value": distance_pct,
            "description": f"Liquidation distance: {distance_pct:.1f}%"
        }
    else:
        risk_factors["liquidation"] = {
            "score": 50,
            "value": None,
            "description": "Liquidation distance: Unknown"
        }
    
    # Volatility Risk (placeholder - would need historical data)
    volatility_score = 30  # Default moderate risk
    risk_factors["volatility"] = {
        "score": volatility_score,
        "value": None,
        "description": "Volatility: Moderate"
    }
    
    # Calculate weighted overall score
    total_score = sum(
        risk_factors[factor]["score"] * weights[factor] 
        for factor in weights.keys()
    )
    
    # Determine risk level
    if total_score >= 75:
        risk_level = "CRITICAL"
        risk_color = "#ef4444"  # red
    elif total_score >= 55:
        risk_level = "HIGH"
        risk_color = "#f97316"  # orange
    elif total_score >= 35:
        risk_level = "MEDIUM"
        risk_color = "#facc15"  # yellow
    else:
        risk_level = "LOW"
        risk_color = "#22c55e"  # green
    
    return {
        "overall_score": round(total_score, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "severity_score": round(total_score, 1),
        "factors": risk_factors,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
    }

def generate_risk_alerts(positions: Dict[str, Any], portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate risk alerts based on current positions and portfolio state."""
    alerts = []
    current_time = dt.datetime.now(dt.timezone.utc)

    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def alert_payload(**kwargs: Any) -> Dict[str, Any]:
        payload = {
            "timestamp": current_time.isoformat(),
            **kwargs,
        }
        # Mirror severity into alert_type for frontend compatibility
        if "severity" in payload and "alert_type" not in payload:
            payload["alert_type"] = payload["severity"]
        return payload
    
    # Portfolio-level alerts
    if portfolio:
        total_pnl_pct = portfolio.get("total_unrealized_pnl_pct", 0)
        
        if total_pnl_pct <= -5:
            alerts.append(alert_payload(
                id=f"portfolio_critical_{int(current_time.timestamp())}",
                type="portfolio_loss",
                severity="critical",
                title="Critical Portfolio Loss",
                message=f"Portfolio loss: {total_pnl_pct:.2f}%",
                value=total_pnl_pct,
                threshold=-5.0,
                action="Consider reducing position sizes or implementing stop losses",
                severity_score=9,
            ))
        elif total_pnl_pct <= -3:
            alerts.append(alert_payload(
                id=f"portfolio_warning_{int(current_time.timestamp())}",
                type="portfolio_loss",
                severity="warning",
                title="Portfolio Loss Warning",
                message=f"Portfolio loss: {total_pnl_pct:.2f}%",
                value=total_pnl_pct,
                threshold=-3.0,
                action="Monitor positions closely",
                severity_score=6,
            ))
    
    # Position-level alerts
    for symbol, position in positions.items():
        # High leverage alert
        leverage = safe_float(position.get("leverage", 1))
        if leverage >= 10:
            alerts.append(alert_payload(
                id=f"leverage_{symbol}_{int(current_time.timestamp())}",
                type="high_leverage",
                severity="warning",
                title=f"High Leverage - {symbol}",
                message=f"Leverage: {leverage}x",
                symbol=symbol,
                value=leverage,
                threshold=10.0,
                action="Consider reducing leverage",
                severity_score=6,
            ))
        
        # Liquidation risk alert
        liquidation_price = position.get("liquidation_price") or position.get("liquidationPrice") or position.get("liqPrice")
        mark_price = position.get("mark_price") or position.get("markPrice") or position.get("indexPrice")
        if liquidation_price and mark_price:
            distance_pct = safe_float(abs(liquidation_price - mark_price) / mark_price * 100)
            if distance_pct <= 15:
                severity = "critical" if distance_pct <= 5 else "warning"
                severity_score = 9 if severity == "critical" else 6
                alerts.append(alert_payload(
                    id=f"liquidation_{symbol}_{int(current_time.timestamp())}",
                    type="liquidation_risk",
                    severity=severity,
                    title=f"Liquidation Risk - {symbol}",
                    message=f"Distance to liquidation: {distance_pct:.1f}%",
                    symbol=symbol,
                    value=distance_pct,
                    threshold=15.0,
                    action="Add margin or reduce position size",
                    severity_score=severity_score,
                ))
        
        # Large position loss alert
        pnl_pct = safe_float(position.get("percentage") or position.get("current_pnl_pct"))
        if pnl_pct <= -5:
            alerts.append(alert_payload(
                id=f"position_loss_{symbol}_{int(current_time.timestamp())}",
                type="position_loss",
                severity="critical",
                title=f"Large Position Loss - {symbol}",
                message=f"Position loss: {pnl_pct:.2f}%",
                symbol=symbol,
                value=pnl_pct,
                threshold=-5.0,
                action="Consider closing position or setting stop loss",
                severity_score=8,
            ))
    
    return alerts

def get_mock_risk_data() -> Dict[str, Any]:
    """Fallback mock data when real system is unavailable."""
    now = dt.datetime.now(dt.timezone.utc)
    positions = {
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "size": 0.5,
            "position_size": 0.5,
            "entry_price": 42000.0,
            "mark_price": 43500.0,
            "current_price": 43500.0,
            "unrealized_pnl": 750.0,
            "current_pnl": 750.0,
            "percentage": 3.57,
            "current_pnl_pct": 3.57,
            "leverage": 3.0,
            "notional": 21750.0,
            "stop_loss": 40000.0,
            "take_profit": 46000.0,
            "tp1": 46000.0,
            "risk_amount": 1000.0,
            "reward_amount": 2000.0,
            "risk_reward_ratio": 2.0,
            "risk_level": "LOW",
            "position_health": "HEALTHY",
            "confidence_score": 3,
            "regime_score": 3,
            "confidence_factors": ["Trend", "Momentum"],
            "liquidation_price": 38000.0,
            "last_updated": now.isoformat(),
        },
        "ETH/USDT:USDT": {
            "symbol": "ETH/USDT:USDT",
            "side": "short",
            "size": 2.0,
            "position_size": 2.0,
            "entry_price": 2600.0,
            "mark_price": 2550.0,
            "current_price": 2550.0,
            "unrealized_pnl": 100.0,
            "current_pnl": 100.0,
            "percentage": 1.92,
            "current_pnl_pct": 1.92,
            "leverage": 5.0,
            "notional": 5200.0,
            "stop_loss": 2700.0,
            "take_profit": 2400.0,
            "tp1": 2400.0,
            "risk_amount": 300.0,
            "reward_amount": 400.0,
            "risk_reward_ratio": 1.33,
            "risk_level": "MEDIUM",
            "position_health": "WARNING",
            "confidence_score": 2,
            "liquidation_price": 3000.0,
            "last_updated": now.isoformat(),
        },
    }

    for pos in positions.values():
        seed = {
            "percentage": pos.get("percentage"),
            "leverage": pos.get("leverage"),
            "liquidationPrice": pos.get("liquidation_price"),
            "markPrice": pos.get("mark_price"),
        }
        pos["risk_score"] = calculate_position_risk_score(seed)
        pos["timestamp"] = now.isoformat()

    portfolio = {
        "total_positions": len(positions),
        "total_notional": 26950.0,
        "total_unrealized_pnl": 850.0,
        "total_unrealized_pnl_pct": 3.15,
        "total_risk_if_all_sl_hit": 1300.0,
        "total_reward_if_all_tp_hit": 2400.0,
        "portfolio_risk_reward_ratio": 1.85,
        "risk_pct_of_notional": 4.82,
        "positions_at_risk": [symbol for symbol, data in positions.items() if data.get("risk_level") in {"MEDIUM", "HIGH", "CRITICAL"}],
        "timestamp": now.isoformat(),
    }

    alerts = generate_risk_alerts(positions, portfolio)

    return {
        "timestamp": now.isoformat(),
        "status": "mock",
        "positions": positions,
        "portfolio": portfolio,
        "account": {
            "balance": 50000.0,
            "equity": 50850.0,
            "margin_used": 26950.0,
            "margin_free": 23100.0,
            "margin_ratio": 34.1,
        },
        "alerts": alerts,
        "risk_thresholds": get_risk_thresholds(),
        "connection_status": "mock",
        "health_status": "warning" if portfolio["positions_at_risk"] else "healthy",
        "data_freshness": {
            "last_update": now.isoformat(),
            "update_interval": 5,
            "status": "mock",
        },
    }

async def update_risk_data_periodically():
    """Background task to update risk data periodically."""
    while True:
        try:
            # Update risk data every 5 seconds
            await asyncio.sleep(5)
            
            # Get fresh data
            fresh_data = await get_current_risk_data()
            
            # Broadcast to all connected WebSocket clients
            if clients:
                await broadcast_to_clients(fresh_data)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic update: {e}")
            await asyncio.sleep(10)  # Wait longer on error

async def broadcast_to_clients(data: Dict[str, Any]):
    """Broadcast data to all connected WebSocket clients."""
    if not clients:
        return
    
    message = json.dumps(data, default=str)
    disconnected_clients = []
    
    for client in clients:
        try:
            await client.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        if client in clients:
            clients.remove(client)

# API Endpoints
@app.get("/")
async def root():
    """Serve the main risk monitoring dashboard."""
    return FileResponse("templates/risk_dashboard.html")

@app.get("/api/risk-analysis")
async def get_risk_analysis():
    """Get current risk analysis data."""
    try:
        data = await get_current_risk_data()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error in risk analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    global risk_manager
    
    health_status = {
        "status": "healthy",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "risk_manager_available": risk_manager is not None,
        "active_websocket_connections": len(clients),
        "data_freshness": "live" if risk_manager else "mock",
        "connected_clients": len(clients)
    }
    
    return JSONResponse(content=health_status)

@app.websocket("/ws/risk-stream")
async def websocket_risk_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time risk data streaming."""
    await websocket.accept()
    clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(clients)}")
    
    try:
        # Send initial data
        initial_data = await get_current_risk_data()
        await websocket.send_text(json.dumps(initial_data, default=str))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong, refresh commands, etc.)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if not message:
                    continue

                stripped = message.strip()
                lowered = stripped.lower()
                payload: Dict[str, Any] = {}
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        payload = {}

                command = (payload.get("type") or payload.get("command") or "").lower()

                if lowered == "ping" or command == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
                    }))
                    continue

                if lowered == "refresh" or command == "refresh":
                    fresh_snapshot = await get_current_risk_data()
                    await websocket.send_text(json.dumps(fresh_snapshot, default=str))
                    continue
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
                }))
            except WebSocketDisconnect:
                break
            except Exception as message_error:
                logger.warning(f"WebSocket message handling error: {message_error}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in clients:
            clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(clients)}")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting Real-Time Risk Monitoring Server")
    print("=" * 80)
    print("📊 Features:")
    print("  • Real-time position risk analysis")
    print("  • WebSocket streaming for live updates")
    print("  • Comprehensive risk metrics and thresholds")
    print("  • Security and performance optimizations")
    print("  • Connection stability and error handling")
    print("=" * 80)
    print("🌐 Endpoints:")
    print("  • Health check: http://localhost:8001/healthz")
    print("  • Risk analysis: http://localhost:8001/api/risk-analysis")
    print("  • WebSocket stream: ws://localhost:8001/ws/risk-stream")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )
