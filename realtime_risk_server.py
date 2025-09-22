#!/usr/bin/env python3
"""
Real-Time Risk Reporting Server
===============================

This server provides real-time risk reporting that simulates live execution of risk manager files.
It continuously updates risk metrics, position analysis, and portfolio health indicators.

Features:
- Real-time position monitoring simulation
- Live risk metric calculations
- Dynamic volatility analysis
- Adaptive stop-loss updates
- Portfolio correlation tracking
- Real-time alerts and notifications
- WebSocket streaming for instant updates
"""

import asyncio
import json
import math
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Use built-in libraries for calculations to avoid import issues
import statistics

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskAlert:
    """Represents a risk alert."""
    id: str
    type: str
    severity: str
    symbol: str
    message: str
    timestamp: datetime
    value: float
    threshold: float
    action: str

@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics."""
    atr: float
    atr_pct: float
    garch_volatility: float
    har_volatility: float
    volatility_regime: str
    volatility_trend: str

@dataclass
class AdaptiveStopLoss:
    """Adaptive stop-loss configuration."""
    current_stop: Optional[float]
    suggested_stop: float
    trailing_stop: Optional[float]
    atr_multiplier: float
    volatility_adjustment: float
    risk_level: str
    needs_adjustment: bool
    adjustment_reason: str

app = FastAPI(title="Real-Time Risk Reporting Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
connected_clients: List[WebSocket] = []
risk_state: Dict[str, Any] = {}
last_update_time = datetime.now()

# Risk analysis configuration
RISK_CONFIG = {
    "update_interval": 2.0,  # seconds
    "price_volatility": 0.02,  # 2% price movement simulation
    "alert_thresholds": {
        "critical_pnl": -0.05,  # -5% PnL
        "warning_pnl": -0.03,   # -3% PnL
        "high_risk_ratio": 0.8,  # 80% risk utilization
        "liquidation_buffer": 0.1  # 10% liquidation buffer
    },
    "portfolio_limits": {
        "max_risk_per_position": 0.02,  # 2% per position
        "max_total_risk": 0.10,  # 10% total portfolio risk
        "correlation_threshold": 0.7  # 70% correlation limit
    }
}

# Sample positions data structure
SAMPLE_POSITIONS = {
    "BTC/USDT:USDT": {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 43500.0,
        "current_price": 44200.0,
        "position_size": 0.5,
        "notional": 22100.0,
        "leverage": 10.0,
        "unrealized_pnl": 350.0,
        "unrealized_pnl_pct": 1.58,
        "liquidation_price": 39150.0,
        "margin_ratio": 0.15,
        "last_update": datetime.now().isoformat()
    },
    "ETH/USDT:USDT": {
        "symbol": "ETH/USDT:USDT",
        "side": "long",
        "entry_price": 2650.0,
        "current_price": 2680.0,
        "position_size": 8.0,
        "notional": 21440.0,
        "leverage": 8.0,
        "unrealized_pnl": 240.0,
        "unrealized_pnl_pct": 1.13,
        "liquidation_price": 2318.75,
        "margin_ratio": 0.18,
        "last_update": datetime.now().isoformat()
    },
    "SOL/USDT:USDT": {
        "symbol": "SOL/USDT:USDT",
        "side": "short",
        "entry_price": 105.0,
        "current_price": 103.5,
        "position_size": -200.0,
        "notional": -20700.0,
        "leverage": 15.0,
        "unrealized_pnl": 300.0,
        "unrealized_pnl_pct": 1.43,
        "liquidation_price": 113.4,
        "margin_ratio": 0.12,
        "last_update": datetime.now().isoformat()
    }
}

class RealTimeRiskAnalyzer:
    """Real-time risk analysis engine that simulates live risk manager execution."""
    
    def __init__(self):
        self.positions = SAMPLE_POSITIONS.copy()
        self.portfolio_metrics = {}
        self.alerts = []
        self.volatility_models = {}
        self.correlation_matrix = {}
        self.last_analysis_time = datetime.now()
        
    def simulate_price_movement(self, current_price: float, volatility: float = 0.02) -> float:
        """Simulate realistic price movement with volatility."""
        # Generate random walk with mean reversion
        random_change = random.gauss(0, volatility)
        # Add some mean reversion
        mean_reversion = -0.1 * random_change if abs(random_change) > volatility else 0
        new_price = current_price * (1 + random_change + mean_reversion)
        return max(new_price, current_price * 0.95)  # Prevent extreme drops
    
    def calculate_atr(self, symbol: str, periods: int = 14) -> float:
        """Calculate Average True Range for volatility measurement."""
        # Simulate ATR based on current price and volatility
        current_price = self.positions[symbol]["current_price"]
        base_atr = current_price * 0.02  # 2% base ATR
        volatility_factor = random.uniform(0.8, 1.2)
        return base_atr * volatility_factor
    
    def calculate_adaptive_stop_loss(self, symbol: str) -> Dict[str, Any]:
        """Calculate adaptive stop-loss levels based on current market conditions."""
        position = self.positions[symbol]
        current_price = position["current_price"]
        entry_price = position["entry_price"]
        side = position["side"]
        atr = self.calculate_atr(symbol)
        
        # ATR multiplier based on volatility regime
        volatility_regime = "normal"  # Could be "low", "normal", "high"
        atr_multipliers = {"low": 1.5, "normal": 2.0, "high": 2.5}
        multiplier = atr_multipliers[volatility_regime]
        
        if side == "long":
            suggested_stop = current_price - (atr * multiplier)
            trailing_stop = max(suggested_stop, entry_price * 0.95)  # Never below 5% from entry
        else:
            suggested_stop = current_price + (atr * multiplier)
            trailing_stop = min(suggested_stop, entry_price * 1.05)  # Never above 5% from entry
        
        return {
            "suggested_stop": suggested_stop,
            "trailing_stop": trailing_stop,
            "atr": atr,
            "atr_multiplier": multiplier,
            "volatility_regime": volatility_regime,
            "needs_adjustment": abs(current_price - entry_price) / entry_price > 0.02
        }
    
    def calculate_risk_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for a position."""
        position = self.positions[symbol]
        current_price = position["current_price"]
        entry_price = position["entry_price"]
        position_size = position["position_size"]
        leverage = position["leverage"]
        
        # Calculate PnL
        if position["side"] == "long":
            pnl = (current_price - entry_price) * abs(position_size)
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - current_price) * abs(position_size)
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        # Update position with new values
        position["unrealized_pnl"] = pnl
        position["unrealized_pnl_pct"] = pnl_pct
        position["current_price"] = current_price
        position["last_update"] = datetime.now().isoformat()
        
        # Risk calculations
        notional = abs(position_size * current_price)
        margin_used = notional / leverage
        liquidation_distance = abs(current_price - position["liquidation_price"]) / current_price
        
        # Adaptive stop-loss
        adaptive_levels = self.calculate_adaptive_stop_loss(symbol)
        
        # Risk assessment
        risk_level = "LOW"
        if pnl_pct < -3:
            risk_level = "HIGH"
        elif pnl_pct < -1:
            risk_level = "MEDIUM"
        
        position_health = "HEALTHY"
        if liquidation_distance < 0.1:
            position_health = "CRITICAL"
        elif liquidation_distance < 0.2:
            position_health = "WARNING"
        elif pnl_pct > 2:
            position_health = "PROFITABLE"
        
        return {
            "symbol": symbol,
            "risk_level": risk_level,
            "position_health": position_health,
            "liquidation_distance": liquidation_distance,
            "margin_ratio": position["margin_ratio"],
            "dollar_risk": margin_used * 0.1,  # Assume 10% risk per position
            "dollar_reward": margin_used * 0.2,  # Assume 20% reward target
            "risk_reward_ratio": 2.0,
            "adaptive_stop_loss": adaptive_levels["suggested_stop"],
            "adaptive_trailing_stop": adaptive_levels["trailing_stop"],
            "atr": adaptive_levels["atr"],
            "volatility_regime": adaptive_levels["volatility_regime"],
            "needs_adjustment": adaptive_levels["needs_adjustment"],
            "confidence_score": random.uniform(0.6, 0.9),  # Simulated confidence
            "last_calculated": datetime.now().isoformat()
        }
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-wide risk metrics."""
        total_notional = sum(abs(pos["notional"]) for pos in self.positions.values())
        total_pnl = sum(pos["unrealized_pnl"] for pos in self.positions.values())
        total_pnl_pct = (total_pnl / total_notional * 100) if total_notional > 0 else 0
        
        # Risk utilization
        total_margin = sum(abs(pos["notional"]) / pos["leverage"] for pos in self.positions.values())
        risk_utilization = total_margin / 100000 if total_margin > 0 else 0  # Assume 100k account
        
        # Portfolio health
        portfolio_health = "HEALTHY"
        if total_pnl_pct < -5:
            portfolio_health = "CRITICAL"
        elif total_pnl_pct < -2:
            portfolio_health = "WARNING"
        elif total_pnl_pct > 3:
            portfolio_health = "EXCELLENT"
        
        # Correlation analysis (simplified)
        correlation_risk = "LOW"
        if len(self.positions) > 2:
            correlation_risk = random.choice(["LOW", "MEDIUM", "HIGH"])
        
        return {
            "total_positions": len(self.positions),
            "total_notional": total_notional,
            "total_unrealized_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "portfolio_health": portfolio_health,
            "risk_utilization": risk_utilization,
            "correlation_risk": correlation_risk,
            "diversification_score": random.uniform(0.7, 0.95),
            "sharpe_ratio": random.uniform(1.2, 2.5),
            "max_drawdown": random.uniform(0.02, 0.08),
            "var_95": total_notional * 0.03,  # 3% VaR
            "last_updated": datetime.now().isoformat()
        }
    
    def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate comprehensive real-time alerts based on current conditions."""
        alerts = []
        current_time = datetime.now()
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        # Portfolio-level alerts
        total_risk = portfolio_metrics["total_risk_exposure"]
        if total_risk > RISK_CONFIG["portfolio_limits"]["max_total_risk"]:
            alerts.append({
                "id": f"portfolio_risk_{int(time.time())}",
                "type": "CRITICAL",
                "category": "Portfolio Risk",
                "symbol": "PORTFOLIO",
                "message": f"Portfolio risk exposure {total_risk*100:.1f}% exceeds limit {RISK_CONFIG['portfolio_limits']['max_total_risk']*100:.1f}%",
                "timestamp": current_time.isoformat(),
                "action_required": True,
                "severity": "CRITICAL",
                "value": total_risk,
                "threshold": RISK_CONFIG["portfolio_limits"]["max_total_risk"],
                "recommended_action": "Reduce position sizes or close high-risk positions"
            })
        
        # Correlation risk alerts
        if portfolio_metrics["correlation_risk"] > RISK_CONFIG["portfolio_limits"]["correlation_threshold"]:
            alerts.append({
                "id": f"correlation_risk_{int(time.time())}",
                "type": "WARNING",
                "category": "Correlation Risk",
                "symbol": "PORTFOLIO",
                "message": f"High correlation detected: {portfolio_metrics['correlation_risk']*100:.1f}% - Diversification needed",
                "timestamp": current_time.isoformat(),
                "action_required": True,
                "severity": "MEDIUM",
                "value": portfolio_metrics["correlation_risk"],
                "threshold": RISK_CONFIG["portfolio_limits"]["correlation_threshold"],
                "recommended_action": "Consider diversifying positions across uncorrelated assets"
            })
        
        for symbol, position in self.positions.items():
            pnl_pct = position["unrealized_pnl_pct"]
            margin_ratio = position["margin_ratio"]
            
            # Enhanced PnL alerts with trend analysis
            if pnl_pct < RISK_CONFIG["alert_thresholds"]["critical_pnl"] * 100:
                alerts.append({
                    "id": f"critical_pnl_{symbol}_{int(time.time())}",
                    "type": "CRITICAL",
                    "category": "PnL Loss",
                    "symbol": symbol,
                    "message": f"{symbol}: CRITICAL LOSS {pnl_pct:.2f}% - IMMEDIATE ACTION REQUIRED",
                    "timestamp": current_time.isoformat(),
                    "action_required": True,
                    "severity": "CRITICAL",
                    "value": pnl_pct,
                    "threshold": RISK_CONFIG["alert_thresholds"]["critical_pnl"] * 100,
                    "recommended_action": "Close position immediately or add stop loss",
                    "urgency": "IMMEDIATE",
                    "sound_alert": True
                })
            elif pnl_pct < RISK_CONFIG["alert_thresholds"]["warning_pnl"] * 100:
                alerts.append({
                    "id": f"warning_pnl_{symbol}_{int(time.time())}",
                    "type": "WARNING",
                    "category": "PnL Loss",
                    "symbol": symbol,
                    "message": f"{symbol}: Loss warning {pnl_pct:.2f}% - Monitor position closely",
                    "timestamp": current_time.isoformat(),
                    "action_required": False,
                    "severity": "MEDIUM",
                    "value": pnl_pct,
                    "threshold": RISK_CONFIG["alert_thresholds"]["warning_pnl"] * 100,
                    "recommended_action": "Consider setting stop loss or reducing position size"
                })
            
            # Enhanced liquidation alerts with distance calculation
            liquidation_distance = abs(position["current_price"] - position["liquidation_price"]) / position["current_price"]
            if liquidation_distance < 0.05:  # Within 5%
                alerts.append({
                    "id": f"liquidation_imminent_{symbol}_{int(time.time())}",
                    "type": "CRITICAL",
                    "category": "Liquidation Risk",
                    "symbol": symbol,
                    "message": f"{symbol}: LIQUIDATION IMMINENT - Only {liquidation_distance*100:.1f}% away!",
                    "timestamp": current_time.isoformat(),
                    "action_required": True,
                    "severity": "CRITICAL",
                    "value": liquidation_distance,
                    "threshold": 0.05,
                    "recommended_action": "Add margin immediately or close position",
                    "urgency": "IMMEDIATE",
                    "sound_alert": True,
                    "liquidation_price": position["liquidation_price"],
                    "current_price": position["current_price"]
                })
            elif liquidation_distance < 0.15:  # Within 15%
                alerts.append({
                    "id": f"liquidation_risk_{symbol}_{int(time.time())}",
                    "type": "WARNING",
                    "category": "Liquidation Risk",
                    "symbol": symbol,
                    "message": f"{symbol}: Liquidation risk - {liquidation_distance*100:.1f}% from liquidation",
                    "timestamp": current_time.isoformat(),
                    "action_required": True,
                    "severity": "HIGH",
                    "value": liquidation_distance,
                    "threshold": 0.15,
                    "recommended_action": "Monitor closely and prepare to add margin",
                    "liquidation_price": position["liquidation_price"],
                    "current_price": position["current_price"]
                })
            
            # Margin ratio alerts
            if margin_ratio < 0.1:  # Below 10% margin
                alerts.append({
                    "id": f"low_margin_{symbol}_{int(time.time())}",
                    "type": "CRITICAL",
                    "category": "Margin Risk",
                    "symbol": symbol,
                    "message": f"{symbol}: Low margin ratio {margin_ratio*100:.1f}% - Add funds urgently",
                    "timestamp": current_time.isoformat(),
                    "action_required": True,
                    "severity": "CRITICAL",
                    "value": margin_ratio,
                    "threshold": 0.1,
                    "recommended_action": "Add margin or reduce position size immediately",
                    "urgency": "HIGH"
                })
            elif margin_ratio < 0.2:  # Below 20% margin
                alerts.append({
                    "id": f"margin_warning_{symbol}_{int(time.time())}",
                    "type": "WARNING",
                    "category": "Margin Risk",
                    "symbol": symbol,
                    "message": f"{symbol}: Margin ratio {margin_ratio*100:.1f}% getting low",
                    "timestamp": current_time.isoformat(),
                    "action_required": False,
                    "severity": "MEDIUM",
                    "value": margin_ratio,
                    "threshold": 0.2,
                    "recommended_action": "Consider adding margin or reducing leverage"
                })
            
            # Position size risk alerts
            position_risk = abs(position["notional"]) / 100000  # Assuming 100k portfolio
            if position_risk > RISK_CONFIG["portfolio_limits"]["max_risk_per_position"]:
                alerts.append({
                    "id": f"position_size_risk_{symbol}_{int(time.time())}",
                    "type": "WARNING",
                    "category": "Position Size",
                    "symbol": symbol,
                    "message": f"{symbol}: Position size {position_risk*100:.1f}% exceeds recommended {RISK_CONFIG['portfolio_limits']['max_risk_per_position']*100:.1f}%",
                    "timestamp": current_time.isoformat(),
                    "action_required": True,
                    "severity": "MEDIUM",
                    "value": position_risk,
                    "threshold": RISK_CONFIG["portfolio_limits"]["max_risk_per_position"],
                    "recommended_action": "Consider reducing position size for better risk management"
                })
        
        # Sort alerts by severity and timestamp
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 4), x["timestamp"]), reverse=True)
        
        return alerts
    
    async def update_positions(self):
        """Update all positions with new prices and recalculate metrics."""
        for symbol in self.positions:
            # Simulate price movement
            current_price = self.positions[symbol]["current_price"]
            new_price = self.simulate_price_movement(current_price, RISK_CONFIG["price_volatility"])
            self.positions[symbol]["current_price"] = new_price
        
        # Calculate risk metrics for all positions
        position_analyses = {}
        for symbol in self.positions:
            position_analyses[symbol] = self.calculate_risk_metrics(symbol)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        # Generate alerts
        alerts = self.generate_alerts()
        
        # Update global state
        global risk_state
        risk_state = {
            "timestamp": datetime.now().isoformat(),
            "positions": self.positions,
            "analysis": position_analyses,
            "portfolio": portfolio_metrics,
            "alerts": alerts,
            "system_status": "LIVE",
            "update_frequency": RISK_CONFIG["update_interval"],
            "data_freshness": "REAL_TIME"
        }
        
        self.last_analysis_time = datetime.now()

# Initialize the risk analyzer
risk_analyzer = RealTimeRiskAnalyzer()

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "service": "Real-Time Risk Reporting Server",
        "status": "LIVE",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/risk-stream",
            "api": "/api/risk-analysis",
            "health": "/healthz"
        },
        "update_interval": RISK_CONFIG["update_interval"],
        "last_update": risk_state.get("timestamp", "Not yet updated")
    }

@app.get("/api/risk-analysis")
async def get_risk_analysis():
    """Get current risk analysis snapshot."""
    return JSONResponse(risk_state)

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connected_clients": len(connected_clients),
        "last_update": risk_state.get("timestamp", "Never"),
        "positions_tracked": len(risk_state.get("positions", {}))
    }

@app.websocket("/ws/risk-stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time risk data streaming."""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        # Send initial data
        if risk_state:
            await websocket.send_text(json.dumps(risk_state))
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"Client removed. Total clients: {len(connected_clients)}")

async def broadcast_updates():
    """Broadcast updates to all connected WebSocket clients."""
    if not connected_clients:
        return
    
    message = json.dumps(risk_state)
    disconnected_clients = []
    
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        if client in connected_clients:
            connected_clients.remove(client)
    disconnected_clients = []
    
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        if client in connected_clients:
            connected_clients.remove(client)

async def real_time_update_loop():
    """Main loop for real-time risk analysis updates."""
    logger.info("Starting real-time update loop")
    
    while True:
        try:
            # Update risk analysis
            await risk_analyzer.update_positions()
            
            # Broadcast to connected clients
            await broadcast_updates()
            
            logger.info(f"Updated risk analysis for {len(risk_state.get('positions', {}))} positions")
            
        except Exception as e:
            logger.error(f"Error in update loop: {e}")
        
        # Wait for next update
        await asyncio.sleep(RISK_CONFIG["update_interval"])

@app.on_event("startup")
async def startup_event():
    """Initialize the real-time update loop on startup."""
    # Perform initial analysis
    await risk_analyzer.update_positions()
    
    # Start the real-time update loop
    asyncio.create_task(real_time_update_loop())
    
    logger.info("Real-Time Risk Reporting Server started successfully")

if __name__ == "__main__":
    print("🚀 Starting Real-Time Risk Manager Server...")
    print("📊 Real-time risk analysis and reporting")
    print("🔄 WebSocket streaming enabled")
    print("⚡ Live position monitoring active")
    print("\n🌐 Server will be available at: http://localhost:8001")
    print("📡 WebSocket endpoint: ws://localhost:8001/ws/risk-stream")
    print("📈 API endpoint: http://localhost:8001/api/risk-analysis")
    
    uvicorn.run(
        "realtime_risk_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )