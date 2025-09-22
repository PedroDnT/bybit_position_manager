import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .position_risk_manager import PositionRiskManager
from .garch_vol_triggers import get_live_price_bybit, dynamic_levels_from_state

# Paths
ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_HTML = ROOT / "assets" / "dashboard.html"
RISK_JSON_PATH = ROOT / "risk_analysis.json"

app = FastAPI(title="Bybit Position Manager UI", version="1.0.0")

# Allow local dev frontends if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
risk_state: Dict[str, Any] = {"positions": {}, "portfolio": {}, "account": {}}
manager: PositionRiskManager | None = None
clients: List[WebSocket] = []
broadcast_lock = asyncio.Lock()


@app.get("/")
async def index():
    return FileResponse(str(DASHBOARD_HTML))


@app.get("/api/analysis")
async def api_analysis():
    return JSONResponse(risk_state)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        # Send initial snapshot
        await ws.send_text(json.dumps(risk_state))
        while True:
            await asyncio.sleep(30)
            try:
                await ws.send_text(json.dumps({"type": "ping"}))
            except Exception:
                # Client closed or cannot accept messages anymore; break loop
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            clients.remove(ws)
        except ValueError:
            pass


async def _broadcast_update():
    if not clients:
        return
    async with broadcast_lock:
        payload = json.dumps(risk_state)
        living: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(payload)
                living.append(ws)
            except Exception:
                # drop broken connections
                continue
        clients[:] = living


async def _load_fallback_json():
    global risk_state
    if RISK_JSON_PATH.exists():
        try:
            with RISK_JSON_PATH.open("r") as f:
                data = json.load(f)
            # normalize to expected structure
            if isinstance(data, dict) and "positions" in data:
                risk_state = data
            elif isinstance(data, dict):
                risk_state = {"positions": data, "portfolio": {}, "account": {}}
        except Exception:
            pass


async def init_manager_and_analyze():
    global manager, risk_state
    try:
        # Construct manager (may raise if API keys missing)
        manager = await asyncio.to_thread(PositionRiskManager)
        await asyncio.to_thread(manager.fetch_positions)
        await asyncio.to_thread(manager.analyze_all_positions)
        # Save JSON to disk via existing __main__ behavior would normally happen elsewhere;
        # here we just take the in-memory snapshot
        risk_state = {
            "positions": manager.risk_analysis,
            "portfolio": manager._calculate_portfolio_metrics(),
            "account": manager.account_metrics,
        }
    except Exception as e:
        # Fallback to static JSON if live manager cannot start
        manager = None
        await _load_fallback_json()


async def realtime_update_loop(interval: int = 5):
    """Periodic loop to update dynamic levels and live prices, broadcasting to clients."""
    global risk_state
    # Ensure initial state exists
    if not risk_state["positions"]:
        await init_manager_and_analyze()
        await _broadcast_update()
    while True:
        try:
            if manager and manager.positions and manager.risk_analysis:
                # Update dynamic fields similar to monitor_positions but non-blocking
                for pos in manager.positions:
                    sym = pos['symbol']
                    analysis = manager.risk_analysis.get(sym)
                    if not analysis:
                        continue
                    # blocking IO -> offload
                    live = await asyncio.to_thread(get_live_price_bybit, manager.exchange, sym)
                    if live is None:
                        live = analysis.get('current_price')
                    sigma_H = analysis.get('sigma_H')
                    atr = analysis.get('atr')
                    side = analysis.get('side')
                    entry = analysis.get('entry_price')
                    if None in (live, sigma_H, atr, side, entry):
                        continue
                    dyn = dynamic_levels_from_state(
                        current_price=float(live),
                        entry_price=float(entry),
                        side=side,
                        sigma_H=float(sigma_H),
                        atr=float(atr),
                        base_k=float(analysis.get('k_multiplier', 2.0)),
                        base_m=float(analysis.get('m_multiplier', 3.0)),
                        cfg=manager.cfg,
                    )
                    # Update fields
                    analysis['current_price'] = float(live)
                    analysis['dynamic_stop_loss'] = float(dyn['SL'])
                    analysis['dynamic_take_profit'] = float(dyn['TP'])
                    analysis['dynamic_sl_distance'] = abs(float(entry) - float(dyn['SL']))
                    analysis['dynamic_tp_distance'] = abs(float(dyn['TP']) - float(entry))
                    analysis['dynamic_p_tp'] = float(dyn.get('p_tp', 0.0))
                    analysis['dynamic_reasons'] = dyn.get('reasons', [])
                # Refresh top-level snapshot
                risk_state = {
                    "positions": manager.risk_analysis,
                    "portfolio": manager._calculate_portfolio_metrics(),
                    "account": manager.account_metrics,
                }
            # Broadcast to clients
            await _broadcast_update()
        except Exception:
            # Do not crash the loop on transient errors
            pass
        await asyncio.sleep(interval)


@app.on_event("startup")
async def on_startup():
    # Initialize once and start update loop
    await init_manager_and_analyze()
    asyncio.create_task(realtime_update_loop())


# Optional: simple health check
@app.get("/healthz")
async def health():
    ok = bool(risk_state and risk_state.get("positions") is not None)
    return {"ok": ok}