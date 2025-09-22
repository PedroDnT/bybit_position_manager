"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { ChevronRight, TrendingUp, AlertTriangle, DollarSign, BarChart3, Bell, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import PortfolioOverviewPage from "./portfolio-overview/page"
import PositionsPage from "./positions/page"
import RiskAnalysisPage from "./risk-analysis/page"
import AlertsPage from "./alerts/page"
import PerformancePage from "./performance/page"

type PositionAnalysis = Record<string, any>

type PortfolioMetrics = {
  total_positions: number
  total_notional: number
  total_unrealized_pnl: number
  total_risk_if_all_sl_hit: number
  total_reward_if_all_tp_hit: number
  portfolio_risk_reward_ratio: number
  positions_at_risk?: string[]
  risk_pct_of_notional?: number
}

type RiskState = {
  timestamp?: string
  positions: Record<string, PositionAnalysis>
  portfolio?: PortfolioMetrics
  account?: Record<string, any>
}

function wsUrlFromHttp(httpUrl: string) {
  try {
    const u = new URL(httpUrl)
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:"
    return u.toString().replace(/\/$/, "")
  } catch {
    return httpUrl
  }
}

export default function RiskManagerTerminal() {
  const [activeSection, setActiveSection] = useState("overview")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [risk, setRisk] = useState<RiskState>({ positions: {}, portfolio: undefined, account: undefined })
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const API_BASE = (process.env.NEXT_PUBLIC_RISK_API_URL || "http://localhost:8001").replace(/\/$/, "")

  // Initial fetch
  useEffect(() => {
    let cancelled = false
    async function fetchInitial() {
      try {
        const res = await fetch(`${API_BASE}/api/analysis`, { cache: "no-store" })
        const data = (await res.json()) as RiskState
        if (!cancelled) {
          setRisk(data)
          setLoading(false)
          setLastUpdate(new Date().toISOString())
        }
      } catch (e) {
        // keep loading false but risk may be empty
        if (!cancelled) setLoading(false)
      }
    }
    fetchInitial()
    return () => {
      cancelled = true
    }
  }, [API_BASE])

  // WebSocket live updates
  useEffect(() => {
    const wsUrl = wsUrlFromHttp(`${API_BASE}`) + "/ws/stream"
    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data)
          // Ignore pings
          if (msg && msg.type === "ping") return
          if (msg && (msg.positions || msg.portfolio || msg.account)) {
            setRisk(msg)
            setLastUpdate(new Date().toISOString())
          }
        } catch {
          // ignore malformed
        }
      }
      ws.onerror = () => {
        // noop
      }
      ws.onclose = () => {
        wsRef.current = null
      }
      return () => {
        ws.close()
      }
    } catch {
      // ws not available
      return () => {}
    }
  }, [API_BASE])

  const positionsArray = useMemo(() => {
    const rec = risk?.positions || {}
    return Object.values(rec || {}) as PositionAnalysis[]
  }, [risk])

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div
        className={`bg-neutral-900 border-r border-neutral-700 transition-all duration-300 fixed left-0 top-0 h-full z-50 md:relative md:z-auto md:h-auto ${sidebarCollapsed ? "-translate-x-full md:translate-x-0 md:w-16" : "translate-x-0 md:w-72"} md:transform-none`}
      >
        <div className="p-4">
          <div className="flex items-center justify-between mb-8">
            <div className={`${sidebarCollapsed ? "hidden" : "block"}`}>
              <h1 className="text-orange-500 font-bold text-lg tracking-wider">RISK TERMINAL</h1>
              <p className="text-neutral-500 text-xs">v3.2.1 LIVE TRADING</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="text-neutral-400 hover:text-orange-500"
            >
              <ChevronRight
                className={`w-4 h-4 sm:w-5 sm:h-5 transition-transform ${sidebarCollapsed ? "" : "rotate-180"}`}
              />
            </Button>
          </div>

          <nav className="space-y-2">
            {[
              { id: "overview", icon: BarChart3, label: "PORTFOLIO" },
              { id: "positions", icon: TrendingUp, label: "POSITIONS" },
              { id: "risk", icon: AlertTriangle, label: "RISK ANALYSIS" },
              { id: "alerts", icon: Bell, label: "ALERTS" },
              { id: "performance", icon: DollarSign, label: "PERFORMANCE" },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={`w-full flex items-center gap-3 p-3 rounded transition-colors ${
                  activeSection === item.id
                    ? "bg-orange-500 text-white"
                    : "text-neutral-400 hover:text-white hover:bg-neutral-800"
                }`}
              >
                <item.icon className="w-5 h-5 md:w-5 md:h-5 sm:w-6 sm:h-6" />
                {!sidebarCollapsed && <span className="text-sm font-medium">{item.label}</span>}
              </button>
            ))}
          </nav>

          {!sidebarCollapsed && (
            <div className="mt-8 p-4 bg-neutral-800 border border-neutral-700 rounded">
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-2 h-2 ${loading ? "bg-neutral-500" : "bg-green-500"} rounded-full animate-pulse`}></div>
                <span className="text-xs text-white">{loading ? "CONNECTING" : "MARKET LIVE"}</span>
              </div>
              <div className="text-xs text-neutral-500">
                <div>POSITIONS: {risk?.portfolio?.total_positions ?? Object.keys(risk.positions || {}).length}</div>
                <div>P&L: ${risk?.portfolio?.total_unrealized_pnl?.toLocaleString?.() ?? "-"}</div>
                <div>RISK: ${risk?.portfolio?.total_risk_if_all_sl_hit?.toLocaleString?.() ?? "-"}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Mobile Overlay */}
      {!sidebarCollapsed && (
        <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={() => setSidebarCollapsed(true)} />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col md:ml-0">
        {/* Top Toolbar */}
        <div className="h-16 bg-neutral-800 border-b border-neutral-700 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="text-sm text-neutral-400">
              RISK TERMINAL / <span className="text-orange-500">LIVE TRADING</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-xs text-neutral-500">LAST UPDATE: {lastUpdate ? new Date(lastUpdate).toUTCString() : "--"}</div>
            <Button
              variant="ghost"
              size="icon"
              className="text-neutral-400 hover:text-orange-500"
              onClick={async () => {
                try {
                  const res = await fetch(`${API_BASE}/api/analysis`, { cache: "no-store" })
                  const data = (await res.json()) as RiskState
                  setRisk(data)
                  setLastUpdate(new Date().toISOString())
                } catch {}
              }}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Dashboard Content */}
        <div className="flex-1 overflow-auto">
          {activeSection === "overview" && (
            <PortfolioOverviewPage
              loading={loading}
              portfolio={risk.portfolio}
              positionsArray={positionsArray}
            />
          )}
          {activeSection === "positions" && (
            <PositionsPage loading={loading} positionsArray={positionsArray} />
          )}
          {activeSection === "risk" && (
            <RiskAnalysisPage loading={loading} portfolio={risk.portfolio} positionsArray={positionsArray} />
          )}
          {activeSection === "alerts" && <AlertsPage />}
          {activeSection === "performance" && <PerformancePage />}
        </div>
      </div>
    </div>
  )
}
