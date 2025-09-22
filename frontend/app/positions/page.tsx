"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { TrendingUp, TrendingDown, AlertTriangle } from "lucide-react"

// Basic shape coming from backend (risk_analysis.json / FastAPI)
export type PositionLike = {
  symbol?: string
  side?: string
  entry_price?: number
  current_price?: number
  position_size?: number
  notional?: number
  leverage?: number
  current_pnl?: number
  current_pnl_pct?: number
  position_health?: string
  action_required?: string
  stop_loss?: number
  tp1?: number
  tp2?: number
  trail_stop_suggestion?: number
  regime_score?: number
  confidence_factors?: string[]
}

type PositionsPageProps = {
  loading?: boolean
  positionsArray?: PositionLike[]
}

export default function PositionsPage({ loading, positionsArray = [] }: PositionsPageProps) {
  const getHealthColor = (health?: string) => {
    switch (health) {
      case "CRITICAL":
        return "text-red-500 bg-red-500/10 border-red-500/30"
      case "WARNING":
        return "text-orange-500 bg-orange-500/10 border-orange-500/30"
      case "PROFITABLE":
        return "text-green-500 bg-green-500/10 border-green-500/30"
      default:
        return "text-blue-500 bg-blue-500/10 border-blue-500/30"
    }
  }

  const getHealthBadgeVariant = (health?: string) => {
    switch (health) {
      case "CRITICAL":
        return "destructive"
      case "WARNING":
        return "secondary"
      case "PROFITABLE":
        return "default"
      default:
        return "outline"
    }
  }

  const confidencePercent = (p: PositionLike): number => {
    // Heuristic: base on regime_score (1-4) and confidence_factors count
    const regime = typeof p.regime_score === "number" ? p.regime_score : 2
    const base = Math.min(Math.max(regime, 1), 4) / 4 // 0.25..1
    const bonus = (p.confidence_factors?.length || 0) * 0.05 // +5% each factor
    return Math.min(1, base + bonus)
  }

  const positions = positionsArray.filter(Boolean)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">ACTIVE POSITIONS</h1>
          <p className="text-neutral-400">Real-time position monitoring and risk assessment</p>
        </div>
        <div className="flex items-center gap-4">
          <div className={`w-3 h-3 ${loading ? "bg-neutral-500" : "bg-green-500"} rounded-full animate-pulse`} />
          <Button
            variant="outline"
            size="sm"
            className="border-orange-500 text-orange-500 hover:bg-orange-500 hover:text-white bg-transparent"
          >
            Close All Positions
          </Button>
          <Button variant="outline" size="sm" className="border-neutral-600 text-neutral-400 bg-transparent">
            Export Report
          </Button>
        </div>
      </div>

      {/* Positions Grid */}
      <div className="grid grid-cols-1 gap-4">
        {positions.length === 0 && (
          <div className="text-sm text-neutral-400">No active positions detected.</div>
        )}
        {positions.map((p) => {
          const symbol = p.symbol || "—"
          const health = p.position_health || "NORMAL"
          const entryPrice = p.entry_price ?? 0
          const currentPrice = p.current_price ?? 0
          const size = p.position_size ?? 0
          const notional = p.notional ?? 0
          const lev = p.leverage ?? 0
          const pnl = p.current_pnl ?? 0
          const pnlPct = p.current_pnl_pct ?? 0
          const sl = p.stop_loss
          const tp1 = p.tp1
          const tp2 = p.tp2
          const trail = p.trail_stop_suggestion
          const conf = confidencePercent(p)

          return (
            <Card key={symbol} className={`bg-neutral-800 border p-6 ${getHealthColor(health)}`}>
              <div className="flex items-center justify-between">
                {/* Symbol and Health */}
                <div className="flex items-center gap-4">
                  <div>
                    <h3 className="font-mono text-lg font-bold text-white">{symbol}</h3>
                    <Badge variant={getHealthBadgeVariant(health)} className="text-xs mt-1">
                      {health}
                    </Badge>
                  </div>
                </div>

                {/* Key Metrics in Same Row */}
                <div className="flex items-center gap-8">
                  {/* Entry/Current Prices */}
                  <div className="text-center">
                    <p className="text-xs text-neutral-400 mb-1">ENTRY / CURRENT</p>
                    <p className="font-mono text-sm text-white">${entryPrice.toFixed(6)}</p>
                    <p className="font-mono text-sm text-neutral-300">${currentPrice.toFixed(6)}</p>
                  </div>

                  {/* Size/Notional */}
                  <div className="text-center">
                    <p className="text-xs text-neutral-400 mb-1">SIZE / NOTIONAL</p>
                    <p className="font-mono text-sm text-white">{size.toLocaleString()}</p>
                    <p className="font-mono text-sm text-neutral-300">${notional.toLocaleString()}</p>
                  </div>

                  {/* P&L */}
                  <div className="text-center">
                    <p className="text-xs text-neutral-400 mb-1">P&L</p>
                    <div className="flex items-center gap-2">
                      {pnl >= 0 ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                      )}
                      <div>
                        <p className={`font-mono text-sm font-bold ${pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          ${pnl.toFixed(2)}
                        </p>
                        <p className={`font-mono text-xs ${pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {pnlPct.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Leverage */}
                  <div className="text-center">
                    <p className="text-xs text-neutral-400 mb-1">LEVERAGE</p>
                    <p className="font-mono text-lg font-bold text-white">{lev}x</p>
                  </div>
                </div>
              </div>

              {/* Action Required Alert */}
              {(p.action_required || sl || tp1 || tp2 || trail) && (
                <div className="mt-4 p-3 bg-neutral-900 border border-orange-500/30 rounded">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-orange-500" />
                    <p className="text-sm text-orange-500 font-medium">ACTION REQUIRED</p>
                  </div>
                  {p.action_required && (
                    <p className="text-sm text-neutral-300 mt-1">{p.action_required}</p>
                  )}

                  <div className="mt-3 grid grid-cols-5 gap-4 pt-3 border-t border-neutral-700">
                    <div className="text-center">
                      <p className="text-xs text-neutral-400 mb-1">CONFIDENCE</p>
                      <p
                        className={`font-mono text-sm font-bold ${
                          conf >= 0.7 ? "text-green-500" : conf >= 0.5 ? "text-orange-500" : "text-red-500"
                        }`}
                      >
                        {(conf * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-neutral-400 mb-1">SL</p>
                      <p className="font-mono text-sm text-red-400">{sl ? `$${sl.toFixed(5)}` : "—"}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-neutral-400 mb-1">TP1</p>
                      <p className="font-mono text-sm text-green-400">{tp1 ? `$${tp1.toFixed(5)}` : "—"}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-neutral-400 mb-1">TP2</p>
                      <p className="font-mono text-sm text-green-400">{tp2 ? `$${tp2.toFixed(5)}` : "—"}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-neutral-400 mb-1">TRAIL</p>
                      <p className="font-mono text-sm text-neutral-300">{trail ? `$${trail.toFixed(5)}` : "—"}</p>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          )
        })}
      </div>
    </div>
  )
}
