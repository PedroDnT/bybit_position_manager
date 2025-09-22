"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, AlertTriangle, DollarSign } from "lucide-react"

type PortfolioOverviewProps = {
  loading?: boolean
  portfolio?: {
    total_positions?: number
    total_notional?: number
    total_unrealized_pnl?: number
    total_risk_if_all_sl_hit?: number
    total_reward_if_all_tp_hit?: number
    portfolio_risk_reward_ratio?: number
    risk_pct_of_notional?: number
    positions_at_risk?: string[]
  }
  positionsArray?: any[]
}

export default function PortfolioOverviewPage({ loading, portfolio, positionsArray = [] }: PortfolioOverviewProps) {
  const totalPositions = portfolio?.total_positions ?? positionsArray.length
  const totalNotional = portfolio?.total_notional ?? 0
  const totalUnrealizedPnl = portfolio?.total_unrealized_pnl ?? 0
  const totalRisk = portfolio?.total_risk_if_all_sl_hit ?? 0
  const totalReward = portfolio?.total_reward_if_all_tp_hit ?? 0
  const riskRewardRatio = portfolio?.portfolio_risk_reward_ratio ?? 0
  const riskPctOfNotional = portfolio?.risk_pct_of_notional ?? 0

  const positionsAtRisk: string[] = (portfolio?.positions_at_risk && portfolio.positions_at_risk.length > 0)
    ? portfolio.positions_at_risk
    : positionsArray
        .filter((p: any) => ["CRITICAL", "WARNING"].includes(p?.position_health))
        .map((p: any) => p?.symbol)
        .slice(0, 9)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">PORTFOLIO OVERVIEW</h1>
          <p className="text-neutral-400">Real-time portfolio risk assessment</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 ${loading ? "bg-neutral-500" : "bg-green-500"} rounded-full animate-pulse`}></div>
          <span className={`text-sm ${loading ? "text-neutral-400" : "text-green-500"}`}>{loading ? "CONNECTING" : "LIVE DATA"}</span>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Total Notional</p>
              <p className="text-2xl font-bold text-white">${Number(totalNotional).toLocaleString()}</p>
            </div>
            <DollarSign className="w-8 h-8 text-blue-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Unrealized P&L</p>
              <p className={`text-2xl font-bold ${Number(totalUnrealizedPnl) >= 0 ? "text-green-500" : "text-red-500"}`}>
                ${Number(totalUnrealizedPnl).toLocaleString()}
              </p>
            </div>
            {Number(totalUnrealizedPnl) >= 0 ? (
              <TrendingUp className="w-8 h-8 text-green-500" />
            ) : (
              <TrendingDown className="w-8 h-8 text-red-500" />
            )}
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Total Risk</p>
              <p className="text-2xl font-bold text-orange-500">${Number(totalRisk).toLocaleString()}</p>
              <p className="text-xs text-neutral-500">{Number(riskPctOfNotional).toFixed(2)}% of notional</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-orange-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide" role="heading" aria-level={2}>Risk/Reward</p>
              <p className="text-2xl font-bold text-green-500">{Number(riskRewardRatio).toFixed(2)}</p>
              <p className="text-xs text-neutral-500">Reward: ${Number(totalReward).toLocaleString()}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-500" />
          </div>
        </Card>
      </div>

      {/* Risk Alert Section */}
      <Card className="bg-neutral-800 border-red-500/50 p-6">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-6 h-6 text-red-500" />
          <h2 className="text-xl font-bold text-white">POSITIONS AT RISK</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {positionsAtRisk.length === 0 && (
            <div className="text-neutral-400 text-sm">No positions currently flagged as high risk.</div>
          )}
          {positionsAtRisk.map((symbol) => (
            <div key={symbol} className="bg-neutral-900 border border-red-500/30 rounded p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-white">{symbol}</span>
                <Badge variant="destructive" className="text-xs">HIGH RISK</Badge>
              </div>
              <p className="text-xs text-neutral-400">Requires immediate attention</p>
            </div>
          ))}
        </div>
      </Card>

      {/* Portfolio Health Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-neutral-800 border-neutral-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">PORTFOLIO COMPOSITION</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Active Positions</span>
              <span className="text-white font-mono">{totalPositions}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Average Position Size</span>
              <span className="text-white font-mono">${(Number(totalNotional) / (totalPositions || 1)).toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Risk per Position</span>
              <span className="text-orange-500 font-mono">${(Number(totalRisk) / (totalPositions || 1)).toFixed(2)}</span>
            </div>
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">RISK METRICS</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Portfolio Beta</span>
              <span className="text-white font-mono">—</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Max Drawdown</span>
              <span className="text-red-500 font-mono">—</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Sharpe Ratio</span>
              <span className="text-green-500 font-mono">—</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
