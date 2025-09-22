"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, TrendingUp, Shield, Target } from "lucide-react"

// Minimal types aligned with app/page.tsx data shape
export type PositionLike = {
  symbol?: string
  atr_pct?: number
  atr_percent?: number
  volatility_index?: number
  volatility?: number
  // optional additional fields used elsewhere
}

export type PortfolioMetrics = {
  total_risk_if_all_sl_hit?: number
  total_reward_if_all_tp_hit?: number
  portfolio_risk_reward_ratio?: number
}

type RiskAnalysisPageProps = {
  loading?: boolean
  portfolio?: PortfolioMetrics
  positionsArray?: PositionLike[]
}

export default function RiskAnalysisPage({ loading, portfolio, positionsArray = [] }: RiskAnalysisPageProps) {
  const portfolioRisk = portfolio?.total_risk_if_all_sl_hit ?? 0
  const portfolioReward = portfolio?.total_reward_if_all_tp_hit ?? 0
  const riskRewardRatio = portfolio?.portfolio_risk_reward_ratio ?? (portfolioRisk > 0 ? portfolioReward / portfolioRisk : 0)

  // Build volatility rows from any known volatility-related keys
  const volatilityRows = positionsArray
    .map((p) => {
      const atr = (p as any).atr_pct ?? (p as any).atr_percent ?? undefined
      const vol = (p as any).volatility_index ?? (p as any).volatility ?? undefined
      return {
        symbol: p.symbol ?? "—",
        atrPct: typeof atr === "number" ? atr : undefined,
        volIdx: typeof vol === "number" ? vol : undefined,
      }
    })
    .filter((r) => r.atrPct !== undefined || r.volIdx !== undefined)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">RISK ANALYSIS</h1>
          <p className="text-neutral-400">Advanced portfolio risk assessment and correlation analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-orange-500" />
          <span className="text-sm text-orange-500">{loading ? "SYNCING" : "RISK MONITORING ACTIVE"}</span>
        </div>
      </div>

      {/* Risk Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card className="bg-neutral-800 border-orange-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Portfolio Risk</p>
              <p className="text-2xl font-bold text-orange-500">${portfolioRisk.toLocaleString()}</p>
              <p className="text-xs text-neutral-500">If all stops hit</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-orange-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-green-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Portfolio Reward</p>
              <p className="text-2xl font-bold text-green-500">${portfolioReward.toLocaleString()}</p>
              <p className="text-xs text-neutral-500">If all targets hit</p>
            </div>
            <Target className="w-8 h-8 text-green-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-blue-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Risk/Reward</p>
              <p className="text-2xl font-bold text-blue-500">{riskRewardRatio.toFixed(2)}</p>
              <p className="text-xs text-neutral-500">Portfolio ratio</p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
      </div>

      {/* Volatility Analysis - Render if we have any rows */}
      {volatilityRows.length > 0 && (
        <Card className="bg-neutral-800 border-neutral-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">VOLATILITY ANALYSIS</h3>
          <div className="space-y-3">
            {volatilityRows.map((item) => (
              <div key={item.symbol} className="flex items-center justify-between p-3 bg-neutral-900 rounded">
                <div className="flex items-center gap-4">
                  <span className="font-mono text-white">{item.symbol}</span>
                  {typeof item.atrPct === "number" && (
                    <Badge
                      variant={item.atrPct >= 4 ? "destructive" : item.atrPct >= 2.5 ? "secondary" : "outline"}
                      className="text-xs"
                    >
                      {item.atrPct >= 4 ? "HIGH" : item.atrPct >= 2.5 ? "MODERATE" : "LOW"}
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-6">
                  {typeof item.atrPct === "number" && (
                    <div className="text-right">
                      <p className="text-xs text-neutral-400">ATR %</p>
                      <p className="font-mono text-white">{item.atrPct.toFixed(2)}%</p>
                    </div>
                  )}
                  {typeof item.volIdx === "number" && (
                    <div className="text-right">
                      <p className="text-xs text-neutral-400">VOL INDEX</p>
                      <p className="font-mono text-neutral-300">{item.volIdx.toFixed(2)}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Risk Metrics Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-neutral-800 border-neutral-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">PORTFOLIO METRICS</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Total Risk (SL)</span>
              <span className="text-red-500 font-mono">${portfolioRisk.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Total Reward (TP)</span>
              <span className="text-green-500 font-mono">${portfolioReward.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Risk/Reward</span>
              <span className="text-blue-500 font-mono">{riskRewardRatio.toFixed(2)}</span>
            </div>
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">RISK LIMITS</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Monitoring</span>
              <span className={`font-mono ${loading ? "text-neutral-400" : "text-green-500"}`}>
                {loading ? "SYNCING" : "ACTIVE"}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Positions Analyzed</span>
              <span className="text-white font-mono">{positionsArray.length}</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
