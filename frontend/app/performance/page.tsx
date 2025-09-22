"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, DollarSign, Percent } from "lucide-react"

export default function PerformancePage() {
  const performanceData = {
    totalPnl: -1197.54,
    dailyPnl: -247.86,
    weeklyPnl: 156.23,
    monthlyPnl: -892.45,
    winRate: 67.5,
    avgWin: 145.67,
    avgLoss: -89.23,
    profitFactor: 1.63,
    sharpeRatio: 0.87,
    maxDrawdown: -3.44,
  }

  const topPerformers = [
    { symbol: "LQTY/USDT", pnl: 23.27, pnlPct: 0.96, status: "winner" },
    { symbol: "BRETT/USDT", pnl: 33.63, pnlPct: 0.67, status: "winner" },
    { symbol: "W/USDT", pnl: -19.25, pnlPct: -0.55, status: "loser" },
  ]

  const bottomPerformers = [
    { symbol: "SUI/USDT", pnl: -466.45, pnlPct: -2.97, status: "loser" },
    { symbol: "XRP/USDT", pnl: -247.86, pnlPct: -3.44, status: "loser" },
    { symbol: "AVAX/USDT", pnl: -49.59, pnlPct: -0.42, status: "loser" },
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">PERFORMANCE ANALYTICS</h1>
          <p className="text-neutral-400">Portfolio performance metrics and analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-blue-500">ANALYTICS ACTIVE</span>
        </div>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Total P&L</p>
              <p className={`text-2xl font-bold ${performanceData.totalPnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                ${performanceData.totalPnl.toLocaleString()}
              </p>
            </div>
            {performanceData.totalPnl >= 0 ? (
              <TrendingUp className="w-8 h-8 text-green-500" />
            ) : (
              <TrendingDown className="w-8 h-8 text-red-500" />
            )}
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Win Rate</p>
              <p className="text-2xl font-bold text-blue-500">{performanceData.winRate}%</p>
            </div>
            <Percent className="w-8 h-8 text-blue-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Profit Factor</p>
              <p className="text-2xl font-bold text-green-500">{performanceData.profitFactor}</p>
            </div>
            <DollarSign className="w-8 h-8 text-green-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-neutral-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Max Drawdown</p>
              <p className="text-2xl font-bold text-red-500">{performanceData.maxDrawdown}%</p>
            </div>
            <TrendingDown className="w-8 h-8 text-red-500" />
          </div>
        </Card>
      </div>

      {/* Time-based Performance */}
      <Card className="bg-neutral-800 border-neutral-700 p-6">
        <h3 className="text-lg font-bold text-white mb-4">TIME-BASED PERFORMANCE</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Daily P&L</p>
            <p className={`text-3xl font-bold ${performanceData.dailyPnl >= 0 ? "text-green-500" : "text-red-500"}`}>
              ${performanceData.dailyPnl.toLocaleString()}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Weekly P&L</p>
            <p className={`text-3xl font-bold ${performanceData.weeklyPnl >= 0 ? "text-green-500" : "text-red-500"}`}>
              ${performanceData.weeklyPnl.toLocaleString()}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Monthly P&L</p>
            <p className={`text-3xl font-bold ${performanceData.monthlyPnl >= 0 ? "text-green-500" : "text-red-500"}`}>
              ${performanceData.monthlyPnl.toLocaleString()}
            </p>
          </div>
        </div>
      </Card>

      {/* Top and Bottom Performers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-neutral-800 border-green-500/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            TOP PERFORMERS
          </h3>
          <div className="space-y-3">
            {topPerformers.map((position) => (
              <div key={position.symbol} className="flex items-center justify-between p-3 bg-neutral-900 rounded">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-white">{position.symbol}</span>
                  <Badge variant="default" className="bg-green-500/20 text-green-500 text-xs">
                    WINNER
                  </Badge>
                </div>
                <div className="text-right">
                  <p className="font-mono text-green-500 font-bold">${position.pnl.toFixed(2)}</p>
                  <p className="font-mono text-green-400 text-sm">{position.pnlPct.toFixed(2)}%</p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="bg-neutral-800 border-red-500/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-red-500" />
            BOTTOM PERFORMERS
          </h3>
          <div className="space-y-3">
            {bottomPerformers.map((position) => (
              <div key={position.symbol} className="flex items-center justify-between p-3 bg-neutral-900 rounded">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-white">{position.symbol}</span>
                  <Badge variant="destructive" className="text-xs">
                    LOSER
                  </Badge>
                </div>
                <div className="text-right">
                  <p className="font-mono text-red-500 font-bold">${position.pnl.toFixed(2)}</p>
                  <p className="font-mono text-red-400 text-sm">{position.pnlPct.toFixed(2)}%</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Advanced Metrics */}
      <Card className="bg-neutral-800 border-neutral-700 p-6">
        <h3 className="text-lg font-bold text-white mb-4">ADVANCED METRICS</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Average Win</p>
            <p className="text-xl font-bold text-green-500">${performanceData.avgWin.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Average Loss</p>
            <p className="text-xl font-bold text-red-500">${performanceData.avgLoss.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Sharpe Ratio</p>
            <p className="text-xl font-bold text-blue-500">{performanceData.sharpeRatio.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-neutral-400 uppercase tracking-wide mb-2">Profit Factor</p>
            <p className="text-xl font-bold text-green-500">{performanceData.profitFactor.toFixed(2)}</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
