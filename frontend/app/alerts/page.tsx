"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, Bell, Clock, X } from "lucide-react"

export default function AlertsPage() {
  const alerts = [
    {
      id: 1,
      type: "CRITICAL",
      symbol: "BRETT/USDT:USDT",
      message: "Stop loss too close to liquidation price",
      timestamp: "2025-09-19T19:49:12Z",
      action: "Reduce exposure immediately",
      acknowledged: false,
    },
    {
      id: 2,
      type: "WARNING",
      symbol: "SUI/USDT:USDT",
      message: "Position drawdown exceeds 2% threshold",
      timestamp: "2025-09-19T19:45:30Z",
      action: "Consider tightening stop loss",
      acknowledged: false,
    },
    {
      id: 3,
      type: "WARNING",
      symbol: "XRP/USDT:USDT",
      message: "Position drawdown exceeds 2% threshold",
      timestamp: "2025-09-19T19:42:15Z",
      action: "Review position thesis",
      acknowledged: false,
    },
    {
      id: 4,
      type: "INFO",
      symbol: "LQTY/USDT:USDT",
      message: "Position in profit - consider scaling out",
      timestamp: "2025-09-19T19:40:00Z",
      action: "Trail stop or take partial profits",
      acknowledged: true,
    },
    {
      id: 5,
      type: "SYSTEM",
      symbol: "PORTFOLIO",
      message: "Correlation risk detected in 9 positions",
      timestamp: "2025-09-19T19:35:22Z",
      action: "Risk automatically capped at $905.65",
      acknowledged: false,
    },
  ]

  const getAlertColor = (type: string) => {
    switch (type) {
      case "CRITICAL":
        return "border-red-500 bg-red-500/10"
      case "WARNING":
        return "border-orange-500 bg-orange-500/10"
      case "INFO":
        return "border-blue-500 bg-blue-500/10"
      case "SYSTEM":
        return "border-purple-500 bg-purple-500/10"
      default:
        return "border-neutral-500 bg-neutral-500/10"
    }
  }

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "CRITICAL":
        return <AlertTriangle className="w-5 h-5 text-red-500" />
      case "WARNING":
        return <AlertTriangle className="w-5 h-5 text-orange-500" />
      case "INFO":
        return <Bell className="w-5 h-5 text-blue-500" />
      case "SYSTEM":
        return <Bell className="w-5 h-5 text-purple-500" />
      default:
        return <Bell className="w-5 h-5 text-neutral-500" />
    }
  }

  const getBadgeVariant = (type: string) => {
    switch (type) {
      case "CRITICAL":
        return "destructive"
      case "WARNING":
        return "secondary"
      case "INFO":
        return "default"
      case "SYSTEM":
        return "outline"
      default:
        return "outline"
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }

  const unacknowledgedCount = alerts.filter((alert) => !alert.acknowledged).length

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">RISK ALERTS</h1>
          <p className="text-neutral-400">Real-time risk monitoring and alert management</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-red-500">{unacknowledgedCount} UNACKNOWLEDGED</span>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="border-orange-500 text-orange-500 hover:bg-orange-500 hover:text-white bg-transparent"
          >
            Acknowledge All
          </Button>
        </div>
      </div>

      {/* Alert Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-neutral-800 border-red-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Critical</p>
              <p className="text-2xl font-bold text-red-500">{alerts.filter((a) => a.type === "CRITICAL").length}</p>
            </div>
            <AlertTriangle className="w-6 h-6 text-red-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-orange-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Warning</p>
              <p className="text-2xl font-bold text-orange-500">{alerts.filter((a) => a.type === "WARNING").length}</p>
            </div>
            <AlertTriangle className="w-6 h-6 text-orange-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-blue-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">Info</p>
              <p className="text-2xl font-bold text-blue-500">{alerts.filter((a) => a.type === "INFO").length}</p>
            </div>
            <Bell className="w-6 h-6 text-blue-500" />
          </div>
        </Card>

        <Card className="bg-neutral-800 border-purple-500/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-neutral-400 uppercase tracking-wide">System</p>
              <p className="text-2xl font-bold text-purple-500">{alerts.filter((a) => a.type === "SYSTEM").length}</p>
            </div>
            <Bell className="w-6 h-6 text-purple-500" />
          </div>
        </Card>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {alerts.map((alert) => (
          <Card
            key={alert.id}
            className={`border p-4 ${getAlertColor(alert.type)} ${alert.acknowledged ? "opacity-60" : ""}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4 flex-1">
                {getAlertIcon(alert.type)}
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <Badge variant={getBadgeVariant(alert.type)} className="text-xs">
                      {alert.type}
                    </Badge>
                    <span className="font-mono text-sm text-white">{alert.symbol}</span>
                    <div className="flex items-center gap-1 text-xs text-neutral-500">
                      <Clock className="w-3 h-3" />
                      {formatTimestamp(alert.timestamp)}
                    </div>
                  </div>
                  <p className="text-white mb-2">{alert.message}</p>
                  <p className="text-sm text-neutral-400">
                    <span className="font-medium">Recommended Action:</span> {alert.action}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {!alert.acknowledged && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="text-xs border-neutral-600 text-neutral-400 hover:bg-neutral-700 bg-transparent"
                  >
                    Acknowledge
                  </Button>
                )}
                <Button variant="ghost" size="icon" className="text-neutral-400 hover:text-red-500">
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Alert Settings */}
      <Card className="bg-neutral-800 border-neutral-700 p-6">
        <h3 className="text-lg font-bold text-white mb-4">ALERT SETTINGS</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-white mb-3">Risk Thresholds</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 text-sm">Position Drawdown</span>
                <span className="text-white font-mono text-sm">2.0%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 text-sm">Portfolio Risk</span>
                <span className="text-white font-mono text-sm">$2,000</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 text-sm">Correlation Limit</span>
                <span className="text-white font-mono text-sm">0.7</span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="text-sm font-medium text-white mb-3">Notification Settings</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 text-sm">Email Alerts</span>
                <span className="text-green-500 text-sm">Enabled</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 text-sm">SMS Alerts</span>
                <span className="text-green-500 text-sm">Enabled</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 text-sm">Push Notifications</span>
                <span className="text-green-500 text-sm">Enabled</span>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
