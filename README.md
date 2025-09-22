# Advanced Crypto Risk Manager

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive cryptocurrency trading risk management system that combines real-time position monitoring with advanced volatility analysis. This tool provides sophisticated risk assessment capabilities including GARCH volatility modeling, portfolio correlation analysis, and adaptive stop-loss mechanisms.

## 🚀 Features

- **📊 Real-time Position Monitoring**: Track open positions across multiple exchanges
- **📈 Advanced Volatility Analysis**: GARCH modeling for volatility forecasting  
- **🔗 Portfolio Correlation Analysis**: Understand position interdependencies
- **🎯 Adaptive Stop-Loss**: Dynamic stop-loss adjustment based on market conditions
- **📋 Risk Metrics**: Comprehensive risk assessment and reporting
- **🔄 Multi-Exchange Support**: Currently supports Bybit with extensible architecture
- **⚡ High Performance**: Optimized for real-time trading environments
- **🛡️ Risk Management**: Advanced position sizing and risk control

## 📁 Project Structure

```
market_analysis/
├── __init__.py
├── __main__.py              # Main entry point and CLI interface
├── adaptive_stop_loss.py    # Adaptive stop-loss implementation
├── atr_sl_gpt.py           # ATR-based stop-loss logic
├── confidence.py           # Confidence interval calculations
├── config.py               # Configuration management
├── demo_adaptive_stop_loss.py  # Demo script for adaptive stop-loss
├── garch_vol_triggers.py   # GARCH volatility analysis
├── get_position.py         # Position data retrieval from exchanges
├── position_risk_manager.py # Main risk management logic
├── reporting.py            # Report generation and output formatting
├── test_adaptive_stop_loss.py # Tests for adaptive stop-loss
├── utils.py                # Utility functions and helpers
└── web_server.py           # FastAPI web server for frontend integration

Frontend Dashboard:
└── frontend/
    ├── app/                # Next.js 14 App Router pages
    │   ├── page.tsx       # Main dashboard
    │   ├── layout.tsx     # Root layout
    │   ├── portfolio-overview/ # Portfolio metrics page
    │   ├── positions/     # Position monitoring page
    │   ├── risk-analysis/ # Risk analysis page
    │   ├── alerts/        # Alerts management page
    │   └── performance/   # Performance tracking page
    ├── components/        # Reusable React components
    │   └── ui/           # UI component library
    ├── lib/              # Frontend utilities
    ├── public/           # Static assets
    ├── package.json      # Node.js dependencies
    └── next.config.mjs   # Next.js configuration

Configuration & Setup:
├── settings.toml           # Your configuration file (create from example)
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container deployment
└── README.md              # This documentation

Tests:
└── tests/
    ├── test_*.py          # Comprehensive test suite
    └── test_data.py       # Test data and fixtures
```

## 🚀 Quick Start

There are two ways to run the application: directly via `pip` or using Docker.

### 1. Local Installation & Execution

#### Installation

```
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

3.  Create a `.env` file in the root directory with your API credentials (this is read by the application):
    ```
    BYBIT_API_KEY=your_api_key_here
    BYBIT_API_SECRET=your_api_secret_here
    ```

#### Running the Application

Once installed, you can run the analysis using the new command-line tool:

```bash
risk-manager
```

The command prints a per-position table that surfaces the GARCH-driven stop loss, take profit, and trailing stop guidance alongside risk budget, sizing multipliers, and the volatility model that produced the levels.

To run in real-time monitoring mode:

```bash
# Monitor continuously with defaults (10s interval, unlimited iterations)
risk-manager --monitor

# Monitor with custom polling interval and limit the number of iterations
risk-manager --monitor --interval 60 --iterations 10
```

Flags:
- --monitor: enable continuous monitoring loop
- --interval: seconds between refreshes (default: 10)
- --iterations: max iterations before exiting (default: unlimited)
- --print-report: print the full risk report to stdout after analysis (still writes risk_analysis.json)

### 2. Docker Execution

Alternatively, you can use Docker to build and run the application in a containerized environment.

#### Building the Image

From the project root directory, build the Docker image:
```bash
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"
export BYBIT_SANDBOX="false"
```

## 🧪 Running Tests

Execute the test suite to ensure everything is working correctly:

```bash
docker run --rm -v "$(pwd)/.env":/app/.env -v "$(pwd)/settings.toml":/app/settings.toml risk-manager-app
```
The `--rm` flag will automatically remove the container when it exits. The `-v` flags mount your local configuration files into the container at runtime.

### 🔬 Running Tests

```bash
pytest -q
```

## 🖥️ Frontend Dashboard

The project includes a modern, real-time web dashboard built with Next.js that provides a comprehensive interface for monitoring your trading positions and risk metrics.

### 🚀 Frontend Quick Start

#### Prerequisites
- Node.js 18+ and npm/pnpm
- Backend API server running on `http://localhost:8000`

#### Installation & Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   pnpm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   # or
   pnpm dev
   ```

4. **Access the dashboard:**
   Open your browser and navigate to `http://localhost:3000`

### 📊 Dashboard Features

#### **Main Dashboard (`/`)**
- **Real-time Risk Overview**: Live risk scores and portfolio health status
- **Position Summary**: Quick view of all open positions with key metrics
- **Alert Center**: Critical alerts and notifications
- **Connection Status**: Backend API connectivity indicator
- **Navigation Sidebar**: Access to all dashboard sections

#### **Portfolio Overview (`/portfolio-overview`)**
- **Portfolio Metrics**: Total notional, unrealized PnL, risk utilization
- **Risk Distribution**: Visual breakdown of risk across positions
- **Performance Summary**: Portfolio-level performance indicators
- **Risk-Reward Analysis**: Portfolio-wide risk-reward ratios

#### **Positions (`/positions`)**
- **Position Table**: Detailed view of all open positions
- **Individual Position Cards**: Per-position risk analysis
- **Stop-Loss & Take-Profit Levels**: GARCH-based recommendations
- **Liquidation Buffers**: Safety margin indicators
- **Position Health Scores**: Risk assessment for each position

#### **Risk Analysis (`/risk-analysis`)**
- **GARCH Volatility Models**: Advanced volatility forecasting
- **Correlation Matrix**: Position interdependency analysis
- **Risk Metrics**: VaR, expected shortfall, and other risk measures
- **Scenario Analysis**: Stress testing and what-if scenarios

#### **Alerts (`/alerts`)**
- **Active Alerts**: Current risk alerts and warnings
- **Alert History**: Past alerts and their resolution
- **Alert Configuration**: Customize alert thresholds and notifications
- **Severity Filtering**: Filter alerts by severity level

#### **Performance (`/performance`)**
- **P&L Tracking**: Historical performance analysis
- **Risk-Adjusted Returns**: Sharpe ratio and other risk-adjusted metrics
- **Drawdown Analysis**: Maximum drawdown and recovery periods
- **Performance Attribution**: Breakdown by position and strategy

### 🎨 Dashboard Interface

#### **Color-Coded Risk Levels**
- 🔴 **Critical**: Immediate attention required
- 🟠 **High**: Monitor closely
- 🟡 **Medium**: Standard monitoring
- 🟢 **Low**: Normal operation

#### **Real-Time Updates**
- WebSocket connection for live data updates
- Automatic refresh of risk metrics
- Real-time alert notifications
- Connection status indicators

#### **Responsive Design**
- Mobile-friendly interface
- Collapsible sidebar for smaller screens
- Adaptive layouts for different screen sizes
- Touch-friendly controls

### 🔧 Frontend Configuration

#### **Environment Variables**
Create a `.env.local` file in the frontend directory:
```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Dashboard Settings
NEXT_PUBLIC_REFRESH_INTERVAL=5000
NEXT_PUBLIC_ALERT_SOUND=true
```

#### **Build for Production**
```bash
# Build the application
npm run build

# Start production server
npm start
```

#### **Docker Deployment**
```bash
# Build frontend Docker image
docker build -t risk-manager-frontend ./frontend

# Run frontend container
docker run -p 3000:3000 risk-manager-frontend
```

### 🛠️ Development

#### **Project Structure**
```
frontend/
├── app/                    # Next.js 14 App Router
│   ├── page.tsx           # Main dashboard
│   ├── layout.tsx         # Root layout
│   ├── globals.css        # Global styles
│   ├── portfolio-overview/ # Portfolio section
│   ├── positions/         # Positions section
│   ├── risk-analysis/     # Risk analysis section
│   ├── alerts/           # Alerts section
│   └── performance/      # Performance section
├── components/            # Reusable components
│   └── ui/               # UI component library
├── lib/                  # Utility functions
├── public/               # Static assets
└── styles/               # Additional styles
```

#### **Key Technologies**
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library
- **WebSocket**: Real-time data connection

#### **API Integration**
The frontend connects to the backend API for:
- Position data retrieval
- Risk analysis results
- Real-time updates via WebSocket
- Alert management
- Performance metrics

#### **Customization**
- Modify `tailwind.config.ts` for custom styling
- Update `components/ui/` for custom UI components
- Configure `next.config.mjs` for build settings
- Customize alert sounds and notifications

### 🔍 Troubleshooting

#### **Common Issues**

1. **Dashboard shows "Page not found"**
   ```bash
   # Clear Next.js cache and restart
   rm -rf .next
   npm run dev
   ```

2. **API connection errors**
   - Ensure backend server is running on port 8000
   - Check CORS settings in backend configuration
   - Verify API endpoints are accessible

3. **WebSocket connection issues**
   - Check WebSocket URL configuration
   - Ensure firewall allows WebSocket connections
   - Verify backend WebSocket implementation

4. **Build errors**
   ```bash
   # Clear node modules and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

### 📱 Mobile Access

The dashboard is fully responsive and can be accessed on mobile devices:
- Touch-friendly interface
- Optimized layouts for small screens
- Swipe gestures for navigation
- Mobile-specific alert notifications

## 🔧 Core Components & Logic Flow

### 1. Position Risk Manager (`position_risk_manager.py`)

- Pulls open positions, current prices, and account info
- Computes ATR and volatility forecasts
- Chooses effective SL/TP based on regime and confidence
- Computes R, R:R, and position sizing guidance
- Budgets risk off account equity with configurable 0.5–1% guard rails and reports the applied dollars
- Enforces a drawdown-aware portfolio risk cap that scales planned losses when they exceed a configurable fraction of equity
- Emits per-position analysis dict and portfolio report

#### Liquidation buffer guardrail

- The `[risk].liquidation_buffer_multiple` setting controls how far the liquidation price must sit beyond the recommended stop.
- By default the guardrail requires the liquidation price to be at least **2.0 ×** the stop distance (roughly twice the ATR-based cushion).
- Increase the multiple to demand more breathing room during high leverage or volatile regimes; lower it if exchange margin rules already provide a generous cushion.
- Reports surface both the observed buffer ratio and the configured threshold so thin cushions are easy to spot.

... existing code ...

### 🧠 Advanced Risk Logic Explained

#### 1. Confidence Scoring Model
- Combines multiple indicators and regime detection to assign confidence to signals

... existing code ...

#### 4. State-Anchored Dynamic SL/TP with Probability Model
- Uses current-price anchoring with volatility horizon to dynamically update SL/TP
- Incorporates probability of hitting TP before SL and adapts the effective TP multiplier

Key configuration keys:
- [risk] min_equity_risk_frac / max_equity_risk_frac for equity-based risk guard rails
- [portfolio] max_portfolio_risk_frac / min_portfolio_risk_frac / drawdown_multipliers for the equity-level risk throttle
- [vol] horizon_hours
- [stops] breakeven_after_R, atr_trail_mult_initial, atr_trail_mult_late
- [prob] prob_alpha, prob_target, m_min, m_max, m_step

Example (library usage):
```python
from market_analysis.garch_vol_triggers import (
    get_klines_bybit, compute_atr, garch_sigma_ann_and_sigma_H,
    blended_sigma_h, backtest_dynamic_levels
)
from market_analysis.config import settings
import ccxt

ex = ccxt.bybit()
symbol = "BTC/USDT"
# Load OHLCV, compute ATR
raw = get_klines_bybit(ex, symbol=symbol, timeframe="1h", limit=1000)
atr = compute_atr(raw, period=14)
# Choose an entry index and price (e.g., last 200th bar)
entry_idx = len(raw) - 200
entry_price = float(raw['close'].iloc[entry_idx])
side = "long"
# Volatility horizon (e.g., 4h) using GARCH + HAR blending
sigma_ann_garch, sigma_H_garch = garch_sigma_ann_and_sigma_H(raw['close'], interval="1h", horizon_hours=4)
# Blend with ATR-based proxy (use [vol] config section)
sigH = blended_sigma_h(sigma_ann_garch, None, atr_abs=atr.iloc[entry_idx], price=entry_price, cfg=settings.get('vol', {}))
# Choose baseline multiples for test (example values)
base_k = 2.0
base_m = 3.5
res = backtest_dynamic_levels(
    df=raw,
    entry_idx=entry_idx,
    side=side,
    entry_price=entry_price,
    sigma_H=sigH,
    atr_series=atr,
    base_k=base_k,
    base_m=base_m,
    cfg=settings,
)
print(res)
```

Result dictionary includes which method (static vs dynamic) reached TP or SL first, the first hit indices, and example levels for inspection.

## 💾 Saved Artifacts

- risk_analysis.json: full human-readable text report is written after each run.
- Console: live table summary; add --print-report to also print the full report to stdout.
- JSON fields in per-position analysis are also persisted under the hood (e.g., tp1, tp2, scaleout_frac1/2, leave_runner_frac, dynamic_p_tp, k_eff, m_eff).
