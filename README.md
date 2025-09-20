# Advanced Crypto Position Risk Management System

A comprehensive cryptocurrency trading risk management system that combines real-time position monitoring with advanced volatility analysis using GARCH and HAR-RV models, plus portfolio correlation analysis.

## 🎯 Overview

This system provides sophisticated risk management for cryptocurrency trading by:

- **Real-time Position Monitoring**: Fetches live positions from Bybit
- **Advanced Volatility Analysis**: Uses GARCH(1,1) and HAR-RV models for volatility forecasting
- **Dynamic SL/TP Calculation**: Calculates optimal stop-loss and take-profit levels based on volatility
- **Position Sizing**: Recommends optimal position sizes based on target risk
- **Portfolio Risk Assessment**: Analyzes overall portfolio risk and provides actionable recommendations
- **Account Overview**: Displays total equity, available balance, and today's realized/unrealized PnL
- **Correlation Analysis**: Identifies correlated positions and applies cluster risk caps
- **Configuration Management**: Dynamic settings loading with fallback to defaults

## 📁 Project Structure

```
market_analysis/
├── position_risk_manager.py    # Main risk management system
├── garch_vol_triggers.py       # GARCH and HAR-RV volatility models
├── get_position.py             # Position fetching utilities
├── atr_sl_gpt.py              # ATR-based risk management
├── requirements.txt           # Python dependencies
├── settings.toml             # Configuration file (create from settings.example.toml)
├── settings.example.toml     # Example configuration
├── risk_analysis.json        # Generated risk analysis output
└── README.md                 # This file
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
docker build -t risk-manager-app .
```

#### Running the Container

Run the application inside a Docker container. You will need to mount your `.env` file and `settings.toml` into the container so it can access your configuration and API keys.

```bash
docker run --rm -v "$(pwd)/.env":/app/.env -v "$(pwd)/settings.toml":/app/settings.toml risk-manager-app
```
The `--rm` flag will automatically remove the container when it exits. The `-v` flags mount your local configuration files into the container at runtime.

### 🔬 Running Tests

```bash
pytest -q
```

## 🔧 Core Components & Logic Flow

### 1. Position Risk Manager (`position_risk_manager.py`)

- Pulls open positions, current prices, and account info
- Computes ATR and volatility forecasts
- Chooses effective SL/TP based on regime and confidence
- Computes R, R:R, and position sizing guidance
- Emits per-position analysis dict and portfolio report

... existing code ...

### 🧠 Advanced Risk Logic Explained

#### 1. Confidence Scoring Model
- Combines multiple indicators and regime detection to assign confidence to signals

... existing code ...

#### 4. State-Anchored Dynamic SL/TP with Probability Model
- Uses current-price anchoring with volatility horizon to dynamically update SL/TP
- Incorporates probability of hitting TP before SL and adapts the effective TP multiplier

Key configuration keys:
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
