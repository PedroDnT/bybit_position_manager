# Advanced Crypto Position Risk Management System

A comprehensive cryptocurrency trading risk management system that combines real-time position monitoring with advanced volatility analysis using GARCH and HAR-RV models.

## 🎯 Overview

This system provides sophisticated risk management for cryptocurrency trading by:

- **Real-time Position Monitoring**: Fetches live positions from Bybit
- **Advanced Volatility Analysis**: Uses GARCH(1,1) and HAR-RV models for volatility forecasting
- **Dynamic SL/TP Calculation**: Calculates optimal stop-loss and take-profit levels based on volatility
- **Position Sizing**: Recommends optimal position sizes based on target risk
- **Portfolio Risk Assessment**: Analyzes overall portfolio risk and provides actionable recommendations

## 📁 Project Structure

```
market_analysis/
├── position_risk_manager.py    # Main risk management system
├── garch_vol_triggers.py       # GARCH and HAR-RV volatility models
├── get_position.py             # Position fetching utilities
├── atr_sl_gpt.py              # ATR-based risk management
├── garch_vol_triggers.py      # Advanced volatility analysis
├── position_risk_manager.py   # Comprehensive position analysis
├── requirements.txt           # Python dependencies
├── settings.toml             # Configuration file
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repository>
cd market_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy `settings.example.toml` to `settings.toml`
2. Add your Bybit API credentials:
```toml
[bybit]
api_key = "your_api_key"
api_secret = "your_api_secret"
sandbox = false  # Set to true for testnet
```

### Run Position Risk Analysis

```bash
python position_risk_manager.py
```

## 🔧 Core Components

### 1. Position Risk Manager (`position_risk_manager.py`)

The main system that analyzes all open positions and provides comprehensive risk management recommendations.

**Key Features:**
- Fetches real-time positions from Bybit
- Analyzes volatility using multiple methods (GARCH, HAR-RV, ATR)
- Calculates optimal SL/TP levels based on volatility
- Provides position sizing recommendations
- Generates detailed risk reports

**Output Example:**
```
Position 1: ARB/USDT:USDT
----------------------------------------
Current Status:
  Entry: $0.460210 | Current: $0.462700 | PnL: 0.56%
  Size: 331.0 | Notional: $152.33 | Leverage: 15.0x

Volatility Analysis:
  Method: HAR-RV
  ATR(20): $0.006705 (1.46% of price)
  HAR-RV σ(annual): 16.2%
  GARCH σ(annual): 93.3%

🎯 Recommended Levels:
  STOP LOSS: $0.458600 (-0.35% from entry)
    💰 Optimal Risk: $3.08 (for optimal size: 1916.08)
    💰 Current Risk: $0.53 (for current size: 331.00)
    ✅ Safe from liquidation
  TAKE PROFIT: $0.463400 (0.69% from entry)
    💰 Optimal Reward: $6.11
    💰 Current Reward: $1.06
    📊 Risk/Reward: 1.98:1
    ℹ️  POSITION SIZE SMALL: Current is 0.2x optimal
```

### 2. GARCH Volatility Triggers (`garch_vol_triggers.py`)

Advanced volatility analysis using GARCH(1,1) and HAR-RV models for forecasting future volatility.

**Key Functions:**
- `garch_sigma_ann_and_sigma_H()`: Fits GARCH(1,1) model and forecasts volatility
- `sigma_ann_and_sigma_H_from_har()`: Uses HAR-RV model for volatility forecasting
- `sl_tp_and_size()`: Calculates optimal SL/TP levels and position sizes

**Volatility Models:**

#### GARCH(1,1) Model
- Models volatility clustering and mean reversion
- Provides short-term volatility forecasts
- More sensitive to recent market conditions

#### HAR-RV (Heterogeneous Autoregressive Realized Volatility)
- Uses realized volatility from different time horizons
- More stable long-term volatility estimates
- Better for trend-following strategies

#### ATR (Average True Range)
- Simple volatility measure based on price ranges
- Used as fallback when advanced models fail
- Good for quick volatility assessment

### 3. Position Fetching (`get_position.py`)

Utilities for fetching real-time position data from Bybit.

**Features:**
- Real-time position data
- Liquidation price calculations
- PnL tracking
- Leverage information

## 📊 Understanding the Outputs

### Position Analysis Breakdown

#### 1. Current Status
```
Entry: $0.460210 | Current: $0.462700 | PnL: 0.56%
Size: 331.0 | Notional: $152.33 | Leverage: 15.0x
```
- **Entry**: Position entry price
- **Current**: Live market price
- **PnL**: Unrealized profit/loss percentage
- **Size**: Position size in base currency
- **Notional**: Position value in USDT
- **Leverage**: Current leverage used

#### 2. Volatility Analysis
```
Method: HAR-RV
ATR(20): $0.006705 (1.46% of price)
HAR-RV σ(annual): 16.2%
GARCH σ(annual): 93.3%
```
- **Method**: Primary volatility model used for calculations
- **ATR(20)**: 20-period Average True Range in dollars and percentage
- **HAR-RV σ**: Annualized volatility from HAR-RV model
- **GARCH σ**: Annualized volatility from GARCH model

#### 3. Risk Management Levels
```
STOP LOSS: $0.458600 (-0.35% from entry)
  💰 Optimal Risk: $3.08 (for optimal size: 1916.08)
  💰 Current Risk: $0.53 (for current size: 331.00)
  ✅ Safe from liquidation

TAKE PROFIT: $0.463400 (0.69% from entry)
  💰 Optimal Reward: $6.11
  💰 Current Reward: $1.06
  📊 Risk/Reward: 1.98:1
```
- **SL/TP Levels**: Calculated based on volatility and multipliers
- **Optimal Risk/Reward**: Based on optimal position size for 2% risk
- **Current Risk/Reward**: Based on actual position size
- **Risk/Reward Ratio**: Reward divided by risk (target: >1.5:1)

#### 4. Position Health Assessment
- **🟢 NORMAL**: Position within normal parameters
- **🟡 WARNING**: Position needs attention (PnL < -2%)
- **🔴 CRITICAL**: Position needs immediate action (PnL < -5%)
- **💚 PROFITABLE**: Position in profit, consider trailing stops

### Risk/Reward Calculation Logic

The system calculates risk/reward using **optimal position sizing**:

1. **Target Risk**: 2% of position notional
2. **Volatility Forecast**: Uses GARCH/HAR-RV models
3. **SL Distance**: `k × σ_H × entry_price` (k = 0.8-1.5 based on leverage)
4. **TP Distance**: `m × σ_H × entry_price` (m = 1.8-2.5 based on leverage)
5. **Optimal Size**: `target_risk / sl_distance`
6. **Risk/Reward**: `tp_distance / sl_distance`

### Position Sizing Recommendations

The system compares current vs optimal position sizes:

- **Current < 0.5× Optimal**: Position too small for proper risk management
- **Current > 1.5× Optimal**: Position too large, consider reducing
- **0.5× ≤ Current ≤ 1.5× Optimal**: Position size appropriate

## 🎛️ Configuration Options

### Risk Parameters

```python
# Leverage-based risk multipliers
if leverage >= 20:
    k_sl = 0.8  # Very tight stop for high leverage
    m_tp = 1.8  # Conservative target
elif leverage >= 15:
    k_sl = 1.0
    m_tp = 2.0
elif leverage >= 10:
    k_sl = 1.2
    m_tp = 2.2
else:
    k_sl = 1.5
    m_tp = 2.5
```

### Volatility Analysis Settings

```python
# Timeframe and lookback
timeframe = "4h"           # Data timeframe
lookback_days = 30        # Historical data period
horizon_hours = 4         # Volatility forecast horizon

# Risk target
target_risk_pct = 0.02    # 2% of position notional
```

## 📈 Usage Examples

### 1. Basic Position Analysis
```bash
python position_risk_manager.py
```

### 2. Custom Volatility Analysis
```python
from garch_vol_triggers import analyze_multiple_symbols_bybit

# Analyze multiple symbols
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
results = analyze_multiple_symbols_bybit(symbols, timeframe="4h", days_back=90)
```

### 3. Individual Symbol Analysis
```python
from garch_vol_triggers import get_klines_bybit, garch_sigma_ann_and_sigma_H

# Get data and analyze
df = get_klines_bybit("BTC/USDT", "1h", days_back=30)
sigma_ann, sigma_H, garch_res = garch_sigma_ann_and_sigma_H(df["close"])
```

## 🔍 Troubleshooting

### Common Issues

1. **"No module named 'pandas'"**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **"Error fetching data from Bybit"**
   - Check API credentials in `settings.toml`
   - Verify internet connection
   - Check if using correct symbol format (e.g., "BTC/USDT:USDT" for Bybit)

3. **"GARCH failed"**
   - Need at least 500 data points for stable GARCH fit
   - Try increasing `lookback_days` parameter

4. **"HAR-RV failed"**
   - Need sufficient historical data
   - Try different timeframe or longer lookback period

### Performance Tips

- Use `sandbox=True` for testing
- Reduce `lookback_days` for faster analysis
- Cache volatility calculations for frequently analyzed symbols

## 📚 Technical Details

### Volatility Models

#### GARCH(1,1) Model
```
σ²_t = ω + α₁r²_{t-1} + β₁σ²_{t-1}
```
Where:
- `σ²_t`: Conditional variance at time t
- `r²_{t-1}`: Squared return at time t-1
- `ω, α₁, β₁`: Model parameters

#### HAR-RV Model
```
log(RV_{t+1}) = c + β_D log(RV_D) + β_W log(RV_W) + β_M log(RV_M)
```
Where:
- `RV_D`: Daily realized volatility
- `RV_W`: Weekly realized volatility  
- `RV_M`: Monthly realized volatility

### Risk Calculation Formula

```
SL_distance = k × σ_H × entry_price
TP_distance = m × σ_H × entry_price
Optimal_Size = target_risk / SL_distance
```

Where `σ_H` is the volatility forecast for the target horizon.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Always:
- Test thoroughly on paper trading first
- Never risk more than you can afford to lose
- Verify all calculations independently
- Consider professional financial advice

The authors are not responsible for any financial losses incurred through the use of this software.
