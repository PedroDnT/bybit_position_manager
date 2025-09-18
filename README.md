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
└── utils.py                # Utility functions and helpers

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

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning the repository)

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bybit_position_manager
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package**:
   ```bash
   # For development (recommended)
   pip install -e .
   
   # Or install dependencies directly
   pip install -r requirements.txt
   ```

4. **Configure your settings**:
   ```bash
   # Create your configuration file
   cp settings.example.toml settings.toml
   # Edit with your API credentials and preferences
   nano settings.toml
   ```

5. **Run the risk manager**:
   ```bash
   risk-manager
   ```

## 📦 Installation Options

### Development Installation (Recommended)

For development and testing, install in editable mode:

```bash
pip install -e .
```

This allows you to modify the code and see changes immediately without reinstalling.

### Production Installation

For production use:

```bash
pip install .
```

### Docker Installation

You can also run the system using Docker:

```bash
# Build the container
docker build -t crypto-risk-manager .

# Run the risk manager
docker run -v $(pwd)/settings.toml:/app/settings.toml crypto-risk-manager
```

## ⚙️ Configuration

### API Credentials Setup

1. **Create your configuration file**:
   ```bash
   cp settings.example.toml settings.toml
   ```

2. **Edit the configuration** with your exchange credentials:
   ```toml
   [exchange]
   api_key = "your_api_key_here"
   api_secret = "your_api_secret_here"
   sandbox = false  # Set to true for testing
   
   [risk_management]
   max_portfolio_risk = 0.02  # 2% max portfolio risk
   default_stop_loss = 0.05   # 5% default stop loss
   ```

### Environment Variables

Alternatively, you can use environment variables:

```bash
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"
export BYBIT_SANDBOX="false"
```

## 🧪 Running Tests

Execute the test suite to ensure everything is working correctly:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_adaptive_stop_loss.py -v
pytest tests/test_garch_vol_triggers.py -v

# Run with coverage
pytest tests/ --cov=market_analysis --cov-report=html
```

## 📖 Usage Guide

### Basic Usage

Once installed and configured, you can run the risk manager with:

```bash
risk-manager
```

This will:
1. Connect to your exchange using the configured API credentials
2. Fetch all open positions
3. Analyze risk metrics for each position
4. Apply adaptive stop-loss levels where appropriate
5. Generate a comprehensive risk analysis report
6. Save results to `risk_analysis.json`

### Command Line Options

```bash
# Run with verbose output
risk-manager --verbose

# Run in dry-run mode (no actual trades)
risk-manager --dry-run

# Specify custom configuration file
risk-manager --config custom_settings.toml

# Show help
risk-manager --help
```

### Programmatic Usage

You can also use the risk manager programmatically:

```python
from market_analysis.position_risk_manager import PositionRiskManager
from market_analysis.config import load_config

# Load configuration
config = load_config()

# Initialize risk manager
risk_manager = PositionRiskManager(config)

# Run analysis
results = risk_manager.analyze_positions()

# Access individual components
for position in results['positions']:
    print(f"Symbol: {position['symbol']}")
    print(f"Risk Level: {position['risk_level']}")
    print(f"Recommended Action: {position['recommendation']}")
```

### Understanding the Output

The risk manager generates several types of output:

#### Console Output
- **Position Summary**: Overview of all open positions
- **Risk Metrics**: Key risk indicators for your portfolio
- **Recommendations**: Actionable suggestions for risk management
- **Quick Reference**: Summary table of important metrics

#### JSON Report (`risk_analysis.json`)
- Detailed position analysis
- Volatility forecasts
- Correlation matrices
- Historical performance metrics

#### Example Output
```
📊 PORTFOLIO RISK ANALYSIS
==========================

💰 Account Overview:
   Total Equity: $10,000.00
   Available Balance: $8,500.00
   Today's PnL: +$150.00 (+1.5%)

📈 Open Positions (3):
   BTC/USDT: $2,000 (Long) - Risk: Medium
   ETH/USDT: $1,500 (Short) - Risk: Low  
   SOL/USDT: $1,000 (Long) - Risk: High

⚠️  Risk Recommendations:
   • Consider reducing SOL position size
   • BTC correlation with ETH detected
   • Portfolio risk: 2.1% (within limits)

✅ Adaptive stop-losses applied to 3 positions
```

## 🔧 Core Components & Logic Flow

### 1. Position Risk Manager (`position_risk_manager.py`)

The main system orchestrates the entire risk analysis workflow:

**Logic Flow:**
```
1. Initialize → Load Configuration → Fetch Positions
2. For each position:
   a. Analyze volatility (GARCH + HAR-RV + ATR blend)
   b. Calculate optimal SL/TP levels
   c. Determine position sizing recommendations
   d. Assess position health
3. Calculate portfolio metrics
4. Apply correlation analysis and cluster risk caps
5. Generate comprehensive report
6. Export to JSON
```

**Key Features:**
- **Configuration Loading**: Dynamic settings with fallback to defaults
- **Multi-Model Volatility**: Blends GARCH, HAR-RV, and ATR for robust estimates
- **Dynamic Risk Parameters**: Adjusts SL/TP multipliers based on leverage
- **Portfolio Correlation**: Identifies correlated positions and caps cluster risk
- **Health Assessment**: Categorizes positions as NORMAL/WARNING/CRITICAL/PROFITABLE
- **JSON Export**: Saves analysis for external processing

**Output Example:**
```
Position 1: ARB/USDT:USDT
----------------------------------------
Current Status:
  Entry: $0.460210 | Current: $0.462700 | PnL: 0.56%
  Size: 331.0 | Notional: $152.33 | Leverage: 15.0x

Volatility Analysis:
  Method: VOL_BLEND (GARCH 30% + HAR 40% + ATR 30%)
  ATR(20): $0.006705 (1.46% of price)
  HAR-RV σ(annual): 16.2%
  GARCH σ(annual): 93.3%
  Blended σ(4h): 2.1%

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

Risk Assessment:
  Status: 🟢 NORMAL
  Action: Set SL/TP as recommended
```

### 🧠 Advanced Risk Logic Explained

Beyond simple volatility metrics, the system employs a multi-layered approach to dynamically adjust risk parameters based on market conditions and trade confidence. This results in more nuanced and context-aware risk management.

#### 1. Confidence Scoring Model

Instead of treating all trade setups equally, the system calculates a **Confidence Score** for each position to quantify the quality of the setup. This score is based on a blend of five distinct factors:

1.  **Trend Strength (EMA Crossover):** Checks if the short-term trend (20-period EMA) is aligned with the position's direction relative to the longer-term trend (50-period EMA). A score is awarded if the trend is favorable.
2.  **Breakout Confirmation (Donchian Channels):** Determines if the current price has recently broken out of its 20-period price range, providing confirmation for the trade's direction.
3.  **Volatility Regime:** Compares the short-term volatility (20-period standard deviation of returns) to the longer-term median volatility (100-period). A score is awarded for low-volatility regimes (less noise) and penalized for high-volatility regimes.
4.  **Price Momentum (RSI Proxy):** Uses a 14-period RSI calculation to gauge momentum. A score is awarded if the RSI is above 50 for long positions or below 50 for short positions.
5.  **Volatility Model Stability:** Compares the annualized volatility forecasts from the GARCH and HAR-RV models. If the models are in close agreement, it increases confidence. If they diverge significantly, it reduces confidence, indicating market uncertainty.

The final score is clamped between -2 and +5 and directly influences the risk parameters.

#### 2. Dynamic Risk Target Adjustment

The system can dynamically adjust the percentage of capital risked on a trade based on the **Confidence Score**. This allows for taking slightly more risk on high-quality setups and less risk on low-quality ones.

-   The base risk is defined in `settings.toml` (e.g., `base_target_pct = 0.025`).
-   A multiplier is applied based on the score:
    -   High Confidence (Score ≥ 4): **1.2x** multiplier (e.g., 3.0% risk)
    -   Medium Confidence (Score ≥ 2): **1.0x** multiplier (e.g., 2.5% risk)
    -   Low Confidence (Score ≥ 0): **0.9x** multiplier (e.g., 2.25% risk)
    -   Negative Confidence (Score < 0): **0.8x** multiplier (e.g., 2.0% risk)
-   The final risk target is clipped within a professional range (e.g., 2.0% to 3.0%) defined in the configuration.

#### 3. Dynamic Stop-Loss and Take-Profit Multipliers

The multipliers used to set the Stop-Loss (`k`) and Take-Profit (`m`) distances are not static. they are adjusted using a three-factor model to adapt to market conditions:

1.  **Base Multiplier (Leverage):** The initial `k` and `m` values are selected from the configuration based on the position's leverage. Higher leverage results in tighter base multipliers.
2.  **Volatility Adjustment:** The multipliers are then adjusted based on the current volatility regime (measured by ATR as a percentage of price). In very high-volatility environments, stops are widened to avoid premature stop-outs, while in low-volatility environments, they are tightened.
3.  **Confidence Adjustment:** Finally, the multipliers are fine-tuned based on the **Confidence Score**. A higher score results in slightly tighter stops and more aggressive profit targets, as the system has more confidence in the trade's direction.

This multi-factor approach ensures that the final SL/TP levels are tailored specifically to the asset's current leverage, volatility, and the quality of the trade setup.

### 2. GARCH Volatility Triggers (`garch_vol_triggers.py`)

Advanced volatility analysis using multiple models:

**Volatility Models:**

#### GARCH(1,1) Model
- Models volatility clustering and mean reversion
- Provides short-term volatility forecasts
- More sensitive to recent market conditions
- Weight: 30% in blended estimate

#### HAR-RV (Heterogeneous Autoregressive Realized Volatility)
- Uses realized volatility from different time horizons
- More stable long-term volatility estimates
- Better for trend-following strategies
- Weight: 40% in blended estimate

#### ATR (Average True Range)
- Simple volatility measure based on price ranges
- Used as fallback when advanced models fail
- Good for quick volatility assessment
- Weight: 30% in blended estimate

**Blending Logic:**
```python
# Outlier detection and blending
if garch_sigma and har_sigma:
    ratio = garch_sigma / har_sigma
    if ratio > outlier_threshold:
        # Use HAR if GARCH is outlier
        blended_sigma = har_sigma
    else:
        # Weighted blend
        blended_sigma = (w_garch * garch_sigma + 
                        w_har * har_sigma + 
                        w_atr * atr_sigma)
else:
    # Fallback to ATR
    blended_sigma = atr_sigma
```

### 3. Portfolio Correlation Analysis

**New Feature**: Automatically identifies correlated positions and applies risk caps:

```python
# Correlation clustering algorithm
1. Fetch 4h returns for all positions (60-day lookback)
2. Calculate correlation matrix
3. Group positions with |correlation| ≥ 0.7 into clusters
4. Cap total cluster risk at 50% of portfolio risk budget
5. Scale down cluster members proportionally 
```

**Example Output:**
```
📊 PORTFOLIO SUMMARY
----------------------------------------
Total Positions: 5
Total Notional: $2,450.33
Total Unrealized PnL: $45.67
Total Risk (if all SL hit): $89.23
Total Reward (if all TP hit): $156.78
Portfolio Risk/Reward: 1.76:1

⚠️  Positions at Risk: BTC/USDT:USDT, ETH/USDT:USDT
```

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
Method: VOL_BLEND (GARCH 30% + HAR 40% + ATR 30%)
ATR(20): $0.006705 (1.46% of price)
HAR-RV σ(annual): 16.2%
GARCH σ(annual): 93.3%
Blended σ(4h): 2.1%
```
- **Method**: Shows the blending approach used
- **ATR(20)**: 20-period Average True Range in dollars and percentage
- **HAR-RV σ**: Annualized volatility from HAR-RV model
- **GARCH σ**: Annualized volatility from GARCH model
- **Blended σ**: Final volatility estimate used for calculations

#### 3. Risk Management Levels
```
STOP LOSS: $0.458600 (-0.35% from entry)
  💰 Optimal Risk: $3.08 (for optimal size: 1916.08)
  💰 Current Risk: $0.53 (for current size: 331.00)
  ✅ Safe from liquidation

TAKE PROFIT: $0.463400 (0.69% from entry)
  💰 Optimal Reward: $6.1/1
  💰 Current Reward: $1.06
  📊 Risk/Reward: 1.98:1
```
- **SL/TP Levels**: Calculated based on volatility and multipliers
- **Optimal Risk/Reward**: Based on optimal position size for target risk
- **Current Risk/Reward**: Based on actual position size
- **Risk/Reward Ratio**: Reward divided by risk (target: >1.5:1)

#### 4. Position Health Assessment
- **🟢 NORMAL**: Position within normal parameters
- **🟡 WARNING**: Position needs attention (PnL < -2%)
- **🔴 CRITICAL**: Position needs immediate action (PnL < -5%)
- **💚 PROFITABLE**: Position in profit, consider trailing stops

### Risk/Reward Calculation Logic

The system calculates risk/reward using **optimal position sizing**:

1. **Target Risk**: Configurable (default 2.5% of position notional)
2. **Volatility Forecast**: Uses blended GARCH/HAR-RV/ATR models
3. **SL Distance**: `k × σ_H × entry_price` (k = 0.8-1.8 based on leverage)
4. **TP Distance**: `m × σ_H × entry_price` (m = 1.8-4.0 based on leverage)
5. **Optimal Size**: `target_risk / sl_distance`
6. **Risk/Reward**: `tp_distance / sl_distance`

### Position Sizing Recommendations

The system compares current vs optimal position sizes:

- **Current < 0.5× Optimal**: Position too small for proper risk management
- **Current > 1.5× Optimal**: Position too large, consider reducing
- **0.5× ≤ Current ≤ 1.5× Optimal**: Position size appropriate

## 🎛️ Configuration Options

The system uses a TOML configuration file (`settings.toml`) for comprehensive customization. Below are all available configuration sections:

### Exchange Configuration

```toml
[exchange]
api_key = "your_api_key_here"
api_secret = "your_api_secret_here"
sandbox = false                    # Use testnet/sandbox environment
testnet = false                    # Alternative sandbox setting
rate_limit = 10                    # API calls per second limit
timeout = 30                       # Request timeout in seconds
```

### Risk Management Parameters

```toml
[risk_management]
max_portfolio_risk = 0.02          # Maximum portfolio risk (2%)
position_size_limit = 0.1          # Maximum position size as % of portfolio (10%)
correlation_threshold = 0.7        # Correlation threshold for position clustering
cluster_risk_multiplier = 0.8      # Risk multiplier for correlated positions
default_stop_loss = 0.05           # Default stop loss percentage (5%)
max_leverage = 10                  # Maximum allowed leverage
risk_free_rate = 0.02              # Risk-free rate for Sharpe ratio calculations
```

### Risk Parameters

```toml
[risk]
base_target_pct = 0.025      # Base risk target (2.5%)
min_target_pct = 0.020       # Minimum risk target (2.0%)
max_target_pct = 0.030       # Maximum risk target (3.0%)
use_dynamic = true           # Enable dynamic risk adjustment
exchange_timeout_ms = 10000  # HTTP timeout for exchange calls (10 seconds)

[stops]
# Leverage-based SL multipliers (professional 2–4× ATR range)
k_sl_lev20 = 1.5            # High leverage: tighter stops
k_sl_lev15 = 1.8            # Medium-high leverage
k_sl_lev10 = 2.2            # Medium leverage
k_sl_low   = 2.5            # Low leverage: wider stops allowed

# Leverage-based TP multipliers
m_tp_lev20 = 3.0            # Conservative target for high leverage
m_tp_lev15 = 3.5            # Medium-high leverage
m_tp_lev10 = 4.0            # Medium leverage
m_tp_low   = 4.5            # Aggressive target for low leverage
```

### Adaptive Stop-Loss Settings

```toml
[adaptive_stop_loss]
enabled = true                     # Enable adaptive stop-loss functionality
min_stop_distance = 0.01           # Minimum stop distance (1%)
max_stop_distance = 0.15           # Maximum stop distance (15%)
volatility_multiplier = 2.0        # ATR multiplier for stop distance
trailing_enabled = true            # Enable trailing stops
update_frequency = 300             # Update frequency in seconds
```

### Volatility Analysis Settings

```toml
[vol]
blend_w_garch = 0.30        # GARCH weight in blend
blend_w_har   = 0.40        # HAR-RV weight in blend
blend_w_atr   = 0.30        # ATR weight in blend
garch_har_outlier_ratio = 2.0  # Outlier detection threshold
horizon_hours = 4           # Volatility forecast horizon

[volatility]
garch_window = 252                 # GARCH model lookback window (days)
confidence_level = 0.95            # Confidence level for VaR calculations
volatility_threshold = 0.3         # High volatility threshold (30%)
forecast_horizon = 5               # Volatility forecast horizon (days)
min_observations = 50              # Minimum data points for analysis
```

### Portfolio Correlation Settings

```toml
[portfolio]
corr_lookback_days = 60     # Days for correlation calculation
corr_threshold = 0.7        # Correlation threshold for clustering
cluster_risk_cap_pct = 0.5  # Max risk per cluster (% of total)

[correlation]
lookback_period = 30               # Correlation calculation period (days)
min_correlation = 0.5              # Minimum correlation for clustering
update_frequency = "1h"            # How often to update correlations
max_cluster_size = 5               # Maximum positions per correlation cluster
correlation_decay = 0.94           # Exponential decay factor for correlation
```

### Reporting and Output

```toml
[reporting]
output_format = "json"             # Output format: json, csv, html
save_to_file = true                # Save analysis to file
file_path = "risk_analysis.json"   # Output file path
include_charts = false             # Include charts in HTML output
verbose_logging = false            # Enable detailed logging
```

### Data Sources and Market Data

```toml
[data]
price_source = "exchange"          # Price data source: exchange, external
historical_days = 365              # Days of historical data to fetch
cache_enabled = true               # Enable data caching
cache_duration = 3600              # Cache duration in seconds
backup_data_source = "coingecko"   # Backup data source
```

### Advanced Settings

```toml
[advanced]
parallel_processing = true         # Enable parallel position analysis
max_workers = 4                    # Maximum worker threads
memory_limit = "1GB"               # Memory usage limit
performance_monitoring = false     # Enable performance metrics
debug_mode = false                 # Enable debug output
```

### Notification Settings

```toml
[notifications]
enabled = false                    # Enable notifications
webhook_url = ""                   # Discord/Slack webhook URL
email_enabled = false              # Enable email notifications
email_smtp_server = ""             # SMTP server for emails
risk_threshold_alerts = true       # Alert on high risk positions
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

### 4. Configuration Management
```python
from position_risk_manager import load_settings

# Load configuration with fallback
cfg = load_settings("settings.toml")
risk_target = cfg.get('risk', {}).get('base_target_pct', 0.025)
```

## 🚨 Troubleshooting

### Common Issues

#### API Connection Errors
```
Error: Failed to connect to exchange API
```
**Solutions:**
- Verify your API credentials in `settings.toml`
- Check if your IP is whitelisted on the exchange
- Ensure you have the correct permissions (read positions, read account)
- Try using sandbox mode first: `sandbox = true`
- Check your internet connection and firewall settings

#### Authentication Errors
```
Error: Invalid API key or signature
```
**Solutions:**
- Double-check your API key and secret in the configuration
- Ensure there are no extra spaces or characters
- Verify the API key has the required permissions
- Check if the API key is expired or disabled

#### No Positions Found
```
Warning: No open positions found
```
**Solutions:**
- Verify you have open positions on the exchange
- Check if you're using the correct account (main vs. sub-account)
- Ensure the exchange module is correctly configured
- Try running in sandbox mode to test connectivity

#### GARCH Model Errors
```
Error: GARCH model failed to converge
```
**Solutions:**
- Insufficient historical data: The model needs at least 50 data points
- High volatility in data: Try adjusting the `garch_window` parameter
- Convergence issues: Reduce the forecast horizon or increase the window size
- Check for data quality issues (missing values, outliers)

#### Configuration Issues
```
Error: Configuration file not found
```
**Solutions:**
- Copy `settings.example.toml` to `settings.toml`
- Ensure all required fields are filled
- Check TOML syntax for any formatting errors
- Validate configuration with: `python -c "import tomli; tomli.load(open('settings.toml', 'rb'))"`

#### Memory and Performance Issues
```
Error: Out of memory or slow performance
```
**Solutions:**
- Reduce the `garch_window` size
- Limit the number of positions analyzed
- Disable parallel processing: `parallel_processing = false`
- Increase memory limit in configuration
- Close other applications to free up memory

#### Adaptive Stop-Loss Errors
```
Error: Failed to calculate adaptive levels
```
**Solutions:**
- Check if market data is available for the symbol
- Verify ATR calculation has sufficient data
- Ensure position data contains required fields
- Try disabling adaptive stop-loss temporarily

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Environment variable
export DEBUG=1
risk-manager

# Or in configuration
[advanced]
debug_mode = true
verbose_logging = true
```

### Log Analysis

Check the log files for detailed error information:

```bash
# View recent logs
tail -f risk_manager.log

# Search for specific errors
grep -i "error" risk_manager.log
grep -i "failed" risk_manager.log
```

### Performance Optimization

If the system is running slowly:

1. **Reduce data requirements:**
   ```toml
   [volatility]
   garch_window = 100  # Reduce from 252
   
   [correlation]
   lookback_period = 14  # Reduce from 30
   ```

2. **Enable caching:**
   ```toml
   [data]
   cache_enabled = true
   cache_duration = 3600
   ```

3. **Optimize parallel processing:**
   ```toml
   [advanced]
   parallel_processing = true
   max_workers = 2  # Adjust based on your CPU
   ```

### Getting Help

If you encounter issues:

1. **Check the logs** for detailed error messages
2. **Verify your configuration** matches the expected format
3. **Test with minimal settings** to isolate the problem
4. **Check GitHub Issues** for similar problems
5. **Create a new issue** with:
   - Error message and full stack trace
   - Your configuration (remove sensitive data)
   - System information (OS, Python version)
   - Steps to reproduce the issue

### Reporting Bugs

When reporting bugs, please include:

```bash
# System information
python --version
pip list | grep -E "(ccxt|pandas|numpy|arch)"

# Configuration (sanitized)
cat settings.toml | sed 's/api_key.*/api_key = "***"/' | sed 's/api_secret.*/api_secret = "***"/'

# Error logs
tail -50 risk_manager.log
```

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

### Correlation Analysis Algorithm

```
1. Fetch 4h returns for all positions (configurable lookback)
2. Calculate pairwise correlation matrix
3. Apply threshold-based clustering:
   - Start with each symbol as its own cluster
   - Merge clusters if any member has |corr| ≥ threshold
4. For each cluster:
   - Calculate total cluster risk
   - If cluster risk > cap_pct × total_risk:
     - Scale down all cluster members proportionally
```

## 🤝 Contributing

We welcome contributions to improve the Advanced Crypto Risk Manager! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### 🚀 Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/bybit_position_manager.git
   cd bybit_position_manager
   ```
3. **Set up development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .[test]
   ```

### 🛠️ Development Setup

#### Prerequisites
- Python 3.10+
- Git
- A code editor (VS Code, PyCharm, etc.)

#### Environment Setup
```bash
# Install development dependencies
pip install -e .[test]

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# Verify installation
pytest --version
python -c "import market_analysis; print('✅ Installation successful')"
```

#### Development Tools
```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 mypy

# Testing
pip install pytest pytest-cov pytest-mock
```

### 📝 Development Workflow

#### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
# or  
git checkout -b docs/documentation-improvement
```

#### 2. Make Your Changes
- Follow the existing code style and patterns
- Add type hints to new functions
- Include docstrings for public methods
- Update tests for modified functionality

#### 3. Test Your Changes
```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_adaptive_stop_loss.py -v

# Run with coverage
pytest --cov=market_analysis --cov-report=html

# Test specific functionality
python -m market_analysis.demo_adaptive_stop_loss
```

#### 4. Code Quality Checks
```bash
# Format code
black market_analysis/ tests/
isort market_analysis/ tests/

# Lint code
flake8 market_analysis/ tests/
mypy market_analysis/

# Check for common issues
python -m py_compile market_analysis/*.py
```

### 📋 Contribution Guidelines

#### Code Style
- **Follow PEP 8** guidelines
- **Use type hints** for function parameters and return values
- **Add docstrings** to all public functions and classes
- **Keep functions focused** and modular (single responsibility)
- **Use meaningful variable names** and avoid abbreviations

#### Example Code Style
```python
from typing import Dict, List, Optional
import pandas as pd

def calculate_position_risk(
    position: Dict[str, float], 
    market_data: pd.DataFrame,
    risk_params: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate risk metrics for a trading position.
    
    Args:
        position: Position data containing size, entry price, etc.
        market_data: Historical market data for volatility calculation
        risk_params: Optional risk parameters override
        
    Returns:
        Dictionary containing calculated risk metrics
        
    Raises:
        ValueError: If position data is invalid
    """
    # Implementation here
    pass
```

#### Testing Guidelines
- **Write tests** for all new functionality
- **Maintain test coverage** above 80%
- **Use descriptive test names** that explain what is being tested
- **Include edge cases** and error conditions
- **Mock external dependencies** (API calls, file I/O)

#### Example Test
```python
import pytest
from unittest.mock import Mock, patch
from market_analysis.position_risk_manager import PositionRiskManager

def test_calculate_position_risk_with_valid_data():
    """Test position risk calculation with valid input data."""
    # Arrange
    position = {"symbol": "BTC/USDT", "size": 1000, "entryPrice": 50000}
    market_data = pd.DataFrame({"close": [50000, 51000, 49000]})
    
    # Act
    result = calculate_position_risk(position, market_data)
    
    # Assert
    assert "risk_score" in result
    assert result["risk_score"] > 0
    assert result["risk_score"] <= 1
```

### 🎯 Areas for Contribution

#### High Priority
- **🔄 Exchange Support**: Add support for Binance, OKX, Kraken
- **📊 Risk Models**: Implement VaR, CVaR, maximum drawdown models
- **⚡ Performance**: Optimize GARCH calculations and data processing
- **🧪 Testing**: Increase test coverage and add integration tests

#### Medium Priority
- **📱 UI/UX**: Improve console output and add web dashboard
- **📈 Visualization**: Add charts and graphs for risk analysis
- **🔔 Notifications**: Implement email, Slack, Discord alerts
- **📚 Documentation**: Add tutorials and advanced usage examples

#### Low Priority
- **🐳 Docker**: Improve containerization and deployment
- **☁️ Cloud**: Add cloud deployment options (AWS, GCP, Azure)
- **🔌 Plugins**: Create plugin system for custom risk models
- **📊 Reporting**: Enhanced PDF and HTML report generation

### 🔍 Code Review Process

#### Before Submitting
1. **Self-review** your code changes
2. **Run the full test suite** and ensure all tests pass
3. **Check code coverage** hasn't decreased significantly
4. **Update documentation** if you've changed APIs or added features
5. **Test manually** with real or simulated data

#### Pull Request Guidelines
Your PR should include:
- **Clear title** describing the change
- **Detailed description** of what was changed and why
- **Link to related issues** if applicable
- **Screenshots** for UI changes
- **Breaking changes** clearly documented
- **Test results** showing all tests pass

#### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### 🐛 Bug Reports

When reporting bugs, please include:
- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, package versions)
- **Error messages** and stack traces
- **Configuration** (sanitized, no API keys)

### 💡 Feature Requests

For new features, please provide:
- **Use case description** - why is this needed?
- **Proposed solution** - how should it work?
- **Alternative solutions** - what other approaches were considered?
- **Additional context** - any relevant examples or references

### 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: Tag maintainers for review help
- **Documentation**: Check existing docs and examples first

### 🏆 Recognition

Contributors will be:
- **Listed in CONTRIBUTORS.md**
- **Mentioned in release notes** for significant contributions
- **Given credit** in documentation for major features

Thank you for contributing to make crypto trading safer and more efficient! 🚀

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Advanced Crypto Risk Manager

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ⚠️ Disclaimer

**IMPORTANT: READ CAREFULLY BEFORE USING THIS SOFTWARE**

### Financial Risk Warning
- **This software is for educational and informational purposes only**
- **Not intended as financial advice or investment recommendations**
- **Trading cryptocurrencies involves substantial risk of loss**
- **Past performance does not guarantee future results**
- **You may lose some or all of your invested capital**

### Software Limitations
- **No warranty or guarantee of accuracy**
- **Market conditions can change rapidly**
- **Technical analysis has inherent limitations**
- **Software bugs or errors may occur**
- **Internet connectivity issues may affect performance**

### User Responsibilities
- **Conduct your own research and due diligence**
- **Understand the risks before trading**
- **Never invest more than you can afford to lose**
- **Consider consulting qualified financial advisors**
- **Test thoroughly with small amounts first**
- **Keep your API keys and credentials secure**

### Liability Disclaimer
The authors, contributors, and maintainers of this project:
- **Are not responsible for any financial losses**
- **Do not guarantee software performance or accuracy**
- **Provide this software "as is" without warranties**
- **Are not liable for any damages or losses**
- **Do not provide financial or investment advice**

**USE AT YOUR OWN RISK**

## 🔗 Additional Resources

### Documentation
- [API Documentation](docs/api.md) - Detailed API reference
- [Configuration Guide](docs/configuration.md) - Advanced configuration options
- [Risk Models Guide](docs/risk-models.md) - Understanding risk calculations
- [Exchange Integration](docs/exchanges.md) - Supported exchanges and setup

### Community
- [GitHub Discussions](https://github.com/your-username/bybit_position_manager/discussions) - Community Q&A
- [Issue Tracker](https://github.com/your-username/bybit_position_manager/issues) - Bug reports and feature requests
- [Wiki](https://github.com/your-username/bybit_position_manager/wiki) - Additional guides and tutorials

### Related Projects
- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange trading library
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical analysis library
- [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation
- [NumPy](https://numpy.org/) - Numerical computing

### Support
- 📧 **Email**: support@example.com
- 💬 **Discord**: [Join our server](https://discord.gg/example)
- 🐦 **Twitter**: [@CryptoRiskMgr](https://twitter.com/example)
- 📖 **Blog**: [Risk Management Insights](https://blog.example.com)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**🔔 Watch for updates and new features**

**🍴 Fork to contribute your improvements**

Made with ❤️ for the crypto trading community

</div>
