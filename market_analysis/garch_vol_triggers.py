# pip installs (uncomment if needed)
# !pip install pandas numpy ccxt arch

import math
import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import ccxt

from .config import settings

# ---------- CCXT Bybit Data Fetcher ----------

def get_live_price_bybit(exchange: ccxt.Exchange, symbol="BTC/USDT"):
    """
    Get real-time live price from Bybit using ticker endpoint.
    
    Returns: float with current live price
    """
    try:
        # Fetch current ticker for live price
        ticker = exchange.fetch_ticker(symbol)
        live_price = float(ticker['last'])
        return live_price
        
    except Exception as e:
        print(f"Error fetching live price from Bybit: {e}")
        return None

def get_klines_bybit(exchange: ccxt.Exchange, symbol="BTC/USDT", timeframe="4h", since=None, limit=1000):
    """
    Fetch OHLCV data from Bybit using ccxt.
    
    exchange: An initialized ccxt.Exchange object
    symbol: ccxt format like "BTC/USDT", "ETH/USDT"
    timeframe: ccxt format like "1m", "5m", "15m", "1h", "4h", "1d"
    since: datetime object or timestamp in ms, or None for recent data
    limit: max number of candles (Bybit max ~1000 per request)
    
    Returns: DataFrame with time-indexed OHLCV
    """
    # Convert datetime to timestamp if needed
    if isinstance(since, dt.datetime):
        since = int(since.timestamp() * 1000)
    
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
        if not ohlcv:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime and set as index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(None)
        df = df.set_index('datetime')
        df = df.drop('timestamp', axis=1)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data from Bybit: {e}")
        return pd.DataFrame()

def get_multiple_timeframes_bybit(exchange: ccxt.Exchange, symbol="BTC/USDT", timeframes=["4h"], days_back=120):
    """
    Fetch multiple timeframes for the same symbol from Bybit.
    
    Returns: dict with timeframe as key, DataFrame as value
    """
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(days=days_back)
    
    data = {}
    for tf in timeframes:
        print(f"Fetching {symbol} {tf} data from Bybit...")
        df = get_klines_bybit(exchange, symbol, tf, since=start_time, limit=1000)
        if not df.empty:
            data[tf] = df
        time.sleep(0.1)  # Rate limiting
    
    return data

# ---------- Enhanced Bybit Market Info ----------

def get_bybit_market_info(exchange: ccxt.Exchange, symbol="BTC/USDT"):
    """
    Get market information like tick size, min quantity, etc. from Bybit
    """
    try:
        markets = exchange.load_markets()
        if symbol in markets:
            market = markets[symbol]
            return {
                'symbol': symbol,
                'base': market['base'],
                'quote': market['quote'],
                'type': market['type'],  # 'spot', 'swap', 'future'
                'tick_size': market['precision']['price'],
                'min_quantity': market['limits']['amount']['min'],
                'max_quantity': market['limits']['amount']['max'],
                'min_cost': market['limits']['cost']['min'],
                'contract_size': market.get('contractSize', 1),
                'maker_fee': market['maker'],
                'taker_fee': market['taker'],
            }
    except Exception as e:
        print(f"Error getting market info: {e}")
    
    return None

# ---------- ATR computation (same as before) ----------

def compute_atr(df, period=14):
    """
    df: DataFrame with columns: high, low, close
    Returns: Series ATR aligned with df index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

# ---------- Realized volatility + HAR-RV (standardized by timeframe) ----------

def realized_variance_close_to_close(prices: pd.Series):
    """
    Close-to-close realized variance proxy using log returns.
    Returns RV per bar (r^2), and annualized volatility if needed externally.
    """
    rets = np.log(prices).diff()
    rv = rets**2
    return rv, rets

def realized_vol_annualized_from_bar_sigma(bar_sigma, bars_per_year):
    """
    Convert per-bar sigma to annualized sigma using sqrt(time).
    """
    return bar_sigma * math.sqrt(bars_per_year)

def bars_per_year_from_interval(interval: str):
    """
    Rough conversion for crypto (24/7).
    """
    m = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
         "12h": 720, "1d": 1440}
    if interval not in m:
        raise ValueError("Unsupported interval for bars_per_year.")
    minutes = m[interval]
    bars_per_day = 24*60 // minutes
    bars_per_year = 365 * bars_per_day
    return bars_per_year

def bars_per_day_from_interval(interval: str) -> int:
    """Number of bars per 24h day for a given interval (24/7)."""
    m = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
         "12h": 720, "1d": 1440}
    if interval not in m:
        raise ValueError("Unsupported interval for bars_per_day.")
    minutes = m[interval]
    return 24*60 // minutes

def har_rv_nowcast(rv_series: pd.Series, *, bars_per_day: int):
    """
    Simple HAR-RV with timeframe-aware windows:
    RV_{t+1} = c + bD*RV_D + bW*RV_W + bM*RV_M
    Where RV_D is mean RV over one day of bars, RV_W over 7 days, RV_M over ~30 days.
    Returns: nowcast of next-bar RV (and sigma).
    """
    eps = 1e-12
    y = np.log(rv_series.shift(-1) + eps)
    D = int(bars_per_day)
    W = int(bars_per_day * 7)
    M = int(bars_per_day * 30)
    X = pd.DataFrame(index=rv_series.index)
    X["RV_D"] = np.log(rv_series.rolling(window=D, min_periods=D).mean() + eps)
    X["RV_W"] = np.log(rv_series.rolling(window=W, min_periods=W).mean() + eps)
    X["RV_M"] = np.log(rv_series.rolling(window=M, min_periods=M).mean() + eps)

    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 200:
        latest_rv = rv_series.iloc[-1]
        return latest_rv, math.sqrt(latest_rv)

    Y = df.iloc[:, 0].values
    Xmat = np.column_stack([np.ones(len(df)), df["RV_D"].values, df["RV_W"].values, df["RV_M"].values])

    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0]

    # Build last sample with proper windows
    rv_rolling = rv_series.rolling
    last_D = float(rv_rolling(D).mean().iloc[-1])
    last_W = float(rv_rolling(W).mean().iloc[-1])
    last_M = float(rv_rolling(M).mean().iloc[-1])
    last = pd.DataFrame({
        "RV_D": [np.log(last_D + eps)],
        "RV_W": [np.log(last_W + eps)],
        "RV_M": [np.log(last_M + eps)]
    })
    X_last = np.array([1.0, last["RV_D"].iloc[0], last["RV_W"].iloc[0], last["RV_M"].iloc[0]])
    log_rv_next = float(X_last @ beta)
    rv_next = max(np.exp(log_rv_next) - eps, eps)
    sigma_next = math.sqrt(rv_next)
    return rv_next, sigma_next

def sigma_ann_and_sigma_H_from_har(prices: pd.Series, interval="1h", horizon_hours=4):
    """
    Compute annualized sigma from HAR-RV nowcast and scale to horizon H in hours.
    """
    rv, rets = realized_variance_close_to_close(prices)
    rv = rv.dropna()
    # Use timeframe-aware HAR windows
    bpd = bars_per_day_from_interval(interval)
    rv_next, sigma_bar = har_rv_nowcast(rv, bars_per_day=bpd)
    bars_year = bars_per_year_from_interval(interval)
    sigma_ann = realized_vol_annualized_from_bar_sigma(sigma_bar, bars_year)

    hours_per_year = 24*365
    sigma_H = sigma_ann * math.sqrt(horizon_hours / hours_per_year)
    return sigma_ann, sigma_H

# ---------- GARCH(1,1) via arch package (same as before) ----------

from arch import arch_model

def garch_sigma_ann_and_sigma_H(prices: pd.Series, interval="1h", horizon_hours=4):
    """
    Fit GARCH(1,1) to log returns and produce annualized sigma and horizon-scaled sigma_H.
    """
    rets = 100 * np.log(prices).diff().dropna()
    if len(rets) < 500:
        raise ValueError("Need at least 500 returns for a stable GARCH fit.")

    am = arch_model(rets, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    f = res.forecast(horizon=1, reindex=False)
    var_next = float(f.variance.values[-1, 0])
    sigma_bar = math.sqrt(var_next) / 100.0

    bars_year = bars_per_year_from_interval(interval)
    sigma_ann = realized_vol_annualized_from_bar_sigma(sigma_bar, bars_year)

    hours_per_year = 24*365
    sigma_H = sigma_ann * math.sqrt(horizon_hours / hours_per_year)
    return sigma_ann, sigma_H, res

# ---------- New: Volatility blending & outlier clamp ----------

HOURS_PER_YEAR = 365 * 24

def _ann_to_horizon_sigma(sig_ann: float, horizon_hours: int) -> float:
    return float(sig_ann) * math.sqrt(horizon_hours / HOURS_PER_YEAR)

def blended_sigma_h(sigma_ann_garch: float | None,
                    sigma_ann_har: float | None,
                    atr_abs: float,
                    price: float,
                    cfg: dict,
                    *,
                    bar_hours: int) -> float:
    """
    Blend horizon-volatilities from GARCH, HAR, and ATR into a single horizon sigma in ABSOLUTE price units.
    Internally:
    - Convert GARCH/HAR annualized sigma to horizon FRACTIONAL sigma.
    - Convert ATR per-bar absolute move to horizon FRACTIONAL sigma via sqrt(N) with N = horizon_hours / bar_hours,
      optionally scaled by atr_to_sigma_factor for calibration.
    - Blend the FRACTIONAL components by weights, then multiply by price to return ABSOLUTE horizon sigma.
    """
    vol_cfg = (cfg or {}).get('vol', {})
    horizon_hours = int(vol_cfg.get('horizon_hours', 4))
    w_g = float(vol_cfg.get('blend_w_garch', 0.30))
    w_h = float(vol_cfg.get('blend_w_har', 0.40))
    w_a = float(vol_cfg.get('blend_w_atr', 0.30))
    outlier_ratio = float(vol_cfg.get('garch_har_outlier_ratio', 2.0))
    atr_to_sigma_factor = float(vol_cfg.get('atr_to_sigma_factor', 1.0))

    # FRACTIONAL horizon sigmas
    sigH_garch_frac = _ann_to_horizon_sigma(sigma_ann_garch, horizon_hours) if sigma_ann_garch else None
    sigH_har_frac   = _ann_to_horizon_sigma(sigma_ann_har,   horizon_hours) if sigma_ann_har else None

    # ATR per bar -> fractional per bar -> horizon fractional via sqrt(N)
    if price > 0 and bar_hours > 0:
        sigma_bar_atr_frac = (float(atr_abs) / float(price)) * atr_to_sigma_factor
        n_bars = max(horizon_hours / float(bar_hours), 1e-9)
        sigH_atr_frac = sigma_bar_atr_frac * math.sqrt(n_bars)
    else:
        sigH_atr_frac = 0.0

    # Outlier downweight for GARCH vs HAR in annualized domain (uses input annualized sigmas)
    if sigma_ann_har and sigma_ann_garch and (sigma_ann_garch > outlier_ratio * sigma_ann_har):
        w_g = w_g * 0.3
        s = w_g + w_h + w_a
        w_g, w_h, w_a = w_g/s, w_h/s, w_a/s

    parts_frac = []
    weights = []
    if sigH_garch_frac is not None:
        parts_frac.append(sigH_garch_frac); weights.append(w_g)
    if sigH_har_frac is not None:
        parts_frac.append(sigH_har_frac);   weights.append(w_h)
    # ATR available always
    parts_frac.append(sigH_atr_frac); weights.append(w_a)

    if not parts_frac or sum(weights) == 0:
        return float(price) * sigH_atr_frac

    # Normalize weights of available components
    s = sum(weights)
    weights = [w/s for w in weights]
    sigH_blend_frac = sum(w*p for w,p in zip(weights, parts_frac))
    sigH_blend_abs = float(price) * sigH_blend_frac
    return sigH_blend_abs

# ---------- New: GARCH sanity validator ----------

def validate_garch_result(returns: np.ndarray, res, sigma_ann: float, horizon_hours: int) -> list[str]:
    issues: list[str] = []
    if np.isnan(returns).any() or np.isinf(returns).any():
        issues.append("NaN/Inf in returns")
    if abs(np.mean(returns)) > 5e-3:
        issues.append("Large mean in returns; did you use raw prices?")

    params = getattr(res, "params", {})
    omega = params.get("omega", None)
    alpha = params.get("alpha[1]", params.get("alpha", None))
    beta  = params.get("beta[1]",  params.get("beta",  None))
    if omega is not None and omega <= 0: issues.append("omega <= 0")
    if alpha is not None and alpha < 0:  issues.append("alpha < 0")
    if beta  is not None and beta  < 0:  issues.append("beta < 0")
    if (alpha is not None) and (beta is not None) and (alpha + beta) >= 1.0:
        issues.append("alpha+beta >= 1 (non-stationary)")

    sigH = _ann_to_horizon_sigma(sigma_ann, horizon_hours)
    if (sigH <= 0) or (sigH > 0.50):
        issues.append(f"Horizon sigma out of range: {sigH:.3f}")
    return issues

# ---------- Dynamic SL/TP and Position Sizing (same as before, but helpers will be used upstream) ----------

def sl_tp_and_size(entry_price, sigma_H, k=1.2, m=2.0, side="long", R=100.0, tick_size=None):
    """
    entry_price: current/entry price
    sigma_H: forecasted volatility for the chosen horizon in fractional terms
    k: stop multiplier
    m: TP multiplier
    side: "long" or "short"
    R: target dollar risk per position
    tick_size: minimum price increment (optional, for rounding)
    """
    sigma_price = entry_price * sigma_H
    sl_distance = k * sigma_price
    tp_distance = m * sigma_price

    if side.lower() == "long":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    # Round to tick size if provided
    if tick_size:
        sl = round(sl / tick_size) * tick_size
        tp = round(tp / tick_size) * tick_size

    Q = R / sl_distance if sl_distance > 0 else 0.0
    return {
        "SL": sl,
        "TP": tp,
        "SL_distance": sl_distance,
        "TP_distance": tp_distance,
        "Q": Q
    }

# ---------- New: trailing stop helper ----------

def compute_trailing_stop(entry: float, direction: str, atr: float, cfg: dict, r_unrealized: float) -> float:
    stops_cfg = (cfg or {}).get('stops', {})
    trail_initial = float(stops_cfg.get('atr_trail_mult_initial', 2.0))
    trail_late    = float(stops_cfg.get('atr_trail_mult_late', 1.5))
    trail_mult = trail_late if r_unrealized >= 2.0 else trail_initial
    if direction == "long":
        return max(entry - atr * trail_mult, 0.0)
    else:
        return entry + atr * trail_mult

