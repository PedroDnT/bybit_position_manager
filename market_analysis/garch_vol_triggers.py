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

# ---------- Realized volatility + HAR-RV (same as before) ----------

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

def har_rv_nowcast(rv_series: pd.Series):
    """
    Simple HAR-RV: RV_t+1 = c + bD*RV_D + bW*RV_W + bM*RV_M
    Returns: nowcast of next-bar RV (and sigma).
    """
    eps = 1e-12
    y = np.log(rv_series.shift(-1) + eps)
    X = pd.DataFrame(index=rv_series.index)
    X["RV_D"] = np.log(rv_series.rolling(window=24, min_periods=24).mean() + eps)
    X["RV_W"] = np.log(rv_series.rolling(window=24*7, min_periods=24*7).mean() + eps)
    X["RV_M"] = np.log(rv_series.rolling(window=24*30, min_periods=24*30).mean() + eps)

    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 200:
        latest_rv = rv_series.iloc[-1]
        return latest_rv, math.sqrt(latest_rv)

    Y = df.iloc[:, 0].values
    Xmat = np.column_stack([np.ones(len(df)), df["RV_D"].values, df["RV_W"].values, df["RV_M"].values])

    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0]

    last = pd.DataFrame({
        "RV_D": [np.log(rv_series.rolling(24).mean().iloc[-1] + eps)],
        "RV_W": [np.log(rv_series.rolling(24*7).mean().iloc[-1] + eps)],
        "RV_M": [np.log(rv_series.rolling(24*30).mean().iloc[-1] + eps)]
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
    rv_next, sigma_bar = har_rv_nowcast(rv)
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
                    cfg: dict) -> float:
    vol_cfg = (cfg or {}).get('vol', {})
    horizon_hours = int(vol_cfg.get('horizon_hours', 4))
    w_g = float(vol_cfg.get('blend_w_garch', 0.30))
    w_h = float(vol_cfg.get('blend_w_har', 0.40))
    w_a = float(vol_cfg.get('blend_w_atr', 0.30))
    outlier_ratio = float(vol_cfg.get('garch_har_outlier_ratio', 2.0))

    sigH_garch = _ann_to_horizon_sigma(sigma_ann_garch, horizon_hours) if sigma_ann_garch else None
    sigH_har   = _ann_to_horizon_sigma(sigma_ann_har,   horizon_hours) if sigma_ann_har else None
    sigH_atr   = float(atr_abs)

    if sigma_ann_har and sigma_ann_garch and (sigma_ann_garch > outlier_ratio * sigma_ann_har):
        w_g = w_g * 0.3
        s = w_g + w_h + w_a
        w_g, w_h, w_a = w_g/s, w_h/s, w_a/s

    parts = []
    weights = []
    if sigH_garch is not None:
        parts.append(sigH_garch); weights.append(w_g)
    if sigH_har is not None:
        parts.append(sigH_har);   weights.append(w_h)
    # ATR available always
    parts.append(sigH_atr); weights.append(w_a)

    if not parts or sum(weights) == 0:
        return sigH_atr
    # Normalize weights of available components
    s = sum(weights)
    weights = [w/s for w in weights]
    sigH_blend = sum(w*p for w,p in zip(weights, parts))
    return sigH_blend

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


# ---------- New: Probability-based TP-before-SL model and dynamic levels ----------

def probability_hit_tp_before_sl(price: float,
                                  tp: float,
                                  sl: float,
                                  sigma_price: float,
                                  side: str,
                                  alpha: float = 1.0) -> float:
    """
    Approximate probability that TP is hit before SL given current state.

    Heuristic model inspired by first-passage probabilities for driftless Brownian motion.
    We use a smooth Bradley-Terry style logistic on distance asymmetry normalized by sigma.

    Args:
        price: current price
        tp: take-profit price
        sl: stop-loss price
        sigma_price: expected one-horizon price sigma (absolute, not %)
        side: 'long' or 'short'
        alpha: softness parameter for the logistic (higher -> softer)

    Returns:
        Probability in [0,1] that TP is hit before SL.
    """
    if sigma_price <= 0:
        return 0.5

    if side.lower() == 'long':
        d_up = max(0.0, tp - price)
        d_dn = max(0.0, price - sl)
    else:
        d_up = max(0.0, price - tp)  # for shorts, TP is below
        d_dn = max(0.0, sl - price)

    # If one side is already violated, return degenerate probability
    if d_up == 0 and d_dn == 0:
        return 0.5
    if d_up == 0:
        return 1.0
    if d_dn == 0:
        return 0.0

    # Normalize asymmetry by sigma and alpha
    z = (d_up - d_dn) / max(1e-12, (sigma_price * alpha))
    # Map to probability
    p_tp = 1.0 / (1.0 + math.exp(z))
    return float(max(0.0, min(1.0, p_tp)))


def dynamic_levels_from_state(current_price: float,
                              entry_price: float,
                              side: str,
                              sigma_H: float,
                              atr: float,
                              base_k: float,
                              base_m: float,
                              cfg: dict) -> dict:
    """
    Compute dynamic SL/TP anchored on CURRENT price, adapting to unrealized R and
    probabilistic win chance.

    Logic:
    - Start from base ATR multiples (base_k, base_m) but anchor distances to current price.
    - If unrealized R >= breakeven threshold, tighten SL to at least breakeven or ATR trail.
    - Optimize TP multiplier to satisfy target probability of hitting TP before SL.

    Returns dict with fields: SL, TP, k_eff, m_eff, p_tp, reasons[list[str]].
    """
    vol_cfg = (cfg or {}).get('vol', {})
    stops_cfg = (cfg or {}).get('stops', {})
    prob_cfg  = (cfg or {}).get('prob', {})

    horizon_hours = int(vol_cfg.get('horizon_hours', 4))
    alpha = float(prob_cfg.get('prob_alpha', 1.0))
    p_target = float(prob_cfg.get('prob_target', 0.55))
    m_min = float(prob_cfg.get('m_min', 2.0))
    m_max = float(prob_cfg.get('m_max', 6.0))
    m_step = float(prob_cfg.get('m_step', 0.25))
    breakeven_R = float(stops_cfg.get('breakeven_after_R', 1.0))

    # One-horizon absolute sigma from current price
    sigma_price = max(1e-12, current_price * float(sigma_H))

    # Base distances anchored to current price
    base_sl_dist = base_k * sigma_price
    base_tp_dist = base_m * sigma_price

    # Approximate unrealized R using base SL distance as R-unit
    if side.lower() == 'long':
        r_unreal = max(0.0, (current_price - entry_price) / max(1e-12, base_sl_dist))
        sl_dyn = current_price - base_sl_dist
        tp_dyn = current_price + base_tp_dist
    else:
        r_unreal = max(0.0, (entry_price - current_price) / max(1e-12, base_sl_dist))
        sl_dyn = current_price + base_sl_dist
        tp_dyn = current_price - base_tp_dist

    reasons: list[str] = []

    # Tighten SL when trade has accrued sufficient unrealized R
    if r_unreal >= breakeven_R:
        trail = compute_trailing_stop(entry=current_price, direction=side, atr=atr, cfg=cfg, r_unrealized=r_unreal)
        if side.lower() == 'long':
            # ensure at least breakeven
            sl_dyn = max(trail, entry_price)
            reasons.append(f"Tightened SL to max(trail, breakeven), r_unreal={r_unreal:.2f}R")
        else:
            sl_dyn = min(trail, entry_price)
            reasons.append(f"Tightened SL to min(trail, breakeven), r_unreal={r_unreal:.2f}R")

    # Optimize TP multiplier to satisfy probability target
    k_eff = base_k
    best_tp = tp_dyn
    best_m  = base_m
    best_p  = probability_hit_tp_before_sl(
        price=current_price, tp=tp_dyn, sl=sl_dyn,
        sigma_price=sigma_price, side=side, alpha=alpha
    )

    m_val = m_min
    while m_val <= m_max:
        if side.lower() == 'long':
            tp_try = current_price + m_val * sigma_price
        else:
            tp_try = current_price - m_val * sigma_price
        p = probability_hit_tp_before_sl(
            price=current_price, tp=tp_try, sl=sl_dyn,
            sigma_price=sigma_price, side=side, alpha=alpha
        )
        # choose the smallest m that meets target; otherwise keep the best expected value
        if p >= p_target:
            best_tp = tp_try
            best_m = m_val
            best_p = p
            reasons.append(f"TP optimized to meet prob target {p_target:.2f} (m={m_val:.2f}, p={p:.2f})")
            break
        # fallback tracking: keep the highest p if none meets target
        if p > best_p:
            best_p = p
            best_tp = tp_try
            best_m = m_val
        m_val = round(m_val + m_step, 6)

    return {
        'SL': float(sl_dyn),
        'TP': float(best_tp),
        'k_eff': float(k_eff),
        'm_eff': float(best_m),
        'p_tp': float(best_p),
        'r_unreal': float(r_unreal),
        'reasons': reasons,
        'sigma_price': float(sigma_price),
        'horizon_hours': horizon_hours,
    }


def backtest_dynamic_levels(df: pd.DataFrame,
                            entry_idx: int,
                            side: str,
                            entry_price: float,
                            sigma_H: float,
                            atr_series: pd.Series,
                            base_k: float,
                            base_m: float,
                            cfg: dict) -> dict:
    """
    Simple historical backtest procedure comparing static (entry-anchored) vs dynamic (state-anchored)
    SL/TP logic to determine which level is hit first after an entry point.

    Methodology:
    - Compute static SL/TP once using entry_price and (base_k, base_m) with given sigma_H.
    - For each subsequent bar i>entry_idx, compute dynamic SL/TP anchored to the previous close
      and previous ATR (to avoid lookahead), then check if the current bar's high/low breaches
      the dynamic or static levels.
    - Record the index of the first hit for each method and whether TP or SL was hit first.

    Args:
        df: OHLCV DataFrame with columns ['open','high','low','close']
        entry_idx: Index (integer position) in df that represents the entry bar
        side: 'long' or 'short'
        entry_price: executed entry price
        sigma_H: horizon volatility fraction (e.g., from blended_sigma_h / price)
        atr_series: ATR series aligned with df (e.g., df['ATR20'])
        base_k: base stop-loss multiple
        base_m: base take-profit multiple
        cfg: configuration dict

    Returns:
        dict with keys:
          - static_first: 'TP' | 'SL' | None
          - static_hit_index: int | None
          - dynamic_first: 'TP' | 'SL' | None
          - dynamic_hit_index: int | None
          - static_levels: {SL, TP}
          - example_dynamic: one example of dynamic levels at entry+1 for inspection
    """
    assert 0 <= entry_idx < len(df) - 1, "entry_idx must allow at least one forward bar"

    # Static levels computed once from entry
    static = sl_tp_and_size(
        entry_price=entry_price,
        sigma_H=sigma_H,
        k=base_k,
        m=base_m,
        side=side,
        R=100.0,
        tick_size=None,
    )
    static_sl = float(static['SL'])
    static_tp = float(static['TP'])

    static_first = None
    static_hit_index = None

    dynamic_first = None
    dynamic_hit_index = None

    # Iterate forward bars
    for i in range(entry_idx + 1, len(df)):
        prev_close = float(df['close'].iloc[i - 1])
        atr_prev = float(atr_series.iloc[i - 1]) if atr_series is not None else 0.0
        # compute dynamic levels anchored to prev close
        dyn = dynamic_levels_from_state(
            current_price=prev_close,
            entry_price=entry_price,
            side=side,
            sigma_H=sigma_H,
            atr=atr_prev,
            base_k=base_k,
            base_m=base_m,
            cfg=cfg,
        )
        dyn_sl = float(dyn['SL'])
        dyn_tp = float(dyn['TP'])

        bar_high = float(df['high'].iloc[i])
        bar_low = float(df['low'].iloc[i])

        # Check static hits
        if static_first is None:
            if side.lower() == 'long':
                if bar_high >= static_tp:
                    static_first = 'TP'
                    static_hit_index = i
                elif bar_low <= static_sl:
                    static_first = 'SL'
                    static_hit_index = i
            else:
                if bar_low <= static_tp:
                    static_first = 'TP'
                    static_hit_index = i
                elif bar_high >= static_sl:
                    static_first = 'SL'
                    static_hit_index = i

        # Check dynamic hits
        if dynamic_first is None:
            if side.lower() == 'long':
                if bar_high >= dyn_tp:
                    dynamic_first = 'TP'
                    dynamic_hit_index = i
                elif bar_low <= dyn_sl:
                    dynamic_first = 'SL'
                    dynamic_hit_index = i
            else:
                if bar_low <= dyn_tp:
                    dynamic_first = 'TP'
                    dynamic_hit_index = i
                elif bar_high >= dyn_sl:
                    dynamic_first = 'SL'
                    dynamic_hit_index = i

        # If both determined, we can stop early
        if static_first is not None and dynamic_first is not None:
            break

    example_dynamic = dynamic_levels_from_state(
        current_price=float(df['close'].iloc[min(entry_idx + 1, len(df) - 1)]),
        entry_price=entry_price,
        side=side,
        sigma_H=sigma_H,
        atr=float(atr_series.iloc[min(entry_idx + 1, len(atr_series) - 1)]) if atr_series is not None else 0.0,
        base_k=base_k,
        base_m=base_m,
        cfg=cfg,
    )

    return {
        'static_first': static_first,
        'static_hit_index': static_hit_index,
        'dynamic_first': dynamic_first,
        'dynamic_hit_index': dynamic_hit_index,
        'static_levels': {'SL': static_sl, 'TP': static_tp},
        'example_dynamic': example_dynamic,
    }

