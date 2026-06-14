from collections import deque
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from src.indicators.cycle import (
    adaptive_atr_ehlers,
    ehler_dominant_cycle,
    get_typical_price,
)
from statsmodels.tsa.stattools import adfuller

_session_range_cache: dict = {}


def add_session_ranges(df: pd.DataFrame, start_time: str = "03:00", end_time: str = "04:30") -> pd.DataFrame:
    # Module-level cache keyed by (df id, start_time, end_time) — avoids recomputation across optimization runs
    cache_key = (id(df), start_time, end_time)
    if cache_key in _session_range_cache:
        df[["RangeHigh", "RangeLow", "SessionRange"]] = _session_range_cache[cache_key]
        return df

    session_data = df.between_time(start_time, end_time)
    daily_ranges = session_data.resample("D").agg(RangeHigh=("High", "max"), RangeLow=("Low", "min"))
    daily_ranges["SessionRange"] = (
        (daily_ranges["RangeHigh"] - daily_ranges["RangeLow"]) / daily_ranges["RangeHigh"]
    ) * 100

    df_days = df.index.floor("D")
    ranges = daily_ranges.reindex(df_days).ffill()
    df[["RangeHigh", "RangeLow", "SessionRange"]] = ranges.to_numpy(copy=False)

    _session_range_cache[cache_key] = df[["RangeHigh", "RangeLow", "SessionRange"]].to_numpy(copy=True)
    return df


# helper
def rolling_percentile_rank_sw(s: pd.Series, window: int) -> pd.Series:
    x = s.to_numpy(dtype=np.float64, copy=False)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    sw = sliding_window_view(x, window_shape=window)
    last = sw[:, -1:]
    pct = (sw <= last).mean(axis=1)
    out[window - 1 :] = pct.astype(np.float64, copy=False)
    return pd.Series(out, index=s.index, name=f"{s.name}_pct_rank_{window}" if s.name else None)


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def resample_ohlc(df, resample_rule):
    new_df = (
        df.resample(resample_rule, label="left", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    return new_df


# ==============================
# Volatility & Range
# ==============================
def calculate_atr(data: pd.DataFrame, atr_length: int = 14, multiplier: float = 1.0, method: str = "rma") -> pd.Series:
    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)

    # True Range calculation
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    if method.lower() == "sma":
        atr = tr.rolling(window=atr_length).mean()
    elif method.lower() == "ema":
        atr = tr.ewm(span=atr_length, adjust=False).mean()
    elif method.lower() == "rma":
        # Wilder's smoothing (RMA)
        atr = tr.ewm(alpha=1 / atr_length, adjust=False).mean()
    else:
        raise ValueError("Method must be 'sma', 'ema', or 'rma'")

    return atr * multiplier


def calculate_bb(
    df: pd.DataFrame, column: str, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger bands."""
    close = df[column]

    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=1)

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_bb_width(
    df: pd.DataFrame,
    column: str = "Close",
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Calculate Bollinger Bands width normalized."""
    close = df[column]

    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=1)

    # Width calculation
    width_absolute = 2 * num_std * std
    return width_absolute / (middle + 1e-9)  # Avoid division by zero


def calculate_keltner_channel(
    df: pd.DataFrame, length: int = 20, mult: float = 2.0, atr_length: int = 10
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channel bands."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Middle line (EMA of close)
    middle = close.ewm(span=length, adjust=False).mean()

    # ATR calculation using Average True Range method
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use RMA (Wilder's smoothing) for ATR
    atr = true_range.ewm(alpha=1 / atr_length, adjust=False).mean()

    # Upper and lower bands
    upper = middle + (atr * mult)
    lower = middle - (atr * mult)

    return upper, middle, lower


def calculate_chandelier_exit(
    df: pd.DataFrame, period: int = 22, atr_period: int = 22, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Calculate Chandelier Exit levels."""
    high = df["High"]
    low = df["Low"]

    # ATR using simple moving average
    atr = calculate_atr(df, atr_length=atr_period, multiplier=multiplier, method="sma")

    # Highest high and lowest low
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()

    # Chandelier levels
    chandelier_long = highest_high - (atr * multiplier)
    chandelier_short = lowest_low + (atr * multiplier)

    return chandelier_long, chandelier_short


def compute_combined_volatility(
    df: pd.DataFrame,
    price_col: str = "Close",
    window_range: int = 20,
    window_geo: int = 252,
    trading_periods: int = 252,
) -> pd.DataFrame:

    # Work with Series directly from DataFrame (preserve index)
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df[price_col]

    # Pre-compute logs as Series (keeps index)
    log_hl = np.log(h / l)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_co = np.log(c / o)

    # Parkinson volatility
    park_roll = log_hl.rolling(window_range, min_periods=window_range).mean()
    df["parkinson_vol"] = np.sqrt(park_roll / (4.0 * np.log(2.0))) * np.sqrt(trading_periods)

    # Rogers-Satchell volatility
    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs_var_roll = rs_var.rolling(window_range, min_periods=window_range).mean()
    df["rs_vol"] = np.sqrt(rs_var_roll) * np.sqrt(trading_periods)

    # Yang-Zhang volatility
    u = np.log(o / c.shift(1))  # Keep as Series with .shift()
    v = log_co

    sigma_co2 = u.rolling(window_range, min_periods=window_range).var(ddof=1)
    sigma_oc2 = v.rolling(window_range, min_periods=window_range).var(ddof=1)
    sigma_rs2 = rs_var.rolling(window_range, min_periods=window_range).mean()

    T = float(window_range)
    k = 0.34 / (1.34 + (T + 1.0) / (T - 1.0))
    yz_var = sigma_co2 + k * sigma_oc2 + (1.0 - k) * sigma_rs2
    df["yang_zhang_vol"] = np.sqrt(yz_var) * np.sqrt(trading_periods)

    # Geometric volatility - convert to numpy only for Numba
    returns = c.pct_change()
    returns_array = returns.values  # Convert to numpy array for Numba

    # Call Numba function with numpy array
    geo_vol_values = _compute_geo_vol_numba(returns_array, window_geo, trading_periods)
    df["geo_vol_252"] = geo_vol_values

    # Calibration and normalization
    vol_cols = ["parkinson_vol", "rs_vol", "yang_zhang_vol"]

    # Create boolean mask for valid rows
    valid_mask = df["geo_vol_252"].notna()
    for col in vol_cols:
        valid_mask &= df[col].notna()

    # Normalize using vectorized operations
    normalized_cols = []
    for col in vol_cols:
        ratio = df.loc[valid_mask, "geo_vol_252"] / df.loc[valid_mask, col]
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        scale_factor = ratio.median() if ratio.notna().sum() > 0 else 1.0

        norm_col = col + "_norm"
        df[norm_col] = df[col] * scale_factor
        normalized_cols.append(norm_col)

    # Average of normalized volatilities
    df["vol_avg_norm"] = df[normalized_cols].mean(axis=1)

    return df


def add_overnight_and_week_gap_features(
    df: pd.DataFrame,
    include_weekly: bool = False,
    include_lagged: bool = False,
    open_time: str | None = None,  # e.g. "09:30"
    close_time: str | None = None,  # e.g. "16:00"
) -> pd.DataFrame:
    """
    Always added:
      - OvernightGap
      - GapPct
      - GapVsPriorDay:   1  if session open > previous day's full High(max)
                        -1  if session open < previous day's full Low(min)
                         NA otherwise (inside range or first day)
      - GapDirection:     1 if GapPct > 0
                        -1 if GapPct < 0
                         0 if GapPct == 0
                         NA if GapPct is NA (e.g., first day / missing prev_close)

    When open_time/close_time are supplied:
      - Open/Close use only the specified intraday window (e.g. RTH).
      - High/Low still use full day (standard for "yesterday's range").
    """
    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    dates = df.index.normalize()

    # === Time-window filter (only for Open/Close) ===
    if open_time is not None or close_time is not None:
        mask = pd.Series(True, index=df.index)
        if open_time is not None:
            mask &= df.index.time >= pd.Timestamp(open_time).time()
        if close_time is not None:
            mask &= df.index.time <= pd.Timestamp(close_time).time()
        filtered_df = df[mask]
    else:
        filtered_df = df

    # Full-day range for accurate "yesterday highs/lows"
    daily_range = df.groupby(dates, observed=True).agg(high_max=("High", "max"), low_min=("Low", "min"))

    # Session-based Open/Close (respects open_time/close_time)
    daily_session = filtered_df.groupby(filtered_df.index.normalize(), observed=True).agg(
        open_first=("Open", "first"), close_last=("Close", "last")
    )

    daily_agg = daily_range.join(daily_session)

    # Core overnight calculations
    daily_agg["prev_close"] = daily_agg["close_last"].shift(1)
    daily_agg["overnight_gap"] = daily_agg["open_first"] - daily_agg["prev_close"]
    daily_agg["gap_pct"] = (daily_agg["overnight_gap"] / daily_agg["prev_close"]) * 100

    # Previous day's full range
    daily_agg["prev_high"] = daily_agg["high_max"].shift(1)
    daily_agg["prev_low"] = daily_agg["low_min"].shift(1)

    # === GapVsPriorDay: 1 if open > prev_high, -1 if open < prev_low, else NA ===
    daily_agg["GapVsPriorDay"] = pd.Series(pd.NA, index=daily_agg.index, dtype="Int64")
    daily_agg.loc[daily_agg["open_first"] > daily_agg["prev_high"], "GapVsPriorDay"] = 1
    daily_agg.loc[daily_agg["open_first"] < daily_agg["prev_low"], "GapVsPriorDay"] = -1

    # === GapDirection: sign of GapPct (1 / -1 / 0 / NA) ===
    daily_agg["GapDirection"] = pd.Series(pd.NA, index=daily_agg.index, dtype="Int64")
    daily_agg.loc[daily_agg["gap_pct"] > 0, "GapDirection"] = 1
    daily_agg.loc[daily_agg["gap_pct"] < 0, "GapDirection"] = -1
    daily_agg.loc[daily_agg["gap_pct"] == 0, "GapDirection"] = 0

    # === Broadcast (fast .map) ===
    df["OvernightGap"] = dates.map(daily_agg["overnight_gap"])
    df["GapPct"] = dates.map(daily_agg["gap_pct"])
    df["GapVsPriorDay"] = dates.map(daily_agg["GapVsPriorDay"])
    df["GapDirection"] = dates.map(daily_agg["GapDirection"])

    # === WEEKLY GAPS (optional) ===
    if include_weekly:
        week_series = df.index.to_period("W").to_timestamp()
        weekly_agg = filtered_df.groupby(filtered_df.index.to_period("W").to_timestamp(), observed=True).agg(
            open_first=("Open", "first"), close_last=("Close", "last")
        )

        weekly_agg["prev_week_close"] = weekly_agg["close_last"].shift(1)
        weekly_agg["week_gap"] = weekly_agg["open_first"] - weekly_agg["prev_week_close"]
        weekly_agg["week_gap_pct"] = weekly_agg["week_gap"] / weekly_agg["prev_week_close"] * 100

        df["CurrentWeekGap"] = week_series.map(weekly_agg["week_gap"])
        df["CurrentWeekGapPct"] = week_series.map(weekly_agg["week_gap_pct"])

    # === LAGGED FEATURES (optional) ===
    if include_lagged:
        daily_agg["prev_day_gap"] = daily_agg["overnight_gap"].shift(1)
        daily_agg["prev_day_gap_pct"] = daily_agg["gap_pct"].shift(1)

        df["PreviousDayGap"] = dates.map(daily_agg["prev_day_gap"])
        df["PreviousDayGapPct"] = dates.map(daily_agg["prev_day_gap_pct"])

    return df


# ==============================
# Momentum & Oscillators
# ==============================
def calculate_rsi(df: pd.DataFrame, length: int | None = None) -> float:
    """Calculate RSI."""
    rma = lambda x, n: x.ewm(alpha=1 / n, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = loss.abs()
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.ffill()
    return rsi


def calculate_adx(df: pd.DataFrame, dilen: int = 14, adxlen: int = 14):
    """Calculate ADX."""
    rma = lambda x, n: x.ewm(alpha=1 / n, adjust=False).mean()
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    pDM = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=df.index)
    mDM = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=df.index)
    tr = pd.Series(
        np.fmax.reduce([(h - l).values, np.abs(h.values - c.shift().values), np.abs(l.values - c.shift().values)]),
        index=df.index,
    )
    atr = rma(tr, dilen)
    pDI = 100 * rma(pDM, dilen) / atr
    mDI = 100 * rma(mDM, dilen) / atr
    dx = (pDI - mDI).abs() / (pDI + mDI).replace(0, 1)
    return 100 * rma(dx, adxlen)


def williams_r(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Calculate Williams %R oscillator.
    pd.Series: Williams %R values (range: -100 to 0)
    """
    highest_high = df["High"].rolling(window=lookback).max()
    lowest_low = df["Low"].rolling(window=lookback).min()

    wr = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low)
    return wr


def calculate_stoch(df: pd.DataFrame, k: int = 18, smooth_k: int = 2, d: int = 6) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D lines).

    Port of Pine Script:
    k = ta.sma(ta.stoch(close, high, low, periodK), smoothK)
    d = ta.sma(k, periodD)

    Risk: Generates frequent false signals in ranging markets. Most effective
    in trending conditions with overbought/oversold confirmation.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data
    k : int
        Lookback period for %K calculation (default: 18)
    smooth_k : int
        Smoothing period for %K (default: 2)
    d : int
        Smoothing period for %D signal line (default: 6)

    Returns
    -------
    Tuple[pd.Series, pd.Series]: (%K line, %D line)

    """
    # Calculate highest high and lowest low over k periods
    lowest_low = df["Low"].rolling(window=k, min_periods=k).min()
    highest_high = df["High"].rolling(window=k, min_periods=k).max()

    # Avoid division by zero
    price_range = (highest_high - lowest_low).replace(0, np.nan)

    # Raw stochastic %K calculation
    raw_k = 100.0 * (df["Close"] - lowest_low) / price_range

    # Smooth %K using MA
    k_line = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()

    # Standard: %D is another SMA of %K
    d_line = k_line.rolling(window=d, min_periods=d).mean()

    return k_line, d_line


def calculate_stoch_rsi(
    df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI oscillator - applies stochastic calculation to RSI values."""
    # Calculate RSI using existing function
    rsi = calculate_rsi(df, rsi_period)

    # Apply stochastic calculation to RSI values
    lowest_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    highest_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()

    # Avoid division by zero
    rsi_range = (highest_rsi - lowest_rsi).replace(0, np.nan)

    # Raw Stochastic RSI
    raw_stoch_rsi = 100.0 * (rsi - lowest_rsi) / rsi_range

    # Smooth %K and calculate %D using SMA
    stoch_rsi_k = raw_stoch_rsi.rolling(window=smooth_k, min_periods=smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d, min_periods=smooth_d).mean()

    return stoch_rsi_k, stoch_rsi_d


def squeeze_momentum(
    df: pd.DataFrame,
    length: int = 20,
    mult: float = 2.0,
    lengthKC: int = 20,
    multKC: float = 1.5,
    useTrueRange: bool = True,
) -> pd.DataFrame:
    """Squeeze Momentum with regular rolling OLS for momentum ('val' = slope)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # === BOLLINGER BANDS ===
    upperBB, _, lowerBB = calculate_bb(df, "Close", length, mult)

    # === KELTNER CHANNELS ===
    upperKC, _, lowerKC = calculate_keltner_channel(df, lengthKC, multKC)

    # === SQUEEZE DETECTION ===
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~(sqzOn | sqzOff)

    # === MOMENTUM CALCULATION (regular OLS) ===
    highest_len = high.rolling(window=lengthKC, min_periods=lengthKC).max()
    lowest_len = low.rolling(window=lengthKC, min_periods=lengthKC).min()
    mid_hl = (highest_len + lowest_len) / 2.0
    sma_c = close.rolling(window=lengthKC, min_periods=lengthKC).mean()

    # Source series used by LazyBear; we keep the same definition
    source = close - ((mid_hl + sma_c) / 2.0)

    # Regular OLS: return slope as momentum
    slope, intercept = _rolling_ols(source, lengthKC)
    val = slope  # momentum proxy: positive slope = upward momentum, negative = downward

    return val, sqzOn
    # return pd.DataFrame({'val': val,'sqzOn': sqzOn.astype(int),'sqzOff': sqzOff.astype(int),'noSqz': noSqz.astype(int),'upperBB': upperBB,'lowerBB': lowerBB,'upperKC': upperKC,'lowerKC': lowerKC}, index=df.index)


def calculate_roc(df: pd.DataFrame, period: int) -> pd.Series:
    roc = df["Close"].pct_change(periods=period) * 100
    return roc


# ==============================
# Trend & Moving Averages
# ==============================
def rma(series: pd.Series, length: int = 10) -> pd.Series:
    return series.ewm(1 / length, adjust=False).mean()


def calculate_ma(
    df: pd.DataFrame, column: str = "Close", period: int = 20, method: str = "SMA", timeframe: str | None = None
) -> pd.Series:
    """
    Calculate Moving Average with optional timeframe resampling.

    method : str
        MA type: 'SMA', 'EMA', or 'TEMA' (default: 'SMA')
    timeframe : str, optional
        Pandas resample string (e.g., '1D', '1H', '4H'). If None, uses original timeframe.
    """
    # Validate method
    valid_methods = ["SMA", "EMA", "TEMA"]
    if method.upper() not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    method = method.upper()

    # Branch 1: Resampling path (with look-ahead bias protection)
    if timeframe is not None and timeframe != "":
        # Resample to target timeframe
        # Assumption: df has OHLCV columns; if not, only column parameter is resampled
        agg_dict = {}
        if "Open" in df.columns:
            agg_dict["Open"] = "first"
        if "High" in df.columns:
            agg_dict["High"] = "max"
        if "Low" in df.columns:
            agg_dict["Low"] = "min"
        if "Close" in df.columns:
            agg_dict["Close"] = "last"
        if "Volume" in df.columns:
            agg_dict["Volume"] = "sum"

        # If the target column isn't in the dict, add it
        if column not in agg_dict:
            agg_dict[column] = "last"

        resampled = df.resample(timeframe).agg(agg_dict).dropna()

        # Calculate MA on resampled data
        if method == "SMA":
            ma_values = resampled[column].rolling(window=period, min_periods=period).mean()

        elif method == "EMA":
            ma_values = resampled[column].ewm(span=period, adjust=False).mean()

        elif method == "TEMA":
            # Triple EMA: reduces lag compared to standard EMA
            close = resampled[column]
            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            ma_values = 3 * (ema1 - ema2) + ema3

        ma_shifted = ma_values.shift(1)
        ma_series = ma_shifted.reindex(df.index, method="ffill")

    # Branch 2: Direct calculation on original timeframe (NO shift)
    else:
        if method == "SMA":
            ma_series = df[column].rolling(window=period, min_periods=period).mean()

        elif method == "EMA":
            ma_series = df[column].ewm(span=period, adjust=False).mean()

        elif method == "TEMA":
            close = df[column]
            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            ma_series = 3 * (ema1 - ema2) + ema3

    return ma_series


def calculate_dew(
    df: pd.DataFrame,
    dpo_period: int = 20,
    wma_period: int = 30,
    envelope_period: int = 10,
    envelope_offset: float = 0.06,
) -> pd.DataFrame:
    """Vectorized DEW component calculation with optimized memory access."""
    prices = df["Close"].values  # NumPy array for faster computation
    n = len(prices)

    # DPO - vectorized with minimum overhead
    sma = pd.Series(prices).rolling(window=dpo_period, min_periods=10).mean().values
    offset = (dpo_period // 2) + 1
    dpo = np.concatenate([np.full(offset, np.nan), prices[:-offset]]) - sma

    # WMA - optimized weight calculation
    weights = np.arange(1, wma_period + 1, dtype=np.float64)
    weight_sum = weights.sum()

    wma = np.full(n, np.nan)
    for i in range(wma_period - 1, n):
        wma[i] = np.dot(weights, prices[i - wma_period + 1 : i + 1]) / weight_sum

    # Envelopes - pure NumPy EMA
    alpha = 2.0 / (envelope_period + 1)
    ema = np.full(n, np.nan)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    upper_envelope = ema * (1 + envelope_offset)
    lower_envelope = ema * (1 - envelope_offset)

    # Crossovers - vectorized boolean operations
    price_above_wma = prices > wma
    price_cross_above_wma = price_above_wma & ~np.concatenate([[False], price_above_wma[:-1]])
    price_cross_below_wma = ~price_above_wma & np.concatenate([[False], price_above_wma[:-1]])

    dpo_above_zero = dpo > 0
    dpo_cross_above_zero = dpo_above_zero  # & ~np.concatenate([[False], dpo_above_zero[:-1]])
    dpo_cross_below_zero = ~dpo_above_zero  # & np.concatenate([[False], dpo_above_zero[:-1]])

    # Breaches
    breach_upper = prices > upper_envelope
    breach_lower = prices < lower_envelope

    # Signals - vectorized logic
    standard_up = price_cross_above_wma & dpo_cross_above_zero
    standard_down = price_cross_below_wma & dpo_cross_below_zero
    exception_up = breach_lower & (price_cross_above_wma | dpo_cross_above_zero)
    exception_down = breach_upper & (price_cross_below_wma | dpo_cross_below_zero)

    signal = np.zeros(n, dtype=np.float32)
    signal[standard_up | exception_up] = 1
    signal[standard_down | exception_down] = -1

    # Forward fill signal
    for i in range(1, n):
        if signal[i] == 0:
            signal[i] = signal[i - 1]

    return signal


# ==============================
# Volume & Price-Volume
# ==============================
def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV) indicator."""
    price_direction = df["Close"].diff().fillna(0)
    volume_multiplier = np.where(price_direction > 0, 1, np.where(price_direction < 0, -1, 0))
    signed_volume = df["Volume"] * volume_multiplier
    obv = signed_volume.cumsum() + 0
    obv.iloc[0] = 0
    return obv


def add_vwap(df: pd.DataFrame, column: str = "typical", freq: str = "D") -> pd.Series:
    x = df
    p = (x["High"] + x["Low"] + x["Close"]) / 3.0 if column.lower() == "typical" else x[column].astype("float64")

    v = x["Volume"].astype("float64")

    valid = p.notna() & v.notna()
    pv = (p * v).where(valid, 0.0)
    vv = v.where(valid, 0.0)

    g = pd.Grouper(freq=freq)
    cum_pv = pv.groupby(g).cumsum()
    cum_vv = vv.groupby(g).cumsum()

    s = (cum_pv / cum_vv.replace(0.0, np.nan)).astype("float64")
    s.name = "VWAP"
    return s


# ==============================
# Pattern Recognition
# ==============================
def is_doji(df: pd.DataFrame, doji_threshold: float = 0.1) -> pd.Series:
    candle_range = df["High"] - df["Low"]
    body_size = (df["Close"] - df["Open"]).abs()
    candle_range = candle_range.replace(0, np.nan)

    body_ratio = body_size / candle_range

    return body_ratio < doji_threshold


def is_upper_wick_rejection(df: pd.DataFrame, tail_threshold: float = 0.5, doji_threshold: float = 0.1) -> pd.Series:
    candle_range = df["High"] - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)

    # Calculate body size
    body_size = abs(df["Close"] - df["Open"])

    # Check for valid candles with significant upper wicks that are not dojis
    valid_candles = candle_range > 0
    rejection = (upper_wick / candle_range) >= tail_threshold
    not_doji = (body_size / candle_range) > doji_threshold

    return valid_candles & rejection & not_doji


def is_lower_wick_rejection(df: pd.DataFrame, tail_threshold: float = 0.5, doji_threshold: float = 0.1) -> pd.Series:
    candle_range = df["High"] - df["Low"]
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]

    # Calculate body size
    body_size = abs(df["Close"] - df["Open"])

    # Check for valid candles with significant lower wicks that are not dojis
    valid_candles = candle_range > 0
    rejection = (lower_wick / candle_range) >= tail_threshold
    not_doji = (body_size / candle_range) > doji_threshold

    return valid_candles & rejection & not_doji


def calculate_fractals(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    use_high_low: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if use_high_low:
        highs = df["High"].astype(float)
        lows = df["Low"].astype(float)
    else:
        highs = pd.Series(np.maximum(df["Close"].values, df["Open"].values), index=df.index, name="HighHL")
        lows = pd.Series(np.minimum(df["Close"].values, df["Open"].values), index=df.index, name="LowHL")

    # Left window: max/min of bars [t-left, t-1] (excludes current bar)
    prev_left_max = highs.shift(1).rolling(window=left, min_periods=left).max()
    prev_left_min = lows.shift(1).rolling(window=left, min_periods=left).min()

    # Right window: max/min of bars [t+1, t+right]
    next_right_max = highs.rolling(window=right, min_periods=right).max().shift(-right)
    next_right_min = lows.rolling(window=right, min_periods=right).min().shift(-right)

    # Strict inequalities to avoid duplicates at flat peaks/troughs
    is_fh = (highs > prev_left_max) & (highs > next_right_max)
    is_fl = (lows < prev_left_min) & (lows < next_right_min)

    # Values at detection point (has lookahead - for charting/analysis)
    fh_value = highs.where(is_fh)
    fl_value = lows.where(is_fl)

    # Values shifted to availability point (lookahead-free - for trading)
    # A fractal detected at bar t is available at bar t+right
    fh_avail = fh_value.shift(right).rename("FractalHighAvail")
    fl_avail = fl_value.shift(right).rename("FractalLowAvail")

    return fh_value, fl_value, fh_avail, fl_avail


@njit(cache=True)
def _zigzag_core(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, threshold: np.ndarray, use_high_low: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core zigzag detection with strict high-low alternation."""
    n = len(high)

    zigzag_values = np.full(n, np.nan)
    pivot_indices = np.full(n, -1, dtype=np.int32)
    pivot_types = np.full(n, 0, dtype=np.int8)
    pivot_available_at = np.full(n, -1, dtype=np.int32)  # Bar when pivot is confirmed

    if n < 2:
        return zigzag_values, pivot_indices, pivot_types, pivot_available_at

    # Select price arrays for extreme detection
    if use_high_low:
        upper_prices = high
        lower_prices = low
    else:
        upper_prices = close
        lower_prices = close

    # Establish initial trend from first 20 bars
    current_extreme = close[0]
    is_uptrend = True

    for i in range(1, min(20, n)):
        if close[i] > current_extreme + threshold[i]:
            is_uptrend = True
            break
        elif close[i] < current_extreme - threshold[i]:
            is_uptrend = False
            break
        current_extreme = close[i]

    # Initialize tracking
    if is_uptrend:
        current_extreme = upper_prices[0]
        current_extreme_idx = 0
        for i in range(1, min(20, n)):
            if upper_prices[i] > current_extreme:
                current_extreme = upper_prices[i]
                current_extreme_idx = i
    else:
        current_extreme = lower_prices[0]
        current_extreme_idx = 0
        for i in range(1, min(20, n)):
            if lower_prices[i] < current_extreme:
                current_extreme = lower_prices[i]
                current_extreme_idx = i

    pivot_count = 0
    start_search_idx = max(1, current_extreme_idx + 1)

    last_pivot_type = np.int8(0)
    last_pivot_idx = np.int32(-1)
    last_pivot_value = np.nan
    prev_pivot_idx = np.int32(-1)

    for i in range(start_search_idx, n):
        if is_uptrend:
            # Track highest high
            if upper_prices[i] > current_extreme:
                current_extreme = upper_prices[i]
                current_extreme_idx = i

            # Check for reversal to downtrend
            if lower_prices[i] < current_extreme - threshold[i]:
                new_pivot_type = np.int8(1)

                if last_pivot_type == new_pivot_type:
                    # Replace weaker high
                    if current_extreme > last_pivot_value:
                        zigzag_values[last_pivot_idx] = np.nan
                        pivot_types[last_pivot_idx] = 0
                        pivot_available_at[last_pivot_idx] = -1

                        zigzag_values[current_extreme_idx] = current_extreme
                        pivot_types[current_extreme_idx] = new_pivot_type
                        pivot_indices[pivot_count - 1] = current_extreme_idx

                        last_pivot_idx = current_extreme_idx
                        last_pivot_value = current_extreme

                    is_uptrend = False
                    current_extreme = lower_prices[i]
                    current_extreme_idx = i

                elif last_pivot_type == 0 or last_pivot_type == -1:
                    # Valid alternation: mark new high pivot
                    zigzag_values[current_extreme_idx] = current_extreme
                    pivot_types[current_extreme_idx] = new_pivot_type
                    pivot_indices[pivot_count] = current_extreme_idx

                    # Previous pivot is now confirmed (available at current bar i)
                    if prev_pivot_idx >= 0:
                        pivot_available_at[prev_pivot_idx] = i

                    prev_pivot_idx = last_pivot_idx
                    last_pivot_type = new_pivot_type
                    last_pivot_idx = current_extreme_idx
                    last_pivot_value = current_extreme
                    pivot_count += 1

                    is_uptrend = False
                    current_extreme = lower_prices[i]
                    current_extreme_idx = i
        else:
            # Track lowest low
            if lower_prices[i] < current_extreme:
                current_extreme = lower_prices[i]
                current_extreme_idx = i

            # Check for reversal to uptrend
            if upper_prices[i] > current_extreme + threshold[i]:
                new_pivot_type = np.int8(-1)

                if last_pivot_type == new_pivot_type:
                    # Replace weaker low
                    if current_extreme < last_pivot_value:
                        zigzag_values[last_pivot_idx] = np.nan
                        pivot_types[last_pivot_idx] = 0
                        pivot_available_at[last_pivot_idx] = -1

                        zigzag_values[current_extreme_idx] = current_extreme
                        pivot_types[current_extreme_idx] = new_pivot_type
                        pivot_indices[pivot_count - 1] = current_extreme_idx

                        last_pivot_idx = current_extreme_idx
                        last_pivot_value = current_extreme

                    is_uptrend = True
                    current_extreme = upper_prices[i]
                    current_extreme_idx = i

                elif last_pivot_type == 0 or last_pivot_type == 1:
                    # Valid alternation: mark new low pivot
                    zigzag_values[current_extreme_idx] = current_extreme
                    pivot_types[current_extreme_idx] = new_pivot_type
                    pivot_indices[pivot_count] = current_extreme_idx

                    # Previous pivot is now confirmed
                    if prev_pivot_idx >= 0:
                        pivot_available_at[prev_pivot_idx] = i

                    prev_pivot_idx = last_pivot_idx
                    last_pivot_type = new_pivot_type
                    last_pivot_idx = current_extreme_idx
                    last_pivot_value = current_extreme
                    pivot_count += 1

                    is_uptrend = True
                    current_extreme = upper_prices[i]
                    current_extreme_idx = i

    return zigzag_values, pivot_indices, pivot_types, pivot_available_at


def calculate_zigzag(
    df: pd.DataFrame,
    atr_divisor: float = 5.0,
    use_high_low: bool = False,
) -> pd.DataFrame:
    """Calculate adaptive zigzag with lookahead-free availability tracking."""
    typical = get_typical_price(df)
    dominant_cycle = ehler_dominant_cycle(typical)
    atr_adaptive = adaptive_atr_ehlers(df, adaptive_period=2 * dominant_cycle)

    threshold_series = (atr_adaptive / atr_divisor).ffill().fillna(np.inf)
    threshold_array = threshold_series.values

    zigzag_values, pivot_indices, pivot_types, pivot_available_at = _zigzag_core(
        df["High"].values, df["Low"].values, df["Close"].values, threshold_array, use_high_low=use_high_low
    )

    result_df = df.copy()
    result_df["zigzag"] = zigzag_values
    result_df["pivot_high"] = pivot_types == 1
    result_df["pivot_low"] = pivot_types == -1
    result_df["pivot_available_at"] = pivot_available_at  # Bar index when confirmed
    result_df["zigzag_threshold"] = threshold_array

    return result_df


def calculate_confirmed_pivots(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
) -> pd.DataFrame:
    """Combine fractals + zigzag into lookahead-safe confirmed pivots."""
    fh_value, fl_value, _, _ = calculate_fractals(
        df, left=fractal_lookback, right=fractal_lookback, use_high_low=use_high_low
    )

    zz_result = calculate_zigzag(df, atr_divisor=atr_divisor, use_high_low=use_high_low)

    is_confirmed_high = ~fh_value.isna() & zz_result["pivot_high"]
    is_confirmed_low = ~fl_value.isna() & zz_result["pivot_low"]
    n = len(df)

    confirmed_high = fh_value.where(is_confirmed_high)
    confirmed_low = fl_value.where(is_confirmed_low)

    high_idx = np.flatnonzero(is_confirmed_high)
    low_idx = np.flatnonzero(is_confirmed_low)

    # Fractal availability: pivot bar + right lookback
    fh_avail = np.full(n, np.nan, dtype=np.float64)
    fl_avail = np.full(n, np.nan, dtype=np.float64)
    if high_idx.size > 0:
        fh_avail[high_idx] = high_idx + fractal_lookback
    if low_idx.size > 0:
        fl_avail[low_idx] = low_idx + fractal_lookback

    # Zigzag availability
    zz_avail = zz_result["pivot_available_at"].replace(-1, np.nan).to_numpy(dtype=np.float64)

    # Confirmed availability: later of fractal and zigzag availability
    confirmed_high_avail = np.where(is_confirmed_high, np.fmax(fh_avail, zz_avail), np.nan)
    confirmed_low_avail = np.where(is_confirmed_low, np.fmax(fl_avail, zz_avail), np.nan)

    result = pd.DataFrame(index=df.index)
    result["confirmed_high"] = confirmed_high
    result["confirmed_low"] = confirmed_low
    result["confirmed_high_avail"] = confirmed_high_avail
    result["confirmed_low_avail"] = confirmed_low_avail
    result["atr_adaptive"] = zz_result["zigzag_threshold"] * atr_divisor
    return result


def wyckoff_spring_textbook_signal(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
    confirmed_pivots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Textbook Wyckoff spring (lookahead-safe, one signal per structure).

    Structure:
      - H[-3] < H[-2]
      - H[-1] < H[-2]
      - Exactly one confirmed low between H[-2] and H[-1]
      - That low lies in H[-3] +/- ATR/atr_divisor at H[-1] confirmation bar

    Signal:
      - Emitted exactly once at H[-1] confirmation bar
      - limit_buy_price equals spring low price
    """
    cp = (
        confirmed_pivots
        if confirmed_pivots is not None
        else calculate_confirmed_pivots(
            df,
            fractal_lookback=fractal_lookback,
            atr_divisor=atr_divisor,
            use_high_low=use_high_low,
        )
    )

    n = len(df)
    high_mask = ~cp["confirmed_high"].isna() & ~cp["confirmed_high_avail"].isna()
    low_mask = ~cp["confirmed_low"].isna() & ~cp["confirmed_low_avail"].isna()

    high_positions = np.flatnonzero(high_mask)
    low_positions = np.flatnonzero(low_mask)

    # Build event lists with numpy array indexing (avoids slow .iloc)
    high_avail_arr = cp["confirmed_high_avail"].to_numpy()
    high_val_arr = cp["confirmed_high"].to_numpy()
    low_avail_arr = cp["confirmed_low_avail"].to_numpy()
    low_val_arr = cp["confirmed_low"].to_numpy()

    h_avails = high_avail_arr[high_positions].astype(np.intp)
    h_vals = high_val_arr[high_positions]
    order_h = np.argsort(h_avails, kind="mergesort")
    high_events = [(h_avails[j], int(high_positions[j]), h_vals[j]) for j in order_h]

    l_avails = low_avail_arr[low_positions].astype(np.intp)
    l_vals = low_val_arr[low_positions]
    order_l = np.argsort(l_avails, kind="mergesort")
    low_events = [(l_avails[j], int(low_positions[j]), l_vals[j]) for j in order_l]

    atr = cp["atr_adaptive"].to_numpy(dtype=np.float64)

    spring_signal = np.zeros(n, dtype=np.int8)
    limit_buy_price = np.full(n, np.nan, dtype=np.float64)
    spring_low_price = np.full(n, np.nan, dtype=np.float64)
    spring_low_pivot_idx = np.full(n, np.nan, dtype=np.float64)

    h_m3_price = np.full(n, np.nan, dtype=np.float64)
    h_m2_price = np.full(n, np.nan, dtype=np.float64)
    h_m1_price = np.full(n, np.nan, dtype=np.float64)
    h_m3_idx = np.full(n, np.nan, dtype=np.float64)
    h_m2_idx = np.full(n, np.nan, dtype=np.float64)
    h_m1_idx = np.full(n, np.nan, dtype=np.float64)

    support_lower = np.full(n, np.nan, dtype=np.float64)
    support_upper = np.full(n, np.nan, dtype=np.float64)
    spring_valid = np.zeros(n, dtype=np.int8)

    recent_highs: deque = deque(maxlen=3)
    available_lows: list[tuple[int, float]] = []
    emitted_structures = set()
    l_ptr = 0

    # Group high events by availability bar for event-driven iteration
    high_groups: list[tuple[int, list[tuple[int, float]]]] = []
    for avail_bar, pivot_idx, value in high_events:
        if not high_groups or high_groups[-1][0] != avail_bar:
            high_groups.append((avail_bar, []))
        high_groups[-1][1].append((pivot_idx, value))

    # Event-driven: iterate only over bars where new highs arrive (O(num_events) vs O(n))
    for avail_bar, group in high_groups:
        while l_ptr < len(low_events) and low_events[l_ptr][0] <= avail_bar:
            available_lows.append((low_events[l_ptr][1], low_events[l_ptr][2]))
            l_ptr += 1

        for pivot_idx, value in group:
            recent_highs.append((pivot_idx, value))

        if len(recent_highs) < 3:
            continue

        i = avail_bar
        h3_idx, h3 = recent_highs[0]
        h2_idx, h2 = recent_highs[1]
        h1_idx, h1 = recent_highs[2]
        atr_i = atr[i]

        h_m3_idx[i] = h3_idx
        h_m2_idx[i] = h2_idx
        h_m1_idx[i] = h1_idx
        h_m3_price[i] = h3
        h_m2_price[i] = h2
        h_m1_price[i] = h1

        if np.isnan(atr_i) or atr_i <= 0:
            continue

        lows_between = [(low_idx, low_val) for low_idx, low_val in available_lows if h2_idx < low_idx < h1_idx]

        if len(lows_between) != 1:
            continue

        low_idx, low_val = lows_between[0]
        spring_low_pivot_idx[i] = low_idx
        spring_low_price[i] = low_val

        zone_half_width = atr_i / atr_divisor
        support_lower[i] = h3 - zone_half_width
        support_upper[i] = h3 + zone_half_width

        in_support_zone = support_lower[i] <= low_val <= support_upper[i]
        is_valid = (h3 < h2) and (h1 < h2) and in_support_zone
        if not is_valid:
            continue

        spring_valid[i] = 1
        structure_key = (h3_idx, h2_idx, h1_idx, low_idx)
        if structure_key in emitted_structures:
            continue

        emitted_structures.add(structure_key)
        spring_signal[i] = 1
        limit_buy_price[i] = low_val

    out = df.copy()
    out["spring_signal"] = spring_signal
    out["limit_buy_price"] = limit_buy_price
    out["spring_low_price"] = spring_low_price
    out["spring_low_pivot_idx"] = spring_low_pivot_idx

    out["h_m3_price"] = h_m3_price
    out["h_m2_price"] = h_m2_price
    out["h_m1_price"] = h_m1_price
    out["h_m3_idx"] = h_m3_idx
    out["h_m2_idx"] = h_m2_idx
    out["h_m1_idx"] = h_m1_idx

    out["support_lower"] = support_lower
    out["support_upper"] = support_upper
    out["spring_valid"] = spring_valid
    return out


def wyckoff_upthrust_textbook_signal(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
    confirmed_pivots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Textbook Wyckoff upthrust (lookahead-safe, one signal per structure).

    Structure (bearish mirror of textbook spring):
      - L[-3] > L[-2]
      - L[-1] > L[-2]
      - Exactly one confirmed high between L[-2] and L[-1]
      - That high lies in L[-3] +/- ATR/atr_divisor at L[-1] confirmation bar

    Signal:
      - Emitted exactly once at L[-1] confirmation bar
      - limit_sell_price equals upthrust high price
    """
    cp = (
        confirmed_pivots
        if confirmed_pivots is not None
        else calculate_confirmed_pivots(
            df,
            fractal_lookback=fractal_lookback,
            atr_divisor=atr_divisor,
            use_high_low=use_high_low,
        )
    )

    n = len(df)
    high_mask = ~cp["confirmed_high"].isna() & ~cp["confirmed_high_avail"].isna()
    low_mask = ~cp["confirmed_low"].isna() & ~cp["confirmed_low_avail"].isna()

    high_positions = np.flatnonzero(high_mask)
    low_positions = np.flatnonzero(low_mask)

    # Build event lists with numpy array indexing (avoids slow .iloc)
    high_avail_arr = cp["confirmed_high_avail"].to_numpy()
    high_val_arr = cp["confirmed_high"].to_numpy()
    low_avail_arr = cp["confirmed_low_avail"].to_numpy()
    low_val_arr = cp["confirmed_low"].to_numpy()

    h_avails = high_avail_arr[high_positions].astype(np.intp)
    h_vals = high_val_arr[high_positions]
    order_h = np.argsort(h_avails, kind="mergesort")
    high_events = [(h_avails[j], int(high_positions[j]), h_vals[j]) for j in order_h]

    l_avails = low_avail_arr[low_positions].astype(np.intp)
    l_vals = low_val_arr[low_positions]
    order_l = np.argsort(l_avails, kind="mergesort")
    low_events = [(l_avails[j], int(low_positions[j]), l_vals[j]) for j in order_l]

    atr = cp["atr_adaptive"].to_numpy(dtype=np.float64)

    upthrust_signal = np.zeros(n, dtype=np.int8)
    limit_sell_price = np.full(n, np.nan, dtype=np.float64)
    upthrust_high_price = np.full(n, np.nan, dtype=np.float64)
    upthrust_high_pivot_idx = np.full(n, np.nan, dtype=np.float64)

    l_m3_price = np.full(n, np.nan, dtype=np.float64)
    l_m2_price = np.full(n, np.nan, dtype=np.float64)
    l_m1_price = np.full(n, np.nan, dtype=np.float64)
    l_m3_idx = np.full(n, np.nan, dtype=np.float64)
    l_m2_idx = np.full(n, np.nan, dtype=np.float64)
    l_m1_idx = np.full(n, np.nan, dtype=np.float64)

    resistance_lower = np.full(n, np.nan, dtype=np.float64)
    resistance_upper = np.full(n, np.nan, dtype=np.float64)
    upthrust_valid = np.zeros(n, dtype=np.int8)

    recent_lows: deque = deque(maxlen=3)
    available_highs: list[tuple[int, float]] = []
    emitted_structures = set()
    h_ptr = 0

    # Group low events by availability bar for event-driven iteration
    low_groups: list[tuple[int, list[tuple[int, float]]]] = []
    for avail_bar, pivot_idx, value in low_events:
        if not low_groups or low_groups[-1][0] != avail_bar:
            low_groups.append((avail_bar, []))
        low_groups[-1][1].append((pivot_idx, value))

    # Event-driven: iterate only over bars where new lows arrive (O(num_events) vs O(n))
    for avail_bar, group in low_groups:
        while h_ptr < len(high_events) and high_events[h_ptr][0] <= avail_bar:
            available_highs.append((high_events[h_ptr][1], high_events[h_ptr][2]))
            h_ptr += 1

        for pivot_idx, value in group:
            recent_lows.append((pivot_idx, value))

        if len(recent_lows) < 3:
            continue

        i = avail_bar
        l3_idx, l3 = recent_lows[0]
        l2_idx, l2 = recent_lows[1]
        l1_idx, l1 = recent_lows[2]
        atr_i = atr[i]

        l_m3_idx[i] = l3_idx
        l_m2_idx[i] = l2_idx
        l_m1_idx[i] = l1_idx
        l_m3_price[i] = l3
        l_m2_price[i] = l2
        l_m1_price[i] = l1

        if np.isnan(atr_i) or atr_i <= 0:
            continue

        highs_between = [(high_idx, high_val) for high_idx, high_val in available_highs if l2_idx < high_idx < l1_idx]

        if len(highs_between) != 1:
            continue

        high_idx, high_val = highs_between[0]
        upthrust_high_pivot_idx[i] = high_idx
        upthrust_high_price[i] = high_val

        zone_half_width = atr_i / atr_divisor
        resistance_lower[i] = l3 - zone_half_width
        resistance_upper[i] = l3 + zone_half_width

        in_resistance_zone = resistance_lower[i] <= high_val <= resistance_upper[i]
        is_valid = (l3 > l2) and (l1 > l2) and in_resistance_zone
        if not is_valid:
            continue

        upthrust_valid[i] = 1
        structure_key = (l3_idx, l2_idx, l1_idx, high_idx)
        if structure_key in emitted_structures:
            continue

        emitted_structures.add(structure_key)
        upthrust_signal[i] = 1
        limit_sell_price[i] = high_val

    out = df.copy()
    out["upthrust_signal"] = upthrust_signal
    out["limit_sell_price"] = limit_sell_price
    out["upthrust_high_price"] = upthrust_high_price
    out["upthrust_high_pivot_idx"] = upthrust_high_pivot_idx

    out["l_m3_price"] = l_m3_price
    out["l_m2_price"] = l_m2_price
    out["l_m1_price"] = l_m1_price
    out["l_m3_idx"] = l_m3_idx
    out["l_m2_idx"] = l_m2_idx
    out["l_m1_idx"] = l_m1_idx

    out["resistance_lower"] = resistance_lower
    out["resistance_upper"] = resistance_upper
    out["upthrust_valid"] = upthrust_valid
    return out


def wyckoff_textbook_signal(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
    confirmed_pivots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Unified textbook Wyckoff signal:
      - spring -> Signal = 1
      - upthrust -> Signal = -1
      - no pattern -> Signal = 0

    Returns only input df columns plus a single `Signal` column.
    """
    cp = (
        confirmed_pivots
        if confirmed_pivots is not None
        else calculate_confirmed_pivots(
            df,
            fractal_lookback=fractal_lookback,
            atr_divisor=atr_divisor,
            use_high_low=use_high_low,
        )
    )

    spring_df = wyckoff_spring_textbook_signal(
        df,
        fractal_lookback=fractal_lookback,
        atr_divisor=atr_divisor,
        use_high_low=use_high_low,
        confirmed_pivots=cp,
    )
    upthrust_df = wyckoff_upthrust_textbook_signal(
        df,
        fractal_lookback=fractal_lookback,
        atr_divisor=atr_divisor,
        use_high_low=use_high_low,
        confirmed_pivots=cp,
    )

    signal = np.zeros(len(df), dtype=np.int8)
    spring_mask = spring_df["spring_signal"].to_numpy(dtype=np.int8) == 1
    upthrust_mask = upthrust_df["upthrust_signal"].to_numpy(dtype=np.int8) == 1

    signal[spring_mask] = 1
    signal[upthrust_mask] = -1

    # If both occur on the same bar, neutralize ambiguity.
    both_mask = spring_mask & upthrust_mask
    signal[both_mask] = 0

    out = df.copy()
    out["Signal"] = signal
    return out


# ==============================
# Event & News Processing
# ==============================
def process_news(
    price_df: pd.DataFrame,
    news_raw: pd.DataFrame,
    n_bars: int = 1,
    timezone: str = "US/Eastern",
    currencies: Sequence[str] = ("USD",),
    impacts: Sequence[str] = ("red", "High Impact Expected"),
) -> pd.Series:
    """
    Vectorized news-to-bar mapping with currency/impact filters and neighborhood expansion.

    Marks price bars near news events (bank holidays mark full day, regular events mark
    exact or next-bar match). Neighborhood expansion via binary dilation with kernel size 2*n_bars+1.
    """
    # Align price index to timezone (no DataFrame copy needed, only need the index)
    idx = pd.DatetimeIndex(price_df.index)
    idx = idx.tz_localize(timezone) if idx.tz is None else idx.tz_convert(timezone)
    sorted_idx = idx.sort_values()
    news_flags = pd.Series(np.zeros(len(sorted_idx), dtype=np.int8), index=sorted_idx, name="News")

    if news_raw is None or news_raw.empty:
        return news_flags

    # Resolve incoming schema to canonical names
    _norm = lambda name: "".join(ch for ch in str(name).lower() if ch.isalnum())
    col_map = {_norm(c): c for c in news_raw.columns}

    date_col = next((col_map[k] for k in ["datetime", "date", "time", "timestamp"] if k in col_map), None)
    currency_col = next((col_map[k] for k in ["currency", "ccy"] if k in col_map), None)
    impact_col = col_map.get("impact")
    event_col = col_map.get("event")

    missing = [
        name
        for name, col in [
            ("date/time", date_col),
            ("currency", currency_col),
            ("impact", impact_col),
            ("event", event_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"news_raw missing required columns: {', '.join(missing)}")

    # Select, rename, parse, and filter in one pass
    news = news_raw[[date_col, currency_col, impact_col, event_col]].copy()
    news.columns = ["Date", "currency", "impact", "event"]
    news["Date"] = pd.to_datetime(news["Date"], utc=True, errors="coerce")
    news = news.dropna(subset=["Date"])
    if news.empty:
        return news_flags

    news["Date"] = news["Date"].dt.tz_convert(timezone)
    news["currency"] = news["currency"].astype(str).str.strip().str.upper()
    news["impact"] = news["impact"].astype(str).str.strip().str.lower()

    currencies_norm = {str(c).strip().upper() for c in currencies}
    impacts_norm = {str(i).strip().lower() for i in impacts}
    news = news[news["currency"].isin(currencies_norm) & news["impact"].isin(impacts_norm)]
    if news.empty:
        return news_flags

    # Separate bank holidays (all-day) from regular events
    is_bh = news["event"].astype(str).str.contains("Bank Holiday", case=False, na=False)
    bh_dates = news.loc[is_bh, "Date"].dt.normalize()
    reg_times = news.loc[~is_bh, "Date"]

    # Bank holidays: mark entire day via interval marking (difference-array + cumsum)
    if not bh_dates.empty:
        bh_start = pd.DatetimeIndex(bh_dates.unique())
        bh_end = bh_start + pd.Timedelta(days=1)
        s_idx = news_flags.index.searchsorted(bh_start)
        e_idx = news_flags.index.searchsorted(bh_end, side="right")
        marks = np.zeros(len(news_flags) + 1, dtype=np.int32)
        np.add.at(marks, s_idx, 1)
        np.add.at(marks, e_idx, -1)
        bh_mask = np.cumsum(marks[:-1]) > 0
        if bh_mask.any():
            news_flags.iloc[np.flatnonzero(bh_mask)] = 1

    # Regular events: exact match + forward-align non-exact to next bar
    if len(reg_times) > 0:
        reg_unique = pd.DatetimeIndex(reg_times.unique()).sort_values()
        exact = news_flags.index.intersection(reg_unique)
        news_flags.loc[exact] = 1

        non_exact = reg_unique.difference(exact)
        if len(non_exact) > 0:
            forward_idx = np.unique(news_flags.index.searchsorted(non_exact, side="left"))
            forward_idx = forward_idx[forward_idx < len(news_flags)]
            if len(forward_idx) > 0:
                news_flags.iloc[forward_idx] = 1

    # Neighborhood expansion via convolution (binary dilation)
    if n_bars > 0:
        kernel = np.ones(2 * n_bars + 1, dtype=np.int8)
        news_flags[:] = (np.convolve(news_flags.to_numpy(), kernel, mode="same") > 0).astype(np.int8)

    return news_flags


def day_change(df: pd.DataFrame, end_of_day_time: str = "15:45:00") -> pd.Series:
    """
    Identify day changes in intraday data using DatetimeIndex.
    Flags 1 when the date 2 bars ahead differs from the current bar's date.
    """
    dates = pd.Series(df.index.date, index=df.index)
    lookahead = (dates != dates.shift(-2)).fillna(True).astype(int)

    eod_time = pd.to_datetime(end_of_day_time).time()
    eod_flag = pd.Series((df.index.time == eod_time).astype(int), index=df.index)

    day_change = (lookahead | eod_flag).astype(int)
    return day_change.rename("DayChange")


def calculate_confirmed_market_calls(
    df: pd.DataFrame,
    vvc_lookback: int = 10,
    bsr_period: int = 20,
    rt_period: int = 20,
    momentum_short: int = 10,
    momentum_long: int = 20,
) -> pd.Series:
    """
    Calculate VectorVest Confirmed Market Calls with BSR proxy

    Confirmed Up:
        - Two consecutive 5-day periods show uptrend
        - Daily momentum positive
        - BSR (Buy/Sell Ratio) > 1.00

    Confirmed Down:
        - Two consecutive 5-day periods show downtrend
        - Daily momentum negative
        - BSR < 1.00
    """
    prices = df["Close"]

    # VVC proxy: Use closing price as market composite indicator
    vvc_proxy = prices

    # BSR proxy calculation using technical strength indicators
    # RT (Relative Timing): Price vs moving average ratio
    rt_ma = prices.rolling(window=rt_period, min_periods=rt_period // 2).mean()
    rt_proxy = prices / rt_ma
    rt_proxy = rt_proxy.fillna(1.0)  # Neutral if insufficient data

    # Momentum: Short-term vs long-term trend strength
    momentum_short_ma = prices.rolling(window=momentum_short, min_periods=momentum_short // 2).mean()
    momentum_long_ma = prices.rolling(window=momentum_long, min_periods=momentum_long // 2).mean()
    momentum = momentum_short_ma / momentum_long_ma
    momentum = momentum.fillna(1.0)

    # Buy/Sell vote tallying
    buy_votes = (rt_proxy >= 1.00) & (momentum >= 1.00)
    sell_votes = (rt_proxy < 1.00) | (momentum < 1.00)

    buy_count = buy_votes.rolling(window=bsr_period, min_periods=bsr_period // 2).sum()
    sell_count = sell_votes.rolling(window=bsr_period, min_periods=bsr_period // 2).sum()

    # BSR (Buy/Sell Ratio): Add 1 to avoid division by zero
    bsr_proxy = (buy_count + 1) / (sell_count + 1)

    # Confirmed Call conditions
    price_5d_ago = vvc_proxy.shift(5)
    price_10d_ago = vvc_proxy.shift(10)
    price_1d_ago = vvc_proxy.shift(1)

    # Two consecutive 5-day periods trending
    two_periods_up = (vvc_proxy > price_5d_ago) & (price_5d_ago > price_10d_ago)
    two_periods_down = (vvc_proxy < price_5d_ago) & (price_5d_ago < price_10d_ago)

    # Confirmed signals require all conditions met
    confirmed_up = two_periods_up & (vvc_proxy > price_1d_ago) & (bsr_proxy > 1.00)
    confirmed_down = two_periods_down & (vvc_proxy < price_1d_ago) & (bsr_proxy < 1.00)

    # Build signal series
    signal = pd.Series(0, index=vvc_proxy.index, dtype=np.int8)
    signal[confirmed_up] = 1
    signal[confirmed_down] = -1

    # Forward fill - maintain confirmation state
    signal = signal.replace(0, np.nan).ffill().fillna(0).astype(np.int8)
    return signal


# ==============================
# Helpers (Internal/Private)
# ==============================
def _rolling_ols(y: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Regular rolling OLS with x = 0..n-1 per window."""
    yv = y.to_numpy(dtype=np.float64)
    N = yv.size

    # Rolling sums for Sy
    Sy = pd.Series(yv).rolling(window=n, min_periods=n).sum().to_numpy()

    # Sxy via single convolution with x = [0, 1, ..., n-1]
    x = np.arange(n, dtype=np.float64)
    Sxy_valid = np.convolve(yv, x, mode="valid")  # length N - n + 1
    Sxy = np.full(N, np.nan, dtype=np.float64)
    Sxy[n - 1 :] = Sxy_valid  # align to window endpoints

    # Precompute constants
    Sx = n * (n - 1) / 2.0
    Sxx = n * (n - 1) * (2 * n - 1) / 6.0
    denom = n * Sxx - Sx * Sx

    # OLS coefficients
    b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n

    slope = pd.Series(b, index=y.index, name="slope")
    intercept = pd.Series(a, index=y.index, name="intercept")
    return slope, intercept


@njit(cache=True)
def _zigzag_core(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, threshold: np.ndarray, use_high_low: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core zigzag detection with strict high-low alternation."""
    n = len(high)

    zigzag_values = np.full(n, np.nan)
    pivot_indices = np.full(n, -1, dtype=np.int32)
    pivot_types = np.full(n, 0, dtype=np.int8)
    pivot_available_at = np.full(n, -1, dtype=np.int32)  # Bar when pivot is confirmed

    if n < 2:
        return zigzag_values, pivot_indices, pivot_types, pivot_available_at

    # Select price arrays for extreme detection
    if use_high_low:
        upper_prices = high
        lower_prices = low
    else:
        upper_prices = close
        lower_prices = close

    # Establish initial trend from first 20 bars
    current_extreme = close[0]
    is_uptrend = True

    for i in range(1, min(20, n)):
        if close[i] > current_extreme + threshold[i]:
            is_uptrend = True
            break
        elif close[i] < current_extreme - threshold[i]:
            is_uptrend = False
            break
        current_extreme = close[i]

    # Initialize tracking
    if is_uptrend:
        current_extreme = upper_prices[0]
        current_extreme_idx = 0
        for i in range(1, min(20, n)):
            if upper_prices[i] > current_extreme:
                current_extreme = upper_prices[i]
                current_extreme_idx = i
    else:
        current_extreme = lower_prices[0]
        current_extreme_idx = 0
        for i in range(1, min(20, n)):
            if lower_prices[i] < current_extreme:
                current_extreme = lower_prices[i]
                current_extreme_idx = i

    pivot_count = 0
    start_search_idx = max(1, current_extreme_idx + 1)

    last_pivot_type = np.int8(0)
    last_pivot_idx = np.int32(-1)
    last_pivot_value = np.nan
    prev_pivot_idx = np.int32(-1)

    for i in range(start_search_idx, n):
        if is_uptrend:
            # Track highest high
            if upper_prices[i] > current_extreme:
                current_extreme = upper_prices[i]
                current_extreme_idx = i

            # Check for reversal to downtrend
            if lower_prices[i] < current_extreme - threshold[i]:
                new_pivot_type = np.int8(1)

                if last_pivot_type == new_pivot_type:
                    # Replace weaker high
                    if current_extreme > last_pivot_value:
                        zigzag_values[last_pivot_idx] = np.nan
                        pivot_types[last_pivot_idx] = 0
                        pivot_available_at[last_pivot_idx] = -1

                        zigzag_values[current_extreme_idx] = current_extreme
                        pivot_types[current_extreme_idx] = new_pivot_type
                        pivot_indices[pivot_count - 1] = current_extreme_idx

                        last_pivot_idx = current_extreme_idx
                        last_pivot_value = current_extreme

                    is_uptrend = False
                    current_extreme = lower_prices[i]
                    current_extreme_idx = i

                elif last_pivot_type == 0 or last_pivot_type == -1:
                    # Valid alternation: mark new high pivot
                    zigzag_values[current_extreme_idx] = current_extreme
                    pivot_types[current_extreme_idx] = new_pivot_type
                    pivot_indices[pivot_count] = current_extreme_idx

                    # Previous pivot is now confirmed (available at current bar i)
                    if prev_pivot_idx >= 0:
                        pivot_available_at[prev_pivot_idx] = i

                    prev_pivot_idx = last_pivot_idx
                    last_pivot_type = new_pivot_type
                    last_pivot_idx = current_extreme_idx
                    last_pivot_value = current_extreme
                    pivot_count += 1

                    is_uptrend = False
                    current_extreme = lower_prices[i]
                    current_extreme_idx = i
        else:
            # Track lowest low
            if lower_prices[i] < current_extreme:
                current_extreme = lower_prices[i]
                current_extreme_idx = i

            # Check for reversal to uptrend
            if upper_prices[i] > current_extreme + threshold[i]:
                new_pivot_type = np.int8(-1)

                if last_pivot_type == new_pivot_type:
                    # Replace weaker low
                    if current_extreme < last_pivot_value:
                        zigzag_values[last_pivot_idx] = np.nan
                        pivot_types[last_pivot_idx] = 0
                        pivot_available_at[last_pivot_idx] = -1

                        zigzag_values[current_extreme_idx] = current_extreme
                        pivot_types[current_extreme_idx] = new_pivot_type
                        pivot_indices[pivot_count - 1] = current_extreme_idx

                        last_pivot_idx = current_extreme_idx
                        last_pivot_value = current_extreme

                    is_uptrend = True
                    current_extreme = upper_prices[i]
                    current_extreme_idx = i

                elif last_pivot_type == 0 or last_pivot_type == 1:
                    # Valid alternation: mark new low pivot
                    zigzag_values[current_extreme_idx] = current_extreme
                    pivot_types[current_extreme_idx] = new_pivot_type
                    pivot_indices[pivot_count] = current_extreme_idx

                    # Previous pivot is now confirmed
                    if prev_pivot_idx >= 0:
                        pivot_available_at[prev_pivot_idx] = i

                    prev_pivot_idx = last_pivot_idx
                    last_pivot_type = new_pivot_type
                    last_pivot_idx = current_extreme_idx
                    last_pivot_value = current_extreme
                    pivot_count += 1

                    is_uptrend = True
                    current_extreme = upper_prices[i]
                    current_extreme_idx = i

    return zigzag_values, pivot_indices, pivot_types, pivot_available_at


@njit
def _compute_geo_vol_numba(returns: np.ndarray, window: int, trading_periods: int) -> np.ndarray:
    """Numba-optimized geometric volatility calculation - pure numpy."""
    n = len(returns)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_ret = returns[i - window + 1 : i + 1]

        # Check for NaN
        has_nan = False
        for val in window_ret:
            if np.isnan(val):
                has_nan = True
                break

        if has_nan:
            continue

        # Geometric mean daily return
        prod = 1.0
        for r in window_ret:
            prod *= 1.0 + r
        gmean_day_return = prod ** (1.0 / window) - 1.0

        # Variance (manual calculation)
        mean_ret = 0.0
        for r in window_ret:
            mean_ret += r
        mean_ret /= window

        var_sum = 0.0
        for r in window_ret:
            var_sum += (r - mean_ret) ** 2
        var_daily = var_sum / (window - 1)

        # Annual volatility
        term1 = (var_daily + (1.0 + gmean_day_return) ** 2) ** trading_periods
        term2 = (1.0 + gmean_day_return) ** (2 * trading_periods)
        annual_vol = np.sqrt(term1 - term2)

        result[i] = annual_vol

    return result


# Fractional Differentation
@njit(cache=True)
def fracdiff_weights(d: float, max_len: int = 10000, tol: float = 1e-5) -> np.ndarray:
    w = np.empty(max_len, dtype=np.float64)
    w[0] = 1.0

    for k in range(1, max_len):
        w_k = -w[k - 1] * (d - k + 1) / k
        if abs(w_k) < tol:
            return w[:k]  # Return truncated array
        w[k] = w_k

    return w


@njit(cache=True, parallel=False)
def _fracdiff_core(prices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = len(prices)
    m = len(weights)
    result = np.empty(n, dtype=np.float64)

    # Leading NaNs until window is full
    for i in range(m - 1):
        result[i] = np.nan

    # Vectorized dot product for each window
    for i in range(m - 1, n):
        result[i] = np.dot(weights, prices[i - m + 1 : i + 1])

    return result


def fracdiff_ffd(series: pd.Series, d: float, tol: float = 1e-5) -> pd.Series:
    """Fixed-width fractional differentiation: x_t = sum_{k=0}^{m} w_k * p_{t-k}."""
    # Compute weights (reversed for convolution)
    w = fracdiff_weights(d=d, tol=tol)
    w_rev = w[::-1]

    # Convert to numpy, apply core computation
    prices = series.values.astype(np.float64)
    result = _fracdiff_core(prices, w_rev)

    return pd.Series(result, index=series.index, name=series.name)


def find_optimal_d(
    close_prices: pd.Series,
    adf_threshold: float = -2.86,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_steps: int = 21,
    tol: float = 1e-3,
    min_samples: int = 50,
) -> tuple:
    """
    Find minimum d that achieves stationarity (ADF statistic < threshold).
    (optimal_d, adf_statistic, n_valid_samples)
    """
    log_prices = np.log(close_prices.replace(0, 1e-9))
    d_range = np.linspace(d_min, d_max, d_steps)

    for d in d_range:
        # Apply fracdiff to LOG prices (Numba-accelerated)
        ffd = fracdiff_ffd(log_prices, d, tol=tol)
        clean = ffd.dropna()

        if len(clean) < min_samples:
            continue

        try:
            adf_stat = adfuller(clean, maxlag=10, regression="c", autolag="AIC")[0]
            if adf_stat < adf_threshold:
                return d, adf_stat, len(clean)
        except:
            continue

    # Fallback to full differencing
    ffd = fracdiff_ffd(log_prices, d_max, tol=tol)
    clean = ffd.dropna()
    adf_stat = adfuller(clean, maxlag=10, regression="c", autolag="AIC")[0]

    return d_max, adf_stat, len(clean)
