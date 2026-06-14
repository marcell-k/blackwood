import numpy as np
import pandas as pd
from src.indicators.core import (
    add_overnight_and_week_gap_features,
    calculate_adx,
    calculate_atr,
    calculate_bb,
    calculate_bb_width,
    calculate_keltner_channel,
    calculate_ma,
    calculate_roc,
    calculate_rsi,
    rolling_percentile_rank_sw,
    rolling_zscore,
)
from src.indicators.cycle import (
    adaptive_atr_ehlers,
    adaptive_cci,
    adaptive_rsi,
    adaptive_stochastic,
    ehler_dominant_cycle,
    get_typical_price,
)


# Features
def add_session_features(df: pd.DataFrame, start_time: str = "03:00", end_time: str = "04:30") -> pd.DataFrame:
    # Current session range calculation (existing logic)
    session_data = df.between_time(start_time, end_time)

    daily_ranges = (
        session_data.groupby(session_data.index.date)
        .agg({"High": "max", "Low": "min"})
        .rename(columns={"High": "RangeHigh", "Low": "RangeLow"})
    )

    daily_ranges["SessionRange"] = (
        (daily_ranges["RangeHigh"] - daily_ranges["RangeLow"]) / daily_ranges["RangeHigh"]
    ) * 100
    daily_ranges["SessionRange"] = daily_ranges["SessionRange"].replace([np.inf, -np.inf], np.nan)

    # Previous full day range calculation (0:00-23:15)
    # Financial logic: 23:15 end avoids including end-of-day settlement/auction bars in 24h markets
    prev_day_data = df.between_time("00:00", "23:15")

    prev_day_ranges = (
        prev_day_data.groupby(prev_day_data.index.date)
        .agg({"High": "max", "Low": "min"})
        .rename(columns={"High": "PrevDayHigh", "Low": "PrevDayLow"})
    )

    prev_day_ranges["PrevDayRange"] = (
        (prev_day_ranges["PrevDayHigh"] - prev_day_ranges["PrevDayLow"]) / prev_day_ranges["PrevDayHigh"]
    ) * 100
    prev_day_ranges["PrevDayRange"] = prev_day_ranges["PrevDayRange"].replace([np.inf, -np.inf], np.nan)

    prev_day_ranges = prev_day_ranges.shift(1)
    df_dates = pd.Series(df.index.date, index=df.index)

    # Current session columns
    df["RangeHigh"] = df_dates.map(daily_ranges["RangeHigh"]).ffill()
    df["RangeLow"] = df_dates.map(daily_ranges["RangeLow"]).ffill()
    df["SessionRange"] = df_dates.map(daily_ranges["SessionRange"]).ffill()

    # Previous day columns
    df["PrevDayHigh"] = df_dates.map(prev_day_ranges["PrevDayHigh"]).ffill()
    df["PrevDayLow"] = df_dates.map(prev_day_ranges["PrevDayLow"]).ffill()
    df["PrevDayRange"] = df_dates.map(prev_day_ranges["PrevDayRange"]).ffill()

    # Comparison ratios (vectorized operations)
    # Ratio 1: Session high relative to previous day high (breakout detection)
    df["RangeHigh_vs_PrevDayHigh"] = (df["RangeHigh"] / df["PrevDayHigh"]) * 100
    df["RangeHigh_vs_PrevDayHigh"] = df["RangeHigh_vs_PrevDayHigh"].replace([np.inf, -np.inf], np.nan)

    # Ratio 2: Session high position within previous day's range (normalized 0-100+ scale)
    df["RangeHigh_vs_PrevDayRange"] = (
        (df["RangeHigh"] - df["PrevDayHigh"]) / (df["PrevDayHigh"] - df["PrevDayLow"])
    ) * 100
    df["RangeHigh_vs_PrevDayRange"] = df["RangeHigh_vs_PrevDayRange"].replace([np.inf, -np.inf], np.nan)

    # Ratio 3: Session low relative to previous day low (breakdown detection)
    df["RangeLow_vs_PrevDayLow"] = (df["RangeLow"] / df["PrevDayLow"]) * 100
    df["RangeLow_vs_PrevDayLow"] = df["RangeLow_vs_PrevDayLow"].replace([np.inf, -np.inf], np.nan)

    # Ratio 4: Session low position within previous day's range
    df["RangeLow_vs_PrevDayRange"] = (
        (df["RangeLow"] - df["PrevDayHigh"]) / (df["PrevDayHigh"] - df["PrevDayLow"])
    ) * 100
    df["RangeLow_vs_PrevDayRange"] = df["RangeLow_vs_PrevDayRange"].replace([np.inf, -np.inf], np.nan)
    df["SessionRange_vs_PrevDayRange"] = df["SessionRange"] / df["PrevDayRange"]
    df["SessionRange_vs_PrevDayRange"] = df["SessionRange_vs_PrevDayRange"].replace([np.inf, -np.inf], np.nan)

    return df


def add_features(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    df_orig = df.copy()

    df_15m = (
        df.resample("15min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    df_30m = (
        df.resample("30min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    df_1h = (
        df.resample("1h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    )

    # ── 15m features ──
    features_15m = pd.DataFrame(index=df_15m.index)
    df_15m = add_session_features(df, start_time, end_time)
    features_15m["RangeHigh_vs_PrevDayHigh"] = df_15m["RangeHigh_vs_PrevDayHigh"]
    features_15m["RangeHigh_vs_PrevDayRange"] = df_15m["RangeHigh_vs_PrevDayRange"]
    features_15m["RangeLow_vs_PrevDayLow"] = df_15m["RangeLow_vs_PrevDayLow"]
    features_15m["RangeLow_vs_PrevDayRange"] = df_15m["RangeLow_vs_PrevDayRange"]
    features_15m["SessionRange_vs_PrevDayRange"] = df_15m["SessionRange_vs_PrevDayRange"]

    price_15m = get_typical_price(df_15m)
    period_15m = ehler_dominant_cycle(price_15m, cycpart=1.0)

    adp_rsi_15m = adaptive_rsi(df_15m, cyc_part=1.0)
    if not isinstance(adp_rsi_15m, pd.Series):
        adp_rsi_15m = pd.Series(adp_rsi_15m, index=df_15m.index)
    features_15m["adp_rsi"] = adp_rsi_15m

    features_15m["rsi_divergence"] = np.sign(df_15m["Close"].diff()) - np.sign(features_15m["adp_rsi"].diff())

    # ATR: keep only adaptive, short (5), medium (21) — pruned 2, 4, 10, 14, 20, 40
    atr_adaptive = adaptive_atr_ehlers(df_15m, period_15m)
    features_15m["atr_adaptive"] = rolling_percentile_rank_sw(atr_adaptive, 89)

    close = df_15m.Close
    features_15m["atr_5"] = calculate_atr(df, atr_length=5) / close
    features_15m["atr_21"] = calculate_atr(df, atr_length=21) / close

    # Volatility: keep realized_vol_20 + vol_of_vol_40, pruned vol_of_vol_20
    log_returns = np.log(close / close.shift(1))
    features_15m["realized_vol_20"] = log_returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    features_15m["vol_of_vol_40"] = log_returns.rolling(window=40, min_periods=20).std()

    # Volatility term structure: short/long ATR ratio
    atr_short = calculate_atr(df_15m, 5, 1, "rma")
    atr_long = calculate_atr(df_15m, 21, 1, "rma")
    features_15m["vol_term_structure"] = (atr_short / atr_long).replace([np.inf, -np.inf], np.nan)

    # Momentum: keep ROC_20, pruned roc_momentum_rank
    roc_15m = calculate_roc(df_15m, 20)
    features_15m["ROC_20"] = rolling_zscore(roc_15m, 89)

    features_15m["adx_4"] = calculate_adx(df_15m, 4, 4)
    features_15m["adx_10"] = calculate_adx(df_15m, 2, 2)

    dollar_vol = df_15m["Volume"] * df_15m["Close"]
    features_15m["dollar_vol_pressure"] = (np.sign(df_15m["Close"].diff()) * dollar_vol).rolling(
        21
    ).sum() / dollar_vol.rolling(21).sum()

    features_15m["momentum_composite"] = (
        rolling_zscore(features_15m["ROC_20"], 50)
        + rolling_zscore(features_15m["adp_rsi"] - 50, 50)
        + rolling_zscore(features_15m["adx_10"], 50)
    ) / 3.0

    features_15m["stoch_k_1"] = adaptive_stochastic(df_15m, cyc_part=1.0)
    features_15m["stoch_k_2"] = adaptive_stochastic(df_15m, cyc_part=2.0)
    features_15m["stoch2_stoch1"] = features_15m["stoch_k_1"] - features_15m["stoch_k_2"]
    features_15m["adap_cci"] = adaptive_cci(df, cyc_part=1.0)

    # ── New 15m features ──

    # Calendar: cyclical day-of-week encoding
    day_of_week = df_15m.index.dayofweek
    features_15m["dow_sin"] = np.sin(2 * np.pi * day_of_week / 5)
    features_15m["dow_cos"] = np.cos(2 * np.pi * day_of_week / 5)

    # Bollinger Band position: price within bands (0=lower, 1=upper)
    bb_upper, bb_mid, bb_lower = calculate_bb(df_15m, "Close", 20, 2.0)
    features_15m["bb_position"] = ((close - bb_lower) / (bb_upper - bb_lower)).clip(0, 1)

    # Squeeze: BB width / KC width — low values = consolidation
    bb_width = calculate_bb_width(df_15m, "Close", 20, 2.0)
    kc_upper, kc_mid, kc_lower = calculate_keltner_channel(df_15m, 20, 1.5, 10)
    kc_width = (kc_upper - kc_lower) / kc_mid
    features_15m["squeeze"] = (bb_width / kc_width).replace([np.inf, -np.inf], np.nan)

    # Overnight gap
    gap_df = add_overnight_and_week_gap_features(df_15m, include_weekly=False, include_lagged=False)
    features_15m["overnight_gap"] = gap_df["GapPct"]

    # Candle body ratio: conviction proxy (5-bar rolling mean)
    body = (df_15m["Close"] - df_15m["Open"]).abs()
    total_range = df_15m["High"] - df_15m["Low"]
    features_15m["body_ratio"] = (body / total_range).replace([np.inf, -np.inf], np.nan).rolling(5).mean()

    # ── 1h features ──
    features_1h = pd.DataFrame(index=df_1h.index)
    features_1h["adx_10"] = calculate_adx(df_1h, 2, 2)
    features_1h["rsi"] = calculate_rsi(df_1h, 14)
    bb_upper_1h, _, bb_lower_1h = calculate_bb(df_1h, "Close", 20, 2.0)
    features_1h["bb_position"] = ((df_1h["Close"] - bb_lower_1h) / (bb_upper_1h - bb_lower_1h)).clip(0, 1)

    # ── 30m features ──
    features_30m = pd.DataFrame(index=df_30m.index)

    close_daily = df_30m["Close"]
    returns_daily = close_daily.pct_change().fillna(0)
    adp_rsi_30m = adaptive_rsi(df_30m, cyc_part=1.0)
    if not isinstance(adp_rsi_30m, pd.Series):
        adp_rsi_30m = pd.Series(adp_rsi_30m, index=df_30m.index)

    features_30m["slow_vol"] = rolling_zscore((returns_daily.rolling(20, min_periods=1).std() * np.sqrt(252)), 89)

    features_30m["rsi_divergence"] = np.sign(df_30m["Close"].diff()) - np.sign(adp_rsi_30m.diff())
    price_30m = get_typical_price(df_30m)
    period_30m = ehler_dominant_cycle(price_30m, cycpart=1.0)

    # ATR: keep only adaptive + one representative (atr_10), pruned 2, 4, 20
    atr_adaptive_30m = adaptive_atr_ehlers(df_30m, period_30m)
    features_30m["atr_adaptive"] = rolling_percentile_rank_sw(atr_adaptive_30m, 89)
    features_30m["atr_10"] = rolling_percentile_rank_sw(calculate_atr(df_30m, 10, 1, "rma"), 89)

    # Momentum: keep ROC_20, pruned roc_momentum_rank
    roc_30m = calculate_roc(df_30m, 20)
    features_30m["ROC_20"] = rolling_zscore(roc_30m, 89)

    features_30m["adx_4"] = calculate_adx(df_30m, 4, 4)
    features_30m["adx_10"] = calculate_adx(df_30m, 2, 2)

    close_daily_arr = close_daily.to_numpy(np.float64)
    dominant_period_daily = ehler_dominant_cycle(close_daily_arr, cycpart=0.2)
    features_30m["dominant_period"] = rolling_zscore(pd.Series(dominant_period_daily, index=df_30m.index), 89)
    features_30m["adaptive_atr"] = rolling_zscore(
        rolling_percentile_rank_sw(adaptive_atr_ehlers(df_30m, dominant_period_daily), 89), 89
    )

    adp_rsi_daily = adaptive_rsi(df_30m, cyc_part=0.2)
    features_30m["adaptive_rsi"] = adp_rsi_daily

    # Price distance from MA (z-scored)
    ma_50 = calculate_ma(df_30m, "Close", 50, "ema")
    features_30m["price_vs_ma"] = rolling_zscore((df_30m["Close"] - ma_50) / calculate_atr(df_30m, 10, 1, "rma"), 89)

    # ── Align and merge ──
    features_15m_shifted = features_15m.shift(1)
    features_1h_shifted = features_1h.shift(1)
    features_30m_shifted = features_30m.shift(1)

    features_15m_aligned = features_15m_shifted.reindex(df_orig.index, method="ffill")
    features_30m_aligned = features_30m_shifted.reindex(df_orig.index, method="ffill")
    features_1h_aligned = features_1h_shifted.reindex(df_orig.index, method="ffill")

    features_15m_aligned = features_15m_aligned.add_suffix("_15m")
    features_30m_aligned = features_30m_aligned.add_suffix("_30m")
    features_1h_aligned = features_1h_aligned.add_suffix("_1h")

    result = pd.concat([features_15m_aligned, features_1h_aligned, features_30m_aligned], axis=1)

    feature_cols = (
        list(features_15m_aligned.columns) + list(features_1h_aligned.columns) + list(features_30m_aligned.columns)
    )
    result[feature_cols] = result[feature_cols].fillna(0.0)

    return result
