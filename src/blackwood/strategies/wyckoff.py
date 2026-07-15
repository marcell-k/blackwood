from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from blackwood.indicators.cycle import (
    adaptive_atr_ehlers,
    ehler_dominant_cycle,
    get_typical_price,
)


@dataclass
class WyckoffConfig:
    fractal_lookback: int = 5
    atr_divisor: float = 5.0
    use_high_low: bool = True


def _high_low_series(df: pd.DataFrame, use_high_low: bool) -> tuple[pd.Series, pd.Series]:
    """Return (highs, lows) series based on OHLC or Open/Close extremes."""
    if use_high_low:
        return df["High"].astype(float), df["Low"].astype(float)
    highs = pd.Series(
        np.maximum(df["Close"].values, df["Open"].values),
        index=df.index,
        name="HighHL",
    )
    lows = pd.Series(
        np.minimum(df["Close"].values, df["Open"].values),
        index=df.index,
        name="LowHL",
    )
    return highs, lows


def _build_sorted_events(cp: pd.DataFrame, col_value: str, col_avail: str) -> list[tuple[int, int, float]]:
    """Build sorted (avail_bar, pivot_idx, value) event list from confirmed pivots."""
    mask = cp[col_value].notna() & cp[col_avail].notna()
    positions = np.flatnonzero(mask.to_numpy())
    avail_vals = cp[col_avail].to_numpy()
    price_vals = cp[col_value].to_numpy()
    return sorted((int(avail_vals[idx]), int(idx), float(price_vals[idx])) for idx in positions)


def calculate_fractals(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    use_high_low: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    highs, lows = _high_low_series(df, use_high_low)

    # Left window: max/min of bars [t-left, t-1] (excludes current bar)
    prev_left_max = highs.shift(1).rolling(window=left, min_periods=left).max()
    prev_left_min = lows.shift(1).rolling(window=left, min_periods=left).min()

    # Right window: max/min of bars [t+1, t+right]
    next_right_max = highs.rolling(window=right, min_periods=right).max().shift(-right)
    next_right_min = lows.rolling(window=right, min_periods=right).min().shift(-right)

    # Strict inequalities to avoid duplicates at flat peaks/troughs
    is_fractal_high = (highs > prev_left_max) & (highs > next_right_max)
    is_fractal_low = (lows < prev_left_min) & (lows < next_right_min)

    fh_value = highs.where(is_fractal_high)
    fl_value = lows.where(is_fractal_low)

    # Shifted to availability point (lookahead-free)
    fh_avail = fh_value.shift(right).rename("FractalHighAvail")
    fl_avail = fl_value.shift(right).rename("FractalLowAvail")

    return fh_value, fl_value, fh_avail, fl_avail


@njit(cache=True)
def _zigzag_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    threshold: np.ndarray,
    use_high_low: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core zigzag detection with strict high-low alternation."""
    n = len(high)
    zigzag_values = np.full(n, np.nan)
    pivot_indices = np.full(n, -1, dtype=np.int32)
    pivot_types = np.full(n, 0, dtype=np.int8)
    pivot_available_at = np.full(n, -1, dtype=np.int32)

    if n < 2:
        return zigzag_values, pivot_indices, pivot_types, pivot_available_at

    upper_prices = high if use_high_low else close
    lower_prices = low if use_high_low else close

    # Establish initial trend from first 20 bars
    current_extreme = close[0]
    is_uptrend = True
    scan_end = min(20, n)

    for i in range(1, scan_end):
        if close[i] > current_extreme + threshold[i]:
            is_uptrend = True
            break
        elif close[i] < current_extreme - threshold[i]:
            is_uptrend = False
            break
        current_extreme = close[i]

    # Initialize tracking to the best extreme in the first 20 bars
    prices_for_init = upper_prices if is_uptrend else lower_prices
    current_extreme = prices_for_init[0]
    current_extreme_idx = 0
    for i in range(1, scan_end):
        if is_uptrend and upper_prices[i] > current_extreme:
            current_extreme = upper_prices[i]
            current_extreme_idx = i
        elif not is_uptrend and lower_prices[i] < current_extreme:
            current_extreme = lower_prices[i]
            current_extreme_idx = i

    pivot_count = 0
    last_pivot_type = np.int8(0)
    last_pivot_idx = np.int32(-1)
    last_pivot_value = np.nan
    prev_pivot_idx = np.int32(-1)
    start_search_idx = max(1, current_extreme_idx + 1)

    for i in range(start_search_idx, n):
        if is_uptrend:
            if upper_prices[i] > current_extreme:
                current_extreme = upper_prices[i]
                current_extreme_idx = i

            if lower_prices[i] < current_extreme - threshold[i]:
                new_pivot_type = np.int8(1)
                if last_pivot_type == new_pivot_type:
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
                    zigzag_values[current_extreme_idx] = current_extreme
                    pivot_types[current_extreme_idx] = new_pivot_type
                    pivot_indices[pivot_count] = current_extreme_idx
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
            if lower_prices[i] < current_extreme:
                current_extreme = lower_prices[i]
                current_extreme_idx = i

            if upper_prices[i] > current_extreme + threshold[i]:
                new_pivot_type = np.int8(-1)
                if last_pivot_type == new_pivot_type:
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
                    zigzag_values[current_extreme_idx] = current_extreme
                    pivot_types[current_extreme_idx] = new_pivot_type
                    pivot_indices[pivot_count] = current_extreme_idx
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
    """Adaptive zigzag with lookahead-free availability tracking."""
    typical = get_typical_price(df)
    dominant_cycle = ehler_dominant_cycle(typical)
    atr_adaptive = adaptive_atr_ehlers(df, adaptive_period=2 * dominant_cycle)

    threshold_array = (atr_adaptive / atr_divisor).ffill().fillna(np.inf).values

    zigzag_values, _, pivot_types, pivot_available_at = _zigzag_core(
        df["High"].values,
        df["Low"].values,
        df["Close"].values,
        threshold_array,
        use_high_low=use_high_low,
    )

    result_df = df.copy()
    result_df["zigzag"] = zigzag_values
    result_df["pivot_high"] = pivot_types == 1
    result_df["pivot_low"] = pivot_types == -1
    result_df["pivot_available_at"] = pivot_available_at
    result_df["zigzag_threshold"] = threshold_array
    return result_df


def calculate_confirmed_pivots(
    df: pd.DataFrame,
    cfg: WyckoffConfig | None = None,
) -> pd.DataFrame:
    """
    Combine fractals + zigzag into lookahead-safe confirmed pivots.
    Returns: confirmed_high, confirmed_low, confirmed_high_avail,
             confirmed_low_avail, atr_adaptive.
    """
    if cfg is None:
        cfg = WyckoffConfig()

    fh_value, fl_value, _, _ = calculate_fractals(
        df,
        left=cfg.fractal_lookback,
        right=cfg.fractal_lookback,
        use_high_low=cfg.use_high_low,
    )
    zz = calculate_zigzag(df, atr_divisor=cfg.atr_divisor, use_high_low=cfg.use_high_low)

    is_confirmed_high = fh_value.notna() & zz["pivot_high"]
    is_confirmed_low = fl_value.notna() & zz["pivot_low"]

    confirmed_high = fh_value.where(is_confirmed_high)
    confirmed_low = fl_value.where(is_confirmed_low)

    n = len(df)
    high_idx = np.flatnonzero(is_confirmed_high)
    low_idx = np.flatnonzero(is_confirmed_low)

    # Fractal availability: pivot bar + right lookback
    fh_avail = np.full(n, np.nan)
    fl_avail = np.full(n, np.nan)
    if high_idx.size:
        fh_avail[high_idx] = high_idx + cfg.fractal_lookback
    if low_idx.size:
        fl_avail[low_idx] = low_idx + cfg.fractal_lookback

    # Zigzag availability
    zz_avail = zz["pivot_available_at"].replace(-1, np.nan).to_numpy(dtype=np.float64)

    # Confirmed availability: later of fractal and zigzag
    confirmed_high_avail = np.where(is_confirmed_high, np.fmax(fh_avail, zz_avail), np.nan)
    confirmed_low_avail = np.where(is_confirmed_low, np.fmax(fl_avail, zz_avail), np.nan)

    return pd.DataFrame(
        {
            "confirmed_high": confirmed_high,
            "confirmed_low": confirmed_low,
            "confirmed_high_avail": confirmed_high_avail,
            "confirmed_low_avail": confirmed_low_avail,
            "atr_adaptive": zz["zigzag_threshold"].values * cfg.atr_divisor,
        },
        index=df.index,
    )


def _scan_wyckoff_structure(
    df: pd.DataFrame,
    cp: pd.DataFrame,
    atr_divisor: float,
    direction: str,
) -> pd.DataFrame:
    """
    Unified scanner for textbook Wyckoff spring / upthrust.

    Textbook Wyckoff spring (lookahead-safe, one signal per structure).

    Structure:
      - H[-3] < H[-2]
      - H[-1] < H[-2]
      - Exactly one confirmed low between H[-2] and H[-1]
      - That low lies in H[-3] +/- ATR/atr_divisor at H[-1] confirmation bar

    Signal:
      - Emitted exactly once at H[-1] confirmation bar
      - limit_buy_price equals spring low price

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
    is_spring = direction == "spring"

    # Decide which pivots form the "structure" vs. the "test"
    struct_col = "confirmed_high" if is_spring else "confirmed_low"
    test_col = "confirmed_low" if is_spring else "confirmed_high"
    struct_avail = f"{struct_col}_avail"
    test_avail = f"{test_col}_avail"

    struct_events = _build_sorted_events(cp, struct_col, struct_avail)
    test_events = _build_sorted_events(cp, test_col, test_avail)
    atr = cp["atr_adaptive"].to_numpy(dtype=np.float64)

    n = len(df)

    # Output arrays
    signal = np.zeros(n, dtype=np.int8)
    limit_price = np.full(n, np.nan)
    test_price_out = np.full(n, np.nan)
    test_pivot_idx_out = np.full(n, np.nan)
    s3_price = np.full(n, np.nan)
    s2_price = np.full(n, np.nan)
    s1_price = np.full(n, np.nan)
    s3_idx_out = np.full(n, np.nan)
    s2_idx_out = np.full(n, np.nan)
    s1_idx_out = np.full(n, np.nan)
    zone_lower = np.full(n, np.nan)
    zone_upper = np.full(n, np.nan)
    valid_flag = np.zeros(n, dtype=np.int8)

    recent_struct: deque = deque(maxlen=3)
    available_tests: list[tuple[int, float]] = []
    emitted = set()
    s_ptr = 0
    t_ptr = 0

    # Comparison functions for spring vs upthrust
    if is_spring:

        def structure_ok(p3, p2, p1):
            return p3 < p2 and p1 < p2
    else:

        def structure_ok(p3, p2, p1):
            return p3 > p2 and p1 > p2

    for i in range(n):
        new_struct = False

        if is_spring:
            # Absorb tests (lows) first, then structure (highs)
            while t_ptr < len(test_events) and test_events[t_ptr][0] <= i:
                available_tests.append((test_events[t_ptr][1], test_events[t_ptr][2]))
                t_ptr += 1
            while s_ptr < len(struct_events) and struct_events[s_ptr][0] <= i:
                recent_struct.append((struct_events[s_ptr][1], struct_events[s_ptr][2]))
                s_ptr += 1
                new_struct = True
        else:
            # Absorb tests (highs) first, then structure (lows)
            while t_ptr < len(test_events) and test_events[t_ptr][0] <= i:
                available_tests.append((test_events[t_ptr][1], test_events[t_ptr][2]))
                t_ptr += 1
            while s_ptr < len(struct_events) and struct_events[s_ptr][0] <= i:
                recent_struct.append((struct_events[s_ptr][1], struct_events[s_ptr][2]))
                s_ptr += 1
                new_struct = True

        if not new_struct or len(recent_struct) < 3:
            continue

        p3_idx, p3 = recent_struct[0]
        p2_idx, p2 = recent_struct[1]
        p1_idx, p1 = recent_struct[2]
        atr_i = atr[i]

        s3_idx_out[i] = p3_idx
        s2_idx_out[i] = p2_idx
        s1_idx_out[i] = p1_idx
        s3_price[i] = p3
        s2_price[i] = p2
        s1_price[i] = p1

        if np.isnan(atr_i) or atr_i <= 0:
            continue

        tests_between = [(t_idx, t_val) for t_idx, t_val in available_tests if p2_idx < t_idx < p1_idx]

        if len(tests_between) != 1:
            continue

        test_idx, test_val = tests_between[0]
        test_pivot_idx_out[i] = test_idx
        test_price_out[i] = test_val

        zone_hw = atr_i / atr_divisor
        zone_lower[i] = p3 - zone_hw
        zone_upper[i] = p3 + zone_hw
        in_zone = zone_lower[i] <= test_val <= zone_upper[i]

        if not (structure_ok(p3, p2, p1) and in_zone):
            continue

        valid_flag[i] = 1
        key = (p3_idx, p2_idx, p1_idx, test_idx)
        if key in emitted:
            continue
        emitted.add(key)

        signal[i] = 1
        limit_price[i] = test_val

    # Build output DataFrame
    if is_spring:
        return _pack_spring_output(
            df,
            signal,
            limit_price,
            test_price_out,
            test_pivot_idx_out,
            s3_price,
            s2_price,
            s1_price,
            s3_idx_out,
            s2_idx_out,
            s1_idx_out,
            zone_lower,
            zone_upper,
            valid_flag,
        )
    return _pack_upthrust_output(
        df,
        signal,
        limit_price,
        test_price_out,
        test_pivot_idx_out,
        s3_price,
        s2_price,
        s1_price,
        s3_idx_out,
        s2_idx_out,
        s1_idx_out,
        zone_lower,
        zone_upper,
        valid_flag,
    )


def _pack_spring_output(df, signal, limit_price, test_price, test_idx, s3p, s2p, s1p, s3i, s2i, s1i, zl, zu, valid):
    out = df.copy()
    out["spring_signal"] = signal
    out["limit_buy_price"] = limit_price
    out["spring_low_price"] = test_price
    out["spring_low_pivot_idx"] = test_idx
    out["h_m3_price"] = s3p
    out["h_m2_price"] = s2p
    out["h_m1_price"] = s1p
    out["h_m3_idx"] = s3i
    out["h_m2_idx"] = s2i
    out["h_m1_idx"] = s1i
    out["support_lower"] = zl
    out["support_upper"] = zu
    out["spring_valid"] = valid
    return out


def _pack_upthrust_output(df, signal, limit_price, test_price, test_idx, s3p, s2p, s1p, s3i, s2i, s1i, zl, zu, valid):
    out = df.copy()
    out["upthrust_signal"] = signal
    out["limit_sell_price"] = limit_price
    out["upthrust_high_price"] = test_price
    out["upthrust_high_pivot_idx"] = test_idx
    out["l_m3_price"] = s3p
    out["l_m2_price"] = s2p
    out["l_m1_price"] = s1p
    out["l_m3_idx"] = s3i
    out["l_m2_idx"] = s2i
    out["l_m1_idx"] = s1i
    out["resistance_lower"] = zl
    out["resistance_upper"] = zu
    out["upthrust_valid"] = valid
    return out


def wyckoff_spring_textbook_signal(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
    confirmed_pivots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Textbook Wyckoff spring (lookahead-safe, one signal per structure)."""
    cfg = WyckoffConfig(fractal_lookback, atr_divisor, use_high_low)
    cp = confirmed_pivots.copy() if confirmed_pivots is not None else calculate_confirmed_pivots(df, cfg)
    return _scan_wyckoff_structure(df, cp, atr_divisor, direction="spring")


def wyckoff_upthrust_textbook_signal(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
    confirmed_pivots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Textbook Wyckoff upthrust (lookahead-safe, one signal per structure)."""
    cfg = WyckoffConfig(fractal_lookback, atr_divisor, use_high_low)
    cp = confirmed_pivots.copy() if confirmed_pivots is not None else calculate_confirmed_pivots(df, cfg)
    return _scan_wyckoff_structure(df, cp, atr_divisor, direction="upthrust")


def wyckoff_signal(
    df: pd.DataFrame,
    fractal_lookback: int = 5,
    atr_divisor: float = 5.0,
    use_high_low: bool = True,
    confirmed_pivots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Unified textbook Wyckoff signal:
      spring → Signal = 1,  upthrust → Signal = -1,  none → 0.
    """
    cfg = WyckoffConfig(fractal_lookback, atr_divisor, use_high_low)
    cp = confirmed_pivots.copy() if confirmed_pivots is not None else calculate_confirmed_pivots(df, cfg)

    spring_df = _scan_wyckoff_structure(df, cp, atr_divisor, direction="spring")
    upthrust_df = _scan_wyckoff_structure(df, cp, atr_divisor, direction="upthrust")

    spring_mask = spring_df["spring_signal"].to_numpy(dtype=np.int8) == 1
    upthrust_mask = upthrust_df["upthrust_signal"].to_numpy(dtype=np.int8) == 1

    spring_limit = spring_df["limit_buy_price"].to_numpy(dtype=np.float64)
    upthrust_limit = upthrust_df["limit_sell_price"].to_numpy(dtype=np.float64)

    signal = np.zeros(len(df), dtype=np.int8)
    signal[spring_mask] = 1
    signal[upthrust_mask] = -1
    signal[spring_mask & upthrust_mask] = 0  # neutralize ambiguity

    limit_price = np.full(len(df), np.nan, dtype=np.float64)
    limit_price[spring_mask] = spring_limit[spring_mask]
    limit_price[upthrust_mask] = upthrust_limit[upthrust_mask]
    limit_price[spring_mask & upthrust_mask] = np.nan  # neutralize ambiguity

    out = df.copy()
    out["Signal"] = signal
    out["Limit"] = limit_price
    out["atr"] = cp["atr_adaptive"].values
    out["spring_h2"] = spring_df["h_m2_price"].values
    out["upthrust_l2"] = upthrust_df["l_m2_price"].values
    return out
