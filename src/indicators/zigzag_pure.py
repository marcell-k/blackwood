import numpy as np
import pandas as pd


def get_typical_price(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate typical price (H+L+C)/3.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High', 'Low', 'Close' columns.

    Returns
    -------
    np.ndarray
        Typical price values as float64 array.

    """
    return ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)


def _homodyne_discriminator_kernel_pure(
    price: np.ndarray, cycpart: float, min_period: float = 6.0, max_period: float = 50.0
) -> np.ndarray:
    """
    Pure NumPy implementation of Ehlers Homodyne Discriminator.

    Estimates dominant market cycle period using Hilbert Transform technique.
    This is a line-by-line translation of the Numba JIT version, preserving
    exact algorithm logic including recursive amplitude correction and period feedback.

    Parameters
    ----------
    price : np.ndarray
        Price series (typically typical price).
    cycpart : float
        Scalar applied to final period (default 0.5 for half-cycle).
    min_period : float
        Minimum allowed period in bars (default 6).
    max_period : float
        Maximum allowed period in bars (default 50).

    Returns
    -------
    np.ndarray
        Dominant cycle period estimates in bars.

    Notes
    -----
    Algorithm cannot be vectorized due to:
    - Recursive amplitude correction depends on period[i-1]
    - Period smoothing uses previous period values
    - Sequential dependencies in Hilbert Transform calculations

    """
    n = len(price)

    # Initialize arrays for state variables
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    jI = np.zeros(n)
    jQ = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    smooth_period = np.zeros(n)
    dom_cycle = np.zeros(n)

    for i in range(n):
        if i < min_period + 1:
            smooth[i] = price[i]
            period[i] = 0.0
            smooth_period[i] = 0.0
            continue

        # 1. Smooth Data (4-bar WMA)
        # Formula: (4*P + 3*P[1] + 2*P[2] + 1*P[3]) / 10
        smooth[i] = (4 * price[i] + 3 * price[i - 1] + 2 * price[i - 2] + 1 * price[i - 3]) / 10.0

        # 2. Amplitude Correction using PREVIOUS Period
        amp_corr = 0.075 * period[i - 1] + 0.54

        # 3. Detrender (Hilbert Transform part 1)
        detrender[i] = (
            0.0962 * smooth[i] + 0.5769 * smooth[i - 2] - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]
        ) * amp_corr

        # 4. Compute Q1 (Quadrature) and I1 (InPhase)
        q1[i] = (
            0.0962 * detrender[i] + 0.5769 * detrender[i - 2] - 0.5769 * detrender[i - 4] - 0.0962 * detrender[i - 6]
        ) * amp_corr
        i1[i] = detrender[i - 3]

        # 5. Advance Phase of I1 and Q1 by 90 degrees (jI, jQ)
        jI[i] = (0.0962 * i1[i] + 0.5769 * i1[i - 2] - 0.5769 * i1[i - 4] - 0.0962 * i1[i - 6]) * amp_corr
        jQ[i] = (0.0962 * q1[i] + 0.5769 * q1[i - 2] - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6]) * amp_corr

        # 6. Phasor Addition for 3-bar averaging
        i2[i] = i1[i] - jQ[i]
        q2[i] = q1[i] + jI[i]

        # 7. Smooth I2 and Q2
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 2]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 2]

        # 8. Homodyne Discriminator
        # Signal * ComplexConjugate(Signal[1])
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]

        # 9. Smooth Re and Im
        re[i] = 0.2 * re[i] + 0.8 * re[i - 1]
        im[i] = 0.2 * im[i] + 0.8 * im[i - 1]

        # 10. Calculate Period
        # Logic: Period = 2*pi / PhaseChange. PhaseChange = arctan(Im/Re)
        if im[i] != 0 and re[i] != 0:
            period[i] = 2 * np.pi / np.arctan(im[i] / re[i])
        else:
            period[i] = period[i - 1]  # Carry forward if undefined

        # 11. Clamp Period change rate (max 50% increase, max 33% decrease)
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]

        # 12. Clamp hard bounds (6 to 50 bars)
        if period[i] < min_period:
            period[i] = min_period
        if period[i] > max_period:
            period[i] = max_period

        # 13. Smooth Period
        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

        # 14. Final Domain Cycle Calculation
        val = smooth_period[i] * cycpart
        dom_cycle[i] = val

    return dom_cycle


def ehler_dominant_cycle_pure(
    price: np.ndarray, cycpart: float = 0.5, min_period: int = 6, max_period: int = 50
) -> np.ndarray:
    """
    Pure NumPy wrapper for Ehlers dominant cycle detector.

    Parameters
    ----------
    price : np.ndarray
        Price series for cycle detection.
    cycpart : float
        Multiplier for final period (0.5 = half-cycle).
    min_period : int
        Minimum cycle period in bars.
    max_period : int
        Maximum cycle period in bars.

    Returns
    -------
    np.ndarray
        Dominant cycle period estimates.

    """
    return _homodyne_discriminator_kernel_pure(price, cycpart, min_period, max_period)


def adaptive_atr_ehlers_pure(
    data: pd.DataFrame,
    adaptive_period: np.ndarray,
    min_eff_len: float = 3.0,
    max_eff_len: float = 50.0,
    base_multiplier: float = 1.0,
    min_multiplier: float = 0.5,
    max_multiplier: float = 2.0,
    period_ref: float = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Adaptive ATR driven by Ehlers dominant cycle period.

    Pure NumPy/Pandas implementation (no Numba). This function was already
    pure in the original codebase - copied as-is with no modifications.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'High', 'Low', 'Close'.
    adaptive_period : np.ndarray
        Output from ehler_dominant_cycle_pure(), in bars.
    min_eff_len : float
        Minimum effective ATR length (in bars) for stability.
    max_eff_len : float
        Maximum effective ATR length (in bars).
    base_multiplier : float
        Baseline ATR multiplier around which the adaptive multiplier will vary.
    min_multiplier : float
        Minimum allowed ATR multiplier (prevents ultra-tight stops).
    max_multiplier : float
        Maximum allowed ATR multiplier (prevents runaway widening).
    period_ref : float, optional
        Reference period for scaling the multiplier.
        If None, uses the midpoint of the observed adaptive_period.

    Returns
    -------
    atr_adaptive : pd.Series
        Recursive ATR with time-varying alpha corresponding to half the
        adaptive Ehlers period, bounded by [min_eff_len, max_eff_len].
    mult_adaptive : pd.Series
        Adaptive ATR multiplier derived from the Ehlers period.
    stop_unit : pd.Series
        Per-bar stop "unit" = atr_adaptive * mult_adaptive.

    Notes
    -----
    - Effective ATR length at bar t:
          eff_len_t = clip(0.5 * period_t, min_eff_len, max_eff_len)
      and smoothing factor:
          alpha_t = 1.0 / eff_len_t

    - ATR recursion:
          ATR_t = (1 - alpha_t) * ATR_{t-1} + alpha_t * TR_t

    - Multiplier from Ehlers period:
          mult_t = clip(min_multiplier,
                        max_multiplier,
                        base_multiplier * (period_t / period_ref))

    """
    idx = data.index
    high = data["High"].to_numpy(dtype=float, copy=False)
    low = data["Low"].to_numpy(dtype=float, copy=False)
    close = data["Close"].to_numpy(dtype=float, copy=False)

    period_arr = np.asarray(adaptive_period, dtype=float)
    n = idx.size

    # --- True Range (standard) ---
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])

    # --- Effective ATR length = period, clamped ---
    eff_len = np.clip(period_arr, min_eff_len, max_eff_len)

    # Smoothing factor per bar
    alpha = np.divide(
        1.0,
        eff_len,
        out=np.full_like(eff_len, np.nan, dtype=float),
        where=np.isfinite(eff_len),
    )

    # --- Recursive ATR with time-varying alpha ---
    atr_adaptive = np.full(n, np.nan, dtype=float)

    # Seed ATR as simple mean of initial window
    seed_len = int(max(min_eff_len, 3))
    if seed_len < n:
        atr_adaptive[seed_len - 1] = tr[:seed_len].mean()
        start_idx = seed_len
    else:
        atr_adaptive[0] = tr[0]
        start_idx = 1

    for t in range(start_idx, n):
        # If period is NaN, just carry forward previous ATR
        if not np.isfinite(alpha[t]):
            atr_adaptive[t] = atr_adaptive[t - 1]
            continue

        a_t = alpha[t]
        # Guard against extreme values
        if a_t <= 0.0:
            prev_alpha = alpha[t - 1] if t > 0 else np.nan
            a_t = prev_alpha if np.isfinite(prev_alpha) else 1.0 / min_eff_len
        elif a_t >= 1.0:
            a_t = 1.0

        atr_adaptive[t] = (1.0 - a_t) * atr_adaptive[t - 1] + a_t * tr[t]

    atr_series = pd.Series(atr_adaptive, index=idx, name="ATR_EhlersAdaptive")

    # --- Adaptive multiplier from Ehlers period ---
    # Reference period: midpoint of observed adaptive_period if not given
    if period_ref is None:
        valid_period = period_arr[np.isfinite(period_arr)]
        period_ref = 0.5 * (valid_period.min() + valid_period.max()) if valid_period.size > 0 else 20.0

    # Raw multiplier proportional to period / period_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        mult_raw = base_multiplier * (period_arr / period_ref)

    # Replace invalids with base_multiplier
    mult_raw[~np.isfinite(mult_raw)] = base_multiplier

    mult_clipped = np.clip(mult_raw, min_multiplier, max_multiplier)
    mult_series = pd.Series(mult_clipped, index=idx, name="ATR_Multiplier_Ehlers")

    # Final per-bar stop unit
    stop_unit = atr_series * mult_series
    stop_unit.name = "ATR_StopUnit_Ehlers"

    return atr_series, mult_series, stop_unit


def _zigzag_core_pure(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, threshold: np.ndarray, use_high_low: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core zigzag detection with strict high-low alternation.

    Pure NumPy implementation (no Numba). This is a line-by-line translation
    of the JIT-compiled version, preserving exact stateful algorithm logic.

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    threshold : np.ndarray
        Per-bar threshold for reversal detection (typically ATR-based).
    use_high_low : bool
        If True, use high/low for extremes; if False, use close/close.

    Returns
    -------
    zigzag_values : np.ndarray
        Zigzag pivot values (NaN where no pivot).
    pivot_indices : np.ndarray
        Indices of detected pivots.
    pivot_types : np.ndarray
        Pivot types: 1 for high, -1 for low, 0 for none.
    pivot_available_at : np.ndarray
        Bar index when each pivot becomes confirmed (lookahead-free).

    Notes
    -----
    Algorithm cannot be vectorized due to:
    - Stateful tracking of current trend (uptrend/downtrend)
    - Dynamic extreme value updates as new bars arrive
    - Pivot replacement logic when stronger extremes are found
    - Sequential confirmation of pivots only after next pivot is detected

    Temporal integrity:
    - pivot_available_at[i] indicates the bar index when pivot at index i
      becomes tradeable (i.e., after the next opposing pivot is detected)
    - No look-ahead bias: all decisions use only current and past bars

    """
    n = len(high)

    zigzag_values = np.full(n, np.nan)
    pivot_indices = np.full(n, -1, dtype=np.int32)
    pivot_types = np.full(n, 0, dtype=np.int8)
    pivot_available_at = np.full(n, -1, dtype=np.int32)

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


def calculate_zigzag_pure(
    df: pd.DataFrame,
    atr_divisor: float = 5.0,
    use_high_low: bool = False,
) -> pd.DataFrame:
    """
    Calculate adaptive zigzag with lookahead-free availability tracking.

    Pure NumPy/Pandas implementation without Numba dependencies.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC price data with columns: 'High', 'Low', 'Close'.
        Index should be datetime for proper time-series handling.
    atr_divisor : float
        Divisor applied to adaptive ATR to create reversal threshold.
        Lower values = more sensitive (more pivots).
        Higher values = less sensitive (fewer pivots).
    use_high_low : bool
        If True, use high/low prices for extreme detection.
        If False, use close/close (default).

    Returns
    -------
    pd.DataFrame
        Copy of input df with additional columns:
        - 'zigzag': Pivot values (NaN where no pivot)
        - 'pivot_high': Boolean, True at high pivots
        - 'pivot_low': Boolean, True at low pivots
        - 'pivot_available_at': Bar index when pivot becomes tradeable
        - 'zigzag_threshold': Adaptive threshold used for detection

    Examples
    --------
    >>> from src.data.loaders import load_security
    >>> df = load_security('NQ', '2024-01-01', '2024-12-31')
    >>> result = calculate_zigzag_pure(df, atr_divisor=5.0)
    >>> pivots = result[result['zigzag'].notna()]
    >>> print(f"Found {len(pivots)} pivots")

    Notes
    -----
    Performance: ~200-1000ms for 10k bars (10-50x slower than Numba version).
    For production use, prefer calculate_zigzag() from src/indicators/core.py.

    Temporal integrity: All calculations use only current and past data.
    The pivot_available_at column indicates when each pivot becomes tradeable
    (i.e., after the next opposing pivot is detected).

    """
    typical = get_typical_price(df)
    dominant_cycle = ehler_dominant_cycle_pure(typical)
    atr_adaptive, _, _ = adaptive_atr_ehlers_pure(df, adaptive_period=2 * dominant_cycle)

    threshold_series = (atr_adaptive / atr_divisor).ffill().fillna(np.inf)
    threshold_array = threshold_series.values

    zigzag_values, pivot_indices, pivot_types, pivot_available_at = _zigzag_core_pure(
        df["High"].values, df["Low"].values, df["Close"].values, threshold_array, use_high_low=use_high_low
    )

    result_df = df.copy()
    result_df["zigzag"] = zigzag_values
    result_df["pivot_high"] = pivot_types == 1
    result_df["pivot_low"] = pivot_types == -1
    result_df["pivot_available_at"] = pivot_available_at
    result_df["zigzag_threshold"] = threshold_array

    return result_df
