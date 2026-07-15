import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def extend_signal_numba(signal: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(len(signal), dtype=np.bool_)

    for i in range(len(signal)):
        if signal[i]:
            end_idx = min(i + n, len(signal))
            for j in range(i, end_idx):
                result[j] = True
        elif result[i]:
            continue
    return result


def get_typical_price(df: pd.DataFrame) -> pd.Series:
    return ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)


@jit(nopython=True)
def _homodyne_discriminator_kernel(price, cycpart, min_period=6.0, max_period=50.0):
    """
    Numba-optimized kernel for the Homodyne Discriminator.
    Iterates row-by-row to handle recursive amplitude correction and period feedback.
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
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 2]  # Note: Ehlers often uses i-2 here for smoothing
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
            # use arctan to get radians, result is in bars
            period[i] = 2 * np.pi / np.arctan(im[i] / re[i])
        else:
            period[i] = period[i - 1]  # Carry forward if undefined

        # 11. Clamp Period change rate (max 50% increase, max 33% decrease)
        # Prevents wild swings in period calculation
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
        # Applies the user scalar 'cycpart' (usually 0.5 to get half-cycle)
        val = smooth_period[i] * cycpart
        dom_cycle[i] = val

    return dom_cycle


def ehler_dominant_cycle(
    price: pd.Series, cycpart: float = 0.5, min_period: int = 6, max_period: int = 50
) -> pd.Series:
    dom_cycle_vals = _homodyne_discriminator_kernel(price, cycpart, min_period, max_period)
    return dom_cycle_vals


@jit(nopython=True)
def _adaptive_ehlers_filter_kernel(price, dominant_period, momentum_ratio=0.25, length_mult: float = 1.0):
    """
    Adaptive Ehlers Filter using dominant cycle period.

    The filter uses momentum-weighted coefficients where both the lookback window
    and the momentum lag scale with the measured cycle period.

    Parameters
    ----------
    price : ndarray
        Price array (e.g., (H+L)/2 or Close).
    dominant_period : ndarray
        Adaptive period in bars (from Homodyne Discriminator).
    momentum_ratio : float
        Fraction of period to use as momentum lag (default 0.25 = quarter cycle).

    Returns
    -------
    filt : ndarray
        Adaptive filtered output.

    """
    n = len(price)
    filt = np.zeros(n)

    for i in range(n):
        # Get current period (clamp to reasonable bounds)
        period = dominant_period[i]
        if period < 6:
            period = 6
        if period > 50:
            period = 50

        # Adaptive lookback length (integer bars)
        length = int(period * length_mult)
        mom_lag = int(momentum_ratio * period)
        if mom_lag < 1:
            mom_lag = 1

        # Check if we have enough history
        if i < length + mom_lag:
            filt[i] = price[i]
            continue

        # Compute momentum-based coefficients
        num = 0.0
        sum_coef = 0.0

        for k in range(length):
            # Coefficient = absolute momentum over mom_lag bars
            coef = abs(price[i - k] - price[i - k - mom_lag])

            # Accumulate weighted price and coefficient sum
            num += coef * price[i - k]
            sum_coef += coef

        # Normalized weighted average
        if sum_coef > 1e-10:
            filt[i] = num / sum_coef
        else:
            # Fallback to simple average if no momentum detected
            simple_avg = 0.0
            for k in range(length):
                simple_avg += price[i - k]
            filt[i] = simple_avg / length

    return filt


def adaptive_ehlers_filter(price, dominant_period, momentum_ratio: float = 0.25, length_mult: float = 1.0):
    price_arr = price.values.astype(np.float64) if isinstance(price, pd.Series) else np.asarray(price, dtype=np.float64)

    if isinstance(dominant_period, pd.Series):
        period_arr = dominant_period.values.astype(np.float64)
    else:
        period_arr = np.asarray(dominant_period, dtype=np.float64)

    filt_vals = _adaptive_ehlers_filter_kernel(price_arr, period_arr, momentum_ratio, length_mult)
    return filt_vals


def adaptive_atr_ehlers(
    data: pd.DataFrame,
    adaptive_period: pd.Series,
    min_eff_len: float = 3.0,
    max_eff_len: float = 50.0,
    base_multiplier: float = 1.0,
    min_multiplier: float = 0.5,
    max_multiplier: float = 2.0,
    period_ref: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    r"""
    Adaptive ATR driven by Ehlers dominant cycle period.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'High', 'Low', 'Close'.
    adaptive_period : pd.Series
        Output from `adaptive_period_ehlers(price)`, in bars.
        Expected to be bounded e.g. between 6 and 50 bars.
    min_eff_len : float, default 3.0
        Minimum effective ATR length (in bars) for stability.
        Used to clamp 0.5 * period_t.
    max_eff_len : float, default 50.0
        Maximum effective ATR length (in bars).
    base_multiplier : float, default 1.0
        Baseline ATR multiplier around which the adaptive multiplier will vary.
    min_multiplier : float, default 0.5
        Minimum allowed ATR multiplier (prevents ultra-tight stops).
    max_multiplier : float, default 2.0
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
        Per-bar stop “unit” = atr_adaptive * mult_adaptive.
        For a long trade, a 1R stop would be: entry_price - stop_unit_t.

    Notes
    -----
    - Effective ATR length at bar t is:
          eff_len_t = clip(0.5 * period_t, min_eff_len, max_eff_len)
      and smoothing factor:
          alpha_t = 1.0 / eff_len_t

    - ATR recursion:
          ATR_t = (1 - alpha_t) * ATR_{t-1} + alpha_t * TR_t

    - Multiplier from Ehlers period:
          mult_t = clip(min_multiplier,
                        max_multiplier,
                        base_multiplier * (period_t / period_ref))

      where period_ref is a fixed reference period (e.g. midpoint).

    """
    idx = data.index
    high = data["High"].astype(float).values
    low = data["Low"].astype(float).values
    close = data["Close"].astype(float).values

    period_arr = adaptive_period
    n = len(idx)

    # --- True Range (standard) ---
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])

    # --- Effective ATR length = period, clamped ---
    eff_len = period_arr
    eff_len = np.clip(eff_len, min_eff_len, max_eff_len)

    # Smoothing factor per bar
    alpha = 1.0 / eff_len

    # --- Recursive ATR with time-varying alpha ---
    atr_adaptive = np.full(n, np.nan, dtype=float)

    # Seed ATR as simple mean of initial window where we have valid TR
    # Use first non-NaN alpha values; here alpha is never NaN if period is valid.
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
            a_t = alpha[t - 1] if t > 0 and np.isfinite(alpha[t - 1]) else 1.0 / min_eff_len
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

    mult_raw[~np.isfinite(mult_raw)] = base_multiplier

    mult_clipped = np.clip(mult_raw, min_multiplier, max_multiplier)
    mult_series = pd.Series(mult_clipped, index=idx, name="ATR_Multiplier_Ehlers")

    # Final per-bar stop unit
    stop_unit = atr_series * mult_series
    stop_unit.name = "ATR_StopUnit_Ehlers"

    return atr_series  # , mult_series, stop_unit


@jit(nopython=True)
def _adaptive_rsi_kernel(close_prices: np.ndarray, smooth_period: np.ndarray, cyc_part: float):
    n = len(close_prices)
    rsi = np.full(n, np.nan)

    for t in range(1, n):  # Start at 1 (need previous close)
        sp = smooth_period[t]

        # Skip if period is invalid
        if not np.isfinite(sp) or sp < 1.0:
            continue

        # Adaptive window length
        length = int(cyc_part * sp)
        if length < 1:
            length = 1

        # Window boundaries
        start = t - length + 1
        if start < 1:
            start = 1

        # Accumulate up/down moves
        cu = 0.0
        cd = 0.0

        for k in range(start, t + 1):
            diff = close_prices[k] - close_prices[k - 1]
            if diff > 0.0:
                cu += diff
            elif diff < 0.0:
                cd += -diff

        # Calculate RSI (avoid division by zero)
        total = cu + cd
        if total > 1e-10:
            rsi[t] = 100.0 * cu / total
        else:
            # No movement in window - use neutral 50
            rsi[t] = 50.0

    return rsi


@jit(nopython=True)
def _laguerre_rsi_kernel(
    price: np.ndarray,
    smooth_period: np.ndarray,
    cyc_part: float,
    gamma_fixed: float,
    use_adaptive_gamma: bool,
) -> np.ndarray:
    """
    Ehlers-style Laguerre RSI kernel.

    Returns values on a 0..100 scale. Leading values can be NaN during warmup.
    """
    n = len(price)
    out = np.full(n, np.nan)

    if n == 0:
        return out

    # Laguerre recursive state
    L0 = price[0]
    L1 = price[0]
    L2 = price[0]
    L3 = price[0]

    for t in range(1, n):
        p = price[t]

        # Gamma selection
        if use_adaptive_gamma:
            sp = smooth_period[t]
            if (not np.isfinite(sp)) or sp < 1.0:
                continue

            length = int(cyc_part * sp)
            if length < 1:
                length = 1

            # EMA mapping: alpha = 2/(length+1), gamma = 1-alpha
            alpha = 2.0 / (length + 1.0)
            gamma = 1.0 - alpha
        else:
            gamma = gamma_fixed

        # Keep strictly inside (0, 1) for numerical stability
        if gamma < 1e-6:
            gamma = 1e-6
        elif gamma > 1.0 - 1e-6:
            gamma = 1.0 - 1e-6

        # Laguerre recursion update
        L0_old = L0
        L1_old = L1
        L2_old = L2

        L0 = (1.0 - gamma) * p + gamma * L0
        L1 = -gamma * L0 + L0_old + gamma * L1
        L2 = -gamma * L1 + L1_old + gamma * L2
        L3 = -gamma * L2 + L2_old + gamma * L3

        # Up/down accumulators
        cu = 0.0
        cd = 0.0

        d01 = L0 - L1
        if d01 >= 0.0:
            cu += d01
        else:
            cd += -d01

        d12 = L1 - L2
        if d12 >= 0.0:
            cu += d12
        else:
            cd += -d12

        d23 = L2 - L3
        if d23 >= 0.0:
            cu += d23
        else:
            cd += -d23

        denom = cu + cd
        if denom > 1e-12:
            out[t] = 100.0 * (cu / denom)
        else:
            out[t] = 50.0

    return out


def laguerre_rsi(
    df: pd.DataFrame,
    gamma: float = 0.5,
    price_col: str = "Close",
) -> pd.Series:
    """
    Classic Laguerre RSI with fixed gamma.

    Returns
    -------
    pd.Series
        Laguerre RSI on a 0..100 scale. Leading values can be NaN during warmup.

    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {price_col}")

    price = df[price_col].to_numpy(np.float64)
    dummy_sp = np.ones(len(price), dtype=np.float64)  # not used when adaptive gamma is off
    vals = _laguerre_rsi_kernel(price, dummy_sp, 1.0, gamma, False)
    return pd.Series(vals, index=df.index, name=f"LagRSI(gamma={gamma})")


def adaptive_laguerre_rsi(
    df: pd.DataFrame,
    cyc_part: float = 0.5,
    min_period: float = 6.0,
    max_period: float = 50.0,
) -> pd.Series:
    """
    Adaptive Laguerre RSI where gamma follows Ehlers dominant cycle period.

    This lives side-by-side with `adaptive_rsi` for comparative research.

    Returns
    -------
    pd.Series
        Adaptive Laguerre RSI on a 0..100 scale.
        Leading values can be NaN during cycle warmup.

    """
    required_cols = ["High", "Low", "Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    typical = ((df["High"] + df["Low"] + df["Close"]) / 3.0).to_numpy(np.float64)
    smooth_period = ehler_dominant_cycle(typical, cycpart=1.0, min_period=min_period, max_period=max_period)

    vals = _laguerre_rsi_kernel(typical, smooth_period, cyc_part, 0.5, True)
    return pd.Series(vals, index=df.index, name="AdaptiveLagRSI")


def adaptive_rsi(
    df: pd.DataFrame,
    cyc_part: float = 0.5,
    min_period: float = 6.0,
    max_period: float = 50.0,
) -> pd.Series:
    """
    Ehlers-style adaptive RSI based on homodyne period estimation.

    The RSI lookback window adapts to the dominant market cycle,
    making it more responsive during short cycles (trending) and
    smoother during long cycles (ranging).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    cyc_part : float, default 0.5
        Fraction of the dominant period used as RSI lookback.
        - 0.5 = half-cycle (default, matches Ehlers' convention)
        - 1.0 = full cycle (smoother)
        - 0.25 = quarter-cycle (faster, noisier)
    min_period, max_period : float
        Hard bounds for the instantaneous period estimate.

    Returns
    -------
    pd.Series
        Adaptive RSI on a 0..100 scale.
        Leading values can be NaN during cycle warmup.

    Notes
    -----
    - This indicator remains available alongside `adaptive_laguerre_rsi`.
    - Cycle estimation uses typical price, while RSI momentum uses Close deltas.

    """
    required_cols = ["High", "Low", "Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Calculate typical price for cycle detection
    price = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)
    close = df["Close"].to_numpy(np.float64)

    # Get dominant cycle period (returns numpy array)
    smooth_period = ehler_dominant_cycle(price, cycpart=1.0, min_period=min_period, max_period=max_period)

    # Calculate adaptive RSI
    rsi_values = _adaptive_rsi_kernel(close, smooth_period, cyc_part)

    # Return as Series with original index
    return pd.Series(rsi_values, index=df.index, name="AdaptiveRSI")


@jit(nopython=True)
def _adaptive_stochastic_kernel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, smooth_period: np.ndarray, cyc_part: float
) -> np.ndarray:
    r"""
    JIT-compiled kernel for adaptive stochastic calculation.

    Stochastic Formula:
    $$
    \text{Stochastic}_t = \frac{C_t - LL}{HH - LL} \times 100
    $$

    where:
    - $C_t$ = Close price at bar t
    - $LL$ = Lowest low over adaptive window
    - $HH$ = Highest high over adaptive window
    - Adaptive window = $\lfloor \text{cyc_part} \times \text{smooth_period}_t \rfloor$

    Parameters
    ----------
    high : np.ndarray
        High prices (length n)
    low : np.ndarray
        Low prices (length n)
    close : np.ndarray
        Close prices (length n)
    smooth_period : np.ndarray
        Dominant cycle period estimate at each bar (from ehler_dominant_cycle)
    cyc_part : float
        Fraction of cycle period to use as lookback window

    Returns
    -------
    np.ndarray
        Stochastic values (0-100 scale), NaN where insufficient data

    """
    n = len(close)
    stoch = np.full(n, np.nan)

    for t in range(1, n):  # Start at 1 (need at least 2 bars for window)
        sp = smooth_period[t]

        # Skip if period is invalid
        if not np.isfinite(sp) or sp < 1.0:
            continue

        # Adaptive window length (half-cycle default per Ehlers)
        length = int(cyc_part * sp)
        if length < 1:
            length = 1

        # Window boundaries
        start = t - length + 1
        if start < 0:
            start = 0

        # Find highest high and lowest low in adaptive window
        HH = high[start]
        LL = low[start]

        for k in range(start, t + 1):
            if high[k] > HH:
                HH = high[k]
            if low[k] < LL:
                LL = low[k]

        # Calculate stochastic (avoid division by zero)
        range_hl = HH - LL
        if range_hl > 1e-10:  # Non-zero range
            stoch[t] = 100.0 * (close[t] - LL) / range_hl
        else:
            # No range in window (flat market) - use neutral 50
            # This prevents division by zero and represents indecision
            stoch[t] = 50.0

    return stoch


def adaptive_stochastic(
    df: pd.DataFrame,
    cyc_part: float = 0.5,
    min_period: float = 6.0,
    max_period: float = 50.0,
) -> pd.Series:
    # Validate required columns
    required_cols = ["High", "Low", "Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Calculate typical price for cycle detection (Ehlers convention)
    price = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)

    # Extract OHLC arrays
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    close = df["Close"].to_numpy(np.float64)

    # Get dominant cycle period (returns numpy array)
    # NOTE: Assumes ehler_dominant_cycle function exists in scope
    smooth_period = ehler_dominant_cycle(
        price,
        cycpart=1.0,  # Use full cycle for period measurement
        min_period=min_period,
        max_period=max_period,
    )

    # Calculate adaptive stochastic
    stoch_values = _adaptive_stochastic_kernel(high, low, close, smooth_period, cyc_part)

    # Return as Series with original index
    return pd.Series(stoch_values, index=df.index, name="AdaptiveStochastic")


@jit(nopython=True)
def _mean_absolute_deviation_kernel(
    typical_price: np.ndarray, sma: np.ndarray, smooth_period: np.ndarray, cyc_part: float
) -> np.ndarray:
    """
    Calculate mean absolute deviation (MAD) using adaptive windows.

    MAD = (1/n) * Σ|TP_i - SMA_i| over adaptive window length

    Parameters
    ----------
    typical_price : np.ndarray
        Typical price series (H+L+C)/3
    sma : np.ndarray
        Simple moving average of typical price (pre-computed)
    smooth_period : np.ndarray
        Dominant cycle period at each bar
    cyc_part : float
        Fraction of cycle to use as window length

    Returns
    -------
    np.ndarray
        Mean absolute deviation at each bar

    """
    n = len(typical_price)
    mad = np.full(n, np.nan)

    for t in range(n):
        sp = smooth_period[t]

        # Skip if period is invalid or SMA is NaN
        if not np.isfinite(sp) or sp < 1.0 or not np.isfinite(sma[t]):
            continue

        # Adaptive window length
        length = int(cyc_part * sp)
        if length < 1:
            length = 1

        # Window boundaries
        start = t - length + 1
        if start < 0:
            start = 0

        # Accumulate absolute deviations
        sum_abs_dev = 0.0
        count = 0

        for k in range(start, t + 1):
            if np.isfinite(typical_price[k]) and np.isfinite(sma[k]):
                sum_abs_dev += abs(typical_price[k] - sma[k])
                count += 1

        # Calculate MAD (avoid division by zero)
        if count > 0:
            mad[t] = sum_abs_dev / count
        else:
            mad[t] = np.nan

    return mad


def adaptive_cci(
    df: pd.DataFrame,
    cyc_part: float = 0.5,
    min_period: float = 6.0,
    max_period: float = 50.0,
    constant: float = 0.015,
) -> pd.Series:
    """
    Ehlers-style adaptive CCI based on homodyne period estimation.

    The CCI lookback window adapts to the dominant market cycle,
    making it more responsive during short cycles (trending) and
    smoother during long cycles (ranging).

    Mathematical Formulation:
    $$
    CCI_t = \\frac{TP_t - SMA_t(TP)}{c \\cdot MAD_t}
    $$

    where:
    - $TP_t = (H_t + L_t + C_t) / 3$ (Typical Price)
    - $SMA_t(TP)$ = Simple Moving Average over adaptive window
    - $MAD_t = \\frac{1}{n} \\sum_{i=t-n+1}^{t} |TP_i - SMA_i|$ (Mean Absolute Deviation)
    - $c = 0.015$ (Lambert's constant, normalizes ~70-80% of values within ±100)
    - $n = \\lfloor cyc\\_part \\times period_t \\rfloor$ (adaptive window length)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    cyc_part : float, default 0.5
        Fraction of the dominant period used as CCI lookback.
        - 0.5 = half-cycle (default, matches Ehlers' convention)
        - 1.0 = full cycle (smoother)
        - 0.25 = quarter-cycle (faster, noisier)
    min_period, max_period : float
        Hard bounds for the instantaneous period estimate.
    constant : float, default 0.015
        Lambert's scaling constant. Standard CCI uses 0.015.
        Modify only if empirically justified for specific asset class.

    Returns
    -------
    pd.Series
        Adaptive CCI values (typically range ±100, but unbounded).
        NaN where insufficient data for cycle detection or calculation.

    Notes
    -----
    - Unlike fixed-period CCI, this adapts to market regime changes
    - Shorter detected cycles → faster CCI response (good for trends)
    - Longer detected cycles → smoother CCI (good for ranges)
    - No look-ahead bias: all calculations use [0:t] historical data only

    Validation Requirements
    -----
    - Backtest with strict point-in-time simulation (no future data leakage)
    - Verify cycle period bounds (min_period, max_period) match asset frequency
    - Compare distributions vs fixed-period CCI(20) for regime detection

    Performance Characteristics
    -----
    - Time: O(n) for rolling SMA + O(n×k) for MAD kernel ≈ O(n×k_avg)
    - Space: O(n) for arrays (4 arrays × n bars)
    - Bottleneck: MAD calculation (JIT-compiled, ~10x faster than pure Python)

    """
    # Calculate typical price (used for both cycle detection and CCI)
    typical_price = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)

    # Get dominant cycle period (returns numpy array)
    # Assumes ehler_dominant_cycle is available from your codebase
    smooth_period = ehler_dominant_cycle(typical_price, cycpart=1.0, min_period=min_period, max_period=max_period)

    # Calculate adaptive SMA using pandas rolling
    # Strategy: Use maximum possible window for rolling, then select adaptive values
    max_window = int(np.ceil(max_period * cyc_part))

    # Convert to Series for rolling operations
    tp_series = pd.Series(typical_price, index=df.index)

    # Pre-compute all possible SMAs up to max window
    # min_periods=1 allows calculation even with partial windows
    sma_series = tp_series.rolling(window=max_window, min_periods=1).mean()
    sma = sma_series.to_numpy(np.float64)

    # Calculate adaptive mean absolute deviation (MAD)
    mad = _mean_absolute_deviation_kernel(typical_price, sma, smooth_period, cyc_part)

    # Calculate CCI
    # CCI = (TP - SMA) / (constant * MAD)
    cci = np.full(len(typical_price), np.nan)

    # Vectorized calculation where MAD is valid
    valid_mask = np.isfinite(mad) & (mad > 1e-10)
    cci[valid_mask] = (typical_price[valid_mask] - sma[valid_mask]) / (constant * mad[valid_mask])

    # Return as Series with original index
    return pd.Series(cci, index=df.index, name="Adaptive_CCI")


def rolling_geometric_mean(returns: pd.Series, window: int) -> pd.Series:
    # 1. Log returns (handle zeros/negatives carefully)
    log_returns = np.log1p(returns.fillna(0))

    # 2. Rolling sum using optimized convolution or cumsum
    values = log_returns.values
    ret_cumsum = np.cumsum(np.insert(values, 0, 0.0))

    # Rolling Sum = CumSum[i] - CumSum[i-window]
    rolling_sum = ret_cumsum[window:] - ret_cumsum[:-window]

    # 3. Average and Exponentiate
    mean_log = rolling_sum / window
    geo_mean = np.expm1(mean_log)
    result = np.full(len(returns), np.nan)
    result[window - 1 :] = geo_mean

    return pd.Series(result, index=returns.index, name="geometric_mean")


def rolling_geometric_bollinger_bands(price: pd.Series, window: int = 20, num_std: float = 2.0):
    """Geometric Bollinger Bands with adaptive Ehlers middle band."""
    # --- Returns ---
    returns = price.pct_change()

    # Rolling geometric mean of returns
    rolling_gmean = rolling_geometric_mean(returns, window=window)

    # Rolling variance
    rolling_var = returns.rolling(window).var(ddof=1)

    # --- Geometric Volatility ---
    gmean_plus_1 = 1 + rolling_gmean
    part_a = rolling_var + gmean_plus_1**2

    # Avoid exploding with large powers by clipping
    annualized_var = (part_a.clip(lower=1e-12)) ** window - (gmean_plus_1.clip(lower=1e-12)) ** (2 * window)

    rolling_geo_std = np.sqrt(annualized_var.clip(lower=0)) * 100  # percent

    # Ensure Series alignment
    rolling_geo_std = pd.Series(rolling_geo_std.values, index=price.index, name="geo_std")

    # --- Middle Band (adaptive Ehlers filter) ---
    middle_band_np = adaptive_ehlers_filter(
        price=price.values.astype(np.float64),
        dominant_period=ehler_dominant_cycle(price.values.astype(np.float64), cycpart=1.0, min_period=6, max_period=50),
        length_mult=0.5,
        momentum_ratio=0.7,
    )

    # Convert back to aligned Series
    middle_band = pd.Series(middle_band_np, index=price.index, name="middle_band")

    # --- Final Bands ---
    # Geometric std as percentage → convert to price points
    price_std = middle_band * (rolling_geo_std / 100)

    upper_band = middle_band + num_std * price_std
    lower_band = middle_band - num_std * price_std

    upper_band.name = "upper_band"
    lower_band.name = "lower_band"

    return pd.DataFrame(
        {
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "rolling_geo_std": rolling_geo_std,
        }
    )


@jit(nopython=True)
def _rolling_linreg_numba(data: np.ndarray, periods: np.ndarray) -> np.ndarray:
    n = len(data)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        period = int(periods[i])

        if period < 2 or i < 1:
            result[i] = 0.0
            continue

        # Window bounds
        start_idx = max(0, i - period + 1)
        window_len = i - start_idx + 1

        if window_len < 2:
            result[i] = 0.0
            continue

        # Extract window
        y = data[start_idx : i + 1]
        x = np.arange(window_len, dtype=np.float64)

        # Calculate linear regression coefficients
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = 0.0
        denominator = 0.0

        for j in range(window_len):
            x_diff = x[j] - x_mean
            numerator += x_diff * (y[j] - y_mean)
            denominator += x_diff * x_diff

        if denominator != 0.0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Value at last position (current bar)
            result[i] = slope * x[-1] + intercept
        else:
            result[i] = 0.0

    return result


@jit(nopython=True)
def _rolling_highest_lowest(high: np.ndarray, low: np.ndarray, periods: np.ndarray) -> tuple:
    n = len(high)
    highest = np.zeros(n, dtype=np.float64)
    lowest = np.zeros(n, dtype=np.float64)

    for i in range(n):
        period = int(periods[i])

        if period < 1:
            period = 20

        start_idx = max(0, i - period + 1)

        highest[i] = np.max(high[start_idx : i + 1])
        lowest[i] = np.min(low[start_idx : i + 1])

    return highest, lowest


@jit(nopython=True)
def _rolling_mean_variable(data: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Calculate rolling mean with variable periods.

    Parameters
    ----------
    data : np.ndarray
        Input data
    periods : np.ndarray
        Rolling periods at each position

    Returns
    -------
    np.ndarray
        Rolling mean values

    """
    n = len(data)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        period = int(periods[i])

        if period < 1:
            period = 20

        start_idx = max(0, i - period + 1)
        result[i] = np.mean(data[start_idx : i + 1])

    return result


def _calculate_squeeze_momentum(df: pd.DataFrame, source: pd.Series, dominant_period: np.ndarray) -> pd.Series:
    """
    Calculate squeeze momentum value using linear regression.

    Formula:
    val = linreg(source - avg(avg(highest(high, period), lowest(low, period)),
                              sma(close, period)),
                 period, 0)

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'High', 'Low', 'Close'
    source : pd.Series
        Source price series (typically typical price)
    dominant_period : np.ndarray
        Dominant period at each position

    Returns
    -------
    pd.Series
        Linear regression momentum values

    """
    # Convert to numpy arrays for Numba
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    source_arr = source.to_numpy(dtype=np.float64)

    # Ensure dominant_period is the right type
    if not isinstance(dominant_period, np.ndarray):
        dominant_period = np.array(dominant_period, dtype=np.float64)

    # 1. Calculate highest high and lowest low
    highest_high, lowest_low = _rolling_highest_lowest(high, low, dominant_period)

    # 2. Midpoint of range
    midpoint = (highest_high + lowest_low) / 2.0

    # 3. Rolling SMA of close
    sma_close = _rolling_mean_variable(close, dominant_period)

    # 4. Average of midpoint and SMA (basis)
    basis = (midpoint + sma_close) / 2.0

    # 5. Deviation from basis
    deviation = source_arr - basis

    # 6. Linear regression on deviation
    linreg_val = _rolling_linreg_numba(deviation, dominant_period)

    return pd.Series(linreg_val, index=df.index)


def calculate_keltner_channel(df):
    price = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)
    dominant_period = ehler_dominant_cycle(price, cycpart=1.0, min_period=6, max_period=50)
    # --- Middle Band (adaptive Ehlers filter) ---
    middle = adaptive_ehlers_filter(price=price, dominant_period=dominant_period, length_mult=0.5, momentum_ratio=0.7)
    atr = adaptive_atr_ehlers(df, dominant_period)
    upper = middle + (atr * 1.5)
    lower = middle - (atr * 1.5)
    return pd.DataFrame({"middle_band": middle, "upper_band": upper, "lower_band": lower, "atr": atr})


def squeeze_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Squeeze Indicator with adaptive dominant cycle period.

    The squeeze occurs when Bollinger Bands are inside Keltner Channels,
    indicating low volatility and potential breakout conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with columns: 'High', 'Low', 'Close', 'Open'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'sqz_on': bool, True when squeeze is on (BBands inside Keltner)
        - 'sqz_off': bool, True when squeeze is off (BBands outside Keltner)
        - 'sqz_val': float, momentum value based on linear regression
        - 'dominant_period': float, adaptive cycle period from Ehlers
        - 'bb_upper', 'bb_middle', 'bb_lower': Bollinger Band values
        - 'kc_upper', 'kc_middle', 'kc_lower': Keltner Channel values

    """
    # 1. Calculate typical price (HLC/3)
    price = (df["High"] + df["Low"] + df["Close"]) / 3

    # 2. Get dominant cycle period using Ehlers method
    dominant_period = ehler_dominant_cycle(price.to_numpy(np.float64), cycpart=1.0, min_period=6, max_period=50)

    # 3. Calculate Bollinger Bands (geometric method with 0.7 std dev)
    bbands = rolling_geometric_bollinger_bands(price, window=20, num_std=0.7)

    # 4. Calculate Keltner Channels
    kcdf = calculate_keltner_channel(df)

    # 5. Determine squeeze conditions
    # Squeeze ON: BBands compressed inside Keltner (low volatility)
    sqz_on = (bbands["lower_band"] > kcdf["lower_band"]) & (bbands["upper_band"] < kcdf["upper_band"])

    # Squeeze OFF: BBands expanded outside Keltner (high volatility)
    sqz_off = (bbands["lower_band"] < kcdf["lower_band"]) & (bbands["upper_band"] > kcdf["upper_band"])

    # 6. Calculate momentum value using linear regression
    sqz_val = _calculate_squeeze_momentum(df=df, source=price, dominant_period=dominant_period)

    # 7. Combine results into output DataFrame
    result = pd.DataFrame(index=df.index)
    result["sqz_on"] = sqz_on
    result["sqz_off"] = sqz_off
    result["sqz_val"] = sqz_val
    # result['dominant_period'] = dominant_period

    # Include band values for reference/plotting
    # result['bb_upper'] = bbands['upper_band']
    # result['bb_middle'] = bbands['middle_band']
    # result['bb_lower'] = bbands['lower_band']
    # result['kc_upper'] = kcdf['upper_band']
    # result['kc_middle'] = kcdf['middle_band']
    # result['kc_lower'] = kcdf['lower_band']

    return result


def calculate_chandelier_exit(df: pd.DataFrame, multiplier: float = 3.0):
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    price = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)
    period = ehler_dominant_cycle(price=price, cycpart=1.0, min_period=6, max_period=50)
    atr, _, _ = adaptive_atr_ehlers(df, period / 2)

    # Highest high and lowest low
    highest_high, lowest_low = _rolling_highest_lowest(high, low, period)

    # Chandelier levels
    chandelier_long = highest_high - (atr * multiplier)
    chandelier_short = lowest_low + (atr * multiplier)

    return chandelier_long, chandelier_short


@jit(nopython=True)
def _adaptive_smi_kernel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, smooth_period: np.ndarray, cyc_part: float
) -> np.ndarray:
    r"""
    JIT-compiled kernel for adaptive SMI calculation.

    SMI Formula:
    $$
    \text{SMI}_t = \frac{2(C_t - M)}{HH - LL} \times 100 = \frac{2C_t - HH - LL}{HH - LL} \times 100
    $$

    where:
    - $C_t$ = Close price at bar t
    - $M$ = Midpoint of range = $(HH + LL) / 2$
    - $LL$ = Lowest low over adaptive window
    - $HH$ = Highest high over adaptive window
    - Adaptive window = $\lfloor \text{cyc_part} \times \text{smooth_period}_t \rfloor$

    Key Differences from Stochastic:
    - Range: -100 to +100 (vs 0 to 100 for Stochastic)
    - Reference: Midpoint of HH-LL range (vs LL for Stochastic)
    - Interpretation:
        * Positive values (>0): Close above midpoint (bullish momentum)
        * Negative values (<0): Close below midpoint (bearish momentum)
        * Zero: Close at exact midpoint (neutral)

    Parameters
    ----------
    high : np.ndarray
        High prices (length n)
    low : np.ndarray
        Low prices (length n)
    close : np.ndarray
        Close prices (length n)
    smooth_period : np.ndarray
        Dominant cycle period estimate at each bar (from ehler_dominant_cycle)
    cyc_part : float
        Fraction of cycle period to use as lookback window

    Returns
    -------
    np.ndarray
        SMI values (-100 to +100 scale), NaN where insufficient data

    """
    n = len(close)
    smi = np.full(n, np.nan)

    for t in range(1, n):  # Start at 1 (need at least 2 bars for window)
        sp = smooth_period[t]

        # Skip if period is invalid
        if not np.isfinite(sp) or sp < 1.0:
            continue

        # Adaptive window length (half-cycle default per Ehlers)
        length = int(cyc_part * sp)
        if length < 1:
            length = 1

        # Window boundaries
        start = t - length + 1
        if start < 0:
            start = 0

        # Find highest high and lowest low in adaptive window
        HH = high[start]
        LL = low[start]

        for k in range(start, t + 1):
            if high[k] > HH:
                HH = high[k]
            if low[k] < LL:
                LL = low[k]

        # Calculate SMI (avoid division by zero)
        range_hl = HH - LL
        if range_hl > 1e-10:  # Non-zero range
            midpoint = (HH + LL) / 2.0
            smi[t] = 200.0 * (close[t] - midpoint) / range_hl
        else:
            # No range in window (flat market) - use neutral 0
            # Zero represents close at midpoint (perfect equilibrium)
            smi[t] = 0.0

    return smi


def adaptive_smi(
    df: pd.DataFrame,
    cyc_part: float = 0.5,
    min_period: float = 6.0,
    max_period: float = 50.0,
) -> pd.Series:
    """
    Calculate adaptive Stochastic Momentum Index using Ehlers dominant cycle.

    The SMI measures where price closes relative to the midpoint of the high-low
    range over an adaptive lookback period. Unlike traditional stochastic (0-100),
    SMI ranges from -100 to +100, providing clearer momentum direction signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC dataframe with columns: ['High', 'Low', 'Close']
    cyc_part : float, default=0.5
        Fraction of dominant cycle to use as lookback (0.5 = half cycle)
    min_period : float, default=6.0
        Minimum cycle period constraint for Ehlers algorithm
    max_period : float, default=50.0
        Maximum cycle period constraint for Ehlers algorithm

    Returns
    -------
    pd.Series
        Adaptive SMI values (-100 to +100) with original DataFrame index

    Examples
    --------
    >>> # Half-cycle SMI (standard)
    >>> df['SMI'] = adaptive_smi(df, cyc_part=0.5)
    >>>
    >>> # Full-cycle SMI (smoother, slower)
    >>> df['SMI_Full'] = adaptive_smi(df, cyc_part=1.0)
    >>>
    >>> # Quarter-cycle SMI (more responsive)
    >>> df['SMI_Fast'] = adaptive_smi(df, cyc_part=0.25)

    Trading Signals
    ---------------
    - SMI > 40: Overbought zone (potential reversal/sell)
    - SMI < -40: Oversold zone (potential reversal/buy)
    - SMI crosses above 0: Bullish momentum shift
    - SMI crosses below 0: Bearish momentum shift
    - Divergence between SMI and price: Trend weakening signal

    """
    # Validate required columns
    required_cols = ["High", "Low", "Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Calculate typical price for cycle detection (Ehlers convention)
    price = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(np.float64)

    # Extract OHLC arrays
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    close = df["Close"].to_numpy(np.float64)

    # Get dominant cycle period (returns numpy array)
    # NOTE: Assumes ehler_dominant_cycle function exists in scope
    smooth_period = ehler_dominant_cycle(
        price,
        cycpart=1.0,  # Use full cycle for period measurement
        min_period=min_period,
        max_period=max_period,
    )

    # Calculate adaptive SMI
    smi_values = _adaptive_smi_kernel(high, low, close, smooth_period, cyc_part)

    # Return as Series with original index
    return pd.Series(smi_values, index=df.index, name="AdaptiveSMI")


# Transform
def fisher_transform(series: pd.Series, clip_value: float = 0.999) -> pd.Series:
    """
    Apply Fisher Transform to convert bounded price oscillators to approximately normal distribution.
    """
    # Handle NaN values
    if series.isna().all():
        return pd.Series(np.nan, index=series.index)

    # Clip values to prevent log(0) or log(negative)
    # Fisher Transform requires input in (-1, 1) exclusive
    clipped = np.clip(series, -clip_value, clip_value)

    # Apply Fisher Transform: 0.5 * ln((1+x)/(1-x))
    # Equivalent to arctanh(x) but explicit for clarity
    fisher = 0.5 * np.log((1.0 + clipped) / (1.0 - clipped))

    return pd.Series(fisher, index=series.index, name=f"Fisher_{series.name}" if series.name else "Fisher")


def cube_root_transform(series: pd.Series, preserve_sign: bool = True) -> pd.Series:
    """
    Apply cube root transform to reduce positive skewness and compress extreme values.
    """
    if preserve_sign:
        # Sign-preserving cube root: sign(x) * |x|^(1/3)
        # This maintains negative values and monotonicity
        result = np.sign(series) * np.power(np.abs(series), 1 / 3)
    else:
        # Standard cube root (requires non-negative values)
        if (series < 0).any():
            raise ValueError(
                "Standard cube root requires non-negative values. Use preserve_sign=True for negative values."
            )
        result = np.power(series, 1 / 3)

    return pd.Series(result, index=series.index, name=f"CubeRoot_{series.name}" if series.name else "CubeRoot")


def inverse_fisher_transform(series: pd.Series, normalize: bool = True) -> pd.Series:
    """
    Apply Inverse Fisher Transform to compress unbounded oscillators into [-1, +1] range.
    """
    # Optional: z-score normalization if input is raw values
    if not normalize:
        mean = series.mean()
        std = series.std(ddof=1)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        series = (series - mean) / std

    # Apply IFT: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    ift = np.tanh(series)

    return pd.Series(ift, index=series.index, name=f"IFT_{series.name}" if series.name else "IFT")
