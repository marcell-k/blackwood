from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
from src.indicators.core import calculate_fractals, calculate_zigzag
from src.indicators.cycle import (
    adaptive_atr_ehlers,
    ehler_dominant_cycle,
    get_typical_price,
)


def calculate_combined_pivots(
    df: pd.DataFrame,
    fractal_lookback: int = 2,
    fractal_use_high_low: bool = True,
    atr_divisor: float = 5.0,
    zigzag_use_high_low: bool = False,
) -> pd.DataFrame:
    """Detect pivots that are BOTH fractals AND zigzag pivots."""
    len(df)

    # Calculate fractals
    fh_value, fl_value, fh_avail, fl_avail = calculate_fractals(
        df,
        left=fractal_lookback,
        right=fractal_lookback,
        use_high_low=fractal_use_high_low,
    )

    # Calculate zigzag
    zz_result = calculate_zigzag(
        df,
        atr_divisor=atr_divisor,
        use_high_low=zigzag_use_high_low,
    )

    # Get fractal detection indices (where fractal occurred, not where available)
    fh_detection_mask = ~fh_value.isna()
    fl_detection_mask = ~fl_value.isna()

    # Fractal availability: detection_bar + fractal_lookback
    fh_available_at = pd.Series(np.nan, index=df.index)
    fl_available_at = pd.Series(np.nan, index=df.index)

    fh_detection_indices = np.where(fh_detection_mask)[0]
    fl_detection_indices = np.where(fl_detection_mask)[0]

    for idx in fh_detection_indices:
        fh_available_at.iloc[idx] = idx + fractal_lookback
    for idx in fl_detection_indices:
        fl_available_at.iloc[idx] = idx + fractal_lookback

    # Zigzag availability from core function
    zz_available_at = zz_result['pivot_available_at'].replace(-1, np.nan)

    # Combined pivots: must be BOTH fractal AND zigzag
    is_combined_high = fh_detection_mask & zz_result['pivot_high']
    is_combined_low = fl_detection_mask & zz_result['pivot_low']

    # Combined availability: max of both components
    combined_high_available_at = pd.Series(np.nan, index=df.index)
    combined_low_available_at = pd.Series(np.nan, index=df.index)

    for idx in np.where(is_combined_high)[0]:
        f_avail = fh_available_at.iloc[idx]
        z_avail = zz_available_at.iloc[idx]
        if not np.isnan(f_avail) and not np.isnan(z_avail):
            combined_high_available_at.iloc[idx] = max(f_avail, z_avail)
        elif not np.isnan(f_avail):
            combined_high_available_at.iloc[idx] = f_avail
        elif not np.isnan(z_avail):
            combined_high_available_at.iloc[idx] = z_avail

    for idx in np.where(is_combined_low)[0]:
        f_avail = fl_available_at.iloc[idx]
        z_avail = zz_available_at.iloc[idx]
        if not np.isnan(f_avail) and not np.isnan(z_avail):
            combined_low_available_at.iloc[idx] = max(f_avail, z_avail)
        elif not np.isnan(f_avail):
            combined_low_available_at.iloc[idx] = f_avail
        elif not np.isnan(z_avail):
            combined_low_available_at.iloc[idx] = z_avail

    # Build result
    result = df.copy()
    result['combined_high'] = np.where(is_combined_high, fh_value, np.nan)
    result['combined_low'] = np.where(is_combined_low, fl_value, np.nan)
    result['combined_high_available_at'] = combined_high_available_at
    result['combined_low_available_at'] = combined_low_available_at

    # Component columns for debugging
    result['fractal_high'] = fh_value
    result['fractal_low'] = fl_value
    result['fractal_high_available_at'] = fh_available_at
    result['fractal_low_available_at'] = fl_available_at
    result['zigzag_high'] = zz_result['zigzag'].where(zz_result['pivot_high'])
    result['zigzag_low'] = zz_result['zigzag'].where(zz_result['pivot_low'])
    result['zigzag_available_at'] = zz_available_at

    return result

def get_available_pivots_at_bar(
    result_df: pd.DataFrame,
    bar_index: int,
    use_combined: bool = True
) -> dict[str, list[tuple[int, float]]]:
    """Get all pivots available (confirmed) at a specific bar."""
    if use_combined:
        high_col = 'combined_high'
        low_col = 'combined_low'
        high_avail_col = 'combined_high_available_at'
        low_avail_col = 'combined_low_available_at'
    else:
        high_col = 'fractal_high'
        low_col = 'fractal_low'
        high_avail_col = 'fractal_high_available_at'
        low_avail_col = 'fractal_low_available_at'

    available_highs = []
    available_lows = []

    # Vectorized filtering
    high_values = result_df[high_col].values
    low_values = result_df[low_col].values
    high_avail = result_df[high_avail_col].values
    low_avail = result_df[low_avail_col].values

    # Find highs available at bar_index
    for i in range(bar_index):
        if not np.isnan(high_values[i]) and not np.isnan(high_avail[i]) and high_avail[i] <= bar_index:
            available_highs.append((i, high_values[i]))

        if not np.isnan(low_values[i]) and not np.isnan(low_avail[i]) and low_avail[i] <= bar_index:
            available_lows.append((i, low_values[i]))

    return {'highs': available_highs, 'lows': available_lows}

def find_sr_zones(
    df: pd.DataFrame,
    lookback: int = 100,
    atr_divide: float = 10,
    min_fractals: int = 3,
    fractal_lookback: int = 5,
    expiration_bars: int | None = 200,
    use_high_low: bool = True,
    current_bar_idx: int | None = None,
) -> pd.DataFrame:
    df = df.copy()

    # Calculate fractals with availability tracking
    fh_value, fl_value, fh_avail, fl_avail = calculate_fractals(
        df,
        left=fractal_lookback,
        right=fractal_lookback,
        use_high_low=use_high_low
    )
    df['fractal_high'] = fh_value
    df['fractal_low'] = fl_value
    df['fractal_high_avail'] = fh_avail
    df['fractal_low_avail'] = fl_avail

    # Determine current detection point
    current_idx = len(df) - 1 if current_bar_idx is None else min(current_bar_idx, len(df) - 1)
    lookback_start_idx = max(0, current_idx - lookback + 1)

    # Calculate ATR for zone width
    typical = get_typical_price(df)
    dominant_cycle = ehler_dominant_cycle(typical)
    atr_adaptive = adaptive_atr_ehlers(df, adaptive_period=2 * dominant_cycle)
    zone_half_width = atr_adaptive.iloc[current_idx] / atr_divide

    # ===== VECTORIZED FRACTAL COLLECTION (matches original loop logic) =====
    window_slice = slice(lookback_start_idx, current_idx + 1)
    fh_window = df['fractal_high'].values[window_slice]
    fl_window = df['fractal_low'].values[window_slice]
    window_indices = np.arange(lookback_start_idx, current_idx + 1)

    # Masks: fractal exists AND it is confirmed/available by current_idx
    # Availability = bar_index + fractal_lookback <= current_idx
    mask_high = ~np.isnan(fh_window) & (window_indices + fractal_lookback <= current_idx)
    mask_low  = ~np.isnan(fl_window) & (window_indices + fractal_lookback <= current_idx)

    available_high_values = fh_window[mask_high]
    available_high_indices = window_indices[mask_high]
    available_low_values = fl_window[mask_low]
    available_low_indices = window_indices[mask_low]

    fractal_values = np.concatenate([available_high_values, available_low_values])
    fractal_indices = np.concatenate([available_high_indices, available_low_indices])

    if len(fractal_values) < min_fractals:
        return pd.DataFrame(columns=[
            'zone_low', 'zone_high', 'first_idx', 'last_idx',
            'fractal_count', 'bars_since_last'
        ])

    # ===== ZONE CLUSTERING ALGORITHM =====
    sort_order = np.argsort(fractal_values)
    sorted_values = fractal_values[sort_order]
    sorted_indices = fractal_indices[sort_order]

    zones = []
    n_fractals = len(sorted_values)

    for i in range(n_fractals):
        zone_center = sorted_values[i]
        zone_low = zone_center - zone_half_width
        zone_high = zone_center + zone_half_width

        left = bisect_left(sorted_values, zone_low)
        right = bisect_right(sorted_values, zone_high)
        count = right - left

        if count >= min_fractals:
            # Optimize center using mean of fractals in initial zone
            fractals_in_zone = sorted_values[left:right]
            cluster_center = np.mean(fractals_in_zone)
            optimal_zone_low = cluster_center - zone_half_width
            optimal_zone_high = cluster_center + zone_half_width

            recount_left = bisect_left(sorted_values, optimal_zone_low)
            recount_right = bisect_right(sorted_values, optimal_zone_high)
            final_count = recount_right - recount_left

            if final_count >= min_fractals:
                final_indices = sorted_indices[recount_left:recount_right]
                first_idx = int(np.min(final_indices))
                last_idx = int(np.max(final_indices))
                bars_since_last = current_idx - last_idx

                if expiration_bars is None or bars_since_last <= expiration_bars:
                    zones.append({
                        'zone_low': optimal_zone_low,
                        'zone_high': optimal_zone_high,
                        'first_idx': first_idx,
                        'last_idx': last_idx,
                        'fractal_count': final_count,
                        'bars_since_last': bars_since_last,
                        'cluster_center': cluster_center
                    })

    if not zones:
        return pd.DataFrame(columns=[
            'zone_low', 'zone_high', 'first_idx', 'last_idx',
            'fractal_count', 'bars_since_last'
        ])

    zones_df = pd.DataFrame(zones)
    zones_df = zones_df.sort_values(
        by=['fractal_count', 'cluster_center'],
        ascending=[False, True]
    ).reset_index(drop=True)

    # ===== DEDUPLICATE OVERLAPPING ZONES =====
    unique_zones = []
    for _, zone in zones_df.iterrows():
        z_low, z_high = zone['zone_low'], zone['zone_high']
        is_duplicate = False
        for existing in unique_zones:
            e_low, e_high = existing['zone_low'], existing['zone_high']
            overlap_low = max(z_low, e_low)
            overlap_high = min(z_high, e_high)
            if overlap_low < overlap_high:
                overlap_amount = overlap_high - overlap_low
                min_width = min(z_high - z_low, e_high - e_low)
                if overlap_amount / min_width > 0.5:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_zones.append(zone.to_dict())

    result_df = pd.DataFrame(unique_zones)
    if not result_df.empty:
        result_df = result_df[[
            'zone_low', 'zone_high', 'first_idx', 'last_idx',
            'fractal_count', 'bars_since_last'
        ]]

    return result_df

def _merge_overlapping_zones(
    zones_df: pd.DataFrame,
    overlap_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Merge overlapping zones using sweep-line algorithm - O(n log n).

    Zones with overlap ratio > threshold are merged, keeping the strongest.
    """
    if zones_df.empty:
        return pd.DataFrame(columns=[
            'zone_low', 'zone_high', 'first_idx', 'last_idx',
            'fractal_count', 'first_detected_idx'
        ])

    consolidated = []
    current = zones_df.iloc[0].to_dict()
    current['first_detected_idx'] = current.get('detection_idx', current.get('first_idx'))

    for idx in range(1, len(zones_df)):
        candidate = zones_df.iloc[idx]

        overlap_low = max(current['zone_low'], candidate['zone_low'])
        overlap_high = min(current['zone_high'], candidate['zone_high'])

        if overlap_low < overlap_high:
            overlap_amount = overlap_high - overlap_low
            current_width = current['zone_high'] - current['zone_low']
            candidate_width = candidate['zone_high'] - candidate['zone_low']
            overlap_ratio = overlap_amount / min(current_width, candidate_width)

            if overlap_ratio >= overlap_threshold:
                # Merge zones
                current['zone_low'] = min(current['zone_low'], candidate['zone_low'])
                current['zone_high'] = max(current['zone_high'], candidate['zone_high'])
                current['first_idx'] = min(current['first_idx'], candidate['first_idx'])
                current['last_idx'] = max(current['last_idx'], candidate['last_idx'])
                current['fractal_count'] = max(
                    current['fractal_count'], candidate['fractal_count']
                )
                current['first_detected_idx'] = min(
                    current['first_detected_idx'],
                    candidate.get('detection_idx', candidate['first_idx'])
                )
                continue

        # No merge - finalize current and start new
        consolidated.append(current.copy())
        current = candidate.to_dict()
        current['first_detected_idx'] = current.get('detection_idx', current.get('first_idx'))

    consolidated.append(current)

    result_df = pd.DataFrame(consolidated)
    result_df = result_df.drop(columns=['detection_idx', 'cluster_center', 'bars_since_last'], errors='ignore')
    return result_df

def find_sr_zones_rolling(
    df: pd.DataFrame,
    window: int = 100,
    atr_divide: float = 10,
    min_fractals: int = 3,
    fractal_lookback: int = 5,
    expiration_bars: int | None = 200,
    step: int | str = 10,
    use_high_low: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Rolling S/R zone detection with timestamp-based validity.
    
    Returns DataFrame with timestamp columns for live trading compatibility.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "find_sr_zones_rolling requires DatetimeIndex for timestamp-based zone validity. "
            f"Current index type: {type(df.index).__name__}"
        )

    df = df.copy()

    # Precompute fractals once for entire dataset
    fh_value, fl_value, fh_avail, fl_avail = calculate_fractals(
        df,
        left=fractal_lookback,
        right=fractal_lookback,
        use_high_low=use_high_low
    )
    df['fractal_high'] = fh_value
    df['fractal_low'] = fl_value
    df['fractal_high_avail'] = fh_avail
    df['fractal_low_avail'] = fl_avail

    all_zones = []
    n_bars = len(df)
    max_window = window + fractal_lookback

    # Detection point generation
    if isinstance(step, str):
        resampled_idx = df.resample(
            step,
            label='left',
            closed='left'
        ).first().index

        detection_indices = df.index.get_indexer(resampled_idx, method='ffill')
        detection_indices = detection_indices[
            (detection_indices >= window - 1) &
            (detection_indices < n_bars)
        ]
    else:
        detection_indices = np.arange(window - 1, n_bars, step)

    if show_progress:
        try:
            from tqdm import tqdm
            detection_iterator = tqdm(
                detection_indices,
                desc="Detecting S/R zones",
                unit="bar",
                total=len(detection_indices)
            )
        except ImportError:
            detection_iterator = detection_indices
    else:
        detection_iterator = detection_indices

    # Rolling window detection
    for i in detection_iterator:
        window_start_idx = max(0, i - max_window)
        df_window = df.iloc[window_start_idx:i + 1].copy()
        current_bar_window_idx = i - window_start_idx

        zones = find_sr_zones(
            df_window,
            lookback=window,
            atr_divide=atr_divide,
            min_fractals=min_fractals,
            fractal_lookback=fractal_lookback,
            expiration_bars=expiration_bars,
            use_high_low=use_high_low,
            current_bar_idx=current_bar_window_idx,
        )

        if not zones.empty:
            # Convert indices to global + add timestamps
            zones['first_idx'] = zones['first_idx'] + window_start_idx
            zones['last_idx'] = zones['last_idx'] + window_start_idx
            zones['detection_idx'] = i

            # Add timestamp columns (vectorized)
            zones['first_detected_time'] = df.index[i]
            zones['last_fractal_time'] = df.index[zones['last_idx'].astype(int).values]

            if expiration_bars is not None:
                expire_indices = (zones['last_idx'] + expiration_bars).clip(upper=n_bars - 1).astype(int)
                zones['expires_at_idx'] = expire_indices
                zones['expires_at_time'] = df.index[expire_indices.values]
            else:
                zones['expires_at_idx'] = n_bars
                zones['expires_at_time'] = pd.NaT

            # Prune expired zones
            current_time = df.index[i]
            if expiration_bars is not None:
                zones = zones[
                    zones['expires_at_time'].isna() |
                    (zones['expires_at_time'] > current_time)
                ].copy()

            if not zones.empty:
                all_zones.append(zones)

    # Consolidation
    if not all_zones:
        return pd.DataFrame(columns=[
            'zone_low', 'zone_high', 'first_idx', 'last_idx',
            'fractal_count', 'first_detected_idx', 'first_detected_time',
            'last_fractal_time', 'expires_at_idx', 'expires_at_time'
        ])

    combined_zones = pd.concat(all_zones, ignore_index=True)
    combined_zones = combined_zones.sort_values('zone_low').reset_index(drop=True)

    consolidated = _merge_overlapping_zones(combined_zones, overlap_threshold=0.7)

    # Recompute timestamps after merge (vectorized)
    if not consolidated.empty:
        consolidated['first_detected_time'] = df.index[consolidated['first_detected_idx'].astype(int).values]
        consolidated['last_fractal_time'] = df.index[consolidated['last_idx'].astype(int).values]

        if expiration_bars is not None:
            expire_indices = (consolidated['last_idx'] + expiration_bars).clip(upper=n_bars - 1).astype(int)
            consolidated['expires_at_idx'] = expire_indices
            consolidated['expires_at_time'] = df.index[expire_indices.values]
        else:
            consolidated['expires_at_idx'] = n_bars
            consolidated['expires_at_time'] = pd.NaT

    return consolidated[[
        'zone_low', 'zone_high', 'first_idx', 'last_idx',
        'fractal_count', 'first_detected_idx', 'first_detected_time',
        'last_fractal_time', 'expires_at_idx', 'expires_at_time'
    ]]

def add_in_zone_column(
    df: pd.DataFrame,
    zones: pd.DataFrame,
) -> pd.DataFrame:
    # Initialize result column as int8 for memory efficiency
    df = df.copy()
    df['in_zone'] = np.zeros(len(df), dtype=np.int8)

    # Handle empty zones edge case
    if zones.empty:
        return df

    # Extract close prices as numpy array for vectorized operations
    close = df['Close'].values
    n = len(close)

    # Compute previous close (shift by 1)
    # First element set to NaN to prevent false entry signals at bar 0
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan  # No previous candle for first bar
    prev_close[1:] = close[:-1]

    # Accumulate triggers across ALL zones (no priority/sorting needed anymore)
    has_from_below = np.zeros(n, dtype=bool)
    has_from_above = np.zeros(n, dtype=bool)

    # Process every zone (order irrelevant now)
    for _, zone in zones.iterrows():
        zone_low = zone['zone_low']
        zone_high = zone['zone_high']
        first_idx = int(zone['first_detected_idx'])
        expires_idx = int(zone['expires_at_idx'])

        # ===== TIME VALIDITY MASK =====
        time_mask = np.zeros(n, dtype=bool)
        time_mask[first_idx:expires_idx] = True

        # ===== ENTRY FROM BELOW (Direction = -1) =====
        entry_from_below = (prev_close < zone_low) & (close >= zone_low) & (close <= zone_high)
        gap_through_up = (prev_close < zone_low) & (close > zone_high)
        from_below = entry_from_below | gap_through_up

        # ===== ENTRY FROM ABOVE (Direction = 1) =====
        entry_from_above = (prev_close > zone_high) & (close >= zone_low) & (close <= zone_high)
        gap_through_down = (prev_close > zone_high) & (close < zone_low)
        from_above = entry_from_above | gap_through_down

        # Apply time validity
        valid_from_below = from_below & time_mask
        valid_from_above = from_above & time_mask

        # Accumulate triggers
        has_from_below |= valid_from_below
        has_from_above |= valid_from_above

    # Resolve final signals
    conflict_mask = has_from_below & has_from_above
    only_below_mask = has_from_below & ~has_from_above
    only_above_mask = has_from_above & ~has_from_below

    df['in_zone'] = np.select(
        condlist=[conflict_mask, only_below_mask, only_above_mask],
        choicelist=[2, -1, 1],
        default=0
    ).astype(np.int8)

    return df

def check_in_zone_at_bar(
    zones: pd.DataFrame,
    current_close: float,
    previous_close: float,
    current_bar_idx: int,
) -> int:
    if zones.empty:
        return 0

    # Filter active zones: detected before current bar, not expired
    active_mask = (
        (zones['first_detected_idx'] <= current_bar_idx) &
        (zones['expires_at_idx'] > current_bar_idx)
    )
    active_zones = zones[active_mask]

    if active_zones.empty:
        return 0

    zone_lows = active_zones['zone_low'].values
    zone_highs = active_zones['zone_high'].values

    # Vectorized trigger detection across all active zones
    entry_from_below = (
        (previous_close < zone_lows) &
        (current_close >= zone_lows) &
        (current_close <= zone_highs)
    )
    gap_through_up = (previous_close < zone_lows) & (current_close > zone_highs)
    from_below_triggers = entry_from_below | gap_through_up

    entry_from_above = (
        (previous_close > zone_highs) &
        (current_close >= zone_lows) &
        (current_close <= zone_highs)
    )
    gap_through_down = (previous_close > zone_highs) & (current_close < zone_lows)
    from_above_triggers = entry_from_above | gap_through_down

    has_from_below = np.any(from_below_triggers)
    has_from_above = np.any(from_above_triggers)

    # Resolve conflicts
    if has_from_below and has_from_above:
        return 2
    elif has_from_below:
        return -1
    elif has_from_above:
        return 1
    else:
        return 0
