from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from sambo.plot import plot_objective
from scipy.signal import argrelextrema

from blackwood.visualization.style import DEFAULT_STYLE

if TYPE_CHECKING:
    from numpy.typing import NDArray


def visualize_chart(
    price_df: pd.DataFrame,
    bbands_df: pd.DataFrame | list[pd.DataFrame] | None = None,
    bbands_names: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    days_to_plot: int = 10,
    signal_col: str | None = None,
    overlay_cols: str | Sequence[str] | None = None,
    bottom_col: str | Sequence[str | tuple] | None = None,
    sr_zones: pd.DataFrame | None = None,
    show_fractals: bool = True,
    show_zigzag_pivots: bool = False,
    session_hours: tuple[float | None, float | None] = (None, None),
    resample: str | None = None,
) -> go.Figure:
    """
    Create a candlestick chart with up to 2 Bollinger Bands, optional signals, overlays, and bottom panels.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data with DatetimeIndex and columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
        Optional columns:
        - 'fractal_high', 'fractal_low' for fractal visualization
        - 'pivot_high', 'pivot_low' for zigzag pivot visualization
    bbands_df : pd.DataFrame or list[pd.DataFrame]
        Single Bollinger Bands dataframe OR list of up to 2 Bollinger Bands dataframes.
        Each dataframe should have same index as price_df and columns:
        'middle_band', 'upper_band', 'lower_band'.
    bbands_names : list[str], optional
        Names for each Bollinger Bands (e.g., ["BB 20", "BB 50"]).
        If None, defaults to ["BB 1", "BB 2"].
    start_date, end_date : str, optional
        Date range to plot.
    days_to_plot : int
        Days from end to plot if no date range specified.
    signal_col : str, optional
        Column for buy/sell signals (1=buy, -1=sell).
    overlay_cols : str or list[str], optional
        Columns to plot as overlays on the main price chart (e.g., moving averages, VWAP).
        Max 5 overlays for visual clarity.
    bottom_col : str or list[str or tuple], optional
        Additional indicators for bottom panels. Can be:
        - Single string: 'rsi'
        - List of strings: ['rsi', 'macd'] - each gets own row
        - List with tuples: [('stoch_k', 'stoch_d'), 'rsi'] - tuple items share a row
        Max 5 bottom panels total.
    sr_zones : pd.DataFrame, optional
        Support/Resistance zones with columns:
        ['zone_low', 'zone_high', 'first_idx', 'last_idx', 'fractal_count']
        Zones are drawn as rectangles from first_idx to last_idx.
    show_fractals : bool, default=True
        Whether to plot fractal_high and fractal_low as scatter points.
    show_zigzag_pivots : bool, default=False
        Whether to plot pivot_high and pivot_low from zigzag indicator as scatter points.
        Pivots represent significant swing highs/lows after volatility-adjusted filtering.
    resample : str, optional
        Resampling frequency (e.g., '5min', '1H', '1D'). If provided, resamples all data
        using OHLC aggregation before plotting. Reduces memory and improves rendering speed.

    """
    style = DEFAULT_STYLE

    # 0. Normalize bbands_df to list (None → empty list, hide bands)
    if bbands_df is None:
        bbands_list = []
    elif isinstance(bbands_df, pd.DataFrame):
        bbands_list = [bbands_df]
    else:
        bbands_list = list(bbands_df)[:2]

    # 0.5. Store original index mapping before resampling (for SR zones)
    original_idx_to_timestamp = dict(enumerate(price_df.index))

    # Resample data if requested (before any filtering)
    if resample is not None:
        # Resample price_df with OHLC aggregation
        ohlc_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

        # Include optional columns if they exist
        for col in ["fractal_high", "fractal_low", "pivot_high", "pivot_low", "zigzag", "atr_adapt"]:
            if col in price_df.columns:
                if col in ["pivot_high", "pivot_low"]:
                    ohlc_dict[col] = "max"
                elif col == "zigzag":
                    ohlc_dict[col] = "last"
                else:
                    ohlc_dict[col] = "last"

        # Add signal_col if specified
        if signal_col is not None and signal_col in price_df.columns:
            ohlc_dict[signal_col] = "last"

        # Add overlay_cols
        if overlay_cols is not None:
            cols_to_add = [overlay_cols] if isinstance(overlay_cols, str) else overlay_cols
            for col in cols_to_add:
                if col in price_df.columns and col not in ohlc_dict:
                    ohlc_dict[col] = "last"

        # Add bottom_col columns
        if bottom_col is not None:
            if isinstance(bottom_col, str):
                bottom_cols_flat = [bottom_col]
            else:
                bottom_cols_flat = []
                for item in bottom_col:
                    if isinstance(item, str):
                        bottom_cols_flat.append(item)
                    elif isinstance(item, (tuple, list)):
                        bottom_cols_flat.extend(item)

            for col in bottom_cols_flat:
                if col in price_df.columns and col not in ohlc_dict:
                    ohlc_dict[col] = "last"

        price_df = price_df.resample(resample).agg(ohlc_dict).dropna(subset=["Close"])  # pyright: ignore[]

        # Resample bbands_df list
        bbands_list = [
            bb.resample(resample).agg({"middle_band": "last", "upper_band": "last", "lower_band": "last"}).dropna()
            for bb in bbands_list
        ]

    n_bbands = len(bbands_list)

    # Set default names if not provided
    if bbands_names is None:
        bbands_names = [f"BB {i + 1}" for i in range(n_bbands)]
    else:
        bbands_names = list(bbands_names)[:n_bbands]
        while len(bbands_names) < n_bbands:
            bbands_names.append(f"BB {len(bbands_names) + 1}")

    # 1. Normalize overlay_cols
    if overlay_cols is None:
        overlay_cols_list: list[str] = []
    elif isinstance(overlay_cols, str):
        overlay_cols_list = [overlay_cols]
    else:
        overlay_cols_list = [c for c in overlay_cols if isinstance(c, str)]

    overlay_cols_list = overlay_cols_list[:5]

    # 2. Normalize bottom_col - parse groups (tuples) and singles
    bottom_groups: list[tuple[str, ...]] = []
    if bottom_col is None:
        pass
    elif isinstance(bottom_col, str):
        bottom_groups = [(bottom_col,)]
    else:
        for item in bottom_col:
            if isinstance(item, str):
                bottom_groups.append((item,))
            elif isinstance(item, (tuple, list)):
                filtered = tuple(c for c in item if isinstance(c, str))
                if filtered:
                    bottom_groups.append(filtered)

    bottom_groups = bottom_groups[:5]
    n_bottom = len(bottom_groups)

    # 3. Filter data for plotting window
    if start_date and end_date:
        plot_data = price_df.loc[start_date:end_date].copy()
        bb_plot_data_list = [bb.loc[start_date:end_date].copy() for bb in bbands_list]
    elif days_to_plot is not None and not price_df.empty:
        last_timestamp = price_df.index.max()
        start_timestamp = last_timestamp - pd.Timedelta(days=days_to_plot)

        plot_data = price_df[price_df.index >= start_timestamp].copy()
        bb_plot_data_list = [bb[bb.index >= start_timestamp].copy() for bb in bbands_list]
    else:
        plot_data = price_df.copy()
        bb_plot_data_list = [bb.copy() for bb in bbands_list]

    # Align indices - find common index across all dataframes
    common_index = plot_data.index
    for bb_data in bb_plot_data_list:
        common_index = common_index.intersection(bb_data.index)

    plot_data = plot_data.loc[common_index]
    bb_plot_data_list = [bb.loc[common_index] for bb in bb_plot_data_list]

    # Validate and filter columns that exist
    overlay_cols_list = [c for c in overlay_cols_list if c in plot_data.columns]

    # Filter bottom groups to only include existing columns
    valid_bottom_groups = []
    for group in bottom_groups:
        valid_cols = tuple(c for c in group if c in plot_data.columns)
        if valid_cols:
            valid_bottom_groups.append(valid_cols)

    bottom_groups = valid_bottom_groups
    n_bottom = len(bottom_groups)

    # 4. Layout setup
    if n_bottom == 0:
        n_rows = 1
        row_heights = [1.0]
    else:
        main_height = 0.7
        remaining = 0.3 / n_bottom
        row_heights = [main_height] + [remaining] * n_bottom
        n_rows = 1 + n_bottom

    # 5. Create subplot titles
    subplot_titles = ["Price & Bollinger Bands"]
    for group in bottom_groups:
        if len(group) == 1:
            subplot_titles.append(group[0])
        else:
            subplot_titles.append(" & ".join(group))

    # 6. Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # 7. Candlestick (main price action)
    fig.add_trace(
        go.Candlestick(
            x=plot_data.index,
            open=plot_data["Open"],
            high=plot_data["High"],
            low=plot_data["Low"],
            close=plot_data["Close"],
            name="Price",
            increasing_line_color=style.accent3,
            decreasing_line_color=style.accent4,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # 8. Multiple Bollinger Bands with distinct colors
    band_colors = [
        {"middle": "rgba(0, 123, 255, 0.8)", "bands": "rgba(0, 123, 255, 0.6)", "fill": "rgba(0, 123, 255, 0.15)"},
        {"middle": "rgba(255, 99, 71, 0.8)", "bands": "rgba(255, 99, 71, 0.6)", "fill": "rgba(255, 99, 71, 0.15)"},
    ]

    for idx, (bb_data, bb_name) in enumerate(zip(bb_plot_data_list, bbands_names, strict=True)):
        colors = band_colors[idx % len(band_colors)]

        # Middle band
        fig.add_trace(
            go.Scatter(
                x=bb_data.index,
                y=bb_data["middle_band"],
                mode="lines",
                line=dict(width=1.5, color=colors["middle"], dash="dash"),
                name=f"{bb_name} Middle",
                hovertemplate=f"{bb_name} Middle: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Upper band
        fig.add_trace(
            go.Scatter(
                x=bb_data.index,
                y=bb_data["upper_band"],
                fill=None,
                mode="lines",
                line=dict(width=1, color=colors["bands"]),
                name=f"{bb_name} Upper",
                hovertemplate=f"{bb_name} Upper: %{{y:.2f}}<extra></extra>",
                legendgroup=bb_name,
            ),
            row=1,
            col=1,
        )

        # Lower band with fill
        fig.add_trace(
            go.Scatter(
                x=bb_data.index,
                y=bb_data["lower_band"],
                fill="tonexty",
                fillcolor=colors["fill"],
                mode="lines",
                line=dict(width=1, color=colors["bands"]),
                name=f"{bb_name} Lower",
                hovertemplate=f"{bb_name} Lower: %{{y:.2f}}<extra></extra>",
                showlegend=False,
                legendgroup=bb_name,
            ),
            row=1,
            col=1,
        )

    # 8.5. Fractals visualization (Williams Fractals)
    if show_fractals:
        has_fractal_high = "fractal_high" in plot_data.columns
        has_fractal_low = "fractal_low" in plot_data.columns

        if has_fractal_high:
            fractal_highs = plot_data["fractal_high"].dropna()
            if not fractal_highs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fractal_highs.index,
                        y=fractal_highs.values,
                        mode="markers",
                        marker=dict(
                            symbol="circle", size=10, color="rgba(255, 0, 0, 0.7)", line=dict(width=1, color="white")
                        ),
                        name="Fractal High",
                        hovertemplate="Fractal High: %{y:.5f}<extra></extra>",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        if has_fractal_low:
            fractal_lows = plot_data["fractal_low"].dropna()
            if not fractal_lows.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fractal_lows.index,
                        y=fractal_lows.values,
                        mode="markers",
                        marker=dict(
                            symbol="circle", size=10, color="rgba(0, 255, 0, 0.7)", line=dict(width=1, color="white")
                        ),
                        name="Fractal Low",
                        hovertemplate="Fractal Low: %{y:.5f}<extra></extra>",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

    # 8.6. ZigZag Pivots visualization (Adaptive ATR-based pivots)
    if show_zigzag_pivots:
        has_pivot_high = "pivot_high" in plot_data.columns
        has_pivot_low = "pivot_low" in plot_data.columns

        if has_pivot_high:
            if "zigzag" in plot_data.columns:
                pivot_high_mask = plot_data["pivot_high"]
                pivot_highs_data = plot_data.loc[pivot_high_mask, "zigzag"].dropna()
            else:
                pivot_high_mask = plot_data["pivot_high"]
                pivot_highs_data = plot_data.loc[pivot_high_mask, "High"]

            if not pivot_highs_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_highs_data.index,
                        y=pivot_highs_data.values,
                        mode="markers",
                        marker=dict(
                            symbol="diamond",
                            size=12,
                            color="rgba(255, 140, 0, 0.8)",
                            line=dict(width=2, color="rgba(139, 0, 0, 0.9)"),
                        ),
                        name="ZigZag Pivot High",
                        hovertemplate="Pivot High: %{y:.5f}<extra></extra>",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        if has_pivot_low:
            if "zigzag" in plot_data.columns:
                pivot_low_mask = plot_data["pivot_low"]
                pivot_lows_data = plot_data.loc[pivot_low_mask, "zigzag"].dropna()
            else:
                pivot_low_mask = plot_data["pivot_low"]
                pivot_lows_data = plot_data.loc[pivot_low_mask, "Low"]

            if not pivot_lows_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_lows_data.index,
                        y=pivot_lows_data.values,
                        mode="markers",
                        marker=dict(
                            symbol="diamond",
                            size=12,
                            color="rgba(0, 191, 255, 0.8)",
                            line=dict(width=2, color="rgba(0, 0, 139, 0.9)"),
                        ),
                        name="ZigZag Pivot Low",
                        hovertemplate="Pivot Low: %{y:.5f}<extra></extra>",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

    # 9. Overlay indicators on price chart
    overlay_colors = [
        "rgba(255, 159, 64, 0.9)",
        "rgba(153, 102, 255, 0.9)",
        "rgba(255, 99, 132, 0.9)",
        "rgba(75, 192, 192, 0.9)",
        "rgba(54, 162, 235, 0.9)",
    ]

    for i, col_name in enumerate(overlay_cols_list):
        color = overlay_colors[i % len(overlay_colors)]

        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[col_name],
                mode="lines",
                name=col_name,
                line=dict(width=2, color=color),
                hovertemplate=f"%{{y:.2f}}<extra>{col_name}</extra>",
                opacity=0.8,
            ),
            row=1,
            col=1,
        )

    # 10. Optional signals
    if signal_col is not None and signal_col in plot_data.columns:
        buy_signals = plot_data[plot_data[signal_col] == 1]
        sell_signals = plot_data[plot_data[signal_col] == -1]

        if "atr_adapt" in plot_data.columns:
            buy_offset = buy_signals["atr_adapt"]
            sell_offset = sell_signals["atr_adapt"]
        else:
            buy_offset = buy_signals["Close"] * 0.001
            sell_offset = sell_signals["Close"] * 0.001

        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals["Close"] - buy_offset,
                    mode="markers",
                    marker=dict(symbol="triangle-up", color=style.accent3, size=12, line=dict(width=1, color="white")),
                    name="Buy",
                    hovertemplate="Buy @ %{y:.5f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals["Close"] + sell_offset,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down", color=style.accent4, size=12, line=dict(width=1, color="white")
                    ),
                    name="Sell",
                    hovertemplate="Sell @ %{y:.5f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # 11. Bottom panels for oscillators/indicators
    panel_colors = [
        "rgba(75, 192, 192, 0.9)",
        "rgba(255, 99, 132, 0.9)",
        "rgba(255, 159, 64, 0.9)",
        "rgba(153, 102, 255, 0.9)",
        "rgba(54, 162, 235, 0.9)",
    ]

    for group_idx, group in enumerate(bottom_groups, start=2):
        for color_idx, col_name in enumerate(group):
            color = panel_colors[color_idx % len(panel_colors)]

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(width=1.5, color=color),
                    hovertemplate=f"%{{y:.4f}}<extra>{col_name}</extra>",
                ),
                row=group_idx,
                col=1,
            )

        if any(
            any(keyword in col.lower() for keyword in ["rsi", "macd", "momentum", "oscillator", "stoch"])
            for col in group
        ):
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=group_idx, col=1)

        yaxis_label = group[0] if len(group) == 1 else " & ".join(group)
        fig.update_yaxes(title_text=yaxis_label, row=group_idx, col=1)

    # 12. Support/Resistance Zones as rectangles
    if sr_zones is not None and not sr_zones.empty:
        zone_colors = [
            "rgba(255, 193, 7, 0.15)",
            "rgba(156, 39, 176, 0.15)",
        ]

        zone_border_colors = [
            "rgba(255, 193, 7, 0.6)",
            "rgba(156, 39, 176, 0.6)",
        ]

        sr_zones["first_detected_idx"] = sr_zones["first_detected_idx"].astype(int)
        sr_zones["expires_at_idx"] = sr_zones["expires_at_idx"].astype(int)
        sr_zones["fractal_count"] = sr_zones["fractal_count"].astype(int)

        for counter, (_, zone_row) in enumerate(sr_zones.iterrows()):
            zone_low = zone_row["zone_low"]
            zone_high = zone_row["zone_high"]

            first_detected_idx = zone_row["first_detected_idx"]
            expires_at_idx = zone_row["expires_at_idx"]
            fractal_count = zone_row["fractal_count"]

            x0 = original_idx_to_timestamp[first_detected_idx]
            x1 = original_idx_to_timestamp[expires_at_idx]

            # Pyright loves this because 'counter' is explicitly an int
            color_idx = counter % len(zone_colors)
            fill_color = zone_colors[color_idx]
            border_color = zone_border_colors[color_idx]

            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=zone_low,
                y1=zone_high,
                fillcolor=fill_color,
                line=dict(color=border_color, width=1.5, dash="dot"),
                layer="below",
                row=1,
                col=1,
            )

            mid_x = x0 + (x1 - x0) / 2
            mid_y = (zone_low + zone_high) / 2

            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"{fractal_count}",
                showarrow=False,
                font=dict(size=10, color=border_color.replace("0.6", "0.9")),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor=border_color,
                borderwidth=1,
                borderpad=2,
                row=1,
                col=1,
                xref="x",
                yref="y",
            )

    # 13. Apply style and finalize
    fig = style.apply(fig)
    fig.update_layout(
        title="Price with Multiple Bollinger Bands & Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        height=700 + 150 * n_bottom,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if session_hours[0] is not None and session_hours[1] is not None:
        fig.update_xaxes(rangebreaks=[dict(bounds=session_hours, pattern="hour")])

    fig.update_xaxes(matches="x")
    fig.update_yaxes(fixedrange=False)

    return fig


def visualize_chart_date_range(
    price_df: pd.DataFrame,
    bbands_df: pd.DataFrame | list[pd.DataFrame] | None = None,
    bbands_names: list[str] | None = None,
    *,
    start_date: str,
    end_date: str,
    signal_col: str | None = None,
    overlay_cols: str | Sequence[str] | None = None,
    bottom_col: str | Sequence[str | tuple] | None = None,
    session_hours: tuple[float | None, float | None] = (None, None),
) -> go.Figure:
    """
    Candlestick chart with up to two Bollinger Band sets, optional signals, overlays, and
    bottom panels, plotted between start_date and end_date (inclusive under .loc).
    """
    style = DEFAULT_STYLE

    # 0) Normalize Bollinger Bands input
    if bbands_df is None:
        bbands_list: list[pd.DataFrame] = []
    elif isinstance(bbands_df, pd.DataFrame):
        bbands_list = [bbands_df]
    else:
        bbands_list = list(bbands_df)[:2]

    n_bbands = len(bbands_list)

    # Default or trimmed Bollinger Band names
    if bbands_names is None:
        bbands_names = [f"BB {i + 1}" for i in range(n_bbands)]
    else:
        bbands_names = list(bbands_names)[:n_bbands]
        while len(bbands_names) < n_bbands:
            bbands_names.append(f"BB {len(bbands_names) + 1}")

    # 1) Normalize overlay column names
    if overlay_cols is None:
        overlay_cols_list: list[str] = []
    elif isinstance(overlay_cols, str):
        overlay_cols_list = [overlay_cols]
    else:
        overlay_cols_list = [c for c in overlay_cols if isinstance(c, str)]
    overlay_cols_list = overlay_cols_list[:5]

    # 2) Normalize bottom column groups
    bottom_groups: list[tuple[str, ...]] = []
    if isinstance(bottom_col, str):
        bottom_groups = [(bottom_col,)]
    elif bottom_col is not None:
        for item in bottom_col:
            if isinstance(item, str):
                bottom_groups.append((item,))
            elif isinstance(item, (tuple, list)):
                filtered = tuple(c for c in item if isinstance(c, str))
                if filtered:
                    bottom_groups.append(filtered)

    bottom_groups = bottom_groups[:5]
    n_bottom = len(bottom_groups)

    # 3) Filter data by explicit date range (inclusive under .loc)
    plot_data = price_df.loc[start_date:end_date]
    bb_plot_data_list = [bb.loc[start_date:end_date] for bb in bbands_list]

    # Align indices (intersection across price and all bands)
    common_index = plot_data.index
    for bb_data in bb_plot_data_list:
        common_index = common_index.intersection(bb_data.index)

    plot_data = plot_data.loc[common_index]
    bb_plot_data_list = [bb.loc[common_index] for bb in bb_plot_data_list]

    # Validate overlay columns against actual data
    overlay_cols_list = [c for c in overlay_cols_list if c in plot_data.columns]

    # Validate bottom groups against actual data
    valid_bottom_groups: list[tuple[str, ...]] = []
    for group in bottom_groups:
        valid_cols = tuple(c for c in group if c in plot_data.columns)
        if valid_cols:
            valid_bottom_groups.append(valid_cols)
    bottom_groups = valid_bottom_groups
    n_bottom = len(bottom_groups)

    # 4) Layout setup
    if n_bottom == 0:
        n_rows = 1
        row_heights = [1.0]
    else:
        main_height = 0.7
        remaining = 0.3 / n_bottom
        row_heights = [main_height] + [remaining] * n_bottom
        n_rows = 1 + n_bottom

    # 5) Subplot titles
    subplot_titles = ["Price & Bollinger Bands"]
    for group in bottom_groups:
        subplot_titles.append(group[0] if len(group) == 1 else " & ".join(group))

    # 6) Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # 7) Candlestick
    fig.add_trace(
        go.Candlestick(
            x=plot_data.index,
            open=plot_data["Open"],
            high=plot_data["High"],
            low=plot_data["Low"],
            close=plot_data["Close"],
            name="Price",
            increasing_line_color=style.accent3,
            decreasing_line_color=style.accent4,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # 8) Bollinger Bands
    band_colors = [
        {
            "middle": "rgba(0, 123, 255, 0.8)",
            "bands": "rgba(0, 123, 255, 0.6)",
            "fill": "rgba(0, 123, 255, 0.15)",
        },
        {
            "middle": "rgba(255, 99, 71, 0.8)",
            "bands": "rgba(255, 99, 71, 0.6)",
            "fill": "rgba(255, 99, 71, 0.15)",
        },
    ]

    for idx, (bb_data, bb_name) in enumerate(zip(bb_plot_data_list, bbands_names, strict=True)):
        colors = band_colors[idx % len(band_colors)]

        fig.add_trace(
            go.Scatter(
                x=bb_data.index,
                y=bb_data["middle_band"],
                mode="lines",
                line=dict(width=1.5, color=colors["middle"], dash="dash"),
                name=f"{bb_name} Middle",
                hovertemplate=f"{bb_name} Middle: %{{y:.2f}}<extra></extra>",
                legendgroup=bb_name,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=bb_data.index,
                y=bb_data["upper_band"],
                fill=None,
                mode="lines",
                line=dict(width=1, color=colors["bands"]),
                name=f"{bb_name} Upper",
                hovertemplate=f"{bb_name} Upper: %{{y:.2f}}<extra></extra>",
                legendgroup=bb_name,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=bb_data.index,
                y=bb_data["lower_band"],
                fill="tonexty",
                fillcolor=colors["fill"],
                mode="lines",
                line=dict(width=1, color=colors["bands"]),
                name=f"{bb_name} Lower",
                hovertemplate=f"{bb_name} Lower: %{{y:.2f}}<extra></extra>",
                showlegend=False,
                legendgroup=bb_name,
            ),
            row=1,
            col=1,
        )

    # 9) Overlay indicators
    overlay_colors = [
        "rgba(255, 159, 64, 0.9)",
        "rgba(153, 102, 255, 0.9)",
        "rgba(255, 99, 132, 0.9)",
        "rgba(75, 192, 192, 0.9)",
        "rgba(54, 162, 235, 0.9)",
    ]

    for i, col_name in enumerate(overlay_cols_list):
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[col_name],
                mode="lines",
                name=col_name,
                line=dict(width=2, color=overlay_colors[i % len(overlay_colors)]),
                hovertemplate=f"%{{y:.2f}}<extra>{col_name}</extra>",
                opacity=0.8,
            ),
            row=1,
            col=1,
        )

    # 10) Optional signals (vectorized masking, no intermediate DataFrames)
    if signal_col is not None and signal_col in plot_data.columns:
        signal_series = plot_data[signal_col]
        close = plot_data["Close"]

        buy_mask = signal_series == 1
        sell_mask = signal_series == -1

        if "atr_adapt" in plot_data.columns:
            atr = plot_data["atr_adapt"]
            buy_offset = atr[buy_mask]
            sell_offset = atr[sell_mask]
        else:
            buy_offset = close[buy_mask] * 0.001
            sell_offset = close[sell_mask] * 0.001

        if buy_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index[buy_mask],
                    y=close[buy_mask] - buy_offset,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        color=style.accent3,
                        size=12,
                        line=dict(width=1, color="white"),
                    ),
                    name="Buy",
                    hovertemplate="Buy @ %{y:.5f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if sell_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index[sell_mask],
                    y=close[sell_mask] + sell_offset,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        color=style.accent4,
                        size=12,
                        line=dict(width=1, color="white"),
                    ),
                    name="Sell",
                    hovertemplate="Sell @ %{y:.5f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # 11) Bottom panels
    panel_colors = [
        "rgba(75, 192, 192, 0.9)",
        "rgba(255, 99, 132, 0.9)",
        "rgba(255, 159, 64, 0.9)",
        "rgba(153, 102, 255, 0.9)",
        "rgba(54, 162, 235, 0.9)",
    ]

    for group_idx, group in enumerate(bottom_groups, start=2):
        for color_idx, col_name in enumerate(group):
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(
                        width=1.5,
                        color=panel_colors[color_idx % len(panel_colors)],
                    ),
                    hovertemplate=f"%{{y:.4f}}<extra>{col_name}</extra>",
                ),
                row=group_idx,
                col=1,
            )

        # Zero-line for oscillators / momentum-style indicators
        if any(
            any(keyword in col.lower() for keyword in ["rsi", "macd", "momentum", "oscillator", "stoch"])
            for col in group
        ):
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=group_idx,
                col=1,
            )

        fig.update_yaxes(title_text=(" & ".join(group)), row=group_idx, col=1)

    # 12) Apply style + layout
    fig = style.apply(fig)
    fig.update_layout(
        title=f"Price with Multiple Bollinger Bands & Indicators ({start_date} → {end_date})",
        xaxis_title="Date",
        yaxis_title="Price",
        height=700 + 200 * n_bottom,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Session rangebreaks
    if session_hours[0] is not None and session_hours[1] is not None:
        fig.update_xaxes(rangebreaks=[dict(bounds=session_hours, pattern="hour")])

    fig.update_xaxes(matches="x")
    fig.update_yaxes(fixedrange=False)

    return fig


def plot_price_with_session_ranges(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    days_to_plot: int = 10,
    show_fractals: bool = False,
    show_prev_day_range: bool = True,
) -> go.Figure:
    style = DEFAULT_STYLE
    if start_date and end_date:
        plot_data = df.loc[start_date:end_date].copy()
    else:
        # Use normalized index to get unique trading days efficiently
        normalized_index = cast("pd.DatetimeIndex", df.index).normalize()
        unique_dates = pd.Index(normalized_index.unique())
        recent_dates = unique_dates[-days_to_plot:]
        if len(recent_dates) == 0:
            plot_data = df.iloc[0:0].copy()
        else:
            start_day = recent_dates[0]
            plot_data = df.loc[normalized_index >= start_day].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=plot_data.index,
            open=plot_data["Open"],
            high=plot_data["High"],
            low=plot_data["Low"],
            close=plot_data["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # ---- Previous day range boxes ----
    if show_prev_day_range and {"PrevDayHigh", "PrevDayLow"}.issubset(plot_data.columns):
        prev_high = plot_data["PrevDayHigh"]
        prev_low = plot_data["PrevDayLow"]

        plot_data["prev_day_group"] = ((prev_high != prev_high.shift()) | (prev_low != prev_low.shift())).cumsum()

        for _, group_data in plot_data[prev_high.notna()].groupby("prev_day_group"):
            if group_data.empty:
                continue

            prev_day_high = group_data["PrevDayHigh"].iloc[0]
            prev_day_low = group_data["PrevDayLow"].iloc[0]

            x0, x1 = group_data.index[0], group_data.index[-1]

            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=prev_day_low,
                y1=prev_day_high,
                fillcolor="rgba(255, 152, 0, 0.12)",
                line=dict(color="rgba(230, 81, 0, 0.5)", width=1, dash="dash"),
                layer="below",
            )

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[prev_day_high, prev_day_high],
                    mode="lines",
                    line=dict(color="#E65100", width=1.5, dash="dot"),
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[prev_day_low, prev_day_low],
                    mode="lines",
                    line=dict(color="#E65100", width=1.5, dash="dot"),
                    showlegend=False,
                )
            )

    # ---- Current session range boxes ----
    range_high = plot_data["RangeHigh"]
    range_low = plot_data["RangeLow"]

    plot_data["range_group"] = ((range_high != range_high.shift()) | (range_low != range_low.shift())).cumsum()

    for _, group_data in plot_data[range_high.notna()].groupby("range_group"):
        if group_data.empty:
            continue

        current_range_high = group_data["RangeHigh"].iloc[0]
        current_range_low = group_data["RangeLow"].iloc[0]

        x0, x1 = group_data.index[0], group_data.index[-1]

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=current_range_low,
            y1=current_range_high,
            fillcolor="rgba(135, 206, 250, 0.2)",
            line=dict(color="rgba(70, 130, 180, 0.6)", width=1.5),
            layer="below",
        )

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[current_range_high, current_range_high],
                mode="lines",
                line=dict(color="#1976D2", width=2, dash="dash"),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[current_range_low, current_range_low],
                mode="lines",
                line=dict(color="#1976D2", width=2, dash="dash"),
                showlegend=False,
            )
        )

    # ---- Optional fractal overlays ----
    if show_fractals and {"Fractal_high", "Fractal_low"}.issubset(plot_data.columns):
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data["Fractal_high"],
                mode="markers",
                name="Fractal High",
                marker=dict(
                    symbol="triangle-down",
                    color="#7B1FA2",
                    size=9,
                    line=dict(color="white", width=1),
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data["Fractal_low"],
                mode="markers",
                name="Fractal Low",
                marker=dict(
                    symbol="triangle-up",
                    color="#FBC02D",
                    size=9,
                    line=dict(color="white", width=1),
                ),
            )
        )

    # ---- Base layout (semantic, not stylistic) ----
    fig.update_layout(
        title="Price Action with Session Range (Blue) and Previous Day Range (Orange)",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=700,
        showlegend=True,
    )

    style.apply(fig)
    return fig


class BacktestVisualizer:
    def __init__(
        self,
        equity_dict: dict[str, pd.Series],
        risk_lookback: int = 252,
        extrema_order: int = 50,  # Typical: 20-100; higher = smoother/fewer points
        create_portfolios: bool = False,
    ) -> None:
        """
        Initialize BacktestVisualizer with equity curves.

        Parameters
        ----------
        equity_dict : Dict[str, pd.Series]
            Dictionary mapping strategy names to equity curves.
        risk_lookback : int, default=252
            Lookback window for volatility calculations in equal-risk portfolio.
        extrema_order : int, default=50
            Order parameter for scipy.signal.argrelextrema.
            Higher values = fewer local extrema detected (smoother interpolation).
            Typical range: 20-100 depending on data frequency and volatility.
        create_portfolios : bool, default=False
            If True, automatically add equal-weight and equal-risk portfolios.

        """
        self.equity_dict = equity_dict
        self.risk_lookback = risk_lookback
        self.extrema_order = extrema_order
        self.style = DEFAULT_STYLE
        self.create_portfolios = create_portfolios

        if self.create_portfolios:
            self._add_equal_weight_portfolio()
            # self._add_equal_risk_portfolio()

        self.colors = [
            self.style.accent1,
            self.style.accent2,
            self.style.accent3,
            self.style.accent4,
            self.style.accent5,
            self.style.accent6,
        ]

    def _get_color(self, idx: int) -> str:
        return self.colors[idx % len(self.colors)]

    def apply_style(self, fig: go.Figure) -> go.Figure:
        return self.style.apply(fig)

    def _prepare_strategy_data(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        strategy_equities = {
            name: equity
            for name, equity in self.equity_dict.items()
            if "Portfolio" not in name and "Leveraged" not in name
        }
        if len(strategy_equities) < 2:
            return None

        equity_df = pd.DataFrame(strategy_equities).ffill()
        returns_df = equity_df.pct_change()
        return equity_df, returns_df

    def _finalize_portfolio(
        self,
        portfolio_returns: pd.Series,
        equity_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        name: str,
    ) -> None:
        first_valid_idx = returns_df.notna().any(axis=1).idxmax()
        initial_capital = equity_df.loc[first_valid_idx].mean(skipna=True)
        portfolio_equity = initial_capital * (1 + portfolio_returns).cumprod()
        if not portfolio_equity.empty:
            portfolio_equity.iloc[0] = initial_capital
        self.equity_dict[name] = portfolio_equity

    def _add_equal_weight_portfolio(self) -> None:
        data = self._prepare_strategy_data()
        if data is None:
            return
        equity_df, returns_df = data
        portfolio_returns = returns_df.mean(axis=1, skipna=True)
        self._finalize_portfolio(portfolio_returns, equity_df, returns_df, "Equal-Weight Portfolio")

    def _add_equal_risk_portfolio(self) -> None:
        data = self._prepare_strategy_data()
        if data is None:
            return
        equity_df, returns_df = data

        volatility_df = returns_df.ewm(span=self.risk_lookback, min_periods=self.risk_lookback).std()
        inv_vol = 1 / volatility_df
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        weights_df = inv_vol.div(inv_vol.sum(axis=1, skipna=True), axis=0)

        available_count = returns_df.notna().sum(axis=1)
        equal_weight = 1 / available_count
        equal_weight = equal_weight.replace([np.inf, -np.inf], 0)
        weights_df = weights_df.fillna(equal_weight)

        portfolio_returns = (weights_df * returns_df).sum(axis=1, skipna=True)
        self._finalize_portfolio(portfolio_returns, equity_df, returns_df, "Equal-Risk Portfolio")

    def _calculate_drawdown(self, equity: pd.Series) -> tuple[pd.Series, dict[str, int]]:
        running_max: pd.Series = equity.cummax()
        drawdown: pd.Series = (equity - running_max) / running_max * 100
        drawdown_depth: pd.Series = running_max - equity

        if len(equity) == 0 or drawdown_depth.max() == 0:
            metadata: dict[str, int] = {"dd_start": 0, "dd_end": 0, "dd_peak": 0}
            return drawdown, metadata

        # Work in plain numpy/positional space to avoid the get_loc()
        # `int | slice | np_1darray_bool` union poisoning every downstream type.
        equity_values: NDArray[np.float64] = equity.to_numpy(dtype=float)
        drawdown_depth_values: NDArray[np.float64] = drawdown_depth.to_numpy(dtype=float)
        n = len(equity_values)

        dd_peak_idx: int = int(np.argmax(drawdown_depth_values))
        dd_start_idx: int = int(np.argmax(equity_values[: dd_peak_idx + 1]))
        equity_at_start: float = float(equity_values[dd_start_idx])

        recovery_mask = equity_values[dd_peak_idx:] >= equity_at_start
        if recovery_mask.any():
            recovery_idx: int = dd_peak_idx + int(np.argmax(recovery_mask))
            if recovery_idx > dd_peak_idx and equity_values[recovery_idx - 1] < equity_at_start:
                prev_value = float(equity_values[recovery_idx - 1])
                curr_value = float(equity_values[recovery_idx])
                dd_end_idx = round(
                    float(np.interp(equity_at_start, [prev_value, curr_value], [recovery_idx - 1, recovery_idx]))
                )

            else:
                dd_end_idx = recovery_idx
        else:
            dd_end_idx = n - 1

        metadata = {
            "dd_start": dd_start_idx,
            "dd_end": dd_end_idx,
            "dd_peak": dd_peak_idx,
        }
        return drawdown, metadata

    def _select_interest_points(
        self,
        equity: pd.Series,
        drawdown: pd.Series,
        dd_metadata: dict[str, int],
    ) -> pd.Index:
        n = len(equity)
        critical = [
            0,
            n - 1,
            equity.idxmax(),
            equity.idxmin(),
            drawdown.idxmax(),
            dd_metadata["dd_start"],
            dd_metadata["dd_peak"],
            dd_metadata["dd_end"],
            min(dd_metadata["dd_end"] + 1, n - 1),
        ]

        critical_int = []
        for pt in critical:
            if isinstance(pt, (int, np.integer)):
                critical_int.append(int(pt))
            else:
                critical_int.append(equity.index.get_loc(pt))

        equity_values = equity.values
        local_max = argrelextrema(equity_values, np.greater, order=self.extrema_order)[0]
        local_min = argrelextrema(equity_values, np.less, order=self.extrema_order)[0]

        all_points = np.unique(np.concatenate([critical_int, local_max, local_min]))
        all_points = all_points[(all_points >= 0) & (all_points < n)]
        return pd.Index(all_points)

    def _interpolate_series(
        self,
        series: pd.Series,
        select_indices: pd.Index,
    ) -> pd.Series:
        selected = series.iloc[select_indices]
        sparse = selected.reindex(series.index)
        interpolated = sparse.interpolate(method="linear", limit_direction="both")
        return interpolated

    def plot_equity_curves(
        self,
        equity_type: Literal["strategies", "portfolios", "all"] = "strategies",
        show_drawdown: bool = True,
        interpolate: bool = True,
    ) -> go.Figure:
        filtered_equity = {}
        for name, equity in self.equity_dict.items():
            include = False
            if equity_type == "strategies":
                # Exclude portfolios, benchmarks, and leveraged
                include = "Portfolio" not in name and "Benchmark" not in name and "Leveraged" not in name
            elif equity_type == "portfolios":
                include = "Portfolio" in name or "Benchmark" in name
            elif equity_type == "all":
                include = True
            if include:
                filtered_equity[name] = equity

        if not filtered_equity:
            fig = go.Figure()
            fig.add_annotation(text="No matching equity curves")
            return self.apply_style(fig)

        if show_drawdown:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Equity Curves (Log Scale)", "Drawdown"),
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
            )
        else:
            fig = go.Figure()

        for idx, (name, equity) in enumerate(filtered_equity.items()):
            color = self._get_color(idx)
            line_width = 3 if "Portfolio" in name else 2

            if interpolate:
                drawdown, dd_metadata = self._calculate_drawdown(equity)
                interest_points = self._select_interest_points(equity, drawdown, dd_metadata)
                equity_plot = self._interpolate_series(equity, interest_points)
                drawdown_plot = self._interpolate_series(drawdown, interest_points)
            else:
                equity_plot = equity
                drawdown_plot = self._calculate_drawdown(equity)[0]

            fig.add_trace(
                go.Scatter(
                    x=equity_plot.index,
                    y=equity_plot,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line=dict(color=color, width=line_width),
                    hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            if show_drawdown:
                fig.add_trace(
                    go.Scatter(
                        x=drawdown_plot.index,
                        y=drawdown_plot,
                        mode="lines",
                        name=f"{name} DD",
                        legendgroup=name,
                        showlegend=False,
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(200, 100, 100, 0.2)",
                        hovertemplate="<b>Drawdown</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

        title_parts = [equity_type.title()]
        fig.update_layout(
            title_text=f"<b>{' '.join(title_parts)}</b>",
            hovermode="x unified",
            height=700 if show_drawdown else 500,
        )
        if show_drawdown:
            fig.update_yaxes(type="log", title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        else:
            fig.update_yaxes(type="log", title_text="Equity ($)")
        fig.update_xaxes(title_text="Date")
        return self.apply_style(fig)

    def return_portfolio(self, type: str, resample: str | None = None) -> pd.Series:
        equity = self.equity_dict[type]
        if resample:
            equity = equity.resample(resample).last().dropna()
        return equity


def create_combined_objective_plots_all_dims(
    optimize_results, param_names: list[str], estimator: str, max_cols: int = 5
) -> dict[str, Figure]:
    """
    Create combined objective function plots for all parameters across WFO periods.
    Grid dimensions are calculated dynamically based on the number of optimization results.
    """
    style = DEFAULT_STYLE
    combined_figures: dict[str, Figure] = {}

    # Dynamic grid calculation
    n_results = len(optimize_results)
    ncols = min(n_results, max_cols)
    nrows = ceil(n_results / ncols) if ncols > 0 else 1
    num_axes = nrows * ncols

    for dim_idx, param_name in enumerate(param_names):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
        axes_flat = np.asarray(axes).ravel() if num_axes > 1 else np.array([axes])

        num_to_plot = min(n_results, num_axes)
        individual_figs = []

        for i in range(num_to_plot):
            try:
                individual_fig = plot_objective(optimize_results[i], estimator=estimator, plot_dims=[dim_idx])
                individual_figs.append(individual_fig)
            except Exception as e:
                print(f"Warning: Could not create plot for WFO period {i + 1}, parameter {param_name}: {e}")
                individual_figs.append(None)

        def _copy_objective_to_ax(source_ax: Axes, target_ax: Axes) -> None:
            y_candidates = []

            for line in source_ax.get_lines():
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                if x.size == 0 or y.size == 0:
                    continue
                alpha = line.get_alpha() if line.get_alpha() is not None else 1.0

                if x.size >= 2 and np.allclose(x, x[0]) and y.size >= 2 and np.nanmin(y) >= 0.0 and np.nanmax(y) <= 1.0:
                    target_ax.axvline(
                        x=float(x[0]),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        alpha=alpha,
                    )
                else:
                    target_ax.plot(
                        x,
                        y,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        alpha=alpha,
                    )
                    y_candidates.append(y)

            for collection in source_ax.collections:
                offsets = np.asarray(collection.get_offsets(), dtype=np.float64)
                if offsets.size == 0:
                    continue
                offsets = np.asarray(offsets)
                alpha = collection.get_alpha() if collection.get_alpha() is not None else 1.0
                target_ax.scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    c=collection.get_facecolor(),
                    s=collection.get_sizes(),
                    alpha=alpha,
                )
                y_candidates.append(offsets[:, 1])

            target_ax.set_xlabel(source_ax.get_xlabel())
            target_ax.set_ylabel(source_ax.get_ylabel())

            if y_candidates:
                finite_parts = [y[np.isfinite(y)] for y in y_candidates if y.size > 0]
                if finite_parts:
                    y_all = np.concatenate(finite_parts)
                    if y_all.size > 0:
                        y_min, y_max = np.min(y_all), np.max(y_all)
                        span = y_max - y_min
                        pad = 0.08 * span if span > 0 else max(1e-6, 0.05 * (abs(y_min) + abs(y_max) + 1.0))
                        target_ax.set_ylim(y_min - pad, y_max + pad)

        for i in range(num_to_plot):
            ax = axes_flat[i]
            individual_fig = individual_figs[i]

            if individual_fig is None:
                ax.text(0.5, 0.5, f"No Data\nPeriod {i + 1}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"WFO Period {i + 1} - No Data", fontsize=8)
            else:
                try:
                    source_ax = individual_fig.axes[0]
                    _copy_objective_to_ax(source_ax, ax)
                    ax.set_title(f"WFO Period {i + 1}")
                except Exception as e:
                    print(f"Warning: Could not extract plot data for WFO period {i + 1}: {e}")
                    ax.text(0.5, 0.5, f"Plot Error\nPeriod {i + 1}", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"WFO Period {i + 1} - Error")
                finally:
                    plt.close(individual_fig)

            style.apply_mpl(fig=fig, ax=ax)

        # Hide unused axes
        for j in range(num_to_plot, num_axes):
            ax = axes_flat[j]
            ax.axis("off")
            style.apply_mpl(fig=fig, ax=ax)

        fig.suptitle(
            f"SAMBO Objective Function Analysis - Parameter: {param_name}",
            fontsize=11,
            fontweight="bold",
            color=style.font_color,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

        combined_figures[param_name] = fig

    return combined_figures


@dataclass
class ChartConfig:
    primary_color: str
    secondary_color: str | None = None
    title: str = ""
    yaxis_title: str = ""
    yaxis_type: str = "linear"
    height: int = 600
    trace_mode: str = "lines"
    trace_width: int = 2
    marker_size: int = 3
    show_hline_zero: bool = False
    layout_overrides: dict[str, Any] | None = None  # Additional layout customization
