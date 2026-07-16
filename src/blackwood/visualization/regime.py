from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from numpy.typing import NDArray

from blackwood.visualization.style import DEFAULT_STYLE


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def plot_regime_candlesticks(
    df: pd.DataFrame,
    title: str = "Regime Detection",
    width: int = 1100,
    height: int = 600,
    session_hours: tuple[float, float] | None = None,
    mark_transitions: bool = False,
    transition_window: int = 1,
    use_session_hours: bool = False,
    per_bar_shading: bool = False,
) -> go.Figure:
    """
    Regime visualization optimized for dark theme (DEFAULT_STYLE).
    New parameter:
    - per_bar_shading: bool = False
        If True, shades the background with an individual rectangle for EVERY bar,
        colored strictly by that bar's regime value. No blocking of consecutive regimes,
        no transitions (disabled automatically). This gives pure per-bar coloring
        with no overlaps, no gaps, and no special transition handling.
        Visually seamless when consecutive bars share the same regime.
        If False (default), uses efficient block shading for consecutive same-regime bars
        (fewer shapes, better performance on large datasets) while remaining visually
        identical to per-bar mode when regimes are persistent.
    """
    # Step 1: Lazy Regime Column Detection
    regime_col = next((col for col in ["regime_label", "regime", "Regime", "regime_id"] if col in df.columns), None)
    if regime_col is None:
        raise ValueError("No regime column found in ['regime_label', 'regime', 'Regime', 'regime_id']")

    # Zero-copy array views
    regime_labels: NDArray[np.int_] = df[regime_col].to_numpy(dtype=int)
    dates: NDArray[np.datetime64] = df.index.to_numpy()
    open_prices: NDArray[np.float64] = df["Open"].to_numpy(dtype=float)
    high_prices: NDArray[np.float64] = df["High"].to_numpy(dtype=float)
    low_prices: NDArray[np.float64] = df["Low"].to_numpy(dtype=float)
    close_prices: NDArray[np.float64] = df["Close"].to_numpy(dtype=float)
    n_bars = len(df)

    if n_bars == 0:
        raise ValueError("DataFrame is empty")

    # Step 2: Accent palette from DEFAULT_STYLE
    accents = [
        DEFAULT_STYLE.accent1,  # light blue
        DEFAULT_STYLE.accent6,  # orange
        DEFAULT_STYLE.accent3,  # green
        DEFAULT_STYLE.accent2,  # coral red
        DEFAULT_STYLE.accent4,  # dark red
        DEFAULT_STYLE.accent5,  # gray
    ]
    regime_names = ["Blue", "Orange", "Green", "Coral", "Red", "Gray"]

    # Step 3: Color mapping
    unique_regimes = np.unique(regime_labels)
    sorted_regimes = np.sort(unique_regimes)
    color_map = {
        rid: (accents[i % len(accents)], regime_names[i % len(regime_names)]) for i, rid in enumerate(sorted_regimes)
    }

    # Step 4: Price bounds & padding
    price_min = np.nanmin(low_prices)
    price_max = np.nanmax(high_prices)
    price_pad = (price_max - price_min) * 0.05
    y_bottom = price_min - price_pad
    y_top = price_max + price_pad

    # Step 5: Extend dates for perfect bar coverage
    last_delta = dates[-1] - dates[-2] if n_bars > 1 else np.timedelta64(1, "D")
    extended_end = dates[-1] + last_delta
    extended_dates = np.append(dates, extended_end)

    # Step 6: Helper for rectangle shapes
    def _make_rect(
        t_start: np.datetime64, t_end: np.datetime64, fill_color: str, layer: str = "below"
    ) -> dict[str, Any]:
        return dict(
            type="rect",
            xref="x",
            yref="y",
            x0=t_start,
            x1=t_end,
            y0=y_bottom,
            y1=y_top,
            fillcolor=fill_color,
            layer=layer,
            line_width=0,
        )

    # Step 7: Build regime background shapes
    shapes = []
    fill_alpha = 0.25
    add_transitions = mark_transitions and not per_bar_shading

    if per_bar_shading:
        for i in range(n_bars):
            rid = regime_labels[i]
            line_color, _ = color_map[rid]
            fill_color = hex_to_rgba(line_color, fill_alpha)
            shapes.append(_make_rect(dates[i], extended_dates[i + 1], fill_color))
    else:
        change_indices = np.where(regime_labels[:-1] != regime_labels[1:])[0] + 1
        starts = np.concatenate(([0], change_indices))
        ends = np.concatenate((change_indices, [n_bars]))

        for start_idx, end_idx in zip(starts, ends, strict=True):
            rid = regime_labels[start_idx]
            line_color, _ = color_map[rid]
            fill_color = hex_to_rgba(line_color, fill_alpha)
            shapes.append(_make_rect(dates[start_idx], extended_dates[end_idx], fill_color))

        transition_mask = None
        if add_transitions and n_bars > 1:
            transition_mask = np.zeros(n_bars, dtype=bool)
            for change_idx in change_indices:
                start = max(0, change_idx - transition_window)
                end = min(n_bars, change_idx + transition_window + 1)
                transition_mask[start:end] = True

            transition_fill = hex_to_rgba(DEFAULT_STYLE.accent5, 0.35)
            if transition_mask.any():
                padded = np.concatenate(([False], transition_mask, [False]))
                diff = np.diff(padded.astype(int))
                trans_starts = np.where(diff == 1)[0]
                trans_ends = np.where(diff == -1)[0]
                for s, e in zip(trans_starts, trans_ends, strict=True):
                    shapes.append(_make_rect(dates[s], extended_dates[e], transition_fill, layer="above"))

    # Step 8: Range breaks
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if use_session_hours and session_hours:
        start_h, end_h = session_hours
        if not (start_h == 0 and end_h == 24):
            rangebreaks.append(dict(bounds=[end_h, start_h], pattern="hour"))

    # Step 9: Figure setup
    fig = go.Figure()
    DEFAULT_STYLE.apply(fig)

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name="Price",
            increasing_fillcolor=DEFAULT_STYLE.accent3,
            increasing_line_color=DEFAULT_STYLE.accent3,
            decreasing_fillcolor=DEFAULT_STYLE.accent2,
            decreasing_line_color=DEFAULT_STYLE.accent2,
            showlegend=False,
        )
    )

    # Step 10: Legend/stats
    regime_stats: list[dict[str, str | float]] = []
    for rid in sorted_regimes:
        line_color, name = color_map[rid]
        mask = regime_labels == rid
        regime_stats.append({"label": f"Regime {rid}", "name": name, "pct": 100.0 * mask.mean(), "color": line_color})

    if add_transitions:
        transition_mask: NDArray[np.bool_] | None = None
        transition_pct = 100.0 * transition_mask.mean() if transition_mask is not None else 0.0
        regime_stats.append(
            {
                "label": "Transition",
                "name": f"±{transition_window} bars",
                "pct": transition_pct,
                "color": DEFAULT_STYLE.accent5,
            }
        )

    # Adaptive spacing
    base_spacing = 0.08
    adaptive_spacing = min(base_spacing, (0.7 * height / 600) / max(len(regime_stats), 1))

    annotations = [
        dict(
            xref="paper",
            yref="paper",
            x=1.01,
            y=0.95 - i * adaptive_spacing,
            xanchor="left",
            yanchor="middle",
            text=f"<b>{stat['label']}</b> <span style='font-size:9px'>({stat['name']})</span><br>"
            f"<span style='font-size:10px'>({stat['pct']:.1f}%)</span>",
            font=dict(size=11, color=stat["color"], family="Arial"),
            showarrow=False,
            bgcolor=DEFAULT_STYLE.plot_bgcolor,
            bordercolor=stat["color"],
            borderwidth=2,
            borderpad=4,
        )
        for i, stat in enumerate(regime_stats)
    ]

    # Step 11: Final layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, family="Arial")),
        width=width,
        height=height,
        margin=dict(l=50, r=150, t=80, b=50),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        shapes=shapes,
        annotations=annotations,
    )
    fig.update_xaxes(rangebreaks=rangebreaks)

    return fig
