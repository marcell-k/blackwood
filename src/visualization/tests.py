from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.visualization.style import DEFAULT_STYLE


class PermutationPlotter:
    """Permutation test visualization with PlotStyle theming."""

    def __init__(self):
        self.style = DEFAULT_STYLE

    @staticmethod
    def _daily(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")
        return df.resample("D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(how="all")

    def plot_original_vs_permuted(
        self,
        original_df: pd.DataFrame,
        permuted_dfs: list[pd.DataFrame],
        n_to_show: int = 3,
        title: str = "Original vs Permuted",
    ) -> None:
        """Plot original vs permuted data with PlotStyle theming."""
        base = self._daily(original_df)
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=base.index,
                open=base["Open"],
                high=base["High"],
                low=base["Low"],
                close=base["Close"],
                name="Original",
                increasing_line_color=self.style.accent3,
                decreasing_line_color=self.style.accent2,
            )
        )

        colors = [self.style.accent1, self.style.accent5, self.style.accent4, self.style.accent6, self.style.accent3]
        for i, p in enumerate(permuted_dfs[:n_to_show]):
            d = self._daily(p)
            fig.add_trace(
                go.Candlestick(
                    x=d.index,
                    open=d["Open"],
                    high=d["High"],
                    low=d["Low"],
                    close=d["Close"],
                    name=f"Permutation {i + 1}",
                    increasing_line_color=colors[i % len(colors)],
                    decreasing_line_color=colors[i % len(colors)],
                    opacity=0.7,
                )
            )

        fig.update_layout(
            title={"text": f"<b>{title}</b>", "x": 0.5},
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        # Apply PlotStyle theming
        self.style.apply(fig)
        fig.show()

    def plot_permutation_histogram(
        self,
        perm_values: Sequence[float],
        real_value: float,
        metric_name: str = "Metric",
        title: str = "Permutation Distribution",
    ) -> None:
        """Plot permutation histogram with PlotStyle theming."""
        arr = np.asarray(perm_values, dtype=float)
        arr = arr[~np.isnan(arr)]
        fig = go.Figure()
        bins = min(50, max(10, int(np.sqrt(len(arr)))) or 10)

        fig.add_trace(
            go.Histogram(
                x=arr,
                nbinsx=bins,
                name="Permuted",
                marker_color=self.style.accent1,
                marker_line_color=self.style.line,
                opacity=0.7,
            )
        )

        fig.add_vline(
            x=real_value, line_color=self.style.accent2, line_width=2, annotation_text="Real", annotation_position="top"
        )

        fig.update_layout(
            title={"text": f"<b>{title}</b>", "x": 0.5},
            xaxis_title=metric_name,
            yaxis_title="Frequency",
            showlegend=False,
        )

        # Apply PlotStyle theming
        self.style.apply(fig)
        fig.show()

    def plot_permutation_results(
        self,
        fold_results: list[dict[str, Any]],
        fold_idx: int = 0,
        metrics: list[str] = ["Sharpe Ratio", "Profit Factor"],
        width: int = 1000,
        height: int = 300,
    ) -> None:
        """
        Plot permutation test results for multiple metrics with PlotStyle theming.

        Parameters
        ----------
        fold_results : List[Dict[str, Any]]
            Results from PermutationWalkForwardTester.evaluate_split()
        fold_idx : int
            Fold index to plot
        metrics : List[str]
            List of metrics to plot (must be 2 metrics)

        """
        if fold_idx >= len(fold_results):
            print(f"Error: fold_idx {fold_idx} exceeds available folds {len(fold_results)}")
            return

        if len(metrics) != 2:
            raise ValueError("Exactly 2 metrics required for side-by-side plot")

        fold_data = fold_results[fold_idx]
        metric1, metric2 = metrics

        real_oos_1 = fold_data.get(f"{metric1}_real_oos_perf", np.nan)
        perm_oos_1 = fold_data.get(f"{metric1}_perm_oos_perfs", [])
        p_value_1 = fold_data.get(f"{metric1}_oos_p_value", np.nan)

        real_oos_2 = fold_data.get(f"{metric2}_real_oos_perf", np.nan)
        perm_oos_2 = fold_data.get(f"{metric2}_perm_oos_perfs", [])
        p_value_2 = fold_data.get(f"{metric2}_oos_p_value", np.nan)

        valid_perm_1 = [p for p in perm_oos_1 if not np.isnan(p)]
        valid_perm_2 = [p for p in perm_oos_2 if not np.isnan(p)]

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
        fig.update_annotations(font_size=11)

        if valid_perm_1:
            n_bins_1 = min(30, max(10, len(set(valid_perm_1))))

            fig.add_trace(
                go.Histogram(
                    x=valid_perm_1,
                    nbinsx=n_bins_1,
                    name=f"Permuted {metric1}",
                    marker_color=self.style.accent1,
                    marker_line_color=self.style.line,
                    marker_line_width=1,
                    opacity=0.7,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            if not np.isnan(real_oos_1):
                fig.add_vline(
                    x=real_oos_1,
                    line_color=self.style.accent2,
                    line_width=3,
                    annotation_text=f"Real: {real_oos_1:.3f}",
                    annotation_position="top",
                    row=1,
                    col=1,
                )

            if not np.isnan(p_value_1):
                color_1 = self.style.accent3 if p_value_1 < 0.05 else self.style.accent4
                significance_1 = "Significant" if p_value_1 < 0.05 else "Not Significant"

                p_value_text_1 = f"<b>P-value: {p_value_1:.4f}</b><br><sub>{significance_1}</sub>"

                fig.add_annotation(
                    text=p_value_text_1,
                    xref="x1",
                    yref="y1",
                    x=0.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    bgcolor=color_1,
                    bordercolor=self.style.line,
                    borderwidth=1,
                    font=dict(size=11, color=self.style.font_color),
                    opacity=0.9,
                    row=1,
                    col=1,
                )

        if valid_perm_2:
            n_bins_2 = min(30, max(10, len(set(valid_perm_2))))

            fig.add_trace(
                go.Histogram(
                    x=valid_perm_2,
                    nbinsx=n_bins_2,
                    name=f"Permuted {metric2}",
                    marker_color=self.style.accent4,
                    marker_line_color=self.style.line,
                    marker_line_width=1,
                    opacity=0.7,
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

            if not np.isnan(real_oos_2):
                fig.add_vline(
                    x=real_oos_2,
                    line_color=self.style.accent2,
                    line_width=3,
                    annotation_text=f"Real: {real_oos_2:.3f}",
                    annotation_position="top",
                    row=1,
                    col=2,
                )

            if not np.isnan(p_value_2):
                color_2 = self.style.accent3 if p_value_2 < 0.05 else self.style.accent4
                significance_2 = "Significant" if p_value_2 < 0.05 else "Not Significant"

                p_value_text_2 = f"<b>P-value: {p_value_2:.4f}</b><br><sub>{significance_2}</sub>"

                fig.add_annotation(
                    text=p_value_text_2,
                    xref="x2",
                    yref="y2",
                    x=0.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    bgcolor=color_2,
                    bordercolor=self.style.line,
                    borderwidth=1,
                    font=dict(size=11, color=self.style.font_color),
                    opacity=0.9,
                    row=1,
                    col=2,
                )

        fig.update_layout(
            title=dict(
                text=f"<b>Permutation Test Results - Fold {fold_idx + 1}</b>",
                x=0.5,
                font=dict(size=14, family="Arial Black", color=self.style.font_color),
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=0.97, xanchor="right", x=1, font=dict(color=self.style.font_color)
            ),
        )

        fig.update_xaxes(
            title_text=metric1,
            row=1,
            col=1,
            tickfont=dict(color=self.style.font_color),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=1,
        )
        fig.update_xaxes(
            title_text=metric2,
            row=1,
            col=2,
            tickfont=dict(color=self.style.font_color),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=1,
        )
        fig.update_yaxes(
            title_text="Frequency",
            row=1,
            col=1,
            tickfont=dict(color=self.style.font_color),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=1,
        )
        fig.update_yaxes(
            title_text="Frequency",
            row=1,
            col=2,
            tickfont=dict(color=self.style.font_color),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=1,
        )

        self.style.apply(fig)
        fig.show()


def _to_daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to daily OHLC with first/max/min/last aggregation."""
    daily = df.resample("D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(how="all")
    return daily


def _fold_color_from_p(p: float | None) -> tuple[str, str, str]:
    """Map a p-value to (fill_rgba, line_rgba, label) using PlotStyle colors."""
    style = DEFAULT_STYLE

    if p is None or not np.isfinite(p):
        return ("rgba(128,128,128,0.20)", "rgba(128,128,128,1.0)", "p=N/A")
    p = float(np.clip(p, 0.0, 1.0))

    if p < 0.05:
        # Extract RGB from hex color
        hex_color = style.accent3.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (f"rgba({r},{g},{b},0.22)", style.accent3, f"p={p:.4f} (<0.05)")
    elif p < 0.1:
        hex_color = style.accent4.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (f"rgba({r},{g},{b},0.22)", style.accent4, f"p={p:.4f} (<0.1)")
    else:
        hex_color = style.accent2.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (f"rgba({r},{g},{b},0.22)", style.accent2, f"p={p:.4f} (≥0.05)")


def plot_cross_validation_folds(
    fold_dfs: list[pd.DataFrame],
    fold_p_values: Sequence[float | None] | None = None,
) -> go.Figure:
    """Visualize fold time spans with p-value coded background shading using PlotStyle."""
    plot_style = DEFAULT_STYLE

    # Validate and standardize folds
    cleaned: list[pd.DataFrame] = []
    cleaned.extend(fold_dfs)

    # Combine and resample to daily OHLC
    combined = pd.concat(cleaned, axis=0, ignore_index=False).loc[:, ["Open", "High", "Low", "Close"]]
    daily = _to_daily_ohlc(combined)

    # Base candlestick
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=daily.index,
                open=daily["Open"],
                high=daily["High"],
                low=daily["Low"],
                close=daily["Close"],
                name="OHLC",
                increasing_line_color=plot_style.accent3,
                decreasing_line_color=plot_style.accent2,
                showlegend=False,
            )
        ]
    )

    # Add fold overlays with colors from p-values
    if fold_p_values is None:
        fold_p_values = [None] * len(cleaned)

    if len(fold_p_values) != len(cleaned):
        raise ValueError("fold_p_values must have the same length as fold_dfs.")

    for i, (df, pval) in enumerate(zip(cleaned, fold_p_values)):
        if df.empty:
            continue
        x0, x1 = df.index.min(), df.index.max()
        fill_rgba, line_rgba, label = _fold_color_from_p(pval)

        # Vertical rectangle spanning full y-range
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=fill_rgba,
            line_width=0,
            opacity=0.22,
            layer="below",
        )
        # Legend proxy with fold number and p-value label
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"Fold {i + 1} • {label}",
                marker=dict(color=line_rgba, size=10),
            )
        )

    # Layout and axes
    fig.update_layout(
        title=dict(
            text="<b>Cross-Validation Folds: Daily Candlestick Chart</b>",
            x=0.5,
            font=dict(size=20, color=plot_style.font_color),
        ),
        height=600,
        width=1200,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color=plot_style.font_color)
        ),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], tickfont=dict(color=plot_style.font_color))
    fig.update_yaxes(tickfont=dict(color=plot_style.font_color))
    plot_style.apply(fig)
    return fig
