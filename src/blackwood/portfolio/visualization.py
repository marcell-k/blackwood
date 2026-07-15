from dataclasses import dataclass
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import normalized_mutual_info_score

from blackwood.visualization.style import DEFAULT_STYLE


@dataclass
class MonteCarloResults:
    """
    Container for Monte Carlo simulation results.

    Attributes:
        simulated_equity: Array of simulated equity curves (n_sims, n_days)
        simulated_max_dd: Maximum drawdown for each simulation (n_sims,)
        simulated_avg_dd: Average drawdown for each simulation (n_sims,)
        actual_equity: Original equity curve
        actual_max_dd: Actual maximum drawdown
        actual_avg_dd: Actual average drawdown
        percentiles_max_dd: Percentile dict for max DD {5: val, 25: val, ...}
        percentiles_avg_dd: Percentile dict for avg DD
        actual_max_dd_percentile: Where actual max DD ranks (0-100)
        actual_avg_dd_percentile: Where actual avg DD ranks (0-100)
        dates: DateTime index for plotting

    """

    simulated_equity: np.ndarray
    simulated_max_dd: np.ndarray
    simulated_avg_dd: np.ndarray
    actual_equity: pd.Series
    actual_max_dd: float
    actual_avg_dd: float
    percentiles_max_dd: dict[int, float]
    percentiles_avg_dd: dict[int, float]
    actual_max_dd_percentile: float
    actual_avg_dd_percentile: float
    dates: pd.DatetimeIndex


class PortfolioVisualizer:
    def __init__(self, all_results: dict, returns: pd.DataFrame):
        self.all_results = all_results
        self.returns = returns
        self.style = DEFAULT_STYLE
        self.colors = [
            self.style.accent1,
            self.style.accent2,
            self.style.accent3,
            self.style.accent4,
            self.style.accent5,
            self.style.accent6,
        ]

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        valid_mask = equity.notna()
        drawdown = pd.Series(np.nan, index=equity.index, dtype=float)

        if not valid_mask.any():
            return drawdown

        equity_valid = equity.loc[valid_mask]
        running_max = equity_valid.cummax()
        drawdown.loc[valid_mask] = ((equity_valid - running_max) / running_max) * 100
        return drawdown

    def plot_equity_and_drawdown(self, show_drawdown: bool = True, filter_type: str = "all") -> go.Figure:
        """
        Plot equity curves and drawdowns with filtering options.

        Args:
            show_drawdown: If True, show drawdown panel below equity
            filter_type: What to display:
                'all' - Everything (strategies + normalized + portfolios + leveraged)
                'strategies' - Only original strategies (no Portfolio, no Normalized, no Leveraged)
                'normalized' - Only normalized strategies (_Normalized suffix)
                'portfolios' - Only portfolios (no Leveraged)
                'leveraged' - Only leveraged portfolios (_Leveraged suffix)
                'portfolios_all' - All portfolios (with and without leverage)

        Returns:
            Plotly Figure

        """
        filtered_results = {}

        for name, data in self.all_results.items():
            include = False

            if filter_type == "all":
                include = True

            elif filter_type == "strategies":
                include = "Portfolio" not in name and "Normalized" not in name and "Leveraged" not in name

            elif filter_type == "normalized":
                include = "Normalized" in name

            elif filter_type == "portfolios":
                include = "Portfolio" in name and "Leveraged" not in name

            elif filter_type == "leveraged":
                include = "_L" in name

            elif filter_type == "portfolios_all":
                include = "Portfolio" in name

            if include:
                filtered_results[name] = data

        if len(filtered_results) == 0:
            print(f"No results match filter_type='{filter_type}'")
            return go.Figure()

        if show_drawdown:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Multi-Strategy Equity Curves (Log Scale)", "Drawdown Comparison"),
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                x_title="Date",
            )
        else:
            fig = go.Figure()

        for idx, (name, data) in enumerate(filtered_results.items()):
            equity = data["results"]["equity"]
            color = self.colors[idx % len(self.colors)]

            is_portfolio = any(keyword in name.lower() for keyword in ["portfolio", "combined", "basket"])
            line_width = 4 if is_portfolio else 2.5

            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line=dict(color=color, width=line_width),
                    hovertemplate=("<b>%{fullData.name}</b><br>Date: %{x}<br>Equity: $%{y:,.0f}<br><extra></extra>"),
                ),
                row=1,
                col=1,
            )

            if show_drawdown:
                drawdown = self._calculate_drawdown(equity)
                rgb = tuple(int(color[1:][i : i + 2], 16) for i in (0, 2, 4))

                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        mode="lines",
                        name=f"{name} DD",
                        legendgroup=name,
                        showlegend=False,
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)",
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<br><extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )

        title_suffix = {
            "all": "All Strategies",
            "strategies": "Original Strategies Only",
            "normalized": "Normalized Strategies Only",
            "portfolios": "Portfolios (Unleveraged)",
            "leveraged": "Portfolios (Leveraged)",
            "portfolios_all": "All Portfolios",
        }.get(filter_type, "Multi-Strategy")

        fig.update_layout(
            title=dict(
                text=f"<b>{title_suffix} - Performance Analysis</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=20, color=self.style.font_color),
            ),
            hovermode="x unified",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor=self.style.plot_bgcolor,
                bordercolor=self.style.line,
                borderwidth=1,
                font=dict(color=self.style.font_color),
            ),
            height=700 if show_drawdown else 500,
            font=dict(color=self.style.font_color),
        )

        axis_style = dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.style.grid,
            showline=True,
            linewidth=1,
            linecolor=self.style.line,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )

        fig.update_xaxes(row=1, col=1, **axis_style, showticklabels=False)
        fig.update_yaxes(title_text="Equity Value ($) - Log Scale", type="log", row=1, col=1, **axis_style)

        if show_drawdown:
            fig.update_xaxes(title_text="Date", row=2, col=1, **axis_style, showticklabels=True)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, **axis_style)

        self.style.apply(fig)
        return fig

    def plot_normalized_with_rebalances(self, rebalance_freq: str = "QE", show_drawdown: bool = True) -> go.Figure:

        normalized_results = {k: v for k, v in self.all_results.items() if "Normalized" in k}

        if len(normalized_results) == 0:
            print("No normalized strategies found. Make sure to create them first.")
            return go.Figure()

        idx = self.returns.index
        quarter_ends = pd.date_range(idx.min(), idx.max(), freq=rebalance_freq)

        rebalance_dates = idx[idx.get_indexer(quarter_ends, method="ffill")]
        rebalance_dates = pd.DatetimeIndex(sorted(set(rebalance_dates)))

        if show_drawdown:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Normalized Strategies - Equal Volatility (Log Scale)", "Drawdown Comparison"),
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                x_title="Date",
            )
        else:
            fig = go.Figure()

        for idx, (name, data) in enumerate(normalized_results.items()):
            equity = data["results"]["equity"]
            color = self.colors[idx % len(self.colors)]

            clean_name = name.replace("_Normalized", "")

            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity,
                    mode="lines",
                    name=clean_name,
                    legendgroup=clean_name,
                    line=dict(color=color, width=2.5),
                    hovertemplate=("<b>%{fullData.name}</b><br>Date: %{x}<br>Equity: $%{y:,.0f}<br><extra></extra>"),
                ),
                row=1,
                col=1,
            )

            if show_drawdown:
                drawdown = self._calculate_drawdown(equity)
                rgb = tuple(int(color[1:][i : i + 2], 16) for i in (0, 2, 4))

                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        mode="lines",
                        name=f"{clean_name} DD",
                        legendgroup=clean_name,
                        showlegend=False,
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)",
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<br><extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )

        for rebal_date in rebalance_dates:
            fig.add_vline(x=rebal_date, line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dash"), row=1, col=1)

            if show_drawdown:
                fig.add_vline(
                    x=rebal_date, line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dash"), row=2, col=1
                )

        fig.update_layout(
            title=dict(
                text=f"<b>Normalized Strategies with {rebalance_freq} Rebalancing</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=20, color=self.style.font_color),
            ),
            hovermode="x unified",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor=self.style.plot_bgcolor,
                bordercolor=self.style.line,
                borderwidth=1,
                font=dict(color=self.style.font_color),
            ),
            height=700 if show_drawdown else 500,
            font=dict(color=self.style.font_color),
        )

        axis_style = dict(
            showgrid=False,
            gridwidth=0,
            gridcolor=None,
            showline=True,
            linewidth=1,
            linecolor=self.style.line,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )

        fig.update_xaxes(row=1, col=1, **axis_style, showticklabels=False)
        fig.update_yaxes(title_text="Equity Value ($) - Log Scale", type="log", row=1, col=1, **axis_style)

        if show_drawdown:
            fig.update_xaxes(title_text="Date", row=2, col=1, **axis_style, showticklabels=True)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, **axis_style)

        self.style.apply(fig)
        return fig

    def plot_allocation_timeline(self, strategy_name: str) -> go.Figure:
        weights_df = self.all_results[strategy_name]["results"]["weights_history"]

        fig = go.Figure()

        for idx, col in enumerate(weights_df.columns):
            color = self.colors[idx % len(self.colors)]

            fig.add_trace(
                go.Scatter(
                    x=weights_df.index,
                    y=weights_df[col],
                    mode="lines+markers",
                    name=col,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6, color=color),
                    hovertemplate=("<b>%{fullData.name}</b><br>Date: %{x}<br>Weight: %{y:.2%}<br><extra></extra>"),
                )
            )

        fig.update_layout(
            title=dict(
                text=f"<b>{strategy_name} - Allocation Over Time</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color=self.style.font_color),
            ),
            xaxis_title="Rebalance Date",
            yaxis_title="Allocation Weight",
            hovermode="x unified",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor=self.style.plot_bgcolor,
                bordercolor=self.style.line,
                borderwidth=1,
                font=dict(color=self.style.font_color),
            ),
            height=500,
            font=dict(color=self.style.font_color),
        )

        axis_style = dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.style.grid,
            showline=True,
            linewidth=1,
            linecolor=self.style.line,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )

        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style, tickformat=".0%", range=[0, 1])

        self.style.apply(fig)
        return fig

    def plot_all_allocation_timelines(self) -> dict[str, go.Figure]:
        figures = {}

        for strategy_name in self.all_results:
            fig = self.plot_allocation_timeline(strategy_name)
            figures[strategy_name] = fig

        return figures

    def plot_allocation_comparison(self) -> go.Figure:
        final_alloc_dict = {}

        for name, data in self.all_results.items():
            weights = data["results"]["weights_history"]
            if not weights.empty:
                final_alloc_dict[name] = weights.iloc[-1]

        final_alloc_df = pd.DataFrame(final_alloc_dict).T

        fig = go.Figure()

        for idx, col in enumerate(final_alloc_df.columns):
            color = self.colors[idx % len(self.colors)]

            fig.add_trace(
                go.Bar(
                    x=final_alloc_df.index,
                    y=final_alloc_df[col],
                    name=col,
                    marker_color=color,
                    hovertemplate=("<b>%{fullData.name}</b><br>Strategy: %{x}<br>Weight: %{y:.2%}<br><extra></extra>"),
                )
            )

        fig.update_layout(
            title=dict(
                text="<b>Final Portfolio Allocations by Strategy</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color=self.style.font_color),
            ),
            xaxis_title="Optimization Strategy",
            yaxis_title="Allocation Weight",
            barmode="group",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor=self.style.plot_bgcolor,
                bordercolor=self.style.line,
                borderwidth=1,
                font=dict(color=self.style.font_color),
                title=dict(text="Trading Strategy", font=dict(color=self.style.font_color)),
            ),
            height=500,
            font=dict(color=self.style.font_color),
        )

        axis_style = dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.style.grid,
            showline=True,
            linewidth=1,
            linecolor=self.style.line,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )

        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style, tickformat=".0%", range=[0, 1])

        self.style.apply(fig)
        return fig

    def plot_returns_distribution(self) -> go.Figure:
        fig = make_subplots(
            rows=1, cols=len(self.returns.columns), subplot_titles=[col for col in self.returns.columns]
        )

        for idx, col in enumerate(self.returns.columns):
            color = self.colors[idx % len(self.colors)]

            fig.add_trace(
                go.Histogram(
                    x=self.returns[col].values * 100,
                    nbinsx=50,
                    name=col,
                    marker_color=color,
                    opacity=0.7,
                    hovertemplate=("Return: %{x:.2f}%<br>Count: %{y}<br><extra></extra>"),
                ),
                row=1,
                col=idx + 1,
            )

        fig.update_layout(
            title=dict(
                text="<b>Distribution of Daily Returns</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color=self.style.font_color),
            ),
            showlegend=False,
            height=400,
            font=dict(color=self.style.font_color),
        )

        axis_style = dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.style.grid,
            showline=True,
            linewidth=1,
            linecolor=self.style.line,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )

        for i in range(1, len(self.returns.columns) + 1):
            fig.update_xaxes(title_text="Daily Return (%)", row=1, col=i, **axis_style)
            fig.update_yaxes(title_text="Frequency", row=1, col=i, **axis_style)

        self.style.apply(fig)
        return fig

    def show_all_plots(self):
        self.plot_equity_and_drawdown().show()

        for strategy_name in self.all_results:
            self.plot_allocation_timeline(strategy_name).show()

        self.plot_allocation_comparison().show()
        self.plot_returns_distribution().show()

    def plot_correlation_and_nmi(
        self,
        n_bins: int = 10,
        filter_type: str = "all",
        method: Literal["arithmetic", "geometric", "min", "max"] = "arithmetic",
    ):
        """
        Plot Pearson correlation and NMI matrices using Matplotlib.
        """
        # --- filter strategies ---
        filtered = []
        for name in self.returns.columns:
            include = False

            if filter_type == "all":
                include = True
            elif filter_type == "strategies":
                include = "Portfolio" not in name and "Normalized" not in name and "Leveraged" not in name
            elif filter_type == "normalized":
                include = "Normalized" in name
            elif filter_type == "portfolios":
                include = "Portfolio" in name and "Leveraged" not in name
            elif filter_type == "leveraged":
                include = "Leveraged" in name
            elif filter_type == "portfolios_all":
                include = "Portfolio" in name

            if include:
                filtered.append(name)

        if not filtered:
            return None

        data = self.returns[filtered]

        # --- correlation ---
        corr = data.corr().values

        # --- NMI ---
        disc = pd.DataFrame(index=data.index, columns=data.columns)
        for col in data.columns:
            disc[col] = pd.qcut(data[col], q=n_bins, labels=False, duplicates="drop")

        n = len(filtered)
        nmi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    nmi[i, j] = 1.0
                else:
                    nmi[i, j] = normalized_mutual_info_score(
                        disc.iloc[:, i].values,
                        disc.iloc[:, j].values,
                        average_method=method,
                    )

        # --- plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Pearson correlation
        im0 = axes[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        axes[0].set_title("Pearson Correlation\nLinear Dependencies")
        axes[0].set_xticks(range(n))
        axes[0].set_yticks(range(n))
        axes[0].set_xticklabels(filtered, rotation=45, ha="right")
        axes[0].set_yticklabels(filtered)

        for i in range(n):
            for j in range(n):
                axes[0].text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=9)

        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046)
        cbar0.set_label("Correlation")

        # NMI
        im1 = axes[1].imshow(nmi, cmap="plasma", vmin=0, vmax=1, aspect="auto")

        axes[1].set_title(f"Normalized Mutual Information\nLinear + Non-Linear ({n_bins} quantiles)")
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(filtered, rotation=45, ha="right")
        axes[1].set_yticklabels(filtered)

        for i in range(n):
            for j in range(n):
                axes[1].text(j, i, f"{nmi[i, j]:.3f}", ha="center", va="center", fontsize=9)

        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046)
        cbar1.set_label("NMI")

        fig.suptitle("Dependency Analysis", fontsize=18, color=DEFAULT_STYLE.font_color)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        DEFAULT_STYLE.apply_mpl(fig)
        return fig

    def plot_rolling_correlation(
        self,
        rebalance_freq: str = "QE",
        filter_type: str = "normalized",
        window: int = 60,
    ):
        """
        Plot rolling correlation at rebalance frequency using Matplotlib.
        """
        # --- filter strategies ---
        filtered = []
        for name in self.returns.columns:
            include = False

            if filter_type == "all":
                include = True
            elif filter_type == "strategies":
                include = "Portfolio" not in name and "Normalized" not in name and "Leveraged" not in name
            elif filter_type == "normalized":
                include = "Normalized" in name
            elif filter_type == "portfolios":
                include = "Portfolio" in name and "Leveraged" not in name
            elif filter_type == "leveraged":
                include = "Leveraged" in name
            elif filter_type == "portfolios_all":
                include = "Portfolio" in name

            if include:
                filtered.append(name)

        if len(filtered) < 2:
            return None

        data = self.returns[filtered]

        idx = data.index
        rebal_dates = pd.date_range(idx.min(), idx.max(), freq=rebalance_freq)
        rebal_dates = idx[idx.get_indexer(rebal_dates, method="ffill")]
        rebal_dates = pd.DatetimeIndex(sorted(set(rebal_dates)))

        fig, ax = plt.subplots(figsize=(14, 6))

        color_idx = 0
        for i, s1 in enumerate(filtered):
            for s2 in filtered[i + 1 :]:
                vals = []
                dates = []

                for d in rebal_dates:
                    w0 = d - pd.Timedelta(days=window)
                    win = data.loc[w0:d, [s1, s2]]
                    if len(win) > 1:
                        vals.append(win.corr().iloc[0, 1])
                        dates.append(d)

                if vals:
                    ax.plot(
                        dates,
                        vals,
                        color=self.colors[color_idx % len(self.colors)],
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        label=f"{s1} vs {s2}",
                    )
                    color_idx += 1

        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"Rolling Correlation ({rebalance_freq} rebalance)\nWindow: {window} days")
        ax.set_xlabel("Rebalance Date")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True)

        fig.tight_layout()
        DEFAULT_STYLE.apply_mpl(fig)
        return fig

    def plot_consecutive_drawdown_days_distribution(
        self,
        portfolio_name: str,
    ):

        equity = self.all_results[portfolio_name]["results"]["equity"]
        dd = self._calculate_drawdown(equity)

        is_dd = dd < 0
        durations = []
        count = 0

        for i in range(len(is_dd)):
            if is_dd.iloc[i]:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0

        if count > 0:
            durations.append(count)

        if not durations:
            return None

        durations = pd.Series(durations)

        total_dd_days = is_dd.sum()
        if abs(total_dd_days - durations.sum()) > 1:
            raise ValueError("Drawdown duration mismatch")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(durations, bins=30, alpha=0.75, color=DEFAULT_STYLE.accent2)

        mean_d = durations.mean()
        med_d = durations.median()
        max_d = durations.max()

        stats_text = (
            f"Mean: {mean_d:.1f} days\n"
            f"Median: {med_d:.1f} days\n"
            f"Max: {max_d:.0f} days\n"
            f"Count: {len(durations)} episodes"
        )

        ax.text(
            0.98,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.4),
            color=DEFAULT_STYLE.font_color,
        )

        ax.set_title(f"{portfolio_name} — Consecutive Drawdown Days\nTime spent continuously underwater")
        ax.set_xlabel("Consecutive Days in Drawdown")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        fig.tight_layout()
        DEFAULT_STYLE.apply_mpl(fig)
        return fig

    def plot_drawdown_duration_distribution(
        self,
        portfolio_name: str,
    ):
        equity = self.all_results[portfolio_name]["results"]["equity"].copy()
        running_max = equity.cummax()

        new_high = running_max.diff().fillna(0) > 0
        is_peak = new_high | (equity.index == equity.index[0])
        peak_dates = equity.index[is_peak]

        if len(peak_dates) < 2:
            return None

        durations = []
        magnitudes = []

        for i in range(len(peak_dates) - 1):
            peak_date = peak_dates[i]
            peak_val = equity.loc[peak_date]

            future = equity.loc[equity.index > peak_date]
            recovered = future >= peak_val

            if recovered.any():
                recovery_date = recovered.idxmax()
                duration = (recovery_date - peak_date).days
                durations.append(duration)

                trough = equity.loc[peak_date:recovery_date].min()
                magnitudes.append((trough - peak_val) / peak_val)

        last_peak = peak_dates[-1]
        if equity.iloc[-1] < equity.loc[last_peak]:
            duration = (equity.index[-1] - last_peak).days
            durations.append(duration)

            trough = equity.loc[last_peak:].min()
            magnitudes.append((trough - equity.loc[last_peak]) / equity.loc[last_peak])

        durations = pd.Series(durations)
        magnitudes = pd.Series(magnitudes)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(durations, bins=30, alpha=0.75, color=DEFAULT_STYLE.accent4)

        stats = (
            f"Mean Duration: {durations.mean():.1f} days\n"
            f"Median: {durations.median():.1f} days\n"
            f"Max: {durations.max():.0f} days\n"
            f"Mean DD: {magnitudes.mean() * 100:.2f}%\n"
            f"Count: {len(durations)} cycles"
        )

        ax.text(
            0.98,
            0.95,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.4),
            color=DEFAULT_STYLE.font_color,
        )

        ax.set_title(f"{portfolio_name} — Peak-to-Recovery Drawdown Duration")
        ax.set_xlabel("Duration (days)")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        fig.tight_layout()
        DEFAULT_STYLE.apply_mpl(fig)
        return fig

    def plot_leverage_and_volatility(
        self,
        portfolio_name: str,
    ):
        # --- 1. Extract data ---
        leverage_history = self.all_results[portfolio_name]["results"]["leverage_history"]

        # --- 2. Data preparation ---
        df = pd.DataFrame(leverage_history)
        df["date"] = pd.to_datetime(df["date"])

        # --- 3. Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        fig.suptitle(
            f"Leverage & Volatility Analysis for: {portfolio_name}", fontsize=18, y=0.98, color=DEFAULT_STYLE.font_color
        )

        # --- Top: Leverage ---
        ax1.plot(
            df["date"],
            df["leverage"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Applied Leverage",
            color=DEFAULT_STYLE.accent1,
        )

        ax1.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="1x (No Leverage)",
        )

        ax1.set_ylabel("Leverage Ratio (x)")
        ax1.set_title("Leverage Applied at Each Rebalance")
        ax1.grid(True, which="both")
        ax1.legend()

        # --- Bottom: Volatility ---
        ax2.plot(
            df["date"],
            df["realized_vol"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Realized Volatility (pre-leverage)",
        )

        ax2.set_ylabel("Annualized Volatility")
        ax2.set_title("Portfolio Volatility at Each Rebalance (Pre-Leverage)")
        ax2.grid(True, which="both")
        ax2.legend()

        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # --- X-axis formatting ---
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        # --- Layout ---
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        DEFAULT_STYLE.apply_mpl(fig)
        return fig

    def plot_monte_carlo_results(self, mc_results: MonteCarloResults):
        """
        Visualizes Monte Carlo simulation results using matplotlib (2x2 layout).

        Panels:
            1. Simulated equity curve fans + actual
            2. Simulated drawdown fans + actual
            3. Histogram of maximum drawdowns (log frequency)
            4. Histogram of average drawdowns (log frequency)
        """
        equity_p05 = np.percentile(mc_results.simulated_equity, 5, axis=0)
        equity_p25 = np.percentile(mc_results.simulated_equity, 25, axis=0)
        equity_p50 = np.percentile(mc_results.simulated_equity, 50, axis=0)
        equity_p75 = np.percentile(mc_results.simulated_equity, 75, axis=0)
        equity_p95 = np.percentile(mc_results.simulated_equity, 95, axis=0)

        # Vectorized drawdown computation
        simulated_dd_curves = np.zeros_like(mc_results.simulated_equity)
        for i in range(mc_results.simulated_equity.shape[0]):
            equity_series = mc_results.simulated_equity[i]
            peak = np.maximum.accumulate(equity_series)
            simulated_dd_curves[i] = (equity_series / peak - 1.0) * 100.0

        dd_p05 = np.percentile(simulated_dd_curves, 5, axis=0)
        dd_p25 = np.percentile(simulated_dd_curves, 25, axis=0)
        dd_p50 = np.percentile(simulated_dd_curves, 50, axis=0)
        dd_p75 = np.percentile(simulated_dd_curves, 75, axis=0)
        dd_p95 = np.percentile(simulated_dd_curves, 95, axis=0)

        actual_dd = self._calculate_drawdown(mc_results.actual_equity).values
        dates = mc_results.dates

        # Figure layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_eq, ax_dd = axes[0]
        ax_maxdd, ax_avgdd = axes[1]

        # Panel 1: Equity curves
        ax_eq.fill_between(dates, equity_p05, equity_p95, alpha=0.15, label="5–95%", color=DEFAULT_STYLE.accent1)
        ax_eq.fill_between(dates, equity_p25, equity_p75, alpha=0.30, label="25–75%", color=DEFAULT_STYLE.accent1)

        ax_eq.plot(dates, equity_p50, linestyle="--", linewidth=1, label="Median", color=DEFAULT_STYLE.paper_bgcolor)
        ax_eq.plot(
            dates, mc_results.actual_equity.values, linewidth=2.5, label="Actual Strategy", color=DEFAULT_STYLE.accent1
        )

        ax_eq.set_yscale("log")
        ax_eq.set_title("Monte Carlo Equity Curve Distribution", color=DEFAULT_STYLE.font_color, fontsize=18)
        ax_eq.set_xlabel("Date")
        ax_eq.set_ylabel("Equity ($)")
        ax_eq.legend(frameon=False)

        # Panel 2: Drawdown curves
        ax_dd.fill_between(dates, dd_p05, dd_p95, alpha=0.15, color=DEFAULT_STYLE.accent5)
        ax_dd.fill_between(dates, dd_p25, dd_p75, alpha=0.30, color=DEFAULT_STYLE.accent5)

        ax_dd.plot(dates, dd_p50, linestyle="--", linewidth=1, color="#ba2b2b")
        ax_dd.plot(dates, actual_dd, linewidth=2.5, color=DEFAULT_STYLE.accent4)

        ax_dd.set_title("Monte Carlo Drawdown Distribution", color=DEFAULT_STYLE.font_color, fontsize=25)
        ax_dd.set_xlabel("Date")
        ax_dd.set_ylabel("Drawdown (%)")

        # Panel 3: Maximum drawdown histogram
        ax_maxdd.hist(mc_results.simulated_max_dd, bins=50, log=True, alpha=0.85, color=DEFAULT_STYLE.accent4)

        ax_maxdd.axvline(
            mc_results.actual_max_dd,
            linestyle="--",
            linewidth=2,
            color=DEFAULT_STYLE.paper_bgcolor,
        )

        ax_maxdd.text(
            0.98,
            0.95,
            f"Actual: {mc_results.actual_max_dd:.2f}%\nPercentile: {mc_results.actual_max_dd_percentile:.1f}",
            transform=ax_maxdd.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.9),
            backgroundcolor=DEFAULT_STYLE.accent2,
            color=DEFAULT_STYLE.font_color,
            fontsize=12,
        )

        ax_maxdd.set_title("Maximum Drawdown Distribution")
        ax_maxdd.set_xlabel("Maximum Drawdown (%)")
        ax_maxdd.set_ylabel("Frequency")

        # Panel 4: Average drawdown histogram
        ax_avgdd.hist(
            mc_results.simulated_avg_dd,
            bins=50,
            log=True,
            alpha=0.85,
            color=DEFAULT_STYLE.accent2,
        )

        ax_avgdd.axvline(
            mc_results.actual_avg_dd,
            linestyle="--",
            linewidth=2,
            color=DEFAULT_STYLE.paper_bgcolor,
        )

        ax_avgdd.text(
            0.98,
            0.95,
            f"Actual: {mc_results.actual_avg_dd:.2f}%\nPercentile: {mc_results.actual_avg_dd_percentile:.1f}",
            transform=ax_avgdd.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.9),
            backgroundcolor=DEFAULT_STYLE.accent2,
            color=DEFAULT_STYLE.font_color,
            fontsize=12,
        )

        ax_avgdd.set_title("Average Drawdown Distribution")
        ax_avgdd.set_xlabel("Average Drawdown (%)")
        ax_avgdd.set_ylabel("Frequency")

        # Final layout
        fig.suptitle("Monte Carlo Drawdown Analysis", fontsize=20, color=DEFAULT_STYLE.font_color)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        DEFAULT_STYLE.apply_mpl(fig)
        return fig
