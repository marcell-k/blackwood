from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backtesting import Backtest
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from scipy.stats import t
from sklearn.metrics import r2_score

from blackwood.config import ANNUAL_TRADING_DAYS, CASH, RISK_FREE_RATE
from blackwood.evaluation.utils import (
    BinnedIndicatorAnalyzer,
    BinStrategy,
    PerformanceAnalyzer,
    ZoneMetrics,
)
from blackwood.visualization.core import BacktestVisualizer, ChartConfig
from blackwood.visualization.style import DEFAULT_STYLE, PlotStyle


class PerformanceTablePrinter:
    KEY_METRICS: frozenset[str] = frozenset(
        {
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "# Trades",
            "Profit Factor",
            "Win Rate [%]",
            "Return (Ann.) [%]",
            "Volatility (Ann.) [%]",
            "Max. Drawdown [%]",
            "Max. Drawdown Duration",
            "Ulcer Index",
            "Avg RRR",
        }
    )

    @staticmethod
    def print_report(
        strategy_names: tuple[str, ...],
        metrics_list: list[dict[str, Any] | pd.Series],
        title: str = "PERFORMANCE ANALYSIS",
        important_only: bool = True,
    ) -> None:
        # Normalize and filter metrics: convert Series to dict, remove internal (_) keys
        processed_metrics_list: list[dict[str, Any]] = []
        for raw in metrics_list:
            if isinstance(raw, pd.Series):
                metrics_dict = raw.to_dict()
            elif isinstance(raw, dict):
                metrics_dict = raw
            else:
                continue

            filtered = {k: v for k, v in metrics_dict.items() if isinstance(k, str) and not k.startswith("_")}
            if filtered:
                processed_metrics_list.append(filtered)

        if not processed_metrics_list:
            print("No valid metrics available to display.")
            return

        n_strategies = len(strategy_names)
        headers = [name.upper() for name in strategy_names]

        # Collect metrics in order of first appearance
        all_metrics = []
        seen = set()
        for metrics in processed_metrics_list:
            for key in metrics:
                if key not in seen and (not important_only or key in PerformanceTablePrinter.KEY_METRICS):
                    all_metrics.append(key)
                    seen.add(key)

        # Dynamic column width based on formatted values
        lengths = []
        for metric in all_metrics:
            for metrics in processed_metrics_list:
                val = metrics.get(metric, "N/A")
                if isinstance(val, timedelta):
                    s = f"{val.days} days"
                elif isinstance(val, (float, np.floating)):
                    s = f"{val:.3f}" if any(tok in metric for tok in ("Ratio", "RRR", "Beta")) else f"{val:.2f}"
                else:
                    s = str(val)
                lengths.append(len(s))
        col_width = max(12, min(18, max(lengths or [0]) + 2))
        metric_width = 32
        total_width = metric_width + n_strategies * (col_width + 3) + 3

        print("\n" + "=" * total_width)
        print(title.upper().center(total_width))
        print("=" * total_width)

        print(f"{'Metric':<{metric_width}}" + " | ".join(f"{h:^{col_width}}" for h in headers))
        print("-" * total_width)

        for metric in all_metrics:
            row = f"{metric:<{metric_width}}"
            for metrics in processed_metrics_list:
                val = metrics.get(metric, "N/A")
                if isinstance(val, timedelta):
                    s = f"{val.days} days"
                elif isinstance(val, (float, np.floating)):
                    s = f"{val:.3f}" if any(tok in metric for tok in ("Ratio", "RRR", "Beta")) else f"{val:.2f}"
                elif isinstance(val, int):
                    s = f"{val}d"
                else:
                    s = str(val)
                row += f" | {s:>{col_width}}"
            print(row)

        print("=" * total_width + "\n")


class RegimePerformanceAnalyzer:
    def __init__(self, df: pd.DataFrame, regime_column: str = "Entry_regime", indicator_name: str = "Regime") -> None:
        if regime_column not in df.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")

        def regime_grouping_strategy(trades_df: pd.DataFrame) -> pd.Series:
            return trades_df[regime_column].astype(int)

        self._analyzer = PerformanceAnalyzer(
            df=df, grouping_strategy=regime_grouping_strategy, indicator_name=indicator_name
        )

        self.unique_regimes = sorted(df[regime_column].dropna().unique().astype(int))
        self.df = self._analyzer.df
        self.regime_column = regime_column
        self.indicator_name = indicator_name

    def run_analysis(self) -> dict[str, Any]:
        return self._analyzer.run_analysis()


@dataclass
class TimeAnalyzer:
    trades: pd.DataFrame
    type: str
    time_interval: int = 30

    def __post_init__(self) -> None:
        df = self.trades.copy()
        df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=False, errors="coerce")

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["DayOfWeek"] = df["EntryTime"].dt.day_name()
        df["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=weekday_order, ordered=True)
        present_days = df["DayOfWeek"].dropna().unique()
        filtered_order = [d for d in weekday_order if d not in ["Saturday", "Sunday"] or d in present_days]
        df["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=filtered_order, ordered=True)

        if self.type == "day":
            self._analyzer = PerformanceAnalyzer(df, grouping_strategy=lambda t: t["DayOfWeek"], indicator_name="Day")
        elif self.type == "time":
            self._analyzer = PerformanceAnalyzer(
                df,
                grouping_strategy=lambda t: t["EntryTime"].dt.floor(f"{self.time_interval}min").dt.time,
                indicator_name="Time",
            )

    def run_analysis(self, trade_direction: str) -> dict[str, dict[str, dict[str, ZoneMetrics]]]:
        return self._analyzer.run_analysis(trade_direction=trade_direction)


class EquityAnalyzer:
    CHART_CONFIGS: dict[str, ChartConfig] | None = None

    @staticmethod
    def _initialize_configs(plot_style: PlotStyle) -> dict[str, ChartConfig]:
        return {
            "rrr": ChartConfig(
                primary_color=plot_style.accent1,
                secondary_color=plot_style.accent4,
                title="<b>Cumulative Sum of RiskRewardRatio Over Trades</b>",
                yaxis_title="Cumulative RiskRewardRatio",
                yaxis_type="linear",
                height=600,
                trace_mode="lines+markers",
                trace_width=2,
                marker_size=3,
                show_hline_zero=True,
            ),
            "equity": ChartConfig(
                primary_color=plot_style.accent3,
                title="<b>Trading Strategy Equity Curve (Log Scale)</b>",
                yaxis_title="Equity (Log Scale)",
                yaxis_type="log",
                height=600,
                trace_mode="lines",
                trace_width=3,
                show_hline_zero=False,
            ),
            "drawdown": ChartConfig(
                primary_color=plot_style.accent2,
                secondary_color=plot_style.accent3,
                title="<b>Equity and Drawdown Analysis</b>",
                yaxis_title="Drawdown (%)",
                yaxis_type="linear",
                height=800,
                trace_mode="lines",
                trace_width=2,
                show_hline_zero=False,
            ),
        }

    @staticmethod
    def _get_default_axis_style(plot_style: PlotStyle) -> dict[str, Any]:
        return {
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": plot_style.grid,
            "tickfont": {"color": plot_style.font_color},
            "title_font": {"color": plot_style.font_color},
        }

    @staticmethod
    def _create_figure_from_config(
        chart_type: str,
        traces: list[go.Scatter],
        plot_style: PlotStyle,
        custom_layout: dict[str, Any] | None = None,
        xaxis_title: str = "Trade Number",
    ) -> go.Figure:
        if EquityAnalyzer.CHART_CONFIGS is None:
            EquityAnalyzer.CHART_CONFIGS = EquityAnalyzer._initialize_configs(plot_style)

        config = EquityAnalyzer.CHART_CONFIGS[chart_type]
        axis_style = EquityAnalyzer._get_default_axis_style(plot_style)

        fig = go.Figure(data=traces)

        base_layout = {
            "title": {"text": config.title, "x": 0.5, "font": {"color": plot_style.font_color, "size": 16}},
            "xaxis_title": xaxis_title,
            "yaxis_title": config.yaxis_title,
            "yaxis_type": config.yaxis_type,
            "width": 1000,
            "height": config.height,
            "hovermode": "x unified",
            "font": {"color": plot_style.font_color},
            "legend": {"font": {"color": plot_style.font_color}},
        }

        if config.layout_overrides:
            base_layout.update(config.layout_overrides)
        if custom_layout:
            base_layout.update(custom_layout)

        fig.update_layout(**base_layout)
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

        if config.show_hline_zero:
            fig.add_hline(y=0, line_dash="dash", line_color=plot_style.accent2, opacity=0.7)

        plot_style.apply(fig)
        return fig

    @staticmethod
    def plot_and_analyze_risk_reward_cumsum(
        df: pd.DataFrame,
        rrr_col: str = "RiskRewardRatio",
    ) -> go.Figure | None:
        plot_style = DEFAULT_STYLE
        df_sorted = df.sort_values(by="EntryTime").reset_index(drop=True)
        cumsum_rr = np.asarray(df_sorted[rrr_col].cumsum(), dtype=np.float64)
        trade_numbers = np.arange(1, len(cumsum_rr) + 1)

        coeffs = np.polyfit(trade_numbers, cumsum_rr, deg=1)
        y_fit = np.polyval(coeffs, trade_numbers)
        global_r2 = r2_score(cumsum_rr, y_fit)

        config = EquityAnalyzer._initialize_configs(plot_style)["rrr"]

        traces = [
            go.Scatter(
                x=trade_numbers,
                y=cumsum_rr,
                mode=config.trace_mode,
                name="Cumulative RiskRewardRatio",
                line={"color": config.primary_color, "width": config.trace_width},
                marker={"size": config.marker_size, "color": config.primary_color},
            ),
            go.Scatter(
                x=trade_numbers,
                y=y_fit,
                mode="lines",
                name=f"Trend (R²={global_r2:.3f})",
                line={"color": config.secondary_color, "width": 2, "dash": "dash"},
                opacity=0.7,
            ),
        ]

        return EquityAnalyzer._create_figure_from_config("rrr", traces, plot_style)

    @staticmethod
    def plot_and_analyze_equity(df: pd.DataFrame) -> go.Figure | None:
        plot_style = DEFAULT_STYLE
        if df.empty:
            return None

        df = df.copy()
        df["EntryBar"] = pd.to_datetime(df["EntryBar"])
        return_pct = df.groupby("EntryBar")["ReturnPct"].sum()
        if return_pct.empty:
            return None

        equity = np.concatenate([[1.0], np.asarray((1 + return_pct).cumprod(), dtype=np.float64)])
        trade_numbers = np.arange(len(equity))

        config = EquityAnalyzer._initialize_configs(plot_style)["equity"]
        traces = [
            go.Scatter(
                x=trade_numbers,
                y=equity,
                mode=config.trace_mode,
                name="Equity Curve",
                line={"color": config.primary_color, "width": config.trace_width},
                hovertemplate="<b>Trade:</b> %{x}<br><b>Equity:</b> %{y:.4f}<extra></extra>",
            )
        ]

        return EquityAnalyzer._create_figure_from_config("equity", traces, plot_style)

    @staticmethod
    def plot_drawdown_analysis(df: pd.DataFrame) -> go.Figure | None:
        plot_style = DEFAULT_STYLE
        if df.empty:
            return None

        df = df.copy()
        df["EntryBar"] = pd.to_datetime(df["EntryBar"])
        return_pct = df.groupby("EntryBar")["ReturnPct"].sum()
        if return_pct.empty:
            return None

        equity = np.concatenate([[1.0], np.asarray((1 + return_pct).cumprod(), dtype=np.float64)])
        trade_numbers = np.arange(len(equity))
        running_max = np.maximum.accumulate(equity)
        drawdown_pct = ((equity - running_max) / running_max) * 100

        config = EquityAnalyzer._initialize_configs(plot_style)["drawdown"]
        try:
            r, g, b = (
                int(config.primary_color[1:3], 16),
                int(config.primary_color[3:5], 16),
                int(config.primary_color[5:7], 16),
            )
            fillcolor = f"rgba({r}, {g}, {b}, 0.3)"
        except Exception:
            fillcolor = "rgba(255, 100, 100, 0.3)"

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Equity Curve", "Drawdown (%)"],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
        )

        fig.add_trace(
            go.Scatter(
                x=trade_numbers,
                y=equity,
                mode="lines",
                name="Equity",
                line={"color": config.secondary_color, "width": 2},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=trade_numbers,
                y=drawdown_pct,
                mode="lines",
                name="Drawdown",
                line={"color": config.primary_color, "width": 2},
                fill="tozeroy",
                fillcolor=fillcolor,
            ),
            row=2,
            col=1,
        )

        axis_style = EquityAnalyzer._get_default_axis_style(plot_style)
        fig.update_layout(
            title={"text": config.title, "x": 0.5, "font": {"color": plot_style.font_color, "size": 16}},
            height=config.height,
            showlegend=False,
            font={"color": plot_style.font_color},
        )
        fig.update_xaxes(title_text="Trade Number", row=2, col=1, **axis_style)
        fig.update_yaxes(title_text="Equity", row=1, col=1, **axis_style)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, **axis_style)
        plot_style.apply(fig)
        return fig


@dataclass
class TradingSequenceAnalyzer:
    """Visualizes individual trade Risk-Reward ratios as a heatmap and analyzes losing streaks."""

    trades_df: pd.DataFrame
    trades_per_row: int = 100
    rr_column: str = "RiskRewardRatio"
    rr_min: float = -1.1
    rr_center: float = 0.0
    rr_max: float = 2.5

    def __post_init__(self) -> None:
        self.rr_values: np.ndarray = np.asarray(self.trades_df[self.rr_column], dtype=np.float64).copy()
        self.style = DEFAULT_STYLE
        self._colormap_cache: LinearSegmentedColormap | None = None
        self.rr_matrix = self._create_rr_matrix()
        self.streak_stats = self._compute_losing_streaks()

    def _create_rr_matrix(self) -> np.ndarray:
        n_trades = len(self.rr_values)
        n_rows = int(np.ceil(n_trades / self.trades_per_row))
        total_cells = n_rows * self.trades_per_row
        padded = (
            np.pad(self.rr_values, (0, total_cells - n_trades), constant_values=np.nan)
            if n_trades < total_cells
            else self.rr_values
        )
        return padded.reshape(n_rows, self.trades_per_row)

    def _compute_losing_streaks(self) -> dict[str, Any]:
        rr = self.rr_values
        if len(rr) == 0:
            return {
                "max_losing_streak": 0,
                "avg_losing_streak": 0.0,
                "total_losing_streaks": 0,
                "losing_streaks": [0],
                "median_losing_streak": 0.0,
                "p75_losing_streak": 0.0,
                "p95_losing_streak": 0.0,
            }

        is_loss = rr < 0
        transitions = np.concatenate([[False], is_loss[:-1]]) != is_loss
        group_ids = np.cumsum(transitions)
        losing_groups = group_ids[is_loss]

        if len(losing_groups) == 0:
            return {
                "max_losing_streak": 0,
                "avg_losing_streak": 0.0,
                "total_losing_streaks": 0,
                "losing_streaks": [0],
                "median_losing_streak": 0.0,
                "p75_losing_streak": 0.0,
                "p95_losing_streak": 0.0,
            }

        _, counts = np.unique(losing_groups, return_counts=True)
        return {
            "losing_streaks": counts.tolist(),
            "max_losing_streak": int(counts.max()),
            "avg_losing_streak": float(counts.mean()),
            "total_losing_streaks": len(counts),
            "median_losing_streak": float(np.median(counts)),
            "p75_losing_streak": float(np.percentile(counts, 75)),
            "p95_losing_streak": float(np.percentile(counts, 95)),
        }

    def _get_cached_colormap(self) -> LinearSegmentedColormap:
        if self._colormap_cache is None:
            colors = ["#8B0000", self.style.accent2, self.style.accent6, self.style.accent3, "#006400"]
            self._colormap_cache = LinearSegmentedColormap.from_list("RR_Diverging", colors, N=256)
            self._colormap_cache.set_bad(color=self.style.muted, alpha=0.3)
        return self._colormap_cache

    def create_matrix_heatmap(
        self, figsize: tuple[float, float] = (18, 10), show_values: bool = False, cell_fontsize: int = 7
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        cmap = self._get_cached_colormap()
        norm = TwoSlopeNorm(vmin=self.rr_min, vcenter=self.rr_center, vmax=self.rr_max)

        ax.imshow(self.rr_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

        ax.set_xticks(np.arange(self.rr_matrix.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.rr_matrix.shape[0]) - 0.5, minor=True)
        ax.grid(which="minor", color=self.style.grid, linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        if show_values:
            for i in range(self.rr_matrix.shape[0]):
                for j in range(self.rr_matrix.shape[1]):
                    val = self.rr_matrix[i, j]
                    if not np.isnan(val):
                        color = self.style.font_color if abs(val) > 1.0 else self.style.plot_bgcolor
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=cell_fontsize,
                            fontweight="bold",
                        )

        cbar = plt.colorbar(ax.imshow(self.rr_matrix, cmap=cmap, norm=norm), fraction=0.046, pad=0.04)
        cbar.set_label("Risk-Reward Ratio per Trade", rotation=270, labelpad=20, fontsize=12, fontweight="bold")

        rows, cols = self.rr_matrix.shape
        ax.set_xticks(np.arange(0, cols, 5))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(1, cols + 1, 5))
        ax.set_yticklabels(np.arange(1, rows + 1))
        ax.set_xlabel(f"Trade Position (within row of {self.trades_per_row})", fontweight="bold")
        ax.set_ylabel("Row Number", fontweight="bold")
        ax.set_title("Trading Sequence RR Heatmap (Individual Trades)", fontweight="bold", pad=20)

        self.style.apply_mpl(fig, ax)
        cbar.ax.tick_params(labelsize=self.style.font_size)
        for label in cbar.ax.get_yticklabels():
            label.set_color(self.style.font_color)

        plt.tight_layout()
        return fig

    def create_losing_streak_histogram(self, figsize: tuple[float, float] = (12, 6), bins: int = 20) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        streaks = self.streak_stats["losing_streaks"]
        actual_bins = min(bins, len(set(streaks))) if streaks else 1

        ax.hist(
            streaks, bins=actual_bins, color=self.style.accent2, edgecolor=self.style.accent4, linewidth=1.5, alpha=0.7
        )

        mean = self.streak_stats["avg_losing_streak"]
        max_streak = self.streak_stats["max_losing_streak"]
        median = self.streak_stats["median_losing_streak"]

        ax.axvline(mean, color=self.style.accent1, linestyle="--", linewidth=2.5, label=f"Mean: {mean:.1f}", alpha=0.9)
        ax.axvline(
            median, color=self.style.accent6, linestyle="-.", linewidth=2.5, label=f"Median: {median:.0f}", alpha=0.9
        )

        ax.set_xlabel("Consecutive Losing Trades (RR < 0)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(
            f"Losing Streak Distribution\nTotal Streaks: {self.streak_stats['total_losing_streaks']} | "
            f"Longest: {max_streak}",
            fontweight="bold",
            pad=20,
        )

        legend = ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        for text in legend.get_texts():
            text.set_color(self.style.font_color)

        self.style.apply_mpl(fig, ax)
        plt.tight_layout()
        return fig

    def print_detailed_statistics(self) -> None:
        print(f"\n{'=' * 70}")
        print("TRADING SEQUENCE ANALYSIS - DETAILED STATISTICS")
        print(f"{'=' * 70}")
        print(f"\nMaximum Losing Streak: {self.streak_stats['max_losing_streak']} trades")
        print(f"Average Losing Streak: {self.streak_stats['avg_losing_streak']:.1f} trades")
        print(f"Median Losing Streak: {self.streak_stats['median_losing_streak']:.0f} trades")
        print(f"75th Percentile: {self.streak_stats['p75_losing_streak']:.0f} trades")
        print(f"95th Percentile: {self.streak_stats['p95_losing_streak']:.0f} trades")
        print(f"Total Losing Streaks: {self.streak_stats['total_losing_streaks']}")

    def run_complete_analysis(self, show_matrix: bool = True, show_histogram: bool = True) -> tuple[Figure, Figure]:
        self.print_detailed_statistics()
        heatmap_fig = self.create_matrix_heatmap()
        histogram_fig = self.create_losing_streak_histogram()

        if show_matrix:
            plt.figure(heatmap_fig.number)
            plt.show()
        if show_histogram:
            plt.figure(histogram_fig.number)
            plt.show()

        return heatmap_fig, histogram_fig


class StrategyVsBuyHoldAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        stats: pd.Series | dict[str, Any],
        strategy_name: str = "Strategy",
        show_drawdown: bool = True,
    ) -> None:
        self.stats = stats
        self.data = df
        self.bh_stats = self.get_bh_stats()
        self.strategy_name = strategy_name
        self.show_drawdown = show_drawdown
        self.style = DEFAULT_STYLE

    def get_bh_stats(self) -> pd.Series:
        from blackwood.strategies.base import BuyAndHoldStrategy

        bt = Backtest(
            self.data,
            BuyAndHoldStrategy,
            cash=CASH,
            spread=0,
            commission=0,
            trade_on_close=True,
            exclusive_orders=False,
            finalize_trades=True,
        )
        return bt.run()

    def _extract_equity_dict(self) -> dict[str, pd.Series]:
        strategy_equity = (
            self.stats["_equity_curve"]["Equity"].resample("D").last().dropna()
            if isinstance(self.stats, (pd.Series, dict)) and "_equity_curve" in self.stats
            else pd.Series()
        )
        bh_equity = self.bh_stats["_equity_curve"]["Equity"].resample("D").last().dropna()
        return {self.strategy_name: strategy_equity, "Buy & Hold": bh_equity}

    def print_summary(self) -> None:
        PerformanceTablePrinter.print_report(
            strategy_names=(self.strategy_name, "Buy & Hold"),
            metrics_list=[self.stats, self.bh_stats],
            title=f"{self.strategy_name.upper()} VS BUY & HOLD COMPARISON",
            important_only=True,
        )

    def create_plot(self) -> go.Figure:
        equity_dict = self._extract_equity_dict()
        visualizer = BacktestVisualizer(equity_dict=equity_dict, create_portfolios=False)
        fig = visualizer.plot_equity_curves(equity_type="all", show_drawdown=self.show_drawdown, interpolate=True)
        fig.update_layout(
            title=dict(text=f"<b>Performance Analysis: {self.strategy_name} vs Buy & Hold</b>", x=0.5, xanchor="center")
        )
        return fig

    def analyze_and_display(self, print_summary: bool = True, show_plot: bool = True) -> go.Figure | None:
        if print_summary:
            self.print_summary()
        if show_plot:
            fig = self.create_plot()
            fig.show()
            return fig
        return None

    def show_plot(self) -> None:
        self.create_plot().show()


@dataclass
class DailyNmbTradeAnalyzer:
    trades: pd.DataFrame
    max_plots: int = 4
    early_late_split: int = 1

    def __post_init__(self) -> None:
        self.trades_df = self.trades.copy()
        self.trades_df["EntryDate"] = self.trades_df["EntryTime"].dt.date
        self.trades_df = self.trades_df.sort_values(["EntryDate", "EntryTime"])
        self.trades_df["TradeOrder"] = self.trades_df.groupby("EntryDate").cumcount() + 1

        self.max_trades_per_day = int(self.trades_df["TradeOrder"].max())
        self.plot_trades = min(self.max_trades_per_day, self.max_plots)
        self.grouped = self.trades_df.groupby("TradeOrder")

        self.style = DEFAULT_STYLE
        print(f"Maximum trades per day: {self.max_trades_per_day}")

    def _create_distribution_plots(self) -> None:
        cols = min(2, self.plot_trades)
        rows = int(np.ceil(self.plot_trades / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
        axes = np.atleast_2d(axes).flatten()

        colors = [
            self.style.accent1,
            self.style.accent2,
            self.style.accent3,
            self.style.accent4,
            self.style.accent5,
            self.style.accent6,
        ]

        for i, order in enumerate(range(1, self.plot_trades + 1)):
            ax = axes[i]
            try:
                group = self.grouped.get_group(order)
            except KeyError:
                ax.axis("off")
                continue

            returns = group["ReturnPct"].values * 100
            rr_finite = group["RiskRewardRatio"][np.isfinite(group["RiskRewardRatio"])]

            if len(returns) == 0:
                ax.axis("off")
                continue

            n_bins = min(25, max(8, len(returns) // 3))
            ax.hist(
                returns,
                bins=n_bins,
                color=colors[i % len(colors)],
                alpha=0.75,
                edgecolor=self.style.line,
                linewidth=0.8,
            )

            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            win_rate = np.mean(group["ReturnPct"] > 0) * 100
            avg_rr = np.mean(rr_finite) if len(rr_finite) > 0 else np.nan
            med_rr = np.median(rr_finite) if len(rr_finite) > 0 else np.nan

            annotation = [f"Mean: {mean_ret:.2f}%", f"Std: {std_ret:.2f}%", f"Win Rate: {win_rate:.1f}%"]
            if not np.isnan(avg_rr):
                annotation.append(f"Avg R/R: {avg_rr:.2f}")
                if not np.isnan(med_rr):
                    annotation.append(f"Med R/R: {med_rr:.2f}")

            ax.text(
                0.02,
                0.98,
                "\n".join(annotation),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=self.style.plot_bgcolor, edgecolor=self.style.line, alpha=0.9),
                color=self.style.font_color,
            )

            ax.set_xlabel("Return %", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"Order {order} (n={len(returns)})", fontsize=11)

        for j in range(self.plot_trades, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"ReturnPct Distribution by Trade Order (First {self.plot_trades} Orders)",
            fontsize=14,
            y=0.995,
            color=self.style.font_color,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        self.style.apply_mpl(fig)
        plt.show()

    def _generate_summary_statistics(self) -> pd.DataFrame:
        if self.grouped.ngroups == 0:
            return pd.DataFrame()

        summary = self.grouped.agg(
            Count=("ReturnPct", "count"),
            **{"Mean Return %": ("ReturnPct", lambda x: float(np.mean(np.asarray(x, dtype=np.float64))) * 100)},
            **{"Std Dev %": ("ReturnPct", lambda x: float(np.std(np.asarray(x, dtype=np.float64))) * 100)},
            **{"Win Rate %": ("ReturnPct", lambda x: float(np.mean(np.asarray(x, dtype=np.float64) > 0)) * 100)},
            **{"Best %": ("ReturnPct", lambda x: float(np.max(np.asarray(x, dtype=np.float64))) * 100)},
            **{"Worst %": ("ReturnPct", lambda x: float(np.min(np.asarray(x, dtype=np.float64))) * 100)},
            **{"Median %": ("ReturnPct", lambda x: float(np.median(np.asarray(x, dtype=np.float64))) * 100)},
        )

        summary = summary[summary["Count"] > 0]
        summary = summary.head(min(25, self.max_trades_per_day))
        summary.reset_index(inplace=True)
        summary["TradeOrder"] = summary["TradeOrder"].astype(int)
        return summary.sort_values("TradeOrder")

    def _print_comprehensive_analysis(self, summary_df: pd.DataFrame) -> None:
        print("\n" + "=" * 90)
        print("COMPREHENSIVE TRADE ORDER ANALYSIS WITH RISK/REWARD METRICS")
        print("=" * 90)
        display_cols = [
            "TradeOrder",
            "Count",
            "Mean Return %",
            "Std Dev %",
            "Win Rate %",
            "Best %",
            "Worst %",
            "Median %",
            "Avg R/R",
            "Med R/R",
            "R/R Count",
        ]
        print(summary_df[display_cols].round(2).to_string(index=False))

        if len(summary_df) >= self.early_late_split:
            early = summary_df.head(self.early_late_split)
            early_mean = early["Mean Return %"].mean()
            early_win = early["Win Rate %"].mean()
            early_rr = early["Avg R/R"].mean()

            print(
                f"\nEARLY TRADES (1-{self.early_late_split}): "
                f"Avg Return {early_mean:.2f}%, Win Rate {early_win:.1f}%"
                f"{f', Avg R/R {early_rr:.2f}' if not np.isnan(early_rr) else ''}"
            )

            if len(summary_df) >= self.early_late_split * 2:
                later = summary_df.iloc[self.early_late_split : self.early_late_split * 2]
                later_mean = later["Mean Return %"].mean()
                later_win = later["Win Rate %"].mean()
                later_rr = later["Avg R/R"].mean()

                print(
                    f"LATER TRADES ({self.early_late_split + 1}-{self.early_late_split * 2}): "
                    f"Avg Return {later_mean:.2f}%, Win Rate {later_win:.1f}%"
                    f"{f', Avg R/R {later_rr:.2f}' if not np.isnan(later_rr) else ''}"
                )

                ret_change = ((later_mean - early_mean) / abs(early_mean) * 100) if early_mean != 0 else 0
                win_change = later_win - early_win
                print(
                    f"\nChange (Early → Later): Return {ret_change:+.1f}%, Win Rate {win_change:+.1f} pts"
                    f"{f', R/R {later_rr - early_rr:+.2f}' if not np.isnan(early_rr + later_rr) else ''}"
                )

    def run_full_analysis(self) -> pd.DataFrame:
        self._create_distribution_plots()
        summary_df = self._generate_summary_statistics()
        self._print_comprehensive_analysis(summary_df)
        return summary_df


class TradingPerformanceAnalyzer:
    def __init__(self, equity_data: pd.Series | pd.DataFrame, trades: pd.DataFrame | None = None) -> None:
        self.equity_curve = equity_data["Equity"].copy()
        self.full_equity_data = equity_data.copy()
        self.trades = trades.copy() if trades is not None else None
        self.style = DEFAULT_STYLE
        self._process_returns()

    def _process_returns(self) -> None:
        self.monthly_data = self.equity_curve.resample("ME").last().to_frame("Equity")
        self.monthly_data["MonthlyReturn"] = self.monthly_data["Equity"].pct_change()
        self.monthly_data["MonthlyReturn_Pct"] = self.monthly_data["MonthlyReturn"] * 100

        self.yearly_data = self.monthly_data.resample("YE").agg({"Equity": "last"})
        self.yearly_data["YearlyReturn"] = self.yearly_data["Equity"].pct_change()
        self.yearly_data["YearlyReturn_Pct"] = self.yearly_data["YearlyReturn"] * 100

        monthly_returns = self.monthly_data["MonthlyReturn"].dropna()
        if len(monthly_returns) > 0:
            self.yearly_compound_returns_pct = (
                monthly_returns.groupby(pd.DatetimeIndex(monthly_returns.index).year).apply(
                    lambda x: (1 + x).prod() - 1
                )
                * 100
            )
        else:
            self.yearly_compound_returns_pct = pd.Series(dtype=float)

    def _calculate_comprehensive_metrics(self) -> dict[str, Any]:
        monthly = self.monthly_data["MonthlyReturn"].dropna()
        if len(monthly) == 0:
            return {
                "monthly_return_pct": 0.0,
                "yearly_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "win_rate_pct": 0.0,
                "total_months": 0,
                "total_trades": 0,
            }

        mean_monthly = monthly.mean()
        std_monthly = monthly.std(ddof=1)
        expected_yearly = mean_monthly * 12
        annualized_std = std_monthly * np.sqrt(12)

        n = len(monthly)
        if n >= 2 and std_monthly > 0:
            margin = t.ppf(0.975, n - 1) * (std_monthly / np.sqrt(n)) * np.sqrt(12)
            conf_lower, conf_upper = expected_yearly - margin, expected_yearly + margin
        else:
            conf_lower, conf_upper = np.nan, np.nan

        downside_std = monthly[monthly < 0].std(ddof=1) * np.sqrt(12) if len(monthly[monthly < 0]) >= 2 else np.nan

        rolling_max = self.equity_curve.expanding().max()
        max_dd = abs(((self.equity_curve / rolling_max) - 1).min()) * 100 if len(self.equity_curve) > 0 else 0.0

        sharpe = expected_yearly / annualized_std if annualized_std > 0 else np.nan
        sortino = expected_yearly / downside_std if pd.notna(downside_std) and downside_std > 0 else np.nan
        win_rate = (monthly > 0).mean() * 100

        return {
            "monthly_return_pct": mean_monthly * 100,
            "yearly_return_pct": expected_yearly * 100,
            "monthly_std_pct": std_monthly * 100,
            "yearly_std_pct": annualized_std * 100,
            "conf_lower_pct": conf_lower * 100 if pd.notna(conf_lower) else np.nan,
            "conf_upper_pct": conf_upper * 100 if pd.notna(conf_upper) else np.nan,
            "downside_std_pct": downside_std * 100 if pd.notna(downside_std) else np.nan,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate_pct": win_rate,
            "total_months": len(monthly),
            "total_trades": len(self.trades) if self.trades is not None else 0,
            "total_return_pct": ((self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1) * 100
            if len(self.equity_curve) > 1
            else 0,
        }

    def build_report_text(self) -> str:
        m = self._calculate_comprehensive_metrics()
        LABEL_WIDTH = 25
        VALUE_WIDTH = 20
        lines = ["=" * 45, "STRATEGY PERFORMANCE REPORT", "=" * 45]

        sections = {
            "RETURN ANALYSIS": [
                ("Average Monthly Return:", f"{m['monthly_return_pct']:.2f}%"),
                ("Expected Yearly Return:", f"{m['yearly_return_pct']:.2f}%"),
                ("95% Confidence Interval:", f"[{m['conf_lower_pct']:.1f}%, {m['conf_upper_pct']:.1f}%]")
                if pd.notna(m["conf_lower_pct"])
                else None,
            ],
            "RISK ANALYSIS": [
                ("Monthly Volatility:", f"{m['monthly_std_pct']:.2f}%"),
                ("Annualized Volatility:", f"{m['yearly_std_pct']:.2f}%"),
                ("Downside Deviation:", f"{m['downside_std_pct']:.2f}%") if pd.notna(m["downside_std_pct"]) else None,
                ("Maximum Drawdown:", f"{m['max_drawdown_pct']:.2f}%"),
            ],
            "RISK-ADJUSTED METRICS": [
                ("Sharpe Ratio:", f"{m['sharpe_ratio']:.3f}") if pd.notna(m["sharpe_ratio"]) else None,
                ("Sortino Ratio:", f"{m['sortino_ratio']:.3f}") if pd.notna(m["sortino_ratio"]) else None,
                ("Win Rate:", f"{m['win_rate_pct']:.1f}%"),
            ],
            "TRADING STATISTICS": [
                ("Total Months Analyzed:", f"{m['total_months']}"),
                ("Total Trades:", f"{m['total_trades']}"),
            ],
        }

        for title, items in sections.items():
            lines.append(f"\n{title.center(45)}")
            lines.append("-" * 45)
            for label, value in [i for i in items if i]:
                lines.append(f"{label:<{LABEL_WIDTH}}{value:>{VALUE_WIDTH}}")

        if len(self.yearly_compound_returns_pct) > 0:
            lines.append("\nYEARLY RETURNS".center(45))
            lines.append("-" * 45)
            for year, ret in self.yearly_compound_returns_pct.items():
                lines.append(f"{year!s:<{LABEL_WIDTH}}{f'{ret:.2f}%':>{VALUE_WIDTH}}")

        return "\n".join(lines)

    def print_full_report(self) -> dict[str, Any]:
        print(self.build_report_text())
        return self._calculate_comprehensive_metrics()

    def plot_comprehensive_charts(self) -> Figure | None:
        if len(self.equity_curve) == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()

        # Equity curve
        axes[0].plot(self.equity_curve.index, self.equity_curve.values, color=self.style.accent1, linewidth=2.5)
        axes[0].fill_between(self.equity_curve.index, 0, self.equity_curve.values, color=self.style.accent1, alpha=0.2)
        axes[0].set_yscale("log")
        axes[0].set_title("Equity Curve Over Time (Log Scale)")
        axes[0].grid(True, alpha=0.3)

        # Monthly returns
        monthly_pct = self.monthly_data["MonthlyReturn_Pct"].dropna()
        if len(monthly_pct) > 0:
            colors = [self.style.accent3 if v > 0 else self.style.accent2 for v in monthly_pct.values]
            axes[1].bar(monthly_pct.index, monthly_pct.values, color=colors, width=25)
            axes[1].axhline(0, color=self.style.muted, linestyle="--", alpha=0.7)
        axes[1].set_title("Monthly Returns")
        axes[1].grid(True, alpha=0.3)

        # Yearly or distribution
        if len(self.yearly_compound_returns_pct) > 0:
            years = self.yearly_compound_returns_pct.index
            colors = [
                self.style.accent3 if v > 0 else self.style.accent2 for v in self.yearly_compound_returns_pct.values
            ]
            axes[2].bar(years, self.yearly_compound_returns_pct.values, color=colors)
            axes[2].axhline(0, color=self.style.muted, linestyle="--", alpha=0.7)
            axes[2].set_title("Yearly Returns")
        else:
            axes[2].hist(monthly_pct.values, bins=15, color=self.style.accent4, alpha=0.7)
            axes[2].set_title("Return Distribution")
        axes[2].grid(True, alpha=0.3)

        # Drawdown
        rolling_max = self.equity_curve.expanding().max()
        dd = ((self.equity_curve / rolling_max) - 1) * 100
        axes[3].plot(dd.index, dd.values, color=self.style.accent2, linewidth=2)
        axes[3].fill_between(dd.index, 0, dd.values, color=self.style.accent2, alpha=0.3)
        axes[3].axhline(0, color=self.style.muted, linestyle="--", alpha=0.7)
        axes[3].set_title("Drawdown Analysis")
        axes[3].grid(True, alpha=0.3)

        fig.suptitle("Trading Strategy Performance Dashboard", fontsize=14, color=self.style.font_color)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        self.style.apply_mpl(fig)
        return fig

    def create_detailed_drawdown_chart(self) -> Figure:
        rolling_max = self.equity_curve.expanding().max()
        drawdown = ((self.equity_curve / rolling_max) - 1) * 100

        in_dd = drawdown < -0.1
        periods = []
        start = None
        for i, val in enumerate(in_dd):
            if val and start is None:
                start = i
            elif not val and start is not None:
                period = drawdown.iloc[start:i]
                periods.append(
                    {
                        "start": drawdown.index[start],
                        "end": drawdown.index[i - 1],
                        "max_dd": period.min(),
                        "duration_days": (drawdown.index[i - 1] - drawdown.index[start]).days,
                    }
                )
                start = None
        if start is not None:
            period = drawdown.iloc[start:]
            periods.append(
                {
                    "start": drawdown.index[start],
                    "end": drawdown.index[-1],
                    "max_dd": period.min(),
                    "duration_days": (drawdown.index[-1] - drawdown.index[start]).days,
                }
            )

        fig, ax = plt.subplots(figsize=(12, 6))
        accent_rgba = (*[int(self.style.accent2[i : i + 2], 16) / 255 for i in (1, 3, 5)], 0.4)

        drawdown_arr = drawdown.to_numpy(dtype=np.float64)
        ax.fill_between(drawdown.index, 0, drawdown_arr, color=accent_rgba)
        ax.plot(drawdown.index, drawdown_arr, color=self.style.accent2, linewidth=2)

        top5 = sorted(periods, key=lambda x: x["max_dd"])[:5]
        for p in top5:
            ax.axvspan(p["start"], p["end"], color=accent_rgba, alpha=0.5)

        for i, p in enumerate(top5):
            mid = p["start"] + (p["end"] - p["start"]) / 2
            ax.annotate(
                f"#{i + 1}: {p['max_dd']:.1f}%\n{p['duration_days']}d",
                xy=(mid, p["max_dd"]),
                xytext=(0, -30),
                textcoords="offset points",
                ha="center",
                color=self.style.font_color,  # Text color
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.style.plot_bgcolor),
                arrowprops=dict(arrowstyle="->", color=self.style.font_color),
            )  # Arrow color

        ax.axhline(0, linestyle="--", color=self.style.muted, linewidth=1)
        ax.set_title("Detailed Drawdown Analysis (Underwater Plot)", fontsize=16, fontweight="bold", pad=20)
        ax.grid(True, color=self.style.grid, linewidth=0.5, alpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        self.style.apply_mpl(fig)
        plt.tight_layout()
        return fig


class TradingDashboard:
    def __init__(
        self,
        strategy_names: tuple[str, ...],
        equity_data: tuple[pd.DataFrame, ...],
        trade_data: tuple[pd.DataFrame, ...],
        ohlc_data: tuple[pd.DataFrame, ...] | None = None,
        benchmark_sharpe: float = 1.0,
        initial_capital: float = CASH,
        extrema_order: int = 3,
    ) -> None:
        self.strategy_names = strategy_names
        self.equity_data = equity_data
        self.trade_data = trade_data
        self.ohlc_data = ohlc_data
        self.initial_capital = initial_capital
        self.n_strategies = len(strategy_names)
        self.style = DEFAULT_STYLE
        self.extrema_order = extrema_order
        self.risk_free_rate = RISK_FREE_RATE
        self.annual_trading_days = ANNUAL_TRADING_DAYS
        self.benchmark_sharpe = benchmark_sharpe
        self._metrics_cache = {}

    def calculate_performance_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, Any]:
        from blackwood.metrics.core import compute_all_metrics

        return dict(
            compute_all_metrics(
                equity=equity_df["Equity"],
                risk_free_rate=self.risk_free_rate,
                annual_trading_days=self.annual_trading_days,
                pnl=np.array(trades_df["PnL"]),
                returns_pct=np.array(trades_df["ReturnPct"]),
                benchmark_sharpe=self.benchmark_sharpe,
            )
        )

    def create_dashboard(
        self,
        show_drawdown: bool = True,
        interpolate: bool = True,
        equity_type: Literal["strategies", "portfolios", "all"] = "strategies",
    ) -> tuple[dict | tuple, str]:
        from blackwood.visualization.core import BacktestVisualizer

        metrics_list = [
            self.calculate_performance_metrics(self.equity_data[i], self.trade_data[i])
            for i in range(self.n_strategies)
        ]

        self._metrics_cache = dict(zip(self.strategy_names, metrics_list, strict=True))

        equity_dict = {
            name: df["Equity"]
            for name, df in zip(self.strategy_names, self.equity_data, strict=True)
            if not df.empty and "Equity" in df.columns
        }

        visualizer = BacktestVisualizer(equity_dict=equity_dict, extrema_order=self.extrema_order)
        fig = visualizer.plot_equity_curves(
            equity_type=equity_type, show_drawdown=show_drawdown, interpolate=interpolate
        )
        fig.show()

        PerformanceTablePrinter.print_report(
            strategy_names=self.strategy_names,
            metrics_list=metrics_list,
            title="WALK FORWARD PERFORMANCE ANALYSIS",
            important_only=True,
        )

        formatted_string = "; ".join(str(v).replace(".", ",") for metrics in metrics_list for v in metrics)
        return (metrics_list[0] if self.n_strategies == 1 else tuple(metrics_list), formatted_string)

    def get_strategy_metrics(self, strategy_name: str) -> dict[str, Any]:
        if strategy_name not in self._metrics_cache:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        return self._metrics_cache[strategy_name]

    def compare_strategies(self, metric: str) -> dict[str, Any]:
        return {name: m.get(metric, 0) for name, m in self._metrics_cache.items()}

    def create_comparison_chart(self, metrics_to_compare: list[str]) -> go.Figure:
        fig = make_subplots(
            rows=len(metrics_to_compare),
            cols=1,
            subplot_titles=[f"<b>{m}</b>" for m in metrics_to_compare],
            vertical_spacing=0.1,
        )
        colors = [self.style.accent1, self.style.accent2, self.style.accent3][: self.n_strategies]

        for i, metric in enumerate(metrics_to_compare):
            values = [self._metrics_cache[n].get(metric, 0) for n in self.strategy_names]
            fig.add_trace(
                go.Bar(x=list(self.strategy_names), y=values, marker=dict(color=colors, opacity=0.8), showlegend=False),
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            title="<b>Strategy Performance Comparison</b>",
            height=300 * len(metrics_to_compare),
            font=dict(color=self.style.font_color),
        )
        self.style.apply(fig)
        return fig


def create_rsi_analyzer(df: pd.DataFrame) -> BinnedIndicatorAnalyzer:
    return BinnedIndicatorAnalyzer(
        df=df,
        indicator_column="RSI",
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=[
            "Extremely Oversold (0-10)",
            "Severely Oversold (10-20)",
            "Oversold (20-30)",
            "Weak (30-40)",
            "Bearish Neutral (40-50)",
            "Bullish Neutral (50-60)",
            "Strong (60-70)",
            "Overbought (70-80)",
            "Severely Overbought (80-90)",
            "Extremely Overbought (90-100)",
        ],
        indicator_name="RSI",
        bin_strategy=BinStrategy.FIXED,
    )


def create_gap_analyzer(df: pd.DataFrame) -> BinnedIndicatorAnalyzer:
    return BinnedIndicatorAnalyzer(
        df=df,
        indicator_column="Entry_Gap_pct",
        bins=[-float("inf"), -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, float("inf")],
        labels=[
            "Extreme Down (<-1%)",
            "Large Down (-1% to -0.5%)",
            "Medium Down (-0.5% to -0.25%)",
            "Small Down (-0.25% to 0%)",
            "Small Up (0% to 0.25%)",
            "Medium Up (0.25% to 0.5%)",
            "Large Up (0.5% to 1%)",
            "Extreme Up (>1%)",
        ],
        indicator_name="Entry Gap",
        bin_strategy=BinStrategy.FIXED,
    )


def create_gap_sigma_analyzer(df: pd.DataFrame) -> BinnedIndicatorAnalyzer:
    return BinnedIndicatorAnalyzer(
        df=df,
        indicator_column="Gap_Sigma",
        bins=[-float("inf"), -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, float("inf")],
        labels=[
            "Extreme Down (<-2σ)",  # noqa: RUF001
            "Large Down (-2σ to -1.5σ)",  # noqa: RUF001
            "Medium Down (-1.5σ to -1σ)",  # noqa: RUF001
            "Small Down (-1σ to -0.5σ)",  # noqa: RUF001
            "Tiny Down (-0.5σ to 0σ)",  # noqa: RUF001
            "Tiny Up (0σ to 0.5σ)",  # noqa: RUF001
            "Small Up (0.5σ to 1σ)",  # noqa: RUF001
            "Medium Up (1σ to 1.5σ)",  # noqa: RUF001
            "Large Up (1.5σ to 2σ)",  # noqa: RUF001
            "Extreme Up (>2σ)",  # noqa: RUF001
        ],
        indicator_name="Gap Sigma",
        bin_strategy=BinStrategy.FIXED,
    )


def create_session_range_analyzer(
    df: pd.DataFrame, percentile_low: float = 10.0, percentile_high: float = 90.0, n_bins: int = 15
) -> BinnedIndicatorAnalyzer:
    return BinnedIndicatorAnalyzer(
        df=df,
        indicator_column="Entry_SessionRange",
        indicator_name="SessionRange %",
        bin_strategy=BinStrategy.PERCENTILE,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        n_bins=n_bins,
    )


def create_vix_analyzer(df: pd.DataFrame) -> BinnedIndicatorAnalyzer:
    return BinnedIndicatorAnalyzer(
        df=df,
        indicator_column="Entry_VIX",
        bins=list(range(0, 35, 5)),
        labels=[
            "VIX <10 (Very Low)",
            "VIX 10-15 (Low)",
            "VIX 15-20 (Moderate)",
            "VIX 20-25 (Elevated)",
            "VIX 25-30 (High)",
            "VIX 30-35 (Very High)",
            "VIX 35+ (Extreme)",
        ],
        indicator_name="VIX",
        bin_strategy=BinStrategy.FIXED,
    )
