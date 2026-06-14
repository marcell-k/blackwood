# Standard library
from dataclasses import dataclass
from datetime import time, timedelta
from enum import Enum
from typing import Any

# Third-party
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local
from src.visualization.style import DEFAULT_STYLE


def _validate_and_clean_df(
    df: pd.DataFrame,
    required_cols: list[str],
    rr_col: str = "RiskRewardRatio",
) -> pd.DataFrame:
    """
    Shared validation and numeric cleaning for trade DataFrames.
    - Ensures required columns exist.
    - Coerces to numeric and drops NaN rows in required columns.
    - Removes infinite values in the risk-reward column when present.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    clean_df = df.copy()
    for col in required_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    if rr_col in clean_df.columns:
        clean_df = clean_df[~np.isinf(clean_df[rr_col])]

    clean_df = clean_df.dropna(subset=required_cols)

    if clean_df.empty:
        raise ValueError("No valid trade data found after cleaning.")

    return clean_df


class PerformanceFormatter:
    """Formatter for trading strategy performance statistics."""

    @staticmethod
    def format_stats_output(
        stats: dict[str, Any],
        *,
        metric_name_width: int = 28,
        value_width: int = 12,
        important_only: bool = True,
    ) -> str:
        """Format trading strategy statistics into a structured, human-readable string."""
        display_names = {
            "# Trades": "Number of Trades",
            "Profit Factor": "Profit Factor",
            "Profit Factor PnL": "Profit Factor PnL",
            "Alpha [%]": "Alpha [%]",
            "Beta": "Beta",
            "Sharpe Ratio": "Sharpe Ratio",
            "Probabilistic Sharpe Ratio 1.0": "Prob. Sharpe Ratio 1.0",
            "Min. Track Record Length": "Min. Track Record Length",
            "Arithmetic Sharpe Ratio": "Arithmetic Sharpe Ratio",
            "Sortino Ratio": "Sortino Ratio",
            "Calmar Ratio": "Calmar Ratio",
            "Lake Area": "Lake Area",
            "Win Rate [%]": "Win Rate [%]",
            "Min Winrate (95%) [%]": "Min Win Rate (95%) [%]",
            "Avg. Trade [%]": "Avg. Trade [%]",
            "Avg. Win [%]": "Avg. Win [%]",
            "Avg. Loss [%]": "Avg. Loss [%]",
            "Avg. Drawdown [%]": "Avg. Drawdown [%]",
            "Return (Ann.) [%]": "Annual Return [%]",
            "Volatility (Ann.) [%]": "Volatility (Ann.) [%]",
            "Max. Drawdown [%]": "Maximum Drawdown [%]",
            "Max. Drawdown Duration": "Max. Drawdown Duration",
            "Avg. Drawdown Duration": "Avg. Drawdown Duration",
            "Ulcer Index": "Ulcer Index",
        }

        def _format_value(metric_key: str, val: Any) -> str:
            if isinstance(val, timedelta):
                days_str = f"{val.days} days"
                return f"{days_str:>{value_width}}"
            if isinstance(val, (str, int)):
                return f"{val:>{value_width}}"
            if isinstance(val, float):
                if "Ratio" in metric_key or "Beta" in metric_key:
                    return f"{val:>{value_width}.3f}"
                return f"{val:>{value_width}.2f}"
            return f"{val!s:>{value_width}}"

        def _append_metrics(metric_keys: list[str], lines: list[str]) -> None:
            for metric_key in metric_keys:
                display_name = display_names.get(metric_key, metric_key)
                val = stats.get(metric_key, 0)
                formatted_val = _format_value(metric_key, val)
                lines.append(f"{display_name:<{metric_name_width}} | {formatted_val}")

        separator_length = metric_name_width + 3 + value_width
        key_metrics = [
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
        ]

        output_lines: list[str] = []

        if important_only:
            output_lines.append("=" * separator_length)
            _append_metrics(key_metrics, output_lines)
            output_lines.append("=" * separator_length)
            return "\n".join(output_lines)

        metric_groups: list[tuple[list[str], str]] = [
            (
                ["# Trades", "Profit Factor", "Profit Factor PnL"],
                "TRADE EXECUTION",
            ),
            (
                [
                    "Alpha [%]",
                    "Beta",
                    "Arithmetic Sharpe Ratio",
                    "Sharpe Ratio",
                    "Probabilistic Sharpe Ratio 1.0",
                    "Min. Track Record Length",
                    "Sortino Ratio",
                    "Calmar Ratio",
                ],
                "RISK-ADJUSTED RETURNS",
            ),
            (
                [
                    "Lake Area",
                    "Win Rate [%]",
                    "Min Winrate (95%) [%]",
                    "Avg. Trade [%]",
                    "Avg. Win [%]",
                    "Avg. Loss [%]",
                ],
                "TRADE STATISTICS",
            ),
            (
                [
                    "Avg. Drawdown [%]",
                    "Max. Drawdown [%]",
                    "Avg. Drawdown Duration",
                    "Max. Drawdown Duration",
                    "Ulcer Index",
                ],
                "DRAWDOWN",
            ),
        ]

        for metrics_list, _section_title in metric_groups:
            output_lines.append("-" * separator_length)
            _append_metrics(metrics_list, output_lines)
            output_lines.append("-" * separator_length)

        _append_metrics(key_metrics, output_lines)
        output_lines.append("=" * separator_length)
        return "\n".join(output_lines)


class PerformanceAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        grouping_strategy: callable,
        indicator_name: str = "Analysis",
    ) -> None:
        self.df = self._validate_data(df)
        self.grouping_strategy = grouping_strategy
        self.indicator_name = indicator_name

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["ReturnPct", "RiskRewardRatio", "PnL"]
        return _validate_and_clean_df(df, required_cols)

    @staticmethod
    def calc_profit_factor(pnl_series: pd.Series) -> float:
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = pnl_series[pnl_series < 0].sum()
        total_activity = gross_profit + abs(gross_loss)
        if total_activity == 0:
            return np.nan
        if gross_loss == 0:
            return float("inf")
        return gross_profit / abs(gross_loss)

    @staticmethod
    def compute_metrics(
        grouped_data: pd.core.groupby.generic.DataFrameGroupBy,
        indicator_col: str | None = None,
    ) -> pd.DataFrame:
        trades = grouped_data.size().rename("Trades")
        winrate = grouped_data["ReturnPct"].apply(lambda x: (x > 0).mean() * 100.0).rename("WinRate")
        winrate = winrate.rename("WinRate")
        profit_factor = grouped_data["PnL"].apply(PerformanceAnalyzer.calc_profit_factor).rename("ProfitFactor")
        rr = grouped_data["RiskRewardRatio"]
        avg_rr = rr.mean().rename("AvgRR")
        med_rr = rr.median().rename("MedianRR")
        std_rr = rr.std(ddof=1).fillna(0.0).rename("StdRR")

        metrics = [trades, winrate, profit_factor, avg_rr, med_rr, std_rr]

        if indicator_col is not None:
            avg_indicator = grouped_data[indicator_col].mean().rename("AvgIndicator")
            metrics.append(avg_indicator)

        return pd.concat(metrics, axis=1)

    def _print_zone_table(
        self,
        trades_df: pd.DataFrame,
        trade_type_name: str,
    ) -> dict[str, Any]:
        trades_df = trades_df.copy()
        trades_df["Grouping_Key"] = self.grouping_strategy(trades_df)
        grp = trades_df.groupby("Grouping_Key")
        table = PerformanceAnalyzer.compute_metrics(grp)
        table = table.sort_index()

        title = f"{self.indicator_name} Analysis - {trade_type_name}"
        print(
            f"\n{title:<26} | {'Trades':>6} | {'WinRate':>7} | {'ProfitF':>7} | "
            f"{'AvgRR':>5} | {'MedianRR':>8} | {'StdRR':>6}"
        )
        print("-" * 83)

        zone_stats: dict[str, Any] = {}
        for group_id, row in table.iterrows():
            group_display = group_id.strftime("%H:%M") if isinstance(group_id, time) else str(group_id)

            pf = row["ProfitFactor"]
            if np.isnan(pf):
                pf_display = "   -   "
            elif np.isinf(pf):
                pf_display = " inf "
            else:
                pf_display = f"{pf:>8.3f}"

            print(
                f"{group_display:<26} |"
                f"{int(row['Trades']):>7d} |"
                f"{row['WinRate']:>7.1f}% |"
                f"{pf_display:>8} |"
                f"{row['AvgRR']:>6.2f} |"
                f"{row['MedianRR']:>9.3f} |"
                f"{row['StdRR']:>7.3f}"
            )

            zone_stats[str(group_id)] = {
                "trades": int(row["Trades"]),
                "win_rate": float(row["WinRate"]),
                "profit_factor": None if np.isnan(pf) else float(pf),
                "avg_rr": float(row["AvgRR"]),
                "median_rr": float(row["MedianRR"]),
                "std_rr": float(row["StdRR"]),
            }

        return zone_stats

    def _analyze_trade_type(
        self,
        trades_df: pd.DataFrame,
        trade_type_name: str,
    ) -> dict[str, Any] | None:
        if trades_df.empty:
            return None
        return self._print_zone_table(trades_df, trade_type_name)

    def run_analysis(self, trade_direction: str = "both") -> dict[str, Any]:
        valid_directions = {"all", "long", "short", "both"}
        if trade_direction not in valid_directions:
            raise ValueError(f"trade_direction must be one of {valid_directions}, got '{trade_direction}'")

        if trade_direction in {"long", "short", "both"} and "Size" not in self.df.columns:
            raise ValueError(f"trade_direction='{trade_direction}' requires 'Size' column in DataFrame")

        if trade_direction == "all":
            all_stats = self._analyze_trade_type(self.df, "All") or {}
            return {"all_analysis": {"summary": all_stats}}

        if trade_direction == "long":
            long_trades = self.df[self.df["Size"] > 0]
            long_stats = self._analyze_trade_type(long_trades, "Long") or {}
            return {"all_analysis": {"summary": long_stats}}

        if trade_direction == "short":
            short_trades = self.df[self.df["Size"] < 0]
            short_stats = self._analyze_trade_type(short_trades, "Short") or {}
            return {"all_analysis": {"summary": short_stats}}

        # both
        long_trades = self.df[self.df["Size"] > 0]
        short_trades = self.df[self.df["Size"] < 0]
        long_stats = self._analyze_trade_type(long_trades, "Long") or {}
        short_stats = self._analyze_trade_type(short_trades, "Short") or {}
        all_stats = self._analyze_trade_type(self.df, "All") or {}

        return {
            "all_analysis": {"summary": all_stats},
            "long_analysis": {"summary": long_stats},
            "short_analysis": {"summary": short_stats},
        }


class BinStrategy(Enum):
    FIXED = "fixed"
    PERCENTILE = "percentile"


@dataclass
class BinnedIndicatorAnalyzer:
    df: pd.DataFrame
    indicator_column: str
    bins: list[float] | None = None
    labels: list[str] | None = None
    indicator_name: str = "Indicator"
    bin_strategy: BinStrategy = BinStrategy.FIXED
    percentile_low: float = 10.0
    percentile_high: float = 90.0
    n_bins: int = 15

    def __post_init__(self) -> None:
        self.df = self._validate_data()
        if self.bin_strategy == BinStrategy.PERCENTILE:
            if self.bins is not None or self.labels is not None:
                raise ValueError("bins and labels must be None when using PERCENTILE strategy")
            self.bins, self.labels = self._calculate_percentile_bins()
        else:
            if self.bins is None or self.labels is None:
                raise ValueError("bins and labels required when using FIXED strategy")
            if len(self.labels) != len(self.bins) - 1:
                raise ValueError(
                    f"labels length ({len(self.labels)}) must equal bins length - 1 ({len(self.bins) - 1})"
                )

    def _validate_data(self) -> pd.DataFrame:
        required_cols = ["Size", "ReturnPct", "RiskRewardRatio", "PnL", self.indicator_column]
        return _validate_and_clean_df(self.df, required_cols)

    def _round_to_clean_step(self, raw_width: float) -> float:
        if raw_width < 0.0375:
            base = 0.025
        elif raw_width < 0.075:
            base = 0.05
        elif raw_width < 0.15:
            base = 0.1
        elif raw_width < 0.35:
            base = 0.2
        elif raw_width < 0.75:
            base = 0.5
        else:
            base = 1.0
        return base * np.ceil(raw_width / base)

    def _calculate_percentile_bins(self) -> tuple[list[float], list[str]]:
        data = self.df[self.indicator_column].values
        p_low = np.percentile(data, self.percentile_low)
        p_high = np.percentile(data, self.percentile_high)
        raw_width = (p_high - p_low) / self.n_bins
        step = self._round_to_clean_step(raw_width)
        start = np.floor(p_low / step) * step
        bins = [start + i * step for i in range(self.n_bins + 1)]
        bins.append(float("inf"))

        labels: list[str] = []
        for i in range(len(bins) - 1):
            if bins[i + 1] == float("inf"):
                labels.append(f"Outlier (>{bins[i]:.2f})")
            else:
                labels.append(f"{bins[i]:.2f}-{bins[i + 1]:.2f}")
        return bins, labels

    def _assign_zone(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        out = trades_df.copy()
        out["Zone"] = pd.cut(
            out[self.indicator_column],
            bins=self.bins,
            labels=self.labels,
            right=False,
            ordered=True,
        )
        out["Zone"] = out["Zone"].astype(pd.CategoricalDtype(categories=self.labels, ordered=True))
        return out

    def _print_zone_table(
        self,
        trades_df: pd.DataFrame,
        trade_type_name: str,
    ) -> dict[str, Any]:
        dfz = self._assign_zone(trades_df)
        grp = dfz.groupby("Zone", observed=False)
        table = PerformanceAnalyzer.compute_metrics(grp, indicator_col=self.indicator_column)
        table = table.reindex(self.labels).fillna(0.0)

        zone_width = min(max(len(max(self.labels, key=len)), 25), 45)
        title = f"{self.indicator_name} - {trade_type_name}"
        print(f"\n{title:<25} | Trades | WinRate | ProfitF |   AvgRR | MedianRR |   StdRR | AvgInd")
        print("-" * 95)

        zone_stats: dict[str, Any] = {}
        for zone, row in table.iterrows():
            pf = row["ProfitFactor"]
            if np.isnan(pf):
                pf_display = "   -   "
            elif np.isinf(pf):
                pf_display = " inf "
            else:
                pf_display = f"{pf:>8.3f}"

            print(
                f"{zone:<{zone_width}} |"
                f"{int(row['Trades']):>7d} |"
                f"{row['WinRate']:>7.1f}% |"
                f"{pf_display:>8} |"
                f"{row['AvgRR']:>8.3f} |"
                f"{row['MedianRR']:>9.3f} |"
                f"{row['StdRR']:>8.3f} |"
                f"{row['AvgIndicator']:>7.2f}"
            )

            zone_stats[str(zone)] = {
                "trades": int(row["Trades"]),
                "win_rate": float(row["WinRate"]),
                "profit_factor": None if np.isnan(pf) else float(pf),
                "avg_rr": float(row["AvgRR"]),
                "median_rr": float(row["MedianRR"]),
                "std_rr": float(row["StdRR"]),
                "avg_indicator": float(row["AvgIndicator"]),
            }

        return zone_stats

    def _analyze_trade_type(
        self,
        trades_df: pd.DataFrame,
        trade_type_name: str,
    ) -> dict[str, Any] | None:
        if trades_df.empty:
            print(f"\n{trade_type_name} trades: No data")
            return None
        return self._print_zone_table(trades_df, trade_type_name)

    def run_analysis(self) -> dict[str, Any]:
        long_trades = self.df[self.df["Size"] > 0]
        short_trades = self.df[self.df["Size"] < 0]

        print("=" * 95)
        print(f"{self.indicator_name.upper()} PERFORMANCE ANALYSIS")
        print("=" * 95)

        all_zone_stats = self._analyze_trade_type(self.df, "All")
        long_zone_stats = self._analyze_trade_type(long_trades, "Long")
        short_zone_stats = self._analyze_trade_type(short_trades, "Short")

        print("=" * 95)

        return {
            "all_analysis": {"summary": all_zone_stats or {}},
            "long_analysis": {"summary": long_zone_stats or {}},
            "short_analysis": {"summary": short_zone_stats or {}},
        }

    def plot_analysis(
        self,
        analysis_results: dict[str, Any],
        trade_type: str = "All",
    ) -> go.Figure:
        plot_style = DEFAULT_STYLE

        key_map = {
            "All": "all_analysis",
            "Long": "long_analysis",
            "Short": "short_analysis",
        }
        key = key_map[trade_type]

        if key not in analysis_results or "summary" not in analysis_results[key]:
            raise ValueError(f"No {trade_type} analysis results found")

        zone_stats = analysis_results[key]["summary"]
        if not zone_stats:
            raise ValueError("No zone statistics to plot")

        zones = list(zone_stats.keys())
        trades = [zone_stats[z]["trades"] for z in zones]
        win_rates = [zone_stats[z]["win_rate"] for z in zones]
        avg_rrs = [zone_stats[z]["avg_rr"] for z in zones]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                f"Trade Count by {self.indicator_name} Zone ({trade_type})",
                f"Win Rate by {self.indicator_name} Zone ({trade_type})",
                f"Average RRR by {self.indicator_name} Zone ({trade_type})",
            ],
        )

        fig.add_trace(
            go.Bar(
                x=zones,
                y=trades,
                name="Trade Count",
                marker=dict(color=plot_style.accent1, line=dict(color=plot_style.line, width=1), opacity=0.8),
                hovertemplate="<b>%{x}</b><br>Trades: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=zones,
                y=win_rates,
                name="Win Rate",
                marker=dict(color=plot_style.accent3, line=dict(color=plot_style.line, width=1), opacity=0.8),
                hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=50, line_dash="dot", line_color=plot_style.muted, opacity=0.7, row=1, col=2)

        rrr_colors = [
            plot_style.accent2 if rr < 0 else plot_style.accent4 if rr < 1.0 else plot_style.accent3 for rr in avg_rrs
        ]
        fig.add_trace(
            go.Bar(
                x=zones,
                y=avg_rrs,
                name="Average RRR",
                marker=dict(color=rrr_colors, line=dict(color=plot_style.line, width=1), opacity=0.8),
                hovertemplate="<b>%{x}</b><br>Avg RRR: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=3,
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color=plot_style.accent2, opacity=0.6, row=1, col=3)
        fig.add_hline(y=1.0, line_dash="dot", line_color=plot_style.muted, opacity=0.7, row=1, col=3)

        fig.update_layout(
            height=500,
            showlegend=False,
            title=dict(
                text=f"<b>{self.indicator_name} Analysis - {trade_type} Trades</b>",
                x=0.5,
                font=dict(size=16, color=plot_style.font_color),
            ),
            font=dict(size=12, color=plot_style.font_color),
            margin=dict(l=60, r=40, t=80, b=120),
        )

        axis_style = dict(
            showline=True,
            linewidth=2,
            linecolor=plot_style.line,
            gridcolor=plot_style.grid,
            gridwidth=1,
            tickfont=dict(color=plot_style.font_color),
            title_font=dict(color=plot_style.font_color),
            tickangle=45,
        )
        fig.update_yaxes(title_text="Trade Count", row=1, col=1, **axis_style)
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2, **axis_style)
        fig.update_yaxes(title_text="Average RRR", row=1, col=3, **axis_style)
        for col in (1, 2, 3):
            fig.update_xaxes(row=1, col=col, **axis_style)

        plot_style.apply(fig)
        return fig
