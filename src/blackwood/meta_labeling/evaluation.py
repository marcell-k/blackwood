from dataclasses import dataclass, field
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from backtesting import Backtest
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from blackwood.config import CASH, MARGIN
from blackwood.visualization.style import DEFAULT_STYLE


def _apply_axis_style(ax: plt.Axes, style) -> None:
    """Apply consistent styling to matplotlib axis."""
    ax.set_facecolor(style.plot_bgcolor)
    ax.grid(True, color=style.grid, alpha=0.6, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(style.line)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=style.font_color)
    ax.xaxis.label.set_color(style.font_color)
    ax.yaxis.label.set_color(style.font_color)
    ax.title.set_color(style.font_color)


def _style_legend(ax: plt.Axes, style, **kwargs) -> None:
    """Create styled legend on axis."""
    leg = ax.legend(facecolor=style.plot_bgcolor, edgecolor=style.line, **kwargs)
    for text in leg.get_texts():
        text.set_color(style.font_color)


@dataclass
class XGBoostModelEvaluator:
    """
    Evaluation and calibration for XGBoost meta-labeling models.
    Provides learning curves, ROC-AUC analysis, and convergence diagnostics.
    """

    model: xgb.XGBClassifier
    X_train: pd.DataFrame
    y_train: pd.Series
    X_dev: pd.DataFrame
    y_dev: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    prior_models: list[xgb.XGBClassifier] = field(default_factory=list)

    metrics: dict[str, dict[str, float]] = field(init=False, default_factory=dict)
    train_loss_history: np.ndarray = field(init=False, default=None)
    val_loss_history: np.ndarray = field(init=False, default=None)
    stage_lengths: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._compute_all_metrics()
        self._extract_loss_history()

    def _compute_split_metrics(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute classification metrics for a single dataset split."""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": roc_auc_score(y, y_proba),
            "n_samples": len(y),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
        }

    def _compute_all_metrics(self) -> None:
        """Compute classification metrics across all splits."""
        splits = [
            ("train", self.X_train, self.y_train),
            ("dev", self.X_dev, self.y_dev),
            ("test", self.X_test, self.y_test),
        ]
        self.metrics = {name: self._compute_split_metrics(X, y) for name, X, y in splits}

    def _extract_loss_history(self) -> None:
        """Extract training and validation loss history from model(s)."""
        train_hist, val_hist = [], []
        self.stage_lengths = []

        for m in [*self.prior_models, self.model]:
            res = m.evals_result() if hasattr(m, "evals_result") else getattr(m, "evals_result_", {})
            if not res or "validation_0" not in res:
                continue

            train_loss = res["validation_0"].get("logloss", [])
            val_loss = res.get("validation_1", {}).get("logloss", [])

            if train_loss:
                self.stage_lengths.append(len(train_loss))
                train_hist.extend(map(float, train_loss))
                if val_loss:
                    val_hist.extend(map(float, val_loss))

        self.train_loss_history = np.asarray(train_hist, dtype=float) if train_hist else None
        self.val_loss_history = np.asarray(val_hist, dtype=float) if val_hist else None

    def plot_learning_curves(self, figsize: tuple[int, int] = (10, 6)):
        """Plot Train vs Validation logloss curves."""
        if self.train_loss_history is None or len(self.train_loss_history) == 0:
            raise ValueError("No training loss history available.")

        style = DEFAULT_STYLE
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(style.paper_bgcolor)

        x = np.arange(1, len(self.train_loss_history) + 1, dtype=int)
        ax.plot(x, self.train_loss_history, label="Train logloss", color=style.accent1, linewidth=2.0)

        if self.val_loss_history is not None and len(self.val_loss_history) == len(self.train_loss_history):
            ax.plot(x, self.val_loss_history, label="Validation logloss", color=style.accent2, linewidth=2.0)

        _apply_axis_style(ax, style)
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Log Loss")
        ax.set_title("Learning Curves (Train vs Validation)")

        if len(self.stage_lengths) > 1:
            for c in np.cumsum(self.stage_lengths)[:-1]:
                ax.axvline(c, color=style.muted, linestyle="--", alpha=0.6)

        _style_legend(ax, style, loc="best")
        fig.tight_layout()
        return fig

    def _plot_single_roc(self, ax: plt.Axes, X: pd.DataFrame, y: pd.Series, title: str, style) -> None:
        """Plot ROC curve on a single axis."""
        y_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot([0, 1], [0, 1], linestyle="--", color=style.muted, alpha=0.5)
        ax.plot(fpr, tpr, color=style.accent1, linewidth=2.5, label=f"ROC (AUC = {roc_auc:.4f})")

        _apply_axis_style(ax, style)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        _style_legend(ax, style, loc="lower right")

    def plot_roc_auc_curves(self, figsize: tuple[int, int] = (12, 5), style=None):
        """Plot ROC-AUC curves for train/dev/test with AUC scores."""
        if style is None:
            style = SimpleNamespace(
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#252526",
                accent1="#4fc3f7",
                accent2="#ff6b6b",
                muted="#808080",
                grid="#3e3e3e",
                line="#5e5e5e",
                font_color="#d4d4d4",
            )

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.patch.set_facecolor(style.paper_bgcolor)

        datasets = [
            ("TRAIN Set", self.X_train, self.y_train),
            ("DEV Set", self.X_dev, self.y_dev),
            ("TEST Set", self.X_test, self.y_test),
        ]
        for (title, X, y), ax in zip(datasets, axes, strict=True):
            self._plot_single_roc(ax, X, y, title, style)

        fig.suptitle("ROC-AUC Curves", fontsize=14, color=style.font_color, y=0.98)
        fig.tight_layout()
        return fig

    def print_metrics_summary_table(self) -> None:
        """Print compact comparison table across all datasets."""
        print(f"\n{'=' * 70}\nMETRICS SUMMARY TABLE\n{'=' * 70}")
        print(f"{'Metric':<15} {'Train':>12} {'Dev':>12} {'Test':>12}")
        print(f"{'-' * 70}")

        metric_pairs = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 Score"),
            ("auc", "AUC-ROC"),
        ]
        for key, label in metric_pairs:
            vals = [self.metrics.get(split, {}).get(key, 0.0) for split in ("train", "dev", "test")]
            print(f"{label:<15} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")

        print(f"{'-' * 70}")
        for key, label in [("n_samples", "Samples"), ("n_positive", "Positive")]:
            vals = [self.metrics.get(split, {}).get(key, 0) for split in ("train", "dev", "test")]
            print(f"{label:<15} {vals[0]:>12d} {vals[1]:>12d} {vals[2]:>12d}")
        print(f"{'=' * 70}\n")


def _interpolate_equity_and_drawdown(stats: dict) -> tuple[pd.Series, pd.Series]:
    """Extract and interpolate equity and drawdown curves from backtest stats."""
    equity_data = stats["_equity_curve"].reset_index(drop=True)
    trades = stats["_trades"]
    equity = equity_data["Equity"]
    drawdown_pct = equity_data["DrawdownPct"]

    dd_end = equity_data["DrawdownDuration"].idxmax()
    dd_start = dd_end if np.isnan(dd_end) else equity[:dd_end].idxmax()
    dd_end_interp = (
        dd_end
        if np.isnan(dd_end)
        else np.interp(equity[dd_start], (equity[dd_end - 1], equity[dd_end]), (dd_end - 1, dd_end))
    )

    interest_pts = [
        equity.index[0],
        equity.index[-1],
        equity.idxmax(),
        drawdown_pct.idxmax(),
        dd_start,
        int(dd_end_interp),
        min(int(dd_end_interp + 1), len(equity) - 1),
    ]
    select = pd.Index(trades["ExitBar"]).union(pd.Index(interest_pts)).unique().dropna()

    equity_interp = equity.iloc[select].reindex(equity.index).interpolate()
    dd_interp = drawdown_pct.iloc[select].reindex(drawdown_pct.index).interpolate() * -100

    return pd.Series(equity_interp.values, index=stats["_equity_curve"].index), pd.Series(
        dd_interp.values, index=stats["_equity_curve"].index
    )


def run_gate_sweep_and_plot(
    df: pd.DataFrame,
    StrategyClass,
    cash: int = CASH,
    spread: int = 0,
    commission: float = (0, 0),
    margin: float = MARGIN,
    gate_col: str = "signal_prob",
    bet_col: str | None = None,
    gate_values_long: np.ndarray = None,
    gate_values_short: np.ndarray = None,
    title: str = "GatedStrategy: Cumulative Return and Drawdown by Probability Gate",
    figsize: tuple[int, int] = (12, 8),
):
    """Run parameter sweep over long/short gating thresholds."""
    style = DEFAULT_STYLE
    gate_values_long = np.arange(0.0, 1.01, 0.1) if gate_values_long is None else gate_values_long
    gate_values_short = gate_values_long.copy() if gate_values_short is None else gate_values_short

    curves_return, curves_dd, rows = [], [], []

    def build_row(stats, g_long, g_short, type_label=None):
        row = {
            "Gate Long": round(float(g_long), 2),
            "Gate Short": round(float(g_short), 2),
            "Vol (Ann) %": round(float(stats.get("Volatility (Ann.) [%]", 0)), 2),
            "#Trades": int(stats.get("# Trades", 0)),
            "Return %": round(float(stats.get("Return [%]", 0)), 2),
            "Max DD %": round(float(stats.get("Max. Drawdown [%]", 0)), 2),
            "Calmar": round(float(stats.get("Calmar Ratio", 0)), 2),
            "Sharpe": round(float(stats.get("Sharpe Ratio", 0)), 2),
        }
        if type_label:
            row["Type"] = type_label
        return row

    bt = Backtest(
        df,
        StrategyClass,
        cash=cash,
        exclusive_orders=False,
        trade_on_close=True,
        spread=spread,
        commission=commission,
        margin=margin,
        finalize_trades=True,
    )

    baseline_added = gate_col is not None or bet_col is not None
    if baseline_added:
        stats_base = bt.run(gate_col=None, gate_min_long=0.0, gate_min_short=0.0, bet_col=None)
        eq, dd = _interpolate_equity_and_drawdown(stats_base)
        curves_return.append(eq)
        curves_dd.append(dd)
        rows.append(build_row(stats_base, 0.0, 0.0, "Baseline"))

    for g_long, g_short in zip(gate_values_long, gate_values_short, strict=True):
        stats = bt.run(gate_col=gate_col, gate_min_long=g_long, gate_min_short=g_short, bet_col=bet_col)
        eq, dd = _interpolate_equity_and_drawdown(stats)
        curves_return.append(eq)
        curves_dd.append(dd)
        rows.append(build_row(stats, g_long, g_short, "Gated" if baseline_added else None))

    stats_df = pd.DataFrame(rows)

    if baseline_added:
        print(f"\n{'BASELINE (No Gating/Betting)':^90}\n{'-' * 90}")
        print(stats_df[stats_df["Type"] == "Baseline"].to_string(index=False))
        print(f"\n{'GATED STRATEGIES':^90}\n{'-' * 90}")
        print(stats_df[stats_df["Type"] == "Gated"].drop(columns=["Type"]).to_string(index=False))
        stats_df = stats_df.set_index(["Type", "Gate Long", "Gate Short"]).sort_index()
    else:
        print(stats_df.to_string(index=False))
        stats_df = stats_df.set_index(["Gate Long", "Gate Short"]).sort_index()
    print("\n" + "=" * 90 + "\n")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(style.paper_bgcolor)

    colors = [style.color0, style.color1, style.color2, style.color3, style.color4, style.color5]

    start_idx = 1 if baseline_added else 0
    ax1.axhline(cash, color=style.muted, lw=1.2, ls="--", alpha=0.7)

    if baseline_added and len(curves_return[0]) > 0:
        ax1.plot(curves_return[0].index, curves_return[0].values, color=colors[0], lw=2.0, label="Baseline", alpha=0.9)

    for i, (g_long, g_short, eq) in enumerate(
        zip(gate_values_long, gate_values_short, curves_return[start_idx:], strict=True), start_idx
    ):
        if len(eq) == 0:
            continue
        label = f"Gate ≥ {g_long:.1f}" if g_long == g_short else f"L≥{g_long:.1f}, S≥{g_short:.1f}"
        ax1.plot(eq.index, eq.values, color=colors[i], lw=2.0 + 0.6 * (g_long in (0.0, 0.5, 1.0)), label=label)

    _apply_axis_style(ax1, style)
    ax1.set_ylabel("Equity ($)", fontweight="bold")
    _style_legend(ax1, style, loc="best", ncol=2)

    ax2.axhline(0, color=style.muted, lw=0.9, alpha=0.7)
    if baseline_added and len(curves_dd[0]) > 0 and not np.allclose(curves_dd[0].values, 0, atol=1e-12):
        dd = curves_dd[0]
        ax2.plot(dd.index, dd.values, color=colors[0], lw=2.0, label=f"Baseline (MDD {dd.min():.1f}%)", alpha=0.9)
        ax2.scatter(dd.index[dd.values.argmin()], dd.min(), color=colors[0], s=30, zorder=3, marker="s")

    for i, (g_long, g_short, dd) in enumerate(
        zip(gate_values_long, gate_values_short, curves_dd[start_idx:], strict=True), start_idx
    ):
        if len(dd) == 0 or np.allclose(dd.values, 0, atol=1e-12):
            continue
        mdd = dd.min()
        label = (
            f"Gate ≥ {g_long:.1f} (MDD {mdd:.1f}%)"
            if g_long == g_short
            else f"L≥{g_long:.1f}, S≥{g_short:.1f} (MDD {mdd:.1f}%)"
        )
        ax2.plot(dd.index, dd.values, color=colors[i], lw=2.0, label=label)
        ax2.scatter(dd.index[dd.values.argmin()], mdd, color=colors[i], s=22, zorder=3)

    _apply_axis_style(ax2, style)
    ax2.set_ylabel("Drawdown [%]", fontweight="bold")
    ax2.set_xlabel("Date", fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45, ha="right")
    _style_legend(ax2, style, loc="best", ncol=2)

    fig.suptitle(title, color=style.font_color, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return stats_df, fig


def plot_probability_distributions(prob_train, prob_dev, prob_test, bins: int = 30, figsize=(18, 5)):
    """Plot predicted probability distributions for train, dev, and test sets."""
    style = DEFAULT_STYLE
    mpl.rcParams.update(
        {
            "figure.facecolor": style.paper_bgcolor,
            "axes.facecolor": style.plot_bgcolor,
            "text.color": style.font_color,
            "axes.labelcolor": style.font_color,
            "xtick.color": style.font_color,
            "ytick.color": style.font_color,
            "grid.color": style.grid,
            "axes.edgecolor": style.line,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=100)
    fig.patch.set_facecolor(style.paper_bgcolor)

    datasets = [
        (prob_train, "Train", style.accent1),
        (prob_dev, "Dev", style.accent2),
        (prob_test, "Test", style.accent3),
    ]

    for (prob, label, color), ax in zip(datasets, axes, strict=True):
        arr = np.asarray(prob)
        ax.hist(arr, bins=bins, alpha=0.7, color=color, edgecolor=style.line, density=True)
        mean_p, median_p = arr.mean(), np.median(arr)
        ax.axvline(mean_p, color=style.font_color, ls="--", lw=2, label=f"Mean: {mean_p:.3f}", alpha=0.8)
        ax.axvline(median_p, color=style.font_color, ls=":", lw=2, label=f"Median: {median_p:.3f}", alpha=0.6)
        ax.set_xlabel("Predicted Probability", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title(f"{label} Set", fontsize=style.title_size, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, facecolor=style.plot_bgcolor)
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_xlim(0, 1)
        ax.text(
            0.02,
            0.98,
            f"N = {len(arr)}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox={"boxstyle": "round", "facecolor": style.plot_bgcolor, "alpha": 0.8, "edgecolor": style.line},
        )

    fig.suptitle("Predicted Probability Distributions (Meta-Label)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
