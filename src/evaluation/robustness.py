from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backtesting import Strategy
from plotly.subplots import make_subplots
from scipy.stats import combine_pvalues
from src.config import CASH, RANDOM_STATE
from src.data.bootstrap import OHLCBootstrap
from src.visualization.style import DEFAULT_STYLE
from tqdm import tqdm


class PermutationWalkForwardTester:
    """
    Permutation testing for a single contiguous IS/OOS split with optional per-permutation re-optimization.
    Supports multiple metrics and augments with Neyman-Pearson LRT p-values per metric.
    """

    def __init__(self, optimizer: Any, cash: float = CASH, commission: tuple | None = None, spread: float = 0.0):
        # Unwrap WalkForwardOptimizer to its inner SamboOptimizer
        self.opt = getattr(optimizer, "optimizer", optimizer)
        self.cash = cash
        self.commission = commission
        self.spread = spread

    def evaluate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy_class: type[Strategy],
        metrics: str | list[str] = "Profit Factor",
        n_permutations: int = 200,
        optimize_on_permutation: bool = False,
        optimize_on_real: bool = True,
        params: dict[str, tuple] | None = None,
        constraint: Callable | None = None,
        max_tries: int = 40,
        random_state: int | None = None,
        progress: bool = True,
        progress_position: int = 1,
        higher_is_better: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        if n_permutations <= 0:
            raise ValueError("n_permutations must be > 0")

        metrics_list = [metrics] if isinstance(metrics, str) else list(metrics)
        primary_metric = metrics_list[0]
        higher_is_better = dict(higher_is_better or {})

        # --- 1) Determine Real strategy params ---
        if optimize_on_real:
            stats_is, _, real_params, _, param_names, _ = self.opt.optimize(
                df=train_df,
                strategy_class=strategy_class,
                cash=self.cash,
                params=params,
                constraint=constraint,
                max_tries=max_tries,
                maximize=primary_metric,
                random_state=random_state,
            )
            real_is_metrics = {m: float(stats_is.get(m, np.nan)) for m in metrics_list}
            real_is_trades = stats_is["# Trades"]

            FrozenRealStrategy = self.opt._apply_optimized_params(
                strategy_class=strategy_class,
                optimized_params=real_params,
                param_names=param_names,
            )
        else:
            FrozenRealStrategy = strategy_class
            real_params = []
            param_names = []
            if params:
                param_names = list(params.keys())
                real_params = [getattr(strategy_class, k, None) for k in param_names]

            bt_real_is = self.opt._make_backtest(
                df=train_df,
                strat=FrozenRealStrategy,
                cash=self.cash,
            )
            stats_is = bt_real_is.run()
            real_is_metrics = {m: float(stats_is.get(m, np.nan)) for m in metrics_list}
            real_is_trades = stats_is["# Trades"]

        # --- 2) Real OOS ---
        bt_real_oos = self.opt._make_backtest(
            df=test_df,
            strat=FrozenRealStrategy,
            cash=self.cash,
        )
        stats_oos = bt_real_oos.run()
        real_oos_metrics = {m: float(stats_oos.get(m, np.nan)) for m in metrics_list}
        real_oos_trades = stats_oos["# Trades"]

        # --- 3) Permutation loop ---
        rng = np.random.default_rng(random_state)
        n_train = len(train_df)
        combined_df = pd.concat([train_df, test_df], axis=0)

        perm_is_metrics = {m: np.full(n_permutations, np.nan, dtype=float) for m in metrics_list}
        perm_oos_metrics = {m: np.full(n_permutations, np.nan, dtype=float) for m in metrics_list}
        perm_is_trades = np.full(n_permutations, np.nan, dtype=float)
        perm_oos_trades = np.full(n_permutations, np.nan, dtype=float)

        # Position hierarchy: bt.optimize()=0 (default), permutations=1, folds=2
        iterable = tqdm(
            range(n_permutations),
            desc="Permutations",
            unit="perm",
            dynamic_ncols=True,
            disable=not progress,
            position=progress_position,  # Position=1 for permutation loop
            leave=False,  # Clear after completion (nested bar behavior)
            miniters=max(1, n_permutations // 20),  # Update every 5%
            mininterval=1.0,  # Minimum 1s between updates (reduce I/O)
        )

        for i in iterable:
            seed_i = int(rng.integers(0, 2**32 - 1))
            perm_combined = OHLCBootstrap(method="permutation", seed=seed_i).generate(combined_df)

            if len(perm_combined) != len(combined_df):
                raise ValueError("Permutation must preserve DataFrame length")

            perm_train_df = perm_combined.iloc[:n_train]
            perm_test_df = perm_combined.iloc[n_train:]

            if optimize_on_permutation:
                # Optimization (bt.optimize shows progress at position=0 by default)
                s_is, _, perm_params, _, perm_names, _ = self.opt.optimize(
                    df=perm_train_df,
                    strategy_class=strategy_class,
                    cash=self.cash,
                    params=params,
                    constraint=constraint,
                    max_tries=max_tries,
                    maximize=primary_metric,
                )
                for m in metrics_list:
                    perm_is_metrics[m][i] = float(s_is.get(m, np.nan))
                perm_is_trades[i] = s_is["# Trades"]

                FrozenPermStrategy = self.opt._apply_optimized_params(
                    strategy_class=strategy_class,
                    optimized_params=perm_params,
                    param_names=perm_names,
                )
                bt_perm_oos = self.opt._make_backtest(
                    df=perm_test_df,
                    strat=FrozenPermStrategy,
                    cash=self.cash,
                )
                s_oos = bt_perm_oos.run()
                for m in metrics_list:
                    perm_oos_metrics[m][i] = float(s_oos.get(m, np.nan))
                perm_oos_trades[i] = s_oos["# Trades"]
            else:
                bt_perm_is = self.opt._make_backtest(
                    df=perm_train_df,
                    strat=FrozenRealStrategy,
                    cash=self.cash,
                )
                s_is = bt_perm_is.run()
                for m in metrics_list:
                    perm_is_metrics[m][i] = float(s_is.get(m, np.nan))
                perm_is_trades[i] = s_is["# Trades"]

                bt_perm_oos = self.opt._make_backtest(
                    df=perm_test_df,
                    strat=FrozenRealStrategy,
                    cash=self.cash,
                )
                s_oos = bt_perm_oos.run()
                for m in metrics_list:
                    perm_oos_metrics[m][i] = float(s_oos.get(m, np.nan))
                perm_oos_trades[i] = s_oos["# Trades"]

            # Update postfix with current permutation metrics
            postfix_str = " | ".join([f"{m[:6]}={perm_oos_metrics[m][i]:.3g}" for m in metrics_list])
            iterable.set_postfix_str(postfix_str, refresh=False)

        # --- 4) Results ---
        results: dict[str, Any] = {
            "optimize_on_permutation": bool(optimize_on_permutation),
            "optimize_on_real": bool(optimize_on_real),
            "metrics": metrics_list,
            "real_is_trades": float(real_is_trades),
            "real_oos_trades": float(real_oos_trades),
            "perm_is_trades": perm_is_trades.tolist(),
            "perm_oos_trades": perm_oos_trades.tolist(),
            "n_permutations": int(n_permutations),
            "real_params": dict(zip(param_names, real_params, strict=True)) if param_names else {},
        }

        for m in metrics_list:
            perm_is = perm_is_metrics[m]
            perm_oos = perm_oos_metrics[m]

            is_p = (np.sum(perm_is >= real_is_metrics[m]) + 1.0) / (perm_is.size + 1.0)
            oos_p = (np.sum(perm_oos >= real_oos_metrics[m]) + 1.0) / (perm_oos.size + 1.0)

            results[f"{m}_real_is_perf"] = float(real_is_metrics[m])
            results[f"{m}_real_oos_perf"] = float(real_oos_metrics[m])
            results[f"{m}_perm_is_perfs"] = perm_is.tolist()
            results[f"{m}_perm_oos_perfs"] = perm_oos.tolist()
            results[f"{m}_is_p_value"] = float(is_p)
            results[f"{m}_oos_p_value"] = float(oos_p)
        return results


class MonteCarloInSampleTester:
    """
    Monte Carlo permutation testing for in-sample strategy validation with PlotStyle theming.

    Tests whether strategy performance on training data significantly exceeds
    what would be expected from random chance by comparing against performance
    on permuted versions of the same data.

    This is particularly useful for:
    - Initial strategy validation before proceeding to out-of-sample testing
    - Understanding baseline performance expectations
    - Detecting potential overfitting in optimization procedures
    """

    def __init__(self, optimizer, style=None):
        """
        Initialize with a configured WalkForwardOptimizer and PlotStyle theming.

        Args:
            optimizer: Pre-configured optimizer with trading parameters
            style: PlotStyle for consistent theming across visualizations (optional)

        """
        self.opt = getattr(optimizer, "optimizer", optimizer)
        self.cash = getattr(optimizer, "cash", CASH)
        self.style = style or DEFAULT_STYLE  # Fixed: respect passed style, fall back to default

    def test_strategy(
        self,
        train_df,
        strategy_class,
        metric: str = "Profit Factor",
        n_permutations: int = 200,
        optimize_on_permutation: bool = False,
        optimize_on_real: bool = True,  # Fixed: renamed logic + default to True (intuitive)
        constraint=None,
        params=None,
        max_tries: int = 40,
        random_state=None,
        progress: bool = True,
        spread: float | None = None,
        commission: float | tuple[float, float] | Callable | None = None,
    ):
        """
        Execute Monte Carlo in-sample permutation test.

        Args:
            train_df: Training data with OHLC structure
            strategy_class: Strategy class to test
            metric: Performance metric for comparison
            n_permutations: Number of random permutations to generate
            optimize_on_permutation: Whether to re-optimize on each permutation
            optimize_on_real: Whether to optimize parameters on the real training data
                              (True = optimize, False = use current/default parameters)
            params: Parameter ranges for optimization
            max_tries: Maximum optimization attempts
            random_state: Random seed for reproducibility
            progress: Show progress bar

        Returns:
            Dictionary containing results (see original docstring for full details)

        """
        # Set default random state if not provided
        if random_state is None:
            random_state = RANDOM_STATE

        # Validation
        if n_permutations <= 0:
            raise ValueError("n_permutations must be > 0")

        # Step 1: Run backtest on real data (with or without optimization)
        param_names = []
        real_params = []

        if optimize_on_real:
            # Optimize parameters on real data
            stats_real, _, real_params, _, param_names, _ = self.opt.optimize(
                df=train_df,
                strategy_class=strategy_class,
                cash=self.cash,
                constraint=constraint,
                params=params,
                max_tries=max_tries,
                maximize=metric,
            )
        else:
            # Use current (default) parameters without optimization
            if params is not None:
                param_names = list(params.keys())
                real_params = [getattr(strategy_class, k, None) for k in param_names]

            strategy_for_real = self.opt._apply_optimized_params(
                strategy_class=strategy_class,
                optimized_params=real_params,
                param_names=param_names,
            )

            bt_real = self.opt._make_backtest(
                df=train_df,
                strat=strategy_for_real,
                cash=self.cash,
            )
            stats_real = bt_real.run()

        # Create frozen strategy with the final parameters (optimized or default)
        FrozenRealStrategy = self.opt._apply_optimized_params(
            strategy_class=strategy_class,
            optimized_params=real_params,
            param_names=param_names,
        )

        # Extract real performance metrics
        real_performance = float(stats_real.get(metric, np.nan))
        real_trades = stats_real.get("# Trades", np.nan)

        # Step 2: Generate permutation distribution
        rng = np.random.default_rng(random_state)
        perm_performances = np.full(n_permutations, np.nan, dtype=float)
        perm_trades = np.full(n_permutations, np.nan, dtype=float)

        # Progress tracking
        iterable = tqdm(
            range(n_permutations),
            desc="Monte Carlo IS Test",
            unit="perm",
            dynamic_ncols=True,
            disable=not progress,
            leave=False,
        )

        for i in iterable:
            # Generate permuted training data
            seed_i = int(rng.integers(0, 2**32 - 1))
            perm_train_df = OHLCBootstrap(method="permutation", seed=seed_i).generate(train_df)

            # Validate permutation integrity
            if len(perm_train_df) != len(train_df):
                raise ValueError("Permutation must preserve data length")

            try:
                if optimize_on_permutation:
                    # Re-optimize strategy on permuted data
                    stats_perm, _, _, _, _, _ = self.opt.optimize(
                        df=perm_train_df,
                        strategy_class=strategy_class,
                        cash=self.cash,
                        params=params,
                        constraint=constraint,
                        max_tries=max_tries,
                        maximize=metric,
                    )

                    perm_performance = float(stats_perm.get(metric, np.nan))
                    perm_trades[i] = stats_perm.get("# Trades", np.nan)

                else:
                    # Use fixed parameters on permuted data
                    bt_perm = self.opt._make_backtest(
                        df=perm_train_df,
                        strat=FrozenRealStrategy,
                        cash=self.cash,
                    )

                    stats_perm = bt_perm.run()
                    perm_performance = float(stats_perm.get(metric, np.nan))
                    perm_trades[i] = stats_perm.get("# Trades", np.nan)

                perm_performances[i] = perm_performance

                # Update progress with recent performance
                iterable.set_postfix(perf=f"{perm_performance:.3g}", real=f"{real_performance:.3g}", refresh=False)

            except Exception:
                # Log failed permutation but continue
                perm_performances[i] = np.nan
                perm_trades[i] = np.nan
                iterable.set_postfix(error="Failed", refresh=False)

        # Step 3: Statistical analysis (unchanged from original)
        valid_perms = perm_performances[~np.isnan(perm_performances)]

        if len(valid_perms) == 0:
            return {
                "error": "All permutations failed",
                "real_performance": real_performance,
                "real_trades": real_trades,
                "n_permutations": n_permutations,
                "n_valid_permutations": 0,
            }

        p_value = (np.sum(valid_perms >= real_performance) + 1) / (len(valid_perms) + 1)
        percentile_rank = (valid_perms < real_performance).mean() * 100

        summary_stats = {
            "mean": float(np.nanmean(valid_perms)),
            "std": float(np.nanstd(valid_perms)),
            "min": float(np.nanmin(valid_perms)),
            "max": float(np.nanmax(valid_perms)),
            "q25": float(np.nanpercentile(valid_perms, 25)),
            "q50": float(np.nanpercentile(valid_perms, 50)),
            "q75": float(np.nanpercentile(valid_perms, 75)),
        }

        return {
            "real_performance": float(real_performance),
            "real_trades": float(real_trades),
            "permutation_performances": perm_performances.tolist(),
            "permutation_trades": perm_trades.tolist(),
            "p_value": float(p_value),
            "percentile_rank": float(percentile_rank),
            "is_significant": bool(p_value < 0.05),
            "summary_stats": summary_stats,
            "n_permutations": int(n_permutations),
            "n_valid_permutations": len(valid_perms),
            "metric": metric,
            "optimize_on_permutation": bool(optimize_on_permutation),
            "optimized_params": dict(zip(param_names, real_params, strict=True)) if param_names else {},
        }

    def create_distribution_plot(self, results):
        """Create comprehensive visualization of Monte Carlo permutation test results with PlotStyle theming."""
        if "error" in results:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {results['error']}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=self.style.font_color),
            )
            self.style.apply(fig)
            return fig

        # Extract data
        real_performance = results["real_performance"]
        perm_performances = np.array(results["permutation_performances"])
        valid_perms = perm_performances[~np.isnan(perm_performances)]

        if len(valid_perms) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid permutations to display",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=self.style.font_color),
            )
            self.style.apply(fig)
            return fig

        p_value = results["p_value"]
        percentile_rank = results["percentile_rank"]
        metric = results["metric"]

        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                f"{metric} Distribution (Histogram)",
                f"{metric} Box Plot Comparison",
                "Cumulative Distribution Function",
                "Performance Ranking",
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # 1. Histogram of permutation distribution
        fig.add_trace(
            go.Histogram(
                x=valid_perms,
                nbinsx=min(30, max(10, len(valid_perms) // 5)),
                name="Permutations",
                marker=dict(color=self.style.accent4, line=dict(color=self.style.line, width=1), opacity=0.7),
                hovertemplate=f"<b>{metric}</b><br>Value: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add vertical line for real performance
        fig.add_vline(
            x=real_performance,
            line=dict(color=self.style.accent2, width=3, dash="dash"),
            annotation_text=f"Strategy: {real_performance:.3f}",
            annotation_position="top",
            annotation_font=dict(color=self.style.font_color, size=12),
            row=1,
            col=1,
        )

        # 2. Box plot comparison
        fig.add_trace(
            go.Box(
                y=valid_perms,
                name="Permutations",
                marker=dict(color=self.style.accent4),
                line=dict(color=self.style.line),
                boxpoints="outliers",
                hovertemplate=f"<b>Permutations</b><br>{metric}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Box(
                y=[real_performance],
                name="Strategy",
                marker=dict(color=self.style.accent2),
                line=dict(color=self.style.line),
                hovertemplate=f"<b>Strategy</b><br>{metric}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Cumulative Distribution Function
        sorted_perms = np.sort(valid_perms)
        y_cdf = np.arange(1, len(sorted_perms) + 1) / len(sorted_perms) * 100

        fig.add_trace(
            go.Scatter(
                x=sorted_perms,
                y=y_cdf,
                mode="lines",
                name="CDF",
                line=dict(color=self.style.accent1, width=3),
                hovertemplate=f"<b>CDF</b><br>{metric}: %{{x:.3f}}<br>Percentile: %{{y:.1f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add point for real performance on CDF
        real_percentile = (valid_perms < real_performance).mean() * 100
        fig.add_trace(
            go.Scatter(
                x=[real_performance],
                y=[real_percentile],
                mode="markers",
                name=f"Strategy ({real_percentile:.1f}%ile)",
                marker=dict(
                    color=self.style.accent2, size=12, symbol="star", line=dict(color=self.style.line, width=2)
                ),
                hovertemplate=f"<b>Strategy Position</b><br>{metric}: %{{x:.3f}}<br>Percentile: %{{y:.1f}}%<extra></extra>",  # noqa: E501
            ),
            row=2,
            col=1,
        )

        # 4. Performance ranking scatter plot
        n_valid = len(valid_perms)
        ranks = np.arange(1, n_valid + 1)
        sorted_idx = np.argsort(valid_perms)

        fig.add_trace(
            go.Scatter(
                x=ranks,
                y=valid_perms[sorted_idx],
                mode="markers",
                name="Permutation Ranking",
                marker=dict(color=self.style.accent4, size=6, opacity=0.7, line=dict(color=self.style.line, width=1)),
                hovertemplate=f"<b>Rank</b>: %{{x}}<br><b>{metric}</b>: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # Add strategy position in ranking
        strategy_rank = np.searchsorted(sorted_perms, real_performance) + 1
        fig.add_trace(
            go.Scatter(
                x=[strategy_rank],
                y=[real_performance],
                mode="markers",
                name=f"Strategy (Rank {strategy_rank})",
                marker=dict(
                    color=self.style.accent2, size=14, symbol="star", line=dict(color=self.style.line, width=2)
                ),
                hovertemplate=f"<b>Strategy</b><br>Rank: {strategy_rank}<br>{metric}: {real_performance:.3f}<extra></extra>",  # noqa: E501
            ),
            row=2,
            col=2,
        )

        # Layout configuration with PlotStyle theming
        fig.update_layout(
            title=dict(
                text=f"<b>Monte Carlo Permutation Test Results: {metric}</b><br>"
                f"<sub>P-value: {p_value:.4f} | Percentile: {percentile_rank:.1f}% | "
                f"{'Significant' if results['is_significant'] else 'Not Significant'}</sub>",
                x=0.5,
                font=dict(size=16, color=self.style.font_color),
            ),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(color=self.style.font_color),
            ),
            font=dict(color=self.style.font_color),
        )

        # Apply consistent axis styling
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

        # Update axes
        fig.update_xaxes(title_text=f"{metric}", row=1, col=1, **axis_style)
        fig.update_yaxes(title_text="Frequency", row=1, col=1, **axis_style)

        fig.update_xaxes(title_text="Category", row=1, col=2, **axis_style)
        fig.update_yaxes(title_text=f"{metric}", row=1, col=2, **axis_style)

        fig.update_xaxes(title_text=f"{metric}", row=2, col=1, **axis_style)
        fig.update_yaxes(title_text="Cumulative Probability (%)", row=2, col=1, **axis_style)

        fig.update_xaxes(title_text="Rank", row=2, col=2, **axis_style)
        fig.update_yaxes(title_text=f"{metric}", row=2, col=2, **axis_style)

        # Add significance indicators
        if results["is_significant"]:
            # Green background tint for significant results
            for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    xref=f"x{'' if row == 1 and col == 1 else (row - 1) * 2 + col} domain",
                    yref=f"y{'' if row == 1 and col == 1 else (row - 1) * 2 + col} domain",
                    fillcolor=self.style.accent3,
                    opacity=0.05,
                    layer="below",
                    line_width=0,
                )

        # Apply PlotStyle theming
        self.style.apply(fig)
        return fig

    def create_summary_plot(self, results):
        """
        Create summary visualization focusing on key statistical insights.

        Args:
            results: Output from test_strategy method

        Returns:
            go.Figure: Focused summary plot

        """
        import plotly.graph_objects as go

        if "error" in results:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {results['error']}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=self.style.font_color),
            )
            self.style.apply(fig)
            return fig

        real_performance = results["real_performance"]
        perm_performances = np.array(results["permutation_performances"])
        valid_perms = perm_performances[~np.isnan(perm_performances)]

        if len(valid_perms) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid permutations to display",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=self.style.font_color),
            )
            self.style.apply(fig)
            return fig

        # Create violin plot with overlaid statistics
        fig = go.Figure()

        # Main distribution violin
        fig.add_trace(
            go.Violin(
                y=valid_perms,
                name="Permutation Distribution",
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.style.accent4,
                line_color=self.style.line,
                opacity=0.7,
                hovertemplate=f"<b>Permutations</b><br>{results['metric']}: %{{y:.3f}}<extra></extra>",
            )
        )

        # Strategy performance line
        fig.add_hline(
            y=real_performance,
            line=dict(color=self.style.accent2, width=4, dash="dash"),
            annotation_text=f"Strategy: {real_performance:.3f} ({results['percentile_rank']:.1f}%ile)",
            annotation_position="right",
            annotation_font=dict(color=self.style.font_color, size=14, family="Arial Bold"),
        )

        # Statistical significance zones
        q95 = np.percentile(valid_perms, 95)
        q99 = np.percentile(valid_perms, 99)

        # Add significance threshold lines
        if q95 < real_performance:
            fig.add_hline(
                y=q95,
                line=dict(color=self.style.accent6, width=2, dash="dot"),
                annotation_text="95th percentile",
                annotation_position="left",
                annotation_font=dict(color=self.style.font_color, size=10),
            )

        if q99 < real_performance:
            fig.add_hline(
                y=q99,
                line=dict(color=self.style.accent5, width=2, dash="dot"),
                annotation_text="99th percentile",
                annotation_position="left",
                annotation_font=dict(color=self.style.font_color, size=10),
            )

        # Layout
        fig.update_layout(
            title=dict(
                text=f"<b>Monte Carlo Test Summary: {results['metric']}</b><br>"
                f"<sub>P-value: {results['p_value']:.4f} | "
                f"{'✅ Significant' if results['is_significant'] else '❌ Not Significant'} | "
                f"Valid Permutations: {results['n_valid_permutations']}/{results['n_permutations']}</sub>",
                x=0.5,
                font=dict(size=16, color=self.style.font_color),
            ),
            yaxis_title=f"{results['metric']}",
            xaxis_title="Distribution",
            height=600,
            showlegend=False,
            font=dict(color=self.style.font_color),
        )

        # Apply axis styling
        fig.update_xaxes(
            showgrid=True,
            gridcolor=self.style.grid,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=self.style.grid,
            tickfont=dict(color=self.style.font_color),
            title_font=dict(color=self.style.font_color),
        )

        # Apply PlotStyle theming
        self.style.apply(fig)
        return fig

    def interpret_results(self, results, significance_level: float = 0.05):
        """
        Print human-readable interpretation of Monte Carlo test results.

        Args:
            results: Output from test_strategy method
            significance_level: Threshold for statistical significance

        """
        print("=" * 70)
        print("MONTE CARLO IN-SAMPLE PERMUTATION TEST RESULTS")
        print("=" * 70)

        if "error" in results:
            print(f"ERROR: {results['error']}")
            return

        real_perf = results["real_performance"]
        p_val = results["p_value"]
        percentile = results["percentile_rank"]
        n_valid = results["n_valid_permutations"]
        n_total = results["n_permutations"]

        print(f"Strategy Performance ({results['metric']}): {real_perf:.4f}")
        print(f"Number of Trades: {int(results['real_trades'])}")
        print(f"Valid Permutations: {n_valid}/{n_total}")
        print()

        print("Statistical Significance:")
        print(f" P-value: {p_val:.6f}")
        print(f" Percentile Rank: {percentile:.1f}%")

        if not np.isnan(p_val):
            if p_val < significance_level:
                print(f" Result: **SIGNIFICANT** (p < {significance_level})")
                print(" → Strategy performance appears genuine, not due to random chance")
            else:
                print(f" Result: Not significant (p ≥ {significance_level})")
                print(" → Strategy performance could be explained by random chance")
        else:
            print(" Result: Unable to calculate significance")

        print()
        print("Permutation Distribution Summary:")
        stats = results["summary_stats"]
        print(f" Mean: {stats['mean']:.4f}")
        print(f" Std: {stats['std']:.4f}")
        print(f" Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f" IQR: [{stats['q25']:.4f}, {stats['q75']:.4f}]")

        print()
        if results["optimize_on_permutation"]:
            print("Note: Each permutation was re-optimized (stronger test)")
        else:
            print("Note: Fixed parameters used on permutations (weaker test)")

        print("=" * 70)

    def run_full_analysis(
        self,
        train_df,
        strategy_class,
        metric: str = "Profit Factor",
        n_permutations: int = 200,
        optimize_on_permutation: bool = False,
        show_plots: bool = True,
        **kwargs,
    ):
        """
        Run complete Monte Carlo analysis with visualization.

        Args:
            train_df: Training data
            strategy_class: Strategy to test
            metric: Performance metric
            n_permutations: Number of permutations
            optimize_on_permutation: Re-optimize on each permutation
            show_plots: Display visualization plots
            **kwargs: Additional arguments for test_strategy

        Returns:
            Complete results dictionary with added visualization figures

        """
        # Run the statistical test
        results = self.test_strategy(
            train_df=train_df,
            strategy_class=strategy_class,
            metric=metric,
            n_permutations=n_permutations,
            optimize_on_permutation=optimize_on_permutation,
            **kwargs,
        )

        # Print interpretation
        self.interpret_results(results)

        # Create and show visualizations
        if show_plots and "error" not in results:
            detailed_fig = self.create_distribution_plot(results)
            summary_fig = self.create_summary_plot(results)

            detailed_fig.show()
            summary_fig.show()

            # Add figures to results
            results["detailed_figure"] = detailed_fig
            results["summary_figure"] = summary_fig

        return results


def _extract_valid_pvals(results: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    p = np.array([r.get(key, np.nan) for r in results], dtype=float)
    m = (~np.isnan(p)) & (p >= 0.0) & (p <= 1.0)
    return p[m]


def _prepare_p_for_method(pvals: np.ndarray, method: str) -> np.ndarray:
    pv = np.asarray(pvals, dtype=float)
    eps = np.finfo(float).tiny
    return np.clip(pv, eps, 1.0 - eps)


def analyze_multi_metric_p_values(
    final_results: list[dict[str, Any]],
    metrics: list[str] | None = None,
    significance_level: float = 0.05,
    methods: tuple[str, ...] = ("fisher", "pearson", "stouffer"),
    primary_method: str = "fisher",
    pvalue_source: str = "np",  # "perm" or "np"
) -> dict[str, dict[str, float]]:
    """
    Combine fold p-values across folds per metric.

    pvalue_source:
      - "perm": combines f"{metric}_oos_p_value"
      - "np":   combines f"{metric}_np_oos_p_value"
    """
    if metrics is None:
        metrics = ["Profit Factor", "Sharpe Ratio"]

    if pvalue_source not in {"perm", "np"}:
        raise ValueError("pvalue_source must be one of {'perm','np'}")

    suffix = "oos_p_value" if pvalue_source == "perm" else "np_oos_p_value"

    print("=" * 80)
    print("MULTI-METRIC P-VALUE ANALYSIS")
    print("=" * 80)

    combined_results: dict[str, dict[str, float]] = {}

    for metric in metrics:
        key = f"{metric}_{suffix}"

        print(f"\n{'=' * 80}")
        print(f"METRIC: {metric}  (using {key})")
        print(f"{'=' * 80}")

        header = (
            f"{'Fold':<6} | {'IS perm p':<10} | {'OOS perm p':<10} | {'OOS NP p':<10} | "
            f"{'IS Trades':<10} | {'OOS Trades':<11} | {'IS Perf':<10} | {'OOS Perf':<10}"
        )
        print(header)
        print("-" * len(header))

        for i, r in enumerate(final_results):
            is_perm = r.get(f"{metric}_is_p_value", np.nan)
            oos_perm = r.get(f"{metric}_oos_p_value", np.nan)
            oos_np = r.get(f"{metric}_np_oos_p_value", np.nan)

            is_tra = r.get("real_is_trades", np.nan)
            oos_tra = r.get("real_oos_trades", np.nan)
            is_met = r.get(f"{metric}_real_is_perf", np.nan)
            oos_met = r.get(f"{metric}_real_oos_perf", np.nan)

            def _fmt(x: float, nd: int) -> str:
                return f"{x:.{nd}f}" if pd.notna(x) else "NaN"

            print(
                f"{i + 1:<6} | "
                f"{_fmt(is_perm, 4):<10} | {_fmt(oos_perm, 4):<10} | {_fmt(oos_np, 4):<10} | "
                f"{(int(is_tra) if pd.notna(is_tra) else 'NaN')!s:<10} | {(int(oos_tra) if pd.notna(oos_tra) else 'NaN')!s:<11} | "  # noqa: E501
                f"{_fmt(is_met, 2):<10} | {_fmt(oos_met, 2):<10}"
            )

        print("-" * len(header))

        pvals = _extract_valid_pvals(final_results, key=key)
        if pvals.size < 2:
            print(f"\nInsufficient valid p-values for {metric} (found {pvals.size}).")
            combined_results[metric] = {}
            continue

        combined: dict[str, float] = {}
        for m in methods:
            pv = _prepare_p_for_method(pvals, m)
            _, p = combine_pvalues(pv, method=m)
            combined[m] = float(p)

        print(f"\nCombined P-Values for {metric} ({pvals.size} valid folds):")
        for m in methods:
            tag = "**SIGNIFICANT**" if combined[m] < significance_level else "Not significant"
            print(f"{m.capitalize():>10s}: {combined[m]:.6f} ({tag})")

        if primary_method in combined:
            p = combined[primary_method]
            tag = "**SIGNIFICANT**" if p < significance_level else "Not significant"
            print(f"Final assessment ({primary_method}): {p:.6f} ({tag})")

        combined_results[metric] = combined

    return combined_results
