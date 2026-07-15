from __future__ import annotations

import math
import multiprocessing as mp
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import TYPE_CHECKING, Any, Literal

import backtesting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from plotly.subplots import make_subplots
from scipy.ndimage import generic_filter

from blackwood.config import RANDOM_STATE
from blackwood.visualization.style import DEFAULT_STYLE, PlotStyle

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from blackwood.metrics.core import Stats

ParameterSpace = dict[str, tuple[Any, ...] | list[int | float]]


@dataclass
class OptimizationResult:
    """
    Unified result object.

    Conventions:
    - stats: pd.Series of "final evaluation stats" for best params, where possible.
      (Portfolio optimization is aggregated; stats includes objective + params.)
    - params: best params dict
    - results_df: long-form dataframe of all evaluated points/trials (if available)
    - optimizer_info: raw optimizer artifacts (heatmap, study, optres, etc.)
    """

    stats: pd.Series | dict
    strategy: type[Strategy]
    params: dict[str, Any] = field(default_factory=dict)
    param_names: list[str] = field(default_factory=list)
    results_df: pd.DataFrame | None = None
    optimizer_info: Any = None
    best_metric: float = float("nan")
    method_used: str = ""
    success: bool = True

    @property
    def best_params(self) -> dict[str, Any]:
        return self.params


class ParameterSpaceHandler:
    """Consistent handling of param spaces across optimizers."""

    @staticmethod
    def build_grids(
        param_space: ParameterSpace,
        float_points: int = 15,
    ) -> tuple[list[str], list[list], dict[str, str]]:
        """
        Build grids for exhaustive search.
        Rules:
        - tuple(...) => categorical
        - [lo, hi]   => numeric range (ints -> inclusive range; floats -> linspace)
        - list([...]) other => treated as categorical values
        """
        param_names = list(param_space.keys())
        grids: list[list] = []
        param_types: dict[str, str] = {}

        for name in param_names:
            space = param_space[name]
            if isinstance(space, tuple):
                param_types[name] = "categorical"
                grids.append(list(space))
                continue

            if isinstance(space, list):
                if len(space) == 2 and all(isinstance(v, (int, float, np.number)) for v in space):
                    low, high = sorted(space)
                    if all(isinstance(v, (int, np.integer)) for v in (low, high)):
                        param_types[name] = "integer"
                        grids.append(list(range(int(low), int(high) + 1)))
                    else:
                        param_types[name] = "float"
                        grids.append(np.linspace(float(low), float(high), num=int(float_points)).tolist())
                else:
                    param_types[name] = "categorical"
                    grids.append(list(space))
                continue

            param_types[name] = "categorical"
            grids.append([space])

        return param_names, grids, param_types

    @staticmethod
    def suggest_for_optuna(trial: Any, name: str, space: tuple | list) -> Any:
        """Optuna suggestion logic."""
        if isinstance(space, tuple):
            return trial.suggest_categorical(name, list(space))
        if isinstance(space, list) and len(space) == 2:
            low, high = sorted(space)
            if all(isinstance(v, (int, np.integer)) for v in (low, high)):
                return trial.suggest_int(name, int(low), int(high))
            return trial.suggest_float(name, float(low), float(high))
        raise ValueError(f"Unsupported param space for {name}: {space}")


class OptimizerVisualizer:
    """Decoupled visualization for results_df-based optimizers."""

    def __init__(self, style: PlotStyle = DEFAULT_STYLE) -> None:
        self.style = style

    def _apply_plotly(self, fig: go.Figure) -> go.Figure:
        return self.style.apply(fig) if hasattr(self.style, "apply") else fig

    def _apply_mpl(self, fig: Figure, ax: Axes) -> None:
        if hasattr(self.style, "apply_mpl"):
            self.style.apply_mpl(fig, ax)

    @staticmethod
    def _neighbor_average(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        return float(np.mean(valid)) if len(valid) else float("nan")

    def plot_heatmap(
        self,
        results_df: pd.DataFrame,
        x_param: str,
        y_param: str,
        metric_col: str = "SharpeRatio",
        size: int = 3,
    ) -> None:
        pivot = results_df.pivot_table(index=y_param, columns=x_param, values=metric_col, aggfunc="mean")
        z = generic_filter(
            pivot.values,
            lambda v: self._neighbor_average(v),
            size=size,
            mode="constant",
            cval=np.nan,
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=[str(v) for v in pivot.columns.tolist()],
                y=[str(v) for v in pivot.index.tolist()],
                colorscale="RdYlGn",
                text=np.round(z, 3),
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title=f"{metric_col} - {y_param} x {x_param} (smoothed)",
            xaxis_title=x_param,
            yaxis_title=y_param,
        )
        self._apply_plotly(fig).show()

    def plot_smoothed(
        self,
        results_df: pd.DataFrame,
        param_col: str,
        metric_col: str = "SharpeRatio",
        window: int = 5,
    ) -> None:
        df = results_df.sort_values(param_col).reset_index(drop=True).copy()
        df["metric_smoothed"] = df[metric_col].rolling(window=window, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        raw_kwargs = dict(marker="o", linestyle="", alpha=0.5, markersize=5, label=f"Raw {metric_col}")
        smooth_kwargs = dict(linestyle="-", linewidth=2.5, label=f"Smoothed (MA{window})")

        muted = getattr(self.style, "muted", None)
        accent2 = getattr(self.style, "accent2", None)
        if muted is not None:
            raw_kwargs["color"] = muted
        if accent2 is not None:
            smooth_kwargs["color"] = accent2

        ax.plot(df[param_col], df[metric_col], **raw_kwargs)
        ax.plot(df[param_col], df["metric_smoothed"], **smooth_kwargs)

        ax.set_title(f"{metric_col} Optimization Landscape", fontweight="bold")
        ax.set_xlabel(param_col.replace("_", " ").title())
        ax.set_ylabel(metric_col)
        ax.legend(loc="best")
        self._apply_mpl(fig, ax)

        plt.tight_layout()
        plt.show()

    def plot_param_heatmaps(
        self,
        results_df: pd.DataFrame,
        params: list[str] | dict[str, Sequence],
        metric_col: str = "SharpeRatio",
        size: int = 2,
    ) -> None:
        param_names = list(params.keys()) if isinstance(params, dict) else list(params)

        n_params = len(param_names)
        if n_params < 1 or n_params > 9:
            raise ValueError(f"Expected 1-9 parameters, got {n_params}")

        missing = [p for p in param_names if p not in results_df.columns]
        if missing:
            raise ValueError(f"Parameters not in results_df: {missing}")

        if n_params == 1:
            self.plot_smoothed(results_df, param_names[0], metric_col=metric_col, window=size)
            return
        if n_params == 2:
            self.plot_heatmap(results_df, param_names[0], param_names[1], metric_col=metric_col, size=size)
            return

        pairs = list(combinations(range(n_params), 2))
        n_pairs = len(pairs)
        cols = math.ceil(math.sqrt(n_pairs))
        rows = math.ceil(n_pairs / cols)

        heatmap_data: list[dict[str, Any]] = []
        z_min, z_max = np.inf, -np.inf

        for i, j in pairs:
            x_p, y_p = param_names[i], param_names[j]
            pivot = results_df.pivot_table(index=y_p, columns=x_p, values=metric_col, aggfunc="mean")
            z = generic_filter(
                pivot.values,
                lambda v: self._neighbor_average(v),
                size=size,
                mode="constant",
                cval=np.nan,
            )
            z_min = min(z_min, float(np.nanmin(z)))
            z_max = max(z_max, float(np.nanmax(z)))

            heatmap_data.append(
                {
                    "x_param": x_p,
                    "y_param": y_p,
                    "x_labels": pivot.columns.values,
                    "y_labels": pivot.index.values,
                    "z": z,
                }
            )

        subplot_specs = [
            [{"type": "xy"} if (r * cols + c) < n_pairs else None for c in range(cols)] for r in range(rows)
        ]
        subplot_titles = [f"{d['x_param']} vs {d['y_param']}" for d in heatmap_data]
        subplot_titles += [""] * (rows * cols - n_pairs)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=subplot_specs,
            horizontal_spacing=0.04,
            vertical_spacing=0.06,
        )

        for idx, d in enumerate(heatmap_data):
            r, c = divmod(idx, cols)
            fig.add_trace(
                go.Heatmap(
                    z=d["z"],
                    x=[str(v) for v in d["x_labels"]],
                    y=[str(v) for v in d["y_labels"]],
                    colorscale="RdYlGn",
                    zmin=z_min,
                    zmax=z_max,
                    text=np.round(d["z"], 3),
                    texttemplate="%{text}",
                    textfont={"size": 9, "color": "black"},
                    hoverongaps=False,
                    colorbar=dict(title=metric_col, thickness=20, len=0.8, x=1.02, xanchor="left")
                    if idx == 0
                    else None,
                    showscale=(idx == 0),
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.update_xaxes(
                title_text=d["x_param"],
                type="category",
                row=r + 1,
                col=c + 1,
                tickfont=dict(size=9),
                title_font=dict(size=10),
            )
            fig.update_yaxes(
                title_text=d["y_param"],
                type="category",
                row=r + 1,
                col=c + 1,
                tickfont=dict(size=9),
                title_font=dict(size=10),
            )

        fig.update_layout(
            title=f"Parameter Optimization Landscape: {metric_col} (Smoothed)",
            width=max(1000, 350 * cols),
            height=max(800, 280 * rows),
            showlegend=False,
        )
        self._apply_plotly(fig).show()

    def plot_session_heatmap(
        self,
        results_df: pd.DataFrame,
        metric_col: str = "SharpeRatio",
        size: int = 3,
    ) -> None:
        """
        Session heatmap (StartTime -> EndTime) based on results_df columns:
        - either start_hour/start_min/end_hour/end_min
        - or StartTime/EndTime in 'HH:MM'
        """
        df = results_df.copy()

        if all(c in df.columns for c in ["start_hour", "start_min", "end_hour", "end_min"]):
            df["StartTime"] = df.apply(lambda r: f"{int(r['start_hour']):02d}:{int(r['start_min']):02d}", axis=1)
            df["EndTime"] = df.apply(lambda r: f"{int(r['end_hour']):02d}:{int(r['end_min']):02d}", axis=1)

        if not all(c in df.columns for c in ["StartTime", "EndTime"]):
            raise ValueError("Session columns missing: need StartTime/EndTime or start_* and end_* columns.")

        df["StartHour"] = df["StartTime"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60.0)
        df["EndHour"] = df["EndTime"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60.0)
        df = df[df["EndHour"] > df["StartHour"] + 0.01].copy()

        pivot = df.pivot_table(index="StartHour", columns="EndHour", values=metric_col, aggfunc="mean")
        z = generic_filter(pivot.values, lambda v: self._neighbor_average(v), size=size, mode="constant", cval=np.nan)

        start_labels = [f"{int(h)}:{int((h % 1) * 60):02d}" for h in pivot.index]
        end_labels = [f"{int(h)}:{int((h % 1) * 60):02d}" for h in pivot.columns]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=end_labels,
                y=start_labels,
                colorscale="RdYlGn",
                text=np.round(z, 3),
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title=f"{metric_col} by Session (Valid Only)",
            xaxis_title="End Time",
            yaxis_title="Start Time",
            yaxis=dict(autorange="reversed"),
            width=950,
            height=720,
        )
        self._apply_plotly(fig).show()


class BaseOptimizer:
    def __init__(self, bt_func: Callable | None = None) -> None:
        self.bt_func = bt_func
        self.style = DEFAULT_STYLE
        self.visualizer = OptimizerVisualizer(self.style)

        self.results_df: pd.DataFrame | None = None
        self.optimization_result: OptimizationResult | None = None

    def _store_result(self, result: OptimizationResult) -> None:
        self.optimization_result = result
        self.results_df = result.results_df

    @staticmethod
    def _to_series(stats: Stats) -> pd.Series:
        if isinstance(stats, pd.Series):
            return stats
        if isinstance(stats, dict):
            return pd.Series(stats)
        if hasattr(stats, "to_dict"):
            return pd.Series(stats.to_dict())
        return pd.Series(dict(stats))

    def _create_result(
        self,
        stats: Any,
        params: dict[str, Any],
        param_names: list[str],
        results_df: pd.DataFrame | None = None,
        optimizer_info: Any = None,
        strategy: type[Any] | None = None,
        method: str = "",
        best_metric: float = float("nan"),
        success: bool = True,
    ) -> OptimizationResult:
        res = OptimizationResult(
            stats=self._to_series(stats),
            strategy=strategy,
            params=dict(params),
            param_names=list(param_names),
            results_df=results_df,
            optimizer_info=optimizer_info,
            best_metric=float(best_metric),
            method_used=str(method),
            success=bool(success),
        )
        self._store_result(res)
        return res


class SamboOptimizer(BaseOptimizer):
    """SAMBO / grid optimizer for backtesting.py Strategy classes."""

    def __init__(
        self,
        margin: float = 1.0,
        spread: float = 0.0,
        commission: float | tuple[float, float] | Callable = 0.0,
        trade_on_close: bool = True,
        exclusive_orders: bool = False,
        bt_func: Callable | None = None,
    ):
        super().__init__(bt_func=bt_func)
        self.margin = float(margin)
        self.spread = float(spread)
        self.commission = commission
        self.trade_on_close = bool(trade_on_close)
        self.exclusive_orders = bool(exclusive_orders)

    @staticmethod
    def _apply_optimized_params(
        strategy_class: type[Any],
        param_names: list[str],
        optimized_params: Sequence[Any],
    ) -> type[Any]:
        class ConfiguredStrategy(strategy_class):
            pass

        for name, val in zip(param_names, optimized_params, strict=True):
            if val is None or (isinstance(val, (float, np.floating)) and not np.isfinite(val)):
                continue
            if isinstance(val, (int, np.integer)) or (isinstance(val, float) and float(val).is_integer()):
                setattr(ConfiguredStrategy, name, int(round(float(val))))
            elif isinstance(val, (float, np.floating)):
                setattr(ConfiguredStrategy, name, float(val))
            else:
                setattr(ConfiguredStrategy, name, val)
        return ConfiguredStrategy

    def _make_backtest(
        self,
        df: pd.DataFrame,
        strat: type[Any],
        cash: float,
        finalize_trades: bool = True,
    ) -> Any:
        if self.bt_func is not None:
            return self.bt_func(df, strat, cash, self.spread, self.commission, finalize_trades)

        return Backtest(
            df,
            strat,
            cash=cash,
            spread=self.spread,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders,
            finalize_trades=finalize_trades,
        )

    @staticmethod
    def _heatmap_to_df(heatmap: pd.Series, metric_name: str) -> pd.DataFrame:
        df = heatmap.reset_index()
        df = df.rename(columns={0: "Metric"})
        df["MetricName"] = metric_name
        if metric_name.lower().startswith("sharpe"):
            df["SharpeRatio"] = df["Metric"]
        return df

    def optimize(
        self,
        df: pd.DataFrame,
        strategy_class: type[Strategy],
        cash: float,
        param_space: ParameterSpace | None = None,
        metric: str = "Sharpe Ratio",
        n_trials: int = 60,
        constraint: Callable | None = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
        method: Literal["sambo", "grid"] = "sambo",
        n_jobs: int | None = None,
        mp_start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
        **_,
    ) -> OptimizationResult:
        if param_space is None:
            bt = self._make_backtest(df, strategy_class, cash, finalize_trades=True)
            stats = bt.run()
            best = float(self._to_series(stats).get(metric, self._to_series(stats).get("Sharpe Ratio", np.nan)))
            return self._create_result(
                stats=stats,
                params={},
                param_names=[],
                results_df=None,
                optimizer_info=None,
                strategy=strategy_class,
                method="no-opt",
                best_metric=best,
            )

        param_names = list(param_space.keys())

        if method == "sambo":
            if n_jobs is not None:
                warnings.warn("SAMBO is single-process; n_jobs ignored.", stacklevel=2)

            bt = self._make_backtest(df, strategy_class, cash, finalize_trades=True)
            stats, _, optres = bt.optimize(
                **param_space,
                constraint=constraint,
                maximize=metric,
                random_state=random_state,
                method="sambo",
                max_tries=int(n_trials),
                return_optimization=True,
                return_heatmap=True,
            )
            optimized_params = list(optres.x)
            optimizer_info = optres
            results_df = None
        else:
            bt = self._make_backtest(df, strategy_class, cash, finalize_trades=True)

            def _pool_factory(processes=None, initializer=None, initargs=()):
                ctx = mp.get_context(mp_start_method)
                procs = processes if processes is not None else n_jobs
                return ctx.Pool(processes=procs, initializer=initializer, initargs=initargs)

            pool_context = bt(backtesting, "Pool", _pool_factory) if self.bt_func is None else nullcontext()

            with pool_context:
                stats, heatmap = bt.optimize(
                    **param_space,
                    constraint=constraint,
                    maximize=metric,
                    random_state=random_state,
                    method="grid",
                    max_tries=int(n_trials),
                    return_heatmap=True,
                )

            if heatmap.isna().all():
                optimized_params = [getattr(strategy_class, n, None) for n in param_names]
            else:
                best_idx = heatmap.idxmax(skipna=True)
                if not isinstance(best_idx, tuple):
                    best_idx = (best_idx,)
                best_map = dict(zip(heatmap.index.names, best_idx, strict=True))
                optimized_params = [best_map.get(n) for n in param_names]

            optimizer_info = heatmap
            results_df = self._heatmap_to_df(heatmap, metric)

        OptimizedStrategy = self._apply_optimized_params(strategy_class, param_names, optimized_params)
        final_bt = self._make_backtest(df, OptimizedStrategy, cash, finalize_trades=True)
        final_stats = final_bt.run()
        final_s = self._to_series(final_stats)
        best_metric = float(final_s.get(metric, final_s.get("Sharpe Ratio", np.nan)))

        return self._create_result(
            stats=final_stats,
            params=dict(zip(param_names, optimized_params, strict=True)),
            param_names=param_names,
            results_df=results_df,
            optimizer_info=optimizer_info,
            strategy=OptimizedStrategy,
            method=method,
            best_metric=best_metric,
        )

    def backtest(self, df: pd.DataFrame, strategy_class: type[Any], cash: float, **kwargs) -> Any:
        return self._make_backtest(df, strategy_class, cash, **kwargs)


class GridOptimizer(BaseOptimizer):
    """Exhaustive Cartesian grid search over a ParameterSpace using bt_func(**bt_kwargs, **params)."""

    def __init__(self, bt_func: Callable) -> None:
        super().__init__(bt_func=bt_func)

    def optimize(
        self,
        param_space: ParameterSpace,
        metric: str = "Sharpe Ratio",
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        verbose: bool = True,
        **bt_kwargs,
    ) -> OptimizationResult:
        param_names, grids, _ = ParameterSpaceHandler.build_grids(param_space)
        total = int(np.prod([len(g) for g in grids]))

        if verbose:
            print(f"Grid search over {total:,} combinations → {param_names}")

        rows: list[dict[str, Any]] = []
        best_params: dict[str, Any] | None = None
        best_score = -np.inf

        for i, vals in enumerate(product(*grids), 1):
            params = dict(zip(param_names, vals, strict=True))
            if constraint and not constraint(params):
                continue

            try:
                stats = self.bt_func(**bt_kwargs, **params)
                s = self._to_series(stats)
                score = float(s.get(metric, np.nan))
            except Exception as exc:
                if verbose:
                    print(f"  skipped {params} → {exc}")
                continue

            row = dict(params)
            row["Metric"] = score
            row["MetricName"] = metric
            row["SharpeRatio"] = float(s.get("Sharpe Ratio", np.nan))
            row["Return"] = float(s.get("Return [%]", np.nan))
            row["NumTrades"] = int(s.get("# Trades", 0)) if pd.notna(s.get("# Trades", np.nan)) else 0
            row["MaxDrawdown"] = float(s.get("Max. Drawdown [%]", np.nan))
            row["WinRate"] = float(s.get("Win Rate [%]", np.nan))
            row["AvgTrade"] = float(s.get("Avg. Trade [%]", np.nan))
            rows.append(row)

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_params = params

            if verbose and i % max(50, total // 20) == 0 and best_params is not None:
                print(f"[{i}/{total}] Best so far: {best_score:.4f} @ {best_params}")

        if not rows:
            raise ValueError("No valid combinations passed constraint / backtest.")

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values("Metric", ascending=False).reset_index(drop=True)

        if best_params is None:
            best_row = results_df.iloc[0]
            best_params = {p: best_row[p] for p in param_names}
            best_score = float(best_row["Metric"])

        # final evaluation stats at best params (unified)
        final_stats = self.bt_func(**bt_kwargs, **best_params)
        final_s = self._to_series(final_stats)
        best_metric = float(final_s.get(metric, best_score))

        if verbose:
            print(f"\nBest {metric}: {best_metric:.4f}")
            print(f"   Params: {best_params}")

        return self._create_result(
            stats=final_stats,
            params=best_params,
            param_names=param_names,
            results_df=results_df,
            optimizer_info=None,
            strategy=None,
            method="grid",
            best_metric=best_metric,
        )

    # plotting wrappers (delegated to visualizer)
    def plot_heatmap(self, x_param: str, y_param: str, metric_col: str = "SharpeRatio", size: int = 3) -> None:
        if self.results_df is None:
            raise ValueError("No results_df available. Run optimize() first.")
        self.visualizer.plot_heatmap(self.results_df, x_param, y_param, metric_col=metric_col, size=size)

    def plot_param_heatmaps(
        self, params: list[str] | dict[str, Sequence], metric_col: str = "SharpeRatio", size: int = 2
    ) -> None:
        if self.results_df is None:
            raise ValueError("No results_df available. Run optimize() first.")
        self.visualizer.plot_param_heatmaps(self.results_df, params=params, metric_col=metric_col, size=size)

    def plot_smoothed(self, param_col: str, metric_col: str = "SharpeRatio", window: int = 5) -> None:
        if self.results_df is None:
            raise ValueError("No results_df available. Run optimize() first.")
        self.visualizer.plot_smoothed(self.results_df, param_col=param_col, metric_col=metric_col, window=window)

    def plot_session_heatmap(self, metric_col: str = "SharpeRatio", size: int = 3) -> None:
        if self.results_df is None:
            raise ValueError("No results_df available. Run optimize() first.")
        self.visualizer.plot_session_heatmap(self.results_df, metric_col=metric_col, size=size)


class OptunaOptimizer(BaseOptimizer):
    """Bayesian optimization via Optuna over ParameterSpace using bt_func(**bt_kwargs, **params)."""

    def __init__(self, bt_func: Callable) -> None:
        super().__init__(bt_func=bt_func)
        self.study: Any | None = None
        self.param_space: ParameterSpace = {}
        self.metric_name: str = "Sharpe Ratio"
        self._optuna = None

    def _get_optuna(self):
        if self._optuna is None:
            import optuna

            self._optuna = optuna
        return self._optuna

    def optimize(
        self,
        param_space: ParameterSpace,
        metric: str = "Sharpe Ratio",
        n_trials: int = 100,
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        random_state: int | None = RANDOM_STATE,
        verbose: bool = True,
        direction: Literal["maximize", "minimize"] = "maximize",
        prune_on_error: bool = True,
        **bt_kwargs,
    ) -> OptimizationResult:
        optuna = self._get_optuna()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.param_space = dict(param_space)
        self.metric_name = metric
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        def objective(trial: Any) -> float:
            params = {
                name: ParameterSpaceHandler.suggest_for_optuna(trial, name, space)
                for name, space in self.param_space.items()
            }
            if constraint and not constraint(params):
                raise optuna.TrialPruned("Constraint failed")

            try:
                stats = self.bt_func(**bt_kwargs, **params)
                s = self._to_series(stats)
                score = float(s.get(metric, np.nan))
                if not np.isfinite(score):
                    raise optuna.TrialPruned("Non-finite metric")
                return score
            except Exception as exc:
                if prune_on_error:
                    raise optuna.TrialPruned(str(exc)) from exc
                raise

        self.study.optimize(objective, n_trials=int(n_trials), show_progress_bar=verbose)

        best_params = dict(self.study.best_params)
        best_value = float(self.study.best_value)

        # build a clean trials df for visualization (param columns + Metric)
        trials_df = self.study.trials_dataframe(attrs=("number", "value", "params", "state"))
        rename_map = {c: c.replace("params_", "") for c in trials_df.columns if c.startswith("params_")}
        trials_df = trials_df.rename(columns=rename_map)
        trials_df["Metric"] = trials_df["value"].astype(float)
        trials_df["MetricName"] = metric
        if metric.lower().startswith("sharpe"):
            trials_df["SharpeRatio"] = trials_df["Metric"]
        self.results_df = trials_df

        # final evaluation stats at best params (unified)
        final_stats = self.bt_func(**bt_kwargs, **best_params)
        final_s = self._to_series(final_stats)
        best_metric = float(final_s.get(metric, best_value))

        if verbose:
            print(f"Best {metric}: {best_metric:.4f}")
            print(f"Best params: {best_params}")

        return self._create_result(
            stats=final_stats,
            params=best_params,
            param_names=list(best_params.keys()),
            results_df=trials_df,
            optimizer_info=self.study,
            strategy=None,
            method="optuna",
            best_metric=best_metric,
        )


class PortfolioOptimizer(BaseOptimizer):
    """
    Optimizes strategy parameters at the portfolio level using run_full_suite_berlin_tz.

    aggregate modes:
    - 'mean': average metric across symbols within each timeframe; then mean across timeframes
    - 'portfolio': compute metric on "Portfolio_EqualWeight" equity curve (forces create_portfolios=True)
    """

    def __init__(self, run_suite_kwargs: dict, aggregate: Literal["mean", "portfolio"] = "portfolio") -> None:
        super().__init__(bt_func=None)
        self._run_suite_kwargs = dict(run_suite_kwargs)
        self.aggregate = aggregate
        if aggregate == "portfolio":
            self._run_suite_kwargs["create_portfolios"] = True

        self.study: Any | None = None
        self.trials_df: pd.DataFrame | None = None

    @staticmethod
    def _make_strategy(base_cls: type, params: dict[str, Any]) -> type:
        return type(f"_PortOpt_{base_cls.__name__}", (base_cls,), dict(params))

    @staticmethod
    def _score_equity(equity: pd.Series, metric: str) -> float:
        from blackwood.metrics.core import compute_all_metrics

        daily = equity.resample("D").last().dropna()
        if len(daily) < 10:
            return float("nan")
        try:
            stats = compute_all_metrics(daily)
            return float(stats.get(metric, float("nan")))
        except Exception:
            return float("nan")

    def _evaluate_params(self, strategy_class: type, params: dict[str, Any], metric: str) -> float:
        from blackwood.data.loaders import run_full_suite_berlin_tz

        try:
            configured_cls = self._make_strategy(strategy_class, params)
            _, equity_by_tf = run_full_suite_berlin_tz(configured_cls, **self._run_suite_kwargs)
        except Exception:
            return float("nan")

        scores: list[float] = []
        for _tf, equity_map in equity_by_tf.items():
            if self.aggregate == "portfolio":
                eq = equity_map.get("Portfolio_EqualWeight")
                if eq is None:
                    continue
                s = self._score_equity(eq, metric)
                if np.isfinite(s):
                    scores.append(s)
            else:
                tf_scores: list[float] = []
                for key, eq in equity_map.items():
                    if key.startswith("Portfolio_") or key.startswith("Benchmark_"):
                        continue
                    s = self._score_equity(eq, metric)
                    if np.isfinite(s):
                        tf_scores.append(s)
                if tf_scores:
                    scores.append(float(np.mean(tf_scores)))

        return float(np.mean(scores)) if scores else float("nan")

    def _optimize_optuna(
        self,
        strategy_class: type,
        param_space: ParameterSpace,
        metric: str,
        method: Literal["sambo", "optuna"],
        n_trials: int,
        constraint: Callable[[dict[str, Any]], bool] | None,
        random_state: int,
        verbose: bool,
        direction: Literal["maximize", "minimize"],
    ) -> OptimizationResult:
        import optuna as _optuna

        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        if method == "sambo":
            try:
                sampler = _optuna.samplers.GPSampler(seed=random_state)
            except AttributeError:
                sampler = _optuna.samplers.TPESampler(seed=random_state)
        else:
            sampler = _optuna.samplers.TPESampler(seed=random_state)

        self.study = _optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial: Any) -> float:
            params = {
                name: ParameterSpaceHandler.suggest_for_optuna(trial, name, space)
                for name, space in param_space.items()
            }
            if constraint and not constraint(params):
                raise _optuna.TrialPruned("Constraint failed")

            score = self._evaluate_params(strategy_class, params, metric)
            if not np.isfinite(score):
                raise _optuna.TrialPruned("Non-finite metric")
            return float(score)

        self.study.optimize(objective, n_trials=int(n_trials), show_progress_bar=verbose)
        best_params = dict(self.study.best_params)
        best_value = float(self.study.best_value)

        trials_df = self.study.trials_dataframe(attrs=("number", "value", "params", "state"))
        rename_map = {c: c.replace("params_", "") for c in trials_df.columns if c.startswith("params_")}
        trials_df = trials_df.rename(columns=rename_map)
        trials_df["Metric"] = trials_df["value"].astype(float)
        trials_df["MetricName"] = metric
        self.trials_df = trials_df

        stats = pd.Series(
            {
                "Objective": best_value,
                "MetricName": metric,
                "Aggregate": self.aggregate,
                **best_params,
            }
        )

        if verbose:
            print(f"Best {metric}: {best_value:.4f}")
            print(f"Best params: {best_params}")

        return self._create_result(
            stats=stats,
            params=best_params,
            param_names=list(best_params.keys()),
            results_df=trials_df,
            optimizer_info=self.study,
            strategy=None,
            method=method,
            best_metric=best_value,
        )

    def _optimize_grid(
        self,
        strategy_class: type,
        param_space: ParameterSpace,
        metric: str,
        constraint: Callable[[dict[str, Any]], bool] | None,
        verbose: bool,
        direction: Literal["maximize", "minimize"],
    ) -> OptimizationResult:
        param_names, grids, _ = ParameterSpaceHandler.build_grids(param_space)
        total = int(np.prod([len(g) for g in grids]))

        if verbose:
            print(f"Portfolio grid search over {total:,} combinations → {param_names}")

        rows: list[dict[str, Any]] = []
        best_params: dict[str, Any] | None = None
        best_score = -np.inf if direction == "maximize" else np.inf

        for i, vals in enumerate(product(*grids), 1):
            params = dict(zip(param_names, vals, strict=True))
            if constraint and not constraint(params):
                continue

            score = self._evaluate_params(strategy_class, params, metric)
            row = dict(params)
            row["Metric"] = float(score) if np.isfinite(score) else float("nan")
            row["MetricName"] = metric
            rows.append(row)

            if np.isfinite(score):
                if direction == "maximize" and score > best_score:
                    best_score, best_params = float(score), params
                if direction == "minimize" and score < best_score:
                    best_score, best_params = float(score), params

            if verbose:
                msg = (
                    f"[{i}/{total}] params={params} → {metric}={score:.4f}"
                    if np.isfinite(score)
                    else f"[{i}/{total}] params={params} → failed"
                )
                print(msg)

        if not rows:
            raise ValueError("No valid combinations produced a result.")

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values("Metric", ascending=(direction == "minimize")).reset_index(drop=True)

        if best_params is None:
            best_row = results_df.dropna(subset=["Metric"]).iloc[0]
            best_params = {p: best_row[p] for p in param_names}
            best_score = float(best_row["Metric"])

        stats = pd.Series(
            {
                "Objective": best_score,
                "MetricName": metric,
                "Aggregate": self.aggregate,
                **best_params,
            }
        )

        if verbose:
            print(f"\nBest {metric}: {best_score:.4f}")
            print(f"   Params: {best_params}")

        return self._create_result(
            stats=stats,
            params=best_params,
            param_names=param_names,
            results_df=results_df,
            optimizer_info=None,
            strategy=None,
            method="grid",
            best_metric=best_score,
        )

    def optimize(
        self,
        strategy_class: type,
        param_space: ParameterSpace,
        metric: str = "Sharpe Ratio",
        method: Literal["sambo", "optuna", "grid"] = "sambo",
        n_trials: int = 60,
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
        direction: Literal["maximize", "minimize"] = "maximize",
    ) -> OptimizationResult:
        if method in ("sambo", "optuna"):
            return self._optimize_optuna(
                strategy_class=strategy_class,
                param_space=param_space,
                metric=metric,
                method=method,
                n_trials=n_trials,
                constraint=constraint,
                random_state=random_state,
                verbose=verbose,
                direction=direction,
            )
        if method == "grid":
            return self._optimize_grid(
                strategy_class=strategy_class,
                param_space=param_space,
                metric=metric,
                constraint=constraint,
                verbose=verbose,
                direction=direction,
            )
