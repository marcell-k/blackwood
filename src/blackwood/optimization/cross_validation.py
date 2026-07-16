from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from blackwood.config import CASH
from blackwood.data.splitters import CPCVSplitter
from blackwood.optimization.walk_forward import WalkForwardOptimizer
from blackwood.typing import CPCVPaths

if TYPE_CHECKING:
    from backtesting import Strategy

if TYPE_CHECKING:
    from backtesting import Strategy


class CPCVAnalyzer:
    """
    Stateful orchestrator for Combinatorial Purged Cross-Validation (CPCV) analysis
    with Walk-Forward Optimization (WFO).

    Encapsulates CPCV splitting, per-fold optimization, OOS testing, and robust
    path-level metrics aggregation. Results are cached as instance state for
    repeated queries without re-computation.

    Workflow:
        1. Instantiate with strategies and optimization configuration
        2. Call run_analysis(df) to execute CPCV+WFO
        3. Query results via get_path_results(), get_aggregated_metrics(), get_best_path()
    """

    def __init__(
        self,
        splitter: CPCVSplitter,
        base_strategy: type[Strategy],
        kelly_strategy: type[Strategy],
        maximize: str = "Rating",
        max_tries: int = 40,
        trade_on_close: bool = True,
        exclusive_orders: bool = False,
        evaluate_kelly: bool = True,
        initial_cash: float = CASH,
    ) -> None:
        """
        Initialize CPCV analyzer with strategies and optimization configuration.
        """
        self.splitter = splitter
        self.base_strategy = base_strategy
        self.kelly_strategy = kelly_strategy
        self.maximize = maximize
        self.max_tries = max_tries
        self.trade_on_close = trade_on_close
        self.exclusive_orders = exclusive_orders
        self.evaluate_kelly = evaluate_kelly
        self.initial_cash = float(initial_cash)

        self._path_results: dict[int, dict] | None = None
        self._aggregated_metrics: dict[str, float] | None = None
        self._best_path_info: dict[str, Any] | None = None

    def run_analysis(self, df: pd.DataFrame) -> CPCVAnalyzer:
        """Execute end-to-end CPCV analysis: validate, split paths, run WFO, aggregate metrics."""
        # Split CPCV paths with purge+embargo
        cpcv_paths = CPCVSplitter.split_paths(self.splitter, df)

        # Run WFO across all paths and cache results
        self._path_results, self._aggregated_metrics, self._best_path_info = self._run_wfo_on_paths(cpcv_paths)

        return self

    def get_path_results(self):
        """Retrieve per-path results (fold count, final equity, path metrics, parameters)."""
        return self._path_results

    def get_aggregated_metrics(self):
        """Retrieve cross-path aggregated metrics (mean/median return, Sharpe, Calmar, etc.)."""
        return self._aggregated_metrics

    def get_best_path(self):
        """Retrieve best path selection based on composite normalized scoring."""
        return self._best_path_info

    # ---------- Time/Calendar Helpers (vectorized) ----------

    @staticmethod
    def _months_between(ts_start: pd.Timestamp, ts_end: pd.Timestamp) -> float:
        """
        Approximate fractional calendar months between timestamps.
        """
        if pd.isna(ts_start) or pd.isna(ts_end) or ts_end <= ts_start:
            return 0.0
        delta_days = (ts_end - ts_start).total_seconds() / 86400.0
        return float(delta_days / 30.4375)

    def _total_test_months(self, test_dfs: list[pd.DataFrame]) -> float:
        """
        Sum fractional months across disjoint test folds.
        """
        months = 0.0
        for df_ in test_dfs:
            if len(df_) > 0:
                months += self._months_between(df_.index.min(), df_.index.max())
        return months

    # ---------- Robust Aggregation Helpers (vectorized, finite-only) ----------

    @staticmethod
    def _nanmean(a: np.ndarray) -> float:
        """Mean of finite values only."""
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else np.nan

    @staticmethod
    def _nanmedian(a: np.ndarray) -> float:
        """Median of finite values only."""
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else np.nan

    @staticmethod
    def _nanstd(a: np.ndarray) -> float:
        """Standard deviation of finite values only."""
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.std()) if a.size else np.nan

    @staticmethod
    def _consistency(arr: np.ndarray, mean_val: float) -> float:
        """Consistency measure for path stability across folds."""
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2 or not np.isfinite(mean_val):
            return 0.0
        std_val = np.nanstd(arr, ddof=1)
        if std_val == 0:
            return 1.0
        cv = abs(std_val / mean_val) if mean_val != 0 else np.inf
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    # ---------- Metrics Calculation ----------

    def _calculate_path_metrics(
        self, stats_test_list: list[dict], final_equity: float, total_test_months: float
    ) -> dict[str, float]:
        """Aggregate fold-level OOS stats into robust path-level metrics."""
        if len(stats_test_list) == 0:
            return {}

        def _col(col: str) -> np.ndarray:
            return np.array([float(s.get(col, np.nan)) for s in stats_test_list], dtype=float)

        trades_arr = _col("# Trades")
        ret_arr = _col("Return (Ann.) [%]")
        sharpe_arr = _col("Sharpe Ratio")
        calmar_arr = _col("Calmar Ratio")
        dd_arr = _col("Max. Drawdown [%]")
        winr_arr = _col("Win Rate [%]")
        pf_arr = _col("Profit Factor")

        total_trades = int(np.nansum(trades_arr))
        avg_return = self._nanmean(ret_arr)
        avg_sharpe = self._nanmean(sharpe_arr)
        avg_calmar = self._nanmean(calmar_arr)
        med_sharpe = self._nanmedian(sharpe_arr)
        med_calmar = self._nanmedian(calmar_arr)
        avg_winr = self._nanmean(winr_arr)
        avg_pf = self._nanmean(pf_arr)
        med_pf = self._nanmedian(pf_arr)

        worst_dd_val = float(np.nanmin(dd_arr))
        worst_dd_mag = abs(worst_dd_val)
        total_return_pct = ((float(final_equity) / self.initial_cash) - 1.0) * 100.0
        trades_per_fold = total_trades / len(stats_test_list)
        trades_per_month = (total_trades / total_test_months) if total_test_months > 0 else np.nan

        returns_consistency = self._consistency(ret_arr, avg_return)
        sharpe_consistency = self._consistency(sharpe_arr, avg_sharpe)
        calmar_consistency = self._consistency(calmar_arr, avg_calmar)

        return {
            "total_return_pct": total_return_pct,
            "avg_annual_return": avg_return,
            "avg_sharpe_ratio": avg_sharpe,
            "median_sharpe_ratio": med_sharpe,
            "avg_calmar_ratio": avg_calmar,
            "median_calmar_ratio": med_calmar,
            "max_drawdown": worst_dd_val,
            "max_drawdown_abs": worst_dd_mag,
            "avg_win_rate": avg_winr,
            "avg_profit_factor": avg_pf,
            "median_profit_factor": med_pf,
            "total_trades": total_trades,
            "trades_per_fold": trades_per_fold,
            "trades_per_month": trades_per_month,
            "returns_consistency": returns_consistency,
            "sharpe_consistency": sharpe_consistency,
            "calmar_consistency": calmar_consistency,
            "n_folds": len(stats_test_list),
            "processing_time": np.nan,  # Timing handled externally; placeholder for compatibility
        }

    def _aggregate_across_paths(self, aggregated_data: dict[str, list[float]]) -> dict[str, float]:
        """Aggregate path-level metrics with finite-only robust summaries across CPCV paths."""
        if len(aggregated_data.get("final_equities", [])) == 0:
            return {}

        final_equities = np.asarray(aggregated_data["final_equities"], dtype=float)
        returns = ((final_equities / self.initial_cash) - 1.0) * 100.0
        returns = returns[np.isfinite(returns)]

        dd_vals = np.asarray(aggregated_data.get("max_drawdown", []), dtype=float)
        dd_vals = dd_vals[np.isfinite(dd_vals)]

        sharpe_vals = np.asarray(aggregated_data.get("avg_sharpe_ratio", []), dtype=float)
        calmar_vals = np.asarray(aggregated_data.get("avg_calmar_ratio", []), dtype=float)
        pf_vals = np.asarray(aggregated_data.get("avg_profit_factor", []), dtype=float)
        pf_vals = pf_vals[np.isfinite(pf_vals)]
        tpm_vals = np.asarray(aggregated_data.get("trades_per_month", []), dtype=float)
        tpm_vals = tpm_vals[np.isfinite(tpm_vals)]
        proc_times = np.asarray(aggregated_data.get("processing_times", []), dtype=float)

        return {
            "mean_return": float(returns.mean()) if returns.size else np.nan,
            "median_return": float(np.median(returns)) if returns.size else np.nan,
            "std_return": float(returns.std()) if returns.size else np.nan,
            "min_return": float(np.min(returns)) if returns.size else np.nan,
            "max_return": float(np.max(returns)) if returns.size else np.nan,
            "positive_paths_pct": float((returns > 0).mean() * 100.0) if returns.size else np.nan,
            "mean_sharpe": self._nanmean(sharpe_vals),
            "median_sharpe": self._nanmedian(sharpe_vals),
            "std_sharpe": self._nanstd(sharpe_vals),
            "mean_calmar": self._nanmean(calmar_vals),
            "median_calmar": self._nanmedian(calmar_vals),
            "std_calmar": self._nanstd(calmar_vals),
            "mean_max_dd": self._nanmean(dd_vals),
            "worst_max_dd": float(np.nanmin(dd_vals)) if dd_vals.size else np.nan,
            "pf_mean": self._nanmean(pf_vals),
            "pf_median": self._nanmedian(pf_vals),
            "trades_per_month_mean": self._nanmean(tpm_vals),
            "trades_per_month_median": self._nanmedian(tpm_vals),
            "total_paths": len(aggregated_data["final_equities"]),
            "avg_processing_time": float(proc_times.mean()) if proc_times.size else np.nan,
            "total_processing_time": float(proc_times.sum()) if proc_times.size else 0.0,
        }

    def _score_paths(self, path_results: dict[int, dict]) -> dict[str, Any]:
        """Identify best performing path via composite normalized scoring."""
        if not path_results:
            return {}

        weights = {
            "avg_calmar_ratio": 0.15,
            "returns_consistency": 0.3,
            "avg_sharpe_ratio": 0.40,
            "avg_win_rate": 0.15,
        }

        # Collect metric arrays for normalization
        metric_arrays: dict[str, np.ndarray] = {}
        for m in weights:
            vals = [res["path_metrics"].get(m, np.nan) for res in path_results.values()]
            metric_arrays[m] = np.asarray(vals, dtype=float)

        # Compute normalized composite scores
        scores: dict[int, float] = {}
        for pid, res in path_results.items():
            s = 0.0
            for m, w in weights.items():
                arr = metric_arrays[m]
                v = float(res["path_metrics"].get(m, np.nan))
                if not np.isfinite(v):
                    continue
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                mn, mx = float(np.min(finite)), float(np.max(finite))
                norm = 1.0 if mx == mn else (v - mn) / (mx - mn)
                s += w * max(0.0, norm)
            scores[pid] = s

        best_id = max(scores.keys(), key=lambda k: scores[k])
        return {
            "best_path_id": best_id,
            "best_path_score": scores[best_id],
            "best_path_metrics": path_results[best_id]["path_metrics"],
            "all_path_scores": scores,
        }

    # ---------- CPCV Split + WFO Orchestration ----------

    def _run_wfo_on_paths(self, cpcv_paths: CPCVPaths) -> tuple[dict[int, dict], dict[str, float], dict[str, Any]]:
        """Run Enhanced WFO across CPCV paths."""
        path_results: dict[int, dict] = {}
        aggregated_data: dict[str, list[float]] = {"final_equities": [], "processing_times": [], "path_ids": []}

        for pid, folds in cpcv_paths.items():
            if len(folds) == 0:
                continue

            # Validate and collect folds for this path
            train_dfs: list[pd.DataFrame] = []
            test_dfs: list[pd.DataFrame] = []
            for train_df, test_df in folds:
                train_dfs.append(train_df)
                test_dfs.append(test_df)

            total_test_months = self._total_test_months(test_dfs)

            # Instantiate Enhanced WFO for this path and run once
            wfo = WalkForwardOptimizer(train_dfs=train_dfs, test_dfs=test_dfs)
            (
                _,  # trades_test_base
                _,  # stats_train_list_base
                stats_test_list_base,
                parameters,
                _,  # bt_test_list_base
                _,  # param_names
                _,  # optimize_results
            ) = wfo.run_wfo(
                base_strategy=self.base_strategy,
                params=None,
                max_tries=self.max_tries,
                maximize=self.maximize,
                verbose=False,
            )
            final_equity = (
                stats_test_list_base[-1].get("Equity Final [$]", self.initial_cash)
                if stats_test_list_base
                else self.initial_cash
            )

            # Optionally run again for the Kelly-sized strategy
            stats_test_list_kelly: list[dict] = []
            sizes: list[Any] = []
            if self.evaluate_kelly:
                (
                    _,  # trades_test_kelly
                    _,  # stats_train_list_kelly
                    stats_test_list_kelly,
                    _,  # parameters_kelly
                    _,  # bt_test_list_kelly
                    _,  # param_names
                    _,  # optimize_results
                ) = wfo.run_wfo(
                    base_strategy=self.kelly_strategy,
                    params=None,
                    max_tries=self.max_tries,
                    maximize=self.maximize,
                    verbose=False,
                )
                if stats_test_list_kelly:
                    final_equity = stats_test_list_kelly[-1].get("Equity Final [$]", final_equity)

            # Choose which OOS stats stream to evaluate
            stats_oos_list = stats_test_list_kelly if self.evaluate_kelly else stats_test_list_base

            # Compose path metrics
            path_metrics = self._calculate_path_metrics(
                stats_test_list=stats_oos_list, final_equity=float(final_equity), total_test_months=total_test_months
            )

            result = {
                "path_id": pid,
                "n_folds": len(folds),
                "final_equity": float(final_equity),
                "path_metrics": path_metrics,
                "stats_test_base": stats_test_list_base,
                "stats_test_kelly": stats_test_list_kelly,
                "parameters": parameters,
                "kelly_sizes": sizes,
            }
            path_results[pid] = result

            # Aggregate fields across paths
            aggregated_data["final_equities"].append(float(final_equity))
            aggregated_data["processing_times"].append(path_metrics.get("processing_time", np.nan))
            aggregated_data["path_ids"].append(pid)
            for k, v in path_metrics.items():
                aggregated_data.setdefault(k, []).append(v)

        aggregated_metrics = self._aggregate_across_paths(aggregated_data)
        best_path_info = self._score_paths(path_results) if aggregated_metrics else {}
        return path_results, aggregated_metrics, best_path_info
