from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from blackwood.config import CASH, IS_MONTHS, MARGIN, OOS_MONTHS
from blackwood.data.splitters import CPCVSplitter
from blackwood.optimization.optimization import SamboOptimizer
from blackwood.typing import CPCVPaths

if TYPE_CHECKING:
    from collections.abc import Callable

    from backtesting import Backtest, Strategy

    from blackwood.metrics.core import Stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from backtesting import Backtest, Strategy

    from blackwood.metrics.core import Stats


class WalkForwardOptimizer:
    """
    Unified walk-forward optimization framework with SAMBO hyperparameter search
    and Walk-Forward Efficiency (WFE) reporting.
    Supports standard and validated modes.
    """

    def __init__(
        self,
        train_dfs: list[pd.DataFrame] | None = None,
        test_dfs: list[pd.DataFrame] | None = None,
        trade_on_close: bool = True,
        exclusive_orders: bool = False,
        commission: float | tuple[float, float] | Callable = (3.5 / 100_000, 0),
        margin: float = MARGIN,
        spread: float = 0.0,
        cash: float = CASH,
        is_months: int = IS_MONTHS,
        oos_months: int = OOS_MONTHS,
        compute_stats_func: Callable | None = None,
        bt_func: Callable | None = None,
    ) -> None:
        self.train_dfs = train_dfs or []
        self.test_dfs = test_dfs or []
        self.cash = float(cash)
        self.is_months = is_months
        self.oos_months = oos_months
        self.compute_stats_func = compute_stats_func

        # State
        self.optimize_results: list[Any] = []
        self.param_names: list[str] = []
        self.parameters: list[list[Any]] = []
        self.stats_train_list: list[dict[str, Any]] = []
        self.stats_test_list: list[dict[str, Any]] = []

        # SAMBO optimizer instance
        self.optimizer = SamboOptimizer(
            margin=margin,
            spread=spread,
            commission=commission,
            trade_on_close=trade_on_close,
            exclusive_orders=exclusive_orders,
            bt_func=bt_func,
        )

    @staticmethod
    def _trades_df(stats: Stats) -> pd.DataFrame:
        if isinstance(stats, dict) and "_trades" in stats:
            return stats["_trades"].copy()
        return getattr(stats, "_trades", pd.DataFrame())

    @staticmethod
    def _fmt_period(d: pd.DataFrame) -> str:
        if d.empty or not hasattr(d.index, "min"):
            return "N/A"
        return f"{d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}"

    def run_wfo(
        self,
        base_strategy: type[Strategy],
        params: dict[str, tuple | list] | None = None,
        constraint: Callable | None = None,
        max_tries: int = 200,
        maximize: str = "Sharpe Ratio",
        verbose: bool = True,
        use_validation: bool = False,
        val_dfs: list[pd.DataFrame] | None = None,
    ) -> tuple[
        pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[list[Any]], list[Backtest], list[str], list[Any]
    ]:
        if use_validation:
            if val_dfs is None or len(val_dfs) != len(self.train_dfs):
                raise ValueError(f"use_validation=True requires val_dfs of length {len(self.train_dfs)}")
            periods = zip(self.train_dfs, val_dfs, self.test_dfs, strict=True)
            has_validation = True
        else:
            periods = zip(self.train_dfs, self.test_dfs, strict=True)
            has_validation = False

        self.optimize_results = []
        self.parameters = []
        self.stats_train_list = []
        self.stats_test_list = []
        param_names: list[str] = []
        trades_test_list: list[pd.DataFrame] = []
        bt_test_list: list[Backtest] = []

        current_equity = float(self.cash)
        peak_equity = float(self.cash)
        n_periods = len(self.train_dfs)

        for idx, period in enumerate(periods):
            start_equity = current_equity

            if has_validation:
                train_df, val_df, test_df = period
            else:
                train_df, test_df = period
                val_df = None

            # Consistently carry forward peak equity to the current train period
            class TrainStrategy(base_strategy):
                initial_peak_equity = peak_equity

            stats_train, OptStrat, opt_vec, _, temp_names, optres = self.optimizer.optimize(
                df=train_df,
                strategy_class=TrainStrategy,
                cash=start_equity,
                params=params,
                constraint=constraint,
                max_tries=max_tries,
                maximize=maximize,
            )

            if not param_names:
                param_names = temp_names
            if optres is not None:
                self.optimize_results.append(optres)
            self.parameters.append(list(opt_vec) if opt_vec else [])

            stats_train_dict = stats_train.to_dict() if isinstance(stats_train, pd.Series) else dict(stats_train)
            stats_train_dict["Starting Equity [$]"] = start_equity
            self.stats_train_list.append(stats_train_dict)

            val_stats_dict: dict[str, Any] | None = None
            if val_df is not None:
                bt_val = self.optimizer.backtest(val_df, OptStrat, cash=start_equity)
                val_stats = bt_val.run()
                val_stats_dict = val_stats.to_dict() if isinstance(val_stats, pd.Series) else dict(val_stats)
                val_stats_dict["Starting Equity [$]"] = start_equity

            # Test period uses OptStrat (inherits initial_peak_equity from TrainStrategy)
            bt_test = self.optimizer.backtest(test_df, OptStrat, cash=start_equity)
            stats_test = bt_test.run()
            stats_test_dict = stats_test.to_dict() if isinstance(stats_test, pd.Series) else dict(stats_test)
            stats_test_dict["Starting Equity [$]"] = start_equity
            self.stats_test_list.append(stats_test_dict)
            bt_test_list.append(bt_test)
            trades_test_list.append(self._trades_df(stats_test))

            current_equity = float(stats_test_dict.get("Equity Final [$]", current_equity))
            peak_equity = max(peak_equity, float(stats_test_dict.get("Equity Peak [$]", peak_equity)))

            if verbose:
                print("\n" + "-" * 70)
                print(f"PROCESSING PERIOD {idx + 1}/{n_periods}")
                if has_validation:
                    print(
                        f"Train: {self._fmt_period(train_df)} | Val: {self._fmt_period(val_df)} | "
                        f"Test: {self._fmt_period(test_df)}"
                    )
                    cols = ["Train", "Validation", "Test"]
                    stats_sources = [stats_train_dict, val_stats_dict, stats_test_dict]
                else:
                    print(f"Train: {self._fmt_period(train_df)} | Test: {self._fmt_period(test_df)}")
                    cols = ["Train", "Test"]
                    stats_sources = [stats_train_dict, stats_test_dict]

                print("=" * 70)
                print(f"{'Metric':<20} " + " ".join(f"{c:>15}" for c in cols))
                print("-" * 70)
                for label, key in [
                    ("Annual Return %", "Return (Ann.) [%]"),
                    ("Max Drawdown %", "Max. Drawdown [%]"),
                    ("# Trades", "# Trades"),
                    ("Profit Factor", "Profit Factor"),
                ]:
                    vals = [s.get(key, np.nan) for s in stats_sources]
                    fmt = "{:>15.0f}" if key == "# Trades" else "{:>15.2f}"
                    print(f"{label:<20} " + " ".join(fmt.format(v) for v in vals))
                if "Rating" in stats_train_dict:
                    ratings = [s.get("Rating", 0) for s in stats_sources]
                    print(f"{'Rating':<20} " + " ".join(f"{r:>15.2f}" for r in ratings))
                print("-" * 70)

        trades_combined = pd.concat(trades_test_list, ignore_index=True) if trades_test_list else pd.DataFrame()
        self.param_names = param_names
        return (
            trades_combined,
            self.stats_train_list,
            self.stats_test_list,
            self.parameters,
            bt_test_list,
            param_names,
            self.optimize_results,
        )

    @staticmethod
    def _calc_monthly_return(stats_dict: dict[str, Any], months: int) -> float:
        equity_final = float(stats_dict.get("Equity Final [$]", np.nan))
        start_equity = float(stats_dict.get("Starting Equity [$]", np.nan))
        if not np.isfinite(start_equity) or start_equity <= 0:
            return np.nan
        total_return = (equity_final - start_equity) / start_equity
        if not np.isfinite(total_return):
            return np.nan
        if total_return <= -1.0:
            return -1.0
        if total_return < 0:
            return total_return / months
        return (1.0 + total_return) ** (1.0 / months) - 1.0

    def wfo_efficiency(
        self,
        metric: str,
        stats_train_list: list[dict],
        stats_test_list: list[dict],
        train_months: int,
        test_months: int,
        is_min_trades: int,
        oos_min_trades: int,
    ) -> tuple[float, float, float, float, list[tuple[float, float]]] | None:
        """Compute Walk-Forward Efficiency (WFE) for a single metric."""
        pairs: list[tuple[float, float]] = []
        is_wins = is_trades = oos_wins = oos_trades = 0

        for st, ts in zip(stats_train_list, stats_test_list, strict=True):
            tr_is = int(pd.to_numeric(st.get("# Trades", np.nan), errors="coerce") or 0)
            tr_oos = int(pd.to_numeric(ts.get("# Trades", np.nan), errors="coerce") or 0)
            if tr_is < is_min_trades or tr_oos < oos_min_trades:
                continue

            if metric == "Win Rate [%]":
                is_wr = pd.to_numeric(st.get("Win Rate [%]", np.nan), errors="coerce")
                oos_wr = pd.to_numeric(ts.get("Win Rate [%]", np.nan), errors="coerce")
                if not np.isfinite(is_wr) or not np.isfinite(oos_wr):
                    continue
                iw = int(np.rint((is_wr / 100.0) * tr_is))
                ow = int(np.rint((oos_wr / 100.0) * tr_oos))
                is_wins += iw
                is_trades += tr_is
                oos_wins += ow
                oos_trades += tr_oos
                pairs.append((float(is_wr), float(oos_wr)))
                continue

            if metric == "Monthly Return":
                is_m = self._calc_monthly_return(st, train_months)
                oos_m = self._calc_monthly_return(ts, test_months)
                if not np.isfinite(is_m) or not np.isfinite(oos_m):
                    continue
                pairs.append((float(is_m), float(oos_m)))
                continue

            is_val = pd.to_numeric(st.get(metric, np.nan), errors="coerce")
            oos_val = pd.to_numeric(ts.get(metric, np.nan), errors="coerce")
            if not np.isfinite(is_val) or not np.isfinite(oos_val):
                continue
            pairs.append((float(is_val), float(oos_val)))

        if not pairs:
            return None

        is_vals = np.array([i for i, _ in pairs], dtype=float)
        oos_vals = np.array([o for _, o in pairs], dtype=float)
        avg_is = float(np.mean(is_vals))
        avg_oos = float(np.mean(oos_vals))
        std_oos = float(np.std(oos_vals))

        if metric == "Win Rate [%]":
            agg_is = (is_wins / is_trades) * 100.0 if is_trades > 0 else np.nan
            agg_oos = (oos_wins / oos_trades) * 100.0 if oos_trades > 0 else np.nan
            wfe = float(agg_oos / agg_is) if np.isfinite(agg_is) and agg_is != 0 else np.nan
            return wfe, float(agg_is), float(agg_oos), std_oos, pairs

        if metric == "Monthly Return":
            s_is = float(np.sum(is_vals))
            s_oos = float(np.sum(oos_vals))
            wfe = float(s_oos / s_is) if s_is != 0 else np.nan
            return wfe, avg_is, avg_oos, std_oos, pairs

        ratios = [o / i for i, o in pairs if i != 0]
        wfe = float(np.mean(ratios)) if ratios else np.nan
        return wfe, avg_is, avg_oos, std_oos, pairs

    def wfo_summary_table(
        self,
        stats_train_list: list[dict] | None = None,
        stats_test_list: list[dict] | None = None,
        metrics: list[str] | None = None,
        title: str = "STRATEGY",
        show_period: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, list[tuple[float, float]]]]:
        stats_train_list = stats_train_list or self.stats_train_list
        stats_test_list = stats_test_list or self.stats_test_list
        train_period_months = self.is_months
        test_period_months = self.oos_months
        is_min_trades = 20
        oos_min_trades = 5
        metrics = metrics or ["Monthly Return", "Profit Factor", "Sortino Ratio", "Calmar Ratio", "Win Rate [%]"]

        rows = []
        details: dict[str, list[tuple[float, float]]] = {}
        total_periods = len(stats_train_list)
        results: dict[str, tuple[float, float, float, float, list[tuple[float, float]]]] = {}

        for m in metrics:
            r = self.wfo_efficiency(
                m,
                stats_train_list,
                stats_test_list,
                train_period_months,
                test_period_months,
                is_min_trades,
                oos_min_trades,
            )
            if r is None:
                continue
            wfe, is_avg, oos_avg, oos_std, pairs = r
            results[m] = r
            rows.append(
                {
                    "Metric": m,
                    "WFE": wfe,
                    "IS Avg": is_avg,
                    "OOS Avg": oos_avg,
                    "OOS Std": oos_std,
                    "Periods": f"{len(pairs)}/{total_periods}",
                }
            )
            details[m] = pairs

        scorecard = pd.DataFrame(rows, columns=["Metric", "WFE", "IS Avg", "OOS Avg", "OOS Std", "Periods"])

        # FORMATTED OUTPUT (unchanged for consistency)
        line = "=" * 120
        print(f"\n{line}")
        print(f"{'WFE SCORECARD':^120}")
        print(
            f"{(title.upper() + ' | Training: ' + str(train_period_months) + 'mo | Testing: ' + str(test_period_months) + 'mo | Periods: ' + str(total_periods)):^120}"  # noqa: E501
        )
        print(f"{line}")
        print("\nPERFORMANCE SUMMARY")
        print(f"{'-' * 120}")
        print(f"{'Metric':<15} | {'WFE':<8} {'IS Avg':<10} {'OOS Avg':<10} {'OOS Std':<8} {'Periods':<8}")
        print(f"{'-' * 120}")

        def fmt_row(m: str, pack: tuple | None) -> str:
            if pack is None:
                return f"{m:<15} | {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8}"
            wfe, is_avg, oos_avg, oos_std, pairs = pack
            periods = f"{len(pairs)}/{total_periods}"
            if m == "Monthly Return":
                return f"{m:<15} | {wfe:<8.3f} {is_avg:<10.4f} {oos_avg:<10.4f} {oos_std:<8.4f} {periods:<8}"
            if m == "Win Rate [%]":
                return f"{m:<15} | {wfe:<8.3f} {is_avg:<10.1f} {oos_avg:<10.1f} {oos_std:<8.1f} {periods:<8}"
            return f"{m:<15} | {wfe:<8.3f} {is_avg:<10.2f} {oos_avg:<10.2f} {oos_std:<8.2f} {periods:<8}"

        for m in metrics:
            print(fmt_row(m, results.get(m)))

        # Trade counts per period
        print("\nTRADE COUNTS PER PERIOD")
        print(f"{'-' * 56}")
        print(f"{'Period':<8} | {'Train':<8} {'Test':<8}")
        print(f"{'-' * 56}")

        def get_trades(lst: list[dict], idx: int) -> int:
            if 0 <= idx < len(lst):
                d = lst[idx] if lst[idx] is not None else {}
                return int(d.get("# Trades", 0))
            return 0

        total_periods_all = max(len(stats_train_list), len(stats_test_list))
        for i in range(total_periods_all):
            t_is = get_trades(stats_train_list, i)
            t_oos = get_trades(stats_test_list, i)
            print(f"P{i:<7} | {t_is:<8} {t_oos:<8}")

        if show_period:
            print("\nPERIOD-BY-PERIOD DETAILED ANALYSIS")
            print(f"{'=' * 100}")
            for m in metrics:
                if m not in results:
                    continue
                pairs = results[m][4]
                print(f"\n{m.upper()}")
                print(f"{'-' * 80}")
                print(f"{'Period':<8} {'Train':<7} {'Test':<7} {'WFE':<6}")
                print(f"{'-' * 80}")
                idx = 0
                for p in range(total_periods):
                    st = stats_train_list[p] if p < len(stats_train_list) else {}
                    ts = stats_test_list[p] if p < len(stats_test_list) else {}
                    ok = (int(st.get("# Trades", 0)) >= is_min_trades) and (
                        int(ts.get("# Trades", 0)) >= oos_min_trades
                    )
                    if ok and idx < len(pairs):
                        is_v, oos_v = pairs[idx]
                        wfe_v = (
                            (oos_v / is_v)
                            if (is_v not in (0, None) and np.isfinite(is_v) and np.isfinite(oos_v))
                            else np.nan
                        )
                        if m == "Monthly Return":
                            row = f"{is_v:<7.3f} {oos_v:<7.3f} {wfe_v:<6.2f}"
                        elif m == "Win Rate [%]":
                            row = f"{is_v:<7.1f} {oos_v:<7.1f} {wfe_v:<6.2f}"
                        else:
                            row = f"{is_v:<7.2f} {oos_v:<7.2f} {wfe_v:<6.2f}"
                        idx += 1
                    else:
                        row = f"{'<20T':<7} {'N/A':<7} {'N/A':<6}"
                    print(f"{('P' + str(p)):<8} {row}")

        return scorecard, details


class CPCVAnalyzer:
    def __init__(
        self,
        splitter: CPCVSplitter,
        base_strategy: type[Strategy],
        maximize: str = "Rating",
        max_tries: int = 40,
        trade_on_close: bool = True,
        exclusive_orders: bool = False,
        initial_cash: float = CASH,
    ) -> None:
        self.splitter = splitter
        self.base_strategy = base_strategy
        self.maximize = maximize
        self.max_tries = max_tries
        self.trade_on_close = trade_on_close
        self.exclusive_orders = exclusive_orders
        self.initial_cash = float(initial_cash)

        self._path_results: dict[int, dict] | None = None
        self._aggregated_metrics: dict[str, float] | None = None
        self._best_path_info: dict[str, Any] | None = None

    def run_analysis(self, df: pd.DataFrame) -> CPCVAnalyzer:
        cpcv_paths = CPCVSplitter.split_paths(self.splitter, df)
        self._path_results, self._aggregated_metrics, self._best_path_info = self._run_wfo_on_paths(cpcv_paths)
        return self

    def get_path_results(self) -> dict[int, dict]:
        return self._path_results or {}

    def get_aggregated_metrics(self) -> dict[str, float]:
        return self._aggregated_metrics or {}

    def get_best_path(self) -> dict[str, Any]:
        return self._best_path_info or {}

    @staticmethod
    def _months_between(ts_start: pd.Timestamp, ts_end: pd.Timestamp) -> float:
        if pd.isna(ts_start) or pd.isna(ts_end) or ts_end <= ts_start:
            return 0.0
        delta_days = (ts_end - ts_start).total_seconds() / 86400.0
        return delta_days / 30.4375

    def _total_test_months(self, test_dfs: list[pd.DataFrame]) -> float:
        months = 0.0
        for df_ in test_dfs:
            if len(df_) > 0:
                months += self._months_between(df_.index.min(), df_.index.max())
        return months

    @staticmethod
    def _nanmean(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else np.nan

    @staticmethod
    def _nanmedian(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else np.nan

    @staticmethod
    def _nanstd(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.std()) if a.size else np.nan

    @staticmethod
    def _consistency(arr: np.ndarray, mean_val: float) -> float:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2 or not np.isfinite(mean_val):
            return 0.0
        std_val = np.nanstd(arr, ddof=1)
        if std_val == 0:
            return 1.0
        cv = abs(std_val / mean_val) if mean_val != 0 else np.inf
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    def _calculate_path_metrics(
        self, stats_test_list: list[dict], final_equity: float, total_test_months: float
    ) -> dict[str, float]:
        if not stats_test_list:
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
        total_return_pct = ((final_equity / self.initial_cash) - 1.0) * 100.0

        return {
            "total_return_pct": total_return_pct,
            "avg_annual_return": self._nanmean(ret_arr),
            "avg_sharpe_ratio": self._nanmean(sharpe_arr),
            "median_sharpe_ratio": self._nanmedian(sharpe_arr),
            "avg_calmar_ratio": self._nanmean(calmar_arr),
            "median_calmar_ratio": self._nanmedian(calmar_arr),
            "max_drawdown": float(np.nanmin(dd_arr)) if dd_arr.size else np.nan,
            "max_drawdown_abs": abs(float(np.nanmin(dd_arr))) if dd_arr.size else np.nan,
            "avg_win_rate": self._nanmean(winr_arr),
            "avg_profit_factor": self._nanmean(pf_arr),
            "median_profit_factor": self._nanmedian(pf_arr),
            "total_trades": total_trades,
            "trades_per_fold": total_trades / len(stats_test_list) if stats_test_list else np.nan,
            "trades_per_month": total_trades / total_test_months if total_test_months > 0 else np.nan,
            "returns_consistency": self._consistency(ret_arr, self._nanmean(ret_arr)),
            "sharpe_consistency": self._consistency(sharpe_arr, self._nanmean(sharpe_arr)),
            "calmar_consistency": self._consistency(calmar_arr, self._nanmean(calmar_arr)),
            "n_folds": len(stats_test_list),
            "processing_time": np.nan,
        }

    def _aggregate_across_paths(self, aggregated_data: dict[str, list[Any]]) -> dict[str, float]:
        if not aggregated_data.get("final_equities"):
            return {}

        n_paths = len(aggregated_data["final_equities"])
        agg = {"n_paths": float(n_paths)}

        # Total return aggregation
        final_equities = np.array([float(x) for x in aggregated_data["final_equities"] if np.isfinite(x)], dtype=float)
        if final_equities.size:
            returns_pct = (final_equities / self.initial_cash - 1) * 100
            agg["mean_total_return_pct"] = float(np.mean(returns_pct))
            agg["median_total_return_pct"] = float(np.median(returns_pct))
            agg["std_total_return_pct"] = float(np.std(returns_pct))

        # Generic numeric metrics
        skip_keys = {"final_equities", "processing_times", "path_ids"}
        for key, values in aggregated_data.items():
            if key in skip_keys:
                continue
            arr = np.array([float(x) for x in values if np.isfinite(float(x or np.nan))], dtype=float)
            if arr.size == 0:
                continue
            agg[f"mean_{key}"] = float(np.mean(arr))
            agg[f"median_{key}"] = float(np.median(arr))
            agg[f"std_{key}"] = float(np.std(arr))

        return agg

    def _score_paths(self, path_results: dict[int, dict]) -> dict[str, Any]:
        if not path_results:
            return {}
        # Select path with highest median Sharpe as a reasonable default robustness criterion
        best_pid = max(
            path_results, key=lambda pid: path_results[pid]["path_metrics"].get("median_sharpe_ratio", -np.inf)
        )
        return {
            "best_path_id": best_pid,
            "best_metrics": path_results[best_pid]["path_metrics"],
            "best_final_equity": path_results[best_pid]["final_equity"],
        }

    def _run_wfo_on_paths(self, cpcv_paths: CPCVPaths) -> tuple[dict[int, dict], dict[str, float], dict[str, Any]]:
        path_results: dict[int, dict] = {}
        aggregated_data: dict[str, list[Any]] = {"final_equities": [], "processing_times": [], "path_ids": []}

        for pid, folds in cpcv_paths.items():
            if not folds:
                continue
            train_dfs = [train for train, _ in folds]
            test_dfs = [test for _, test in folds]
            total_test_months = self._total_test_months(test_dfs)

            wfo = WalkForwardOptimizer(
                train_dfs=train_dfs,
                test_dfs=test_dfs,
                trade_on_close=self.trade_on_close,
                exclusive_orders=self.exclusive_orders,
                cash=self.initial_cash,
            )

            (
                trades_combined,
                stats_train_list,
                stats_test_list,
                parameters,
                bt_test_list,
                _,  # param_names
                _,  # optimize_results
            ) = wfo.run_wfo(
                base_strategy=self.base_strategy,
                params=None,
                max_tries=self.max_tries,
                maximize=self.maximize,
                verbose=False,
            )

            final_equity = self.initial_cash
            if stats_test_list:
                final_equity = float(stats_test_list[-1].get("Equity Final [$]", self.initial_cash))

            path_metrics = self._calculate_path_metrics(
                stats_test_list=stats_test_list, final_equity=final_equity, total_test_months=total_test_months
            )

            path_results[pid] = {
                "path_id": pid,
                "n_folds": len(folds),
                "final_equity": final_equity,
                "path_metrics": path_metrics,
                "stats_train_list": stats_train_list,
                "stats_test_list": stats_test_list,
                "parameters": parameters,
                "trades_combined": trades_combined,
                "bt_test_list": bt_test_list,
            }

            aggregated_data["final_equities"].append(final_equity)
            aggregated_data["processing_times"].append(path_metrics.get("processing_time", np.nan))
            aggregated_data["path_ids"].append(pid)
            for k, v in path_metrics.items():
                aggregated_data.setdefault(k, []).append(v)

        aggregated_metrics = self._aggregate_across_paths(aggregated_data)
        best_path_info = self._score_paths(path_results)

        return path_results, aggregated_metrics, best_path_info
