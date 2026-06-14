from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from src.config import ANNUAL_TRADING_DAYS, CASH
from src.metrics.core import compute_all_metrics
from src.visualization.style import DEFAULT_STYLE


class PortfolioAnalyzer:
    """
    Unified interface for preparing returns, calculating metrics,
    and dynamic volatility-targeted portfolio analysis.

    Refactored for performance (vectorized scaling), readability,
    maintainability, and efficiency.
    """

    def __init__(self, equity_dict: dict[str, pd.Series]):
        """
        Args:
            equity_dict: {'strategy_name': equity_series, ...}

        """
        # Pre-sort once for consistency
        self.equity_dict = {name: series.sort_index() for name, series in equity_dict.items()}
        self.returns: pd.DataFrame | None = None
        self.all_results = None  # TODO: clarify usage if needed

    @staticmethod
    def infer_annualization_factor(index: pd.DatetimeIndex) -> float:
        """Infer annualization factor from median bars-per-day (robust to tz)."""
        if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
            return 252.0

        idx_utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
        day_counts = pd.Series(1, index=idx_utc.normalize()).groupby(level=0).sum()

        if day_counts.empty:
            return 252.0

        bars_per_day = float(day_counts.median())
        if not np.isfinite(bars_per_day) or bars_per_day <= 0:
            return 252.0

        return bars_per_day * 252.0

    @staticmethod
    def _infer_annualization_factor(index: pd.DatetimeIndex) -> float:
        """Backward-compatible alias; prefer `infer_annualization_factor`."""
        return PortfolioAnalyzer.infer_annualization_factor(index)

    @staticmethod
    def _calculate_annual_vol(returns: pd.Series, ann_factor: float, vol_method: str = "arithmetic") -> float:
        """Shared helper – removes duplication between normalization & metrics."""
        if len(returns) < 2:
            return 0.0

        if vol_method == "arithmetic":
            return returns.std(ddof=1) * np.sqrt(ann_factor)

        if vol_method == "geometric":
            # Original geometric formula preserved exactly
            gmean = (1 + returns).prod() ** (1 / len(returns)) - 1
            var_daily = returns.var(ddof=1)
            return np.sqrt((var_daily + (1 + gmean) ** 2) ** ann_factor - (1 + gmean) ** (2 * ann_factor))

        raise ValueError(f"vol_method must be 'arithmetic' or 'geometric', got {vol_method}")

    def prepare_returns(self, resample_rule: str | None = "D") -> pd.DataFrame:
        """
        Convert equity curves to returns.

        Args:
            resample_rule: "D" (default) for daily, None for raw (intraday) bars.

        Returns:
            DataFrame with strategies as columns.

        """
        returns_dict: dict[str, pd.Series] = {}
        for name, equity in self.equity_dict.items():
            if resample_rule:
                equity_series = equity.resample(resample_rule).last().ffill()
            else:
                equity_series = equity

            returns_dict[name] = equity_series.pct_change().dropna()

        self.returns = pd.DataFrame(returns_dict).dropna(how="all")
        return self.returns

    def _get_rebalance_dates(self, index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
        """Map requested frequency anchors to actual dates present in the series."""
        if len(index) == 0:
            return pd.DatetimeIndex([])

        anchors = pd.date_range(index.min(), index.max(), freq=freq)
        mapped = index.get_indexer(anchors, method="ffill")
        valid = mapped[mapped >= 0]
        return pd.DatetimeIndex(sorted(set(index[valid])))

    def create_normalized_strategies_dynamic(
        self,
        rebalance_freq: str | dict[str, str] = "QE",
        lookback_periods: int | dict[str, int] = 60,
        target_vol: float = 0.10,
        initial_capital: float = CASH,
        vol_method: str = "arithmetic",
    ) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
        """
        Create volatility-targeted normalized equity curves with dynamic rebalancing.

        Performance note: scale factors are computed *only* at rebalance dates
        and forward-filled (vectorized). Original O(N) Python loop eliminated.
        """
        if vol_method not in ("arithmetic", "geometric"):
            raise ValueError("vol_method must be 'arithmetic' or 'geometric'")

        if self.returns is None:
            self.prepare_returns()

        normalized_equity_dict: dict[str, pd.Series] = {}
        scale_factors_timeseries_dict: dict[str, pd.DataFrame] = {}

        for strategy_name in self.returns.columns:
            strategy_returns = self.returns[strategy_name]
            clean_returns = strategy_returns.dropna()

            if clean_returns.empty:
                continue

            ann_factor = self.infer_annualization_factor(clean_returns.index)

            # Per-strategy parameters
            freq = rebalance_freq.get(strategy_name, "QE") if isinstance(rebalance_freq, dict) else rebalance_freq
            lookback = (
                lookback_periods.get(strategy_name, 60) if isinstance(lookback_periods, dict) else lookback_periods
            )

            rebalance_dates = self._get_rebalance_dates(clean_returns.index, freq)

            # Compute scale factors only at rebalance points
            scales: dict[Any, float] = {}
            scale_records: list[dict[str, Any]] = []
            for dt in rebalance_dates:
                loc = clean_returns.index.get_loc(dt)
                if loc >= lookback:
                    lookback_ret = clean_returns.iloc[loc - lookback : loc]
                    annual_vol = self._calculate_annual_vol(lookback_ret, ann_factor, vol_method)
                    scale_factor = target_vol / annual_vol if annual_vol > 1e-8 else 1.0

                    scales[dt] = scale_factor
                    scale_records.append(
                        {
                            "date": dt,
                            "annual_vol": annual_vol,
                            "scale_factor": scale_factor,
                            "rebalance_freq": freq,
                            "lookback_periods": lookback,
                            "annualization_factor": ann_factor,
                        }
                    )

            # Vectorized scaling: only valid rebalances → ffill → leading 1.0
            scale_ser = pd.Series(scales).reindex(clean_returns.index).ffill().fillna(1.0)

            normalized_returns = clean_returns * scale_ser
            # Reindex to original (may contain extra NaNs from alignment)
            normalized_returns = normalized_returns.reindex(strategy_returns.index, fill_value=0)

            normalized_equity = (1 + normalized_returns).cumprod() * initial_capital
            normalized_name = f"{strategy_name}_Normalized"
            normalized_equity_dict[normalized_name] = normalized_equity

            scale_factors_timeseries_dict[strategy_name] = (
                pd.DataFrame(scale_records) if scale_records else pd.DataFrame()
            )

        return normalized_equity_dict, scale_factors_timeseries_dict

    @staticmethod
    def calculate_metrics_from_equity(
        equity: pd.Series,
        daily_returns: pd.Series | None = None,
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.0,
    ) -> dict[str, float]:
        """Comprehensive metrics (arithmetic + geometric). Reuses vol helper."""
        if daily_returns is None:
            daily_returns = equity.pct_change().dropna()

        if len(equity) < 2:
            return {"error": "Insufficient data"}

        total_return = equity.iloc[-1] / equity.iloc[0] - 1

        # Geometric path
        if len(daily_returns) >= 2:
            gmean_day = (1 + daily_returns).prod() ** (1 / len(daily_returns)) - 1
            ann_ret_geom = (1 + gmean_day) ** annual_trading_days - 1
            ann_vol_geom = PortfolioAnalyzer._calculate_annual_vol(daily_returns, annual_trading_days, "geometric")
            sharpe_geom = (ann_ret_geom - risk_free_rate) / ann_vol_geom if ann_vol_geom > 1e-8 else 0.0
        else:
            ann_ret_geom = ann_vol_geom = sharpe_geom = 0.0

        # Arithmetic path
        ann_ret_arith = daily_returns.mean() * annual_trading_days
        ann_vol_arith = daily_returns.std(ddof=1) * np.sqrt(annual_trading_days)
        sharpe_arith = (ann_ret_arith - risk_free_rate) / ann_vol_arith if ann_vol_arith > 1e-8 else 0.0

        # Drawdown & Calmar
        cum_ret = (1 + daily_returns).cumprod()
        max_dd = (cum_ret / cum_ret.cummax() - 1).min() if not cum_ret.empty else 0.0
        calmar = ann_ret_geom / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return_geometric": ann_ret_geom,
            "annualized_return_arithmetic": ann_ret_arith,
            "annualized_volatility_geometric": ann_vol_geom,
            "annualized_volatility_arithmetic": ann_vol_arith,
            "sharpe_ratio_geometric": sharpe_geom,
            "sharpe_ratio_arithmetic": sharpe_arith,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
        }

    def calculate_individual_strategy_metrics(self) -> dict[str, dict]:
        """Metrics for each original strategy (daily resampling for consistency)."""
        individual_results = {}
        for name, equity_series in self.equity_dict.items():
            equity_daily = equity_series.resample("D").last().dropna()
            daily_returns = equity_daily.pct_change().fillna(0)

            metrics = self.calculate_metrics_from_equity(equity_daily, daily_returns)

            individual_results[name] = {
                "results": {
                    "equity": equity_daily,
                    "daily_returns": daily_returns,
                    "weights_history": pd.DataFrame(),
                },
                "metrics": metrics,
            }
        return individual_results


class PortfolioReporter:
    """
    Comprehensive reporting utilities for portfolio analysis.
    All parameters passed explicitly - no global dependencies.
    """

    def __init__(self, all_results: dict, returns: pd.DataFrame):
        """
        Initialize reporter.

        Args:
            all_results: Results dictionary from portfolio analysis
            returns: Daily returns DataFrame

        """
        self.all_results = all_results
        self.returns = returns

    def print_comprehensive_metrics(
        self, risk_free_rate: float = 0.0, annual_trading_days: int = 252, precision: int = 2
    ) -> pd.DataFrame:

        def calculate_pnl_and_returns_from_equity(
            equity: pd.Series, initial_capital: float | None = None
        ) -> tuple[np.ndarray, np.ndarray]:

            equity = equity.sort_index()

            if initial_capital is None:
                initial_capital = equity.iloc[0]

            equity_values = equity.values.astype(np.float64)

            pnl = np.diff(equity_values, prepend=equity_values[0])
            pnl[0] = 0.0

            equity_shifted = np.roll(equity_values, 1)
            equity_shifted[0] = initial_capital

            returns_pct = np.where(equity_shifted > 0, ((equity_values - equity_shifted) / equity_shifted) * 100, 0.0)
            returns_pct[0] = 0.0

            return pnl, returns_pct

        metrics_list = []

        for name, result in self.all_results.items():
            equity = result["results"]["equity"]
            pnl, returns_pct = calculate_pnl_and_returns_from_equity(equity)

            # Infer annualization factor from equity's actual index frequency so that
            # daily-resampled equity (after resample_all_results) uses 252, not the
            # original intraday factor that may have been passed via annual_trading_days.
            effective_days = max(1, int(round(PortfolioAnalyzer.infer_annualization_factor(equity.index))))

            metrics_series = compute_all_metrics(
                equity=equity,
                risk_free_rate=risk_free_rate,
                annual_trading_days=effective_days,
                pnl=pnl,
                returns_pct=returns_pct,
            )

            row = {
                "Strategy": name,
                "Start": metrics_series["Start"],
                "End": metrics_series["End"],
                "Duration": metrics_series["Duration"],
                "Return [%]": metrics_series["Return [%]"],
                "Return (Ann.) [%]": metrics_series["Return (Ann.) [%]"],
                "CAGR [%]": metrics_series["CAGR [%]"],
                "Volatility (Ann.) [%]": metrics_series["Volatility (Ann.) [%]"],
                "Sharpe Ratio": metrics_series["Sharpe Ratio"],
                "Sortino Ratio": metrics_series["Sortino Ratio"],
                "Calmar Ratio": metrics_series["Calmar Ratio"],
                "Max. Drawdown [%]": metrics_series["Max. Drawdown [%]"],
                "Ulcer Index [%]": metrics_series["Ulcer Index [%]"],
                "Best Day [%]": metrics_series["Best Day [%]"],
                "Worst Day [%]": metrics_series["Worst Day [%]"],
                "CVaR 95% [%]": metrics_series["CVaR 95% [%]"],
            }

            metrics_list.append(row)

        metrics_df = pd.DataFrame(metrics_list)

        print("\n" + "=" * 150)
        print("COMPREHENSIVE METRICS")
        print("=" * 150)

        with pd.option_context("display.precision", precision, "display.max_columns", None, "display.width", 200):
            print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.{precision}f}"))

        print("=" * 150 + "\n")

        return metrics_df

    def print_dynamic_scaling_factors(
        self,
        scale_factors_timeseries_dict: dict[str, pd.DataFrame],
        rebalance_freq: str | dict[str, str] = "QE",
        lookback_periods: int | dict[str, int] = 60,
        max_periods_to_show: int = 2,
    ):
        """
        Print scaling factors with ALL portfolio optimizer allocations side-by-side,
        honoring per-strategy custom rebalance frequencies and lookbacks.

        Reads per-portfolio actual rebalance schedules (if available) from each portfolio's results:
        - results['rebalance_freq_dict']
        - results['lookback_periods_dict']
        Falls back to provided parameters if not present.
        """
        # 1) Collect all portfolio rebalance dates from portfolio weights (most complete source)
        all_portfolio_dates = set()
        portfolio_meta = {}  # name -> {'freq_dict', 'lookback_dict'}
        for name, result in self.all_results.items():
            if "Portfolio" not in name:
                continue
            res = result.get("results", {})
            weights_df = res.get("weights_history", pd.DataFrame())

            # Capture portfolio-specific freq/lookback dicts if present
            freq_dict = res.get("rebalance_freq_dict", None)
            lb_dict = res.get("lookback_periods_dict", None)
            portfolio_meta[name] = {
                "freq_dict": freq_dict if freq_dict is not None else rebalance_freq,
                "lookback_dict": lb_dict if lb_dict is not None else lookback_periods,
            }

            if len(weights_df) > 0:
                all_portfolio_dates.update(weights_df.index)

        if not all_portfolio_dates:
            print("❌ No portfolio weights found. Ensure backtests ran successfully.\n")
            return

        # 2) Normalize and sort the rebalance dates (naive for comparison)
        all_dates = sorted(pd.Timestamp(d).tz_localize(None) for d in all_portfolio_dates)
        if max_periods_to_show is not None and max_periods_to_show > 0:
            all_dates = all_dates[-max_periods_to_show:]

        # 3) Build a mapping of portfolio allocations by date with normalized timestamps
        portfolio_data = {}
        for name, result in self.all_results.items():
            if "Portfolio" not in name:
                continue
            res = result.get("results", {})
            weights_df = res.get("weights_history", pd.DataFrame())

            if len(weights_df) > 0:
                clean_name = name.replace("Portfolio_", "").replace("_Leveraged", "")
                portfolio_data[clean_name] = {
                    pd.Timestamp(idx).tz_localize(None): row.to_dict() for idx, row in weights_df.iterrows()
                }

        # 4) Pre-normalize scale_factors_timeseries dates and attach per-strategy freq/lookback metadata
        #    Strategy names in scale_factors_timeseries_dict keys should match raw base names (without _Normalized).
        #    If they are normalized names, keep them; allocations lookup below tries both variants.
        normalized_scale_data = {}  # strategy -> DataFrame with 'date_norm', 'annual_vol', 'scale_factor', 'rebalance_freq', 'lookback'
        for strategy_name, scale_df in scale_factors_timeseries_dict.items():
            if scale_df is None or len(scale_df) == 0:
                continue
            sdf = scale_df.copy()
            # Normalize date for matching
            if "date" in sdf.columns:
                sdf["date_norm"] = pd.to_datetime(sdf["date"]).dt.tz_localize(None)
            elif "date_norm" in sdf.columns:
                sdf["date_norm"] = pd.to_datetime(sdf["date_norm"]).dt.tz_localize(None)
            else:
                # If no explicit date column, skip
                continue

            # Attach per-strategy freq/lookback from any portfolio meta that has it, fallback to function args
            # Priority: if any portfolio has custom entries for this strategy, use the first found
            # Otherwise fallback to provided function arguments (scalar or dict)
            eff_freq = None
            eff_lb = None

            # Try to extract from any portfolio that carries metadata for this strategy
            for p_name, meta in portfolio_meta.items():
                fdict = meta["freq_dict"]
                ldict = meta["lookback_dict"]
                if isinstance(fdict, dict) and strategy_name in fdict and eff_freq is None:
                    eff_freq = fdict[strategy_name]
                if isinstance(ldict, dict) and strategy_name in ldict and eff_lb is None:
                    eff_lb = ldict[strategy_name]

            # Fallback to function parameters
            if eff_freq is None:
                eff_freq = rebalance_freq.get(strategy_name) if isinstance(rebalance_freq, dict) else rebalance_freq
            if eff_lb is None:
                eff_lb = lookback_periods.get(strategy_name) if isinstance(lookback_periods, dict) else lookback_periods

            # Attach for each row (constant columns)
            sdf["rebalance_freq"] = eff_freq
            sdf["lookback_periods"] = eff_lb

            normalized_scale_data[strategy_name] = sdf

        # # 5) Print header
        # print(f"Dynamic Scaling Factors + Portfolio Allocations (All Optimizers)")
        # print("="*200)

        # # Compute the union list of portfolio display names
        # portfolio_names = sorted(portfolio_data.keys())
        # col_width = 14
        # strategy_col = 22
        # vol_col = 12
        # scale_col = 17
        # meta_col = 18
        # separator = " | "

        # for rebal_date in all_dates:
        #     print(f"\nRebalance Date: {rebal_date.strftime('%Y-%m-%d')}")
        #     print("-"*200)

        #     # Build matching scale_data for this date across strategies
        #     # We accept exact match on date_norm; if you want nearest match within T days, add tolerance logic here.
        #     scale_snapshot = {}
        #     for strategy_name, sdf in normalized_scale_data.items():
        #         row = sdf[sdf['date_norm'] == rebal_date]
        #         if not row.empty:
        #             r0 = row.iloc[0]
        #             scale_snapshot[strategy_name] = {
        #                 'vol': r0.get('annual_vol', np.nan),
        #                 'scale': r0.get('scale_factor', np.nan),
        #                 'freq': r0.get('rebalance_freq', None),
        #                 'lookback': r0.get('lookback_periods', None)
        #             }

        #     if not scale_snapshot:
        #         print(f"  ⚠️  No scale data for {rebal_date.strftime('%Y-%m-%d')} (may not be a rebalance date for strategies)")
        #         continue

        #     # Header
        #     header = (
        #         f"{'Strategy':<{strategy_col}}{separator}"
        #         f"{'Annual Vol':>{vol_col}}{separator}"
        #         f"{'Scale Factor':>{scale_col}}{separator}"
        #         f"{'Freq/Lookback':^{meta_col}}{separator}"
        #     )
        #     header += separator.join(f"{name:^{col_width}}" for name in portfolio_names)
        #     print(f"  {header}")

        #     sep_line = (
        #         f"  {'-' * strategy_col}{separator}"
        #         f"{'-' * vol_col}{separator}"
        #         f"{'-' * scale_col}{separator}"
        #         f"{'-' * meta_col}{separator}"
        #     )
        #     sep_line += separator.join('-' * col_width for _ in portfolio_names)
        #     print(sep_line)

        #     # Rows: each strategy with scale info + ALL portfolio allocations
        #     for strategy_name in sorted(scale_snapshot.keys()):
        #         vol = scale_snapshot[strategy_name]['vol']
        #         scale = scale_snapshot[strategy_name]['scale']
        #         freq = scale_snapshot[strategy_name]['freq']
        #         lb = scale_snapshot[strategy_name]['lookback']

        #         # Allow both raw and normalized strategy key when pulling allocations
        #         normalized_key = strategy_name if strategy_name.endswith('_Normalized') else f"{strategy_name}_Normalized"
        #         raw_key = strategy_name.replace('_Normalized', '')

        #         row_str = (
        #             f"  {strategy_name:<{strategy_col}}{separator}"
        #             f"{vol:>{vol_col}.2%}{separator}"
        #             f"{scale:>{scale_col-1}.3f}x{separator}"
        #             f"{str(freq)}/{str(lb):<{meta_col-2}}{separator}"
        #         )

        #         for p_name in portfolio_names:
        #             alloc_dict = portfolio_data.get(p_name, {}).get(rebal_date, {})
        #             # Try normalized key first, then raw key
        #             alloc = alloc_dict.get(normalized_key, None)
        #             if alloc is None:
        #                 alloc = alloc_dict.get(raw_key, 0.0)
        #             row_str += f"{alloc*100:>{col_width-1}.2f}%{separator}"

        #         print(row_str.rstrip(separator))

        # print("\n" + "="*200 + "\n")

    def print_scaling_summary(self):
        """
        Print summary statistics of scaling factors across all rebalances (for portfolios).
        """
        print("\n" + "=" * 120)
        print("PORTFOLIO SCALING FACTORS SUMMARY (Average across all rebalances)")
        print("=" * 120)

        for portfolio_name, data in self.all_results.items():
            if "Portfolio" not in portfolio_name:
                continue

            scale_factors_by_date = data["results"].get("scale_factors_by_date", [])

            if not scale_factors_by_date:
                continue

            print(f"\n{portfolio_name}:")
            print("-" * 120)

            strategy_scales = {}

            for entry in scale_factors_by_date:
                annual_vols = entry["annual_vols"]
                scale_factors = entry["scale_factors"]

                for strategy_name in annual_vols.keys():
                    if strategy_name not in strategy_scales:
                        strategy_scales[strategy_name] = {"vols": [], "scales": []}

                    strategy_scales[strategy_name]["vols"].append(annual_vols[strategy_name])
                    strategy_scales[strategy_name]["scales"].append(scale_factors[strategy_name])

            print(
                f"  {'Strategy':<15} | {'Avg Annual Vol':<18} | {'Avg Scale Factor':<18} | {'Min Scale':<12} | {'Max Scale':<12}"
            )
            print(f"  {'-' * 15}|{'-' * 20}|{'-' * 20}|{'-' * 14}|{'-' * 14}")

            for strategy_name in sorted(strategy_scales.keys()):
                vols = strategy_scales[strategy_name]["vols"]
                scales = strategy_scales[strategy_name]["scales"]

                avg_vol = np.mean(vols)
                avg_scale = np.mean(scales)
                min_scale = np.min(scales)
                max_scale = np.max(scales)

                print(
                    f"  {strategy_name:<15} | {avg_vol:>18.2%} | {avg_scale:>18.3f}x | {min_scale:>12.3f}x | {max_scale:>12.3f}x"
                )

            print()

    def print_allocator_diagnostics(
        self,
        portfolio_name: str,
        baseline_portfolio_name: str | None = None,
        test_start: str = "2024-01-01",
        test_end: str = "2025-11-30",
        live_start: str = "2025-12-01",
    ) -> pd.DataFrame:
        """
        Print central allocator diagnostics:
        - average multiplier by strategy
        - % time below/above 1.0
        - turnover contribution
        - performance vs baseline portfolio on test/live windows
        """
        if portfolio_name not in self.all_results:
            raise KeyError(f"Unknown portfolio: {portfolio_name}")

        res = self.all_results[portfolio_name].get("results", {})
        allocator_history = res.get("allocator_history", [])
        if not allocator_history:
            print(f"No allocator history found for {portfolio_name}.")
            return pd.DataFrame()

        mult_rows = []
        turnover_rows = []
        for entry in allocator_history:
            date = pd.Timestamp(entry.get("date"))
            mult_dict = entry.get("multipliers", {})
            for strategy, mult in mult_dict.items():
                mult_rows.append({"date": date, "strategy": strategy, "multiplier": float(mult)})

            overlay_stats = entry.get("overlay_stats", {})
            turnover_rows.append(
                {
                    "date": date,
                    "turnover": float(overlay_stats.get("turnover", np.nan)),
                    "turnover_cap": float(overlay_stats.get("turnover_cap", np.nan)),
                    "turnover_limit_applied": bool(overlay_stats.get("turnover_limit_applied", False)),
                    "bounds_clipped": bool(overlay_stats.get("bounds_clipped", False)),
                }
            )

        multipliers_df = pd.DataFrame(mult_rows)
        turnover_df = pd.DataFrame(turnover_rows)

        summary = (
            multipliers_df.groupby("strategy")["multiplier"]
            .agg(
                avg_multiplier="mean",
                pct_below_1=lambda s: (s < 1.0).mean() * 100.0,
                pct_above_1=lambda s: (s > 1.0).mean() * 100.0,
                min_multiplier="min",
                max_multiplier="max",
            )
            .sort_values("avg_multiplier", ascending=False)
        )

        print("\n" + "=" * 120)
        print(f"CENTRAL ALLOCATOR DIAGNOSTICS: {portfolio_name}")
        print("=" * 120)
        print(summary.round(4).to_string())

        if not turnover_df.empty:
            avg_turnover = turnover_df["turnover"].mean()
            total_turnover = turnover_df["turnover"].sum()
            cap_hit_rate = turnover_df["turnover_limit_applied"].mean() * 100.0
            bounds_clip_rate = turnover_df["bounds_clipped"].mean() * 100.0

            print("\nTurnover Diagnostics")
            print("-" * 120)
            print(f"Allocator rebalances   : {len(turnover_df)}")
            print(f"Average turnover       : {avg_turnover:.4f}")
            print(f"Total turnover         : {total_turnover:.4f}")
            print(f"Turnover cap hit rate  : {cap_hit_rate:.2f}%")
            print(f"Bounds clipped rate    : {bounds_clip_rate:.2f}%")

        def _align_ts(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
            if idx.tz is None:
                return ts.tz_localize(None) if ts.tz is not None else ts
            if ts.tz is None:
                return ts.tz_localize(idx.tz)
            return ts.tz_convert(idx.tz)

        def _period_stats(equity: pd.Series, start: str, end: str | None = None) -> dict[str, float]:
            eq = equity.dropna().sort_index()
            if eq.empty:
                return {"return": np.nan, "sharpe": np.nan}

            start_ts = _align_ts(pd.Timestamp(start), pd.DatetimeIndex(eq.index))
            if end is not None:
                end_ts = _align_ts(pd.Timestamp(end), pd.DatetimeIndex(eq.index))
                period_eq = eq.loc[(eq.index >= start_ts) & (eq.index <= end_ts)]
            else:
                period_eq = eq.loc[eq.index >= start_ts]

            if len(period_eq) < 2:
                return {"return": np.nan, "sharpe": np.nan}

            period_returns = period_eq.pct_change().dropna()
            total_ret = period_eq.iloc[-1] / period_eq.iloc[0] - 1.0
            ann_vol = period_returns.std(ddof=1) * np.sqrt(252)
            ann_ret = period_returns.mean() * 252
            sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
            return {"return": float(total_ret), "sharpe": float(sharpe)}

        if baseline_portfolio_name is None:
            candidate = f"{portfolio_name}_NoAllocator"
            if candidate in self.all_results:
                baseline_portfolio_name = candidate

        if baseline_portfolio_name in self.all_results:
            target_eq = self.all_results[portfolio_name]["results"]["equity"]
            base_eq = self.all_results[baseline_portfolio_name]["results"]["equity"]

            target_test = _period_stats(target_eq, test_start, test_end)
            target_live = _period_stats(target_eq, live_start, None)
            base_test = _period_stats(base_eq, test_start, test_end)
            base_live = _period_stats(base_eq, live_start, None)

            compare_df = pd.DataFrame(
                [
                    {
                        "Portfolio": portfolio_name,
                        "Test Return [%]": target_test["return"] * 100.0,
                        "Test Sharpe": target_test["sharpe"],
                        "Live Return [%]": target_live["return"] * 100.0,
                        "Live Sharpe": target_live["sharpe"],
                    },
                    {
                        "Portfolio": baseline_portfolio_name,
                        "Test Return [%]": base_test["return"] * 100.0,
                        "Test Sharpe": base_test["sharpe"],
                        "Live Return [%]": base_live["return"] * 100.0,
                        "Live Sharpe": base_live["sharpe"],
                    },
                    {
                        "Portfolio": "Delta (Allocator - Baseline)",
                        "Test Return [%]": (target_test["return"] - base_test["return"]) * 100.0,
                        "Test Sharpe": target_test["sharpe"] - base_test["sharpe"],
                        "Live Return [%]": (target_live["return"] - base_live["return"]) * 100.0,
                        "Live Sharpe": target_live["sharpe"] - base_live["sharpe"],
                    },
                ]
            )

            print("\nPerformance vs Baseline")
            print("-" * 120)
            print(compare_df.round(4).to_string(index=False))
        else:
            if baseline_portfolio_name is not None:
                print(f"\nBaseline portfolio not found: {baseline_portfolio_name}")
            else:
                print("\nNo baseline portfolio found. Pass baseline_portfolio_name to compare test/live performance.")

        print("=" * 120 + "\n")
        return summary

    def print_rebalances(self, portfolio_name, rebalance_freq):
        # Extract data
        res = self.all_results[portfolio_name]["results"]
        weights_df = res["weights_history"]
        leverage = res.get("leverage_ratio", 1.0)
        scale_info = res["scale_factors_by_date"]
        leverage_df = res.get("leverage_history_df", pd.DataFrame())

        # Build scale factor DataFrame
        scale_df = pd.DataFrame(
            [s["scale_factors"] for s in scale_info], index=pd.DatetimeIndex([s["date"] for s in scale_info])
        )
        # after building scale_df and before merge
        weights_df = weights_df.copy()
        scale_df = scale_df.copy()

        weights_df.index = pd.to_datetime(weights_df.index, utc=True).normalize()
        scale_df.index = pd.to_datetime(scale_df.index, utc=True).normalize()

        # If scale_df has multiple rows per day after normalize, collapse
        scale_df = scale_df.groupby(level=0).last()

        merged = weights_df.merge(
            scale_df, left_index=True, right_index=True, suffixes=("_weight", "_scale")
        ).sort_index()

        freq_offset = pd.tseries.frequencies.to_offset(rebalance_freq)

        last_dates = merged.index[-2:]

        for date in last_dates:
            row = merged.loc[date]

            lev = leverage
            if not leverage_df.empty and date in leverage_df.index:
                lev = float(leverage_df.loc[date, "leverage"])

            date = pd.Timestamp(date)  # your loop date (tz-aware already)

            current_month_end = date + pd.offsets.MonthEnd(0)  # rollforward to month-end
            end_date = current_month_end + pd.offsets.MonthEnd(1)  # next month-end

            print(f"\nRebalance Period: {date.date()} → {end_date.date()} | Portfolio: {portfolio_name}")
            print(f"Portfolio Leverage (dynamic): {lev:.3f}")
            print("-" * 55)
            print(f"{'Strategy':<20} {'Weight':>10} {'Scale':>10} {'Risk':>12}")
            print("-" * 55)

            for asset in [c.replace("_weight", "") for c in merged.columns if "_weight" in c]:
                weight = row[f"{asset}_weight"]
                scale = row[f"{asset}_scale"]
                risk = weight * scale * lev
                print(f"{asset:<20} {weight:>10.4f} {scale:>10.4f} {risk:>12.4f}")

            print("-" * 55)

    def print_evt_var(self, portfolio_name: str, confidence_level=0.99, threshold_quantile=0.95):
        returns_series = self.all_results[portfolio_name]["results"]["daily_returns"]
        # 1. Convert to Losses (EVT models the right tail of losses)
        # We assume 'returns_series' contains negative values for losses.
        losses = -returns_series.dropna()

        # 2. Determine Threshold (u)
        # The point where "normal" volatility ends and "extreme" volatility begins.
        u = losses.quantile(threshold_quantile)

        # 3. Extract Excesses (y = x - u)
        # We only care about the magnitude by which losses exceed the threshold.
        excesses = losses[losses > u] - u

        # 4. Fit Generalized Pareto Distribution (GPD)
        # We fix location (floc=0) because excesses are defined relative to u.
        # shape (xi) = Tail Index (Key metric for fat tails).
        # scale (sigma) = Scale parameter.
        xi, loc, sigma = genpareto.fit(excesses, floc=0)

        # 5. Calculate VaR Formula
        # VaR = u + (sigma / xi) * [ ((N / Nu) * (1 - p))^(-xi) - 1 ]
        N = len(losses)  # Total observations
        Nu = len(excesses)  # Number of extreme exceedances
        p = confidence_level  # Target probability (0.99)

        term = (N / Nu) * (1 - p)
        var_evt = u + (sigma / xi) * (term ** (-xi) - 1)
        print(
            f"{'EVT VaR (99%)':<16}: {-var_evt:.3%}\n"
            f"{'Tail Index (xi)':<16}: {xi:.4f}\n"
            f"{'Threshold (u)':<16}: {u:.4f}\n"
            f"{'Scale (sigma)':<16}: {sigma:.4f}"
        )

    def analyze_daily_returns_detailed(
        self,
        portfolio_name: str = "Portfolio_E_L",
        risk_free_rate: float = 0.0,
        annual_trading_days: int = ANNUAL_TRADING_DAYS,
    ) -> dict[str, pd.Series]:
        equity = self.all_results[portfolio_name]["results"]["equity"]
        daily_returns = equity.pct_change().fillna(0)
        daily_returns = daily_returns.fillna(0)

        # ===== ANALYSIS 1: ALL DAYS (INCLUDING ZEROS) =====
        # Convert daily returns to equity curve
        equity_all = (1 + daily_returns).cumprod() * CASH  # Start with $10M
        equity_all.index = pd.to_datetime(equity_all.index)

        from src.metrics.core import compute_all_metrics

        metrics_all = compute_all_metrics(
            equity=equity_all, risk_free_rate=risk_free_rate, annual_trading_days=annual_trading_days
        )

        # Add daily-specific metrics
        winning_days = daily_returns[daily_returns > 0]
        losing_days = daily_returns[daily_returns < 0]
        neutral_days = daily_returns[daily_returns == 0]

        metrics_all["Winning Days"] = len(winning_days)
        metrics_all["Losing Days"] = len(losing_days)
        metrics_all["Neutral Days"] = len(neutral_days)
        metrics_all["Win Rate [%]"] = (len(winning_days) / len(daily_returns)) * 100 if len(daily_returns) > 0 else 0.0
        metrics_all["Avg Winning Day [%]"] = (winning_days.mean() * 100) if len(winning_days) > 0 else np.nan
        metrics_all["Avg Losing Day [%]"] = (losing_days.mean() * 100) if len(losing_days) > 0 else np.nan
        metrics_all["Profit Factor"] = (
            (winning_days.sum() / abs(losing_days.sum())) if abs(losing_days.sum()) > 1e-10 else np.inf
        )

        return format_metrics(metrics_all)

    def analyze_weekday(self, portfolio_name: str = "Portfolio_E_L") -> pd.DataFrame:
        """
        Analyzes equity performance by Day of Week and appends a Weekly aggregation row.
        """
        equity = self.all_results[portfolio_name]["results"]["equity"]
        # --- 1. Daily Data Prep ---
        df_daily = pd.DataFrame({"equity": equity})
        df_daily["daily_return"] = df_daily["equity"].pct_change().dropna()
        df_daily["day_name"] = df_daily["index_day"] = df_daily.index.day_name()

        # Calculate Total Return for Contribution basis (Sum of daily arithmetic returns)
        total_arithmetic_return = df_daily["daily_return"].sum()

        # --- 2. Weekly Data Prep ---
        # Resample to Weekly (Friday)
        series_weekly = equity.resample("W").last().pct_change().dropna()

        # --- 3. Shared Metric Calculation Logic ---
        def calculate_row(returns_series, label, contribution_denominator=None):
            if len(returns_series) == 0:
                return {}

            # Core stats
            win_rate = (returns_series > 0).mean() * 100
            avg_ret = returns_series.mean() * 100
            med_ret = returns_series.median() * 100
            std_dev = returns_series.std() * 100
            skew = returns_series.skew()

            # Profit Factor
            gross_profit = returns_series[returns_series > 0].sum()
            gross_loss = abs(returns_series[returns_series < 0].sum())
            pf = (gross_profit / gross_loss) if gross_loss != 0 else np.inf

            # Consistency Inputs
            avg_win = returns_series[returns_series > 0].mean() * 100 if (returns_series > 0).any() else 0
            avg_loss = returns_series[returns_series < 0].mean() * 100 if (returns_series < 0).any() else 0

            # Consistency Score: (WinRate * AvgWin) / |AvgLoss|
            consistency = ((win_rate / 100) * avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf

            # Contribution
            # If denominator is None, we assume this row IS the total (100%)
            total_ret = returns_series.sum()
            if contribution_denominator:
                contrib = total_ret / contribution_denominator * 100
            else:
                contrib = 100.0

            return {
                "Day": label,
                "Trades": len(returns_series),
                "Win Rate (%)": win_rate,
                "Avg Return (%)": avg_ret,
                "Median Return (%)": med_ret,
                "Std Dev (%)": std_dev,
                "Skewness": skew,
                "Profit Factor": pf,
                "Contribution (%)": contrib,
                "Consistency": consistency,
            }

        # --- 4. Build Rows ---
        results = []

        # A. Daily Rows
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in day_order:
            subset = df_daily[df_daily["day_name"] == day]["daily_return"]
            if len(subset) > 0:
                results.append(calculate_row(subset, day, total_arithmetic_return))

        # B. Weekly Row (The "2" Option)
        # Note: Contribution is set to 100% as it represents the full strategy timeframe
        weekly_row = calculate_row(series_weekly, "Weekly (Agg)", contribution_denominator=None)
        results.append(weekly_row)

        return print_mixed_results(pd.DataFrame(results))

    def analyze_monthly_performance(
        self, portfolio_name: str = "Portfolio_E_L", rolling_window: int = 12, resample_rule: str = "ME"
    ) -> pd.DataFrame:
        """
        Calculates monthly returns, volatility, and rolling Sharpe ratio from an equity curve.
        """
        equity_series = self.all_results[portfolio_name]["results"]["equity"]

        # Resample to get month-end equity
        monthly_equity = equity_series.resample(resample_rule).last()

        # Calculate monthly returns
        monthly_returns = monthly_equity.pct_change().dropna()

        # Resample daily returns to calculate monthly volatility
        daily_returns = equity_series.pct_change().dropna()
        monthly_volatility = daily_returns.resample(resample_rule).std() * np.sqrt(252)

        # Combine into a single DataFrame, ensuring index alignment
        df = pd.DataFrame({"Monthly_Return": monthly_returns, "Annualized_Volatility": monthly_volatility}).dropna()

        # Calculate rolling annualized return and volatility for Sharpe
        # Use the raw monthly returns for the calculation
        rolling_annual_return = df["Monthly_Return"].rolling(window=rolling_window).mean() * 12
        # Use the monthly annualized volatility for the rolling average
        rolling_annual_vol = df["Annualized_Volatility"].rolling(window=rolling_window).mean()

        # Calculate rolling Sharpe ratio, handle division by zero
        df["Rolling_Sharpe_Ratio"] = np.where(rolling_annual_vol > 0, rolling_annual_return / rolling_annual_vol, 0.0)

        return df

    def plot_monthly_performance(
        self, portfolio_name: str = "Portfolio_E_L", rolling_window: int = 12, resample_rule: str = "ME"
    ):
        """
        Generate a multi-panel plot for monthly performance metrics using matplotlib,
        """
        style = DEFAULT_STYLE
        monthly_df = self.analyze_monthly_performance(portfolio_name, rolling_window, resample_rule)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"{resample_rule} Performance Analysis", fontsize=16, y=0.95, fontdict={"color": style.font_color})

        # Apply your Matplotlib style (figure/axes backgrounds, grids, spines, ticks, legend frame)

        # --- Plot 1: Monthly Returns ---
        ax1 = axes[0]
        colors = [style.accent3 if x >= 0 else style.accent4 for x in monthly_df["Monthly_Return"]]
        ax1.bar(monthly_df.index, monthly_df["Monthly_Return"], width=20, color=colors, alpha=0.9)
        ax1.set_title(f"{resample_rule} Returns", fontdict={"color": style.font_color})
        ax1.set_ylabel("Return")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # --- Plot 2: Annualized Volatility ---
        ax2 = axes[1]
        ax2.plot(
            monthly_df.index,
            monthly_df["Annualized_Volatility"],
            label="Monthly Ann. Volatility",
            color=style.accent1,
            linewidth=2,
        )
        ax2.fill_between(monthly_df.index, monthly_df["Annualized_Volatility"], color=style.accent1, alpha=0.15)
        ax2.set_title(f"{resample_rule} Annualized Volatility", fontdict={"color": style.font_color})
        ax2.set_ylabel("Volatility")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax2.legend(loc="upper left")

        # --- Plot 3: Rolling Sharpe Ratio ---
        ax3 = axes[2]
        ax3.plot(
            monthly_df.index,
            monthly_df["Rolling_Sharpe_Ratio"],
            label=f"{rolling_window}-Month Rolling Sharpe",
            color=style.accent6,
            linewidth=2,
        )
        ax3.axhline(0, color=style.muted, linestyle="--", linewidth=1)
        ax3.set_title(f"{rolling_window}-{resample_rule} Rolling Sharpe Ratio", fontdict={"color": style.font_color})
        ax3.set_ylabel("Sharpe Ratio")
        ax3.legend(loc="upper left")

        # --- Common Formatting ---
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)

        # Date formatting on the bottom axis (shared x propagates)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))  # Jan + Jul markers [web:11]

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        style.apply_mpl(fig=fig)
        plt.show()


def print_mixed_results(results_df):
    """
    Prints results with specific handling for the Weekly row separator.
    """
    if results_df.empty:
        return

    # Define headers and widths
    headers = {
        "Day": 14,
        "Trades": 7,
        "Win Rate (%)": 13,
        "Avg Return (%)": 16,
        "Median Return (%)": 18,
        "Std Dev (%)": 13,
        "Skewness": 10,
        "Profit Factor": 15,
        "Contribution (%)": 17,
        "Consistency": 12,
    }

    # Formatter strings
    header_str = " | ".join([f"{h:<{w}}" for h, w in headers.items()])
    sep_line = "-" * len(header_str)

    print(sep_line)
    print("PERFORMANCE: DAILY BREAKDOWN + WEEKLY AGGREGATION")
    print(sep_line)
    print(header_str)
    print(sep_line)

    for i, row in results_df.iterrows():
        # Add a separator line before the Weekly row for visual distinction
        if row["Day"] == "Weekly (Agg)":
            print(sep_line)

        print(
            f"{row['Day']:<{headers['Day']}} | "
            f"{row['Trades']:<{headers['Trades']}} | "
            f"{row['Win Rate (%)']:<{headers['Win Rate (%)']}.1f} | "
            f"{row['Avg Return (%)']:<{headers['Avg Return (%)']}.3f} | "
            f"{row['Median Return (%)']:<{headers['Median Return (%)']}.3f} | "
            f"{row['Std Dev (%)']:<{headers['Std Dev (%)']}.3f} | "
            f"{row['Skewness']:<{headers['Skewness']}.2f} | "
            f"{row['Profit Factor']:<{headers['Profit Factor']}.2f} | "
            f"{row['Contribution (%)']:<{headers['Contribution (%)']}.1f} | "
            f"{row['Consistency']:<{headers['Consistency']}.2f}"
        )
    print(sep_line)


def format_metrics(metrics: pd.Series) -> None:
    """Print performance metrics with readable formatting."""
    print("{:<25} {}".format("Metric", "Value"))
    print("-" * 40)

    for metric, value in metrics.items():
        # Numeric format
        if isinstance(value, (int, np.integer)):
            print(f"{metric:<25} {value}")
        elif isinstance(value, (float, np.floating)):
            # Format large/small floats gracefully
            if abs(value) > 1e4 or (abs(value) < 1e-2 and value != 0):
                print(f"{metric:<25} {value:.2e}")
            else:
                print(f"{metric:<25} {value:.2f}")
        else:
            print(f"{metric:<25} {value}")


from dataclasses import dataclass

from src.config import RANDOM_STATE


@dataclass
class MonteCarloResults:
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


class MonteCarloAnalyzer:
    """
    Performs stationary block bootstrap Monte Carlo simulation for equity curves.

    The stationary block bootstrap preserves autocorrelation structure in returns
    (volatility clustering, momentum) while generating alternative historical paths
    to assess whether observed drawdowns are within expected statistical ranges.

    Key Features:
        - Overlapping blocks with geometric length distribution
        - Circular wrapping to avoid boundary bias
        - Fully vectorized simulation (all paths computed in parallel)
        - Percentile-based statistical inference

    Financial Interpretation:
        - Max DD < 25th percentile: Strategy unusually robust (check for overfitting)
        - Max DD in 25-75th percentile: Normal variation
        - Max DD > 75th percentile: Unlucky or biased backtest
    """

    def __init__(self, random_seed: int | None = None):

        self.random_seed = RANDOM_STATE

    def _generate_block_bootstrap_returns(self, returns: np.ndarray, n_sims: int, block_size: int) -> np.ndarray:
        """
        Stationary block bootstrap (Politis & Romano).
        """
        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(self.random_seed)

        n_days = len(returns)
        p = 1.0 / block_size  # geometric parameter

        synthetic = np.empty((n_sims, n_days), dtype=np.float64)

        for i in range(n_sims):
            t = 0
            while t < n_days:
                block_len = self.rng.geometric(p)
                start = self.rng.integers(0, n_days)

                size = min(block_len, n_days - t)

                # Circular wrapping
                idx = (start + np.arange(size)) % n_days
                synthetic[i, t : t + size] = returns[idx]

                t += size

        return synthetic

    def _calculate_equity_curves(self, returns_matrix: np.ndarray, initial_capital: float) -> np.ndarray:
        """
        Converts return matrix to equity curves via compounding.
        """
        # Add 1 to returns to get growth factors: (1 + r)
        growth_factors = 1.0 + returns_matrix

        # Cumulative product along time axis (axis=1)
        cumulative_growth = np.cumprod(growth_factors, axis=1)

        # Multiply by initial capital
        equity_from_returns = initial_capital * cumulative_growth

        # Prepend initial capital as the first column (t=0)
        # Shape: (n_sims, n_days) -> (n_sims, n_days+1)
        initial_capital_col = np.full((returns_matrix.shape[0], 1), initial_capital)
        equity_curves = np.hstack([initial_capital_col, equity_from_returns])

        return equity_curves

    def _calculate_drawdowns(self, equity_curves: np.ndarray) -> np.ndarray:
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (equity_curves - running_max) / running_max * 100.0
        return drawdowns

    def _calculate_drawdown_statistics(self, drawdowns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Max drawdown: most negative value per simulation (axis=1)
        max_dd = -np.min(drawdowns, axis=1)
        # Average drawdown: mean of absolute values per simulation
        in_dd = drawdowns < 0
        avg_dd = np.array(
            [-drawdowns[i, in_dd[i]].mean() if in_dd[i].any() else 0.0 for i in range(drawdowns.shape[0])]
        )

        return max_dd, avg_dd

    def run_simulation(self, equity: pd.Series, n_sims: int = 10000, block_size: int = 20) -> MonteCarloResults:
        """
        Executes Monte Carlo simulation using stationary block bootstrap.
        """
        print("Running Monte Carlo Simulation...")
        print(f"  Simulations: {n_sims:,}")
        print(f"  Block Size: {block_size} days")
        print(f"  Data Points: {len(equity):,}")

        # Step 1: Extract returns from equity
        returns = equity.pct_change().dropna().values
        initial_capital = equity.iloc[0]
        dates = equity.index

        # Validate data
        if len(returns) < block_size:
            raise ValueError(
                f"Equity series too short ({len(returns)} days) for block_size={block_size}. "
                f"Need at least {block_size} days."
            )

        print(f"  Initial Capital: ${initial_capital:,.0f}")
        print(f"  Return Stats: μ={returns.mean():.4f}, σ={returns.std():.4f}")

        # Step 2: Generate synthetic returns via block bootstrap
        print("\n[1/4] Generating block bootstrap returns...")
        synthetic_returns = self._generate_block_bootstrap_returns(returns, n_sims, block_size)

        # Step 3: Calculate equity curves
        print("[2/4] Computing equity curves...")
        simulated_equity = self._calculate_equity_curves(synthetic_returns, initial_capital)

        # Step 4: Calculate drawdowns
        print("[3/4] Calculating drawdowns...")
        simulated_drawdowns = self._calculate_drawdowns(simulated_equity)

        # Step 5: Compute drawdown statistics
        print("[4/4] Computing statistics...")
        simulated_max_dd, simulated_avg_dd = self._calculate_drawdown_statistics(simulated_drawdowns)

        # Calculate actual strategy statistics
        actual_equity_array = self._calculate_equity_curves(returns.reshape(1, -1), initial_capital)

        actual_drawdowns = self._calculate_drawdowns(actual_equity_array)

        actual_max_dd = -np.min(actual_drawdowns)
        actual_avg_dd = np.mean(np.abs(actual_drawdowns))

        # Compute percentiles for distribution analysis
        percentile_levels = [5, 25, 50, 75, 95]
        percentiles_max_dd = {p: np.percentile(simulated_max_dd, p) for p in percentile_levels}
        percentiles_avg_dd = {p: np.percentile(simulated_avg_dd, p) for p in percentile_levels}

        # Calculate where actual strategy ranks in the distribution
        # percentileofscore returns percentage of values ≤ score
        from scipy.stats import percentileofscore

        actual_max_dd_pct = percentileofscore(simulated_max_dd, actual_max_dd, kind="rank")
        actual_avg_dd_pct = percentileofscore(simulated_avg_dd, actual_avg_dd, kind="rank")

        # Print summary
        print("\n" + "=" * 70)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 70)
        print("\nMaximum Drawdown:")
        print(f"  Actual:      {actual_max_dd:>8.2f}% (Percentile: {actual_max_dd_pct:.1f})")
        print(
            f"  Simulated:   5th={percentiles_max_dd[5]:.2f}%, "
            f"25th={percentiles_max_dd[25]:.2f}%, "
            f"Median={percentiles_max_dd[50]:.2f}%, "
            f"75th={percentiles_max_dd[75]:.2f}%, "
            f"95th={percentiles_max_dd[95]:.2f}%"
        )

        print("\nAverage Drawdown:")
        print(f"  Actual:      {actual_avg_dd:>8.2f}% (Percentile: {actual_avg_dd_pct:.1f})")
        print(
            f"  Simulated:   5th={percentiles_avg_dd[5]:.2f}%, "
            f"25th={percentiles_avg_dd[25]:.2f}%, "
            f"Median={percentiles_avg_dd[50]:.2f}%, "
            f"75th={percentiles_avg_dd[75]:.2f}%, "
            f"95th={percentiles_avg_dd[95]:.2f}%"
        )

        print("=" * 70)

        # Package results
        results = MonteCarloResults(
            simulated_equity=simulated_equity,
            simulated_max_dd=simulated_max_dd,
            simulated_avg_dd=simulated_avg_dd,
            actual_equity=equity,
            actual_max_dd=actual_max_dd,
            actual_avg_dd=actual_avg_dd,
            percentiles_max_dd=percentiles_max_dd,
            percentiles_avg_dd=percentiles_avg_dd,
            actual_max_dd_percentile=actual_max_dd_pct,
            actual_avg_dd_percentile=actual_avg_dd_pct,
            dates=equity.index,
        )

        return results
