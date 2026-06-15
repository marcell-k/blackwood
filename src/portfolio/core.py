from typing import Any

import numpy as np
import pandas as pd
from src.config import CASH
from src.metrics.core import compute_all_metrics
from src.portfolio.optimizer import (
    EnsemblePositionSizer,
    EnsembleStrategy,
    OptimizationStrategy,
)
from src.portfolio.risk_allocator import CentralRiskAllocator


class PortfolioBacktester:
    def __init__(
        self,
        strategy: OptimizationStrategy,
        target_vol: float = 0.10,
        apply_leverage: bool = False,
        use_optimal_f_for_ensemble: bool = True,
        central_allocator: CentralRiskAllocator | None = None,
        allocator_freq: str = "W-FRI",
    ):
        self.strategy = strategy
        self.target_vol = target_vol
        self.apply_leverage = apply_leverage
        self.use_optimal_f_for_ensemble = use_optimal_f_for_ensemble
        self.central_allocator = central_allocator
        self.allocator_freq = allocator_freq

    @staticmethod
    def _infer_annualization_factor(index: pd.DatetimeIndex) -> float:
        idx_utc = index.tz_convert("UTC")
        day_counts = pd.Series(1, index=idx_utc.normalize()).groupby(level=0).sum()
        bars_per_day = float(day_counts.median())
        return bars_per_day * 252.0

    @staticmethod
    def _to_utc_day(ts: pd.Timestamp) -> pd.Timestamp:
        ts_utc = ts.tz_localize("UTC") if ts.tz_info is None else ts.tz_convert("UTC")
        return ts_utc.normalize()

    @staticmethod
    def _normalize_risk_rules(risk_rules: dict[str, Any] | None, annualization_factor: float) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "enabled": False,
            "intraday_mode": False,
            "day_boundary": "UTC",
            "close_daily_loss_threshold": 0.038,
            "half_risk_on_drawdown": 0.075,
            "half_risk_off_drawdown": 0.070,
            "half_risk_multiplier": 0.5,
            "timeframe": "30min",
        }
        config = defaults.copy()
        if risk_rules is not None:
            config.update(risk_rules)

        config["enabled"] = bool(config.get("enabled", False))
        config["intraday_mode"] = bool(config.get("intraday_mode", False))
        config["day_boundary"] = str(config.get("day_boundary", "UTC"))
        config["close_daily_loss_threshold"] = float(config.get("close_daily_loss_threshold", 0.038))
        config["half_risk_on_drawdown"] = float(config.get("half_risk_on_drawdown", 0.075))
        config["half_risk_off_drawdown"] = float(config.get("half_risk_off_drawdown", 0.070))
        config["half_risk_multiplier"] = float(config.get("half_risk_multiplier", 0.5))
        config["timeframe"] = str(config.get("timeframe", "30min")).strip()

        if not config["enabled"]:
            return config

        return config

    @staticmethod
    def _resample_returns(returns: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if returns.empty:
            return returns
        gross_returns = 1.0 + returns
        gross_resampled = gross_returns.resample(timeframe).prod(min_count=1)
        returns_resampled = (gross_resampled - 1.0).dropna(how="all")
        return returns_resampled.sort_index()

    def _compute_scale_factors(
        self,
        returns: pd.DataFrame,
        vol_method: str = "geometric",
        min_vol_floor: float = 0.05,
        max_scale_factor: float = 10.0,
        min_obs_for_full_trust: int = 0,
        annualization_factor: float | pd.Series = 252.0,
    ) -> tuple[pd.Series, pd.Series]:
        if vol_method not in ["arithmetic", "geometric"]:
            raise ValueError("vol_method must be either 'arithmetic' or 'geometric'")

        if isinstance(annualization_factor, pd.Series):
            ann_factors = annualization_factor.reindex(returns.columns).fillna(252.0)
        else:
            ann_factors = pd.Series(annualization_factor, index=returns.columns)

        n_obs = returns.notna().sum()

        if vol_method == "arithmetic":
            annual_vols = returns.std(ddof=1) * np.sqrt(ann_factors)

        else:  # geometric — vectorized via log-return identity
            log_returns = np.log1p(returns)
            log_sum = log_returns.sum()  # skipna=True skips NaN per column
            log_gmean = log_sum / n_obs.clip(lower=1)
            gmean_day_return = np.exp(log_gmean) - 1
            var_daily = returns.var(ddof=1)

            annual_vols = pd.Series(np.nan, index=returns.columns, dtype=float)
            has_data = n_obs > 0
            af = ann_factors[has_data]
            g = gmean_day_return[has_data]
            v = var_daily[has_data]
            annual_vols[has_data] = np.sqrt((v + (1 + g) ** 2) ** af - (1 + g) ** (2 * af))

        # Bayesian vol shrinkage: shrink toward cross-sectional median for strategies with limited data
        if min_obs_for_full_trust > 0:
            mature_mask = n_obs >= min_obs_for_full_trust
            prior_vol = annual_vols[mature_mask].dropna().median()
            if np.isnan(prior_vol):
                prior_vol = self.target_vol

            alpha = ((min_obs_for_full_trust - n_obs) / min_obs_for_full_trust).clip(lower=0.0, upper=1.0)
            shrink_mask = (alpha > 0) & annual_vols.notna()
            annual_vols[shrink_mask] = (1 - alpha[shrink_mask]) * annual_vols[shrink_mask] + alpha[
                shrink_mask
            ] * prior_vol

        # Apply volatility floor to prevent division by near-zero
        annual_vols_safe = annual_vols.clip(lower=min_vol_floor)

        # Compute scale factors with floor before division, cap after division
        scale_factors = (self.target_vol / annual_vols_safe).clip(upper=max_scale_factor)

        return scale_factors, annual_vols

    def _setup_allocator(
        self, idx: pd.DatetimeIndex, returns: pd.DataFrame, allocator_start_date: pd.Timestamp | None
    ) -> tuple[pd.Timestamp | None, pd.DatetimeIndex]:
        """Fit the central allocator if needed and compute its rebalance dates."""
        if self.central_allocator is None:
            return None, pd.DatetimeIndex([])

        allocator_start_ts: pd.Timestamp | None = None
        if allocator_start_date is not None:
            allocator_start_ts = pd.Timestamp(allocator_start_date)
            if idx.tz is not None:
                if allocator_start_ts.tz is None:
                    allocator_start_ts = allocator_start_ts.tz_localize(idx.tz)
                else:
                    allocator_start_ts = allocator_start_ts.tz_convert(idx.tz)
            elif allocator_start_ts.tz is not None:
                allocator_start_ts = allocator_start_ts.tz_localize(None)

        if not self.central_allocator.is_fitted:
            if allocator_start_ts is None:
                raise ValueError("central_allocator must be pre-fitted or allocator_start_date must be provided.")
            train_returns = returns.loc[returns.index < allocator_start_ts]
            if train_returns.empty:
                raise ValueError("No train data found before allocator_start_date for allocator.fit().")
            self.central_allocator.fit(train_returns)

        alloc_freq_ends = pd.date_range(idx.min(), idx.max(), freq=self.allocator_freq)
        alloc_idx = idx[idx.get_indexer(alloc_freq_ends, method="ffill")]
        allocator_rebalance_dates = pd.DatetimeIndex(sorted(set(alloc_idx)))

        return allocator_start_ts, allocator_rebalance_dates

    def _build_rebalance_schedule(
        self,
        idx: pd.DatetimeIndex,
        rebalance_freq: str | dict[str, str],
        strategy_names: list[str],
    ) -> tuple[dict[str, set], pd.DatetimeIndex, dict]:
        """Compute per-strategy rebalance date sets, their sorted union, and reverse mapping."""
        # Group strategies by frequency to avoid redundant date_range calls
        freq_to_strategies: dict[str, list[str]] = {}
        for name in strategy_names:
            freq = rebalance_freq.get(name, rebalance_freq) if isinstance(rebalance_freq, dict) else rebalance_freq
            freq_to_strategies.setdefault(freq, []).append(name)

        strategy_rebalance_dates: dict[str, set] = {}
        freq_dates_cache: dict[str, set] = {}
        for freq, names in freq_to_strategies.items():
            if freq not in freq_dates_cache:
                freq_ends = pd.date_range(idx.min(), idx.max(), freq=freq)
                freq_dates_cache[freq] = set(idx[idx.get_indexer(freq_ends, method="ffill")])
            for name in names:
                strategy_rebalance_dates[name] = freq_dates_cache[freq]

        all_rebalance_dates = pd.DatetimeIndex(sorted(set().union(*strategy_rebalance_dates.values())))

        # Reverse mapping: date -> [strategies rebalancing on that date]
        date_to_strategies: dict = {}
        for name, dates in strategy_rebalance_dates.items():
            for d in dates:
                date_to_strategies.setdefault(d, []).append(name)

        return strategy_rebalance_dates, all_rebalance_dates, date_to_strategies

    @staticmethod
    def _pre_compute_rebalance_counts(
        all_rebalance_dates: pd.DatetimeIndex,
        strategy_rebalance_dates: dict[str, set],
        strategy_first_data: dict,
        strategy_names: list[str],
    ) -> dict:
        """Pre-compute past_rebalances_counts at each rebalance date (before increment)."""
        running = dict.fromkeys(strategy_names, 0)
        result = {}
        for date in all_rebalance_dates:
            result[date] = running.copy()
            for strat in strategy_names:
                if (
                    date in strategy_rebalance_dates[strat]
                    and strategy_first_data.get(strat) is not None
                    and strategy_first_data[strat] <= date
                ):
                    running[strat] += 1
        return result

    def _rebalance_step(
        self,
        date,
        i: int,
        returns: pd.DataFrame,
        strategies_rebalancing_today: list[str],
        lookback_periods: int | dict[str, int],
        vol_method: str,
        min_vol_threshold: float,
        min_vol_floor: float,
        max_scale_factor: float,
        verbose: bool,
        warmup_periods: int,
        ramp_periods: int,
        min_obs_for_full_trust: int,
        per_strategy_ann_factors: pd.Series,
        annualization_factor: float,
        strategy_first_data: dict,
        past_rebalances_counts: dict[str, int],
        rebalance_freq: str | dict[str, str],
    ) -> tuple | None:
        """
        Compute new weights and leverage at a rebalance date.

        Returns a 7-tuple (weights, base_weights, scale_factors, leverage,
        leverage_entry, audit_entry, weights_updated), or None if no usable data.
        When no active strategies, weights/leverage fields are None but audit_entry
        is still populated.
        """
        lookback_vals = []
        for s in strategies_rebalancing_today:
            lb = lookback_periods.get(s, lookback_periods) if isinstance(lookback_periods, dict) else lookback_periods
            if lb is not None:
                lookback_vals.append(lb)

        if not lookback_vals:
            if verbose:
                print(f"{date}: No valid lookback periods for rebalancing strategies.")
            return None

        max_lookback = max(lookback_vals)
        lookback_returns = returns.iloc[max(0, i - max_lookback) : i]
        lookback_clean = lookback_returns.dropna(how="all")

        if lookback_clean.empty:
            if verbose:
                print(f"{date}: No usable data in lookback; skipping base rebalance.")
            return None

        valid_data_pct = lookback_clean.notna().sum() / len(lookback_clean)
        lookback_filled = lookback_clean.ffill().fillna(0)

        scale_factors, annual_vols = self._compute_scale_factors(
            lookback_clean,
            vol_method,
            min_vol_floor,
            max_scale_factor,
            min_obs_for_full_trust=min_obs_for_full_trust,
            annualization_factor=per_strategy_ann_factors,
        )

        active_mask = annual_vols >= min_vol_threshold

        if warmup_periods > 0:
            for strat in returns.columns:
                if strategy_first_data.get(strat) is None or past_rebalances_counts[strat] < warmup_periods:
                    active_mask[strat] = False

        active_strategies = annual_vols[active_mask].index.tolist()

        rebalance_freqs_entry = (
            {s: rebalance_freq[s] for s in strategies_rebalancing_today}
            if isinstance(rebalance_freq, dict)
            else dict.fromkeys(strategies_rebalancing_today, rebalance_freq)
        )

        if not active_strategies:
            if verbose:
                print(f"{date}: No active strategies above vol threshold.")
            audit_entry = {
                "date": date,
                "annual_vols": annual_vols.to_dict(),
                "scale_factors": scale_factors.to_dict(),
                "lookback_start": lookback_filled.index[0],
                "lookback_end": lookback_filled.index[-1],
                "valid_data_points": len(lookback_clean),
                "data_quality": valid_data_pct.to_dict(),
                "strategies_rebalancing": strategies_rebalancing_today,
                "rebalance_frequencies": rebalance_freqs_entry,
                "past_rebalances": past_rebalances_counts.copy(),
                "ramp_factors": {},
            }
            zero_weights = pd.Series(0.0, index=returns.columns)
            zero_sf = pd.Series(0.0, index=returns.columns)
            no_lev = {
                "date": date,
                "leverage": 1.0,
                "realized_vol": None,
                "strategies_rebalancing": strategies_rebalancing_today,
                "sizing_method": "none",
            }
            return zero_weights, zero_weights.copy(), zero_sf, 1.0, no_lev, audit_entry, True

        lookback_active = lookback_filled[active_strategies]
        scale_factors_active = scale_factors[active_strategies]
        lookback_returns_normalized = lookback_active * scale_factors_active

        if len(active_strategies) == 1:
            new_weights_active = pd.Series([1.0], index=active_strategies)
        else:
            new_weights_active = self.strategy.compute_weights(lookback_returns_normalized)
            new_weights_active = new_weights_active / new_weights_active.sum()

        new_weights = pd.Series(0.0, index=returns.columns)
        new_weights[active_strategies] = new_weights_active

        ramp_factors: dict[str, float] = {}
        if ramp_periods > 0:
            for strat in active_strategies:
                periods_since_warmup = past_rebalances_counts[strat] - warmup_periods + 1
                ramp_factor = min(1.0, max(0.0, periods_since_warmup / ramp_periods))
                new_weights[strat] *= ramp_factor
                ramp_factors[strat] = ramp_factor
            weight_sum = new_weights.sum()
            if weight_sum > 0:
                new_weights = new_weights / weight_sum

        new_base_weights = new_weights.copy()
        new_scale_factors = scale_factors.copy()
        new_scale_factors[~active_mask] = 0.0

        # Portfolio-level leverage
        fractional_kelly = 0.5
        if self.apply_leverage:
            port_returns_lookback = (lookback_returns_normalized @ new_weights_active).dropna()
            realized_vol = port_returns_lookback.std() * np.sqrt(annualization_factor)

            if isinstance(self.strategy, EnsembleStrategy) and self.use_optimal_f_for_ensemble:
                sizer = EnsemblePositionSizer(
                    ensemble=self.strategy,
                    lookback_window=len(lookback_returns_normalized),
                )
                sizer_result = sizer.compute_targets(
                    returns=lookback_returns_normalized,
                    as_of=lookback_returns_normalized.index[-1],
                )
                risk_fraction = float(sizer_result["risk_fraction"])
                optimal_f = float(sizer_result["optimal_f"])
                largest_loss = float(sizer_result["largest_loss"])
                new_leverage = fractional_kelly * risk_fraction
                leverage_entry = {
                    "date": date,
                    "leverage": new_leverage,
                    "realized_vol": realized_vol,
                    "strategies_rebalancing": strategies_rebalancing_today,
                    "sizing_method": "optimal_f",
                    "risk_fraction": risk_fraction,
                    "optimal_f": optimal_f,
                    "largest_loss": largest_loss,
                }
            else:
                new_leverage = float(self.target_vol / realized_vol) if realized_vol > 0 else 1.0
                leverage_entry = {
                    "date": date,
                    "leverage": new_leverage,
                    "realized_vol": realized_vol,
                    "strategies_rebalancing": strategies_rebalancing_today,
                    "sizing_method": "target_vol",
                }
        else:
            new_leverage = 1.0
            leverage_entry = {
                "date": date,
                "leverage": 1.0,
                "realized_vol": None,
                "strategies_rebalancing": strategies_rebalancing_today,
                "sizing_method": "none",
            }

        audit_entry = {
            "date": date,
            "annual_vols": annual_vols.to_dict(),
            "scale_factors": scale_factors.to_dict(),
            "lookback_start": lookback_filled.index[0],
            "lookback_end": lookback_filled.index[-1],
            "valid_data_points": len(lookback_clean),
            "data_quality": valid_data_pct.to_dict(),
            "strategies_rebalancing": strategies_rebalancing_today,
            "rebalance_frequencies": rebalance_freqs_entry,
            "past_rebalances": past_rebalances_counts.copy(),
            "ramp_factors": ramp_factors,
        }

        return new_weights, new_base_weights, new_scale_factors, new_leverage, leverage_entry, audit_entry, True

    def _finalize_results(
        self,
        equity: pd.Series,
        weights_history: list,
        scale_factors_by_date: list,
        leverage_history: list,
        allocator_history: list,
        risk_rules_audit: list,
        risk_rules_config: dict,
        rebalance_freq: str | dict[str, str],
        lookback_periods: int | dict[str, int],
        warmup_periods: int,
        ramp_periods: int,
        min_obs_for_full_trust: int,
        annualization_factor: float,
    ) -> dict[str, Any]:
        """Assemble and return the backtest results dict."""
        weights_df = pd.DataFrame(weights_history).set_index("date") if weights_history else pd.DataFrame()
        daily_returns = equity.pct_change().fillna(0)
        leverage_history_df = pd.DataFrame(leverage_history).set_index("date") if leverage_history else pd.DataFrame()
        allocator_history_df = (
            pd.DataFrame(
                [
                    {
                        "date": entry.get("date"),
                        "turnover": entry.get("overlay_stats", {}).get("turnover", np.nan),
                        "turnover_cap": entry.get("overlay_stats", {}).get("turnover_cap", np.nan),
                        "turnover_limit_applied": entry.get("overlay_stats", {}).get("turnover_limit_applied", False),
                        "bounds_clipped": entry.get("overlay_stats", {}).get("bounds_clipped", False),
                    }
                    for entry in allocator_history
                ]
            ).set_index("date")
            if allocator_history
            else pd.DataFrame()
        )
        risk_rules_audit_df = pd.DataFrame(risk_rules_audit).set_index("date") if risk_rules_audit else pd.DataFrame()

        last_leverage_ratio = 1.0
        last_realized_vol = None
        if not leverage_history_df.empty:
            last_leverage_ratio = float(leverage_history_df["leverage"].iloc[-1])
            if "realized_vol" in leverage_history_df.columns:
                last_realized_vol = leverage_history_df["realized_vol"].iloc[-1]
                if pd.notna(last_realized_vol):
                    last_realized_vol = float(last_realized_vol)

        return {
            "equity": equity,
            "weights_history": weights_df,
            "daily_returns": daily_returns,
            "strategy": self.strategy,
            "scale_factors_by_date": scale_factors_by_date,
            "leverage_history": leverage_history,
            "leverage_history_df": leverage_history_df,
            "leverage_applied": self.apply_leverage,
            "leverage_ratio": last_leverage_ratio,
            "realized_vol_pre_leverage": last_realized_vol,
            "allocator_enabled": self.central_allocator is not None,
            "allocator_freq": self.allocator_freq if self.central_allocator is not None else None,
            "allocator_history": allocator_history,
            "allocator_history_df": allocator_history_df,
            "risk_rules": risk_rules_config,
            "risk_rules_audit": risk_rules_audit,
            "risk_rules_audit_df": risk_rules_audit_df,
            "normalized": True,
            "rebalance_freq_dict": rebalance_freq,
            "lookback_periods_dict": lookback_periods,
            "warmup_periods": warmup_periods,
            "ramp_periods": ramp_periods,
            "min_obs_for_full_trust": min_obs_for_full_trust,
            "annualization_factor": annualization_factor,
        }

    def backtest(
        self,
        returns: pd.DataFrame,
        rebalance_freq: str | dict[str, str] = "QE",
        lookback_periods: int | dict[str, int] = 252,
        initial_capital: float = CASH,
        vol_method: str = "arithmetic",
        min_vol_threshold: float = 0.0001,
        min_vol_floor: float = 0.05,
        max_scale_factor: float = 10.0,
        verbose: bool = False,
        allocator_start_date: pd.Timestamp | None = None,
        warmup_periods: int = 0,
        ramp_periods: int = 0,
        min_obs_for_full_trust: int = 0,
        risk_rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Backtest strategy with dynamic normalization and per-rebalance leverage.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns for each strategy (columns = strategies, index = dates)
        rebalance_freq : Union[str, dict[str, str]], default 'QE'
            Rebalance frequency or per-strategy dict
        lookback_periods : Union[int, dict[str, int]], default 252
            Lookback window or per-strategy dict
        initial_capital : float, default 10_000_000
        vol_method : str, default 'arithmetic'
        min_vol_threshold : float, default 0.0001
        min_vol_floor : float, default 0.05
        max_scale_factor : float, default 10.0
        verbose : bool, default False
        allocator_start_date : Optional[pd.Timestamp], default None
        warmup_periods : int, default 0
        ramp_periods : int, default 0
        min_obs_for_full_trust : int, default 0
        risk_rules : Optional[dict[str, Any]], default None

        """
        # ---- Setup ----
        idx = returns.index
        annualization_factor = self._infer_annualization_factor(idx)
        risk_rules_config = self._normalize_risk_rules(risk_rules, annualization_factor)
        risk_rules_enabled = bool(risk_rules_config["enabled"])
        if risk_rules_enabled:
            returns = self._resample_returns(returns, risk_rules_config["timeframe"])
            idx = returns.index
            annualization_factor = self._infer_annualization_factor(idx)
            risk_rules_config = self._normalize_risk_rules(risk_rules_config, annualization_factor)

        per_strategy_ann_factors = pd.Series(
            {col: self._infer_annualization_factor(returns[col].dropna().index) for col in returns.columns}
        )

        allocator_start_ts, allocator_rebalance_dates = self._setup_allocator(idx, returns, allocator_start_date)
        strategy_rebalance_dates, all_rebalance_dates, date_to_strategies = self._build_rebalance_schedule(
            idx, rebalance_freq, returns.columns.tolist()
        )
        strategy_first_data = {}
        for col in returns.columns:
            s = returns[col].dropna()
            nonzero = s[s != 0]
            strategy_first_data[col] = nonzero.index[0] if not nonzero.empty else None
        past_rebalances_at = self._pre_compute_rebalance_counts(
            all_rebalance_dates, strategy_rebalance_dates, strategy_first_data, returns.columns.tolist()
        )

        # ---- Pre-compute numpy arrays and lookup sets ----
        returns_np = returns.fillna(0).values
        n_bars, n_assets = returns_np.shape

        all_rebalance_set = set(all_rebalance_dates)
        allocator_effective_set = set()
        if self.central_allocator is not None:
            for d in allocator_rebalance_dates:
                if allocator_start_ts is None or d >= allocator_start_ts:
                    allocator_effective_set.add(d)

        # Sorted event dates (rebalance or allocator) with bar positions
        all_event_dates = sorted(all_rebalance_set | allocator_effective_set)
        if all_event_dates:
            event_bar_positions = idx.get_indexer(pd.DatetimeIndex(all_event_dates))
            events = list(zip(event_bar_positions.tolist(), all_event_dates, strict=False))
        else:
            events = []

        # ---- Initialize state ----
        equity_np = np.empty(n_bars, dtype=np.float64)
        equity_np[0] = float(initial_capital)
        if warmup_periods > 0:
            weights_np = np.zeros(n_assets, dtype=np.float64)
            base_weights_pd = pd.Series(0.0, index=returns.columns)
            scale_factors_np = np.zeros(n_assets, dtype=np.float64)
        else:
            weights_np = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
            base_weights_pd = pd.Series(1.0 / n_assets, index=returns.columns)
            scale_factors_np = np.ones(n_assets, dtype=np.float64)
        leverage = 1.0

        weights_history: list[dict] = []
        scale_factors_by_date: list[dict] = []
        leverage_history: list[dict] = []
        allocator_history: list[dict] = []
        risk_rules_audit: list[dict] = []

        # Shared rebalance kwargs (passed to _rebalance_step unchanged)
        _rb_kwargs = dict(
            returns=returns,
            lookback_periods=lookback_periods,
            vol_method=vol_method,
            min_vol_threshold=min_vol_threshold,
            min_vol_floor=min_vol_floor,
            max_scale_factor=max_scale_factor,
            verbose=verbose,
            warmup_periods=warmup_periods,
            ramp_periods=ramp_periods,
            min_obs_for_full_trust=min_obs_for_full_trust,
            per_strategy_ann_factors=per_strategy_ann_factors,
            annualization_factor=annualization_factor,
            strategy_first_data=strategy_first_data,
            rebalance_freq=rebalance_freq,
        )

        if not risk_rules_enabled:
            # ===== VECTORIZED PATH: segment-based equity computation =====
            segment_start = 1  # bar 0 stays at initial_capital

            for event_bar, event_date in events:
                # Vectorize segment [segment_start, event_bar) with current weights
                if segment_start > 0 and segment_start < event_bar:
                    combined = weights_np * scale_factors_np * leverage
                    seg_ret = returns_np[segment_start:event_bar] @ combined
                    equity_np[segment_start:event_bar] = equity_np[segment_start - 1] * np.cumprod(1.0 + seg_ret)

                # Process rebalance and/or allocator at this event
                is_rebalance = event_date in all_rebalance_set
                is_allocator = event_date in allocator_effective_set
                weights_updated = False
                strategies_today = date_to_strategies.get(event_date, [])

                if is_rebalance:
                    step_result = self._rebalance_step(
                        date=event_date,
                        i=event_bar,
                        strategies_rebalancing_today=strategies_today,
                        past_rebalances_counts=past_rebalances_at[event_date],
                        **_rb_kwargs,
                    )
                    if step_result is not None:
                        new_w, new_bw, new_sf, new_lev, lev_entry, audit, updated = step_result
                        if updated:
                            weights_np = new_w.values.copy()
                            base_weights_pd = new_bw
                            scale_factors_np = new_sf.values.copy()
                            leverage = new_lev
                            leverage_history.append(lev_entry)
                            weights_updated = True
                        if audit is not None:
                            scale_factors_by_date.append(audit)

                if is_allocator:
                    multipliers = self.central_allocator.compute_multipliers(
                        as_of=event_date,
                        returns_hist=returns.iloc[:event_bar],
                    )
                    adjusted = self.central_allocator.apply_overlay(
                        base_weights=base_weights_pd,
                        multipliers=multipliers,
                    )
                    weights_np = adjusted.reindex(returns.columns).fillna(0).values.copy()
                    weights_updated = True
                    allocator_history.append(
                        {
                            "date": event_date,
                            "base_weights": base_weights_pd.to_dict(),
                            "multipliers": multipliers.reindex(base_weights_pd.index).fillna(1.0).to_dict(),
                            "adjusted_weights": adjusted.to_dict(),
                            "score_components": self.central_allocator.last_score_components,
                            "overlay_stats": self.central_allocator.last_overlay_stats,
                            "strategies_rebalancing": strategies_today,
                            "allocator_freq": self.allocator_freq,
                        }
                    )

                if weights_updated:
                    weights_pd = pd.Series(weights_np, index=returns.columns)
                    weights_history.append({"date": event_date, **weights_pd.to_dict()})

                # Apply event bar's return with (possibly updated) weights
                if event_bar > 0:
                    combined = weights_np * scale_factors_np * leverage
                    equity_np[event_bar] = equity_np[event_bar - 1] * (1.0 + returns_np[event_bar] @ combined)

                segment_start = event_bar + 1

            # Final segment after last event
            if segment_start < n_bars:
                combined = weights_np * scale_factors_np * leverage
                seg_ret = returns_np[segment_start:] @ combined
                equity_np[segment_start:] = equity_np[segment_start - 1] * np.cumprod(1.0 + seg_ret)

        else:
            # ===== STATEFUL PATH: numpy-optimized per-bar loop (risk rules) =====
            # Pre-compute UTC days to avoid per-bar timezone work
            utc_days = idx.tz_convert("UTC").normalize() if idx.tz is not None else idx.tz_localize("UTC").normalize()

            risk_state = {
                "peak_equity": float(initial_capital),
                "current_drawdown": 0.0,
                "half_risk_active": False,
                "day_kill_active": False,
                "current_utc_day": utc_days[0] if n_bars > 0 else None,
                "day_start_equity": float(initial_capital),
            }
            half_risk_mult = risk_rules_config["half_risk_multiplier"]
            close_threshold = risk_rules_config["close_daily_loss_threshold"]
            half_on = risk_rules_config["half_risk_on_drawdown"]
            half_off = risk_rules_config["half_risk_off_drawdown"]

            for i in range(n_bars):
                date = idx[i]
                weights_updated_today = False
                strategies_today: list[str] = []

                if date in all_rebalance_set:
                    strategies_today = date_to_strategies.get(date, [])
                    step_result = self._rebalance_step(
                        date=date,
                        i=i,
                        strategies_rebalancing_today=strategies_today,
                        past_rebalances_counts=past_rebalances_at.get(date, dict.fromkeys(returns.columns, 0)),
                        **_rb_kwargs,
                    )
                    if step_result is not None:
                        new_w, new_bw, new_sf, new_lev, lev_entry, audit, updated = step_result
                        if updated:
                            weights_np = new_w.values.copy()
                            base_weights_pd = new_bw
                            scale_factors_np = new_sf.values.copy()
                            leverage = new_lev
                            leverage_history.append(lev_entry)
                            weights_updated_today = True
                        if audit is not None:
                            scale_factors_by_date.append(audit)

                if date in allocator_effective_set:
                    multipliers = self.central_allocator.compute_multipliers(
                        as_of=date,
                        returns_hist=returns.iloc[:i],
                    )
                    adjusted = self.central_allocator.apply_overlay(
                        base_weights=base_weights_pd,
                        multipliers=multipliers,
                    )
                    weights_np = adjusted.reindex(returns.columns).fillna(0).values.copy()
                    weights_updated_today = True
                    allocator_history.append(
                        {
                            "date": date,
                            "base_weights": base_weights_pd.to_dict(),
                            "multipliers": multipliers.reindex(base_weights_pd.index).fillna(1.0).to_dict(),
                            "adjusted_weights": adjusted.to_dict(),
                            "score_components": self.central_allocator.last_score_components,
                            "overlay_stats": self.central_allocator.last_overlay_stats,
                            "strategies_rebalancing": strategies_today,
                            "allocator_freq": self.allocator_freq,
                        }
                    )

                if weights_updated_today:
                    weights_pd = pd.Series(weights_np, index=returns.columns)
                    weights_history.append({"date": date, **weights_pd.to_dict()})

                # ---- Inlined bar return with numpy (risk rules) ----
                if i == 0:
                    risk_rules_audit.append(
                        {
                            "date": date,
                            "utc_day": risk_state["current_utc_day"],
                            "day_return_cum": 0.0,
                            "day_kill_active": False,
                            "kill_triggered_this_bar": False,
                            "current_drawdown": 0.0,
                            "half_risk_active": False,
                            "base_leverage": leverage,
                            "effective_leverage": leverage,
                        }
                    )
                    continue

                # Day boundary detection via pre-computed UTC days
                bar_utc_day = utc_days[i]
                if risk_state["current_utc_day"] is None:
                    risk_state["current_utc_day"] = bar_utc_day
                    risk_state["day_start_equity"] = equity_np[i - 1]
                elif bar_utc_day != risk_state["current_utc_day"]:
                    risk_state["current_utc_day"] = bar_utc_day
                    risk_state["day_start_equity"] = equity_np[i - 1]
                    risk_state["day_kill_active"] = False

                # Effective leverage with risk overlays
                effective_leverage = leverage
                if risk_state["half_risk_active"]:
                    effective_leverage *= half_risk_mult
                if risk_state["day_kill_active"]:
                    effective_leverage = 0.0

                # Portfolio return via numpy dot product
                combined = weights_np * scale_factors_np * effective_leverage
                port_ret = returns_np[i] @ combined
                equity_np[i] = equity_np[i - 1] * (1.0 + port_ret)

                # Update risk state
                current_equity = equity_np[i]
                peak = risk_state["peak_equity"]
                if current_equity > peak:
                    risk_state["peak_equity"] = current_equity
                    peak = current_equity

                risk_state["current_drawdown"] = max(0.0, 1.0 - (current_equity / peak)) if peak > 0 else 0.0

                if risk_state["half_risk_active"]:
                    if risk_state["current_drawdown"] < half_off:
                        risk_state["half_risk_active"] = False
                else:
                    if risk_state["current_drawdown"] >= half_on:
                        risk_state["half_risk_active"] = True

                kill_triggered = False
                day_return_cum = (
                    (current_equity / risk_state["day_start_equity"]) - 1.0
                    if risk_state["day_start_equity"] > 0
                    else 0.0
                )
                if not risk_state["day_kill_active"] and day_return_cum <= -close_threshold:
                    risk_state["day_kill_active"] = True
                    kill_triggered = True

                risk_rules_audit.append(
                    {
                        "date": date,
                        "utc_day": risk_state["current_utc_day"],
                        "day_return_cum": day_return_cum,
                        "day_kill_active": risk_state["day_kill_active"],
                        "kill_triggered_this_bar": kill_triggered,
                        "current_drawdown": risk_state["current_drawdown"],
                        "half_risk_active": risk_state["half_risk_active"],
                        "base_leverage": leverage,
                        "effective_leverage": effective_leverage,
                    }
                )

        equity = pd.Series(equity_np, index=idx, dtype=np.float64)
        return self._finalize_results(
            equity=equity,
            weights_history=weights_history,
            scale_factors_by_date=scale_factors_by_date,
            leverage_history=leverage_history,
            allocator_history=allocator_history,
            risk_rules_audit=risk_rules_audit,
            risk_rules_config=risk_rules_config,
            rebalance_freq=rebalance_freq,
            lookback_periods=lookback_periods,
            warmup_periods=warmup_periods,
            ramp_periods=ramp_periods,
            min_obs_for_full_trust=min_obs_for_full_trust,
            annualization_factor=annualization_factor,
        )

    def get_performance_metrics(self, results: dict) -> dict[str, float]:
        """Calculate performance metrics from backtest results (arithmetic and geometric)."""
        equity = results["equity"]
        daily_returns = results["daily_returns"]
        annualization_factor = float(results.get("annualization_factor", 252.0))

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_days = len(equity)

        # --- Arithmetic Metrics ---
        annualized_return_arith = daily_returns.mean() * annualization_factor
        annualized_volatility_arith = daily_returns.std(ddof=1) * np.sqrt(annualization_factor)
        sharpe_ratio_arith = (
            annualized_return_arith / annualized_volatility_arith if annualized_volatility_arith > 0 else 0
        )

        # --- Geometric Metrics ---
        gmean_day_return = (1 + daily_returns).prod() ** (1 / n_days) - 1
        annualized_return_geom = (1 + gmean_day_return) ** annualization_factor - 1
        var_daily = daily_returns.var(ddof=1)
        annualized_volatility_geom = np.sqrt(
            (var_daily + (1 + gmean_day_return) ** 2) ** annualization_factor
            - (1 + gmean_day_return) ** (2 * annualization_factor)
        )
        sharpe_ratio_geom = annualized_return_geom / annualized_volatility_geom if annualized_volatility_geom > 0 else 0

        # --- Classic CAGR & MaxDD/Calmar calculations ---
        annualized_return = (1 + total_return) ** (annualization_factor / n_days) - 1
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_return_geometric": annualized_return_geom,
            "annualized_return_arithmetic": annualized_return_arith,
            "annualized_volatility": daily_returns.std() * np.sqrt(annualization_factor),
            "annualized_volatility_geometric": annualized_volatility_geom,
            "annualized_volatility_arithmetic": annualized_volatility_arith,
            "sharpe_ratio": annualized_return / (daily_returns.std() * np.sqrt(annualization_factor))
            if daily_returns.std() > 0
            else 0,
            "sharpe_ratio_geometric": sharpe_ratio_geom,
            "sharpe_ratio_arithmetic": sharpe_ratio_arith,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "annualization_factor": annualization_factor,
        }

        if results.get("leverage_applied"):
            metrics["leverage_ratio"] = results["leverage_ratio"]
            metrics["realized_vol_pre_leverage"] = results["realized_vol_pre_leverage"]

        return metrics


class PortfolioMetrics:
    def __init__(self, all_results: dict, returns: pd.DataFrame):
        self.all_results = all_results
        self.returns = returns

    def calculate_all_metrics(self) -> pd.DataFrame:
        metrics_list = []

        for name, data in self.all_results.items():
            metrics = data["metrics"].copy()
            metrics["Strategy"] = name
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list).set_index("Strategy")
        return metrics_df

    def print_metrics(self, metrics_df: pd.DataFrame):
        print(f"\n{'=' * 80}")
        print("ALL STRATEGIES - PERFORMANCE METRICS")
        print(f"{'=' * 80}")
        print(metrics_df.round(6).to_string())

        print(f"\n{'=' * 80}")
        print("RANKED BY SHARPE RATIO")
        print(f"{'=' * 80}")
        ranked = metrics_df[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]].sort_values(
            "sharpe_ratio", ascending=False
        )
        print(ranked.round(6).to_string())

    def print_equity_comparison(self):
        print(f"\n{'=' * 80}")
        print("EQUITY CURVE STATISTICS")
        print(f"{'=' * 80}")

        equity_summary = {}
        for name, data in self.all_results.items():
            equity = data["results"]["equity"]
            total_return = equity.iloc[-1] / equity.iloc[0] - 1

            equity_summary[name] = {
                "Start Value": f"${equity.iloc[0]:,.0f}",
                "End Value": f"${equity.iloc[-1]:,.0f}",
                "Total Return %": f"{total_return * 100:.2f}%",
                "Max Equity": f"${equity.max():,.0f}",
                "Min Equity": f"${equity.min():,.0f}",
                "Peak-to-Trough %": f"{(equity.min() - equity.max()) / equity.max() * 100:.2f}%",
            }

        equity_df = pd.DataFrame(equity_summary).T
        print(equity_df.to_string())

    def print_final_allocations(self) -> pd.DataFrame:
        print(f"\n{'=' * 80}")
        print("FINAL PORTFOLIO ALLOCATIONS (Latest Rebalance)")
        print(f"{'=' * 80}")

        final_alloc_dict = {}

        for name, data in self.all_results.items():
            weights = data["results"]["weights_history"]
            if not weights.empty:
                final_alloc_dict[name] = weights.iloc[-1]

        final_alloc_df = pd.DataFrame(final_alloc_dict).T
        print(final_alloc_df.round(4).to_string())

        return final_alloc_df

    def print_allocation_history(self, strategy_name: str):
        weights_df = self.all_results[strategy_name]["results"]["weights_history"]

        print(f"\n{'=' * 80}")
        print(f"{strategy_name} - PORTFOLIO ALLOCATIONS AT EACH REBALANCE")
        print(f"{'=' * 80}")
        print(f"Total Rebalances: {len(weights_df)}")
        print(f"\n{weights_df.round(4).to_string()}")

    def print_all_allocation_history(self):
        for strategy_name in self.all_results:
            self.print_allocation_history(strategy_name)

    def get_portfolio_returns(self) -> dict[str, pd.Series]:
        portfolio_returns = {}

        for name, data in self.all_results.items():
            portfolio_returns[name] = data["results"]["daily_returns"]

        return portfolio_returns

    def get_portfolio_equity(self) -> dict[str, pd.Series]:
        portfolio_equity = {}

        for name, data in self.all_results.items():
            portfolio_equity[name] = data["results"]["equity"]

        return portfolio_equity

    def export_metrics_csv(self, metrics_df: pd.DataFrame, output_path: str = "portfolio_metrics.csv"):
        metrics_df.to_csv(output_path)
        print(f"Metrics exported to: {output_path}")

    def export_allocations_csv(self, output_path: str = "portfolio_allocations.csv") -> pd.DataFrame:
        allocation_data = []

        for name, data in self.all_results.items():
            weights_df = data["results"]["weights_history"]

            for date, row in weights_df.iterrows():
                row_data = {"Strategy_Optimizer": name, "Rebalance_Date": date}
                row_data.update(row.to_dict())
                allocation_data.append(row_data)

        allocation_export = pd.DataFrame(allocation_data)
        allocation_export.to_csv(output_path, index=False)
        print(f"Allocations exported to: {output_path}")

        return allocation_export

    def get_performance_metrics(
        self, results: dict, risk_free_rate: float = 0.0, annual_trading_days: int = 252
    ) -> dict[str, float]:
        """
        Calculate comprehensive performance metrics using analytics.py.

        Uses compute_all_metrics() for consistency with _stats.py formulas.
        Returns only the numeric metrics (not time/equity values for brevity).

        Args:
            results: Backtest results dict with keys: equity, daily_returns, etc.
            risk_free_rate: Risk-free rate for Sharpe/Sortino (default 0%)
            annual_trading_days: Trading days per year (default 252)

        Returns:
            dict with keys: annualized_return, annualized_volatility, sharpe_ratio,
                            sortino_ratio, calmar_ratio, max_drawdown, etc.

        Assumptions:
            - results['equity'] has DatetimeIndex and is sorted
            - No NaN at start of equity curve

        """
        equity = results["equity"]
        annualization_factor = int(round(float(results.get("annualization_factor", annual_trading_days))))

        # Compute all metrics via analytics
        metrics_series = compute_all_metrics(
            equity=equity,
            risk_free_rate=risk_free_rate,
            annual_trading_days=annualization_factor,
            pnl=None,  # No trade PnL available in portfolio backtest
            returns_pct=None,
        )

        # Extract numeric metrics and map to output dict
        metrics_dict = {
            "annualized_return": metrics_series["Return (Ann.) [%]"] / 100,
            "annualized_volatility": metrics_series["Volatility (Ann.) [%]"] / 100,
            "sharpe_ratio": metrics_series["Sharpe Ratio"],
            "sortino_ratio": metrics_series["Sortino Ratio"],
            "calmar_ratio": metrics_series["Calmar Ratio"],
            "max_drawdown": metrics_series["Max. Drawdown [%]"] / 100,
            "ulcer_index": metrics_series["Ulcer Index [%]"] / 100,
            "total_return": metrics_series["Return [%]"] / 100,
            "cagr": metrics_series["CAGR [%]"] / 100 if not np.isnan(metrics_series["CAGR [%]"]) else np.nan,
            "best_day": metrics_series["Best Day [%]"],
            "worst_day": metrics_series["Worst Day [%]"],
            "cvar_95": metrics_series["CVaR 95% [%]"] / 100,
            "avg_drawdown": metrics_series["Avg. Drawdown [%]"] / 100,
            "max_drawdown_duration": metrics_series["Max. Drawdown Duration"],
            "avg_drawdown_duration": metrics_series["Avg. Drawdown Duration"],
        }

        # Add leverage metrics if applicable
        if results.get("leverage_applied"):
            metrics_dict["leverage_ratio"] = results["leverage_ratio"]
            metrics_dict["realized_vol_pre_leverage"] = results["realized_vol_pre_leverage"]

        return metrics_dict
