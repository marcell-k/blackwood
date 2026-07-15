"""
Monte Carlo Simulation Module.
Unified Monte Carlo simulator supporting IID, Stationary Bootstrap, Semi-parametric (POT-GPD),
and Parametric (GH/Student-t) methods with vectorized batch processing.

Methods:
    - IID: Independent shuffling of returns
    - Stationary Bootstrap: Politis-Romano dependence-preserving resampling
    - Semi-parametric: Stationary Bootstrap + POT-GPD tail modeling
    - Parametric: Generalized Hyperbolic or Student-t distribution fitting

"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import Figure
from scipy.stats import genpareto, kurtosis, norm, skew

from blackwood.config import CASH, RANDOM_STATE
from blackwood.visualization.style import DEFAULT_STYLE


class MonteCarloSimulator:
    """Unified Monte Carlo simulation for trading strategy analysis."""

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        n_simulations: int = 2000,
        initial_cash: float | None = None,
    ) -> None:
        """Initialize Monte Carlo simulator."""
        if isinstance(data, pd.DataFrame):
            self.trades = data.copy()
            returns = data["ReturnPct"].astype(np.float64).to_numpy()
        else:
            self.trades = None
            returns = np.asarray(data, dtype=np.float64)
            if returns.ndim != 1:
                raise ValueError("ndarray input must be 1D array of returns")

        self.returns = returns
        self.n_returns = len(self.returns)
        self.n_simulations = n_simulations
        self.initial_cash = float(initial_cash if initial_cash is not None else CASH)
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.total_years = self._infer_years()
        self._results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}

    @staticmethod
    def equity_from_returns(returns: np.ndarray, initial_cash: float) -> np.ndarray:
        r = np.asarray(returns, dtype=np.float64)
        r_clipped = np.clip(r, -0.999999, None)
        return float(initial_cash) * np.cumprod(1.0 + r_clipped, axis=-1)

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> np.ndarray:
        eq = np.asarray(equity_curve, dtype=np.float64)
        peak = np.maximum.accumulate(eq, axis=-1)
        drawdown = (eq - peak) / peak
        return np.min(drawdown, axis=-1)

    def calmar_ratio(self, equity_curve: np.ndarray) -> np.ndarray:
        eq = np.asarray(equity_curve, dtype=np.float64)
        total_years = self.total_years

        if eq.ndim > 1:
            start_equity = eq[..., 0]
            end_equity = eq[..., -1]
        else:
            start_equity = eq[0]
            end_equity = eq[-1]

        valid_mask = (start_equity > 0) & (end_equity > 0)
        cagr = np.where(
            valid_mask,
            (end_equity / start_equity) ** (1.0 / total_years) - 1.0,
            np.nan,
        )
        mdd = np.abs(self.max_drawdown(eq))
        valid_mdd = (mdd > 0) & np.isfinite(mdd) & valid_mask
        return np.where(valid_mdd, cagr / mdd, np.nan)

    def _infer_years(self) -> float:
        if self.trades is not None and "ExitTime" in self.trades.columns:
            exit_times = pd.to_datetime(self.trades["ExitTime"])
            if len(exit_times) >= 2:
                first_time = exit_times.iloc[0]
                last_time = exit_times.iloc[-1]
                if pd.notna(first_time) and pd.notna(last_time):
                    days = (last_time - first_time).days
                    return max(days / 365.25, 1e-9)
        return max(self.n_returns / 252.0, 1e-9)

    @staticmethod
    def generate_stationary_bootstrap_indices(
        rng: np.random.Generator,
        n_paths: int,
        length: int,
        avg_block: int = 10,
    ) -> np.ndarray:
        avg_block = max(1, int(avg_block))
        p = 1.0 / avg_block

        approx_blocks_needed = length / avg_block + 10 * (length / avg_block) ** 0.5 + 50
        max_blocks = max(10, int(approx_blocks_needed))

        block_lengths = rng.geometric(p, size=(n_paths, max_blocks))
        block_starts = rng.integers(0, length, size=(n_paths, max_blocks))

        indices = np.zeros((n_paths, length), dtype=np.int64)

        for path in range(n_paths):
            pos = 0
            b_idx = 0
            while pos < length and b_idx < max_blocks:
                blen = int(block_lengths[path, b_idx])
                if blen <= 0:
                    b_idx += 1
                    continue

                start = int(block_starts[path, b_idx])
                block = (start + np.arange(blen)) % length
                copy_len = min(blen, length - pos)

                indices[path, pos : pos + copy_len] = block[:copy_len]
                pos += copy_len
                b_idx += 1

            if pos < length:
                indices[path, pos:] = np.arange(pos, length)

        return indices

    def _compute_metrics(self, returns_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        equity_paths = self.equity_from_returns(returns_paths, self.initial_cash)
        max_drawdowns = self.max_drawdown(equity_paths)
        calmar_ratios = self.calmar_ratio(equity_paths)
        return equity_paths, max_drawdowns, calmar_ratios

    def simulate_iid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.tile(np.arange(self.n_returns), (self.n_simulations, 1))
        shuffled_indices = self.rng.permuted(indices, axis=1)
        shuffled_returns = self.returns[shuffled_indices]

        equity_paths, max_drawdowns, calmar_ratios = self._compute_metrics(shuffled_returns)
        self._results["IID"] = (
            equity_paths,
            max_drawdowns,
            calmar_ratios,
            {"method": "iid", "n_simulations": self.n_simulations},
        )
        return equity_paths, max_drawdowns, calmar_ratios

    def simulate_stationary_bootstrap(
        self,
        avg_block: int = 10,
        tail_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self.generate_stationary_bootstrap_indices(self.rng, self.n_simulations, self.n_returns, avg_block)
        bootstrap_returns = self.returns[indices] * tail_scale

        equity_paths, max_drawdowns, calmar_ratios = self._compute_metrics(bootstrap_returns)
        self._results["STATIONARY"] = (
            equity_paths,
            max_drawdowns,
            calmar_ratios,
            {
                "method": "stationary_bootstrap",
                "avg_block": avg_block,
                "tail_scale": tail_scale,
            },
        )
        return equity_paths, max_drawdowns, calmar_ratios

    def simulate_semiparametric(
        self,
        avg_block: int = 10,
        q_hi: float = 0.95,
        q_lo: float = 0.05,
        tail_scale: float = 1.0,
        min_exceedances: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_hi = np.quantile(self.returns, q_hi)
        u_lo = np.quantile(self.returns, q_lo)

        pos_exceed = self.returns[self.returns > u_hi] - u_hi
        neg_exceed = u_lo - self.returns[self.returns < u_lo]

        pos_tail = None
        if pos_exceed.size >= min_exceedances:
            c, loc, scale = genpareto.fit(pos_exceed, floc=0.0)
            pos_tail = {"c": float(c), "loc": float(loc), "scale": float(scale)}

        neg_tail = None
        if neg_exceed.size >= min_exceedances:
            c, loc, scale = genpareto.fit(neg_exceed, floc=0.0)
            neg_tail = {"c": float(c), "loc": float(loc), "scale": float(scale)}

        indices = self.generate_stationary_bootstrap_indices(self.rng, self.n_simulations, self.n_returns, avg_block)
        bootstrap_returns = self.returns[indices].copy()

        pos_mask = bootstrap_returns > u_hi
        if pos_tail is not None and np.any(pos_mask):
            draws = genpareto.rvs(
                pos_tail["c"],
                pos_tail["loc"],
                pos_tail["scale"],
                size=pos_mask.sum(),
                random_state=self.rng,
            )
            bootstrap_returns[pos_mask] = (u_hi + draws) * tail_scale

        neg_mask = bootstrap_returns < u_lo
        if neg_tail is not None and np.any(neg_mask):
            draws = genpareto.rvs(
                neg_tail["c"],
                neg_tail["loc"],
                neg_tail["scale"],
                size=neg_mask.sum(),
                random_state=self.rng,
            )
            bootstrap_returns[neg_mask] = (u_lo - draws) * tail_scale

        equity_paths, max_drawdowns, calmar_ratios = self._compute_metrics(bootstrap_returns)
        self._results["SEMIPARAMETRIC"] = (
            equity_paths,
            max_drawdowns,
            calmar_ratios,
            {
                "method": "semiparametric",
                "avg_block": avg_block,
                "q_hi": q_hi,
                "q_lo": q_lo,
                "tail_scale": tail_scale,
                "pos_tail_fitted": pos_tail is not None,
                "neg_tail_fitted": neg_tail is not None,
            },
        )
        return equity_paths, max_drawdowns, calmar_ratios

    @staticmethod
    def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
        r = np.asarray(returns, dtype=np.float64)
        n = len(r)

        mu = r.mean()
        sig = r.std(ddof=1)
        sr = mu / sig if sig > 0 else 0.0

        g3 = skew(r, bias=False)
        g4 = kurtosis(r, fisher=False, bias=False)

        denom = max(n - 1, 1)
        se = np.sqrt((1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2) / denom)

        z = (sr - sr_benchmark) / se if se > 0 else 0.0
        return float(norm.cdf(z))

    @staticmethod
    def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int = 1, sr_benchmark: float = 0.0) -> float:
        r = np.asarray(returns, dtype=np.float64)
        n = len(r)

        mu = r.mean()
        sig = r.std(ddof=1)
        sr = mu / sig if sig > 0 else 0.0

        m_eff = max(1, int(n_trials))
        sr0 = norm.ppf(1.0 - 1.0 / m_eff) / np.sqrt(max(n - 1, 1))
        sr_threshold = max(sr_benchmark, sr0)

        g3 = skew(r, bias=False)
        g4 = kurtosis(r, fisher=False, bias=False)

        denom = max(n - 1, 1)
        se = np.sqrt((1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2) / denom)

        z = (sr - sr_threshold) / se if se > 0 else 0.0
        return float(norm.cdf(z))

    def spa_reality_check(
        self,
        performance_matrix: np.ndarray,
        avg_block: int = 10,
        n_bootstrap: int = 2000,
        use_spa: bool = True,
    ) -> dict[str, float]:
        X = np.asarray(performance_matrix, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("performance_matrix must be 2-D (T, K)")

        T, _ = X.shape
        d_bar = X.mean(axis=0)
        s_hat = X.std(axis=0, ddof=1)

        sqrt_T = np.sqrt(T)
        safe_s_hat = np.where(s_hat > 0, s_hat, np.inf)

        stat_obs_rc = sqrt_T * np.max(d_bar)

        if use_spa:
            student_obs = np.maximum(d_bar / safe_s_hat, 0.0)
            stat_obs_spa = sqrt_T * np.max(student_obs)
        else:
            stat_obs_spa = 0.0

        indices = self.generate_stationary_bootstrap_indices(self.rng, n_bootstrap, T, avg_block)
        X_boot = X[indices]  # (n_bootstrap, T, K)

        dbar_boot = np.mean(X_boot, axis=1)
        stats_rc = sqrt_T * np.max(dbar_boot, axis=1)

        if use_spa:
            s_boot = np.std(X_boot, axis=1, ddof=1)
            safe_s_boot = np.where(s_boot > 0, s_boot, np.inf)
            student = np.maximum(dbar_boot / safe_s_boot, 0.0)
            stats_spa = sqrt_T * np.max(student, axis=1)
            pval_spa = float(np.mean(stats_spa >= stat_obs_spa))
        else:
            pval_spa = np.nan

        pval_rc = float(np.mean(stats_rc >= stat_obs_rc))

        return {
            "pval_rc": pval_rc,
            "pval_spa": pval_spa,
            "reject_null": (pval_spa if use_spa else pval_rc) < 0.05,
        }

    def calculate_summary_stats(self, method_key: str) -> dict[str, Any]:
        equity_paths, max_drawdowns, calmar_ratios, metadata = self._results[method_key]

        terminal_values = equity_paths[:, -1]
        final_returns = (terminal_values / self.initial_cash) - 1.0

        mean_return = float(final_returns.mean())
        median_return = float(np.median(final_returns))
        std_return = float(final_returns.std())

        sharpe_ratio = mean_return / std_return if std_return > 0 else np.nan

        var_95 = np.percentile(final_returns, 5)
        tail_mask = final_returns <= var_95
        cvar_95 = np.mean(final_returns[tail_mask]) if np.any(tail_mask) else np.nan

        return {
            "Method": metadata.get("method", method_key),
            "Mean Return (%)": mean_return * 100.0,
            "Median Return (%)": median_return * 100.0,
            "Std Dev (%)": std_return * 100.0,
            "Sharpe Ratio": float(sharpe_ratio),
            "Win Rate (%)": float(np.mean(final_returns > 0) * 100.0),
            "Best Case (%)": float(np.max(final_returns) * 100.0),
            "Worst Case (%)": float(np.min(final_returns) * 100.0),
            "Max DD (Median)": float(np.median(max_drawdowns)),
            "Max DD (Worst)": float(np.min(max_drawdowns)),
            "Calmar (Median)": float(np.nanmedian(calmar_ratios)),
            "VaR 95% (%)": float(var_95 * 100.0),
            "CVaR 95% (%)": float(cvar_95 * 100.0) if not np.isnan(cvar_95) else np.nan,
            "Metadata": metadata,
        }

    def print_summary(self, method_keys: list[str] | None = None) -> None:
        keys_to_print = method_keys if method_keys is not None else list(self._results.keys())

        print("\n" + "=" * 80)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 80)

        for key in keys_to_print:
            if key not in self._results:
                print(f"\nWarning: No results for '{key}'")
                continue

            stats = self.calculate_summary_stats(key)

            print(f"\n{stats['Method'].upper()} Results:")
            print("-" * 40)
            print(f"Mean Return: {stats['Mean Return (%)']:>8.2f}%")
            print(f"Median Return: {stats['Median Return (%)']:>8.2f}%")
            print(f"Std Dev: {stats['Std Dev (%)']:>8.2f}%")
            print(f"Sharpe Ratio: {stats['Sharpe Ratio']:>8.2f}")
            print(f"Win Rate: {stats['Win Rate (%)']:>8.1f}%")
            print(f"\nBest Case: {stats['Best Case (%)']:>8.2f}%")
            print(f"Worst Case: {stats['Worst Case (%)']:>8.2f}%")
            print(f"\nMax DD (Median): {stats['Max DD (Median)']:>8.2%}")
            print(f"Max DD (Worst): {stats['Max DD (Worst)']:>8.2%}")
            print(f"Calmar (Median): {stats['Calmar (Median)']:>8.2f}")
            print(f"\nVaR 95%: {stats['VaR 95% (%)']:>8.2f}%")
            print(f"CVaR 95%: {stats['CVaR 95% (%)']:>8.2f}%")

    def plot_equity_percentiles(
        self,
        method_key: str,
        observed_equity: np.ndarray | None = None,
    ) -> Figure:
        equity_paths, _, _, metadata = self._results[method_key]

        pct_levels = [5, 25, 50, 75, 95]
        pct = np.percentile(equity_paths, pct_levels, axis=0)
        x = np.arange(equity_paths.shape[1])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(
            x,
            pct[1],
            pct[3],
            color=DEFAULT_STYLE.muted,
            alpha=0.2,
            label="IQR (P25-P75)",
            linewidth=0,
        )
        ax.plot(
            x,
            pct[2],
            color=DEFAULT_STYLE.font_color,
            linewidth=2.5,
            label="P50 (Median)",
            zorder=3,
        )
        ax.plot(
            x,
            pct[4],
            color=DEFAULT_STYLE.accent3,
            linewidth=1.5,
            linestyle="--",
            label="P95 (Best 5%)",
            zorder=2,
        )
        ax.plot(
            x,
            pct[0],
            color=DEFAULT_STYLE.accent4,
            linewidth=1.5,
            linestyle="--",
            label="P5 (Worst 5%)",
            zorder=2,
        )

        if observed_equity is not None:
            ax.plot(
                np.arange(len(observed_equity)),
                observed_equity,
                color=DEFAULT_STYLE.accent1,
                linewidth=2,
                linestyle=":",
                label="Observed",
                zorder=4,
            )

        ax.set_xlabel("Trading Period", fontsize=DEFAULT_STYLE.font_size)
        ax.set_ylabel("Portfolio Value ($)", fontsize=DEFAULT_STYLE.font_size)
        ax.set_title(
            f"Monte Carlo Equity Percentiles - {metadata.get('method', method_key).upper()}",
            fontsize=DEFAULT_STYLE.title_size,
            pad=15,
        )
        ax.legend(loc="upper left", framealpha=0.9, fontsize=DEFAULT_STYLE.font_size)

        DEFAULT_STYLE.apply_mpl(fig, ax)
        fig.tight_layout()
        return fig

    def plot_drawdown_distribution(
        self,
        method_key: str,
        observed_mdd: float | None = None,
    ) -> plt.Figure:
        _, max_drawdowns, _, metadata = self._results[method_key]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(
            max_drawdowns,
            bins=60,
            color=DEFAULT_STYLE.accent1,
            alpha=0.75,
            edgecolor="none",
            label="Simulated MDD",
        )

        median_mdd = np.median(max_drawdowns)
        ax.axvline(
            median_mdd,
            color=DEFAULT_STYLE.accent6,
            linewidth=2,
            linestyle="--",
            label=f"Median: {median_mdd:.1%}",
        )

        if observed_mdd is not None:
            percentile = np.mean(max_drawdowns <= observed_mdd) * 100
            ax.axvline(
                observed_mdd,
                color=DEFAULT_STYLE.accent4,
                linewidth=3,
                linestyle="-",
                label=f"Observed: {observed_mdd:.1%} (P{percentile:.0f})",
            )

        ax.set_xlabel("Maximum Drawdown", fontsize=DEFAULT_STYLE.font_size)
        ax.set_ylabel("Frequency", fontsize=DEFAULT_STYLE.font_size)
        ax.set_title(
            f"Maximum Drawdown Distribution - {metadata.get('method', method_key).upper()}",
            fontsize=DEFAULT_STYLE.title_size,
            pad=15,
        )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper left", framealpha=0.9, fontsize=DEFAULT_STYLE.font_size)

        DEFAULT_STYLE.apply_mpl(fig, ax)
        fig.tight_layout()
        return fig

    def plot_returns_distribution(
        self,
        method_key: str,
        observed_return: float | None = None,
    ) -> plt.Figure:
        equity_paths, _, _, metadata = self._results[method_key]
        final_returns = (equity_paths[:, -1] / self.initial_cash) - 1.0

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(
            final_returns,
            bins=60,
            color=DEFAULT_STYLE.accent1,
            alpha=0.75,
            edgecolor="none",
            label="Simulated Returns",
        )

        var_95 = np.percentile(final_returns, 5)
        cvar_95 = np.mean(final_returns[final_returns <= var_95])

        ax.axvline(
            var_95,
            color=DEFAULT_STYLE.accent6,
            linewidth=2,
            linestyle="--",
            label=f"VaR 95%: {var_95:.1%}",
        )
        ax.axvline(
            cvar_95,
            color=DEFAULT_STYLE.accent4,
            linewidth=2,
            linestyle=":",
            label=f"CVaR 95%: {cvar_95:.1%}",
        )

        if observed_return is not None:
            percentile = np.mean(final_returns <= observed_return) * 100
            ax.axvline(
                observed_return,
                color=DEFAULT_STYLE.accent3,
                linewidth=3,
                linestyle="-",
                label=f"Observed: {observed_return:.1%} (P{percentile:.0f})",
            )

        ax.set_xlabel("Final Return", fontsize=DEFAULT_STYLE.font_size)
        ax.set_ylabel("Frequency", fontsize=DEFAULT_STYLE.font_size)
        ax.set_title(
            f"Final Returns Distribution - {metadata.get('method', method_key).upper()}",
            fontsize=DEFAULT_STYLE.title_size,
            pad=15,
        )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right", framealpha=0.9, fontsize=DEFAULT_STYLE.font_size)

        DEFAULT_STYLE.apply_mpl(fig, ax)
        fig.tight_layout()
        return fig
