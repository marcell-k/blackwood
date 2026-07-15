import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def merge_equity_curves(
    tests: Iterable[Mapping[str, Any]],
    curve_key: str = "_equity_curve",
    keep_policy: str = "first",
    add_test_id: bool = True,
    sort_result_index: bool = True,
) -> pd.DataFrame:
    if keep_policy not in ("first", "last", False):
        raise ValueError("keep_policy must be one of {'first', 'last', False}")

    frames: list[pd.DataFrame] = []

    for i, test in enumerate(tests):
        curve = test.get(curve_key, None)
        if isinstance(curve, pd.DataFrame) and not curve.empty:
            # Shallow copy to avoid mutating the source; preserve dtypes and index
            dfc = curve.copy(deep=False)

            # Optional source identifier for downstream analysis and debugging
            if add_test_id and "test_id" not in dfc.columns:
                dfc = dfc.assign(test_id=i)

            # Deduplicate within each curve before stacking
            if dfc.index.duplicated().any():
                dfc = dfc[~dfc.index.duplicated(keep=keep_policy)]
            frames.append(dfc)
    if not frames:
        # Empty result if no valid curves found
        return pd.DataFrame()

    # Concatenate while preserving original time index labels
    merged = pd.concat(frames, axis=0, sort=True)

    # Global duplicate sweep after stacking
    if merged.index.duplicated().any():
        merged = merged[~merged.index.duplicated(keep=keep_policy)]

    if sort_result_index:
        merged = merged.sort_index(kind="stable")

    # Standardize index name for downstream consumers
    if merged.index.name is None:
        merged.index.name = "timestamp"

    return merged


def calculate_risk_reward_ratio(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    is_long = df["Size"] > 0
    size_abs = df["Size"].abs()

    # PotentialRisk: will be NaN if OpenSL is None/NaN (NaN propagates automatically)
    df["PotentialRisk"] = np.where(
        is_long, (df["EntryPrice"] - df["OpenSL"]) * size_abs, (df["OpenSL"] - df["EntryPrice"]) * size_abs
    )

    df["RealizedReward"] = np.where(
        is_long,
        size_abs * (df["ExitPrice"] - df["EntryPrice"]) - df["Commission"],
        size_abs * (df["EntryPrice"] - df["ExitPrice"]) - df["Commission"],
    )

    # Division by NaN or 0 returns NaN
    df["RiskRewardRatio"] = df["RealizedReward"] / df["PotentialRisk"].replace(0, np.nan)

    # Convert to numeric first to handle any object types, then replace inf with NaN
    df["RiskRewardRatio"] = pd.to_numeric(df["RiskRewardRatio"], errors="coerce")
    df["RiskRewardRatio"] = df["RiskRewardRatio"].replace([np.inf, -np.inf], np.nan)

    return df


def standard_metrics(stats, trades):
    win_rate = round(100 * (trades.PnL > 0).sum() / len(trades), 2)
    pf_pnl = trades[trades.PnL > 0]["PnL"].sum() / -trades[trades.PnL < 0]["PnL"].sum()
    pf_return = trades[trades.PnL > 0]["ReturnPct"].sum() / -trades[trades.PnL < 0]["ReturnPct"].sum()
    tpy = round(len(trades) / (stats["Duration"].days / 365.25), 2)
    print(f"Winrate: {win_rate}% | Profit Factor: {pf_pnl:.2f}, {pf_return:.2f} | Trades per year: {tpy}")
    return win_rate, pf_pnl, pf_return, tpy


class _Stats(pd.Series):
    """
    Custom Series subclass for metric display with controlled formatting.
    Matches _stats.py behavior:
    - 5 decimal precision for readability
    - All rows displayed (no truncation)
    - Compact column width (max 20 chars)
    """

    def __repr__(self):
        with pd.option_context(
            "display.max_colwidth",
            20,
            "display.max_rows",
            len(self),
            "display.precision",
            5,
            "display.width",
            100,
        ):
            return super().__repr__()

    def __repr_html__(self):
        """Override for Jupyter notebook display."""
        with pd.option_context(
            "display.max_colwidth",
            20,
            "display.max_rows",
            len(self),
            "display.precision",
            5,
            "display.width",
            100,
        ):
            return super().__repr_html__()


def geometric_mean(returns: pd.Series) -> float:
    """Geometric mean of returns (expects decimal returns, e.g., 0.05 for 5%)."""
    # Ensure numeric dtype to prevent object-dtype ufunc errors
    if isinstance(returns, pd.Series):
        returns = returns.astype(np.float64, errors="ignore")
    else:
        returns = pd.Series(returns, dtype=np.float64)

    returns = returns.fillna(0) + 1

    if np.any(returns <= 0):
        return 0.0

    return np.exp(np.log(returns).sum() / len(returns)) - 1


def calculate_returns(equity: pd.Series) -> pd.Series:
    """Convert equity curve to log returns."""
    return np.log(equity / equity.shift(1)).dropna()


def calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series."""
    cummax = equity.cummax()
    return 1 - (equity / cummax)


def compute_drawdown_duration_peaks(dd: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Decompose drawdown series into individual episodes."""
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame("iloc").assign(prev=iloc.shift())
    df = df[df["iloc"] > df["prev"] + 1].astype(np.int64)

    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df["duration"] = df["iloc"].map(dd.index.__getitem__) - df["prev"].map(dd.index.__getitem__)
    df["peak_dd"] = df.apply(lambda row: dd.iloc[row["prev"] : row["iloc"] + 1].max(), axis=1)
    df = df.reindex(dd.index)

    return df["duration"], df["peak_dd"]


def calculate_annualized_metrics(equity: pd.Series, annual_trading_days: int = 252) -> dict:
    """Compute annualized return and volatility (matches _stats.py)."""
    equity_resampled = equity.resample("D").last().dropna()
    day_returns = equity_resampled.pct_change().dropna()

    gmean_return = geometric_mean(day_returns)
    annual_return = (1 + gmean_return) ** annual_trading_days - 1

    ddof = int(bool(day_returns.shape))
    annual_vol = np.sqrt(
        (day_returns.var(ddof=ddof) + (1 + gmean_return) ** 2) ** annual_trading_days
        - (1 + gmean_return) ** (2 * annual_trading_days)
    )

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "gmean_return": gmean_return,
        "day_returns": day_returns,
        "annual_trading_days": annual_trading_days,
    }


def calculate_sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0, annual_trading_days: int = 252) -> float:
    """Sharpe Ratio = (annual_return - rf) / annual_vol."""
    metrics = calculate_annualized_metrics(equity, annual_trading_days)
    annual_vol = metrics["annual_vol"]

    if annual_vol == 0:
        return np.nan

    return (metrics["annual_return"] - risk_free_rate) / annual_vol


def calculate_rolling_sharpe_ratio(
    equity: pd.Series, window: int = 252, risk_free_rate: float = 0.0, annual_trading_days: int = 252
) -> pd.Series:
    """Rolling Sharpe Ratio over specified window."""
    equity_resampled = equity.resample("D").last().dropna()
    day_returns = equity_resampled.pct_change().dropna()

    def rolling_sharpe_fn(returns_arr):
        # Inline geometric mean: exp(mean(log(1 + r))) - 1
        returns_plus_1 = returns_arr + 1
        if np.any(returns_plus_1 <= 0):
            gmean_return = 0.0
        else:
            gmean_return = np.exp(np.log(returns_plus_1).sum() / len(returns_arr)) - 1

        annual_return = (1 + gmean_return) ** annual_trading_days - 1

        ddof = 1 if len(returns_arr) > 1 else 0
        var = returns_arr.var(ddof=ddof)
        annual_vol = np.sqrt(
            (var + (1 + gmean_return) ** 2) ** annual_trading_days - (1 + gmean_return) ** (2 * annual_trading_days)
        )

        if annual_vol == 0:
            return np.nan
        return (annual_return - risk_free_rate) / annual_vol

    rolling_sharpe = day_returns.rolling(window=window).apply(rolling_sharpe_fn, raw=True)
    return rolling_sharpe.dropna()


def calculate_sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0, annual_trading_days: int = 252) -> float:
    """Sortino Ratio = (annual_return - rf) / downside_vol."""
    metrics = calculate_annualized_metrics(equity, annual_trading_days)
    day_returns = metrics["day_returns"]

    downside_returns = day_returns.clip(-np.inf, 0)
    downside_vol = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(metrics["annual_trading_days"])

    if downside_vol == 0:
        return np.nan

    return (metrics["annual_return"] - risk_free_rate) / downside_vol


def calculate_calmar_ratio(equity: pd.Series, annual_trading_days: int = 252) -> float:
    """Calmar Ratio = annual_return / max_drawdown."""
    metrics = calculate_annualized_metrics(equity, annual_trading_days)
    dd = calculate_drawdown_series(equity)
    max_dd = dd.max()

    if max_dd == 0 or np.isnan(max_dd):
        return np.nan

    return metrics["annual_return"] / max_dd


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from peak."""
    dd = calculate_drawdown_series(equity)
    return dd.max()


def calculate_ulcer_index(equity: pd.Series) -> float:
    """Ulcer Index = sqrt(mean(DD^2)) * 100."""
    dd = calculate_drawdown_series(equity)
    return np.sqrt(np.mean(dd**2)) * 100


def calculate_best_worst_day(equity: pd.Series) -> tuple[float, float]:
    """Best and worst single-day returns (in %)."""
    returns = calculate_returns(equity)
    best_day = returns.max() * 100
    worst_day = returns.min() * 100
    return best_day, worst_day


def calculate_cvar(equity: pd.Series, confidence_level: float = 0.95) -> float:
    """CVaR = mean of returns below VaR threshold."""
    returns = calculate_returns(equity)
    alpha = 1 - confidence_level
    var = returns.quantile(alpha)
    cvar = returns[returns <= var].mean()
    return cvar


def compute_psr(day_returns: pd.Series, risk_free_rate: float, benchmark_sr: float = 1.0) -> float:
    skewness = day_returns.skew()
    kurtosis = day_returns.kurt() + 3
    day_returns_mean = day_returns.mean()
    day_returns_std = day_returns.std(ddof=1)
    observed_sr = (day_returns_mean - risk_free_rate) / (day_returns_std or np.nan)
    n_samples = len(day_returns.dropna())

    variance_adjustment = 1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2
    sr_variance = variance_adjustment / (n_samples - 1)
    if sr_variance <= 0:
        return np.nan

    sr_std = np.sqrt(sr_variance)
    test_statistic = (observed_sr - benchmark_sr) / sr_std
    psr = norm.cdf(test_statistic)
    return psr


def compute_min_trl(
    day_returns: pd.Series, risk_free_rate: float, alpha: float = 0.05, benchmark_sr: float = 1.0
) -> float:
    skewness = day_returns.skew()
    kurtosis = day_returns.kurt() + 3
    day_returns_mean = day_returns.mean()
    day_returns_std = day_returns.std(ddof=1)
    observed_sr = (day_returns_mean - risk_free_rate) / (day_returns_std or np.nan)

    if observed_sr <= benchmark_sr:
        return np.inf
    variance_adjustment = 1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2

    if variance_adjustment < 0:
        return np.nan

    z_critical = norm.ppf(1 - alpha)
    min_trl = 1 + variance_adjustment * (z_critical / (observed_sr - benchmark_sr)) ** 2

    if not np.isfinite(min_trl) or min_trl <= 0:
        return np.nan
    return math.ceil(min_trl)


def calculate_trade_metrics(pnl: np.ndarray, returns_pct: np.ndarray) -> dict:
    """Compute trade-level metrics."""
    n_trades = len(pnl)

    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_trade_pct": np.nan,
            "avg_win_pct": np.nan,
            "avg_loss_pct": np.nan,
            "profit_factor": np.nan,
        }

    winning = pnl > 0
    losing = pnl < 0
    n_wins = winning.sum()

    win_rate = n_wins / n_trades

    winning_returns = returns_pct[winning]
    losing_returns = returns_pct[losing]

    avg_trade_pct = geometric_mean(pd.Series(returns_pct)) * 100
    avg_win_pct = geometric_mean(pd.Series(winning_returns)) * 100 if len(winning_returns) > 0 else np.nan
    avg_loss_pct = geometric_mean(pd.Series(losing_returns)) * 100 if len(losing_returns) > 0 else np.nan

    sum_wins = returns_pct[winning].sum() if len(winning_returns) > 0 else 0
    sum_losses = returns_pct[losing].sum() if len(losing_returns) > 0 else 0
    profit_factor = sum_wins / abs(sum_losses) if sum_losses != 0 else np.nan

    return {
        "n_trades": n_trades,
        "win_rate": win_rate * 100,
        "avg_trade_pct": avg_trade_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
    }


def compute_all_metrics(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    annual_trading_days: int = 252,
    pnl: np.ndarray | None = None,
    returns_pct: np.ndarray | None = None,
    benchmark_sharpe: float = 2.0,  # Annualized benchmark for PSR/MinTRL
) -> _Stats:
    """
    Compute comprehensive metrics (matches _stats.py output).

    Returns _Stats object with custom display formatting (5 decimal precision).

    Args:
        equity: Equity curve (pd.Series with DatetimeIndex)
        risk_free_rate: Risk-free rate (default 0%)
        annual_trading_days: Trading days per year (default 252)
        pnl: Trade P&L array (optional)
        returns_pct: Trade returns array (optional)

    Returns:
        _Stats object (custom pd.Series with formatting)

    """
    s = _Stats(dtype=object)

    start_ts = equity.index[0]
    end_ts = equity.index[-1]
    s.loc["Start"] = start_ts.strftime("%Y-%m-%d")
    s.loc["End"] = end_ts.strftime("%Y-%m-%d")
    duration = end_ts - start_ts
    s.loc["Duration"] = duration.days

    s.loc["Equity Final [$]"] = equity.iloc[-1]
    s.loc["Equity Peak [$]"] = equity.max()

    total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
    s.loc["Return [%]"] = total_return * 100

    ann_metrics_geom = calculate_annualized_metrics(equity, annual_trading_days)
    s.loc["Return (Ann.) [%]"] = ann_metrics_geom["annual_return"] * 100
    s.loc["Volatility (Ann.) [%]"] = ann_metrics_geom["annual_vol"] * 100

    # --- Arithmetic Annualized Metrics ---
    day_returns_simple = equity.resample("D").last().dropna().pct_change().dropna()
    ann_return_arith = day_returns_simple.mean() * annual_trading_days * 100
    ann_vol_arith = day_returns_simple.std(ddof=1) * np.sqrt(annual_trading_days) * 100
    s.loc["Return (Ann.) [Arith] [%]"] = ann_return_arith
    s.loc["Volatility (Ann.) [Arith] [%]"] = ann_vol_arith

    time_in_years = (s.loc["Duration"]) / annual_trading_days
    if time_in_years > 0:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / time_in_years) - 1
        s.loc["CAGR [%]"] = cagr * 100
    else:
        s.loc["CAGR [%]"] = np.nan

    s.loc["Sharpe Ratio"] = calculate_sharpe_ratio(equity, risk_free_rate, annual_trading_days)
    s.loc["Sharpe Ratio [Arith]"] = (ann_return_arith / ann_vol_arith) if ann_vol_arith > 0 else np.nan

    s.loc["Sortino Ratio"] = calculate_sortino_ratio(equity, risk_free_rate, annual_trading_days)
    s.loc["Calmar Ratio"] = calculate_calmar_ratio(equity, annual_trading_days)

    daily_benchmark_sr = benchmark_sharpe / np.sqrt(annual_trading_days)
    daily_rf = risk_free_rate / annual_trading_days

    s.loc[f"PSR (Benchmark={benchmark_sharpe:.1f})"] = compute_psr(
        day_returns_simple, daily_rf, benchmark_sr=daily_benchmark_sr
    )
    s.loc["Min TRL (Days)"] = compute_min_trl(day_returns_simple, daily_rf, benchmark_sr=daily_benchmark_sr)

    s.loc["Max. Drawdown [%]"] = calculate_max_drawdown(equity) * 100
    s.loc["Ulcer Index [%]"] = calculate_ulcer_index(equity)

    dd = calculate_drawdown_series(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(dd)
    s.loc["Avg. Drawdown [%]"] = -dd_peaks.mean() * 100
    s.loc["Max. Drawdown Duration"] = dd_dur.max()
    s.loc["Avg. Drawdown Duration"] = dd_dur.mean()

    best_day, worst_day = calculate_best_worst_day(equity)
    s.loc["Best Day [%]"] = best_day
    s.loc["Worst Day [%]"] = worst_day

    s.loc["CVaR 95% [%]"] = calculate_cvar(equity, confidence_level=0.95) * 100

    if pnl is not None and returns_pct is not None:
        trade_metrics = calculate_trade_metrics(pnl, returns_pct)
        s.loc["# Trades"] = trade_metrics["n_trades"]

        s.loc["Win Rate [%]"] = trade_metrics["win_rate"]
        s.loc["Avg. Trade [%]"] = trade_metrics["avg_trade_pct"]
        s.loc["Avg. Win [%]"] = trade_metrics["avg_win_pct"]
        s.loc["Avg. Loss [%]"] = trade_metrics["avg_loss_pct"]
        s.loc["Profit Factor"] = trade_metrics["profit_factor"]

    return s


def compute_trades_metrics(trades):
    returns = trades["ReturnPct"]
    pl = trades["PnL"]
    rrr = trades["RiskRewardRatio"]
    n_trades = trades.shape[0]
    s = _Stats(dtype=object)

    s.loc["Profit Factor"] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc["Win Rate [%]"] = win_rate * 100
    s.loc["Avg Win RRR"] = rrr[rrr > 0].mean()
    s.loc["Avg Loss RRR"] = rrr[rrr < 0].mean()
    s.loc["# Trades"] = n_trades
    return s


def compute_sr_max(n_trials: int, trials_variance: float | None = None) -> float:
    gamma_euler = 0.5772156649
    z_1 = norm.ppf(1.0 - 1.0 / n_trials)
    z_2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))

    expected_max_z = (1.0 - gamma_euler) * z_1 + gamma_euler * z_2

    sr_max = np.sqrt(trials_variance) * expected_max_z

    return sr_max


def compute_dsr(
    day_returns: pd.Series, risk_free_rate: float, n_trials: int, trials_variance: float, benchmark_sr: float = 0.0
) -> float:
    day_returns_clean = day_returns.dropna()
    n_samples = len(day_returns_clean)
    benchmark_sr /= 252

    if n_samples < 2:
        return np.nan

    day_returns_mean = day_returns_clean.mean()
    day_returns_std = day_returns_clean.std(ddof=1)

    if day_returns_std == 0:
        return np.nan

    observed_sr = (day_returns_mean - risk_free_rate) / day_returns_std

    sr_max = benchmark_sr + compute_sr_max(n_trials, trials_variance)

    skewness = day_returns_clean.skew()
    kurtosis = day_returns_clean.kurt() + 3.0
    variance_adjustment = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr**2)
    sr_variance = variance_adjustment / (n_samples - 1)

    if sr_variance <= 0:
        return np.nan

    sr_std = np.sqrt(sr_variance)
    test_statistic = (observed_sr - sr_max) / sr_std
    dsr = norm.cdf(test_statistic)
    return dsr
