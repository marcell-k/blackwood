from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blackwood.data.splitters import CPCVSplitter
from blackwood.visualization.style import DEFAULT_STYLE


def _stable_features(counter: Counter, min_count: float) -> set[str]:
    return {feature for feature, count in counter.items() if isinstance(feature, str) and count > min_count}


def existing_columns(df, ordered_columns):
    return [col for col in ordered_columns if col in set(df.columns)]


def _normalize_feature_exclusions(features_to_exclude: Sequence[str] | None = None) -> set[str]:
    """Normalize exclusion names to support both prefixed and renamed feature columns."""
    exclude_set: set[str] = set()
    for col in features_to_exclude or ():
        if not isinstance(col, str):
            continue
        exclude_set.add(col)
        if col.startswith("Entry_"):
            exclude_set.add(col.removeprefix("Entry_"))
    return exclude_set


STRATEGY_STATE_WINDOWS: tuple[int, ...] = (20, 50, 1000)
CPCVPaths = dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]]


def get_strategy_state_feature_cols(windows: tuple[int, ...] = STRATEGY_STATE_WINDOWS) -> list[str]:
    cols: list[str] = []
    for window in windows:
        cols.append(f"state_rrr_mean_{window}")
        cols.append(f"state_winrate_{window}")
    return cols


def compute_strategy_state_features(
    y_data: pd.DataFrame,
    *,
    windows: tuple[int, ...] = STRATEGY_STATE_WINDOWS,
    rrr_col: str = "RiskRewardRatio",
    label_col: str = "meta_label",
    fill_rrr: float = 0.0,
    fill_winrate: float = 0.5,
) -> pd.DataFrame:
    """
    Build causal strategy state features from prior trades only.
    Features:
    - rolling mean of RiskRewardRatio
    - rolling win-rate where win is (meta_label == 1)
    """
    rrr = pd.to_numeric(y_data[rrr_col], errors="coerce").shift(1)
    wins = (pd.to_numeric(y_data[label_col], errors="coerce") == 1).astype(float).shift(1)

    out = pd.DataFrame(index=y_data.index)
    for window in windows:
        min_periods = min(10, int(window))
        out[f"state_rrr_mean_{window}"] = rrr.rolling(window=window, min_periods=min_periods).mean()
        out[f"state_winrate_{window}"] = wins.rolling(window=window, min_periods=min_periods).mean()

    rrr_cols = [c for c in out.columns if c.startswith("state_rrr_mean_")]
    win_cols = [c for c in out.columns if c.startswith("state_winrate_")]
    out.loc[:, rrr_cols] = out.loc[:, rrr_cols].fillna(float(fill_rrr))
    out.loc[:, win_cols] = out.loc[:, win_cols].fillna(float(fill_winrate))
    return out


def compute_strategy_state_features_with_history(
    y_current: pd.DataFrame,
    *,
    y_history: pd.DataFrame | None = None,
    windows: tuple[int, ...] = STRATEGY_STATE_WINDOWS,
    rrr_col: str = "RiskRewardRatio",
    label_col: str = "meta_label",
    fill_rrr: float = 0.0,
    fill_winrate: float = 0.5,
) -> pd.DataFrame:
    """Compute state features for y_current using optional prior history."""
    combined = pd.concat([y_history, y_current], axis=0) if y_history is not None and not y_history.empty else y_current

    combined_feats = compute_strategy_state_features(
        combined,
        windows=windows,
        rrr_col=rrr_col,
        label_col=label_col,
        fill_rrr=fill_rrr,
        fill_winrate=fill_winrate,
    )
    return combined_feats.loc[y_current.index]


def append_strategy_state_features(
    X: pd.DataFrame,
    y_state: pd.DataFrame,
    *,
    y_history: pd.DataFrame | None = None,
    windows: tuple[int, ...] = STRATEGY_STATE_WINDOWS,
    rrr_col: str = "RiskRewardRatio",
    label_col: str = "meta_label",
    fill_rrr: float = -1.0,
    fill_winrate: float = -1.0,
    features_to_exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    y_current = y_state.loc[X.index]
    state = compute_strategy_state_features_with_history(
        y_current=y_current,
        y_history=y_history,
        windows=windows,
        rrr_col=rrr_col,
        label_col=label_col,
        fill_rrr=fill_rrr,
        fill_winrate=fill_winrate,
    )

    exclude_set = _normalize_feature_exclusions(features_to_exclude)
    state_cols = list(state.columns)
    out = X.copy()
    # Ensure excluded state columns are not kept from prior global precomputation.
    excluded_state_cols = [col for col in state_cols if col in exclude_set and col in out.columns]
    if excluded_state_cols:
        out = out.drop(columns=excluded_state_cols)

    for col in state_cols:
        if col in exclude_set:
            continue
        out[col] = state[col].astype(float)
    return out


def append_strategy_state_features_to_cpcv_paths(
    splitter: CPCVSplitter,
    X: pd.DataFrame,
    y_state: pd.DataFrame,
    paths: CPCVPaths,
    *,
    windows: tuple[int, ...] = STRATEGY_STATE_WINDOWS,
    entry_time_col: str = "EntryTime",
    fill_rrr: float = -1.0,
    fill_winrate: float = -1.0,
    features_to_exclude: Sequence[str] | None = None,
) -> CPCVPaths:
    """
    Recompute and append strategy-state features per CPCV path without look-ahead bias.

    For each path:
    - train state is computed from that path's train history only.
    - test state is computed from path train history + prior rows in path test order.
    """
    exclude_set = _normalize_feature_exclusions(features_to_exclude)
    out: CPCVPaths = {}
    for path_id in sorted(paths):
        path_folds = paths[path_id]

        full_train_df, full_test_df = splitter.get_train_test_for_path(paths, path_id)
        train_order = (
            full_train_df.sort_values(entry_time_col).index
            if entry_time_col in full_train_df.columns
            else full_train_df.index
        )
        test_order = (
            full_test_df.sort_values(entry_time_col).index
            if entry_time_col in full_test_df.columns
            else full_test_df.index
        )

        y_train_path = y_state.loc[train_order]
        y_test_path = y_state.loc[test_order]

        train_base = full_train_df
        test_base = full_test_df
        if exclude_set:
            train_drop_cols = [col for col in train_base.columns if col in exclude_set]
            test_drop_cols = [col for col in test_base.columns if col in exclude_set]
            if train_drop_cols:
                train_base = train_base.drop(columns=train_drop_cols)
            if test_drop_cols:
                test_base = test_base.drop(columns=test_drop_cols)

        train_with_state = append_strategy_state_features(
            train_base.copy(),
            y_train_path,
            windows=windows,
            fill_rrr=fill_rrr,
            fill_winrate=fill_winrate,
            features_to_exclude=exclude_set,
        )
        test_with_state = append_strategy_state_features(
            test_base.copy(),
            y_test_path,
            y_history=y_train_path,
            windows=windows,
            fill_rrr=fill_rrr,
            fill_winrate=fill_winrate,
            features_to_exclude=exclude_set,
        )

        updated_folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for _, fold_test_df in path_folds:
            fold_test_with_state = test_with_state.loc[fold_test_df.index].copy()
            updated_folds.append((train_with_state, fold_test_with_state))
        out[path_id] = updated_folds

    return out


def meta_labeling(trades):
    cond0 = trades.RiskRewardRatio < -0.99
    cond1 = 1 - abs(trades.SL / trades.ExitPrice) < 0.01
    cond2 = trades.RiskRewardRatio > 0.2
    trades["meta_label"] = np.select([cond0, cond1, cond2], [0, 1, 2], -1).astype(np.int8)
    return trades


def meta_labeling_binary(trades: pd.DataFrame) -> pd.DataFrame:
    """Binary meta-label: 1 if realized PnL > 0, else 0. -1 for missing."""
    trades = trades.copy()
    cond0 = trades.RealizedReward < 0
    cond1 = trades.RealizedReward >= 0
    trades["meta_label"] = np.select([cond0, cond1], [0, 1], -1).astype(np.int8)
    return trades


def aggregate_trades_keep_entry_features(
    trades: pd.DataFrame,
    *,
    group_col: str = "EntryBar",
    exit_col: str = "ExitBar",
    size_col: str = "Size",
    entry_price_col: str = "EntryPrice",
    exit_price_col: str = "ExitPrice",
    pnl_col: str = "PnL",
    commission_col: str = "Commission",
    keep_entry_prefix: str = "Entry_",
) -> pd.DataFrame:
    """
    Aggregate partial fills / scale-ins that share the same EntryBar into one row.
    - Weighted prices by abs(Size)
    - Sum PnL/Commission
    - ExitBar = last in group
    - Keep FIRST value for Entry_* features; first value for other untouched cols
    """
    df = trades.copy()

    def _wavg(group: pd.DataFrame, value_col: str) -> float:
        w = group[size_col].abs().to_numpy(dtype=float)
        v = group[value_col].to_numpy(dtype=float)
        wsum = w.sum()
        if wsum == 0:
            return float(v[0]) if len(v) else 0.0
        return float((w * v).sum() / wsum)

    def _wavg_col(group: pd.DataFrame, col: str) -> float:
        if col not in group.columns:
            return float("nan")
        w = group[size_col].abs().to_numpy(dtype=float)
        v = group[col].to_numpy(dtype=float)
        wsum = w.sum()
        if wsum == 0:
            return float(v[0]) if len(v) else 0.0
        return float((w * v).sum() / wsum)

    def _agg_one(group: pd.DataFrame) -> pd.Series:
        first = group.iloc[0]
        last = group.iloc[-1]

        total_size = float(group[size_col].sum())
        abs_total_size = float(group[size_col].abs().sum())

        w_entry = _wavg(group, entry_price_col)
        w_exit = _wavg(group, exit_price_col)
        total_pnl = float(group[pnl_col].sum())
        total_comm = float(group[commission_col].sum())

        position_value = abs_total_size * w_entry
        return_pct = total_pnl / position_value if position_value != 0 else 0.0

        out = dict(first)  # default: first row for untouched cols
        out[group_col] = group.name
        out[exit_col] = last[exit_col]
        out[size_col] = total_size
        out[entry_price_col] = w_entry
        out[exit_price_col] = w_exit
        out[pnl_col] = total_pnl
        out[commission_col] = total_comm
        out["ReturnPct"] = return_pct
        out["PotentialRisk"] = _wavg_col(group, "PotentialRisk")
        out["RealizedReward"] = _wavg_col(group, "RealizedReward")
        out["RiskRewardRatio"] = _wavg_col(group, "RiskRewardRatio")

        entry_cols = [c for c in group.columns if c.startswith(keep_entry_prefix)]
        for c in entry_cols:
            out[c] = first[c]

        return pd.Series(out)

    result = df.groupby(group_col, sort=False, as_index=False).apply(_agg_one)
    if isinstance(result.index, pd.MultiIndex):
        result = result.reset_index(drop=True)

    # Column order: original + any new calculated columns
    cols = list(trades.columns)
    for extra in ["ReturnPct", "PotentialRisk", "RealizedReward", "RiskRewardRatio"]:
        if extra not in cols and extra in result.columns:
            cols.append(extra)
    return result.reindex(columns=cols)


def split_by_entry_time_index(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    entry_col: str = "EntryTime",
    train_frac: float = 0.70,
    val_frac: float = 0.15,  # fraction of total (so test = 1 - train - val)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = trades.copy()
    t = t.sort_values(entry_col).reset_index(drop=True)

    n = len(t)
    n1 = int(n * train_frac)
    n2 = int(n * (train_frac + val_frac))
    n1 = max(1, min(n1, n - 2))
    n2 = max(n1 + 1, min(n2, n - 1))

    # Boundary times (last timestamp included in each segment)
    t_train_end = t.loc[n1 - 1, entry_col]
    t_val_end = t.loc[n2 - 1, entry_col]

    # Filter df by index using boundaries
    df_train = df.loc[df.index <= t_train_end]
    df_val = df.loc[(df.index > t_train_end) & (df.index <= t_val_end)]
    df_test = df.loc[df.index > t_val_end]

    return df_train, df_val, df_test


def probability_to_bet_size(prob_series: pd.Series) -> pd.Series:
    """Convert probabilities to Kelly-like bet fraction in [0,1]."""
    from scipy.stats import norm

    p = prob_series.astype(float).clip(1e-6, 1.0 - 1e-6)
    z = (p - 0.0) / np.sqrt(p * (1 - p))
    bet_sizes = 2.0 * norm.cdf(np.abs(z)) - 1.0
    return pd.Series(np.clip(bet_sizes, 0.0, 1.0), index=prob_series.index)


def prob_rrr_to_size(prob: pd.Series, rrr: pd.Series, fraction: float = 0.5) -> pd.Series:
    """Convert win probability + reward-to-risk ratio (RRR) into fractional-Kelly size in [0, 1]."""
    p = prob.astype(float).clip(1e-6, 1.0 - 1e-6)
    b = rrr.astype(float).clip(1e-6, None)

    # Full Kelly for a binary bet with win payoff b and loss 1
    f_full = (((b + 1.0) * p) - 1.0) / b

    # Fractional Kelly and clamp to [0, 1]
    f = (fraction * f_full).clip(0.0, 1.0)
    return pd.Series(f, index=prob.index)


def plot_probability_distributions(
    prob_col,
    bet_cal,
    bins: int = 30,
    figsize: tuple[int, int] = (12, 5),
):
    style = DEFAULT_STYLE

    prob_values = np.asarray(prob_col, dtype=float).ravel()
    bet_values = np.asarray(bet_cal, dtype=float).ravel()

    prob_values = prob_values[~np.isnan(prob_values)]
    bet_values = bet_values[~np.isnan(bet_values)]

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=100)

    datasets = [
        (prob_values, "Probability", style.accent1, axes[0]),
        (bet_values, "Bet Size", style.accent2, axes[1]),
    ]

    for values, label, color, ax in datasets:
        ax.hist(
            values,
            bins=bins,
            alpha=0.7,
            color=color,
            edgecolor=style.line,
            linewidth=1.2,
            density=True,
        )

        mean_val = np.mean(values)
        median_val = np.median(values)

        ax.axvline(
            mean_val,
            color=style.font_color,
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
            alpha=0.8,
        )
        ax.axvline(
            median_val,
            color=style.font_color,
            linestyle=":",
            linewidth=2,
            label=f"Median: {median_val:.3f}",
            alpha=0.6,
        )

        ax.set_xlabel(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.set_title(f"{label} Distribution", fontsize=style.title_size, fontweight="bold", pad=12)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.text(
            0.02,
            0.98,
            f"N = {len(values)}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            color=style.font_color,
            bbox=dict(boxstyle="round", facecolor=style.plot_bgcolor, alpha=0.8, edgecolor=style.line),
        )

    fig.suptitle(
        "Probability and Bet Size Distributions",
        fontsize=14,
        fontweight="bold",
        color=style.font_color,
        y=0.98,
    )

    style.apply_mpl(fig)

    for ax in fig.get_axes():
        ax.grid(True, alpha=0.3, color=style.grid, linestyle="--", linewidth=0.5)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig
