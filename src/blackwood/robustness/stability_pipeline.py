from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm.notebook import tqdm

if not os.environ.get("MPLCONFIGDIR"):
    _MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "matplotlib-cache")
    os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _MPL_CACHE_DIR

from backtesting import Backtest, Strategy
from joblib import Parallel, delayed

from blackwood.config import CASH, MARGIN, RANDOM_STATE, SPLIT_TIME
from blackwood.data.splitters import CPCVSplitter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

_FOLD_COL_RE = re.compile(r"^fold_(\d+)_(\d+)_sharpe$")


def _get_plotting_tools(force_agg: bool = False):
    """Lazy-load Matplotlib and project styling."""
    import matplotlib.pyplot as plt

    if force_agg:
        plt.switch_backend("Agg")
    from blackwood.visualization.style import DEFAULT_STYLE

    return plt, DEFAULT_STYLE


def _finalize_plot(fig: Figure, plt: Any, show: bool, fallback_filename: str, use_tight_layout: bool = True) -> None:
    if use_tight_layout:
        plt.tight_layout()
    if not show:
        return
    backend = str(plt.get_backend()).lower()
    if "agg" in backend:
        fig.savefig(fallback_filename, dpi=150, bbox_inches="tight")
        print(f"Non-interactive backend '{backend}'. Saved plot to {fallback_filename}.")
        return
    plt.show(block=False)
    plt.pause(0.001)


def extract_param_cols(df: pd.DataFrame, param_space: dict[str, Any]) -> list[str]:
    """Return columns in df that are valid strategy parameters."""
    return [c for c in df.columns if c in param_space]


_VALID_IS_PERF_METRICS = {"blend", "median", "p25", "mean", "sharpe_ulcer_80_20", "sharpe_ulcer"}


def _normalize_is_perf_metric(metric: Any) -> str:
    metric_str = str(metric).strip().lower()
    if metric_str == "sharpe_ulcer":
        return "sharpe_ulcer_80_20"
    return metric_str if metric_str in _VALID_IS_PERF_METRICS else "blend"


def _normalize_blend_weights(weights: Any) -> tuple[float, float]:
    try:
        w_med, w_p25 = float(weights[0]), float(weights[1])
    except Exception:
        w_med, w_p25 = 0.7, 0.3
    total = w_med + w_p25
    if not np.isfinite(total) or total <= 0:
        return 0.7, 0.3
    return float(w_med / total), float(w_p25 / total)


def _blend_sharpe_values(
    median_vals: np.ndarray,
    p25_vals: np.ndarray,
    fallback_vals: np.ndarray,
    w_med: float,
    w_p25: float,
) -> np.ndarray:
    med = np.asarray(median_vals, dtype=float)
    p25 = np.asarray(p25_vals, dtype=float)
    fallback = np.asarray(fallback_vals, dtype=float)
    out = np.full_like(med, np.nan, dtype=float)

    both = np.isfinite(med) & np.isfinite(p25)
    out[both] = w_med * med[both] + w_p25 * p25[both]
    med_only = np.isfinite(med) & ~np.isfinite(out)
    out[med_only] = med[med_only]
    p25_only = np.isfinite(p25) & ~np.isfinite(out)
    out[p25_only] = p25[p25_only]
    fb = np.isfinite(fallback) & ~np.isfinite(out)
    out[fb] = fallback[fb]
    return out


def _minmax_normalize(values: np.ndarray, lower_is_better: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    valid = arr[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if hi > lo:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    else:
        out[finite] = 0.5
    if lower_is_better:
        out[finite] = 1.0 - out[finite]
    return out


def _resolve_is_perf(df: pd.DataFrame, config: StabilityConfig) -> tuple[pd.Series, str, str]:
    policy = _normalize_is_perf_metric(getattr(config, "is_perf_metric", "blend"))
    df_out = df

    if "cpcv_sharpe_blend" not in df_out.columns:
        w_med, w_p25 = _normalize_blend_weights(getattr(config, "is_perf_blend_weights", (0.7, 0.3)))
        median_vals = (
            df_out["cpcv_sharpe_median"].to_numpy(dtype=float)
            if "cpcv_sharpe_median" in df_out.columns
            else np.full(len(df_out), np.nan, dtype=float)
        )
        p25_vals = (
            df_out["cpcv_sharpe_p25"].to_numpy(dtype=float)
            if "cpcv_sharpe_p25" in df_out.columns
            else np.full(len(df_out), np.nan, dtype=float)
        )
        fallback_vals = (
            df_out["mean_sharpe"].to_numpy(dtype=float)
            if "mean_sharpe" in df_out.columns
            else np.full(len(df_out), np.nan, dtype=float)
        )
        df_out["cpcv_sharpe_blend"] = _blend_sharpe_values(
            median_vals=median_vals,
            p25_vals=p25_vals,
            fallback_vals=fallback_vals,
            w_med=w_med,
            w_p25=w_p25,
        )

    if policy == "sharpe_ulcer_80_20":
        sharpe_pref = ("cpcv_sharpe_blend", "cpcv_sharpe_median", "cpcv_sharpe_p25", "mean_sharpe")
        sharpe_col = next((c for c in sharpe_pref if c in df_out.columns), "none")
        sharpe_vals = (
            pd.to_numeric(df_out[sharpe_col], errors="coerce")
            if sharpe_col != "none"
            else pd.Series(np.full(len(df_out), np.nan), index=df_out.index)
        )
        ulcer_col = "cpcv_ulcer_min" if "cpcv_ulcer_min" in df_out.columns else "none"
        if ulcer_col != "none":
            ulcer_vals = pd.to_numeric(df_out[ulcer_col], errors="coerce")
            sharpe_norm = _minmax_normalize(sharpe_vals.to_numpy(dtype=float), lower_is_better=False)
            ulcer_norm = _minmax_normalize(ulcer_vals.to_numpy(dtype=float), lower_is_better=True)
            combined = 0.8 * sharpe_norm + 0.2 * ulcer_norm
            combined = np.where(np.isfinite(combined), combined, sharpe_norm)
            return pd.Series(combined, index=df_out.index), policy, f"{sharpe_col}+{ulcer_col}"
        return sharpe_vals, policy, sharpe_col

    preference = {
        "blend": ("cpcv_sharpe_blend", "cpcv_sharpe_median", "cpcv_sharpe_p25", "mean_sharpe"),
        "median": ("cpcv_sharpe_median", "cpcv_sharpe_blend", "cpcv_sharpe_p25", "mean_sharpe"),
        "p25": ("cpcv_sharpe_p25", "cpcv_sharpe_blend", "cpcv_sharpe_median", "mean_sharpe"),
        "mean": ("mean_sharpe", "cpcv_sharpe_blend", "cpcv_sharpe_median", "cpcv_sharpe_p25"),
    }[policy]
    for col in preference:
        if col in df_out.columns:
            return pd.to_numeric(df_out[col], errors="coerce"), policy, col
    return pd.Series(np.full(len(df_out), np.nan), index=df_out.index), policy, "none"


def _is_perf_soft_penalty(values: np.ndarray, floor: float, softness: float) -> np.ndarray:
    """
    Exponential soft penalty for low IS performance.

    penalty = exp(-max(0, floor - perf) / softness)
    """
    arr = np.asarray(values, dtype=float)
    softness_safe = max(float(softness), 1e-9)
    shortfall = np.maximum(0.0, float(floor) - arr)
    penalty = np.exp(-shortfall / softness_safe)
    penalty = np.clip(penalty, 0.0, 1.0)
    penalty[~np.isfinite(arr)] = 0.0
    return penalty.astype(float)


def _interpret_param_spec(spec: Any) -> dict[str, Any]:
    """
    Interpret param-space semantics consistently with optimization.py.

    Rules:
    - tuple(...) => categorical values
    - list([lo, hi]) numeric => numeric range (sorted bounds)
    - list(...) otherwise => categorical values
    """
    if isinstance(spec, tuple):
        return {"kind": "categorical", "values": list(spec)}

    if isinstance(spec, list):
        if len(spec) == 2 and all(isinstance(v, (int, float, np.number)) for v in spec):
            low, high = sorted((float(spec[0]), float(spec[1])))
            is_int = all(isinstance(v, (int, np.integer)) for v in spec)
            return {"kind": "range", "low": low, "high": high, "is_int": bool(is_int)}
        return {"kind": "categorical", "values": list(spec)}

    return {"kind": "unknown"}


# ---- Config / result containers ----


@dataclass
class StabilityConfig:
    """Configuration for parameter stability pipeline."""

    max_folds: int = 5
    purged_weeks: int = 1
    embargo_weeks: int = 1
    n_bootstrap: int = 200
    block_length_days: int = 20
    performance_percentile: float = 0.2
    cv_threshold: float = 0.8
    stability_radius_per_dim: float = 0.02
    stability_min_neighbors: int = 1
    stability_score_threshold: float = 0.5
    weight_performance: float = 0.65
    weight_stability: float = 0.20
    weight_consistency: float = 0.15
    is_perf_metric: str = "sharpe_ulcer_80_20"
    is_perf_blend_weights: tuple[float, float] = (0.7, 0.3)
    is_perf_floor: float = 0.8
    is_perf_softness: float = 0.2
    phase5_weight_cpcv_perf: float = 0.55
    phase5_weight_cpcv_consistency: float = 0.10
    phase5_weight_bootstrap_stability: float = 0.10
    phase5_weight_proximity_stability: float = 0.10
    phase5_weight_oos_robustness: float = 0.05
    top_n_candidates: int = 20
    n_jobs: int = 2
    cpcv_eval_top_k: int = 0
    oos_sharpe_min: float = 0.8
    oos_degradation_min: float = 0.6
    oos_maxdd_max_pct: float = 25.0
    neigh_n: int = 60
    neigh_n_quick: int = 30
    neigh_radius: float = 0.10
    neigh_pass_min: float = 0.70
    tier1_score: float = 0.8
    tier2_score: float = 0.6
    tier3_score: float = 0.4
    quick_mode: bool = False
    cash: float = CASH
    spread: float = 0
    commission: tuple = (0, 0)
    margin: float = MARGIN
    random_state: int = RANDOM_STATE


@dataclass
class PhaseResult:
    phase_name: str
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    passed: bool = True


class BacktestRunner:
    """Centralized Backtesting.py runner configured by StabilityConfig."""

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config

    def _make_dynamic_strategy(self, base_cls: type[Strategy], params: dict[str, Any]) -> type[Strategy]:
        strategy = type("_S", (base_cls,), dict(params))
        return strategy

    def run(self, df: pd.DataFrame, strategy_class: type[Strategy], params: dict[str, Any]) -> dict:
        bt = Backtest(
            df,
            self._make_dynamic_strategy(strategy_class, params),
            cash=self.config.cash,
            commission=self.config.commission,
            spread=self.config.spread,
            margin=self.config.margin,
            finalize_trades=True,
            trade_on_close=True,
        )
        return bt.run()


class ParameterPerturber:
    """Generate bounded parameter perturbations based on param_space semantics."""

    def __init__(self, config: StabilityConfig, param_space: dict[str, Any] | None = None) -> None:
        self.config = config
        self.param_space = param_space or {}

    def perturb(self, base_params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        r = float(self.config.neigh_radius)
        out: dict[str, Any] = {}
        for name, base_val in base_params.items():
            spec = self.param_space.get(name)
            spec_info = _interpret_param_spec(spec)
            if (
                spec_info["kind"] == "range"
                and isinstance(base_val, (int, float, np.integer, np.floating))
                and np.isfinite(base_val)
            ):
                low = float(spec_info["low"])
                high = float(spec_info["high"])
                if bool(spec_info.get("is_int", False)):
                    span = max(high - low, 1.0)
                    step = int(max(1, round(span * r)))
                    cand = int(base_val) + int(rng.integers(-step, step + 1))
                    out[name] = int(np.clip(cand, low, high))
                else:
                    if abs(float(base_val)) > 1e-9:
                        cand = float(base_val) * (1.0 + rng.uniform(-r, r))
                    else:
                        cand = float(base_val) + (high - low) * rng.uniform(-r, r)
                    out[name] = float(np.clip(cand, low, high))
            elif spec_info["kind"] == "categorical" and spec_info["values"]:
                vals = spec_info["values"]
                out[name] = vals[int(rng.integers(0, len(vals)))] if rng.random() < 0.20 else base_val
            else:
                out[name] = base_val
        return out


class DualFilterSelector:
    """Phase 2: Performance, stability, and consistency filters with proximity-based stability."""

    def __init__(self, config: StabilityConfig, param_space: dict) -> None:
        self.config = config
        self.param_space = param_space

    def _normalize_params(self, df: pd.DataFrame, param_cols: list) -> np.ndarray:
        normalized = np.zeros((len(df), len(param_cols)), dtype=np.float32)
        for i, param in enumerate(param_cols):
            values = df[param].to_numpy(dtype=np.float32, copy=False)
            spec_info = _interpret_param_spec(self.param_space.get(param))
            if spec_info["kind"] == "range":
                low = float(spec_info["low"])
                high = float(spec_info["high"])
                denom = high - low
                normalized[:, i] = (values - low) / denom if denom > 0 else 0.5
            elif spec_info["kind"] == "categorical" and spec_info["values"]:
                vals = spec_info["values"]
                val_to_idx = {v: idx for idx, v in enumerate(vals)}
                n = max(len(vals) - 1, 1)
                normalized[:, i] = np.array([val_to_idx.get(v, 0) for v in df[param]], dtype=np.float32) / n
            else:
                min_val, max_val = values.min(), values.max()
                normalized[:, i] = (values - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        return np.clip(normalized, 0.0, 1.0)

    def _build_neighbor_tree(self, normalized_params: np.ndarray) -> BallTree:
        return BallTree(normalized_params.astype(np.float64), metric="manhattan")

    def _calculate_proximity_stability(
        self,
        df: pd.DataFrame,
        param_cols: list,
        fold_cols: list,
        tree: BallTree,
        normalized_params: np.ndarray,
    ) -> np.ndarray:
        n_samples = len(df)
        radius_threshold = self.config.stability_radius_per_dim * len(param_cols)
        fold_arrays = [df[col].to_numpy(dtype=np.float32, copy=False) for col in fold_cols]
        valid_masks = [~np.isnan(arr) for arr in fold_arrays]
        points = normalized_params.astype(np.float64)
        stability_scores = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            neighbor_idxs_list, distances_list = tree.query_radius(
                points[i : i + 1], r=radius_threshold, return_distance=True
            )
            neighbor_idxs = neighbor_idxs_list[0]
            distances = distances_list[0].astype(np.float32)
            fold_performances = []

            for fold_sharpes, valid_mask in zip(fold_arrays, valid_masks, strict=True):
                if valid_mask.sum() == 0:
                    continue
                neighbor_valid = valid_mask[neighbor_idxs]
                n_valid_neighbors = neighbor_valid.sum()
                if n_valid_neighbors < self.config.stability_min_neighbors:
                    if valid_mask[i]:
                        fold_performances.append(fold_sharpes[i])
                    continue
                valid_dists = distances[neighbor_valid]
                valid_sharpes = fold_sharpes[neighbor_idxs[neighbor_valid]]
                weights = 1.0 / (1.0 + valid_dists)
                fold_performances.append(float(np.dot(weights, valid_sharpes) / weights.sum()))

            if len(fold_performances) < 2:
                continue  # stability_scores[i] already 0.0
            mean_perf = float(np.mean(fold_performances))
            if np.isnan(mean_perf):
                continue
            std_perf = float(np.std(fold_performances, ddof=1))
            cv = np.abs(std_perf / (np.abs(mean_perf) + 1e-6))
            stability_scores[i] = 1.0 / (1.0 + cv)

        return stability_scores

    def apply_filters(self, cpcv_results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        if cpcv_results.empty:
            return pd.DataFrame(), {}

        df = cpcv_results.copy()
        param_cols = [c for c in df.columns if c in self.param_space and not c.startswith(("fold_", "mean_", "std_"))]
        fold_cols = [c for c in df.columns if _FOLD_COL_RE.match(c)]
        if not fold_cols:
            return pd.DataFrame(), {"total": len(df), "top_n": 0}

        normalized_params = self._normalize_params(df, param_cols)
        tree = self._build_neighbor_tree(normalized_params)

        # --- Performance filter ---
        perf_vals, is_perf_metric, is_perf_source_col = _resolve_is_perf(df, self.config)
        perf_threshold = perf_vals.quantile(1 - self.config.performance_percentile)
        df["pass_performance"] = perf_vals >= perf_threshold
        perf_min, perf_max = perf_vals.min(), perf_vals.max()
        df["score_performance"] = (perf_vals - perf_min) / (perf_max - perf_min) if perf_max > perf_min else 0.5
        df["is_perf_metric"] = is_perf_metric
        df["is_perf_source_col"] = is_perf_source_col
        df["is_perf_raw"] = perf_vals.to_numpy(dtype=float)
        df["is_perf_penalty"] = _is_perf_soft_penalty(
            df["is_perf_raw"].to_numpy(dtype=float),
            floor=self.config.is_perf_floor,
            softness=self.config.is_perf_softness,
        )
        df["score_performance_adjusted"] = df["score_performance"] * df["is_perf_penalty"]

        # --- Stability filter ---
        df["stability_score"] = self._calculate_proximity_stability(df, param_cols, fold_cols, tree, normalized_params)
        df["pass_stability"] = df["stability_score"] >= self.config.stability_score_threshold

        # --- Consistency filter ---
        fold_values = df[fold_cols].to_numpy(dtype=np.float64)
        valid_counts = np.sum(~np.isnan(fold_values), axis=1)
        insufficient = valid_counts < 2
        means = np.nanmean(fold_values, axis=1)
        stds = np.full(len(df), np.nan)
        if (~insufficient).any():
            stds[~insufficient] = np.nanstd(fold_values[~insufficient], axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_scores = np.abs(stds / means)
        cv_scores[means == 0] = np.inf
        cv_scores[insufficient] = np.inf
        df["cv_sharpe"] = cv_scores
        df["pass_consistency"] = df["cv_sharpe"] < self.config.cv_threshold
        df["score_consistency"] = 1.0 / (1.0 + np.minimum(cv_scores, 5.0))

        # --- Composite ---
        df["phase2_score"] = (
            self.config.weight_performance * df["score_performance_adjusted"]
            + self.config.weight_stability * df["stability_score"]
            + self.config.weight_consistency * df["score_consistency"]
        )
        df["phase2_component_performance"] = self.config.weight_performance * df["score_performance_adjusted"]
        df["phase2_component_stability"] = self.config.weight_stability * df["stability_score"]
        df["phase2_component_consistency"] = self.config.weight_consistency * df["score_consistency"]
        df["pass_all_filters"] = df["pass_performance"] & df["pass_stability"] & df["pass_consistency"]
        df["phase2_rank"] = df["phase2_score"].rank(method="first", ascending=False).astype(int)
        df["phase2_selected_top_n"] = df["phase2_rank"] <= int(self.config.top_n_candidates)
        df["phase2_dropped_by_top_n"] = ~df["phase2_selected_top_n"]

        top_n = int(self.config.top_n_candidates)
        top_n_scores = df["phase2_score"].nlargest(top_n)
        top_n_cutoff_score = float(top_n_scores.iloc[-1]) if len(top_n_scores) else np.nan
        df["phase2_top_n_cutoff_score"] = top_n_cutoff_score

        top_n_df = df.nlargest(self.config.top_n_candidates, "phase2_score").copy()
        top_n_df["phase2_selected_top_n"] = True
        top_n_df["phase2_dropped_by_top_n"] = False
        filter_counts = {
            "total": len(df),
            "pass_performance": int(df["pass_performance"].sum()),
            "pass_stability": int(df["pass_stability"].sum()),
            "pass_consistency": int(df["pass_consistency"].sum()),
            "pass_all_filters": int(df["pass_all_filters"].sum()),
            "top_n_selected": len(top_n_df),
            "top_n_pass_all": int(top_n_df["pass_all_filters"].sum()),
            "top_n_cutoff_score": top_n_cutoff_score,
            "dropped_by_top_n": int(df["phase2_dropped_by_top_n"].sum()),
        }
        return top_n_df, filter_counts


class PnLBootstrapValidator:
    """Phase 3: PnL-space bootstrap analysis for robustness validation."""

    def __init__(self, config: StabilityConfig, param_space: dict[str, Any]) -> None:
        self.config = config
        self.param_space = param_space
        self.runner = BacktestRunner(config)

    @staticmethod
    def _failed_bootstrap_result(full_row: dict) -> dict:
        return {
            **full_row,
            "bootstrap_sharpe_mean": np.nan,
            "bootstrap_sharpe_ci_lower": np.nan,
            "bootstrap_sharpe_ci_upper": np.nan,
            "bootstrap_sharpe_std": np.nan,
            "bootstrap_calmar_mean": np.nan,
            "bootstrap_stability_score": 0.0,
            "bootstrap_n_valid": 0,
            "pass_bootstrap": False,
        }

    def validate_parameters(
        self, param_sets: pd.DataFrame, train_df: pd.DataFrame, strategy_class: type[Strategy]
    ) -> pd.DataFrame:
        n_iterations = 100 if self.config.quick_mode else self.config.n_bootstrap
        block_length = self.config.block_length_days
        param_cols = extract_param_cols(param_sets, self.param_space)

        if not param_cols:
            exclude_cols = {
                "mean_sharpe",
                "cv_sharpe",
                "stability_score",
                "pass_performance",
                "pass_stability",
                "pass_consistency",
                "pass_all_filters",
                "phase2_score",
                "score_performance",
                "score_consistency",
            }
            exclude_prefixes = (
                "fold_",
                "bootstrap_",
                "mean_",
                "std_",
                "cv_",
                "stability_",
                "pass_",
                "score_",
                "phase2_",
                "cpcv_",
            )
            param_cols = [
                c
                for c in param_sets.columns
                if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)
            ]

        param_dicts = param_sets[param_cols].to_dict("records")
        full_row_dicts = param_sets.to_dict("records")

        base_seed = np.random.SeedSequence(self.config.random_state)
        worker_seed_ints = [int(s.generate_state(1)[0]) for s in base_seed.spawn(len(param_dicts))]

        if self.config.n_jobs == 1:
            results = [
                self._bootstrap_single_parameter(
                    params=params,
                    full_row=full_row,
                    train_df=train_df,
                    strategy_class=strategy_class,
                    n_iterations=n_iterations,
                    block_length=block_length,
                    seed=seed,
                )
                for params, full_row, seed in zip(param_dicts, full_row_dicts, worker_seed_ints, strict=True)
            ]
        else:
            n_workers = self.config.n_jobs if self.config.n_jobs > 0 else -1
            results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                delayed(_pnl_bootstrap_worker)(
                    params=params,
                    full_row=full_row,
                    train_df=train_df,
                    strategy_class=strategy_class,
                    param_space=self.param_space,
                    config=self.config,
                    worker_seed=seed,
                    n_iterations=n_iterations,
                    block_length=block_length,
                )
                for params, full_row, seed in zip(param_dicts, full_row_dicts, worker_seed_ints, strict=True)
            )
        return pd.DataFrame(results)

    def _bootstrap_single_parameter(
        self,
        params: dict,
        full_row: dict,
        train_df: pd.DataFrame,
        strategy_class: type[Strategy],
        n_iterations: int,
        block_length: int,
        seed: int,
    ) -> dict:
        try:
            stats = self.runner.run(train_df, strategy_class, params)
            equity_curve = stats["_equity_curve"]["Equity"]
        except Exception:
            return self._failed_bootstrap_result(full_row)

        try:
            equity_daily = equity_curve.resample("D").last().dropna()
            if len(equity_daily) < 2:
                raise ValueError("Insufficient daily data")
            log_returns = np.log(equity_daily / equity_daily.shift(1)).dropna().values
            initial_equity = equity_daily.iloc[0]
        except Exception:
            return self._failed_bootstrap_result(full_row)

        rng = np.random.default_rng(seed)
        T = len(log_returns)
        sharpe_vals: list[float] = []
        calmar_vals: list[float] = []
        for _ in range(n_iterations):
            sampled_log_returns = self._stationary_block_resample(
                log_returns,
                n_samples=T,
                avg_block_length=block_length,
                rng=rng,
            )
            sharpe, calmar = self._compute_single_path_metrics(sampled_log_returns, initial_equity=initial_equity)
            if np.isfinite(sharpe):
                sharpe_vals.append(float(sharpe))
            if np.isfinite(calmar):
                calmar_vals.append(float(calmar))

        valid_sharpes = np.asarray(sharpe_vals, dtype=float)
        valid_calmars = np.asarray(calmar_vals, dtype=float)

        if len(valid_sharpes) > 0:
            mean_sharpe = float(np.mean(valid_sharpes))
            ci_lower = float(np.percentile(valid_sharpes, 2.5))
            ci_upper = float(np.percentile(valid_sharpes, 97.5))
            std_sharpe = float(np.std(valid_sharpes))
            cv_boot = abs(std_sharpe / mean_sharpe) if mean_sharpe != 0 else np.inf
            stability = 1.0 / (1.0 + cv_boot)
        else:
            mean_sharpe = ci_lower = ci_upper = std_sharpe = np.nan
            stability = 0.0

        return {
            **full_row,
            "bootstrap_sharpe_mean": mean_sharpe,
            "bootstrap_sharpe_ci_lower": ci_lower,
            "bootstrap_sharpe_ci_upper": ci_upper,
            "bootstrap_sharpe_std": std_sharpe,
            "bootstrap_calmar_mean": float(np.mean(valid_calmars)) if len(valid_calmars) > 0 else np.nan,
            "bootstrap_stability_score": stability,
            "bootstrap_n_valid": len(valid_sharpes),
            "pass_bootstrap": stability > 0.5 and len(valid_sharpes) >= n_iterations * 0.8,
        }

    @staticmethod
    def _stationary_block_resample(
        data: np.ndarray, n_samples: int, avg_block_length: int, rng: np.random.Generator
    ) -> np.ndarray:
        n_obs = len(data)
        if avg_block_length >= n_obs:
            return data[rng.integers(0, n_obs, size=n_samples)]

        p = 1.0 / avg_block_length
        # Pre-generate all random numbers in batch (avoids repeated Python-level calls)
        expected_blocks = max(int(np.ceil(n_samples / avg_block_length)) * 3, 32)
        starts = rng.integers(0, n_obs, size=expected_blocks)
        lengths = rng.geometric(p, size=expected_blocks)

        resampled = np.empty(n_samples, dtype=data.dtype)
        idx = 0
        for start, block_len in zip(starts, lengths, strict=True):
            if idx >= n_samples:
                break
            actual = min(int(block_len), n_obs - start, n_samples - idx)
            resampled[idx : idx + actual] = data[start : start + actual]
            idx += actual

        if idx < n_samples:  # safety fallback (extremely rare)
            resampled[idx:] = data[rng.integers(0, n_obs, size=n_samples - idx)]
        return resampled

    @staticmethod
    def _compute_single_path_metrics(
        log_returns: np.ndarray,
        initial_equity: float,
        annual_days: int = 252,
    ) -> tuple[float, float]:
        """Annualized Sharpe and Calmar for one bootstrap path."""
        log_returns = np.asarray(log_returns, dtype=float)
        if log_returns.size < 2:
            return np.nan, np.nan

        pct_ret = np.expm1(log_returns)
        gmean = np.expm1(log_returns.mean())
        annual_ret = (1.0 + gmean) ** annual_days - 1.0
        var = pct_ret.var(ddof=1)
        annual_var = max(
            0.0,
            (var + (1.0 + gmean) ** 2) ** annual_days - (1.0 + gmean) ** (2 * annual_days),
        )
        annual_vol = np.sqrt(annual_var)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan

        equity_path = float(initial_equity) * np.exp(np.cumsum(log_returns))
        running_max = np.maximum.accumulate(equity_path)
        max_dd = float((1.0 - equity_path / running_max).max())
        calmar = annual_ret / max_dd if max_dd > 0 else np.nan
        return float(sharpe), float(calmar)

    @staticmethod
    def _compute_batch_metrics(
        log_returns_batch: np.ndarray, equity_batch: np.ndarray, annual_days: int = 252
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Annualized Sharpe and Calmar for a batch of bootstrap paths.

        Annualized variance via exact moment formula:
          Var = E[X²]^n - E[X]^{2n},  X = (1 + r_t),  n = annual_days
        """
        pct_ret = np.expm1(log_returns_batch)
        gmean = np.expm1(log_returns_batch.mean(axis=1))
        annual_ret = (1.0 + gmean) ** annual_days - 1.0
        var = pct_ret.var(axis=1, ddof=1)
        annual_var = np.maximum(
            0.0,
            (var + (1.0 + gmean) ** 2) ** annual_days - (1.0 + gmean) ** (2 * annual_days),
        )
        annual_vol = np.sqrt(annual_var)
        sharpe = np.where(annual_vol > 0, annual_ret / annual_vol, np.nan)
        running_max = np.maximum.accumulate(equity_batch, axis=1)
        max_dd = (1.0 - equity_batch / running_max).max(axis=1)
        calmar = np.where(max_dd > 0, annual_ret / max_dd, np.nan)
        return sharpe, calmar


def _pnl_bootstrap_worker(
    params: dict,
    full_row: dict,
    train_df: pd.DataFrame,
    strategy_class: type[Strategy],
    param_space: dict[str, Any],
    config: StabilityConfig,
    worker_seed: int,
    n_iterations: int,
    block_length: int,
) -> dict:
    validator = PnLBootstrapValidator(config, param_space=param_space)
    return validator._bootstrap_single_parameter(
        params=params,
        full_row=full_row,
        train_df=train_df,
        strategy_class=strategy_class,
        n_iterations=n_iterations,
        block_length=block_length,
        seed=worker_seed,
    )


def _cpcv_path_worker(
    path_id: str,
    folds: list,
    strategy_class: type[Strategy],
    param_space: dict,
    config: StabilityConfig,
    n_cpcv_trials: int,
    path_seed: int,
    constraint: Callable | None = None,
) -> dict:
    """Worker for parallel CPCV path optimization."""
    from blackwood.optimization.optimization import OptunaOptimizer

    runner = BacktestRunner(config)
    all_param_results: dict[tuple, dict] = {}

    for fold_idx, (train_df, test_df) in enumerate(
        tqdm(folds, desc=f"Path {path_id}", unit="fold", position=1, leave=False)
    ):

        def bt_func_train(df: pd.DataFrame = train_df, **params):
            return runner.run(df, strategy_class, params)

        optimizer = OptunaOptimizer(bt_func=bt_func_train)

        optimizer = OptunaOptimizer(bt_func=bt_func_train)
        optimizer.optimize(
            param_space=param_space,
            metric="Sharpe Ratio",
            constraint=constraint,
            n_trials=n_cpcv_trials,
            random_state=path_seed,
            verbose=False,
        )

        completed = optimizer.results_df.query("state == 'COMPLETE'").copy()
        top_k = config.cpcv_eval_top_k
        if top_k > 0 and len(completed) > top_k and "Metric" in completed.columns:
            completed = completed.nlargest(top_k, "Metric")

        param_cols = [p for p in param_space if p in completed.columns]
        for fold_trial_params in completed[param_cols].to_dict("records"):
            stats_test = runner.run(test_df, strategy_class, fold_trial_params)
            test_sharpe = float(stats_test.get("Sharpe Ratio", np.nan))
            test_calmar = float(stats_test.get("Calmar Ratio", np.nan))
            test_return = float(stats_test.get("Return (Ann.) [%]", np.nan))
            test_ulcer = float(stats_test.get("Ulcer Index [%]", np.nan))

            param_key = tuple(sorted(fold_trial_params.items()))
            if param_key not in all_param_results:
                all_param_results[param_key] = {
                    "params": fold_trial_params.copy(),
                    "fold_sharpes": {},
                    "fold_calmars": {},
                    "fold_returns": {},
                    "fold_ulcers": {},
                }
            fold_key = (path_id, fold_idx)
            all_param_results[param_key]["fold_sharpes"][fold_key] = test_sharpe
            all_param_results[param_key]["fold_calmars"][fold_key] = test_calmar
            all_param_results[param_key]["fold_returns"][fold_key] = test_return
            all_param_results[param_key]["fold_ulcers"][fold_key] = test_ulcer

    return all_param_results


class OOSValidator:
    """Phase 4: Out-of-sample validation on holdout data."""

    def __init__(self, config: StabilityConfig, param_space: dict[str, Any]) -> None:
        self.config = config
        self.param_space = param_space
        self.runner = BacktestRunner(config)
        self.perturber = ParameterPerturber(config, param_space)

    @staticmethod
    def _failed_oos_result(row_dict: dict, param_key: tuple, is_metrics: dict) -> dict:
        cached = is_metrics.get(param_key, {})
        return {
            **row_dict,
            "is_sharpe": cached.get("sharpe", np.nan),
            "is_calmar": cached.get("calmar", np.nan),
            "is_return": cached.get("return", np.nan),
            "oos_sharpe": np.nan,
            "oos_calmar": np.nan,
            "oos_return": np.nan,
            "oos_winrate": np.nan,
            "oos_maxdd": np.nan,
            "oos_maxdd_frac": np.nan,
            "oos_trades": 0,
            "degradation_sharpe": np.nan,
            "degradation_calmar": np.nan,
            "degradation_return": np.nan,
            "pass_oos": False,
            "neigh_pass_rate": 0.0,
            "neigh_sharpe_p05": np.nan,
            "pass_neighborhood": False,
        }

    @staticmethod
    def _safe_ratio(num: float, denom: float) -> float:
        return num / denom if denom != 0 and np.isfinite(denom) else np.nan

    def _neighborhood_seed(self, param_key: tuple) -> int:
        payload = f"{self.config.random_state}|{param_key!r}".encode()
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big") % (2**32)

    def validate_parameters(
        self,
        paramsets: pd.DataFrame,
        holdout_df: pd.DataFrame,
        strategy_class: type[Strategy],
        is_metrics: dict[tuple, dict[str, float]],
    ) -> pd.DataFrame:
        param_cols = extract_param_cols(paramsets, self.param_space)
        results = []

        for row_dict in paramsets.to_dict("records"):
            params = {col: row_dict[col] for col in param_cols}
            param_key = tuple(sorted(params.items()))

            try:
                stats = self.runner.run(holdout_df, strategy_class, params)

                oos_maxdd_raw = float(stats.get("Max. Drawdown [%]", np.nan))
                oos_maxdd = abs(oos_maxdd_raw) if np.isfinite(oos_maxdd_raw) else np.nan
                oos_maxdd_frac = oos_maxdd / 100.0 if np.isfinite(oos_maxdd) else np.nan
                oos_sharpe = float(stats.get("Sharpe Ratio", np.nan))
                oos_calmar = float(stats.get("Calmar Ratio", np.nan))
                oos_return = float(stats.get("Return (Ann.) [%]", np.nan))
                oos_winrate = float(stats.get("Win Rate [%]", np.nan))
                oos_trades = int(stats.get("# Trades", 0))

                cached = is_metrics.get(param_key, {})
                is_sharpe = cached.get("sharpe", np.nan)
                is_calmar = cached.get("calmar", np.nan)
                is_return = cached.get("return", np.nan)

                deg_sharpe = self._safe_ratio(oos_sharpe, is_sharpe)
                deg_calmar = self._safe_ratio(oos_calmar, is_calmar)
                deg_return = self._safe_ratio(oos_return, is_return)

                pass_oos = (
                    np.isfinite(oos_sharpe)
                    and oos_sharpe >= self.config.oos_sharpe_min
                    and np.isfinite(deg_sharpe)
                    and deg_sharpe >= self.config.oos_degradation_min
                    and oos_trades >= 10
                )

                neigh_pass_rate, neigh_sharpe_p05, pass_neighborhood = 0.0, np.nan, False
                if pass_oos and np.isfinite(is_sharpe):
                    n_neigh = self.config.neigh_n_quick if self.config.quick_mode else self.config.neigh_n
                    rng = np.random.default_rng(self._neighborhood_seed(param_key))
                    sharpes: list[float] = []
                    valid = passed = 0
                    for _ in range(int(n_neigh)):
                        p2 = self.perturber.perturb(params, rng)
                        try:
                            st2 = self.runner.run(holdout_df, strategy_class, p2)
                        except Exception:
                            continue
                        s2 = float(st2.get("Sharpe Ratio", np.nan))
                        t2 = int(st2.get("# Trades", 0))
                        if not np.isfinite(s2):
                            continue
                        sharpes.append(s2)
                        valid += 1
                        deg2 = self._safe_ratio(s2, is_sharpe)
                        if (
                            np.isfinite(deg2)
                            and s2 >= self.config.oos_sharpe_min
                            and deg2 >= self.config.oos_degradation_min
                            and t2 >= 10
                        ):
                            passed += 1
                    if valid > 0:
                        neigh_pass_rate = passed / valid
                        neigh_sharpe_p05 = float(np.percentile(sharpes, 5))
                        pass_neighborhood = neigh_pass_rate >= self.config.neigh_pass_min

            except Exception:
                results.append(self._failed_oos_result(row_dict, param_key, is_metrics))
                continue

            results.append(
                {
                    **row_dict,
                    "is_sharpe": is_sharpe,
                    "is_calmar": is_calmar,
                    "is_return": is_return,
                    "oos_sharpe": oos_sharpe,
                    "oos_calmar": oos_calmar,
                    "oos_return": oos_return,
                    "oos_winrate": oos_winrate,
                    "oos_maxdd": oos_maxdd,
                    "oos_maxdd_frac": oos_maxdd_frac,
                    "oos_trades": oos_trades,
                    "degradation_sharpe": deg_sharpe,
                    "degradation_calmar": deg_calmar,
                    "degradation_return": deg_return,
                    "pass_oos": pass_oos,
                    "neigh_pass_rate": neigh_pass_rate,
                    "neigh_sharpe_p05": neigh_sharpe_p05,
                    "pass_neighborhood": pass_neighborhood,
                }
            )

        return pd.DataFrame(results)


class TierClassifier:
    """
    Phase 5: Multi-criteria ranking and tier assignment.

    OOS double-counting prevention:
    - Absolute `oos_sharpe` excluded from composite score.
    - IS/OOS ratio + neighborhood stability aggregated into `norm_oos_robustness`.
    - Composite = IS-dominant + small OOS weight.
    - Hard `pass_*` filters act as gates only.
    """

    def __init__(self, config: StabilityConfig, param_space: dict[str, Any]) -> None:
        self.config = config
        self.param_space = param_space

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Robust normalization via median + MAD (outlier-resistant)."""
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.zeros_like(values)
        median = np.median(finite)
        mad = np.median(np.abs(finite - median))
        if mad < 1e-9:
            out = np.where(np.isfinite(values), 0.5, 0.0)
            return out.astype(float)
        normalized = np.clip((values - median) / (1.4826 * mad), -3, 3)
        normalized = (normalized + 3) / 6
        normalized[~np.isfinite(values)] = 0.0
        return normalized

    def rank_and_classify(self, results_df: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        df = results_df.copy()
        eps = 1e-9

        # IS-only robustness
        perf_vals, is_perf_metric, is_perf_source_col = _resolve_is_perf(df, self.config)
        df["is_perf_metric"] = is_perf_metric
        df["is_perf_source_col"] = is_perf_source_col
        df["is_perf_raw"] = perf_vals.to_numpy(dtype=float)
        df["phase5_is_perf_penalty"] = _is_perf_soft_penalty(
            df["is_perf_raw"].to_numpy(dtype=float),
            floor=self.config.is_perf_floor,
            softness=self.config.is_perf_softness,
        )
        df["norm_cpcv_perf"] = self._normalize(perf_vals.to_numpy(dtype=float))
        df["norm_cpcv_consistency"] = self._normalize((1.0 / (1.0 + df["cv_sharpe"])).to_numpy(dtype=float))
        df["norm_bootstrap_stability"] = self._normalize(df["bootstrap_stability_score"].to_numpy(dtype=float))

        has_prox = "stability_score" in df.columns
        df["norm_proximity_stability"] = (
            self._normalize(df["stability_score"].to_numpy(dtype=float)) if has_prox else 0.0
        )

        # Aggregated OOS robustness
        is_sh = df.get("is_sharpe", df.get("cpcv_sharpe_median", df.get("mean_sharpe", np.nan)))
        oos_sh = df.get("oos_sharpe", np.nan)
        ratio = np.clip(
            (oos_sh.to_numpy(dtype=float) + eps) / (is_sh.to_numpy(dtype=float) + eps),
            1e-3,
            1e3,
        )
        gen_sym = np.exp(-np.abs(np.log(ratio)))
        neighborhood = df.get("neigh_pass_rate", pd.Series(np.zeros(len(df)), index=df.index)).to_numpy(dtype=float)
        oos_robust_raw = 0.55 * gen_sym + 0.45 * np.clip(neighborhood, 0.0, 1.0)
        df["norm_oos_robustness"] = np.clip(self._normalize(oos_robust_raw), 0.0, 1.0)

        # Composite score
        w_perf = float(self.config.phase5_weight_cpcv_perf)
        w_cons = float(self.config.phase5_weight_cpcv_consistency)
        w_boot = float(self.config.phase5_weight_bootstrap_stability)
        w_prox = float(self.config.phase5_weight_proximity_stability) if has_prox else 0.0
        if not has_prox:
            w_perf += float(self.config.phase5_weight_proximity_stability)
        w_oos = float(self.config.phase5_weight_oos_robustness)

        df["phase5_component_cpcv_perf"] = w_perf * df["norm_cpcv_perf"]
        df["phase5_component_cpcv_consistency"] = w_cons * df["norm_cpcv_consistency"]
        df["phase5_component_bootstrap_stability"] = w_boot * df["norm_bootstrap_stability"]
        df["phase5_component_proximity_stability"] = w_prox * df["norm_proximity_stability"]
        df["phase5_component_oos_robustness"] = w_oos * df["norm_oos_robustness"]

        df["phase5_is_part_raw"] = (
            df["phase5_component_cpcv_perf"]
            + df["phase5_component_cpcv_consistency"]
            + df["phase5_component_bootstrap_stability"]
            + df["phase5_component_proximity_stability"]
        )
        df["phase5_is_part_adjusted"] = df["phase5_is_part_raw"] * df["phase5_is_perf_penalty"]

        is_weight_sum = w_perf + w_cons + w_boot + w_prox
        total_w = max(is_weight_sum + w_oos, 1e-9)

        df["composite_score"] = np.clip(
            (df["phase5_is_part_adjusted"] + df["phase5_component_oos_robustness"]).to_numpy(dtype=float) / total_w,
            0.0,
            1.0,
        )

        # Hard gates
        pass_oos_gate = df.get("pass_oos", pd.Series(False, index=df.index))
        pass_neigh_gate = df.get("pass_neighborhood", pd.Series(True, index=df.index))
        all_pass = (
            df["pass_performance"]
            & df["pass_stability"]
            & df["pass_consistency"]
            & df["pass_bootstrap"]
            & pass_oos_gate
            & pass_neigh_gate
        )

        score = df["composite_score"]
        tier_conditions = [
            (score >= self.config.tier1_score) & all_pass,
            (score >= self.config.tier2_score) & all_pass,
            score >= self.config.tier3_score,
        ]
        df["tier"] = np.select(tier_conditions, ["Tier 1", "Tier 2", "Tier 3"], default="Reject")
        df["recommendation"] = np.select(
            tier_conditions,
            [
                "Recommended for live trading - excellent robustness",
                "Acceptable for live trading - good robustness",
                "Use with caution - moderate robustness, some filters failed",
            ],
            default="Do not use - poor robustness or critical failures",
        )

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df


class ParameterStabilityPipeline:
    """
    5-Phase parameter stability testing pipeline.
    Phase 1: CPCV execution with WFO
    Phase 2: Dual filter selection
    Phase 3: Bootstrap validation
    Phase 4: OOS validation on holdout
    Phase 5: Ranking and tier classification
    """

    def __init__(
        self,
        strategy_class: type[Strategy],
        param_space: dict[str, tuple],
        constraints: Callable[[dict[str, Any]], bool] | None = None,
        oos_start: str = SPLIT_TIME,
        n_cpcv_trials: int = 100,
        n_bootstrap: int = 1000,
        config: StabilityConfig | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.constraints = constraints
        self.oos_start = pd.Timestamp(oos_start, tz="UTC")
        self.n_cpcv_trials = n_cpcv_trials
        self.config = config or StabilityConfig(n_bootstrap=n_bootstrap)
        self._phase_results: dict[int, PhaseResult] = {}
        self._train_df: pd.DataFrame | None = None
        self._holdout_df: pd.DataFrame | None = None
        self._reference_train_df: pd.DataFrame | None = None
        self._final_ranking: pd.DataFrame | None = None

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        auto_plot: bool = True,
        show_plots: bool = False,
        reference_train_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        if verbose:
            print("=" * 80)
            print("PARAMETER STABILITY PIPELINE")
            print("=" * 80)

        self._train_df = df[df.index < self.oos_start].copy()
        self._holdout_df = df[df.index >= self.oos_start].copy()
        self._reference_train_df = reference_train_df.copy() if reference_train_df is not None else None
        if verbose:
            print("\nData Split (Temporal Integrity):")
            print(f" Train:   {self._train_df.index.min()} → {self._train_df.index.max()} ({len(self._train_df)} bars)")
            print(
                f" Holdout: {self._holdout_df.index.min()} → {self._holdout_df.index.max()} ({len(self._holdout_df)} bars)"  # noqa: E501
            )

        phases = [
            (1, "PHASE 1: CPCV Execution", self._run_phase1_cpcv),
            (2, "PHASE 2: Dual Filter Selection", lambda v: self._run_phase2_filters(self._phase_results[1].data, v)),
            (3, "PHASE 3: Bootstrap Validation", lambda v: self._run_phase3_bootstrap(self._phase_results[2].data, v)),
            (
                4,
                "PHASE 4: OOS Validation",
                lambda v: self._run_phase4_oos(self._phase_results[3].data, self._phase_results[1].metadata, v),
            ),
            (
                5,
                "PHASE 5: Ranking & Classification",
                lambda v: self._run_phase5_ranking(self._phase_results[4].data, v),
            ),
        ]

        for phase_num, label, run_fn in phases:
            if verbose:
                print(f"\n{'-' * 80}\n{label}")
            result = run_fn(verbose)
            self._phase_results[phase_num] = result
            if phase_num <= 2 and (not result.passed or result.data.empty):
                if verbose:
                    print(f"Phase {phase_num} failed or returned no results. Pipeline terminated.")
                return {"status": "failed", "phase": phase_num, "results": self._phase_results}

        self._final_ranking = self._phase_results[5].data
        self._final_ranking = self._attach_reference_train_sharpe(
            self._final_ranking,
            self._reference_train_df,
            top_n=20,
            verbose=verbose,
        )
        self._phase_results[5].data = self._final_ranking

        if verbose:
            print(f"\n{'=' * 80}\nPIPELINE COMPLETE\n{'=' * 80}")
            self._print_final_summary()
            if auto_plot:
                print(f"\n{'=' * 80}\nGENERATING VISUALIZATIONS\n{'=' * 80}")
                for label, fn in [
                    (" Creating filter funnel plot...", self.plot_filter_funnel),
                    (" Creating OOS degradation plot...", self.plot_oos_degradation),
                    (" Creating tier dashboard...", self.plot_tier_dashboard),
                ]:
                    print(label)
                    fn(show=show_plots)

        return {"status": "success", "phase_results": self._phase_results, "final_ranking": self._final_ranking}

    def _run_phase1_cpcv(self, verbose: bool) -> PhaseResult:

        if verbose:
            print(
                f" Creating CPCV splits: {self.config.max_folds} folds, "
                f"{self.config.purged_weeks}w purge, {self.config.embargo_weeks}w embargo"
            )

        splitter = CPCVSplitter(
            max_paths_to_return=self.config.max_folds,
            purged_weeks=self.config.purged_weeks,
            embargo_weeks=self.config.embargo_weeks,
        )
        cpcv_paths = CPCVSplitter.split_paths(splitter, self._train_df)
        if verbose:
            print(f" Generated {len(cpcv_paths)} CPCV paths")

        base_seed = np.random.SeedSequence(self.config.random_state)
        path_seed_ints = [int(s.generate_state(1)[0]) for s in base_seed.spawn(len(cpcv_paths))]

        n_workers = self.config.n_jobs if self.config.n_jobs > 0 else -1
        path_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_cpcv_path_worker)(
                path_id=path_id,
                folds=folds,
                strategy_class=self.strategy_class,
                param_space=self.param_space,
                config=self.config,
                n_cpcv_trials=self.n_cpcv_trials,
                path_seed=seed,
                constraint=self.constraints,
            )
            for (path_id, folds), seed in zip(cpcv_paths.items(), path_seed_ints, strict=False)
        )

        # Merge path results
        all_param_results: dict[tuple, dict] = {}
        for path_result in path_results:
            for param_key, data in path_result.items():
                if param_key not in all_param_results:
                    all_param_results[param_key] = data
                else:
                    for metric in ("fold_sharpes", "fold_calmars", "fold_returns", "fold_ulcers"):
                        all_param_results[param_key][metric].update(data[metric])

        rows, is_metrics = [], {}
        for param_key, data in all_param_results.items():
            params = data["params"]
            metrics = {}
            for metric_name, key in [
                ("sharpe", "fold_sharpes"),
                ("calmar", "fold_calmars"),
                ("return", "fold_returns"),
                ("ulcer", "fold_ulcers"),
            ]:
                arr = np.asarray(list(data[key].values()), dtype=float)
                arr = arr[np.isfinite(arr)]
                metrics[metric_name] = arr

            def _safe_stat(arr, fn: Callable) -> float:
                return float(fn(arr)) if len(arr) else np.nan

            sharpe_arr = metrics["sharpe"]
            row = {
                **params,
                "mean_sharpe": _safe_stat(sharpe_arr, np.mean),
                "mean_calmar": _safe_stat(metrics["calmar"], np.mean),
                "mean_return": _safe_stat(metrics["return"], np.mean),
                "cpcv_sharpe_median": _safe_stat(sharpe_arr, np.median),
                "cpcv_sharpe_p25": float(np.percentile(sharpe_arr, 25)) if len(sharpe_arr) else np.nan,
                "cpcv_sharpe_min": _safe_stat(sharpe_arr, np.min),
                "cpcv_ulcer_median": _safe_stat(metrics["ulcer"], np.median),
                "cpcv_ulcer_p25": float(np.percentile(metrics["ulcer"], 25)) if len(metrics["ulcer"]) else np.nan,
                "cpcv_ulcer_min": _safe_stat(metrics["ulcer"], np.min),
                **{f"fold_{pid}_{fidx}_sharpe": v for (pid, fidx), v in data["fold_sharpes"].items()},
            }
            w_med, w_p25 = _normalize_blend_weights(self.config.is_perf_blend_weights)
            row["cpcv_sharpe_blend"] = _blend_sharpe_values(
                median_vals=np.array([row["cpcv_sharpe_median"]], dtype=float),
                p25_vals=np.array([row["cpcv_sharpe_p25"]], dtype=float),
                fallback_vals=np.array([row["mean_sharpe"]], dtype=float),
                w_med=w_med,
                w_p25=w_p25,
            )[0]
            rows.append(row)
            is_metrics[param_key] = {
                "sharpe": row["cpcv_sharpe_median"],
                "sharpe_mean": row["mean_sharpe"],
                "sharpe_p25": row["cpcv_sharpe_p25"],
                "sharpe_blend": row["cpcv_sharpe_blend"],
                "ulcer_min": row["cpcv_ulcer_min"],
                "calmar": row["mean_calmar"],
                "return": row["mean_return"],
            }

        result_df = pd.DataFrame(rows)
        if verbose:
            print(" Completed CPCV optimization")
            print(f" Unique parameter sets: {len(result_df)}")
            if not result_df.empty:
                print(f" Best mean Sharpe:   {result_df['mean_sharpe'].max():.3f}")
                print(f" Median mean Sharpe: {result_df['mean_sharpe'].median():.3f}")

        return PhaseResult(
            phase_name="CPCV Execution",
            data=result_df,
            metadata={"is_metrics": is_metrics, "max_folds": self.config.max_folds},
            passed=not result_df.empty,
        )

    def _run_phase2_filters(self, cpcv_df: pd.DataFrame, verbose: bool) -> PhaseResult:
        selector = DualFilterSelector(self.config, self.param_space)
        filtered_df, counts = selector.apply_filters(cpcv_df)
        if verbose:
            total = counts["total"]
            for key, label in [
                ("pass_performance", "performance"),
                ("pass_stability", "stability"),
                ("pass_consistency", "consistency"),
                ("pass_all_filters", "ALL"),
            ]:
                n = counts[key]
                print(f" Pass {label} filter: {n} ({n / total * 100:.1f}%)")
            print(f" → Selected top {counts['top_n_selected']} ({counts['top_n_pass_all']} pass all)")
        return PhaseResult(
            phase_name="Dual Filter Selection",
            data=filtered_df,
            metadata={"filter_counts": counts},
            passed=len(filtered_df) > 0,
        )

    def _run_phase3_bootstrap(self, filtered_df: pd.DataFrame, verbose: bool) -> PhaseResult:
        validator = PnLBootstrapValidator(self.config, param_space=self.param_space)
        if verbose:
            n_iter = 100 if self.config.quick_mode else self.config.n_bootstrap
            print(f" {n_iter} bootstrap iterations * {len(filtered_df)} candidates")
        bootstrap_df = validator.validate_parameters(
            param_sets=filtered_df, train_df=self._train_df, strategy_class=self.strategy_class
        )
        n_passed = int(bootstrap_df["pass_bootstrap"].sum()) if "pass_bootstrap" in bootstrap_df else 0
        if verbose:
            print(f" Passed bootstrap: {n_passed}/{len(bootstrap_df)}")
            if not bootstrap_df.empty:
                print(f" Mean stability score: {bootstrap_df['bootstrap_stability_score'].mean():.3f}")
        return PhaseResult(
            phase_name="Bootstrap Validation",
            data=bootstrap_df,
            metadata={"n_passed_bootstrap": n_passed},
            passed=True,
        )

    def _run_phase4_oos(
        self, bootstrap_df: pd.DataFrame, phase1_metadata: dict[str, Any], verbose: bool
    ) -> PhaseResult:
        validator = OOSValidator(self.config, self.param_space)
        if verbose:
            print(f" Testing {len(bootstrap_df)} parameter sets on holdout...")
        oos_df = validator.validate_parameters(
            bootstrap_df,
            self._holdout_df,
            self.strategy_class,
            phase1_metadata.get("is_metrics", {}),
        )
        n_passed = int(oos_df["pass_oos"].sum()) if "pass_oos" in oos_df else 0
        if verbose:
            print(f" Passed OOS: {n_passed}/{len(oos_df)}")
            if not oos_df.empty and "oos_sharpe" in oos_df:
                print(f" Mean OOS Sharpe:    {oos_df['oos_sharpe'].mean():.3f}")
                print(f" Mean degradation:   {oos_df['degradation_sharpe'].mean():.3f}")
        return PhaseResult(
            phase_name="OOS Validation",
            data=oos_df,
            metadata={"n_passed_oos": n_passed},
            passed=True,
        )

    def _run_phase5_ranking(self, oos_df: pd.DataFrame, verbose: bool) -> PhaseResult:
        ranked_df = TierClassifier(self.config, self.param_space).rank_and_classify(oos_df)
        if verbose:
            tier_counts = ranked_df["tier"].value_counts()
            print(" Tier distribution:")
            for tier in ["Tier 1", "Tier 2", "Tier 3", "Reject"]:
                print(f"  {tier}: {tier_counts.get(tier, 0)}")
        return PhaseResult(
            phase_name="Ranking & Classification",
            data=ranked_df,
            metadata={"tier_counts": ranked_df["tier"].value_counts().to_dict()},
            passed=True,
        )

    def _print_final_summary(self) -> None:
        if self._final_ranking is None or self._final_ranking.empty:
            print("\nNo parameter sets passed all phases.")
            return

        from blackwood.robustness.ranking_display import display_ranking

        display_ranking(self._final_ranking, metric_policy=self.config.is_perf_metric)

    def get_tier_parameters(self, tier: int = 1) -> pd.DataFrame:
        if self._final_ranking is None:
            return pd.DataFrame()
        return self._final_ranking.query(f"tier == 'Tier {tier}'").copy()

    def get_final_ranking(self) -> pd.DataFrame:
        return self._final_ranking.copy() if self._final_ranking is not None else pd.DataFrame()

    def _attach_reference_train_sharpe(
        self,
        ranked_df: pd.DataFrame,
        reference_train_df: pd.DataFrame | None,
        top_n: int = 20,
        verbose: bool = False,
    ) -> pd.DataFrame:
        if ranked_df is None or ranked_df.empty:
            return ranked_df
        if reference_train_df is None:
            return ranked_df
        if reference_train_df.empty:
            if verbose:
                print(" Reference train dataset is empty. Skipping full-train Sharpe enrichment.")
            return ranked_df

        enriched = ranked_df.copy()
        if "full_train_sharpe" not in enriched.columns:
            enriched["full_train_sharpe"] = np.nan
        if "full_train_basis" not in enriched.columns:
            enriched["full_train_basis"] = pd.Series([None] * len(enriched), index=enriched.index, dtype="object")

        param_cols = extract_param_cols(enriched, self.param_space)
        if not param_cols:
            if verbose:
                print(" No parameter columns resolved for full-train Sharpe enrichment.")
            return enriched

        top_n_int = max(0, min(int(top_n), len(enriched)))
        if top_n_int == 0:
            return enriched

        runner = BacktestRunner(self.config)
        success = 0
        for idx in enriched.index[:top_n_int]:
            params = {col: enriched.at[idx, col] for col in param_cols}
            try:
                stats = runner.run(reference_train_df, self.strategy_class, params)
                enriched.at[idx, "full_train_sharpe"] = float(stats.get("Sharpe Ratio", np.nan))
                success += 1
            except Exception:
                enriched.at[idx, "full_train_sharpe"] = np.nan
            enriched.at[idx, "full_train_basis"] = "original_train_split"

        if verbose:
            print(f" Added full-train Sharpe for top {top_n_int} candidates ({success}/{top_n_int} successful).")
        return enriched

    def plot_filter_funnel(self, show: bool = True) -> Figure:
        plt, sty = _get_plotting_tools(force_agg=not show)
        fig, ax = plt.subplots(figsize=(10, 6))
        stage_names = ["Phase 1\nCPCV", "Phase 2\nFilters", "Phase 3\nBootstrap", "Phase 4\nOOS", "Phase 5\nTier 1"]
        counts = []
        for phase in range(1, 6):
            data = self._phase_results.get(phase, PhaseResult("", pd.DataFrame())).data
            if phase == 5:
                counts.append(len(data.query("tier == 'Tier 1'")) if "tier" in data.columns else 0)
            else:
                counts.append(len(data))
        colors = [sty.accent1, sty.accent2, sty.accent3, sty.accent4, sty.accent5]
        bars = ax.bar(stage_names, counts, color=colors, alpha=0.8)
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=12,
            )
        ax.set_ylabel("Number of Parameter Sets", fontsize=12)
        ax.set_title("Parameter Stability Pipeline - Filter Funnel", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        sty.apply_mpl(fig, ax)
        _finalize_plot(fig, plt, show=show, fallback_filename="filter_funnel.png")
        return fig

    def plot_oos_degradation(self, show: bool = True) -> Figure | None:
        plt, sty = _get_plotting_tools(force_agg=not show)
        if 4 not in self._phase_results:
            print("Phase 4 (OOS) not yet executed.")
            return None
        df = self._phase_results[4].data
        if df.empty or "is_sharpe" not in df or "oos_sharpe" not in df:
            print("No OOS data available for plotting.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_df = df.dropna(subset=["is_sharpe", "oos_sharpe"])
        if plot_df.empty:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
            return fig

        is_sharpe = plot_df["is_sharpe"].values
        oos_sharpe = plot_df["oos_sharpe"].values
        colors = np.where(plot_df["pass_oos"].values, sty.accent3, sty.accent2)
        ax.scatter(is_sharpe, oos_sharpe, c=colors, s=80, alpha=0.6, edgecolors="white", linewidth=0.5)

        lo, hi = min(is_sharpe.min(), oos_sharpe.min()), max(is_sharpe.max(), oos_sharpe.max())
        ax.plot([lo, hi], [lo, hi], "w--", alpha=0.5, linewidth=2, label="No degradation (y=x)")
        deg = self.config.oos_degradation_min
        ax.plot(
            [lo, hi],
            [lo * deg, hi * deg],
            color=sty.accent4,
            linestyle=":",
            linewidth=2,
            label=f"{deg:.0%} degradation threshold",
        )

        ax.set_xlabel("In-Sample Sharpe Ratio", fontsize=12)
        ax.set_ylabel("Out-of-Sample Sharpe Ratio", fontsize=12)
        ax.set_title("IS vs OOS Performance Degradation", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        sty.apply_mpl(fig, ax)
        _finalize_plot(fig, plt, show=show, fallback_filename="oos_degradation.png")
        return fig

    def plot_tier_dashboard(self, show: bool = True) -> Figure | None:
        plt, sty = _get_plotting_tools(force_agg=not show)
        if self._final_ranking is None or self._final_ranking.empty:
            print("No ranking data available.")
            return None

        colors_map = {
            "Tier 1": sty.accent3,
            "Tier 2": sty.accent1,
            "Tier 3": sty.accent6,
            "Reject": sty.accent2,
        }
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Pie
        ax1 = fig.add_subplot(gs[0, 0])
        tier_counts = self._final_ranking["tier"].value_counts()
        ax1.pie(
            tier_counts.values,
            labels=tier_counts.index,
            autopct="%1.1f%%",
            colors=[colors_map.get(t, sty.accent5) for t in tier_counts.index],
            startangle=90,
        )
        ax1.set_title("Tier Distribution", fontsize=12, fontweight="bold")

        # Scatter per tier
        ax2 = fig.add_subplot(gs[0, 1])
        for tier, color in colors_map.items():
            sub = self._final_ranking.query(f"tier == '{tier}'")
            if not sub.empty:
                ax2.scatter([tier] * len(sub), sub["composite_score"].values, alpha=0.6, s=60, color=color)
        ax2.set_ylabel("Composite Score", fontsize=11)
        ax2.set_title("Composite Scores by Tier", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Top-10 bar
        ax3 = fig.add_subplot(gs[1, :])
        top10 = self._final_ranking.head(10)
        y_pos = np.arange(len(top10))
        ax3.barh(
            y_pos, top10["composite_score"], color=[colors_map.get(t, sty.accent5) for t in top10["tier"]], alpha=0.8
        )
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"Rank {i + 1} ({t})" for i, t in enumerate(top10["tier"])], fontsize=9)
        ax3.set_xlabel("Composite Score", fontsize=11)
        ax3.set_title("Top 10 Parameter Sets", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="x")
        for pos, score in zip(y_pos, top10["composite_score"], strict=True):
            ax3.text(score, pos, f" {score:.3f}", va="center", fontsize=9)

        fig.suptitle("Parameter Stability Pipeline - Final Dashboard", fontsize=16, fontweight="bold", y=0.98)
        sty.apply_mpl(fig)
        _finalize_plot(fig, plt, show=show, fallback_filename="tier_dashboard.png", use_tight_layout=False)
        return fig
