from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.tree import DecisionTreeRegressor, plot_tree

from blackwood.visualization.style import DEFAULT_STYLE, PlotStyle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from blackwood.optimization.optimization import OptunaOptimizer

ParameterSpace = dict[str, tuple[Any, ...] | list[int | float]]
StabilityMetrics = dict[str, dict[str, Any]]


class OptunaStabilityAnalyzer:
    """
    Stability diagnostics for completed Optuna trials.
    """

    def __init__(
        self,
        trials_df: pd.DataFrame,
        param_space: ParameterSpace,
        metric_name: str = "Sharpe Ratio",
        style: PlotStyle = DEFAULT_STYLE,
    ) -> None:
        self.trials_df = trials_df.copy()
        self.param_space = dict(param_space)
        self.metric_name = metric_name
        self.style = style
        self.completed_trials_df: pd.DataFrame | None = None

    @classmethod
    def from_optimizer(cls, optimizer: OptunaOptimizer) -> OptunaStabilityAnalyzer:
        return cls(
            trials_df=optimizer._require_trials_df(),
            param_space=optimizer.param_space,
            metric_name=optimizer.metric_name,
            style=optimizer.style,
        )

    @staticmethod
    def _safe_cv(series: pd.Series) -> float:
        mean_value = series.mean()
        std_value = series.std()
        if pd.isna(mean_value) or mean_value == 0 or pd.isna(std_value):
            return 0.0
        return float((std_value / mean_value) * 100.0)

    def extract_completed_trials(self) -> pd.DataFrame:
        completed = self.trials_df.loc[self.trials_df["state"].astype(str) == "COMPLETE"].copy()

        rename_map = {f"params_{name}": name for name in self.param_space if f"params_{name}" in completed.columns}
        completed = completed.rename(columns=rename_map)

        if "value" not in completed.columns:
            raise KeyError("Trials dataframe does not contain Optuna 'value' column.")

        completed["score"] = pd.to_numeric(completed["value"], errors="coerce")
        completed = completed.dropna(subset=["score"]).reset_index(drop=True)

        for param_name in self.param_space:
            if param_name in completed.columns:
                completed[param_name] = pd.to_numeric(completed[param_name], errors="coerce")

        if completed.empty:
            raise ValueError("No completed Optuna trials with valid score values.")

        self.completed_trials_df = completed
        return completed

    def _resolve_metrics_params(
        self,
        completed_trials: pd.DataFrame,
        metrics_params: Sequence[str] | None,
    ) -> list[str]:
        available_numeric_params = [
            param_name
            for param_name in self.param_space
            if param_name in completed_trials.columns
            and pd.api.types.is_numeric_dtype(completed_trials[param_name])
            and completed_trials[param_name].notna().any()
        ]

        if metrics_params is None:
            if not available_numeric_params:
                raise ValueError("No numeric parameters available for stability analysis.")
            return available_numeric_params

        if not isinstance(metrics_params, Sequence) or isinstance(metrics_params, (str, bytes)):
            raise ValueError("metrics_params must be a sequence of parameter names.")

        resolved = list(metrics_params)
        if not resolved:
            raise ValueError("metrics_params cannot be empty.")

        missing = [name for name in resolved if name not in completed_trials.columns]
        if missing:
            raise ValueError(f"metrics_params contains unknown parameter(s): {missing}")

        non_numeric = [
            name
            for name in resolved
            if not pd.api.types.is_numeric_dtype(completed_trials[name]) or not completed_trials[name].notna().any()
        ]
        if non_numeric:
            raise ValueError(f"metrics_params must contain numeric columns with data. Invalid: {non_numeric}")

        return resolved

    @staticmethod
    def _validate_compare_params(
        compare_params: tuple[str, str] | Sequence[str],
        completed_trials: pd.DataFrame,
    ) -> tuple[str, str]:
        if not isinstance(compare_params, Sequence) or isinstance(compare_params, (str, bytes)):
            raise ValueError("compare_params must be a tuple/list with exactly 2 names.")

        compare_list = list(compare_params)
        if len(compare_list) != 2:
            raise ValueError("compare_params must have length 2.")

        x_param, y_param = compare_list[0], compare_list[1]
        for param_name in (x_param, y_param):
            if param_name not in completed_trials.columns:
                raise ValueError(f"compare_params contains unknown '{param_name}'.")
            if not pd.api.types.is_numeric_dtype(completed_trials[param_name]):
                raise ValueError(f"compare_params '{param_name}' must be numeric.")
            if not completed_trials[param_name].notna().any():
                raise ValueError(f"compare_params '{param_name}' has no valid values.")

        return str(x_param), str(y_param)

    @staticmethod
    def _resolve_score_sensitivity_param(
        score_sensitivity_param: str | None,
        compare_params: tuple[str, str],
        completed_trials: pd.DataFrame,
    ) -> str:
        selected = compare_params[0] if score_sensitivity_param is None else score_sensitivity_param
        if selected not in completed_trials.columns:
            raise ValueError(f"score_sensitivity_param '{selected}' not found.")
        if not pd.api.types.is_numeric_dtype(completed_trials[selected]):
            raise ValueError(f"score_sensitivity_param '{selected}' must be numeric.")
        if not completed_trials[selected].notna().any():
            raise ValueError(f"score_sensitivity_param '{selected}' has no valid values.")
        return selected

    def _choose_discrete_param(
        self,
        completed_trials: pd.DataFrame,
        preferred: str,
        fallback: str | None = None,
        max_unique: int = 12,
    ) -> str | None:
        def is_candidate(param_name: str) -> bool:
            if param_name not in completed_trials.columns:
                return False
            if not pd.api.types.is_numeric_dtype(completed_trials[param_name]):
                return False
            unique_count = completed_trials[param_name].dropna().nunique()
            return 1 < unique_count <= max_unique

        if is_candidate(preferred):
            return preferred
        if fallback is not None and is_candidate(fallback):
            return fallback

        for param_name in self.param_space:
            if is_candidate(param_name):
                return param_name
        return None

    def compute_stability_metrics(
        self,
        completed_trials: pd.DataFrame,
        top_pct: float = 0.10,
        metrics_params: Sequence[str] | None = None,
    ) -> tuple[StabilityMetrics, pd.DataFrame]:
        if completed_trials.empty:
            raise ValueError("No completed trials available for stability analysis.")
        if not (0 < top_pct <= 1):
            raise ValueError(f"top_pct must be in (0, 1], got {top_pct}.")

        resolved_params = self._resolve_metrics_params(
            completed_trials=completed_trials,
            metrics_params=metrics_params,
        )
        n_top = max(1, int(np.ceil(len(completed_trials) * top_pct)))
        top_trials = completed_trials.nlargest(n_top, "score").copy()

        metrics: StabilityMetrics = {}
        for param_name in resolved_params:
            all_values = completed_trials[param_name].dropna()
            top_values = top_trials[param_name].dropna()
            mode_top = top_values.mode()

            metrics[param_name] = {
                "mean_all": float(all_values.mean()) if len(all_values) else np.nan,
                "std_all": float(all_values.std()) if len(all_values) else np.nan,
                "cv_all": self._safe_cv(all_values) if len(all_values) else np.nan,
                "mean_top": float(top_values.mean()) if len(top_values) else np.nan,
                "std_top": float(top_values.std()) if len(top_values) else np.nan,
                "cv_top": self._safe_cv(top_values) if len(top_values) else np.nan,
                "mode_top": float(mode_top.iloc[0]) if not mode_top.empty else None,
            }

        return metrics, top_trials

    def print_stability_report(
        self,
        completed_trials: pd.DataFrame,
        metrics: StabilityMetrics,
        top_trials: pd.DataFrame,
        top_pct: float = 0.10,
    ) -> None:
        threshold = completed_trials["score"].quantile(1 - top_pct)
        pct_label = int(top_pct * 100)

        print("\n" + "=" * 70)
        print("OPTUNA PARAMETER STABILITY REPORT")
        print("=" * 70)
        print("\nDataset Overview:")
        print(f"  Completed trials: {len(completed_trials)}")
        print(f"  Best {self.metric_name}: {completed_trials['score'].max():.4f}")
        print(f"  Median {self.metric_name}: {completed_trials['score'].median():.4f}")
        print(f"  Std Dev: {completed_trials['score'].std():.4f}")
        print(f"  Top {pct_label}% threshold: {threshold:.4f}")

        print("\nParameter Stability Metrics:")
        print(f"{'Parameter':<22} {'Top Mean':<12} {'Top Std':<12} {'Top CV%':<12} {'All CV%':<12}")
        print("-" * 74)
        for param_name, values in metrics.items():
            print(
                f"{param_name:<22} "
                f"{values['mean_top']:<12.2f} "
                f"{values['std_top']:<12.2f} "
                f"{values['cv_top']:<12.1f} "
                f"{values['cv_all']:<12.1f}"
            )

        discrete_params = [
            param_name
            for param_name in self.param_space
            if param_name in top_trials.columns and top_trials[param_name].nunique(dropna=True) <= 10
        ]
        if discrete_params:
            print(f"\nDiscrete Parameter Distribution (Top {pct_label}%):")
            for param_name in discrete_params:
                distribution = top_trials[param_name].value_counts(dropna=False).to_dict()
                print(f"  {param_name}: {distribution}")

        print("\nStability Interpretation:")
        for param_name, values in metrics.items():
            cv_top = values["cv_top"]
            if pd.isna(cv_top):
                label, note = "INSUFFICIENT DATA", "Not enough valid values"
            elif cv_top < 10:
                label, note = "HIGHLY STABLE", "Strong convergence"
            elif cv_top < 30:
                label, note = "MODERATELY STABLE", "Acceptable variance"
            else:
                label, note = "UNSTABLE", "High sensitivity or weak signal"
            print(f"  {param_name}: {label} (CV={cv_top:.1f}%) - {note}")

        print("\n" + "=" * 70 + "\n")

    @staticmethod
    def _assign_performance_tiers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        base_labels = ["Q1 (Worst)", "Q2", "Q3", "Q4 (Best)"]

        if df["score"].nunique(dropna=True) <= 1:
            single_tier = df.copy()
            single_tier["quartile"] = base_labels[0]
            return single_tier, [base_labels[0]]

        q_count = min(4, int(df["score"].nunique(dropna=True)))
        labels = base_labels[:q_count]

        tiered = df.copy()
        tiered["quartile"] = pd.qcut(
            df["score"],
            q=q_count,
            labels=labels,
            duplicates="drop",
        )
        return tiered, labels

    @staticmethod
    def _safe_corr(x: pd.Series, y: pd.Series) -> float:
        frame = pd.concat([x, y], axis=1).dropna()
        if len(frame) < 2:
            return 0.0
        corr = frame.iloc[:, 0].corr(frame.iloc[:, 1])
        return float(abs(corr)) if pd.notna(corr) else 0.0

    @staticmethod
    def _normalize_zero_one(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        v_min = np.nanmin(values)
        v_max = np.nanmax(values)
        if not np.isfinite(v_min) or not np.isfinite(v_max) or np.isclose(v_min, v_max):
            return np.ones_like(values) * 0.5
        return (values - v_min) / (v_max - v_min)

    @staticmethod
    def _std_95_ci(std_value: float, sample_size: int) -> float:
        if sample_size <= 1 or not np.isfinite(std_value):
            return 0.0
        return float(1.96 * std_value / np.sqrt(max(2 * (sample_size - 1), 1)))

    def _build_param_bins(
        self,
        series: pd.Series,
        max_bins: int = 6,
    ) -> pd.Series:
        clean = series.dropna()
        if clean.empty:
            return pd.Series(index=series.index, dtype=object)

        unique_count = clean.nunique()
        if unique_count <= max_bins:
            return series.round(6).astype(str)

        n_bins = min(max_bins, unique_count)
        binned = pd.qcut(series, q=n_bins, duplicates="drop")
        return binned.astype(str)

    def _fit_importance_tree(
        self,
        completed_trials: pd.DataFrame,
        param_names: Sequence[str],
    ) -> tuple[DecisionTreeRegressor | None, pd.Series]:
        if not param_names:
            return None, pd.Series(dtype=float)

        feature_frame = completed_trials[list(param_names)].copy()
        numeric_features = [name for name in param_names if pd.api.types.is_numeric_dtype(feature_frame[name])]
        if not numeric_features:
            return None, pd.Series(dtype=float)

        x = feature_frame[numeric_features].copy()
        for col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].fillna(x[col].median())

        y = pd.to_numeric(completed_trials["score"], errors="coerce")
        mask = y.notna()
        x = x.loc[mask]
        y = y.loc[mask]
        if len(x) < 20:
            return None, pd.Series(0.0, index=numeric_features, dtype=float)

        min_samples_leaf = max(5, min(20, len(x) // 10))
        tree = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=12,
            random_state=42,
        )
        tree.fit(x, y)

        importances = pd.Series(tree.feature_importances_, index=numeric_features)
        importances = importances.sort_values(ascending=False)
        return tree, importances

    def _resolve_top_three_params(
        self,
        importances: pd.Series,
        fallback_params: Sequence[str],
    ) -> list[str]:
        ordered = [name for name in importances.index if importances[name] > 0]
        for name in fallback_params:
            if name not in ordered:
                ordered.append(name)
        return ordered[:3]

    def _build_radar_matrix(
        self,
        completed_trials: pd.DataFrame,
        top_trials: pd.DataFrame,
        param_names: Sequence[str],
        metrics: StabilityMetrics,
    ) -> pd.DataFrame:
        records: list[dict[str, float]] = []
        for param_name in param_names:
            if param_name not in completed_trials.columns:
                continue

            all_values = completed_trials[param_name].dropna()
            top_values = top_trials[param_name].dropna()
            if all_values.empty:
                continue

            grouped_score = completed_trials[[param_name, "score"]].dropna().groupby(param_name)["score"].mean()
            score_cv = self._safe_cv(grouped_score) if not grouped_score.empty else np.nan
            score_cv_stability = 1.0 / (1.0 + (score_cv / 100.0)) if np.isfinite(score_cv) else 0.0

            cv_top = metrics.get(param_name, {}).get("cv_top", np.nan)
            cv_stability = 1.0 / (1.0 + (float(cv_top) / 100.0)) if np.isfinite(cv_top) else 0.0

            top_presence = 0.0
            if not top_values.empty:
                top_presence = float(top_values.value_counts(normalize=True).max())

            corr_abs = self._safe_corr(completed_trials[param_name], completed_trials["score"])

            all_range = float(all_values.max() - all_values.min()) if len(all_values) > 1 else 0.0
            top_range = float(top_values.max() - top_values.min()) if len(top_values) > 1 else 0.0
            range_util = (top_range / all_range) if all_range > 0 else 0.0
            range_util = float(np.clip(range_util, 0.0, 1.0))

            records.append(
                {
                    "parameter": param_name,
                    "cv_sharpe_stability": float(np.clip(score_cv_stability, 0.0, 1.0)),
                    "cv_stability": float(np.clip(cv_stability, 0.0, 1.0)),
                    "top_presence": float(np.clip(top_presence, 0.0, 1.0)),
                    "abs_corr_sharpe": float(np.clip(corr_abs, 0.0, 1.0)),
                    "range_utilization": range_util,
                }
            )

        return pd.DataFrame(records)

    def _plot_radar_fingerprint(
        self,
        ax: Axes,
        radar_df: pd.DataFrame,
        importance_map: dict[str, float],
    ) -> None:
        axis_labels = [
            "CV Sharpe Stability",
            "CV Stability",
            "Top 10% Presence",
            "|Corr| with Score",
            "Range Utilization",
        ]
        axis_keys = [
            "cv_sharpe_stability",
            "cv_stability",
            "top_presence",
            "abs_corr_sharpe",
            "range_utilization",
        ]
        theta = np.linspace(0, 2 * np.pi, len(axis_labels), endpoint=False)
        theta_closed = np.r_[theta, theta[0]]

        ax.fill_between(theta_closed, 0.65, 1.0, color=self.style.accent3, alpha=0.10)
        ax.fill_between(theta_closed, 0.0, 0.35, color=self.style.accent2, alpha=0.10)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(theta)
        ax.set_xticklabels(axis_labels, fontsize=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax.set_title("Volatility Fingerprint Radar", pad=16)

        if radar_df.empty:
            ax.text(0.5, 0.5, "No radar data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(radar_df))))
        for idx, (_, row) in enumerate(radar_df.iterrows()):
            values = np.array([float(row[key]) for key in axis_keys], dtype=float)
            values = np.clip(values, 0.0, 1.0)
            values_closed = np.r_[values, values[0]]

            pname = str(row["parameter"])
            importance = float(importance_map.get(pname, 0.0))
            line_width = 1.5 + 4.0 * importance

            ax.plot(
                theta_closed,
                values_closed,
                color=colors[idx % len(colors)],
                linewidth=line_width,
                alpha=0.95,
                label=pname,
            )
            ax.fill(theta_closed, values_closed, color=colors[idx % len(colors)], alpha=0.05)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)

    def _plot_performance_tree(
        self,
        ax: Axes,
        tree: DecisionTreeRegressor | None,
        feature_names: Sequence[str],
        completed_trials: pd.DataFrame,
    ) -> None:
        if tree is None or len(feature_names) == 0:
            ax.text(
                0.5,
                0.5,
                "Not enough data to fit decision tree",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Performance Decomposition Tree")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        plot_tree(
            tree,
            feature_names=list(feature_names),
            filled=True,
            rounded=True,
            impurity=False,
            proportion=False,
            ax=ax,
            fontsize=8,
        )
        ax.set_title(
            f"Performance Decomposition Tree\n"
            f"mu={completed_trials['score'].mean():.3f}, "
            f"sigma={completed_trials['score'].std():.3f}, "
            f"n={len(completed_trials)}",
            fontsize=11,
        )

    def _plot_param_impact_box_scatter(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        top_trials: pd.DataFrame,
        param_name: str,
    ) -> None:
        data = completed_trials[[param_name, "score"]].dropna().copy()
        if data.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{param_name} Impact")
            return

        data["bin"] = self._build_param_bins(data[param_name])
        groups = [grp["score"].to_numpy() for _, grp in data.groupby("bin", sort=True)]
        labels = [name for name, _ in data.groupby("bin", sort=True)]
        positions = np.arange(1, len(groups) + 1)

        bp = ax.boxplot(groups, positions=positions, widths=0.55, patch_artist=True)
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, max(1, len(groups))))
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[idx % len(colors)])
            patch.set_alpha(0.45)

        rng = np.random.default_rng(42)
        for idx, group_values in enumerate(groups, start=1):
            jitter = rng.uniform(-0.15, 0.15, size=len(group_values))
            ax.scatter(
                np.full(len(group_values), idx) + jitter,
                group_values,
                s=18,
                alpha=0.25,
                color=self.style.accent1,
                linewidth=0,
            )

        top_map = top_trials.set_index("number", drop=False) if "number" in top_trials.columns else top_trials
        _ = top_map  # keep lint clean while preserving local future use

        for idx, group_values in enumerate(groups, start=1):
            ax.text(
                idx,
                np.nanmax(group_values) if len(group_values) else 0.0,
                f"n={len(group_values)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.axhline(data["score"].mean(), linestyle="--", color=self.style.accent6, linewidth=1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.metric_name)
        ax.set_title(f"{param_name} Impact")

    def _plot_param_impact_violin(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        param_name: str,
    ) -> None:
        data = completed_trials[[param_name, "score"]].dropna().copy()
        if data.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{param_name} Impact")
            return

        data["bin"] = self._build_param_bins(data[param_name])
        grouped = list(data.groupby("bin", sort=True))
        groups = [grp["score"].to_numpy() for _, grp in grouped]
        labels = [str(name) for name, _ in grouped]
        positions = np.arange(1, len(groups) + 1)

        violin = ax.violinplot(groups, positions=positions, showmeans=False, showmedians=True)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(groups))))
        for idx, body in enumerate(violin["bodies"]):
            body.set_facecolor(colors[idx % len(colors)])
            body.set_edgecolor(self.style.line)
            body.set_alpha(0.55)
        violin["cmedians"].set_color(self.style.accent2)
        violin["cmedians"].set_linewidth(1.3)

        for idx, values in enumerate(groups, start=1):
            if len(values) == 0:
                continue
            q1, med, q3 = np.nanpercentile(values, [25, 50, 75])
            ax.scatter([idx], [med], color=self.style.accent3, s=18, zorder=4)
            ax.vlines(idx, q1, q3, color=self.style.accent3, linewidth=2, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.metric_name)
        ax.set_title(f"{param_name} Impact")

    def _plot_param_impact_swarm(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        top_trials: pd.DataFrame,
        param_name: str,
    ) -> None:
        data = completed_trials[[param_name, "score"]].dropna().copy()
        if data.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{param_name} Impact")
            return

        data["bin"] = self._build_param_bins(data[param_name])
        grouped = list(data.groupby("bin", sort=True))
        groups = [grp["score"].to_numpy() for _, grp in grouped]
        labels = [str(name) for name, _ in grouped]
        positions = np.arange(1, len(groups) + 1)

        bp = ax.boxplot(groups, positions=positions, widths=0.48, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(self.style.accent5)
            patch.set_alpha(0.25)

        score_threshold = completed_trials["score"].quantile(0.90)
        rng = np.random.default_rng(7)
        for idx, (_, grp) in enumerate(grouped, start=1):
            x_jitter = rng.uniform(-0.17, 0.17, size=len(grp))
            top_mask = grp["score"] >= score_threshold
            ax.scatter(
                np.full(len(grp), idx) + x_jitter,
                grp["score"].to_numpy(),
                s=np.where(top_mask, 35, 18),
                alpha=np.where(top_mask, 0.75, 0.30),
                c=np.where(top_mask, self.style.accent3, self.style.accent1),
                linewidth=0,
            )

        top_n = len(top_trials)
        ax.text(
            0.02,
            0.95,
            f"Top trials highlighted (n={top_n})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.metric_name)
        ax.set_title(f"{param_name} Impact")

    def _plot_interaction_hexbin(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        x_param: str,
        y_param: str,
    ) -> None:
        data = completed_trials[[x_param, y_param, "score"]].dropna()
        if len(data) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{x_param} × {y_param}")  # noqa: RUF001
            return

        hb = ax.hexbin(
            data[x_param].to_numpy(),
            data[y_param].to_numpy(),
            C=data["score"].to_numpy(),
            reduce_C_function=np.mean,
            gridsize=18,
            mincnt=2,
            cmap="RdYlGn",
        )
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label(self.metric_name, fontsize=8)

        with contextlib.suppress(Exception):
            ax.tricontour(
                data[x_param].to_numpy(),
                data[y_param].to_numpy(),
                data["score"].to_numpy(),
                levels=[0.5, 1.0, 1.5],
                colors=self.style.line,
                linewidths=1.0,
                alpha=0.7,
            )

        top5 = data.nlargest(min(5, len(data)), "score")
        ax.scatter(
            top5[x_param],
            top5[y_param],
            marker="*",
            s=140,
            color=self.style.accent2,
            edgecolors=self.style.paper_bgcolor,
            linewidth=0.8,
            label="Top 5",
        )
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"{x_param} × {y_param}")  # noqa: RUF001

    def _plot_interaction_scatter_size(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        x_param: str,
        y_param: str,
    ) -> None:
        data = completed_trials[[x_param, y_param, "score"]].dropna()
        if len(data) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{x_param} x {y_param}")
            return

        score = data["score"].to_numpy()
        s_min = np.nanmin(score)
        s_max = np.nanmax(score)
        if np.isclose(s_min, s_max):
            sizes = np.full(len(score), 60.0)
        else:
            sizes = 30.0 + (score - s_min) / (s_max - s_min) * 220.0

        top_mask = data["score"] >= data["score"].quantile(0.90)
        colors = np.where(top_mask, self.style.accent3, self.style.accent5)

        ax.scatter(
            data[x_param].to_numpy(),
            data[y_param].to_numpy(),
            s=sizes,
            c=colors,
            alpha=0.42,
            linewidth=0,
        )
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"{x_param} x {y_param}")

    def _plot_interaction_hist2d(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        x_param: str,
        y_param: str,
    ) -> None:
        data = completed_trials[[x_param, y_param, "score"]].dropna()
        if len(data) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{x_param} x {y_param}")
            return

        x = data[x_param].to_numpy()
        y = data[y_param].to_numpy()
        z = data["score"].to_numpy()

        x_edges = np.histogram_bin_edges(x, bins=10)
        y_edges = np.histogram_bin_edges(y, bins=10)
        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        score_sum, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)
        mean_score = np.divide(
            score_sum,
            counts,
            out=np.full_like(score_sum, np.nan, dtype=float),
            where=counts > 0,
        )

        img = ax.imshow(
            mean_score.T,
            origin="lower",
            cmap="RdYlGn",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(self.metric_name, fontsize=8)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                n_count = int(counts[i, j])
                if n_count > 5:
                    ax.text(
                        x_centers[i],
                        y_centers[j],
                        f"n={n_count}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"{x_param} x {y_param}")

    def _plot_stability_ranking(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        top_trials: pd.DataFrame,
        param_names: Sequence[str],
        metrics: StabilityMetrics,
    ) -> None:
        if not param_names:
            ax.text(0.5, 0.5, "No parameters", transform=ax.transAxes, ha="center", va="center")
            ax.set_title("Parameter Stability Ranking")
            return

        table = []
        for param_name in param_names:
            if param_name not in metrics:
                continue
            std_all = float(metrics[param_name]["std_all"])
            std_top = float(metrics[param_name]["std_top"])
            all_n = int(completed_trials[param_name].notna().sum())
            top_n = int(top_trials[param_name].notna().sum())
            ratio = std_top / std_all if std_all not in (0.0, np.nan) and np.isfinite(std_all) else np.nan
            table.append(
                {
                    "param": param_name,
                    "std_all": std_all,
                    "std_top": std_top,
                    "ci_all": self._std_95_ci(std_all, all_n),
                    "ci_top": self._std_95_ci(std_top, top_n),
                    "ratio": ratio,
                }
            )

        rank_df = pd.DataFrame(table).dropna(subset=["std_all", "std_top"])
        if rank_df.empty:
            ax.text(0.5, 0.5, "No ranking data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title("Parameter Stability Ranking")
            return

        rank_df = rank_df.sort_values("ratio", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(rank_df))
        height = 0.35
        ax.barh(
            y_pos - height / 2,
            rank_df["std_all"],
            xerr=rank_df["ci_all"],
            height=height,
            color=self.style.accent5,
            alpha=0.75,
            label="All Trials",
        )
        ax.barh(
            y_pos + height / 2,
            rank_df["std_top"],
            xerr=rank_df["ci_top"],
            height=height,
            color=self.style.accent3,
            alpha=0.85,
            label="Top Trials",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rank_df["param"])
        ax.set_xlabel("Std (95% CI)")
        ax.set_title("Parameter Stability Ranking")
        ax.legend(fontsize=8)

    def _plot_cv_comparison(
        self,
        ax: Axes,
        param_names: Sequence[str],
        metrics: StabilityMetrics,
    ) -> None:
        if not param_names:
            ax.text(0.5, 0.5, "No parameters", transform=ax.transAxes, ha="center", va="center")
            ax.set_title("Coefficient of Variation Comparison")
            return

        np.arange(len(param_names))
        width = 0.38
        cv_all = [float(metrics[p]["cv_all"]) for p in param_names if p in metrics]
        cv_top = [float(metrics[p]["cv_top"]) for p in param_names if p in metrics]
        labels = [p for p in param_names if p in metrics]

        x_plot = np.arange(len(labels))
        ax.bar(
            x_plot - width / 2,
            cv_all,
            width,
            color=self.style.accent6,
            alpha=0.80,
            label="All Trials",
        )
        ax.bar(
            x_plot + width / 2,
            cv_top,
            width,
            color=self.style.accent3,
            alpha=0.85,
            label="Top Trials",
        )
        ax.axhline(50.0, linestyle="--", color=self.style.accent2, linewidth=1.2, label="CV 50%")
        ax.set_xticks(x_plot)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_title("Coefficient of Variation Comparison")
        ax.legend(fontsize=8)

    def _plot_discrete_distribution(
        self,
        ax: Axes,
        completed_trials: pd.DataFrame,
        top_trials: pd.DataFrame,
    ) -> None:
        discrete_params = [
            name
            for name in self.param_space
            if name in completed_trials.columns
            and pd.api.types.is_numeric_dtype(completed_trials[name])
            and 1 < completed_trials[name].dropna().nunique() <= 10
        ]

        if not discrete_params:
            ax.text(
                0.5,
                0.5,
                "No discrete parameters with low cardinality",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Discrete Parameter Distribution")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        width = 0.35
        x_base = np.arange(len(discrete_params))
        cmap = plt.cm.tab20
        all_categories: list[tuple[str, float]] = []
        for param_name in discrete_params:
            values = completed_trials[param_name].dropna().unique().tolist()
            for val in sorted(values):
                all_categories.append((param_name, float(val)))

        all_bottom = np.zeros(len(discrete_params))
        top_bottom = np.zeros(len(discrete_params))
        legend_added: set = set()

        for idx_param, param_name in enumerate(discrete_params):
            values_all = completed_trials[param_name].dropna()
            values_top = top_trials[param_name].dropna()
            cats = sorted(values_all.unique().tolist())

            for idx_cat, category in enumerate(cats):
                pct_all = float((values_all == category).mean() * 100.0) if len(values_all) else 0.0
                pct_top = float((values_top == category).mean() * 100.0) if len(values_top) else 0.0

                color = cmap((idx_cat % 20) / 19 if 19 else 0)
                label = f"{param_name}={int(category) if float(category).is_integer() else category}"
                legend_label = label if label not in legend_added else None
                if legend_label is not None:
                    legend_added.add(label)

                ax.bar(
                    x_base[idx_param] - width / 2,
                    pct_all,
                    width,
                    bottom=all_bottom[idx_param],
                    color=color,
                    alpha=0.75,
                    label=legend_label,
                )
                ax.bar(
                    x_base[idx_param] + width / 2,
                    pct_top,
                    width,
                    bottom=top_bottom[idx_param],
                    color=color,
                    alpha=0.95,
                )
                all_bottom[idx_param] += pct_all
                top_bottom[idx_param] += pct_top

        ax.set_xticks(x_base)
        ax.set_xticklabels(discrete_params, rotation=10, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Discrete Parameter Distribution")

        left_proxy = Patch(facecolor=self.style.accent5, alpha=0.65, label="All trials (left bar)")
        right_proxy = Patch(facecolor=self.style.accent3, alpha=0.85, label="Top trials (right bar)")
        handles, labels = ax.get_legend_handles_labels()
        handles = [left_proxy, right_proxy, *handles]
        labels = ["All trials (left bar)", "Top trials (right bar)", *labels]
        ax.legend(handles=handles[:12], labels=labels[:12], fontsize=7, loc="upper right")

    def plot_stability_dashboard(
        self,
        completed_trials: pd.DataFrame,
        metrics: StabilityMetrics,
        top_trials: pd.DataFrame,
        compare_params: tuple[str, str] = ("start_hour", "end_hour"),
        metrics_params: Sequence[str] | None = None,
        score_sensitivity_param: str | None = None,
        show: bool = True,
    ) -> Figure:
        del compare_params
        del score_sensitivity_param

        param_names = self._resolve_metrics_params(
            completed_trials=completed_trials,
            metrics_params=list(metrics.keys()) if metrics_params is None else metrics_params,
        )
        if not param_names:
            raise ValueError("No parameter metrics available for plotting.")
        if any(param_name not in metrics for param_name in param_names):
            raise ValueError("metrics is missing one or more metrics_params entries.")

        tree_model, importances = self._fit_importance_tree(completed_trials, param_names)
        if importances.empty:
            importance_map = dict.fromkeys(param_names, 0.0)
        else:
            raw_importance = importances.reindex(param_names).fillna(0.0).to_numpy(dtype=float)
            scaled_importance = self._normalize_zero_one(raw_importance)
            importance_map = {name: float(scaled_importance[idx]) for idx, name in enumerate(param_names)}

        top_three_params = self._resolve_top_three_params(importances, param_names)
        while len(top_three_params) < 3 and len(param_names) > len(top_three_params):
            for param_name in param_names:
                if param_name not in top_three_params:
                    top_three_params.append(param_name)
                if len(top_three_params) == 3:
                    break
        if len(top_three_params) < 3:
            top_three_params = (top_three_params + [top_three_params[0]] * 3)[:3] if top_three_params else []

        radar_df = self._build_radar_matrix(
            completed_trials=completed_trials,
            top_trials=top_trials,
            param_names=param_names,
            metrics=metrics,
        )

        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(
            4,
            3,
            height_ratios=[1.2, 1, 1, 1],
            width_ratios=[1, 1, 1.2],
            hspace=0.3,
            wspace=0.3,
        )

        # Row 1
        ax1 = fig.add_subplot(gs[0, 0], projection="polar")
        self._plot_radar_fingerprint(ax1, radar_df, importance_map)

        ax2 = fig.add_subplot(gs[0, 1:3])
        tree_features = list(importances.index) if not importances.empty else param_names
        self._plot_performance_tree(ax2, tree_model, tree_features, completed_trials)

        # Row 2
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        if len(top_three_params) >= 1:
            self._plot_param_impact_box_scatter(ax3, completed_trials, top_trials, top_three_params[0])
        else:
            ax3.text(0.5, 0.5, "No parameter", transform=ax3.transAxes, ha="center", va="center")
        if len(top_three_params) >= 2:
            self._plot_param_impact_violin(ax4, completed_trials, top_three_params[1])
        else:
            ax4.text(0.5, 0.5, "No parameter", transform=ax4.transAxes, ha="center", va="center")
        if len(top_three_params) >= 3:
            self._plot_param_impact_swarm(ax5, completed_trials, top_trials, top_three_params[2])
        else:
            ax5.text(0.5, 0.5, "No parameter", transform=ax5.transAxes, ha="center", va="center")

        # Row 3
        ax6 = fig.add_subplot(gs[2, 0])
        ax7 = fig.add_subplot(gs[2, 1])
        ax8 = fig.add_subplot(gs[2, 2])
        if len(top_three_params) >= 2:
            self._plot_interaction_hexbin(ax6, completed_trials, top_three_params[0], top_three_params[1])
        else:
            ax6.text(0.5, 0.5, "No pair", transform=ax6.transAxes, ha="center", va="center")
        if len(top_three_params) >= 3:
            self._plot_interaction_scatter_size(ax7, completed_trials, top_three_params[0], top_three_params[2])
            self._plot_interaction_hist2d(ax8, completed_trials, top_three_params[1], top_three_params[2])
        else:
            ax7.text(0.5, 0.5, "No pair", transform=ax7.transAxes, ha="center", va="center")
            ax8.text(0.5, 0.5, "No pair", transform=ax8.transAxes, ha="center", va="center")

        # Row 4
        ax9 = fig.add_subplot(gs[3, 0])
        ax10 = fig.add_subplot(gs[3, 1])
        ax11 = fig.add_subplot(gs[3, 2])
        self._plot_stability_ranking(ax9, completed_trials, top_trials, param_names, metrics)
        self._plot_cv_comparison(ax10, param_names, metrics)
        self._plot_discrete_distribution(ax11, completed_trials, top_trials)

        fig.suptitle("Comprehensive Parameter Stability Analysis", fontsize=20, fontweight="bold", y=0.995)
        self.style.apply_mpl(fig)
        fig.subplots_adjust(top=0.96)

        if show:
            plt.show()
        return fig

    def run_stability_test(
        self,
        top_pct: float = 0.10,
        compare_params: tuple[str, str] = ("start_hour", "end_hour"),
        metrics_params: Sequence[str] | None = None,
        score_sensitivity_param: str | None = None,
        show_plot: bool = True,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, StabilityMetrics, pd.DataFrame, Figure]:
        completed_trials = (
            self.completed_trials_df.copy() if self.completed_trials_df is not None else self.extract_completed_trials()
        )
        resolved_metrics_params = self._resolve_metrics_params(
            completed_trials=completed_trials,
            metrics_params=metrics_params,
        )

        metrics, top_trials = self.compute_stability_metrics(
            completed_trials=completed_trials,
            top_pct=top_pct,
            metrics_params=resolved_metrics_params,
        )

        if verbose:
            self.print_stability_report(
                completed_trials=completed_trials,
                metrics=metrics,
                top_trials=top_trials,
                top_pct=top_pct,
            )

        fig = self.plot_stability_dashboard(
            completed_trials=completed_trials,
            metrics=metrics,
            top_trials=top_trials,
            compare_params=compare_params,
            metrics_params=resolved_metrics_params,
            score_sensitivity_param=score_sensitivity_param,
            show=show_plot,
        )

        return completed_trials, metrics, top_trials, fig
