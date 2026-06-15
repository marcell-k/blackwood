# Standard library
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt

# Third-party: Core
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_STATE

# First-party
from src.visualization.style import DEFAULT_STYLE

# Utilities
from tqdm import tqdm


@dataclass
class MDAConfig:
    """Configuration for MDA computation."""

    cv: int = 5
    scoring: str = "neg_log_loss"
    n_perm: int = 1
    n_repeats: int = 3
    random_state: int | None = None
    disable_progress: bool = False


def _as_array_2d(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert input to 2D numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(copy=False)
    return np.asarray(X)


def _as_array_1d(y: pd.Series | np.ndarray) -> np.ndarray:
    """Convert input to 1D numpy array."""
    if isinstance(y, pd.Series):
        return y.to_numpy(copy=False)
    return np.asarray(y)


def _slice_weights(sample_weights: pd.Series | np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    """Extract weights for given indices."""
    if sample_weights is None:
        return None
    if isinstance(sample_weights, pd.Series):
        return sample_weights.iloc[idx].to_numpy(copy=False)
    return np.asarray(sample_weights)[idx]


def _score_classifier(
    clf,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    scoring: str,
    sample_weight: np.ndarray | None = None,
) -> float:
    X_arr = _as_array_2d(X)
    y_arr = _as_array_1d(y)
    if scoring == "neg_log_loss":
        prob = clf.predict_proba(X_arr)
        return -log_loss(
            y_arr,
            prob,
            labels=getattr(clf, "classes_", None),
            sample_weight=sample_weight,
        )
    return accuracy_score(y_arr, clf.predict(X_arr), sample_weight=sample_weight)


def _importance_stats(importance_matrix: np.ndarray, labels: Iterable[str]) -> pd.DataFrame:
    imp_df = pd.DataFrame(importance_matrix, columns=list(labels))
    n_obs = max(imp_df.shape[0], 1)
    return pd.DataFrame(
        {
            "mean": imp_df.mean(axis=0),
            "std": imp_df.std(axis=0) / np.sqrt(n_obs),
        }
    )


class RMTCorrelationProcessor:
    """
    Random Matrix Theory-based correlation matrix processor.
    Implements Marcenko-Pastur denoising and market-mode detoning.
    """

    def __init__(self, remove_market_mode: bool = True):
        self.remove_market_mode = remove_market_mode
        self.mp_bounds = None
        self.n_signal_eigvals = 0
        self.clusters = None
        self.plot_style = DEFAULT_STYLE
        self.linkage_matrix = None
        self.best_silhouette = None
        self.cluster_silhouette = None
        self.feature_silhouette = None

    def _zscore(self, X: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization after forward-fill."""
        X_clean = X.ffill().fillna(0.0)
        means = X_clean.mean()
        stds = X_clean.std().replace(0.0, 1.0)
        return ((X_clean - means) / stds).clip(-10.0, 10.0)

    def _compute_score(
        self,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None,
        scoring: str,
    ) -> float:
        """Delegate to module-level scoring function."""
        return _score_classifier(clf, X, y, scoring=scoring, sample_weight=weights)

    def _create_cv_generator(self, cv: int, n_repeats: int, random_state: int | None):
        """Create RepeatedStratifiedKFold cross-validator."""
        return RepeatedStratifiedKFold(
            n_splits=cv,
            n_repeats=n_repeats,
            random_state=random_state or 42,
        )

    @staticmethod
    def _validate_mda_config(config: MDAConfig) -> None:
        if config.scoring not in {"neg_log_loss", "accuracy"}:
            raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'")
        if config.n_perm < 1:
            raise ValueError("n_perm must be >= 1")

    def _compute_grouped_mda(
        self,
        clf,
        X: pd.DataFrame,
        y: pd.Series,
        group_names: list[str],
        group_indices: list[np.ndarray],
        sample_weights: pd.Series | None,
        config: MDAConfig,
        progress_desc: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        cv_gen = self._create_cv_generator(config.cv, config.n_repeats, config.random_state)
        total_folds = config.cv * config.n_repeats

        scr0 = np.empty(total_folds, dtype=float)
        scr1 = np.empty((total_folds, len(group_names)), dtype=float)

        base_rng = np.random.default_rng(config.random_state)
        X_np = X.to_numpy(copy=False)
        y_np = y.to_numpy(copy=False)

        cv_iterator = enumerate(cv_gen.split(X_np, y_np))
        if not config.disable_progress:
            cv_iterator = tqdm(
                cv_iterator,
                total=total_folds,
                desc=progress_desc,
                unit="fold",
                leave=False,
            )

        for fold_idx, (train_idx, test_idx) in cv_iterator:
            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_test, y_test = X_np[test_idx], y_np[test_idx]
            w_train = _slice_weights(sample_weights, train_idx)
            w_test = _slice_weights(sample_weights, test_idx)

            fit = clone(clf)
            fit.fit(X_train, y_train, sample_weight=w_train)
            scr0[fold_idx] = self._compute_score(fit, X_test, y_test, w_test, config.scoring)

            fold_rng = np.random.default_rng(base_rng.integers(0, 2**63 - 1))
            X_test_perm = X_test.copy()

            for group_idx, feat_idx in enumerate(group_indices):
                feat_idx = np.asarray(feat_idx, dtype=int)
                original_block = X_test[:, feat_idx].copy()
                perm_scores = np.empty(config.n_perm, dtype=float)

                for perm_i in range(config.n_perm):
                    X_test_perm[:, feat_idx] = original_block
                    # Use explicit assignment because integer-array indexing returns a copy.
                    row_perm = fold_rng.permutation(X_test_perm.shape[0])
                    X_test_perm[:, feat_idx] = original_block[row_perm, :]
                    perm_scores[perm_i] = self._compute_score(fit, X_test_perm, y_test, w_test, config.scoring)

                X_test_perm[:, feat_idx] = original_block
                scr1[fold_idx, group_idx] = perm_scores.mean()

        scr0_series = pd.Series(scr0, index=range(total_folds), dtype=float)
        scr1_df = pd.DataFrame(scr1, index=range(total_folds), columns=group_names, dtype=float)

        baseline_scale = np.maximum(np.abs(scr0)[:, None], 1e-12)
        imp_matrix = (scr0[:, None] - scr1) / baseline_scale
        imp = _importance_stats(imp_matrix, group_names)

        return imp, scr0_series, scr1_df

    def _style_table(
        self,
        table,
        df_stats: pd.DataFrame,
        highlight_col: str | None = None,
        highlight_thresholds: list[tuple[float, str]] | None = None,
    ) -> None:
        """Style matplotlib table with header highlighting and optional value-based coloring."""
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Header row
        for i in range(len(df_stats.columns)):
            table[(0, i)].set_facecolor(self.plot_style.accent1)
            table[(0, i)].set_text_props(weight="bold", color=self.plot_style.paper_bgcolor)

        # Data rows with alternating backgrounds
        for i in range(1, len(df_stats) + 1):
            for j in range(len(df_stats.columns)):
                bg = self.plot_style.plot_bgcolor if i % 2 == 0 else self.plot_style.paper_bgcolor
                table[(i, j)].set_facecolor(bg)
                table[(i, j)].set_text_props(color=self.plot_style.font_color)

                # Apply threshold-based coloring if requested
                if highlight_col and df_stats.columns[j] == highlight_col and highlight_thresholds:
                    val_str = df_stats.iloc[i - 1][highlight_col]
                    val = float(val_str.split(" ±")[0]) if " ±" in str(val_str) else float(val_str)
                    for threshold, color in highlight_thresholds:
                        if val > threshold:
                            table[(i, j)].set_facecolor(color)
                            table[(i, j)].set_alpha(0.3)
                            break

    def _sanitize_correlation_to_distance(self, corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Convert correlation matrix to distance matrix (handling NaN/inf)."""
        feature_names = corr.index.to_numpy()
        corr_arr = np.clip(corr.to_numpy(dtype=np.float64, copy=True), -1.0, 1.0)
        np.fill_diagonal(corr_arr, 1.0)
        corr_arr = np.where(np.isfinite(corr_arr), corr_arr, 0.0)

        dist = np.sqrt(0.5 * (1.0 - corr_arr))
        np.clip(dist, 0.0, None, out=dist)
        np.fill_diagonal(dist, 0.0)
        return dist, feature_names

    def denoise_detone_corr(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        r"""
        Denoise + detone correlation matrix using Random Matrix Theory.

        Uses Marcenko-Pastur bounds to identify noise eigenvalues:
        $$\lambda_{\pm} = (1 \pm \sqrt{1/q})^2$$ where $q = T/N$.
        """
        Xz = self._zscore(X)
        T, N = Xz.shape
        q = T / N

        corr = np.corrcoef(Xz.values, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(corr)

        idx = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        lambda_max = (1 + np.sqrt(1 / q)) ** 2
        lambda_min = (1 - np.sqrt(1 / q)) ** 2
        self.mp_bounds = (lambda_min, lambda_max)

        eigvals_dn = np.where(eigvals > lambda_max, eigvals, 0.0)
        self.n_signal_eigvals = np.sum(eigvals_dn > 0)

        corr_dn = eigvecs @ np.diag(eigvals_dn) @ eigvecs.T

        if self.remove_market_mode and eigvals_dn[0] > 0:
            market = np.outer(eigvecs[:, 0], eigvecs[:, 0]) * eigvals_dn[0]
            corr_dt = corr_dn - market
        else:
            corr_dt = corr_dn

        d = np.sqrt(np.diag(corr_dt))
        d = np.where(d == 0, 1, d)
        corr_dt = np.clip(corr_dt / np.outer(d, d), -1.0, 1.0)

        info = {
            "T": T,
            "N": N,
            "q": q,
            "mp_bounds": self.mp_bounds,
            "n_signal_eigvals": self.n_signal_eigvals,
            "eigvals_original": eigvals,
            "eigvals_denoised": eigvals_dn,
            "corr_original": corr,
        }
        return pd.DataFrame(corr_dt, index=X.columns, columns=X.columns), info

    def onc_clustering(self, corr: pd.DataFrame, min_cluster_size: int = 2) -> dict[int, list[str]]:
        """Find optimal number of clusters by maximizing silhouette score."""
        dist, feature_names = self._sanitize_correlation_to_distance(corr)
        n_features = dist.shape[0]

        Z = linkage(squareform(dist, checks=False), method="average")
        self.linkage_matrix = Z

        max_clusters = int(np.sqrt(n_features))
        best_score, best_k = -np.inf, 2

        for k in range(2, min(max_clusters, n_features)):
            labels = fcluster(Z, k, criterion="maxclust")
            score = silhouette_score(dist, labels, metric="precomputed")
            if score > best_score:
                best_score, best_k = score, k

        self.best_silhouette = best_score
        labels = fcluster(Z, best_k, criterion="maxclust")

        clusters = {}
        for label, name in zip(labels, feature_names, strict=True):
            clusters.setdefault(label, []).append(name)

        self.clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
        return self.clusters

    def visualize_transformation(self, X: pd.DataFrame):
        """Create comprehensive visualization of RMT transformation."""
        corr_transformed, info = self.denoise_detone_corr(X)
        corr_original = info["corr_original"]

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        for idx, (corr_data, title) in enumerate(
            [(corr_original, "Original Correlation Matrix"), (corr_transformed.values, "Denoised + Detoned Matrix")]
        ):
            ax = fig.add_subplot(gs[0, idx])
            sns.heatmap(
                corr_data,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={"label": "Correlation"},
                ax=ax,
            )
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")

        ax3 = fig.add_subplot(gs[1, 0])
        x = np.arange(len(info["eigvals_original"]))

        ax3.bar(x, info["eigvals_original"], alpha=0.7, label="Original", color=self.plot_style.accent1)
        ax3.bar(x, info["eigvals_denoised"], alpha=0.7, label="Denoised", color=self.plot_style.accent2)
        ax3.axhline(
            info["mp_bounds"][1],
            color=self.plot_style.accent4,
            linestyle="--",
            linewidth=2,
            label=f"MP Upper ({info['mp_bounds'][1]:.3f})",
        )
        ax3.axhline(
            info["mp_bounds"][0],
            color=self.plot_style.accent6,
            linestyle="--",
            linewidth=2,
            label=f"MP Lower ({info['mp_bounds'][0]:.3f})",
        )
        ax3.set_xlabel("Eigenvalue Index", fontsize=12)
        ax3.set_ylabel("Eigenvalue Magnitude", fontsize=12)
        ax3.set_title("Eigenvalue Spectrum", fontsize=14, fontweight="bold")
        ax3.legend(loc="upper right")
        ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        upper_tri_idx = np.triu_indices_from(corr_original, k=1)

        for corr_vals, label, color in [
            (corr_original[upper_tri_idx], "Original", self.plot_style.accent1),
            (corr_transformed.values[upper_tri_idx], "Transformed", self.plot_style.accent2),
        ]:
            ax4.hist(corr_vals, bins=40, alpha=0.6, label=label, color=color, edgecolor="black", linewidth=0.5)

        ax4.axvline(0, color=self.plot_style.line, linestyle="-", linewidth=1, alpha=0.5)
        ax4.set_xlabel("Correlation", fontsize=12)
        ax4.set_ylabel("Frequency", fontsize=12)
        ax4.set_title("Correlation Distribution", fontsize=14, fontweight="bold")
        ax4.legend()
        ax4.grid(alpha=0.3, axis="y")

        stats_text = (
            f"Matrix: {info['T']} × {info['N']} (q={info['q']:.2f})\n"
            f"Signal eigenvalues: {info['n_signal_eigvals']}\n"
            f"Variance explained: {info['eigvals_denoised'][:5].sum() / info['eigvals_original'].sum():.1%}"
        )
        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor=self.plot_style.accent6, alpha=0.5),
        )

        self.plot_style.apply_mpl(fig)
        return fig, (corr_transformed, info)

    def visualize_onc_clustering(self, X: pd.DataFrame):
        """Visualize ONC clustering results with dendrogram."""
        corr_transformed, _ = self.denoise_detone_corr(X)
        clusters = self.onc_clustering(corr_transformed)

        fig, ax1 = plt.subplots(1, 1, figsize=(18, 5.5))
        dendrogram(
            self.linkage_matrix,
            labels=corr_transformed.columns,
            ax=ax1,
            leaf_font_size=8,
            color_threshold=0,
            above_threshold_color=self.plot_style.accent1,
        )
        ax1.set_title(f"Hierarchical Clustering Dendrogram (Optimal k={len(clusters)})", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Feature", fontsize=12)
        ax1.set_ylabel("Distance", fontsize=12)

        summary_text = (
            f"Optimal clusters: {len(clusters)}\n"
            f"Silhouette score: {self.best_silhouette:.3f}\n"
            f"Total features: {sum(len(v) for v in clusters.values())}"
        )
        ax1.text(
            0.995,
            0.02,
            summary_text,
            transform=ax1.transAxes,
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor=self.plot_style.accent3, alpha=0.5),
            ha="right",
            va="bottom",
            color=self.plot_style.font_color,
        )

        self.plot_style.apply_mpl(fig)
        fig.tight_layout()
        return fig, clusters

    def feature_importance_mda(
        self,
        clf,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        config: MDAConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Calculate the Mean Decrease Accuracy (MDA) for feature importance.

        This implements Snippet 6.3 from Marcos López de Prado's "Advances in
        Financial Machine Learning". It calculates the expected drop in model
        performance when a specific feature's values are randomly permuted.

        Formula:
            MDA_j = E [ (S_baseline - S_perm_j) / |S_baseline| ]

        Where:
            - MDA_j      : The importance score for feature `j`.
            - E          : Expectation (average) across all cross-validation folds.
            - S_baseline : The out-of-sample score of the baseline model.
            - S_perm_j   : The out-of-sample score after shuffling feature `j`.
        """
        config = config or MDAConfig()
        self._validate_mda_config(config)
        feature_names = X.columns.tolist()
        group_indices = [np.array([i], dtype=int) for i in range(len(feature_names))]
        return self._compute_grouped_mda(
            clf=clf,
            X=X,
            y=y,
            group_names=feature_names,
            group_indices=group_indices,
            sample_weights=sample_weights,
            config=config,
            progress_desc="MDA CV Folds",
        )

    def cluster_importance_mda(
        self,
        clf,
        X: pd.DataFrame,
        y: pd.Series,
        clusters: dict[int, list[str]],
        sample_weights: pd.Series | None = None,
        config: MDAConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        r"""
        Clustered Mean Decrease Accuracy - shuffle all features within a cluster together.
        """
        config = config or MDAConfig(n_perm=20, n_repeats=1)
        self._validate_mda_config(config)

        cluster_ids = list(clusters.keys())
        cluster_cols = [f"C{cid}" for cid in cluster_ids]
        feat_to_idx = {feat: idx for idx, feat in enumerate(X.columns)}
        group_indices = [np.array([feat_to_idx[f] for f in clusters[cid]], dtype=int) for cid in cluster_ids]
        imp, scr0, scr1 = self._compute_grouped_mda(
            clf=clf,
            X=X,
            y=y,
            group_names=cluster_cols,
            group_indices=group_indices,
            sample_weights=sample_weights,
            config=config,
            progress_desc="Cluster MDA CV Folds",
        )
        imp.index = imp.index.astype(str)
        return imp, scr0, scr1

    @staticmethod
    def select_clusters_by_mda(
        clusters: dict[int, list[str]],
        cluster_mda: pd.DataFrame,
        min_mean_importance: float = 0.0,
        top_k: int | None = None,
    ) -> tuple[dict[int, list[str]], list[str]]:
        """Select clusters based on MDA importance thresholds."""
        mda_sorted = cluster_mda.sort_values("mean", ascending=False)
        mda_sorted = mda_sorted[mda_sorted["mean"] > min_mean_importance]
        if top_k is not None:
            mda_sorted = mda_sorted.head(top_k)

        keep_ids = [int(str(idx)[1:]) if str(idx).startswith("C") else int(idx) for idx in mda_sorted.index]
        selected_clusters = {cid: clusters[cid] for cid in keep_ids if cid in clusters}
        selected_features = [f for feats in selected_clusters.values() for f in feats]
        return selected_clusters, selected_features

    def feature_importance_mda_within_clusters(
        self,
        clf,
        X: pd.DataFrame,
        y: pd.Series,
        selected_clusters: dict[int, list[str]],
        sample_weights: pd.Series | None = None,
        config: MDAConfig | None = None,
    ) -> dict[int, pd.DataFrame]:
        """Compute MDA importance for features within each cluster."""
        config = config or MDAConfig()
        out = {}
        for cluster_id, features in selected_clusters.items():
            imp_df, baseline_scores, _ = self.feature_importance_mda(
                clf=clf,
                X=X[features],
                y=y,
                sample_weights=sample_weights,
                config=config,
            )
            imp_df = imp_df.sort_values("mean", ascending=False)
            imp_df["cluster_id"] = cluster_id
            imp_df["baseline_score_mean"] = float(baseline_scores.mean())
            out[cluster_id] = imp_df
        return out

    @staticmethod
    def plot_mda_barh(
        imp: pd.DataFrame,
        title: str = "MDA results",
        top_n: int | None = None,
        sort_ascending: bool = True,
        figsize: tuple | None = None,
        mean_col: str = "mean",
        std_col: str = "std",
    ) -> plt.Figure:
        """Horizontal bar plot with error bars (Figure 6.3 style)."""
        plot_style = DEFAULT_STYLE

        if mean_col not in imp.columns or std_col not in imp.columns:
            raise ValueError(f"imp must contain columns '{mean_col}' and '{std_col}'")

        df = imp[[mean_col, std_col]].replace([np.inf, -np.inf], np.nan).dropna()
        df = df.sort_values(mean_col, ascending=sort_ascending)

        if top_n is not None:
            df = df.reindex(df[mean_col].abs().sort_values(ascending=False).head(top_n).index)
            df = df.sort_values(mean_col, ascending=sort_ascending)

        figsize = figsize or (7.2, max(2.6, 0.13 * df.shape[0] + 0.8))
        fig, ax = plt.subplots(figsize=figsize)

        y = np.arange(df.shape[0])
        ax.barh(y, df[mean_col].values, color=plot_style.accent1, alpha=0.85, edgecolor="none")
        ax.errorbar(
            df[mean_col].values,
            y,
            xerr=df[std_col].values,
            fmt="none",
            ecolor=plot_style.accent4,
            elinewidth=1.5,
            capsize=2,
            alpha=0.9,
        )
        ax.axvline(0.0, color=plot_style.line, linewidth=1.0, alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(df.index.astype(str))
        ax.invert_yaxis()

        # Apply global style first, then set slightly smaller typography for dense MDA plots.
        plot_style.apply_mpl(fig=fig, ax=ax)
        ax.set_title(
            title,
            fontsize=max(10, plot_style.title_size - 1),
            fontweight="bold",
            color=plot_style.font_color,
        )
        ax.set_xlabel("Importance (MDA)", fontsize=max(9, plot_style.font_size - 1))
        ax.tick_params(axis="x", labelsize=max(8, plot_style.font_size - 1))
        ax.tick_params(axis="y", labelsize=max(8, plot_style.font_size - 1))
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", alpha=0.25, color=plot_style.grid, linewidth=0.5)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_within_cluster_mda(
        within_cluster: dict[int, pd.DataFrame],
        title: str = "Within-cluster MDA (top features)",
        top_n_total: int = 40,
    ) -> plt.Figure:
        """Plot top features across all selected clusters."""
        frames = []
        for cid, df in within_cluster.items():
            tmp = df.copy()
            if "cluster_id" not in tmp.columns:
                tmp["cluster_id"] = cid
            frames.append(tmp)

        all_imp = pd.concat(frames, axis=0)
        all_imp.index = [f"C{int(all_imp.loc[k, 'cluster_id'])}:{k}" for k in all_imp.index]
        return RMTCorrelationProcessor.plot_mda_barh(
            imp=all_imp[["mean", "std"]], title=title, top_n=top_n_total, sort_ascending=False
        )

    def run_full_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        clf,
        sample_weights: pd.Series | None = None,
        random_state: int | None = None,
        include_clustering_figure: bool = False,
    ) -> dict:
        """Execute complete RMT-based feature importance pipeline."""
        sample_weights = self.transform_rrr_to_sample_weights(y["RiskRewardRatio"])
        fig_transformation, (corr_transformed, rmt_info) = self.visualize_transformation(X)
        y_labels = y["meta_label"]

        if include_clustering_figure:
            fig_clustering, clusters = self.visualize_onc_clustering(X)
        else:
            clusters = self.onc_clustering(corr_transformed)
            fig_clustering = None

        feature_config = MDAConfig(cv=5, scoring="neg_log_loss", n_perm=100, random_state=random_state)
        feat_mda, scr0_feature, scr1_feature = self.feature_importance_mda(
            clf=clf, X=X, y=y_labels, sample_weights=sample_weights, config=feature_config
        )
        fig_feature_mda = self.plot_mda_barh(
            imp=feat_mda, title="Feature MDA (Unclustered)", top_n=50, sort_ascending=False
        )

        cluster_config = MDAConfig(cv=5, scoring="neg_log_loss", n_perm=20, n_repeats=3, random_state=random_state)
        cluster_mda, scr0_cluster, scr1_cluster = self.cluster_importance_mda(
            clf=clf, X=X, y=y_labels, clusters=clusters, sample_weights=sample_weights, config=cluster_config
        )
        fig_cluster_mda = self.plot_mda_barh(imp=cluster_mda, title="Cluster MDA", sort_ascending=False)

        selected_clusters, selected_features = self.select_clusters_by_mda(
            clusters=clusters, cluster_mda=cluster_mda, min_mean_importance=-1.0, top_k=None
        )

        within_config = MDAConfig(cv=5, scoring="neg_log_loss", n_perm=20, random_state=random_state)
        within_cluster_mda = self.feature_importance_mda_within_clusters(
            clf=clf,
            X=X,
            y=y_labels,
            selected_clusters=selected_clusters,
            sample_weights=sample_weights,
            config=within_config,
        )
        fig_within_cluster_mda = self.plot_within_cluster_mda(
            within_cluster=within_cluster_mda, title="Within-Cluster MDA (Top Features)", top_n_total=60
        )

        figures = {
            "transformation": fig_transformation,
            "feature_mda": fig_feature_mda,
            "cluster_mda": fig_cluster_mda,
            "within_cluster_mda": fig_within_cluster_mda,
        }
        if fig_clustering is not None:
            figures["clustering"] = fig_clustering

        return {
            "figures": figures,
            "importance": {
                "feature_mda": feat_mda,
                "cluster_mda": cluster_mda,
                "within_cluster_mda": within_cluster_mda,
            },
            "clusters": {"all": clusters, "selected": selected_clusters, "selected_features": selected_features},
            "diagnostics": {
                "corr_transformed": corr_transformed,
                "rmt_info": rmt_info,
                "baseline_scores": {"feature_mda": scr0_feature, "cluster_mda": scr0_cluster},
            },
        }

    def transform_rrr_to_sample_weights(self, rrr: pd.Series) -> pd.Series:
        weights = np.abs(rrr.values)
        cap = np.percentile(weights, 95)
        weights = np.clip(weights, 0.1, cap) / np.clip(weights, 0.1, cap).mean()
        return pd.Series(weights, index=rrr.index, name="sample_weight")

    def _compute_cluster_cohesion(
        self,
        corr_matrix: np.ndarray,
        feature_names: pd.Index,
        clusters: dict[int, list[str]],
    ) -> dict[int, dict]:
        """Compute within-cluster correlation statistics."""
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        cluster_sizes = {cid: len(feats) for cid, feats in clusters.items()}
        sum_squared_sizes = sum(s**2 for s in cluster_sizes.values())

        metrics = {}
        for cluster_id, feature_list in clusters.items():
            indices = [feature_to_idx[f] for f in feature_list if f in feature_to_idx]

            if len(indices) < 2:
                metrics[cluster_id] = {"mean_corr": 0.0, "weight": 0.0, "size": len(indices)}
                continue

            submatrix = corr_matrix[np.ix_(indices, indices)]
            pairwise_corrs = submatrix[np.triu_indices_from(submatrix, k=1)]

            metrics[cluster_id] = {
                "mean_corr": float(np.mean(pairwise_corrs)),
                "weight": cluster_sizes[cluster_id] ** 2 / sum_squared_sizes,
                "size": len(indices),
            }
        return metrics

    def _get_feature_baseline_corr(
        self, feature_name: str, baseline_metrics: dict[int, dict], clusters: dict[int, list[str]]
    ) -> float:
        """Get feature's average correlation with its cluster members."""
        for cluster_id, feature_list in clusters.items():
            if feature_name in feature_list:
                return baseline_metrics[cluster_id]["mean_corr"]
        return 0.0

    def unsupervised_feature_mda(
        self,
        X: pd.DataFrame,
        clusters: dict[int, list[str]],
        n_perm: int = 100,
        random_state: int | None = None,
        disable_progress: bool = False,
        include_silhouette: bool = True,
    ) -> pd.DataFrame:
        """Unsupervised MDA based on cluster cohesion degradation."""
        if not clusters:
            raise ValueError("clusters dict is empty - run onc_clustering() first")

        all_cluster_features = set(f for feats in clusters.values() for f in feats)
        missing = all_cluster_features - set(X.columns)
        if missing:
            raise ValueError(f"Cluster features not in X: {missing}")

        Xz = self._zscore(X)
        Xz_values = Xz.values
        T, N = Xz_values.shape

        if np.any(~np.isfinite(Xz_values)):
            raise ValueError(f"Z-scored data contains {np.sum(~np.isfinite(Xz_values))} non-finite values.")

        corr_baseline = np.clip(np.corrcoef(Xz_values, rowvar=False), -1.0, 1.0)
        np.fill_diagonal(corr_baseline, 1.0)

        baseline_metrics = self._compute_cluster_cohesion(corr_baseline, X.columns, clusters)
        perm_degradations = np.zeros((N, n_perm), dtype=np.float64)

        rng = np.random.default_rng(random_state)
        corr_working = corr_baseline.copy()

        feature_iterator = enumerate(X.columns)
        if not disable_progress:
            feature_iterator = tqdm(feature_iterator, total=N, desc="Unsupervised MDA", unit="feature", leave=False)

        for feat_idx, feat_name in feature_iterator:
            if feat_name not in all_cluster_features:
                continue

            original_col = Xz_values[:, feat_idx].copy()
            original_corr_row = corr_baseline[feat_idx, :].copy()

            for perm_i in range(n_perm):
                permuted_col = original_col.copy()
                rng.shuffle(permuted_col)

                new_corr_row = np.clip((permuted_col @ Xz_values) / T, -1.0, 1.0)
                corr_working[feat_idx, :] = new_corr_row
                corr_working[:, feat_idx] = new_corr_row
                corr_working[feat_idx, feat_idx] = 1.0

                perm_metrics = self._compute_cluster_cohesion(corr_working, X.columns, clusters)

                degradation, total_weight = 0.0, 0.0
                for cluster_id in clusters:
                    baseline_corr = baseline_metrics[cluster_id]["mean_corr"]
                    perm_corr = perm_metrics[cluster_id]["mean_corr"]
                    weight = baseline_metrics[cluster_id]["weight"]
                    rel_deg = (baseline_corr - perm_corr) / baseline_corr if baseline_corr > 1e-8 else 0.0
                    degradation += weight * rel_deg
                    total_weight += weight

                perm_degradations[feat_idx, perm_i] = degradation / total_weight if total_weight > 0 else 0.0

            corr_working[feat_idx, :] = original_corr_row
            corr_working[:, feat_idx] = original_corr_row

        imp = pd.DataFrame(index=X.columns)
        imp["mean"] = perm_degradations.mean(axis=1)
        imp["std"] = perm_degradations.std(axis=1, ddof=1) / np.sqrt(n_perm)
        imp["baseline_corr"] = [self._get_feature_baseline_corr(f, baseline_metrics, clusters) for f in X.columns]
        imp["baseline_corr"] = imp["baseline_corr"].fillna(0.0)

        if include_silhouette:
            corr_df = pd.DataFrame(corr_baseline, index=X.columns, columns=X.columns)
            feat_sil_df = self.compute_feature_silhouette_scores(corr_df, clusters)
            imp = imp.join(feat_sil_df[["silhouette", "quality"]], how="left")
            imp["silhouette"] = imp["silhouette"].fillna(0.0)

        return imp

    def compute_cluster_silhouette_scores(
        self, corr: pd.DataFrame, clusters: dict[int, list[str]] | None = None
    ) -> pd.DataFrame:
        r"""
        Compute per-cluster silhouette coefficient statistics.
        $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
        """
        clusters = clusters or self.clusters
        if clusters is None:
            raise ValueError("No clusters available. Run onc_clustering() first.")

        dist, feature_names = self._sanitize_correlation_to_distance(corr)

        feature_to_cluster = {feat: cid for cid, feats in clusters.items() for feat in feats}
        labels = np.array([feature_to_cluster.get(f, -1) for f in feature_names])
        valid_mask = labels != -1

        if not valid_mask.any():
            raise ValueError("No features assigned to clusters.")

        silhouette_vals = silhouette_samples(
            dist[np.ix_(valid_mask, valid_mask)], labels[valid_mask], metric="precomputed"
        )

        results = []
        for cid in sorted(clusters.keys()):
            cluster_mask = labels[valid_mask] == cid
            cluster_scores = silhouette_vals[cluster_mask]
            results.append(
                {
                    "cluster_id": cid,
                    "mean": float(cluster_scores.mean()),
                    "std": float(cluster_scores.std()),
                    "min": float(cluster_scores.min()),
                    "max": float(cluster_scores.max()),
                    "n_features": int(cluster_mask.sum()),
                }
            )

        self.cluster_silhouette = pd.DataFrame(results).set_index("cluster_id")
        return self.cluster_silhouette

    def compute_feature_silhouette_scores(
        self, corr: pd.DataFrame, clusters: dict[int, list[str]] | None = None
    ) -> pd.DataFrame:
        """Compute per-feature silhouette scores to identify redundant features."""
        clusters = clusters or self.clusters
        if clusters is None:
            raise ValueError("No clusters available. Run onc_clustering() first.")

        dist, feature_names = self._sanitize_correlation_to_distance(corr)

        feature_to_cluster = {feat: cid for cid, feats in clusters.items() for feat in feats}
        labels = np.array([feature_to_cluster.get(f, -1) for f in feature_names])
        valid_mask = labels != -1

        if not valid_mask.any():
            raise ValueError("No features assigned to clusters.")

        valid_features = feature_names[valid_mask]
        dist_filtered = dist[np.ix_(valid_mask, valid_mask)]
        labels_filtered = labels[valid_mask]

        silhouette_vals = silhouette_samples(dist_filtered, labels_filtered, metric="precomputed")

        quality_map = [(0.7, "Excellent"), (0.5, "Good"), (0.25, "Weak"), (-np.inf, "Poor")]

        results = []
        for feat_idx, feat_name in enumerate(valid_features):
            cluster_id = labels_filtered[feat_idx]
            same_cluster_mask = labels_filtered == cluster_id
            same_cluster_mask[feat_idx] = False

            intra_dist = dist_filtered[feat_idx, same_cluster_mask].mean() if same_cluster_mask.sum() > 0 else 0.0

            other_clusters = np.unique(labels_filtered[labels_filtered != cluster_id])
            nearest_dist = min(
                [dist_filtered[feat_idx, labels_filtered == oc].mean() for oc in other_clusters], default=0.0
            )

            sil_score = silhouette_vals[feat_idx]
            quality = next(q for thresh, q in quality_map if sil_score > thresh)

            results.append(
                {
                    "feature": feat_name,
                    "silhouette": float(sil_score),
                    "cluster_id": int(cluster_id),
                    "intra_cluster_dist": float(intra_dist),
                    "nearest_cluster_dist": float(nearest_dist),
                    "quality": quality,
                }
            )

        self.feature_silhouette = pd.DataFrame(results).set_index("feature")
        return self.feature_silhouette

    @staticmethod
    def filter_redundant_features(
        feature_sil_df: pd.DataFrame,
        clusters: dict[int, list[str]],
        method: str = "threshold",
        threshold: float = 0.25,
        percentile_per_cluster: float = 0.2,
        min_cluster_size: int = 2,
    ) -> tuple[list[str], list[str], pd.DataFrame]:
        """Identify redundant features based on silhouette scores."""
        if method not in {"threshold", "percentile", "both"}:
            raise ValueError(f"method must be 'threshold', 'percentile', or 'both', got '{method}'")

        features_to_remove_set = set()

        if method in {"threshold", "both"}:
            features_to_remove_set.update(feature_sil_df[feature_sil_df["silhouette"] < threshold].index)

        if method in {"percentile", "both"}:
            for cid, feats in clusters.items():
                cluster_sil = feature_sil_df[feature_sil_df["cluster_id"] == cid].sort_values("silhouette")
                n_remove = max(1, int(percentile_per_cluster * len(feats)))
                n_remove = max(0, min(n_remove, len(feats) - min_cluster_size))
                if n_remove > 0:
                    features_to_remove_set.update(cluster_sil.head(n_remove).index)

        final_remove = set()
        for cid, feats in clusters.items():
            cluster_remove = [f for f in feats if f in features_to_remove_set]
            n_can_remove = min(len(cluster_remove), max(0, len(feats) - min_cluster_size))

            if n_can_remove < len(cluster_remove):
                cluster_sil = feature_sil_df[feature_sil_df["cluster_id"] == cid].sort_values("silhouette")
                cluster_remove = cluster_sil.head(n_can_remove).index.tolist()

            final_remove.update(cluster_remove)

        features_to_remove = sorted(final_remove)
        features_to_keep = [f for f in feature_sil_df.index if f not in final_remove]

        summary_rows = [
            {
                "cluster_id": cid,
                "original_size": len(feats),
                "removed_count": len([f for f in feats if f in final_remove]),
                "kept_size": len(feats) - len([f for f in feats if f in final_remove]),
                "removed_features": ", ".join([f for f in feats if f in final_remove]) or "None",
            }
            for cid, feats in clusters.items()
        ]

        return features_to_keep, features_to_remove, pd.DataFrame(summary_rows)

    def visualize_onc_clustering_sih(self, X: pd.DataFrame):
        """Visualize ONC clustering with silhouette quality metrics."""
        corr_transformed, info = self.denoise_detone_corr(X)
        clusters = self.onc_clustering(corr_transformed)
        silhouette_df = self.compute_cluster_silhouette_scores(corr_transformed, clusters)

        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, height_ratios=[1, 1.2], width_ratios=[2, 1.5, 1])

        ax1 = fig.add_subplot(gs[0, :2])
        dendrogram(
            self.linkage_matrix,
            labels=corr_transformed.columns,
            ax=ax1,
            leaf_font_size=8,
            color_threshold=0,
            above_threshold_color=self.plot_style.accent1,
        )
        ax1.set_title(f"Hierarchical Clustering Dendrogram (Optimal k={len(clusters)})", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Feature", fontsize=12)
        ax1.set_ylabel("Distance", fontsize=12)

        ax_sil = fig.add_subplot(gs[0, 2])
        cluster_ids_sorted = silhouette_df.sort_values("mean", ascending=False).index
        y_pos = np.arange(len(cluster_ids_sorted))
        means = silhouette_df.loc[cluster_ids_sorted, "mean"].values
        stds = silhouette_df.loc[cluster_ids_sorted, "std"].values

        colors = [
            self.plot_style.accent3 if m > 0.5 else self.plot_style.accent6 if m > 0.25 else self.plot_style.accent5
            for m in means
        ]

        ax_sil.barh(y_pos, means, color=colors, alpha=0.85, edgecolor="none")
        ax_sil.errorbar(
            means, y_pos, xerr=stds, fmt="none", ecolor=self.plot_style.accent4, elinewidth=1.5, capsize=2, alpha=0.9
        )
        ax_sil.axvline(0.5, color=self.plot_style.accent3, linestyle="--", linewidth=1, alpha=0.5, label="Strong")
        ax_sil.axvline(0.25, color=self.plot_style.accent6, linestyle="--", linewidth=1, alpha=0.5, label="Moderate")
        ax_sil.set_yticks(y_pos)
        ax_sil.set_yticklabels([f"C{cid}" for cid in cluster_ids_sorted])
        ax_sil.set_xlabel("Silhouette Score", fontsize=11)
        ax_sil.set_title("Cluster Quality", fontsize=13, fontweight="bold")
        ax_sil.legend(loc="lower right", fontsize=8)
        ax_sil.grid(True, axis="x", alpha=0.25)
        ax_sil.invert_yaxis()

        ordered_features = [f for cid in sorted(clusters.keys()) for f in clusters[cid]]
        cluster_boundaries = np.cumsum([len(clusters[cid]) for cid in sorted(clusters.keys())])[:-1]
        corr_ordered = corr_transformed.loc[ordered_features, ordered_features]

        ax2 = fig.add_subplot(gs[1, 0])
        sns.heatmap(
            corr_ordered.values,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"label": "Correlation"},
            ax=ax2,
            xticklabels=False,
            yticklabels=False,
        )
        for boundary in cluster_boundaries:
            ax2.axhline(boundary, color=self.plot_style.accent6, linewidth=2, linestyle="--")
            ax2.axvline(boundary, color=self.plot_style.accent6, linewidth=2, linestyle="--")
        ax2.set_title(f"Clustered Correlation Matrix ({len(clusters)} clusters)", fontsize=14, fontweight="bold")

        ax3 = fig.add_subplot(gs[1, 1:])
        ax3.axis("off")

        cluster_stats = []
        for cid in sorted(clusters.keys()):
            features = clusters[cid]
            submatrix = corr_transformed.loc[features, features]
            upper_tri = submatrix.values[np.triu_indices_from(submatrix.values, k=1)]
            sil_mean, sil_std = silhouette_df.loc[cid, "mean"], silhouette_df.loc[cid, "std"]

            cluster_stats.append(
                {
                    "Cluster": f"C{cid}",
                    "Size": len(features),
                    "Avg Corr": f"{upper_tri.mean():.3f}" if len(upper_tri) > 0 else "N/A",
                    "Silhouette": f"{sil_mean:.3f} ± {sil_std:.3f}",
                    "Features": ", ".join(features[:3]) + ("..." if len(features) > 3 else ""),
                }
            )

        df_stats = pd.DataFrame(cluster_stats)
        table = ax3.table(
            cellText=df_stats.values, colLabels=df_stats.columns, cellLoc="left", loc="center", bbox=[0, 0, 1, 1]
        )
        self._style_table(
            table,
            df_stats,
            highlight_col="Silhouette",
            highlight_thresholds=[
                (0.5, self.plot_style.accent3),
                (0.25, self.plot_style.accent6),
                (-np.inf, self.plot_style.accent5),
            ],
        )
        ax3.set_title("Cluster Statistics", fontsize=14, fontweight="bold", pad=20)

        summary_text = (
            f"Optimal clusters: {len(clusters)}\n"
            f"Global silhouette: {self.best_silhouette:.3f}\n"
            f"Mean cluster silhouette: {silhouette_df['mean'].mean():.3f}\n"
            f"Total features: {sum(len(v) for v in clusters.values())}"
        )
        fig.text(
            0.98,
            0.02,
            summary_text,
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor=self.plot_style.accent3, alpha=0.5),
            ha="right",
            color=self.plot_style.font_color,
        )

        self.plot_style.apply_mpl(fig)
        return fig, clusters


class PCAEigenvalueAnalysis:
    """PCA-based feature selection for quantitative trading systems."""

    def __init__(self, variance_threshold: float = 0.95, random_state: int = RANDOM_STATE):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.pca_model_: PCA | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: pd.DataFrame | None = None
        self.scaler_: StandardScaler | None = None

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward/backward fill then impute remaining NaNs with median."""
        X_clean = X.ffill().bfill()
        if X_clean.isna().any().any():
            X_clean = X_clean.fillna(X_clean.median())
        return X_clean

    def compute_pca_eigenvalues(
        self, X: pd.DataFrame, variance_threshold: float | None = None, standardize: bool = True, verbose: bool = False
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Compute PCA eigenvalues and eigenvectors for feature matrix."""
        variance_threshold = variance_threshold or self.variance_threshold
        X_clean = self._clean_data(X)
        if standardize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_clean)
        else:
            X_scaled = X_clean.to_numpy()
        n_components = min(X_scaled.shape[0], X_scaled.shape[1])
        self.pca_model_ = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_model_.fit(X_scaled)
        self.eigenvalues_ = self.pca_model_.explained_variance_
        self.eigenvectors_ = pd.DataFrame(
            self.pca_model_.components_.T, index=X_clean.columns, columns=[f"PC{i + 1}" for i in range(n_components)]
        )
        variance_explained = self.pca_model_.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
        kaiser_threshold = self.eigenvalues_.mean()
        n_components_kaiser = np.sum(self.eigenvalues_ > kaiser_threshold)
        pca_analysis = {
            "eigenvalues": self.eigenvalues_,
            "eigenvectors": self.eigenvectors_,
            "variance_explained": variance_explained,
            "cumulative_variance": cumulative_variance,
            "n_components_threshold": n_components_threshold,
            "n_components_kaiser": n_components_kaiser,
            "kaiser_threshold": kaiser_threshold,
            "total_variance": self.eigenvalues_.sum(),
            "n_features": X.shape[1],
            "n_observations": X.shape[0],
        }
        if verbose:
            print("PCA Eigenvalue Analysis:")
            print(f" Total features: {X.shape[1]}")
            print(f" Total observations: {X.shape[0]}")
            print(f" Total variance: {pca_analysis['total_variance']:.4f}")
            print(f" Components for {variance_threshold:.0%} variance: {n_components_threshold}")
            print(f" Components via Kaiser criterion (λ > {kaiser_threshold:.4f}): {n_components_kaiser}")
        return self.eigenvalues_, self.eigenvectors_, pca_analysis

    def plot_pca_eigenvalues(
        self, pca_analysis: dict, eigenvalues: np.ndarray | None = None, variance_threshold: float | None = None
    ) -> plt.Figure:
        """PLOTTING FUNCTION: Visualize eigenvalues with scree plot and cumulative variance."""
        style = DEFAULT_STYLE
        variance_threshold = variance_threshold or self.variance_threshold
        eigenvalues = eigenvalues if eigenvalues is not None else pca_analysis["eigenvalues"]
        # variance_explained = pca_analysis["variance_explained"]
        cumulative_variance = pca_analysis["cumulative_variance"]
        kaiser_threshold = pca_analysis["kaiser_threshold"]
        n_components_threshold = pca_analysis["n_components_threshold"]
        mpl.rcParams.update(
            {
                "figure.facecolor": style.paper_bgcolor,
                "axes.facecolor": style.plot_bgcolor,
                "text.color": style.font_color,
                "axes.labelcolor": style.font_color,
                "xtick.color": style.font_color,
                "ytick.color": style.font_color,
                "grid.color": style.grid,
                "axes.edgecolor": style.line,
            }
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        components = np.arange(1, len(eigenvalues) + 1)
        ax1.bar(components, eigenvalues, color=style.accent1, edgecolor=style.line, linewidth=0.8)
        ax1.axhline(
            kaiser_threshold,
            color=style.accent2,
            linestyle="--",
            linewidth=1.5,
            label=f"Kaiser Criterion (λ̄ = {kaiser_threshold:.2f})",
        )
        ax1.set_xlabel("Principal Component", fontsize=11)
        ax1.set_ylabel("Eigenvalue (Variance)", fontsize=11)
        ax1.set_title("Scree Plot: Eigenvalues per Component", fontsize=12, pad=10)
        ax1.legend(facecolor=style.plot_bgcolor, edgecolor=style.line, labelcolor=style.font_color)
        ax1.grid(True, alpha=0.3, linestyle="--")
        for i in range(min(5, len(eigenvalues))):
            ax1.text(
                i + 1,
                eigenvalues[i] + eigenvalues.max() * 0.02,
                f"{eigenvalues[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=style.font_color,
            )
        ax2.plot(
            components,
            cumulative_variance,
            color=style.accent2,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Cumulative Variance",
        )
        ax2.axhline(
            variance_threshold,
            color=style.accent5,
            linestyle="--",
            linewidth=1.5,
            label=f"{variance_threshold:.0%} Threshold",
        )
        ax2.axvline(n_components_threshold, color=style.muted, linestyle=":", linewidth=1.2)
        ax2.scatter(
            [n_components_threshold],
            [cumulative_variance[n_components_threshold - 1]],
            color=style.accent5,
            s=100,
            zorder=5,
            edgecolor=style.line,
            linewidth=2,
        )
        ax2.text(
            n_components_threshold + 0.5,
            variance_threshold - 0.05,
            f"PC{n_components_threshold}\n({cumulative_variance[n_components_threshold - 1]:.2%})",
            fontsize=10,
            color=style.font_color,
        )
        ax2.set_xlabel("Principal Component", fontsize=11)
        ax2.set_ylabel("Cumulative Variance Explained", fontsize=11)
        ax2.set_title("Cumulative Variance Explained", fontsize=12, pad=10)
        ax2.set_ylim(0, 1.05)
        ax2.legend(facecolor=style.plot_bgcolor, edgecolor=style.line, labelcolor=style.font_color)
        ax2.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        return fig

    def get_detailed_pc_interpretation(
        self, eigenvectors: pd.DataFrame | None = None, n_components: int | None = None, n_top_features: int = 10
    ) -> pd.DataFrame:
        """Get detailed principal component interpretation with top contributing features."""
        eigenvectors = eigenvectors if eigenvectors is not None else self.eigenvectors_
        if eigenvectors is None:
            raise ValueError("Must call compute_pca_eigenvalues() first or provide eigenvectors")
        n_components = n_components or eigenvectors.shape[1]
        n_components = min(n_components, eigenvectors.shape[1])
        interpretation_data = []
        for i in range(n_components):
            pc_name = f"PC{i + 1}"
            loadings = eigenvectors[pc_name].abs().sort_values(ascending=False)
            for rank, (feature, loading) in enumerate(loadings.head(n_top_features).items(), 1):
                interpretation_data.append(
                    {
                        "PC": pc_name,
                        "Feature": feature,
                        "Loading": eigenvectors.loc[feature, pc_name],
                        "AbsLoading": loading,
                        "Rank": rank,
                    }
                )
        return pd.DataFrame(interpretation_data)

    def print_pc_interpretation(
        self,
        eigenvectors: pd.DataFrame | None = None,
        eigenvalues: np.ndarray | None = None,
        n_components: int = 5,
        n_top_features: int = 10,
        verbose: bool = False,
    ) -> None:
        """Print human-readable PC interpretation."""
        eigenvectors = eigenvectors if eigenvectors is not None else self.eigenvectors_
        eigenvalues = eigenvalues if eigenvalues is not None else self.eigenvalues_
        if eigenvectors is None or eigenvalues is None:
            raise ValueError("Must call compute_pca_eigenvalues() first")
        variance_explained = eigenvalues / eigenvalues.sum()
        n_components = min(n_components, len(eigenvalues))
        if verbose:
            print("\n" + "=" * 80)
            print("PRINCIPAL COMPONENT INTERPRETATION")
            print("=" * 80)
            for i in range(n_components):
                pc_name = f"PC{i + 1}"
                loadings = eigenvectors[pc_name].abs().sort_values(ascending=False)
                print(f"\n{pc_name}: λ={eigenvalues[i]:.4f}, Var={variance_explained[i]:.2%}")
                print("-" * 80)
                for rank, (feature, abs_loading) in enumerate(loadings.head(n_top_features).items(), 1):
                    signed_loading = eigenvectors.loc[feature, pc_name]
                    direction = "+" if signed_loading > 0 else "-"
                    print(f" {rank:2d}. {feature:<30s} {direction} {abs_loading:.4f}")

    def filter_features_by_pca_eigenvalue(
        self,
        pca_analysis: dict,
        pc_interpretation: pd.DataFrame | None = None,
        variance_threshold: float = 0.02,
        loading_threshold: float = 0.3,
    ) -> list[str]:
        """Select features based on PC variance and loading thresholds."""
        if pc_interpretation is None:
            pc_interpretation = self.get_detailed_pc_interpretation(
                n_components=None, n_top_features=pca_analysis["n_features"]
            )

        variance_explained = pca_analysis["variance_explained"]
        variance_map = {f"PC{i + 1}": var for i, var in enumerate(variance_explained)}
        pc_interpretation = pc_interpretation.copy()
        pc_interpretation["VarianceExplained"] = pc_interpretation["PC"].map(variance_map)

        # Keep PCs with variance >= threshold
        important_pcs = pc_interpretation[pc_interpretation["VarianceExplained"] >= variance_threshold]["PC"].unique()

        filtered_df = pc_interpretation[pc_interpretation["PC"].isin(important_pcs)]
        feature_max_loading = filtered_df.groupby("Feature")["AbsLoading"].max()

        return feature_max_loading[feature_max_loading >= loading_threshold].index.tolist()

    def get_features_to_remove(
        self,
        X: pd.DataFrame,
        pca_analysis: dict,
        variance_threshold: float = 0.02,
        loading_threshold: float = 0.3,
        verbose: bool = False,
    ) -> list[str]:
        """Get list of features to remove based on PCA analysis."""
        retained_features = self.filter_features_by_pca_eigenvalue(
            pca_analysis, variance_threshold=variance_threshold, loading_threshold=loading_threshold
        )
        features_to_remove = list(set(X.columns) - set(retained_features))
        if verbose:
            print("\n" + "=" * 27)
            print("PCA-BASED FEATURE FILTERING")
            print("=" * 27)
            print(f"Original features: {len(X.columns)}")
            print(f"Retained features: {len(retained_features)}")
            print(f"Removed features: {len(features_to_remove)}")
            print(f"Reduction: {len(features_to_remove) / len(X.columns):.1%}")
        return features_to_remove

    def transform(self, X: pd.DataFrame, n_components: int | None = None) -> pd.DataFrame:
        """Transform features to principal component space."""
        if self.pca_model_ is None or self.scaler_ is None:
            raise ValueError("Must call compute_pca_eigenvalues() first")
        X_clean = self._clean_data(X)
        X_scaled = self.scaler_.transform(X_clean)
        X_pca = self.pca_model_.transform(X_scaled)
        if n_components is not None:
            X_pca = X_pca[:, :n_components]
            col_names = [f"PC{i + 1}" for i in range(n_components)]
        else:
            col_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, index=X.index, columns=col_names)

    def find_optimal_pca_thresholds(
        self,
        features: pd.DataFrame,
        variance_range: np.ndarray,
        loading_range: np.ndarray,
        target: pd.Series | None = None,
        verbose: bool = True,
        random_state: int = RANDOM_STATE,
    ) -> dict:
        """
        Find optimal variance_threshold and loading_threshold using grid search.
        Strategy: Maximize information retention while minimizing feature count.
        Superior to cross-validation approach as it optimizes information efficiency
        without overfitting to specific target (unless target provided).
        """
        loading_range = np.linspace(0.1, 0.5, 9)
        variance_range = np.linspace(0.01, 0.10, 10)

        # Initialize PCA analyzer
        pca_analyzer = PCAEigenvalueAnalysis(random_state=random_state)
        # Compute PCA once (vectorized)
        eigenvalues, eigenvectors, pca_analysis = pca_analyzer.compute_pca_eigenvalues(features, standardize=True)
        pc_interpretation = pca_analyzer.get_detailed_pc_interpretation(
            eigenvectors=eigenvectors, n_components=None, n_top_features=len(features.columns)
        )
        total_variance = pca_analysis["total_variance"]
        all_mi = 0.0
        if target is not None:
            all_mi = mutual_info_regression(
                features.values, target.values, n_neighbors=5, random_state=random_state
            ).sum()
        # Grid search with validation
        results = []
        for var_thresh, load_thresh in product(variance_range, loading_range):
            try:
                # Filter features with proper validation
                retained_features = pca_analyzer.filter_features_by_pca_eigenvalue(
                    pca_analysis=pca_analysis,
                    pc_interpretation=pc_interpretation,
                    variance_threshold=var_thresh,
                    loading_threshold=load_thresh,
                )
                n_selected = len(retained_features)
                # Validate minimum features for PCA
                if n_selected < 2:
                    continue
                reduction_pct = 100 * (1 - n_selected / len(features.columns))
                # Compute information retention
                if target is not None:
                    # Supervised: MI with target (better for classification)
                    selected_mi = mutual_info_regression(
                        features[retained_features].values, target.values, n_neighbors=5, random_state=random_state
                    ).sum()
                    info_retention = selected_mi / all_mi if all_mi > 0 else 1.0
                else:
                    # Unsupervised: variance retention via PCA (faster, no overfitting)
                    n_samples = features.shape[0]
                    n_components_max = min(n_selected, n_samples)
                    if n_components_max < 1:
                        continue
                    pca_selected = PCA(n_components=n_components_max, random_state=random_state)
                    scaler_selected = StandardScaler()
                    features_selected_scaled = scaler_selected.fit_transform(features[retained_features])
                    pca_selected.fit(features_selected_scaled)
                    selected_variance = pca_selected.explained_variance_.sum()
                    info_retention = selected_variance / total_variance
                # Efficiency score: info per feature (key metric)
                feature_ratio = n_selected / len(features.columns)
                efficiency_score = info_retention / feature_ratio if feature_ratio > 0 else 0
                results.append(
                    {
                        "variance_threshold": var_thresh,
                        "loading_threshold": load_thresh,
                        "n_features": n_selected,
                        "reduction_pct": reduction_pct,
                        "info_retention": info_retention,
                        "efficiency_score": efficiency_score,
                    }
                )
            except Exception as e:
                # Skip invalid configurations
                if verbose:
                    print(f"WARNING: Config (var={var_thresh:.3f}, load={load_thresh:.3f}) failed: {e}")
                continue
        # Validate results
        if len(results) == 0:
            raise ValueError(
                "No valid configurations found. Adjust ranges:\n"
                f" variance_range: {variance_range[0]:.3f} - {variance_range[-1]:.3f}\n"
                f" loading_range: {loading_range[0]:.3f} - {loading_range[-1]:.3f}\n"
                "Suggestions:\n"
                " - Widen ranges (e.g., variance_range=np.linspace(0.005, 0.20, 20))\n"
                " - Lower loading_threshold floor (e.g., loading_range=np.linspace(0.05, 0.5, 10))\n"
                " - Check feature quality (too many zero-variance features?)"
            )
        df_results = pd.DataFrame(results)
        # Find optimal: max efficiency with ≥90% info retention
        valid_results = df_results[df_results["info_retention"] >= 0.90]
        if len(valid_results) > 0:
            optimal_idx = valid_results["efficiency_score"].idxmax()
            selection_method = "≥90% info retention"
        else:
            optimal_idx = df_results["efficiency_score"].idxmax()
            selection_method = "max efficiency (relaxed)"
            if verbose:
                print("WARNING: No configs with ≥90% info retention. Using max efficiency.")
        optimal_row = df_results.loc[optimal_idx]
        if verbose:
            # Visualization with PlotStyle
            style = DEFAULT_STYLE
            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            # Apply global styling
            plt.rcParams.update(
                {
                    "figure.facecolor": style.paper_bgcolor,
                    "axes.facecolor": style.plot_bgcolor,
                    "text.color": style.font_color,
                    "axes.labelcolor": style.font_color,
                    "xtick.color": style.font_color,
                    "ytick.color": style.font_color,
                    "grid.color": style.grid,
                    "axes.edgecolor": style.line,
                }
            )
            # Heatmap 1: Efficiency score
            pivot_efficiency = df_results.pivot_table(
                index="loading_threshold", columns="variance_threshold", values="efficiency_score", aggfunc="mean"
            )
            im1 = axes[0, 0].imshow(pivot_efficiency.values, cmap="viridis", aspect="auto", interpolation="nearest")
            axes[0, 0].set_xticks(range(len(pivot_efficiency.columns)))
            axes[0, 0].set_xticklabels([f"{v:.2f}" for v in pivot_efficiency.columns], rotation=45)
            axes[0, 0].set_yticks(range(len(pivot_efficiency.index)))
            axes[0, 0].set_yticklabels([f"{i:.2f}" for i in pivot_efficiency.index])
            axes[0, 0].set_xlabel("Variance Threshold (PC Importance)", fontsize=11)
            axes[0, 0].set_ylabel("Loading Threshold", fontsize=11)
            axes[0, 0].set_title("Efficiency Score Heatmap", fontsize=12, fontweight="bold")
            cbar1 = plt.colorbar(im1, ax=axes[0, 0])
            cbar1.set_label("Efficiency", rotation=90)
            # Mark optimal on heatmap
            if optimal_row["variance_threshold"] in pivot_efficiency.columns.values:
                opt_var_idx = list(pivot_efficiency.columns).index(optimal_row["variance_threshold"])
                opt_load_idx = list(pivot_efficiency.index).index(optimal_row["loading_threshold"])
                axes[0, 0].scatter(
                    [opt_var_idx],
                    [opt_load_idx],
                    color="red",
                    s=300,
                    marker="*",
                    edgecolors="white",
                    linewidths=2.5,
                    zorder=10,
                    label="Optimal",
                )
                axes[0, 0].legend(loc="upper right", fontsize=9)
            # Heatmap 2: Feature count
            pivot_features = df_results.pivot_table(
                index="loading_threshold", columns="variance_threshold", values="n_features", aggfunc="mean"
            )
            im2 = axes[0, 1].imshow(pivot_features.values, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")
            axes[0, 1].set_xticks(range(len(pivot_features.columns)))
            axes[0, 1].set_xticklabels([f"{v:.2f}" for v in pivot_features.columns], rotation=45)
            axes[0, 1].set_yticks(range(len(pivot_features.index)))
            axes[0, 1].set_yticklabels([f"{i:.2f}" for i in pivot_features.index])
            axes[0, 1].set_xlabel("Variance Threshold (PC Importance)", fontsize=11)
            axes[0, 1].set_ylabel("Loading Threshold", fontsize=11)
            axes[0, 1].set_title("Feature Count Heatmap", fontsize=12, fontweight="bold")
            cbar2 = plt.colorbar(im2, ax=axes[0, 1])
            cbar2.set_label("# Features", rotation=90)
            if optimal_row["variance_threshold"] in pivot_features.columns.values:
                axes[0, 1].scatter(
                    [opt_var_idx],
                    [opt_load_idx],
                    color="red",
                    s=300,
                    marker="*",
                    edgecolors="white",
                    linewidths=2.5,
                    zorder=10,
                    label="Optimal",
                )
                axes[0, 1].legend(loc="upper right", fontsize=9)
            # Tradeoff curve
            scatter = axes[1, 0].scatter(
                df_results["reduction_pct"],
                df_results["info_retention"],
                c=df_results["efficiency_score"],
                cmap="viridis",
                s=120,
                edgecolors="black",
                linewidth=0.8,
                alpha=0.8,
            )
            axes[1, 0].scatter(
                optimal_row["reduction_pct"],
                optimal_row["info_retention"],
                color="red",
                s=400,
                marker="*",
                edgecolors="white",
                linewidth=2.5,
                zorder=10,
                label="Optimal",
            )
            axes[1, 0].axhline(
                0.90, color="orange", linestyle="--", linewidth=2, label="90% retention target", alpha=0.8
            )
            axes[1, 0].set_xlabel("Feature Reduction (%)", fontsize=11)
            axes[1, 0].set_ylabel("Information Retention", fontsize=11)
            axes[1, 0].set_title("Reduction vs Retention Tradeoff", fontsize=12, fontweight="bold")
            axes[1, 0].legend(fontsize=9, loc="best")
            axes[1, 0].grid(True, alpha=0.3, color=style.grid)
            cbar_scatter = plt.colorbar(scatter, ax=axes[1, 0])
            cbar_scatter.set_label("Efficiency", rotation=90)
            # Efficiency distribution
            axes[1, 1].hist(
                df_results["efficiency_score"],
                bins=20,
                edgecolor="black",
                alpha=0.7,
                color=style.accent2,
                linewidth=1.2,
            )
            axes[1, 1].axvline(
                optimal_row["efficiency_score"], color="red", linestyle="--", linewidth=2.5, label="Optimal", alpha=0.9
            )
            axes[1, 1].set_xlabel("Efficiency Score", fontsize=11)
            axes[1, 1].set_ylabel("Frequency", fontsize=11)
            axes[1, 1].set_title("Efficiency Score Distribution", fontsize=12, fontweight="bold")
            axes[1, 1].legend(fontsize=9)
            axes[1, 1].grid(True, alpha=0.3, color=style.grid, axis="y")
            # Set background colors
            for ax in axes.flat:
                ax.set_facecolor(style.plot_bgcolor)
            fig.patch.set_facecolor(style.paper_bgcolor)
            plt.tight_layout()
            plt.savefig("pca_threshold_optimization.png", dpi=150, bbox_inches="tight", facecolor=style.paper_bgcolor)
            plt.show()
            # Print results
            print("=" * 80)
            print("OPTIMAL PCA THRESHOLD SELECTION")
            print("=" * 80)
            print(f"Strategy: {selection_method}")
            print(f"Grid search tested: {len(df_results)} valid configurations")
            print()
            print(f"✓ Optimal variance_threshold: {optimal_row['variance_threshold']:.3f}")
            print(f"✓ Optimal loading_threshold: {optimal_row['loading_threshold']:.3f}")
            print(f"✓ Features selected: {int(optimal_row['n_features'])} / {len(features.columns)}")
            print(f"✓ Reduction: {optimal_row['reduction_pct']:.1f}%")
            print(f"✓ Information retention: {optimal_row['info_retention']:.1%}")
            print(f"✓ Efficiency score: {optimal_row['efficiency_score']:.3f}")
            print()
            # Top 5 configurations
            print("TOP 5 CONFIGURATIONS:")
            print("-" * 80)
            top5 = df_results.nlargest(5, "efficiency_score")[
                [
                    "variance_threshold",
                    "loading_threshold",
                    "n_features",
                    "reduction_pct",
                    "info_retention",
                    "efficiency_score",
                ]
            ]
            print(top5.to_string(index=False))
            print()
        return {
            "optimal_variance_threshold": optimal_row["variance_threshold"],
            "optimal_loading_threshold": optimal_row["loading_threshold"],
            "n_features_selected": int(optimal_row["n_features"]),
            "reduction_pct": optimal_row["reduction_pct"],
            "info_retention": optimal_row["info_retention"],
            "efficiency_score": optimal_row["efficiency_score"],
            "grid_results": df_results,
        }

    @staticmethod
    def _validate_rolling_inputs(X: pd.DataFrame, window: int, method: str) -> None:
        if window < X.shape[1]:
            raise ValueError(f"Window ({window}) must be >= number of features ({X.shape[1]})")
        if window > X.shape[0]:
            raise ValueError(f"Window ({window}) exceeds observations ({X.shape[0]})")
        if method not in {"correlation", "covariance"}:
            raise ValueError("method must be 'correlation' or 'covariance'")

    @staticmethod
    def _standardize_window_values(X_window: pd.DataFrame, standardize: bool) -> np.ndarray:
        values = X_window.to_numpy(dtype=float, copy=True)
        if not standardize:
            return values
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=0)
        stds = np.where(stds > 0.0, stds, 1.0)
        return (values - means) / stds

    def _rolling_eigenvalues(
        self,
        X: pd.DataFrame,
        window: int,
        standardize: bool,
        method: str,
    ) -> np.ndarray:
        self._validate_rolling_inputs(X, window, method)
        n_rows, n_features = X.shape
        out = np.full((n_rows, n_features), np.nan, dtype=float)

        for row_idx in range(window - 1, n_rows):
            X_window = self._clean_data(X.iloc[row_idx - window + 1 : row_idx + 1])
            X_values = self._standardize_window_values(X_window, standardize)
            matrix = np.corrcoef(X_values, rowvar=False) if method == "correlation" else np.cov(X_values, rowvar=False)
            out[row_idx] = np.linalg.eigvalsh(matrix)[::-1]

        return out

    def compute_rolling_max_eigenvalue(
        self, X: pd.DataFrame, window: int, standardize: bool = True, method: str = "correlation"
    ) -> pd.Series:
        """
        Compute maximum eigenvalue of correlation/covariance matrix on rolling basis.
        Critical for detecting market instability, regime changes, and systemic risk.
        Maximum eigenvalue spikes indicate increased correlation clustering.
        """
        eigvals = self._rolling_eigenvalues(X, window, standardize, method)
        return pd.Series(eigvals[:, 0], index=X.index, name="max_eigenvalue")

    def compute_rolling_eigenvalue_spectrum(
        self,
        X: pd.DataFrame,
        window: int,
        n_components: int | None = None,
        standardize: bool = True,
        method: str = "correlation",
    ) -> pd.DataFrame:
        """
        Compute top N eigenvalues of correlation matrix on rolling basis.
        Extended version returning full spectrum for eigenvalue dispersion analysis.
        """
        n_components = X.shape[1] if n_components is None else min(n_components, X.shape[1])
        eigvals = self._rolling_eigenvalues(X, window, standardize, method)
        columns = [f"λ_{i + 1}" for i in range(n_components)]
        return pd.DataFrame(eigvals[:, :n_components], index=X.index, columns=columns)

    def compute_eigenvalue_concentration(self, X: pd.DataFrame, window: int, standardize: bool = True) -> pd.DataFrame:
        """
        Compute eigenvalue concentration metrics on rolling basis.
        Quantifies variance capture by top eigenvalues.
        High concentration = strong common factor (market stress).
        """
        eigvals = self._rolling_eigenvalues(X, window, standardize, method="correlation")
        valid = np.isfinite(eigvals[:, 0])
        totals = eigvals.sum(axis=1)
        totals_sq = (eigvals**2).sum(axis=1)
        ratio_denom = eigvals[:, 1] if eigvals.shape[1] > 1 else np.full(len(eigvals), np.nan)

        result = pd.DataFrame(
            {
                "max_eigenvalue": eigvals[:, 0],
                "var_explained_pc1": np.divide(
                    eigvals[:, 0],
                    totals,
                    out=np.full(len(eigvals), np.nan),
                    where=valid & (totals > 0.0),
                ),
                "eigenvalue_ratio": np.divide(
                    eigvals[:, 0],
                    ratio_denom,
                    out=np.full(len(eigvals), np.nan),
                    where=valid & (ratio_denom > 0.0),
                ),
                "participation_ratio": np.divide(
                    totals**2,
                    totals_sq,
                    out=np.full(len(eigvals), np.nan),
                    where=valid & (totals_sq > 0.0),
                ),
                "marchenko_pastur_upper": np.nan,
                "excess_eigenvalue": np.nan,
            },
            index=X.index,
        )

        gamma = X.shape[1] / window
        mp_upper = (1 + np.sqrt(gamma)) ** 2
        result.loc[valid, "marchenko_pastur_upper"] = mp_upper
        result.loc[valid, "excess_eigenvalue"] = result.loc[valid, "max_eigenvalue"] - mp_upper
        return result


def rank_gaussian_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Transform features via rank percentiles to standard normal (avoid extreme tails)."""
    u = X.rank(method="average", pct=True).clip(1e-6, 1 - 1e-6)
    return pd.DataFrame(stats.norm.ppf(u), index=X.index, columns=X.columns)


CVSplit = namedtuple("CVSplit", ["X_train", "X_test", "y_train", "y_test", "w_train", "w_test"])


def _mda_fold_worker(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_np: np.ndarray,
    y_np: np.ndarray,
    sample_weights: np.ndarray | None,
    clf_template,
    scoring: str,
    n_perm: int,
    fold_seed: int,
) -> dict:
    """
    Process a single fold for MDA computation with vectorized permutation generation.

    Returns dict with 'fold_idx', 'baseline_score', and 'perm_scores' (n_features,).
    """
    X_train, y_train = X_np[train_idx], y_np[train_idx]
    X_test, y_test = X_np[test_idx], y_np[test_idx]
    w_train = _slice_weights(sample_weights, train_idx)
    w_test = _slice_weights(sample_weights, test_idx)

    # Fit classifier for this fold
    # Note: clf_template already has n_jobs=1 when running parallel folds
    fold_clf = clone(clf_template)
    fold_clf.fit(X_train, y_train, sample_weight=w_train)

    # Compute baseline score
    baseline_score = _score_classifier(fold_clf, X_test, y_test, scoring=scoring, sample_weight=w_test)

    # Vectorized permutation generation
    n_features = X_test.shape[1]
    n_test = X_test.shape[0]
    fold_rng = np.random.default_rng(fold_seed)

    # Pre-generate all permutation indices at once (n_features × n_perm × n_test)
    perm_indices = np.empty((n_features, n_perm, n_test), dtype=np.intp)
    for feat_idx in range(n_features):
        for perm_i in range(n_perm):
            perm_indices[feat_idx, perm_i] = fold_rng.permutation(n_test)

    # Single working copy for all permutations
    X_test_work = X_test.copy()
    perm_scores = np.empty(n_features, dtype=float)

    for feat_idx in range(n_features):
        original_col = X_test[:, feat_idx].copy()
        perm_scores_feat = np.empty(n_perm, dtype=float)

        for perm_i in range(n_perm):
            # Permute using pre-generated indices
            X_test_work[:, feat_idx] = original_col[perm_indices[feat_idx, perm_i]]
            perm_scores_feat[perm_i] = _score_classifier(
                fold_clf, X_test_work, y_test, scoring=scoring, sample_weight=w_test
            )

        perm_scores[feat_idx] = perm_scores_feat.mean()
        # Restore original column for next feature
        X_test_work[:, feat_idx] = original_col

    return {
        "fold_idx": fold_idx,
        "baseline_score": baseline_score,
        "perm_scores": perm_scores,
    }


class FeatureSelection:
    """Financial ML feature selection with optional RRR-based sample weighting."""

    _SUPPORTED_METHODS = ("MDI", "MDA", "SFI")
    _SUPPORTED_FALLBACK_MODES = {"snr_positive", "mean_positive", "abs_snr"}
    _RRR_COLUMNS = ("RiskRewardRatio", "RRR")

    def __init__(
        self,
        n_estimators: int = 100,
        cv: int = 5,
        max_samples: float = 1.0,
        scoring: str = "neg_log_loss",
        minWLeaf: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        mda_n_perm: int = 1,
        mda_n_repeats: int = 1,
        mda_shuffle: bool = False,
    ):
        """Initialize FeatureSelection with validation."""
        self.n_estimators = n_estimators
        self.cv = cv
        self.max_samples = max_samples
        self.scoring = scoring
        self.min_weight_fraction_leaf = minWLeaf
        self.minWLeaf = minWLeaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mda_n_perm = mda_n_perm
        self.mda_n_repeats = mda_n_repeats
        self.mda_shuffle = mda_shuffle
        self.plot_style = DEFAULT_STYLE

    @staticmethod
    def _validate_importance_columns(importance_df: pd.DataFrame) -> None:
        required = {"mean", "std"}
        missing = required.difference(importance_df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"importance_df must contain columns: {missing_str}")

    def _resolve_cv_folds(self, cv: int | None) -> int:
        cv_folds = self.cv if cv is None else cv
        if cv_folds < 2:
            raise ValueError(f"cv must be >= 2 for confidence intervals (got {cv_folds})")
        return cv_folds

    def _resolve_target_and_weights(self, y: pd.Series | pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
        """Extract target labels and compute sample weights from y."""
        if isinstance(y, pd.Series):
            return y, None

        if isinstance(y, pd.DataFrame):
            y_target = y["meta_label"] if "meta_label" in y.columns else y.iloc[:, 0]
            return y_target, self.calculate_rrr_weights(y)

        raise TypeError(f"y must be Series or DataFrame, got {type(y)}")

    def _get_cv_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> CVSplit:
        w_train = _slice_weights(sample_weights, train_idx)
        w_test = _slice_weights(sample_weights, test_idx)
        return CVSplit(
            X_train=X.iloc[train_idx],
            X_test=X.iloc[test_idx],
            y_train=y.iloc[train_idx],
            y_test=y.iloc[test_idx],
            w_train=w_train,
            w_test=w_test,
        )

    def _fit_classifier(self, clf, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None):
        clf.fit(_as_array_2d(X), _as_array_1d(y), sample_weight=sample_weight)

    def _compute_score(self, clf, X, y, sample_weight: np.ndarray | None = None) -> float:
        return _score_classifier(clf, X, y, scoring=self.scoring, sample_weight=sample_weight)

    def calculate_rrr_weights(self, y_data: pd.DataFrame | pd.Series) -> pd.Series:
        """Calculate sample weights from RiskRewardRatio (high/low percentiles boost)."""
        if isinstance(y_data, pd.Series):
            return pd.Series(1.0, index=y_data.index)

        rrr_col = next((col for col in self._RRR_COLUMNS if col in y_data.columns), None)
        if rrr_col is None:
            raise ValueError("y_data must contain 'RiskRewardRatio' or 'RRR' column")

        rrr = y_data[rrr_col].astype(float)
        quantiles = rrr.rank(pct=True)
        weights = np.select([quantiles < 0.2, quantiles > 0.8], [2.0, 2.0], default=0.5).astype(float)

        total_weight = weights.sum()
        if total_weight <= 0:
            return pd.Series(1.0, index=y_data.index)

        weights = weights / total_weight * len(weights)
        return pd.Series(weights, index=y_data.index)

    def featImpMDI(self, fit, featNames: list[str]) -> pd.DataFrame:
        """Mean Decrease in Impurity (normalized across estimators)."""
        importances = np.array([tree.feature_importances_ for tree in fit.estimators_])
        mean_imp = importances.mean(axis=0)
        std_imp = importances.std(axis=0) / np.sqrt(len(importances))
        total = mean_imp.sum()

        if total <= 0:
            return pd.DataFrame(
                {"mean": np.zeros_like(mean_imp), "std": np.zeros_like(std_imp)},
                index=featNames,
            )

        return pd.DataFrame({"mean": mean_imp / total, "std": std_imp / total}, index=featNames)

    def featImpMDA(
        self, clf, X: pd.DataFrame, y_target: pd.Series, sample_weights: pd.Series | None
    ) -> tuple[pd.DataFrame, float]:
        """
        Compute Mean Decrease Accuracy (MDA) feature importance.

        Mac-optimized: Sequential outer loop with thread-based inner parallelism (n_jobs=6).
        """
        cv_splits = self._build_mda_cv_splits(X, y_target)
        X_np = X.to_numpy(copy=False)
        y_np = y_target.to_numpy(copy=False)
        features = X.columns.to_numpy()
        n_folds = len(cv_splits)

        # Convert sample_weights to numpy if provided
        sample_weights_np = None
        if sample_weights is not None:
            sample_weights_np = sample_weights.to_numpy(copy=False)

        # Generate deterministic seeds for each fold
        base_seed = np.random.SeedSequence(self.random_state)
        fold_seeds = [int(s.generate_state(1)[0]) for s in base_seed.spawn(n_folds)]

        # Mac-optimized: Sequential outer loop, thread-based inner parallelism
        # Cap at 6 threads to prevent overheating
        clf_template = clone(clf)
        if hasattr(clf_template, "n_jobs"):
            clf_template.n_jobs = 6  # Fixed: use 6 threads max

        # Always sequential (no outer parallelism on Mac)
        results = [
            _mda_fold_worker(
                fold_idx=fold_i,
                train_idx=train_idx,
                test_idx=test_idx,
                X_np=X_np,
                y_np=y_np,
                sample_weights=sample_weights_np,
                clf_template=clf_template,
                scoring=self.scoring,
                n_perm=self.mda_n_perm,
                fold_seed=fold_seeds[fold_i],
            )
            for fold_i, (train_idx, test_idx) in enumerate(cv_splits)
        ]

        # Aggregate results from workers
        baseline_scores = np.array([r["baseline_score"] for r in results])
        perm_scores = np.array([r["perm_scores"] for r in results])

        baseline_mean = float(np.mean(baseline_scores))
        if self.scoring == "neg_log_loss":
            denom = np.maximum(np.abs(perm_scores), 1e-10)
        else:
            denom = np.maximum(1.0 - perm_scores, 1e-10)

        imp_matrix = (baseline_scores[:, None] - perm_scores) / denom

        importance = pd.DataFrame(
            {
                "mean": imp_matrix.mean(axis=0),
                "std": imp_matrix.std(axis=0) / np.sqrt(n_folds),
            },
            index=features,
        )
        return importance, baseline_mean

    def _build_mda_cv_splits(self, X: pd.DataFrame, y_target: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build CV splits for MDA, with optional shuffling and repeats."""
        X_np = X.to_numpy(copy=False)
        y_np = y_target.to_numpy(copy=False)

        if not self.mda_shuffle:
            base_splits = list(KFold(n_splits=self.cv, shuffle=False).split(X_np))
            return base_splits * self.mda_n_repeats

        # Shuffled CV with repeats
        cv_gen = RepeatedStratifiedKFold(
            n_splits=self.cv,
            n_repeats=self.mda_n_repeats,
            random_state=self.random_state,
        )
        try:
            return list(cv_gen.split(X_np, y_np))
        except ValueError:
            # Fallback: manual shuffled splits if stratification fails
            splits = []
            rng = np.random.default_rng(self.random_state)
            for _ in range(self.mda_n_repeats):
                fold_seed = int(rng.integers(0, 2**31 - 1))
                cv_gen = KFold(n_splits=self.cv, shuffle=True, random_state=fold_seed)
                splits.extend(cv_gen.split(X_np))
            return splits

    def get_harmful_features(
        self,
        importance_df: pd.DataFrame,
        cv: int | None = None,
        confidence_level: float = 0.95,
        min_threshold: float = -np.inf,
    ) -> list[str]:
        """
        Identify harmful features with negative mean and CI upper < 0.

        Returns features sorted by most negative mean first.
        """
        cv_folds = self._resolve_cv_folds(cv)
        df = self._prepare_importance_df(
            importance_df=importance_df,
            min_threshold=min_threshold,
            n_top=len(importance_df) + 1,
            cv_folds=cv_folds,
            confidence_level=confidence_level,
        )
        harmful_df = df[df["harmful"]].sort_values("mean", ascending=True)
        return harmful_df.index.tolist()

    def auxFeatImpSFI(
        self,
        featNames: list[str],
        clf,
        X: pd.DataFrame,
        y_target: pd.Series,
        sample_weights: pd.Series | None,
        cv_gen,
    ) -> pd.DataFrame:
        """Single Feature Importance: score each feature independently."""
        X_np = X.to_numpy(copy=False)
        y_np = y_target.to_numpy(copy=False)
        cv_splits = list(cv_gen.split(X_np))
        col_loc = {feat: X.columns.get_loc(feat) for feat in featNames}

        importance_data = {}
        for feat in featNames:
            col_idx = col_loc[feat]
            scores = np.empty(len(cv_splits), dtype=float)

            for split_i, (train_idx, test_idx) in enumerate(cv_splits):
                X_train_f = X_np[train_idx, col_idx].reshape(-1, 1)
                X_test_f = X_np[test_idx, col_idx].reshape(-1, 1)
                y_train_f = y_np[train_idx]
                y_test_f = y_np[test_idx]
                w_train_f = _slice_weights(sample_weights, train_idx)
                w_test_f = _slice_weights(sample_weights, test_idx)

                fold_clf = clone(clf)
                fold_clf.fit(X_train_f, y_train_f, sample_weight=w_train_f)
                scores[split_i] = self._compute_score(fold_clf, X_test_f, y_test_f, w_test_f)

            importance_data[feat] = {
                "mean": float(scores.mean()),
                "std": float(scores.std() / np.sqrt(len(scores))),
            }

        return pd.DataFrame.from_dict(importance_data, orient="index")

    def _create_base_classifier(self):
        base_clf = DecisionTreeClassifier(
            criterion="entropy",
            max_features=1,
            class_weight="balanced",
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            random_state=self.random_state,
        )
        return BaggingClassifier(
            estimator=base_clf,
            n_estimators=self.n_estimators,
            max_features=1.0,
            max_samples=self.max_samples,
            oob_score=True,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def featImportance(
        self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, method: str = "MDA"
    ) -> tuple[pd.DataFrame, float, float]:
        method_key = method.upper()
        if method_key not in self._SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported methods: {', '.join(self._SUPPORTED_METHODS)}")

        y_target, sample_weights = self._resolve_target_and_weights(y)
        clf = self._create_base_classifier()
        self._fit_classifier(
            clf, X, y_target, sample_weights.to_numpy(copy=False) if sample_weights is not None else None
        )

        oob_score = clf.oob_score_ if hasattr(clf, "oob_score_") else np.nan

        method_dispatch = {
            "MDI": lambda: (
                self.featImpMDI(clf, X.columns.tolist()),
                self._weighted_cv_score(clf, X, y_target, sample_weights),
            ),
            "MDA": lambda: self.featImpMDA(clf, X, y_target, sample_weights),
            "SFI": lambda: self._compute_sfi(clf, X, y_target, sample_weights),
        }

        importance, oos_score = method_dispatch[method_key]()
        return importance, oob_score, oos_score

    def _compute_sfi(self, clf, X, y_target, sample_weights):
        cv_gen = KFold(n_splits=self.cv, shuffle=False)
        importance = self.auxFeatImpSFI(X.columns.tolist(), clf, X, y_target, sample_weights, cv_gen)
        return importance, importance["mean"].mean()

    def _weighted_cv_score(self, clf, X, y_target, sample_weights):
        cv_gen = KFold(n_splits=self.cv, shuffle=False)
        scores = np.empty(self.cv, dtype=float)
        for fold_i, (train_idx, test_idx) in enumerate(cv_gen.split(X)):
            split = self._get_cv_split(X, y_target, sample_weights, train_idx, test_idx)
            fold_clf = clone(clf)
            self._fit_classifier(fold_clf, split.X_train, split.y_train, split.w_train)
            scores[fold_i] = self._compute_score(fold_clf, split.X_test, split.y_test, split.w_test)
        return float(scores.mean())

    def plot_mda_importance(
        self,
        importance_df: pd.DataFrame,
        *,
        top_n: int | None = 30,
        sort_ascending: bool = False,
        title: str = "MDA Feature Importance",
    ) -> plt.Figure:
        """Plot MDA feature importances with confidence bars."""
        self._validate_importance_columns(importance_df)
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be >= 1 when provided")

        df = importance_df[["mean", "std"]].replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            raise ValueError("importance_df has no finite rows to plot")

        if top_n is not None:
            top_idx = df["mean"].abs().sort_values(ascending=False).head(top_n).index
            df = df.loc[top_idx]

        df = df.sort_values("mean", ascending=sort_ascending)
        style = self.plot_style
        fig_height = max(3.0, min(18.0, 0.28 * len(df) + 1.2))

        with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
            fig, ax = plt.subplots(figsize=(10, fig_height), dpi=100)

        y_pos = np.arange(len(df))
        bar_colors = np.where(df["mean"].to_numpy() >= 0.0, style.accent1, style.accent4)
        ax.barh(
            y_pos,
            df["mean"].to_numpy(),
            color=bar_colors,
            alpha=0.9,
            edgecolor="none",
            zorder=3,
        )
        ax.errorbar(
            df["mean"].to_numpy(),
            y_pos,
            xerr=df["std"].to_numpy(),
            fmt="none",
            ecolor=style.muted,
            elinewidth=1.5,
            capsize=3,
            alpha=0.9,
            zorder=4,
        )

        ax.axvline(0.0, color=style.line, linewidth=1.0, alpha=0.6, zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df.index.astype(str))
        ax.invert_yaxis()
        ax.set_xlabel("MDA Importance", fontsize=11, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", alpha=0.25, color=style.grid, linewidth=0.5, zorder=1)

        style.apply_mpl(fig=fig, ax=ax)
        plt.tight_layout()
        return fig

    def plot_mda(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        *,
        top_n: int | None = 30,
        sort_ascending: bool = False,
        title: str = "MDA Feature Importance",
    ) -> tuple[plt.Figure, pd.DataFrame, float, float]:
        """
        Convenience wrapper: compute MDA importance and return a ready-to-show plot.
        Returns (figure, importance_df, oob_score, oos_score).
        """
        importance_df, oob_score, oos_score = self.featImportance(X, y, method="MDA")
        plot_title = f"{title}\nOOB={oob_score:.4f}, OOS={oos_score:.4f}"
        fig = self.plot_mda_importance(
            importance_df=importance_df,
            top_n=top_n,
            sort_ascending=sort_ascending,
            title=plot_title,
        )
        return fig, importance_df, oob_score, oos_score

    def print_feature_importance_summary(
        self,
        importance_df: pd.DataFrame,
        n_top: int = 20,
        cv: int | None = None,
        confidence_level: float = 0.95,
        min_threshold: float = -np.inf,
        show_harmful: bool = True,
    ) -> None:
        self._validate_importance_columns(importance_df)
        cv_folds = self._resolve_cv_folds(cv)
        df = self._prepare_importance_df(importance_df, min_threshold, n_top, cv_folds, confidence_level)
        if len(df) == 0:
            self._print_empty_summary(min_threshold)
            return
        self._print_summary_header(df, importance_df, n_top, cv_folds, confidence_level)
        self._print_beneficial_features(df[df["mean"] > 0].sort_values("mean", ascending=False))
        if show_harmful:
            self._print_harmful_features(df[df["mean"] < 0].sort_values("mean", ascending=True))
        self._print_summary_footer(df, confidence_level)

    def _prepare_importance_df(self, importance_df, min_threshold, n_top, cv_folds, confidence_level):
        df = importance_df[importance_df["mean"] >= min_threshold].copy()
        df["abs_mean"] = df["mean"].abs()
        df = df.sort_values("abs_mean", ascending=False).head(n_top)
        if len(df) == 0:
            return df
        df["se"] = df["std"] / np.sqrt(cv_folds)
        t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, cv_folds - 1)
        df["ci_lower"] = df["mean"] - t_critical * df["se"]
        df["ci_upper"] = df["mean"] + t_critical * df["se"]
        df["snr"] = np.where(df["std"] > 1e-10, df["mean"].abs() / df["std"], np.inf)
        df["beneficial"] = (df["mean"] > 0) & (df["ci_lower"] > 0)
        df["harmful"] = (df["mean"] < 0) & (df["ci_upper"] < 0)
        df_positive = df[df["mean"] > 0].sort_values("mean", ascending=False)
        total_positive = df_positive["mean"].sum()
        if total_positive > 0:
            df.loc[df["mean"] > 0, "cumulative_pct"] = df_positive["mean"].cumsum() / total_positive * 100
        return df

    def select_features_from_importance(
        self,
        importance_df: pd.DataFrame,
        *,
        cv: int | None = None,
        confidence_level: float = 0.95,
        min_threshold: float = -np.inf,
        min_features: int = 10,
        max_features: int | None = None,
        fallback_mode: str = "snr_positive",
    ) -> dict[str, object]:
        """
        Select features via confidence intervals with fallback to ranking.

        Primary: Keep beneficial features (mean > 0, CI lower > 0).
        Remove harmful features (mean < 0, CI upper < 0).
        Fallback: Rank non-harmful by SNR/mean if insufficient beneficial features.
        """
        self._validate_importance_columns(importance_df)
        if min_features < 1:
            raise ValueError("min_features must be >= 1")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be >= 1 when provided")
        if fallback_mode not in self._SUPPORTED_FALLBACK_MODES:
            raise ValueError(f"fallback_mode must be one of {self._SUPPORTED_FALLBACK_MODES}")

        cv_folds = self._resolve_cv_folds(cv)

        df = self._prepare_importance_df(
            importance_df=importance_df,
            min_threshold=min_threshold,
            n_top=len(importance_df),
            cv_folds=cv_folds,
            confidence_level=confidence_level,
        )
        if df.empty:
            return {
                "selected_features": [],
                "removed_features": list(importance_df.index),
                "selection_mode": "empty",
                "summary_df": df,
            }

        strict_keep = df[df["beneficial"]].sort_values("mean", ascending=False).copy()
        harmful = set(df.index[df["harmful"]])
        selected = list(strict_keep.index)

        if max_features is not None and len(selected) > max_features:
            selected = selected[:max_features]

        target_min = min_features if max_features is None else min(min_features, max_features)
        selection_mode = "strict"

        if len(selected) < target_min:
            fallback_pool = df.loc[~df.index.isin(harmful)].copy()
            if fallback_mode == "snr_positive":
                fallback_pool["fallback_score"] = np.where(fallback_pool["mean"] > 0, fallback_pool["snr"], 0.0)
            elif fallback_mode == "mean_positive":
                fallback_pool["fallback_score"] = np.where(fallback_pool["mean"] > 0, fallback_pool["mean"], 0.0)
            else:  # abs_snr
                fallback_pool["fallback_score"] = fallback_pool["snr"]

            fallback_pool = fallback_pool.sort_values(
                by=["fallback_score", "mean", "abs_mean"],
                ascending=[False, False, False],
            )
            fallback_features = list(fallback_pool.index[:target_min])
            selected = list(dict.fromkeys(selected + fallback_features))
            selection_mode = "fallback"

        if max_features is not None and len(selected) > max_features:
            selected = selected[:max_features]

        selected_set = set(selected)
        removed_features = [feature for feature in importance_df.index if feature not in selected_set]

        summary_df = df.copy()
        summary_df["selected"] = summary_df.index.isin(selected_set)

        return {
            "selected_features": selected,
            "removed_features": removed_features,
            "selection_mode": selection_mode,
            "summary_df": summary_df,
            "n_beneficial": int(df["beneficial"].sum()),
            "n_harmful": int(df["harmful"].sum()),
            "n_selected": len(selected),
        }

    def _print_empty_summary(self, min_threshold: float) -> None:
        """Print message when no features meet threshold."""
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE SUMMARY")
        print("=" * 70)
        print(f"No features with mean importance >= {min_threshold:.4f}")

    def _print_summary_header(
        self,
        df: pd.DataFrame,
        importance_df: pd.DataFrame,
        n_top: int,
        cv_folds: int,
        confidence_level: float,
    ) -> None:
        """Print summary header with feature and statistical info."""
        print("\n" + "=" * 120)
        print("FEATURE IMPORTANCE SUMMARY (Statistical Analysis)")
        print("=" * 120)
        print(f"Confidence Level: {confidence_level * 100:.1f}% (t-distribution, df={cv_folds - 1})")
        print(f"Cross-Validation Folds: {cv_folds}")
        print(f"Total Features Analyzed: {len(importance_df)}")
        print(f"Features Displayed: {len(df)} (top {n_top} by absolute value)")
        print(f"Beneficial Features (mean > 0, CI > 0): {df['beneficial'].sum()}")
        print(f"Harmful Features (mean < 0, CI < 0): {df['harmful'].sum()}")
        print(f"Uncertain Features (CI crosses 0): {(~(df['beneficial'] | df['harmful'])).sum()}")
        print("=" * 120)

    def _print_beneficial_features(self, df_beneficial: pd.DataFrame) -> None:
        """Print table of beneficial features sorted by mean importance."""
        if len(df_beneficial) == 0:
            return

        print("\n" + "─" * 120)
        print("BENEFICIAL FEATURES (Positive Importance)")
        print("─" * 120)
        header = (
            f"{'Rank':<6} {'Feature':<35} {'Mean':<10} {'Std':<10} "
            f"{'CI Lower':<10} {'CI Upper':<10} {'SNR':<8} {'Sig':<5} {'Cum %':<8}"
        )
        print(header)
        print("─" * 120)

        for rank, (feat, row) in enumerate(df_beneficial.iterrows(), start=1):
            sig = "✓✓✓" if row["beneficial"] else "~"
            snr = f"{row['snr']:.2f}" if np.isfinite(row["snr"]) else "inf"
            cum = row.get("cumulative_pct", 0.0)
            print(
                f"{rank:<6} {str(feat)[:34]:<35} {row['mean']:<10.6f} "
                f"{row['std']:<10.6f} {row['ci_lower']:<10.6f} "
                f"{row['ci_upper']:<10.6f} {snr:<8} {sig:<5} {cum:<8.2f}"
            )

    def _print_harmful_features(self, df_harmful: pd.DataFrame) -> None:
        """Print table of harmful features sorted by most negative mean."""
        if len(df_harmful) == 0:
            return

        print("\n" + "─" * 120)
        print("HARMFUL FEATURES (Negative Importance)")
        print("─" * 120)
        header = (
            f"{'Rank':<6} {'Feature':<35} {'Mean':<10} {'Std':<10} "
            f"{'CI Lower':<10} {'CI Upper':<10} {'SNR':<8} {'Sig':<5}"
        )
        print(header)
        print("─" * 120)

        for rank, (feat, row) in enumerate(df_harmful.iterrows(), start=1):
            sig = "✗✗✗" if row["harmful"] else "~"
            snr = f"{row['snr']:.2f}" if np.isfinite(row["snr"]) else "inf"
            print(
                f"{rank:<6} {str(feat)[:34]:<35} {row['mean']:<10.6f} "
                f"{row['std']:<10.6f} {row['ci_lower']:<10.6f} "
                f"{row['ci_upper']:<10.6f} {snr:<8} {sig:<5}"
            )

    def _print_summary_footer(self, df: pd.DataFrame, confidence_level: float) -> None:
        """Print summary statistics and interpretation guide."""
        print("\n" + "─" * 120)
        print(f"Statistically Beneficial: {df['beneficial'].sum()}/{len(df)}")
        print(f"Statistically Harmful: {df['harmful'].sum()}/{len(df)}")

        df_beneficial = df[df["mean"] > 0]
        if len(df_beneficial) > 0:
            total_pos = df_beneficial["mean"].sum()
            top10_pct = df_beneficial.head(10)["mean"].sum() / total_pos * 100
            finite_snr = df_beneficial["snr"][np.isfinite(df_beneficial["snr"])]
            print(f"Top 10 Cumulative Importance: {top10_pct:.2f}%")
            if len(finite_snr) > 0:
                print(f"Mean SNR (beneficial): {finite_snr.mean():.2f}")

        print("\n" + "=" * 120)
        print("INTERPRETATION: ✓✓✓=KEEP, ✗✗✗=REMOVE, ~=Uncertain")
        print(f"CI = {confidence_level * 100:.0f}% Confidence Interval using t-distribution")
        print("=" * 120 + "\n")

    def _zscore(self, X: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize features (replace zero std with 0 after centering)."""
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0).replace(0.0, np.nan)
        return ((X - mu) / sd).fillna(0.0)

    @staticmethod
    def _validate_analysis_frame(analysis_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
        required_cols = {"feature", "mdi_importance", "mdi_std", x_col}
        missing = required_cols.difference(analysis_df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"analysis_df is missing required columns: {missing_str}")

        cleaned = (
            analysis_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, "mdi_importance", "mdi_std"]).copy()
        )
        if cleaned.empty:
            raise ValueError("analysis_df has no finite rows for plotting/analysis")
        return cleaned

    @staticmethod
    def _compute_correlation_stats(x_vals: np.ndarray, y_vals: np.ndarray) -> dict[str, float]:
        """Compute Pearson and Kendall correlations, return NaN if insufficient data."""
        out = {
            "pearson_corr": np.nan,
            "pearson_pvalue": np.nan,
            "weighted_kendall": np.nan,
            "kendall_pvalue": np.nan,
        }
        if x_vals.size < 2:
            return out

        pearson_corr, pearson_p = stats.pearsonr(y_vals, x_vals)
        out["pearson_corr"] = float(pearson_corr)
        out["pearson_pvalue"] = float(pearson_p)

        weighted_kendall, _ = stats.weightedtau(y_vals, x_vals)
        out["weighted_kendall"] = float(weighted_kendall)

        _, kendall_p = stats.kendalltau(y_vals, x_vals)
        out["kendall_pvalue"] = float(kendall_p)

        return out

    @staticmethod
    def _should_use_log_scale(x_vals: np.ndarray) -> bool:
        finite = x_vals[np.isfinite(x_vals)]
        if finite.size < 2 or np.any(finite <= 0):
            return False
        return (finite.max() / finite.min()) > 100

    def get_feature_spectral_scores(
        self,
        X: pd.DataFrame,
        *,
        n_components: int | None = None,
        method: str = "lambda_loading2",
    ) -> pd.DataFrame:
        n_components = n_components or min(X.shape)
        Xz = self._zscore(X)
        pca = PCA(n_components=n_components, svd_solver="full", random_state=self.random_state)
        pca.fit(Xz.values)
        lambdas = pca.explained_variance_
        loadings = pca.components_.T * np.sqrt(lambdas)
        loading2 = loadings**2

        score_map = {
            "lambda_loading2": (loading2 @ lambdas, "spectral_score"),
            "communality": (loading2.sum(axis=1), "communality"),
        }
        if method not in score_map:
            raise ValueError(f"Unknown method='{method}'")

        scores, score_name = score_map[method]
        top_pc_idx = np.argmax(np.abs(loadings), axis=1)
        return pd.DataFrame(
            {
                "feature": X.columns,
                score_name: scores,
                "top_pc": top_pc_idx + 1,
                "top_pc_loading_abs": np.max(np.abs(loadings), axis=1),
                "top_pc_eigenvalue": lambdas[top_pc_idx],
            }
        ).set_index("feature")

    def create_mdi_eigenvalue_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        *,
        n_components: int | None = None,
        spectral_method: str = "lambda_loading2",
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        mdi_imp, _, _ = self.featImportance(X, y, method="MDI")
        spec = self.get_feature_spectral_scores(X, n_components=n_components, method=spectral_method)

        analysis_df = (
            pd.DataFrame(
                {
                    "mdi_importance": mdi_imp["mean"].values,
                    "mdi_std": mdi_imp["std"].values,
                },
                index=X.columns,
            )
            .join(spec, how="inner")
            .reset_index(names="feature")
        )
        x_col = "spectral_score" if spectral_method == "lambda_loading2" else "communality"
        analysis_df = self._validate_analysis_frame(analysis_df, x_col)
        x_vals = analysis_df[x_col].to_numpy(dtype=float)
        y_vals = analysis_df["mdi_importance"].to_numpy(dtype=float)

        stats_results = self._compute_correlation_stats(x_vals, y_vals)
        stats_results["x_col"] = x_col
        return analysis_df, stats_results

    def plot_mdi_eigenvalue_analysis(
        self,
        analysis_df: pd.DataFrame,
        stats_results: dict,
        show_labels: bool = True,
    ) -> plt.Figure:
        x_col = stats_results.get("x_col", "spectral_score")
        clean_df = self._validate_analysis_frame(analysis_df, x_col)
        x_vals = clean_df[x_col].to_numpy(dtype=float)
        use_log = self._should_use_log_scale(x_vals)

        with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        if use_log:
            ax.set_xscale("log")

        style = self.plot_style
        ax.scatter(
            clean_df[x_col],
            clean_df["mdi_importance"],
            c=style.accent1,
            s=80,
            alpha=0.8,
            edgecolors=style.line,
            linewidth=1.0,
            label="RRR-Weighted Features",
            zorder=3,
        )
        ax.errorbar(
            clean_df[x_col],
            clean_df["mdi_importance"],
            yerr=clean_df["mdi_std"],
            fmt="none",
            ecolor=style.muted,
            alpha=0.6,
            capsize=3,
            capthick=1,
            zorder=1,
        )

        self._add_trend_line(ax, clean_df, x_col, use_log, style)
        if show_labels:
            self._add_feature_labels(ax, clean_df, x_col, style)

        pearson_corr = stats_results.get("pearson_corr", np.nan)
        pearson_pvalue = stats_results.get("pearson_pvalue", np.nan)
        weighted_kendall = stats_results.get("weighted_kendall", np.nan)
        title = (
            "RRR-Weighted MDI vs PCA Spectral Score\n"
            f"Pearson r = {pearson_corr:.4f} (p = {pearson_pvalue:.4f}), "
            f"Weighted Kendall tau = {weighted_kendall:.4f}"
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel(f"{x_col}{' (log scale)' if use_log else ''}", fontsize=12, fontweight="bold")
        ax.set_ylabel("RRR-Weighted MDI Feature Importance", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10, frameon=True)
        ax.set_axisbelow(True)
        style.apply_mpl(fig=fig, ax=ax)
        plt.tight_layout()
        return fig

    def _add_trend_line(self, ax, df, x_col, use_log, style):
        valid = np.isfinite(df[x_col].values) & np.isfinite(df["mdi_importance"].values)
        if valid.sum() <= 2:
            return

        xv = df.loc[valid, x_col].values.astype(float)
        yv = df.loc[valid, "mdi_importance"].values.astype(float)
        if use_log:
            if np.any(xv <= 0):
                return
            xv_fit = np.log(xv)
            if np.unique(xv_fit).size < 2:
                return
            coeffs = np.polyfit(xv_fit, yv, 1)
            x_line = np.linspace(xv_fit.min(), xv_fit.max(), 200)
            ax.plot(
                np.exp(x_line),
                np.poly1d(coeffs)(x_line),
                color=style.accent2,
                linestyle="--",
                linewidth=2.5,
                alpha=0.9,
                label=f"Trend (slope={coeffs[0]:.4f} per log-x)",
            )
            return

        if np.unique(xv).size < 2:
            return
        coeffs = np.polyfit(xv, yv, 1)
        x_line = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(
            x_line,
            np.poly1d(coeffs)(x_line),
            color=style.accent2,
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label=f"Trend (slope={coeffs[0]:.4f})",
        )

    def _add_feature_labels(self, ax, df, x_col, style):
        n_labels = min(25, len(df))
        for _, row in df.nlargest(n_labels, "mdi_importance").iterrows():
            ax.annotate(
                row["feature"],
                (row[x_col], row["mdi_importance"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=style.font_color,
                alpha=0.9,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=style.plot_bgcolor,
                    edgecolor=style.line,
                    alpha=0.8,
                ),
            )
