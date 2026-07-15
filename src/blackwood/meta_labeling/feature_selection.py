import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from blackwood.config import RANDOM_STATE
from blackwood.data.splitters import CPCVSplitter
from blackwood.meta_labeling.selection import (
    MDAConfig,
    PCAEigenvalueAnalysis,
    RMTCorrelationProcessor,
)
from blackwood.meta_labeling.utils import (
    append_strategy_state_features_to_cpcv_paths,
    existing_columns,
)


@dataclass(frozen=True)
class SelectionPassConfig:
    cv: int
    n_repeats: int
    feature_n_perm: int
    cluster_n_perm: int
    n_estimators: int


@dataclass(frozen=True)
class TwoPassSelectionConfig:
    threshold: float = 0.5
    refine_band_frac: float = 0.15
    dtype: object = np.float32
    debug: bool = True
    n_jobs: int = 4
    min_active_features: int = 2
    entry_time_col: str = "EntryTime"

    analyzer_variance_threshold: float = 0.99
    pca_variance_threshold: float = 0.95
    pca_feature_variance_threshold: float = 0.02
    pca_loading_threshold: float = 0.3

    cluster_negative_mean_threshold: float = 0.0
    feature_nonsig_margin_threshold: float = 0.0

    mda_scoring: str = "neg_log_loss"
    mda_disable_progress: bool = True

    tree_criterion: str = "gini"
    tree_max_features: str = "sqrt"
    tree_class_weight: object = None
    tree_min_samples_leaf: int = 50

    bagging_max_features: float = 0.8
    bagging_max_samples: float = 0.8
    bagging_oob_score: bool = False
    remove_market_mode: bool = False

    pass1: SelectionPassConfig = field(
        default_factory=lambda: SelectionPassConfig(
            cv=3,
            n_repeats=1,
            feature_n_perm=12,
            cluster_n_perm=8,
            n_estimators=20,
        )
    )
    pass2: SelectionPassConfig = field(
        default_factory=lambda: SelectionPassConfig(
            cv=5,
            n_repeats=1,
            feature_n_perm=35,
            cluster_n_perm=20,
            n_estimators=35,
        )
    )


FEATURE_SELECTION_CONFIG = TwoPassSelectionConfig()


BALANCED_FEATURE_SELECTION_CONFIG = TwoPassSelectionConfig(
    threshold=0.60,
    refine_band_frac=0.20,
    pca_feature_variance_threshold=0.015,
    pca_loading_threshold=0.25,
    cluster_negative_mean_threshold=-0.005,
    feature_nonsig_margin_threshold=-0.005,
    pass1=SelectionPassConfig(cv=3, n_repeats=1, feature_n_perm=12, cluster_n_perm=8, n_estimators=20),
    pass2=SelectionPassConfig(cv=5, n_repeats=1, feature_n_perm=35, cluster_n_perm=20, n_estimators=35),
)


def _stable_set(counter: Counter[str], min_count: int) -> set[str]:
    return {feature for feature, count in counter.items() if count >= min_count}


def _near_threshold_set(counter: Counter[str], min_count: int, band: int) -> set[str]:
    low = max(0, min_count - band)
    high = min_count + band
    return {feature for feature, count in counter.items() if low <= count <= high}


def _merge_stable_sets(counters: Sequence[Counter[str]], min_count: int) -> set[str]:
    merged: set[str] = set()
    for counter in counters:
        merged.update(_stable_set(counter, min_count))
    return merged


def _merge_near_threshold_sets(
    counters: Sequence[Counter[str]],
    min_count: int,
    band: int,
) -> set[str]:
    merged: set[str] = set()
    for counter in counters:
        merged.update(_near_threshold_set(counter, min_count, band))
    return merged


def _build_mda_configs(
    pass_cfg: SelectionPassConfig,
    cfg: TwoPassSelectionConfig,
) -> tuple[MDAConfig, MDAConfig]:
    common_kwargs = {
        "cv": pass_cfg.cv,
        "n_repeats": pass_cfg.n_repeats,
        "scoring": cfg.mda_scoring,
        "random_state": RANDOM_STATE,
        "disable_progress": cfg.mda_disable_progress,
    }
    return (
        MDAConfig(n_perm=pass_cfg.feature_n_perm, **common_kwargs),
        MDAConfig(n_perm=pass_cfg.cluster_n_perm, **common_kwargs),
    )


def _make_selector_clf(
    pass_cfg: SelectionPassConfig,
    cfg: TwoPassSelectionConfig,
) -> BaggingClassifier:
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(
            criterion=cfg.tree_criterion,
            max_features=cfg.tree_max_features,
            class_weight=cfg.tree_class_weight,
            min_samples_leaf=cfg.tree_min_samples_leaf,
            random_state=RANDOM_STATE,
        ),
        n_estimators=pass_cfg.n_estimators,
        max_features=cfg.bagging_max_features,
        max_samples=cfg.bagging_max_samples,
        oob_score=cfg.bagging_oob_score,
        random_state=RANDOM_STATE,
        n_jobs=6,  # Mac-optimized, max 6 threads
    )


def _path_ids(paths: Iterable) -> list:
    if hasattr(paths, "keys"):
        return list(paths.keys())
    return list(paths)


def _binary_meta_auc(y_true, y_score) -> float:
    return float(
        roc_auc_score(
            np.asarray(y_true, dtype=np.int8),
            np.asarray(y_score, dtype=float),
        )
    )


def _process_path_worker(
    train_df: pd.DataFrame,
    y_train: pd.DataFrame,
    active_features: list[str],
    cfg: TwoPassSelectionConfig,
    pass_cfg: SelectionPassConfig,
    show_plots: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Process a single CPCV path: PCA + correlation denoising + cluster/feature MDA.

    Top-level function for pickle/spawn compatibility with joblib.
    Returns (pca_drop, negative_cluster_features, nonsig_features) as plain lists.
    """
    analyzer = PCAEigenvalueAnalysis(
        variance_threshold=cfg.analyzer_variance_threshold,
        random_state=RANDOM_STATE,
    )
    processor = RMTCorrelationProcessor(remove_market_mode=cfg.remove_market_mode)
    clf = _make_selector_clf(pass_cfg=pass_cfg, cfg=cfg)
    feature_cfg, cluster_cfg = _build_mda_configs(pass_cfg=pass_cfg, cfg=cfg)

    X_tr = train_df.loc[:, active_features].copy().astype(cfg.dtype, copy=False)
    y_tr = y_train.loc[X_tr.index]
    y_lbl = y_tr["meta_label"]
    sample_weights = processor.transform_rrr_to_sample_weights(y_tr["RiskRewardRatio"]).astype(cfg.dtype, copy=False)

    _, _, pca_analysis = analyzer.compute_pca_eigenvalues(
        X_tr,
        variance_threshold=cfg.pca_variance_threshold,
        standardize=True,
        verbose=False,
    )
    plot_figures = []
    if show_plots:
        plot_figures.append(
            analyzer.plot_pca_eigenvalues(
                pca_analysis=pca_analysis,
                variance_threshold=cfg.pca_variance_threshold,
            )
        )
    pca_drop = analyzer.get_features_to_remove(
        X=X_tr,
        pca_analysis=pca_analysis,
        variance_threshold=cfg.pca_feature_variance_threshold,
        loading_threshold=cfg.pca_loading_threshold,
    )

    pca_drop_set = set(pca_drop)
    mda_cols = [f for f in active_features if f not in pca_drop_set]

    neg_cluster_features: list[str] = []
    nonsig_features: list[str] = []

    if len(mda_cols) < cfg.min_active_features:
        if show_plots and plot_figures:
            plt.show()
            for fig in plot_figures:
                plt.close(fig)
        return list(pca_drop), neg_cluster_features, nonsig_features

    X_mda = X_tr.loc[:, mda_cols]
    corr_transformed, _ = processor.denoise_detone_corr(X_mda)
    clusters = processor.onc_clustering(corr_transformed)

    if show_plots:
        fig_transformation, _ = processor.visualize_transformation(X_mda)
        fig_clustering, _ = processor.visualize_onc_clustering(X_mda)
        plot_figures.extend([fig_transformation, fig_clustering])
        plt.show()
        for fig in plot_figures:
            plt.close(fig)

    if clusters:
        cluster_mda, _, _ = processor.cluster_importance_mda(
            clf=clf,
            X=X_mda,
            y=y_lbl,
            clusters=clusters,
            sample_weights=sample_weights,
            config=cluster_cfg,
        )
        negative_cids = (
            cluster_mda.index[cluster_mda["mean"] < cfg.cluster_negative_mean_threshold]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .dropna()
            .astype(int)
            .tolist()
        )
        if negative_cids:
            neg_cluster_features = list(chain.from_iterable(clusters.get(cid, ()) for cid in negative_cids))

    feat_mda, _, _ = processor.feature_importance_mda(
        clf=clf,
        X=X_mda,
        y=y_lbl,
        sample_weights=sample_weights,
        config=feature_cfg,
    )
    nonsig_features = feat_mda.index[
        (feat_mda["mean"] - feat_mda["std"]) < cfg.feature_nonsig_margin_threshold
    ].tolist()

    return list(pca_drop), neg_cluster_features, nonsig_features


def rank_features_by_cpcv_auc(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    paths,
    splitter: CPCVSplitter,
    remaining_features: Sequence[str],
    *,
    state_windows: tuple[int, ...] = (20, 50, 1000),
    entry_time_col: str = "EntryTime",
    append_state_features: bool = False,
    strip_feature_prefix: str | None = "Entry_",
    print_summary: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Score each feature by path-wise CPCV train/validation AUC and aggregate by mean/std.

    When ``append_state_features`` is True, strategy-state columns are recomputed per CPCV path
    with causal history using ``append_strategy_state_features_to_cpcv_paths``.
    """
    features = list(dict.fromkeys(remaining_features))  # preserve order, dedupe
    if not features:
        raise ValueError("remaining_features is empty.")

    paths_for_eval = (
        append_strategy_state_features_to_cpcv_paths(
            splitter=splitter,
            X=X_train,
            y_state=y_train,
            paths=paths,
            windows=state_windows,
            entry_time_col=entry_time_col,
        )
        if append_state_features
        else paths
    )

    auc_rows: list[tuple[str, int, float, float]] = []
    for path_id in _path_ids(paths_for_eval):
        train_df, test_df = splitter.get_train_test_for_path(X_train, paths_for_eval, path_id)

        X_tr = train_df[features]
        X_te = test_df[features]

        meta_train = y_train["meta_label"].loc[train_df.index].to_numpy(copy=False)
        meta_test = y_train["meta_label"].loc[test_df.index].to_numpy(copy=False)

        for col in features:
            pred_train = X_tr[col].to_numpy(copy=False)
            pred_test = X_te[col].to_numpy(copy=False)

            auc_train = _binary_meta_auc(meta_train, pred_train)
            auc_val = _binary_meta_auc(meta_test, pred_test)

            auc_rows.append((col, path_id, auc_train, auc_val))

    df_auc = pd.DataFrame(auc_rows, columns=["feature", "path_id", "auc_train", "auc_val"])

    agg_auc = (
        df_auc.groupby("feature")
        .agg(
            mean_auc_train=("auc_train", "mean"),
            std_auc_train=("auc_train", "std"),
            mean_auc_val=("auc_val", "mean"),
            std_auc_val=("auc_val", "std"),
        )
        .reset_index()
        .sort_values("mean_auc_val", ascending=False)
    )

    if strip_feature_prefix:
        agg_auc["feature"] = agg_auc["feature"].str.removeprefix(strip_feature_prefix)

    if print_summary:
        print("Features by mean validation AUC:")
        print(
            agg_auc.round(3).to_string(
                index=False,
                justify="right",
                col_space=14,
                float_format=lambda x: f"{x:.3f}",
                line_width=200,
            )
        )

    return df_auc, agg_auc, paths_for_eval


def _run_selection_pass(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    paths: Iterable,
    splitter,
    active_features: list[str],
    cfg: TwoPassSelectionConfig,
    pass_cfg: SelectionPassConfig,
    verbose: bool = False,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    pca_counts: Counter[str] = Counter()
    negative_cluster_counts: Counter[str] = Counter()
    nonsig_mda_counts: Counter[str] = Counter()

    if len(active_features) < cfg.min_active_features:
        return pca_counts, negative_cluster_counts, nonsig_mda_counts

    # Pre-extract train DataFrames (avoids pickling splitter)
    path_ids = _path_ids(paths)
    path_data: list[pd.DataFrame] = []
    for path_id in path_ids:
        train_df, _ = splitter.get_train_test_for_path(X_train, paths, path_id)
        missing_features = [f for f in active_features if f not in train_df.columns]
        if missing_features:
            raise ValueError(f"Missing active features in path train_df (path_id={path_id}): {missing_features[:10]}")
        path_data.append(train_df)

    results = []
    remaining_path_data = path_data

    if verbose and path_data:
        results.append(
            _process_path_worker(
                path_data[0],
                y_train,
                active_features,
                cfg,
                pass_cfg,
                show_plots=True,
            )
        )
        remaining_path_data = path_data[1:]

    if cfg.n_jobs == 1 or len(remaining_path_data) <= 1:
        results.extend(
            _process_path_worker(
                td,
                y_train,
                active_features,
                cfg,
                pass_cfg,
                show_plots=False,
            )
            for td in remaining_path_data
        )
    else:
        results.extend(
            Parallel(n_jobs=cfg.n_jobs, backend="loky")(
                delayed(_process_path_worker)(
                    td,
                    y_train,
                    active_features,
                    cfg,
                    pass_cfg,
                    False,
                )
                for td in remaining_path_data
            )
        )

    for pca_drop, neg_feats, nonsig_feats in results:
        pca_counts.update(pca_drop)
        negative_cluster_counts.update(neg_feats)
        nonsig_mda_counts.update(nonsig_feats)

    return pca_counts, negative_cluster_counts, nonsig_mda_counts


def select_stable_features_two_pass(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    paths: Iterable,
    splitter,
    cfg: TwoPassSelectionConfig = FEATURE_SELECTION_CONFIG,
    features_to_exclude: Sequence[str] | None = None,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Select stable features using two-pass CPCV and optional diagnostic plotting.

    When ``verbose`` is ``False`` (default), no plots are generated.
    When ``verbose`` is ``True``, PCA and RMT diagnostic plots are shown for the
    first path in each pass.
    """
    excluded_set = set(features_to_exclude or ())
    excluded_missing = sorted(col for col in excluded_set if col not in X_train.columns and col != cfg.entry_time_col)
    excluded_present = sorted(col for col in excluded_set if col in X_train.columns and col != cfg.entry_time_col)
    feature_cols = [col for col in X_train.columns if col != cfg.entry_time_col and col not in excluded_set]
    n_paths = len(_path_ids(paths))
    if n_paths == 0:
        return excluded_present, feature_cols

    min_count = math.ceil(n_paths * cfg.threshold)
    band = max(1, math.ceil(n_paths * cfg.refine_band_frac))

    pca_1, neg_1, nonsig_1 = _run_selection_pass(
        X_train=X_train,
        y_train=y_train,
        paths=paths,
        splitter=splitter,
        active_features=feature_cols,
        cfg=cfg,
        pass_cfg=cfg.pass1,
        verbose=verbose,
    )
    pass1_counters = (pca_1, neg_1, nonsig_1)
    pass1_remove = _merge_stable_sets(pass1_counters, min_count)
    borderline = sorted(_merge_near_threshold_sets(pass1_counters, min_count, band) & set(feature_cols))

    if len(borderline) >= cfg.min_active_features:
        pca_2, neg_2, nonsig_2 = _run_selection_pass(
            X_train=X_train,
            y_train=y_train,
            paths=paths,
            splitter=splitter,
            active_features=borderline,
            cfg=cfg,
            pass_cfg=cfg.pass2,
            verbose=verbose,
        )
    else:
        pca_2, neg_2, nonsig_2 = Counter(), Counter(), Counter()

    remove_set = set(pass1_remove)
    pass2_counters = (pca_2, neg_2, nonsig_2)
    for feature in borderline:
        pass2_counts = [counter[feature] for counter in pass2_counters]
        if not any(pass2_counts):
            continue
        if any(count >= min_count for count in pass2_counts):
            remove_set.add(feature)
        else:
            remove_set.discard(feature)

    features_to_remove = sorted(remove_set.union(excluded_present))
    remaining_features = [col for col in feature_cols if col not in remove_set]

    if cfg.debug:
        if excluded_missing:
            print(f"features_to_exclude not found in X_train ({len(excluded_missing)}): {excluded_missing[:10]}")
        print(
            f"paths={n_paths} min_count={min_count} band={band} | "
            f"excluded={len(excluded_present)} "
            f"pass1_remove={len(pass1_remove)} borderline={len(borderline)} "
            f"final_remove={len(features_to_remove)}"
        )

    return features_to_remove, remaining_features


def apply_feature_selection_columns(
    X_train: pd.DataFrame,
    remaining_features: Sequence[str],
    X_cal: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    entry_time_col: str = "EntryTime",
    features_to_exclude: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    exclude_set = set(features_to_exclude or ())
    use_cols = [col for col in remaining_features if col not in exclude_set]
    if entry_time_col in X_train.columns and entry_time_col not in exclude_set:
        use_cols.append(entry_time_col)
    use_cols = list(dict.fromkeys(use_cols))

    X_train_selected = X_train.loc[:, use_cols]
    X_cal_selected = X_cal.loc[:, _existing_columns(X_cal, use_cols)].copy() if X_cal is not None else None
    X_test_selected = X_test.loc[:, _existing_columns(X_test, use_cols)].copy() if X_test is not None else None
    return X_train_selected, X_cal_selected, X_test_selected, use_cols
