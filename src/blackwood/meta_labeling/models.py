import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import spearmanr
from sklearn.ensemble import BaggingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from blackwood.config import RANDOM_STATE


class BinaryModelXGBoost:
    def __init__(self, random_state: int = RANDOM_STATE, default_param_dist: dict[str, Any] | None = None):
        self.random_state = random_state
        self.default_param_dist = default_param_dist
        self.feature_names = None

    def _create_lr_decay_function(self, initial_lr: float, decay_rate: float = 0.995, min_lr: float = 1e-4) -> Callable:
        def exponential_decay(current_iter: int) -> float:
            lr = initial_lr * np.power(decay_rate, current_iter)
            return max(lr, min_lr)

        return exponential_decay

    def _score_predictions(
        self, scoring: str, y_true: np.ndarray, y_pred: np.ndarray | None = None, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """Compute a scalar score for Optuna objective."""
        if scoring == "f1":
            if y_pred is None:
                raise ValueError("y_pred is required for f1 scoring.")
            return float(precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[2])
        if scoring == "roc_auc":
            if y_pred_proba is None:
                raise ValueError("y_pred_proba is required for roc_auc scoring.")
            return float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0
        if scoring == "neg_log_loss":
            if y_pred_proba is None:
                raise ValueError("y_pred_proba is required for neg_log_loss scoring.")
            return float(-log_loss(y_true, y_pred_proba))
        if y_pred is None:
            raise ValueError("y_pred is required for accuracy scoring.")
        return float(accuracy_score(y_true, y_pred))

    def optimize_hyperparameters_optuna(
        self,
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xdev: pd.DataFrame,
        ydev: np.ndarray,
        n_trials: int = 100,
        cv_folds: int = 3,
        scoring: str = "neg_log_loss",
        n_jobs: int = 1,
        timeout: int | None = None,
        search_space_fn: Callable | None = None,
        # Time-series CV controls
        tscv_test_size: int | None = None,
        tscv_gap: int = 0,
        es_fraction: float = 0.2,
        use_external_val: bool = True,
        # Bagging controls
        enable_bagging: bool = True,
        bagging_n_jobs: int = -1,  # Parallel bagging (independent of optuna n_jobs)
    ) -> dict:
        r"""
        Hyperparameter optimization with Optuna using StratifiedKFold and optional bagging.

        Assumptions:
            - Xtr/ytr and Xdev/ydev are temporally ordered (earlier rows = earlier time)
            - No missing values in features (user preprocessed data)
            - Binary target (0/1)

        Notes:
            - Bagging memory: ~n_bags × single_model_size (e.g., 10 bags = 10× memory)
            - Bagging training time: ~n_bags × base_time (mitigated by bagging_n_jobs=-1)

        """
        if isinstance(ytr, pd.Series):
            ytr = ytr.values  # Extract NumPy array, drops index metadata
        if isinstance(ydev, pd.Series):
            ydev = ydev.values

        # Ensure 1D arrays (defensive check)
        if ytr.ndim != 1:
            raise ValueError(f"ytr must be 1D array, got shape {ytr.shape}")
        if ydev.ndim != 1:
            raise ValueError(f"ydev must be 1D array, got shape {ydev.shape}")

        def _sequential_es_indices(idx: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
            """Return (fit_idx, es_idx) where es_idx is the last frac of idx in time order."""
            n = len(idx)
            if n < 10 or frac <= 0.0:
                return idx, np.array([], dtype=int)
            cut = max(1, int(round(n * (1.0 - frac))))
            return idx[:cut], idx[cut:]

        def _scale_pos_weight(y: np.ndarray) -> float:
            """Compute class weight for imbalanced binary classification."""
            pos = float(np.sum(y))
            return ((len(y) - pos) / pos) if pos > 0 else 1.0

        if search_space_fn is None:
            if self.default_param_dist is None:
                raise ValueError("search_space_fn must be provided when default_param_dist is None.")

            def search_space_fn(trial: optuna.Trial) -> dict[str, Any]:
                params: dict[str, Any] = {}
                for name, dist in self.default_param_dist.items():
                    if isinstance(dist, (list, tuple, np.ndarray)) and not (isinstance(dist, tuple) and len(dist) == 2):
                        params[name] = trial.suggest_categorical(name, list(dist))
                        continue
                    if isinstance(dist, tuple) and len(dist) == 2:
                        low, high = dist
                        if isinstance(low, int) and isinstance(high, int):
                            params[name] = trial.suggest_int(name, low, high)
                        else:
                            params[name] = trial.suggest_float(name, float(low), float(high))
                        continue
                    if hasattr(dist, "dist") and hasattr(dist, "kwds"):
                        dist_name = dist.dist.name
                        if dist_name == "randint":
                            params[name] = trial.suggest_int(name, dist.kwds["low"], dist.kwds["high"] - 1)
                            continue
                        if dist_name == "uniform":
                            low = dist.kwds["loc"]
                            high = dist.kwds["loc"] + dist.kwds["scale"]
                            params[name] = trial.suggest_float(name, low, high)
                            continue
                    raise ValueError(f"Unsupported distribution type for parameter '{name}'.")
                return params

        start_time = time.time()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n{'=' * 60}\nOPTUNA HYPERPARAMETER OPTIMIZATION\n{'=' * 60}")
        print(f"Validation: {'External (Xdev)' if use_external_val else f'StratifiedKFold (CV={cv_folds})'}")
        print(f"Bagging: {'ENABLED' if enable_bagging else 'DISABLED'}")
        self.feature_names = list(Xtr.columns)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective: train model and return validation score."""
            params = search_space_fn(trial)
            initial_lr = params["learning_rate"]
            n_estimators = params["n_estimators"]
            early_stopping_rounds = params.pop("early_stopping_rounds")

            # Bagging hyperparameters
            if enable_bagging:
                n_estimators_bag = trial.suggest_int("n_estimators_bag", 5, 20)
                max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
                bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            else:
                n_estimators_bag = 1
                max_samples = 1.0
                bootstrap = False

            decay_rate = 0.5 ** (1.0 / n_estimators)
            lr_decay_fn = self._create_lr_decay_function(initial_lr, decay_rate)
            lr_scheduler = xgb.callback.LearningRateScheduler(lr_decay_fn)
            params_no_lr = {k: v for k, v in params.items() if k != "learning_rate"}

            if use_external_val:
                # Train on FULL Xtr, validate on Xdev
                spw = _scale_pos_weight(ytr)

                is_bagged = enable_bagging and n_estimators_bag > 1
                use_early_stopping = early_stopping_rounds if not is_bagged else None

                eval_set = [(Xdev, ydev)] if use_early_stopping else None

                base_estimator = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=spw,
                    tree_method="hist",
                    random_state=self.random_state,
                    verbosity=0,
                    early_stopping_rounds=use_early_stopping,
                    callbacks=[lr_scheduler],
                    learning_rate=initial_lr,
                    **params_no_lr,
                )

                if is_bagged:
                    model = BaggingClassifier(
                        estimator=base_estimator,
                        n_estimators=n_estimators_bag,
                        max_samples=max_samples,
                        bootstrap=bootstrap,
                        random_state=self.random_state,
                        n_jobs=bagging_n_jobs,
                        verbose=0,
                    )
                    model.fit(Xtr, ytr)  # No eval_set for bagging
                else:
                    model = base_estimator
                    model.fit(Xtr, ytr, eval_set=eval_set, verbose=False)

                # Evaluate on Xdev
                if scoring in {"roc_auc", "neg_log_loss"}:
                    y_pred_proba = model.predict_proba(Xdev)[:, 1]
                    score = self._score_predictions(scoring, ydev, y_pred_proba=y_pred_proba)
                else:
                    y_pred = model.predict(Xdev)
                    score = self._score_predictions(scoring, ydev, y_pred=y_pred)

                return float(score)

            else:
                cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

                cv_scores: list[float] = []

                # Stratified folds: preserve class distribution (Win/Loss ratio) in each fold
                for fold_idx, (tr_idx, te_idx) in enumerate(cv_strategy.split(Xtr, ytr)):
                    # Sequential ES split from the end of the training indices
                    fit_idx, es_idx = _sequential_es_indices(tr_idx, es_fraction)

                    X_fit, y_fit = Xtr.iloc[fit_idx], ytr[fit_idx]
                    X_te, y_te = Xtr.iloc[te_idx], ytr[te_idx]
                    spw = _scale_pos_weight(y_fit)

                    # CRITICAL FIX: Disable early stopping if bagging (BaggingClassifier limitation)
                    is_bagged = enable_bagging and n_estimators_bag > 1
                    use_early_stopping = early_stopping_rounds if not is_bagged else None

                    # Prepare eval_set ONLY if NOT bagging
                    eval_set = None
                    if use_early_stopping and len(es_idx) > 0:
                        X_es, y_es = Xtr.iloc[es_idx], ytr[es_idx]
                        eval_set = [(X_es, y_es)]

                    # Base XGBoost estimator
                    base_estimator = xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=spw,
                        tree_method="hist",
                        random_state=self.random_state,
                        verbosity=0,
                        early_stopping_rounds=use_early_stopping,
                        callbacks=[lr_scheduler],
                        learning_rate=initial_lr,  # Add directly here
                        **params_no_lr,
                    )
                    # Wrap in BaggingClassifier if enabled
                    if is_bagged:
                        model = BaggingClassifier(
                            estimator=base_estimator,
                            n_estimators=n_estimators_bag,
                            max_samples=max_samples,
                            bootstrap=bootstrap,
                            random_state=self.random_state,
                            n_jobs=bagging_n_jobs,
                            verbose=0,
                        )
                        # BaggingClassifier: fit without eval_set (no early stopping)
                        model.fit(X_fit, y_fit)
                    else:
                        # Single model: fit WITH eval_set (early stopping enabled)
                        model = base_estimator
                        model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)

                    # Evaluate on forward-in-time validation block
                    if scoring in {"roc_auc", "neg_log_loss"}:
                        y_pred_proba = model.predict_proba(X_te)[:, 1]
                        fold_score = self._score_predictions(scoring, y_te, y_pred_proba=y_pred_proba)
                    else:
                        y_pred = model.predict(X_te)
                        fold_score = self._score_predictions(scoring, y_te, y_pred=y_pred)

                    cv_scores.append(fold_score)
                    trial.report(fold_score, fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return float(np.mean(cv_scores))

        # Configure Optuna study
        sampler = TPESampler(n_startup_trials=10, seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, cv_folds // 2))
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name="xgboost_bagging_time_series_cv"
        )

        print(f"\nStarting optimization (n_trials={n_trials}, cv_folds={cv_folds}, scoring={scoring})...")
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout, show_progress_bar=True)

        # Get full params (including fixed values) by calling search_space_fn with best trial
        full_params = search_space_fn(study.best_trial)
        # Merge: Optuna's suggested params override any conflicts
        best_params = {**full_params, **study.best_params}
        best_score = study.best_value

        print(f"\n{'=' * 50}\nOPTIMIZATION RESULTS\n{'=' * 50}")
        print(f"Best {scoring.upper()}: {best_score:.4f}")
        print("Best Parameters:")
        for param, value in sorted(best_params.items()):
            print(f"  {param}: {value}")
        print(f"Trials completed: {len(study.trials)}")
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"Trials pruned: {pruned_trials}")

        # Final refit on full Xtr with sequential ES from the tail of Xtr
        best_params_copy = best_params.copy()
        early_stopping_rounds = best_params_copy.pop("early_stopping_rounds", None)
        initial_lr = best_params_copy.get("learning_rate", 0.1)

        # Extract bagging params if they exist
        n_estimators_bag = best_params_copy.pop("n_estimators_bag", 1)
        max_samples = best_params_copy.pop("max_samples", 1.0)
        bootstrap = best_params_copy.pop("bootstrap", False)

        n_estimators = best_params_copy.get("n_estimators", 100)
        decay_rate = 0.5 ** (1.0 / n_estimators)
        lr_decay_fn = self._create_lr_decay_function(initial_lr, decay_rate)
        lr_scheduler = xgb.callback.LearningRateScheduler(lr_decay_fn)

        # Sequential ES split on the full training set (tail used as eval_set)
        full_idx = np.arange(len(Xtr))
        fit_idx, es_idx = _sequential_es_indices(full_idx, es_fraction)
        X_fit, y_fit = Xtr.iloc[fit_idx], ytr[fit_idx]

        # CRITICAL FIX: Disable early stopping if bagging
        is_bagged = enable_bagging and n_estimators_bag > 1
        use_early_stopping = early_stopping_rounds if not is_bagged else None

        eval_set = None
        if use_early_stopping and len(es_idx) > 0:
            X_es, y_es = Xtr.iloc[es_idx], ytr[es_idx]
            eval_set = [(X_es, y_es)]

        spw_full = _scale_pos_weight(y_fit)

        # Remove learning_rate from params
        params_no_lr = {k: v for k, v in best_params_copy.items() if k != "learning_rate"}

        # Base estimator
        base_estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw_full,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
            early_stopping_rounds=use_early_stopping,  # ✅ None if bagged
            callbacks=[lr_scheduler],
            learning_rate=initial_lr,
            **params_no_lr,
        )

        # Wrap in BaggingClassifier if bagging was enabled and found optimal
        if is_bagged:
            best_estimator = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators_bag,
                max_samples=max_samples,
                bootstrap=bootstrap,
                random_state=self.random_state,
                n_jobs=bagging_n_jobs,
                verbose=0,
            )
            # BaggingClassifier: fit without eval_set
            best_estimator.fit(X_fit, y_fit)
        else:
            best_estimator = base_estimator
            best_estimator.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)

        # Dev monitoring (unchanged: never used for training)
        y_pred_dev = best_estimator.predict(Xdev)
        dev_f1 = precision_recall_fscore_support(ydev, y_pred_dev, average="binary", zero_division=0)[2]
        dev_auc = roc_auc_score(ydev, best_estimator.predict_proba(Xdev)[:, 1]) if len(np.unique(ydev)) > 1 else 0.0
        print(f"Dev F1: {dev_f1:.4f} | Dev AUC: {dev_auc:.4f}")

        # Store results (best_params already includes early_stopping_rounds, learning_rate from search_space_fn)
        self.optimization_results = {
            "best_score": float(best_score),
            "best_params": {
                **best_params,
                "n_estimators_bag": n_estimators_bag,
                "max_samples": max_samples,
                "bootstrap": bootstrap,
            },
            "best_estimator": best_estimator,
            "optuna_study": study,
        }
        self.model = best_estimator

        print(f"Optimization time: {time.time() - start_time:.2f}s\n")
        return self.optimization_results

    def _compute_permutation_importance(
        self, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10, scoring: str = "f1", dataset_name: str = "Dataset"
    ) -> pd.DataFrame:

        if self.model is None:
            raise ValueError("Train model first (call train_optimized_model)")

        print(f"\nComputing permutation importance on {dataset_name} set (n_repeats={n_repeats})...")

        # sklearn's permutation_importance handles scoring internally
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            perm_result = permutation_importance(
                estimator=self.model,
                X=X,
                y=y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=-1,  # Parallel processing
            )

        # Build DataFrame with results
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm_result.importances_mean,
                "importance_std": perm_result.importances_std,
            }
        )

        # Normalize to [0, 100] scale for interpretability
        max_importance = importance_df["importance_mean"].max()
        if max_importance > 0:
            importance_df["importance_normalized"] = (importance_df["importance_mean"] / max_importance * 100).round(2)
        else:
            importance_df["importance_normalized"] = 0.0

        # Rank features (1 = most important)
        importance_df["rank"] = importance_df["importance_mean"].rank(ascending=False, method="min").astype(int)

        # Sort by importance
        importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

        print(f"✓ Permutation importance computed for {len(importance_df)} features")

        return importance_df[["feature", "importance_mean", "importance_std", "importance_normalized", "rank"]]

    def _compare_permutation_stability(
        self, train_importance: pd.DataFrame, dev_importance: pd.DataFrame, drop_threshold: float = 0.3
    ) -> dict[str, Any]:
        """
        Compare permutation importance between train and dev sets.

        Mathematical formulation:
        $$
        \text{Rank Correlation} = \rho_{\text{Spearman}}(r_{\text{train}}, r_{\text{dev}})
        $$
        $$
        \text{Importance Drop}_i = \frac{I_{\text{train},i} - I_{\text{dev},i}}{I_{\text{train},i}} \times 100
        $$

        Interpretation:
        - Rank correlation > 0.8: High stability (features generalize well)
        - Rank correlation 0.6-0.8: Medium stability (some instability)
        - Rank correlation < 0.6: Low stability (features overfit or regime-specific)

        Args:
            train_importance: Train set permutation importance DataFrame
            dev_importance: Dev set permutation importance DataFrame
            drop_threshold: Threshold for flagging unstable features (default 0.3 = 30% drop)

        Returns:
            Dict with keys:
            - rank_correlation: Spearman correlation between train/dev ranks
            - rank_pvalue: Statistical significance of correlation
            - unstable_features: List of features with >threshold importance drop
            - importance_drops: DataFrame with per-feature drop percentages

        Notes:
            - High rank_correlation (>0.8) = stable feature rankings
            - Large importance drops = potential overfitting to train set
            - Negative drops = feature more important on dev (rare, investigate for data leakage)

        """
        # Merge train and dev importance on feature names
        merged = train_importance[["feature", "importance_mean", "rank"]].merge(
            dev_importance[["feature", "importance_mean", "rank"]], on="feature", suffixes=("_train", "_dev")
        )

        # Compute Spearman rank correlation
        rank_corr, rank_pval = spearmanr(merged["rank_train"], merged["rank_dev"])

        # Compute importance drop percentage
        # Formula: (train - dev) / train * 100
        # Positive = feature weaker on dev, Negative = feature stronger on dev
        merged["importance_drop_pct"] = (
            (merged["importance_mean_train"] - merged["importance_mean_dev"])
            / (merged["importance_mean_train"] + 1e-8)  # Add epsilon to avoid division by zero
        ) * 100

        # Flag unstable features (drop > threshold)
        unstable_mask = merged["importance_drop_pct"] > (drop_threshold * 100)
        unstable_features = merged.loc[unstable_mask, "feature"].tolist()

        # Sort by drop percentage (descending = most unstable first)
        importance_drops = merged.sort_values("importance_drop_pct", ascending=False)

        return {
            "rank_correlation": rank_corr,
            "rank_pvalue": rank_pval,
            "unstable_features": unstable_features,
            "importance_drops": importance_drops,
        }

    def train_optimized_model(
        self,
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xdev: pd.DataFrame,
        ydev: np.ndarray,
        compute_permutation: bool = True,
        permutation_repeats: int = 10,
        permutation_scoring: str = "f1",
        drop_threshold: float = 0.3,
    ) -> xgb.XGBClassifier | BaggingClassifier:
        """
        Train final model using best Optuna params with learning rate decay.

        **CRITICAL**: Model trained ONLY on Xtr (Xdev used for eval_set monitoring, NOT training)

        Training Strategy:
        1. Use best hyperparameters from optimize_hyperparameters_optuna
        2. Train on Xtr with LR decay and early stopping
        3. If bagging enabled, wrap in BaggingClassifier
        4. Monitor dev loss (for early stopping only, NOT for gradient updates)
        5. Optionally compute permutation importance for OOS validation

        Args:
            (unchanged from original)

        Returns:
            Trained XGBoost classifier or BaggingClassifier (if bagging enabled)

        Side Effects:
            - Sets self.model (trained classifier, possibly bagged)
            - Sets self.train_loss, self.val_loss (learning curves, only if NOT bagged)
            - Sets self.permutation_results (if compute_permutation=True)

        Notes:
            - Model trained ONLY on Xtr (Xdev used for eval_set monitoring, not training)
            - Permutation importance computed post-training on true OOS data (Xdev)
            - Unstable features (>30% importance drop) automatically flagged
            - Early stopping uses dev loss to prevent overfitting
            - **Bagging limitation**: Learning curves not available (BaggingClassifier doesn't expose evals_result)

        Assumptions:
            - optimize_hyperparameters_optuna must be called first
            - Xtr and Xdev are temporally separated (no data leakage)
            - ytr and ydev are binary (0/1)

        """
        if self.optimization_results is None:
            raise ValueError("Run optimize_hyperparameters_optuna first")

        print(f"\n{'=' * 60}\nTRAINING OPTIMIZED MODEL (TRAIN SET ONLY)\n{'=' * 60}")
        print(f"Train samples: {len(Xtr)} | Dev samples: {len(Xdev)} (OOS validation)")

        self.feature_names = list(Xtr.columns)
        ytr_values = ytr.values if isinstance(ytr, pd.Series) else ytr
        scale_pos_weight = (
            (len(ytr_values) - np.sum(ytr_values)) / np.sum(ytr_values) if np.sum(ytr_values) > 0 else 1.0
        )

        # Eval set for monitoring (NOT for training)
        eval_set = [(Xtr, ytr), (Xdev, ydev)]

        best_params = self.optimization_results["best_params"].copy()
        early_stopping_rounds = best_params.pop("early_stopping_rounds", None)
        initial_lr = best_params.get("learning_rate", 0.1)

        n_estimators = best_params.get("n_estimators", 100)
        decay_rate = 0.5 ** (1.0 / n_estimators)

        # Extract bagging params
        n_estimators_bag = best_params.pop("n_estimators_bag", 1)
        max_samples = best_params.pop("max_samples", 1.0)
        bootstrap = best_params.pop("bootstrap", False)

        is_bagged = n_estimators_bag > 1

        # CRITICAL FIX: Disable early stopping if bagging
        use_early_stopping = early_stopping_rounds if not is_bagged else None

        # Eval set for monitoring (NOT for training) - only if NOT bagged
        eval_set = None
        if use_early_stopping:
            eval_set = [(Xtr, ytr), (Xdev, ydev)]

        lr_decay_fn = self._create_lr_decay_function(initial_lr, decay_rate)
        lr_scheduler = xgb.callback.LearningRateScheduler(lr_decay_fn)

        print(f"Initial LR: {initial_lr:.4f}, Decay rate: {decay_rate}, Min LR: 1e-4")
        if is_bagged:
            print(f"Bagging: {n_estimators_bag} estimators, max_samples={max_samples:.2f}, bootstrap={bootstrap}")
            print("⚠ Early stopping DISABLED (BaggingClassifier limitation)")

        # Remove learning_rate from params
        params_no_lr = {k: v for k, v in best_params.items() if k != "learning_rate"}

        # Base XGBoost estimator
        base_estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=1 if not is_bagged else 0,
            early_stopping_rounds=use_early_stopping,  # ✅ None if bagged
            callbacks=[lr_scheduler],
            learning_rate=initial_lr,
            **params_no_lr,
        )

        # Train on Xtr ONLY
        if is_bagged:
            # Wrap in BaggingClassifier
            self.model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators_bag,
                max_samples=max_samples,
                bootstrap=bootstrap,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            self.model.fit(Xtr, ytr)  # No eval_set

            # Learning curves not available for bagged models
            self.train_loss = None
            self.val_loss = None
            print(f"\n✓ Bagged training complete ({n_estimators_bag} estimators)")
            print("⚠ Learning curves not available (BaggingClassifier limitation)")
            print("⚠ Early stopping not used (BaggingClassifier limitation)")
        else:
            # Single XGBoost model (original behavior)
            self.model = base_estimator
            self.model.fit(Xtr, ytr, eval_set=eval_set, verbose=0)

            evals_result = self.model.evals_result()
            self.train_loss = np.array(evals_result["validation_0"]["logloss"])
            self.val_loss = np.array(evals_result["validation_1"]["logloss"])

            print("\n✓ Training complete")
            print(f"Final train loss: {self.train_loss[-1]:.4f}")
            print(f"Final dev loss: {self.val_loss[-1]:.4f}")
            print(f"Final LR: {lr_decay_fn(len(self.train_loss) - 1):.6f}")

        # Compute permutation importance if requested
        if compute_permutation:
            # (unchanged from original - permutation works on ensemble predictions)
            train_perm = self._compute_permutation_importance(
                X=Xtr, y=ytr, n_repeats=permutation_repeats, scoring=permutation_scoring, dataset_name="Train"
            )
            dev_perm = self._compute_permutation_importance(
                X=Xdev, y=ydev, n_repeats=permutation_repeats, scoring=permutation_scoring, dataset_name="Dev"
            )
            self.permutation_results = {"train": train_perm, "dev": dev_perm}

            stability = self._compare_permutation_stability(
                train_importance=train_perm, dev_importance=dev_perm, drop_threshold=drop_threshold
            )
            self.permutation_results["stability"] = stability

            # (Print stability summary - unchanged from original)
            print(f"\n{'=' * 50}\nSTABILITY ANALYSIS\n{'=' * 50}")
            print(
                f"Spearman rank correlation (train vs dev): {stability['rank_correlation']:.4f} "
                f"(p={stability['rank_pvalue']:.4f})"
            )

            if stability["rank_correlation"] > 0.8:
                print("✓ HIGH stability: Feature rankings consistent across train/dev")
            elif stability["rank_correlation"] > 0.6:
                print("⚠ MEDIUM stability: Some rank changes between train/dev")
            else:
                print("✗ LOW stability: Significant rank instability detected")

            if stability["unstable_features"]:
                print(
                    f"\n{len(stability['unstable_features'])} features with >{drop_threshold * 100:.0f}% importance drop on dev:"  # noqa: E501
                )
                for feat in stability["unstable_features"][:10]:
                    drop_pct = (
                        stability["importance_drops"]
                        .loc[stability["importance_drops"]["feature"] == feat, "importance_drop_pct"]
                        .values[0]
                    )
                    print(f"  - {feat}: {drop_pct:.1f}% drop")
                if len(stability["unstable_features"]) > 10:
                    print(f"  ... and {len(stability['unstable_features']) - 10} more")
                print("\n→ Consider removing these features (potential overfitting to train set)")
            else:
                print(f"\n✓ No features with >{drop_threshold * 100:.0f}% importance drop")

            stable_features = stability["importance_drops"].nsmallest(10, "importance_drop_pct")
            print("\nTop 10 most stable features (smallest importance drop):")
            for _idx, row in stable_features.iterrows():
                print(
                    f"  {row['rank_train']:2d}. {row['feature']:30s} | "
                    f"Train: {row['importance_mean_train']:.4f} | "
                    f"Dev: {row['importance_mean_dev']:.4f} | "
                    f"Drop: {row['importance_drop_pct']:+.1f}%"
                )

        return self.model

    def get_feature_importance_all(
        self,
        top_n: int | None = None,
        normalize: bool = True,
        include_zero: bool = False,  # ✅ Changed default to False
    ) -> pd.DataFrame:
        """
        Extract all feature importance types with proper feature name mapping.

        Args:
            top_n: Return only top N features (None = all)
            normalize: Scale to [0, 100]
            include_zero: Include features with zero importance (default=False)

        Returns:
            DataFrame with feature importances and original feature names

        """
        if self.model is None:
            raise ValueError("Train model first")

        is_bagged = isinstance(self.model, BaggingClassifier)

        if is_bagged:
            # Collect importances from all bags
            gain_dicts, weight_dicts, cover_dicts = [], [], []
            for estimator in self.model.estimators_:
                booster = estimator.get_booster()
                gain_dicts.append(booster.get_score(importance_type="gain"))
                weight_dicts.append(booster.get_score(importance_type="weight"))
                cover_dicts.append(booster.get_score(importance_type="cover"))

            all_features = set()
            for d in gain_dicts + weight_dicts + cover_dicts:
                all_features.update(d.keys())

            importance_data = []
            for feature in all_features:
                gain_vals = [d.get(feature, 0.0) for d in gain_dicts]
                weight_vals = [d.get(feature, 0.0) for d in weight_dicts]
                cover_vals = [d.get(feature, 0.0) for d in cover_dicts]

                importance_data.append(
                    {
                        "feature": feature,
                        "gain": np.mean(gain_vals),
                        "weight": np.mean(weight_vals),
                        "cover": np.mean(cover_vals),
                        "gain_std": np.std(gain_vals),
                        "weight_std": np.std(weight_vals),
                        "cover_std": np.std(cover_vals),
                    }
                )

            importance_df = pd.DataFrame(importance_data)
        else:
            # Single model
            booster = self.model.get_booster()
            gain_dict = booster.get_score(importance_type="gain")
            weight_dict = booster.get_score(importance_type="weight")
            cover_dict = booster.get_score(importance_type="cover")

            all_features = set(gain_dict.keys()) | set(weight_dict.keys()) | set(cover_dict.keys())

            importance_data = []
            for feature in all_features:
                importance_data.append(
                    {
                        "feature": feature,
                        "gain": gain_dict.get(feature, 0),
                        "weight": weight_dict.get(feature, 0),
                        "cover": cover_dict.get(feature, 0),
                    }
                )

            importance_df = pd.DataFrame(importance_data)
            importance_df["gain_std"] = 0.0
            importance_df["weight_std"] = 0.0
            importance_df["cover_std"] = 0.0

        # ✅ ROBUST FEATURE NAME MAPPING
        if self.feature_names is not None:
            # Convert to list (handles pandas.Index)
            feature_names_list = list(self.feature_names)

            # Get XGBoost's internal feature format
            xgb_features = importance_df["feature"].tolist()

            # Check if XGBoost uses 'f0', 'f1', 'f2' format
            if all(
                isinstance(f, str) and f.startswith("f") and f[1:].isdigit()
                for f in xgb_features[: min(5, len(xgb_features))]
            ):
                # XGBoost using fN format - create mapping
                feature_map = {f"f{i}": name for i, name in enumerate(feature_names_list)}
                importance_df["feature"] = importance_df["feature"].map(feature_map)

                # Check for unmapped features
                unmapped = importance_df["feature"].isna().sum()
                if unmapped > 0:
                    print(f"⚠ WARNING: {unmapped} features could not be mapped")
                    print("  This usually means include_zero=True added phantom features")
                    print("  Try: get_feature_importance_all(include_zero=False)")
                    importance_df["feature"] = importance_df["feature"].fillna("UNKNOWN")
            else:
                # XGBoost already using real feature names - no mapping needed
                pass

        # Add zero-importance features if requested (AFTER mapping)
        if include_zero and self.feature_names is not None:
            feature_names_list = list(self.feature_names)
            existing_features = set(importance_df["feature"].tolist())
            missing_features = set(feature_names_list) - existing_features

            if missing_features:
                zero_data = []
                for feat in missing_features:
                    zero_data.append(
                        {
                            "feature": feat,
                            "gain": 0,
                            "weight": 0,
                            "cover": 0,
                            "gain_std": 0.0,
                            "weight_std": 0.0,
                            "cover_std": 0.0,
                        }
                    )
                zero_df = pd.DataFrame(zero_data)
                importance_df = pd.concat([importance_df, zero_df], ignore_index=True)

        if normalize:
            for col in ["gain", "weight", "cover"]:
                max_val = importance_df[col].max()
                if max_val > 0:
                    importance_df[col] = (importance_df[col] / max_val * 100).round(2)

        importance_df["total_importance"] = (
            importance_df["gain"] + importance_df["weight"] + importance_df["cover"]
        ) / 3
        importance_df["rank"] = importance_df["total_importance"].rank(ascending=False, method="min").astype(int)
        importance_df = importance_df.sort_values("total_importance", ascending=False).reset_index(drop=True)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def get_permutation_importance(self, dataset: str = "dev") -> pd.DataFrame:
        return self.permutation_results[dataset]

    def get_permutation_stability(self) -> dict[str, Any]:
        return self.permutation_results["stability"]

    def get_optuna_param_importance(self) -> pd.DataFrame:
        study = self.optimization_results["optuna_study"]

        try:
            importance_dict = optuna.importance.get_param_importances(study)
        except Exception as e:
            raise ValueError(f"Could not compute parameter importance: {e!s}") from e

        if not importance_dict:
            raise ValueError("No parameter importance available (insufficient trials)")

        importance_df = pd.DataFrame({"parameter": importance_dict.keys(), "importance": importance_dict.values()})

        max_imp = importance_df["importance"].max()
        importance_df["importance_normalized"] = (importance_df["importance"] / max_imp * 100).round(2)

        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df[["rank", "parameter", "importance", "importance_normalized"]]

    def _print_comprehensive_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, dataset_name: str
    ) -> dict[str, Any]:
        """
        Compute and print comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            dataset_name: Name for logging (e.g., 'Train', 'Dev', 'Test')

        Returns:
            Dict with metrics: accuracy, auc, precision, recall, f1, confusion_matrix

        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        precision_bin = precision[1] if len(precision) > 1 else 0.0
        recall_bin = recall[1] if len(recall) > 1 else 0.0
        f1_bin = f1[1] if len(f1) > 1 else 0.0
        f1_macro = np.mean(f1)
        auc = roc_auc_score(y_true, y_pred_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.0
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n{dataset_name.upper()} METRICS\n{'=' * 50}")
        print(f"Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")
        print(f"Binary Precision: {precision_bin:.4f} | Recall: {recall_bin:.4f} | F1: {f1_bin:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y_true, y_pred, target_names=["Loss", "Win"]))

        return {
            "accuracy": accuracy,
            "auc": auc,
            "precision_binary": precision_bin,
            "recall_binary": recall_bin,
            "f1_binary": f1_bin,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
        }

    def dev(
        self, Xtr: pd.DataFrame, ytr: np.ndarray, Xdev: pd.DataFrame, ydev: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate on train and dev sets.

        Args:
            Xtr: Training features
            ytr: Training labels
            Xdev: Dev features
            ydev: Dev labels

        Returns:
            Dict with keys 'train' and 'dev', each containing metrics dict

        """
        if self.model is None:
            raise ValueError("Train model first")

        train_metrics = self._print_comprehensive_metrics(
            ytr, self.model.predict(Xtr), self.model.predict_proba(Xtr), "Train"
        )
        dev_metrics = self._print_comprehensive_metrics(
            ydev, self.model.predict(Xdev), self.model.predict_proba(Xdev), "Dev"
        )

        return {"train": train_metrics, "dev": dev_metrics}

    def test(
        self, Xdev: pd.DataFrame, ydev: np.ndarray, Xte: pd.DataFrame, yte: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate on dev and test sets.

        Args:
            Xdev: Dev features
            ydev: Dev labels
            Xte: Test features
            yte: Test labels

        Returns:
            Dict with keys 'dev' and 'test', each containing metrics dict

        """
        if self.model is None:
            raise ValueError("Train model first")

        dev_metrics = self._print_comprehensive_metrics(
            ydev, self.model.predict(Xdev), self.model.predict_proba(Xdev), "Dev"
        )
        test_metrics = self._print_comprehensive_metrics(
            yte, self.model.predict(Xte), self.model.predict_proba(Xte), "Test"
        )

        return {"dev": dev_metrics, "test": test_metrics}

    @property
    def best_model(self) -> xgb.XGBClassifier:
        """Alias for backward compatibility."""
        return self.model

    def train_production_model(
        self, X_full: pd.DataFrame, y_full: np.ndarray, save_path: str | None = None
    ) -> xgb.XGBClassifier | BaggingClassifier:
        if self.optimization_results is None:
            raise ValueError("Must call optimize_hyperparameters_optuna() first")

        if self.model is None:
            raise ValueError("Must call train_optimized_model() first to get best_iteration")

        # Store feature names
        self.feature_names = list(X_full.columns)

        # Convert labels to numpy
        y_full_values = y_full.values if isinstance(y_full, pd.Series) else y_full
        if y_full_values.ndim != 1:
            raise ValueError(f"y_full must be 1D array, got shape {y_full_values.shape}")

        # Compute class weight on full dataset
        n_positive = np.sum(y_full_values == 1)
        n_negative = len(y_full_values) - n_positive
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        # Extract hyperparameters
        best_params = self.optimization_results["best_params"].copy()
        best_params.pop("early_stopping_rounds")
        n_estimators_tuned = best_params.get("n_estimators")

        # Extract bagging parameters
        n_estimators_bag = best_params.pop("n_estimators_bag", 1)
        max_samples = best_params.pop("max_samples", 1.0)
        bootstrap = best_params.pop("bootstrap", False)

        is_bagged = n_estimators_bag > 1

        # Determine optimal n_estimators from early stopping
        if is_bagged:
            n_estimators_prod = n_estimators_tuned
            print(f"\n⚠ Bagged model: using n_estimators={n_estimators_prod}")
        else:
            if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
                n_estimators_prod = self.model.best_iteration
                trees_saved = n_estimators_tuned - n_estimators_prod
                print(f"\n✓ Using best_iteration={n_estimators_prod} from early stopping")
                print(f"  Early stopping saved {trees_saved} overfitting trees")
            else:
                n_estimators_prod = n_estimators_tuned
                print(f"\n⚠ No early stopping detected, using n_estimators={n_estimators_prod}")

        best_params["n_estimators"] = n_estimators_prod

        if is_bagged:
            print(f"  Bagging: {n_estimators_bag} estimators, max_samples={max_samples:.2f}, bootstrap={bootstrap}")

        # Build base XGBoost estimator (NO CALLBACKS for JSON compatibility)
        base_estimator = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
            early_stopping_rounds=None,
            callbacks=None,  # ✓ No callbacks = JSON compatible
        )
        base_estimator._estimator_type = "classifier"

        # Train model
        if is_bagged:
            production_model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators_bag,
                max_samples=max_samples,
                bootstrap=bootstrap,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
        else:
            production_model = base_estimator
            print(f"\nTraining single XGBoost model ({n_estimators_prod} trees)...")

        production_model.fit(X_full, y_full_values)

        # Store model and metadata
        self.model_production = production_model
        self.production_stats = {
            "n_samples": len(y_full_values),
            "n_features": X_full.shape[1],
            "n_positives": int(n_positive),
            "n_negatives": int(n_negative),
            "positive_rate": float(n_positive / len(y_full_values)),
            "scale_pos_weight": float(scale_pos_weight),
            "n_estimators": n_estimators_prod,
            "n_estimators_tuned": n_estimators_tuned,
            "trees_saved_by_early_stopping": n_estimators_tuned - n_estimators_prod,
            "is_bagged": is_bagged,
            "n_estimators_bag": n_estimators_bag if is_bagged else None,
            "max_samples": float(max_samples) if is_bagged else None,
            "bootstrap": bootstrap if is_bagged else None,
            "feature_names": self.feature_names,
            "train_date": pd.Timestamp.now().isoformat(),
        }

        # Save model in JSON format
        if save_path:
            self._save_production_model_json(production_model, save_path, is_bagged)

        return production_model

    def _save_production_model_json(
        self, model: xgb.XGBClassifier | BaggingClassifier, save_path: str, is_bagged: bool
    ) -> None:
        """
        Save production model in JSON format.

        Single: model.json + metadata.json
        Bagged: model_dir/ with estimator_*.json + bagging_config.json + metadata.json
        """
        import json
        from pathlib import Path

        save_path = Path(save_path)

        # Ensure .json extension for base path
        if save_path.suffix != ".json":
            save_path = save_path.with_suffix(".json")

        if is_bagged:
            # Create directory for bagged model
            model_dir = save_path.parent / save_path.stem
            model_dir.mkdir(exist_ok=True, parents=True)

            print(f"\nSaving bagged model to {model_dir}/")

            # Save each base estimator as separate JSON file
            for i, estimator in enumerate(model.estimators_):
                estimator_path = model_dir / f"estimator_{i:03d}.json"
                estimator.save_model(estimator_path)

            print(f"✓ Saved {len(model.estimators_)} base estimators")

            # Save bagging configuration as JSON
            bagging_config = {
                "model_type": "BaggingClassifier",
                "n_estimators": len(model.estimators_),
                "max_samples": float(model.max_samples),
                "bootstrap": bool(model.bootstrap),
                "random_state": int(model.random_state) if model.random_state is not None else None,
                "estimator_files": [f"estimator_{i:03d}.json" for i in range(len(model.estimators_))],
            }

            config_path = model_dir / "bagging_config.json"
            with open(config_path, "w") as f:
                json.dump(bagging_config, f, indent=2)

            print(f"✓ Bagging config: {config_path}")

            # Save metadata at parent level (same location as single model)
            metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"

        else:
            # Single XGBoost: Direct JSON save
            model.save_model(save_path)
            print(f"\n✓ Model saved: {save_path} (XGBoost native JSON)")

            metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"

        # Save metadata as JSON (not pickle)
        # Convert numpy types to native Python for JSON serialization
        metadata_json = {}
        for k, v in self.production_stats.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                metadata_json[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                metadata_json[k] = float(v)
            elif isinstance(v, (list, tuple)):
                metadata_json[k] = list(v)
            else:
                metadata_json[k] = v

        with open(metadata_path, "w") as f:
            json.dump(metadata_json, f, indent=2)

        print(f"✓ Metadata saved: {metadata_path}")
