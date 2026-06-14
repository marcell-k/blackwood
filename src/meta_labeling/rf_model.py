import time
from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_STATE
from src.data.splitters import CPCVSplitter
from src.visualization.style import DEFAULT_STYLE


def _apply_axis_style(ax: plt.Axes, style=DEFAULT_STYLE) -> None:
    """Apply consistent style to a matplotlib axis."""
    ax.set_facecolor(style.plot_bgcolor)
    ax.grid(True, color=style.grid, alpha=0.35, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(style.line)
        spine.set_linewidth(0.9)
    ax.tick_params(colors=style.font_color)
    ax.xaxis.label.set_color(style.font_color)
    ax.yaxis.label.set_color(style.font_color)
    ax.title.set_color(style.font_color)


def _style_legend(ax: plt.Axes, style=DEFAULT_STYLE, **kwargs) -> None:
    """Create legend using project default style."""
    legend = ax.legend(facecolor=style.plot_bgcolor, edgecolor=style.line, **kwargs)
    for text in legend.get_texts():
        text.set_color(style.font_color)


_SCORE_DISPATCH = {
    "f1": lambda y_true, y_pred, _: float(
        precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[2]
    ),
    "roc_auc": lambda y_true, _, y_proba: float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    "neg_log_loss": lambda y_true, _, y_proba: float(-log_loss(y_true, y_proba, labels=[0, 1])),
    "accuracy": lambda y_true, y_pred, _: float(accuracy_score(y_true, y_pred)),
}


class BinaryModelBagging:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.model: BaggingClassifier | None = None
        self.best_params: dict | None = None
        self.optimization_results: dict | None = None
        self.feature_names: list[str] | None = None

    @staticmethod
    def conservative_search_space(trial: optuna.Trial) -> dict:
        """Optuna space tuned for stronger regularization and lower overfitting risk."""
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 140),
            "min_samples_split": trial.suggest_int("min_samples_split", 50, 260),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.3, 0.5]),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 96),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 5e-2, log=True),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-8, 1e-3, log=True),
            "n_estimators_bag": trial.suggest_int("n_estimators_bag", 500, 2500, log=True),
            "max_samples_bag": trial.suggest_float("max_samples_bag", 0.45, 0.8),
            "max_features_bag": trial.suggest_float("max_features_bag", 0.45, 1.0),
            "bootstrap_features": trial.suggest_categorical("bootstrap_features", [False, True]),
            "class_weight": "balanced",  # Always balanced for meta-labeling
        }

    def _build_model(self, params: dict) -> BaggingClassifier:
        tree_params = {
            "max_depth": params.get("max_depth"),
            "min_samples_leaf": params.get("min_samples_leaf", 1),
            "min_samples_split": params.get("min_samples_split", 2),
            "max_features": params.get("max_features", "sqrt"),
            "random_state": self.random_state,
            "class_weight": params.get("class_weight"),
            "min_impurity_decrease": params.get("min_impurity_decrease", 0.0),
            "ccp_alpha": params.get("ccp_alpha", 0.0),
        }
        if "max_leaf_nodes" in params and params["max_leaf_nodes"] is not None:
            tree_params["max_leaf_nodes"] = int(params["max_leaf_nodes"])

        return BaggingClassifier(
            estimator=DecisionTreeClassifier(**tree_params),
            n_estimators=params.get("n_estimators_bag", 1000),
            max_samples=params.get("max_samples_bag", 1.0),
            max_features=1.0,  # avoid double feature subsampling
            bootstrap=params.get("bootstrap", True),
            bootstrap_features=False,
            oob_score=True if params.get("bootstrap", True) else False,
            random_state=self.random_state,
            n_jobs=6,
        )

    def _get_feature_cols(self, X: pd.DataFrame) -> list[str]:
        return list(X.columns)

    def _evaluate_trial(
        self,
        trial: optuna.Trial,
        suggest_fn: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        splitter: CPCVSplitter,
        paths: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]],
        path_ids: list[int],
        scoring: str,
        score_std_penalty: float,
        overfit_penalty: float,
    ) -> float:
        """Evaluate a single trial across CPCV paths."""
        params = suggest_fn(trial)
        model = self._build_model(params)
        path_scores_test: list[float] = []
        path_scores_train: list[float] = []
        score_fn = _SCORE_DISPATCH.get(scoring, _SCORE_DISPATCH["accuracy"])

        for path_idx, path_id in enumerate(path_ids):
            train_df, test_df = splitter.get_train_test_for_path(X, paths, path_id)
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = y.loc[train_df.index]
            y_test = y.loc[test_df.index]

            model.fit(X_train, y_train)

            if scoring in {"roc_auc", "neg_log_loss"}:
                y_proba_train = model.predict_proba(X_train)[:, 1]
                y_proba = model.predict_proba(X_test)[:, 1]
                score_test = score_fn(y_test.values, None, y_proba)
                score_train = score_fn(y_train.values, None, y_proba_train)
            else:
                y_pred_train = model.predict(X_train)
                y_pred = model.predict(X_test)
                score_test = score_fn(y_test.values, y_pred, None)
                score_train = score_fn(y_train.values, y_pred_train, None)

            path_scores_test.append(score_test)
            path_scores_train.append(score_train)

            trial.report(score_test, path_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_test = float(np.mean(path_scores_test))
        std_test = float(np.std(path_scores_test))
        mean_train = float(np.mean(path_scores_train))
        mean_overfit_gap = float(np.maximum(np.asarray(path_scores_train) - np.asarray(path_scores_test), 0.0).mean())
        objective_score = mean_test - score_std_penalty * std_test - overfit_penalty * mean_overfit_gap

        trial.set_user_attr("mean_test_score", mean_test)
        trial.set_user_attr("std_test_score", std_test)
        trial.set_user_attr("mean_train_score", mean_train)
        trial.set_user_attr("mean_overfit_gap", mean_overfit_gap)
        return objective_score

    def optimize_hyperparameters_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splitter: CPCVSplitter,
        paths: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]],
        n_trials: int = 30,
        scoring: str = "f1",  # Changed from neg_log_loss to optimize for balanced precision/recall
        search_space_fn: Callable | None = None,
        timeout: int | None = None,
        score_std_penalty: float = 0.5,
        overfit_penalty: float = 0.0,
    ) -> dict:
        feature_cols = self._get_feature_cols(X)
        self.feature_names = feature_cols

        def _default_search_space(trial: optuna.Trial) -> dict:
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 50),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
                "n_estimators_bag": trial.suggest_int("n_estimators_bag", 20, 100),
                "max_samples_bag": trial.suggest_float("max_samples_bag", 0.5, 1.0),
                "class_weight": "balanced",  # Always balanced
            }

        suggest_fn = search_space_fn or _default_search_space
        path_ids = sorted(paths.keys())
        n_paths = len(path_ids)

        start_time = time.time()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n{'=' * 56}\nOPTUNA HYPERPARAMETER OPTIMIZATION (BaggingClassifier)\n{'=' * 56}")
        print(f"Features: {len(feature_cols)} | CPCV paths: {n_paths} | Scoring: {scoring}")
        print(f"Objective penalties: std={score_std_penalty:.3f}, overfit_gap={overfit_penalty:.3f}")

        def _objective_fn(trial: optuna.Trial) -> float:
            return self._evaluate_trial(
                trial,
                suggest_fn,
                X,
                y,
                feature_cols,
                splitter,
                paths,
                path_ids,
                scoring,
                score_std_penalty,
                overfit_penalty,
            )

        sampler = TPESampler(n_startup_trials=10, seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, n_paths // 2))
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="bagging_cpcv",
        )

        print(f"Starting optimization (n_trials={n_trials})...")
        study.optimize(
            _objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_trial = study.best_trial
        best_objective = float(best_trial.value)
        best_cv_score = float(best_trial.user_attrs.get("mean_test_score", best_objective))
        best_cv_std = float(best_trial.user_attrs.get("std_test_score", np.nan))
        best_train_score = float(best_trial.user_attrs.get("mean_train_score", np.nan))
        best_overfit_gap = float(best_trial.user_attrs.get("mean_overfit_gap", np.nan))

        print(f"\n{'=' * 50}\nOPTIMIZATION RESULTS\n{'=' * 50}")
        print(f"Best objective: {best_objective:.4f}")
        print(f"Best mean CV {scoring.upper()}: {best_cv_score:.4f} ± {best_cv_std:.4f}")
        print(f"Best mean CV train score: {best_train_score:.4f}")
        print(f"Best mean overfit gap (train-test): {best_overfit_gap:.4f}")
        print("Best Parameters:")
        for param, value in sorted(best_params.items()):
            print(f"  {param}: {value}")
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"Trials completed: {len(study.trials)} | Pruned: {pruned_trials}")

        # Refit on full dataset
        self.best_params = best_params
        best_model = self._build_model(best_params)
        best_model.fit(X[feature_cols], y)
        self.model = best_model

        self.optimization_results = {
            "best_score": best_objective,
            "best_cv_score": best_cv_score,
            "best_cv_std": best_cv_std,
            "best_train_score": best_train_score,
            "best_overfit_gap": best_overfit_gap,
            "best_params": best_params,
            "best_estimator": best_model,
            "optuna_study": study,
            "score_std_penalty": score_std_penalty,
            "overfit_penalty": overfit_penalty,
        }

        print(f"Optimization time: {time.time() - start_time:.2f}s\n")
        return self.optimization_results

    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series) -> BaggingClassifier:
        """Train final model with optimized hyperparameters."""
        if self.feature_names is not None:
            X = X[self.feature_names]
        model = self._build_model(self.best_params)
        model.fit(X, y)
        self.model = model
        return model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for samples."""
        if self.feature_names is not None:
            X = X[self.feature_names]
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for samples."""
        if self.feature_names is not None:
            X = X[self.feature_names]
        return self.model.predict(X)

    def predict_with_threshold(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict using custom probability threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on
        threshold : float
            Decision threshold (default 0.5)

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1)

        """
        if self.feature_names is not None:
            X = X[self.feature_names]
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= threshold).astype(int)


@dataclass
class ModelEvaluation:
    """Evaluate bagging model performance on train/test splits."""

    model: object
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    metrics: dict[str, dict[str, float]] = field(init=False, default_factory=dict)
    _resolved_model: BaggingClassifier = field(init=False)
    _feature_names: list[str] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._resolved_model = self._resolve_model()
        self.metrics = {
            "train": self._compute_split_metrics(self.X_train, self.y_train),
            "test": self._compute_split_metrics(self.X_test, self.y_test),
        }

    def _resolve_model(self) -> BaggingClassifier:
        if isinstance(self.model, BinaryModelBagging):
            model = self.model.model
            self._feature_names = self.model.feature_names
            return model

        if isinstance(self.model, BaggingClassifier):
            if hasattr(self.model, "feature_names_in_"):
                self._feature_names = list(self.model.feature_names_in_)
            return self.model

        raise TypeError("model must be BaggingClassifier or BinaryModelBagging.")

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._feature_names is None:
            return X
        return X[self._feature_names]

    def _compute_split_metrics(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        X_eval = self._prepare_features(X)
        y_true = np.asarray(y).ravel().astype(int)
        y_pred = self._resolved_model.predict(X_eval)
        y_proba = self._resolved_model.predict_proba(X_eval)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        auc = float(roc_auc_score(y_true, y_proba)) if np.unique(y_true).size > 1 else float("nan")
        log_loss_val = float(log_loss(y_true, np.clip(y_proba, 1e-12, 1 - 1e-12), labels=[0, 1]))

        return {
            "log_loss": log_loss_val,
            "auc_roc": auc,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_samples": float(y_true.shape[0]),
            "n_positive": float((y_true == 1).sum()),
            "n_negative": float((y_true == 0).sum()),
        }

    def optimize_threshold(
        self,
        metric: str = "youden",
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
    ) -> dict:
        """Find optimal prediction threshold using ROC curve analysis.

        Parameters
        ----------
        metric : str
            'youden': Maximize Youden's J (TPR - FPR) - recommended
            'f1': Maximize F1 score
            'balanced_accuracy': Maximize (TPR + TNR) / 2
        X, y : optional
            Dataset to optimize on. Uses test set by default.

        Returns
        -------
        dict with keys:
            - optimal_threshold: float
            - metrics_at_threshold: dict (accuracy, precision, recall, f1)
            - fpr, tpr, thresholds: ROC curve arrays
            - per_class_recall: dict with 'loss' and 'win' recalls

        """
        # Use test set by default
        if X is None or y is None:
            X, y = self.X_test, self.y_test

        X_eval = self._prepare_features(X)
        y_true = np.asarray(y).ravel().astype(int)
        y_proba = self._resolved_model.predict_proba(X_eval)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        # Calculate metric for each threshold
        if metric == "youden":
            # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
            scores = tpr - fpr
        elif metric == "f1":
            # Calculate F1 at each threshold
            scores = np.array(
                [
                    precision_recall_fscore_support(
                        y_true, (y_proba >= t).astype(int), average="binary", zero_division=0
                    )[2]
                    for t in thresholds
                ]
            )
        elif metric == "balanced_accuracy":
            # (TPR + TNR) / 2
            tnr = 1 - fpr
            scores = (tpr + tnr) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = float(thresholds[optimal_idx])

        # Compute metrics at optimal threshold
        y_pred_opt = (y_proba >= optimal_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_opt, average="binary", zero_division=0
        )

        # Per-class recall
        recall_loss = float(((y_true == 0) & (y_pred_opt == 0)).sum() / (y_true == 0).sum())
        recall_win = float(((y_true == 1) & (y_pred_opt == 1)).sum() / (y_true == 1).sum())

        return {
            "optimal_threshold": optimal_threshold,
            "metric_used": metric,
            "metric_value": float(scores[optimal_idx]),
            "metrics_at_threshold": {
                "accuracy": float(accuracy_score(y_true, y_pred_opt)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            },
            "per_class_recall": {
                "loss": recall_loss,
                "win": recall_win,
            },
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }

    def predict_with_threshold(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict using custom probability threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on
        threshold : float
            Decision threshold (default 0.5)

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1)

        """
        X_eval = self._prepare_features(X)
        y_proba = self._resolved_model.predict_proba(X_eval)[:, 1]
        return (y_proba >= threshold).astype(int)

    @staticmethod
    def _format_value(value: float) -> str:
        if np.isnan(value):
            return "N/A"
        return f"{value:.4f}"

    def _print_split_classification_report(
        self,
        split_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        class_names: tuple[str, str] = ("Loss", "Win"),
        digits: int = 2,
    ) -> None:
        y_true = np.asarray(y).ravel().astype(int)
        y_pred = self._resolved_model.predict(self._prepare_features(X))

        print(f"\n{split_name.upper()} CLASSIFICATION REPORT")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=list(class_names),
                digits=digits,
                zero_division=0,
            )
        )

    def print_metrics_summary_table(
        self,
        include_classification_report: bool = True,
        class_names: tuple[str, str] = ("Loss", "Win"),
    ) -> None:
        print(f"\n{'=' * 56}\nMODEL EVALUATION SUMMARY\n{'=' * 56}")
        print(f"{'Metric':<18} {'Train':>18} {'Test':>18}")
        print(f"{'-' * 56}")

        metric_pairs = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 Score"),
            ("auc_roc", "AUC-ROC"),
            ("log_loss", "Log Loss"),
        ]
        for key, label in metric_pairs:
            train_val = self._format_value(self.metrics["train"][key])
            test_val = self._format_value(self.metrics["test"][key])
            print(f"{label:<18} {train_val:>18} {test_val:>18}")

        print(f"{'-' * 56}")
        print(
            f"{'Samples':<18} "
            f"{int(self.metrics['train']['n_samples']):>18d} "
            f"{int(self.metrics['test']['n_samples']):>18d}"
        )
        print(
            f"{'Positive':<18} "
            f"{int(self.metrics['train']['n_positive']):>18d} "
            f"{int(self.metrics['test']['n_positive']):>18d}"
        )
        print(f"{'=' * 56}\n")

        if include_classification_report:
            self._print_split_classification_report("Train", self.X_train, self.y_train, class_names=class_names)
            self._print_split_classification_report("Test", self.X_test, self.y_test, class_names=class_names)

    def _plot_single_roc(
        self, ax, X: pd.DataFrame, y: pd.Series, title: str, split_name: str, style=DEFAULT_STYLE
    ) -> None:
        y_true = np.asarray(y).ravel().astype(int)

        ax.plot([0, 1], [0, 1], linestyle="--", color=style.muted, alpha=0.7, label="Random")
        if np.unique(y_true).size < 2:
            ax.text(
                0.5,
                0.5,
                "ROC undefined\n(only one class)",
                ha="center",
                va="center",
                color=style.font_color,
            )
            ax.set_title(title)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            _apply_axis_style(ax, style)
            _style_legend(ax, style, loc="lower right")
            return

        # Reuse AUC from stored metrics instead of recomputing
        auc_val = self.metrics[split_name]["auc_roc"]

        X_eval = self._prepare_features(X)
        y_proba = self._resolved_model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(
            fpr,
            tpr,
            linewidth=2.0,
            color=style.accent1,
            label=f"Model (AUC = {auc_val:.4f})",
        )
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        _apply_axis_style(ax, style)
        _style_legend(ax, style, loc="lower right")

    def plot_roc_auc_curves(self, figsize: tuple[int, int] = (12, 5)):
        """Plot ROC-AUC curves for train and test datasets."""
        style = DEFAULT_STYLE
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor(style.paper_bgcolor)
        self._plot_single_roc(axes[0], self.X_train, self.y_train, "Train ROC Curve", "train", style=style)
        self._plot_single_roc(axes[1], self.X_test, self.y_test, "Test ROC Curve", "test", style=style)
        fig.suptitle("ROC-AUC Curves", y=1.02, color=style.font_color)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        return fig

    def plot_learning_curve(
        self,
        cv: int = 5,
        scoring: str = "neg_log_loss",
        train_sizes: np.ndarray | None = None,
        figsize: tuple[int, int] = (10, 6),
        n_jobs: int = 1,
    ):
        """Plot learning curve with CV scores across training set sizes."""
        X_train_eval = self._prepare_features(self.X_train)
        y_train_arr = np.asarray(self.y_train).ravel().astype(int)

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 8)

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=self._resolved_model,
            X=X_train_eval,
            y=y_train_arr,
            cv=cv_splitter,
            scoring=scoring,
            train_sizes=train_sizes,
            n_jobs=6,  # Mac-optimized, max 6 threads
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        if scoring == "neg_log_loss":
            train_curve = -train_scores
            val_curve = -val_scores
            ylabel, title = "Log Loss", "Learning Curve (Log Loss)"
            test_ref = self.metrics["test"]["log_loss"]
        else:
            train_curve = train_scores
            val_curve = val_scores
            ylabel, title = "ROC-AUC", "Learning Curve (ROC-AUC)"
            test_ref = self.metrics["test"]["auc_roc"]

        train_mean = train_curve.mean(axis=1)
        train_std = train_curve.std(axis=1)
        val_mean = val_curve.mean(axis=1)
        val_std = val_curve.std(axis=1)

        style = DEFAULT_STYLE
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(style.paper_bgcolor)
        ax.plot(train_sizes_abs, train_mean, linewidth=2.0, color=style.accent1, label="Train")
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            color=style.accent1,
            alpha=0.2,
        )
        ax.plot(train_sizes_abs, val_mean, linewidth=2.0, color=style.accent2, label="Cross-validation")
        ax.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            color=style.accent2,
            alpha=0.2,
        )

        if not np.isnan(test_ref):
            ax.axhline(test_ref, linestyle="--", color=style.muted, alpha=0.8, label="Held-out test")

        ax.set_title(title)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel(ylabel)
        _apply_axis_style(ax, style)
        _style_legend(ax, style, loc="best")
        fig.tight_layout()
        return fig

    def plot_threshold_optimization(self, threshold_result: dict, figsize: tuple[int, int] = (12, 5)):
        """Plot ROC curve with optimal threshold marked.

        Parameters
        ----------
        threshold_result : dict
            Output from optimize_threshold()
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure

        """
        style = DEFAULT_STYLE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor(style.paper_bgcolor)

        fpr = threshold_result["fpr"]
        tpr = threshold_result["tpr"]
        thresholds = threshold_result["thresholds"]
        optimal_threshold = threshold_result["optimal_threshold"]

        # Find optimal point on ROC curve
        optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        # Left: ROC curve with optimal point
        ax1.plot([0, 1], [0, 1], linestyle="--", color=style.muted, alpha=0.7, label="Random")
        ax1.plot(fpr, tpr, linewidth=2.0, color=style.accent1, label="ROC curve")
        ax1.scatter(
            optimal_fpr,
            optimal_tpr,
            s=100,
            color=style.accent2,
            marker="o",
            zorder=5,
            label=f"Optimal (threshold={optimal_threshold:.3f})",
        )
        ax1.set_title("ROC Curve with Optimal Threshold")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        _apply_axis_style(ax1, style)
        _style_legend(ax1, style, loc="lower right")

        # Right: Per-class recall comparison
        recalls_default = [
            self.metrics["test"]["recall"],  # Default threshold (binary average)
            1 - self.metrics["test"]["recall"],  # Inverse approximation
        ]
        recalls_optimal = [
            threshold_result["per_class_recall"]["loss"],
            threshold_result["per_class_recall"]["win"],
        ]

        x = np.arange(2)
        width = 0.35
        ax2.bar(x - width / 2, recalls_default, width, label="Threshold=0.5", color=style.accent1, alpha=0.8)
        ax2.bar(
            x + width / 2,
            recalls_optimal,
            width,
            label=f"Threshold={optimal_threshold:.3f}",
            color=style.accent2,
            alpha=0.8,
        )

        ax2.set_title("Per-Class Recall Comparison")
        ax2.set_ylabel("Recall")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["Loss (Class 0)", "Win (Class 1)"])
        ax2.set_ylim([0, 1.05])
        _apply_axis_style(ax2, style)
        _style_legend(ax2, style, loc="best")

        fig.suptitle("Threshold Optimization Results", y=1.02, color=style.font_color)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        return fig
