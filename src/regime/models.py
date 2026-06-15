import pickle

import numpy as np
import pandas as pd
from hmmlearn import hmm
from joblib import Parallel, delayed


class HMMEnsembleRegimeDetector:
    """
    PRODUCTION ENSEMBLE HMM with out-of-sample prediction capability.

    KEY FEATURES:
    - Separate fit() and predict() workflows for proper train/test separation
    - Stores fitted HMM models for reuse on unseen data
    - Prevents look-ahead bias through temporal feature preprocessing
    - Optional EMA smoothing via use_preprocessing parameter
    - Model serialization for deployment

    Mathematical Foundation:
    - Enforces minimum self-transition probability: P(s_t = i | s_{t-1} = i) >= min_self_prob
    - Expected regime duration: E[D] = 1 / (1 - min_self_prob)

    CRITICAL USAGE:
    >>> # TRAIN PHASE
    >>> detector = HMMEnsembleRegimeDetector(n_estimators=1000, n_components=3, use_preprocessing=True)
    >>> detector.fit(features_train)
    >>> states_train, probs_train = detector.predict(features_train)
    >>>
    >>> # TEST PHASE (OUT-OF-SAMPLE)
    >>> states_test, probs_test = detector.predict(features_test)  # Uses fitted models
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        n_components: int = 2,
        min_features: int = 3,
        max_features: int = 8,
        feature_weight_strategy: str = "linear",
        aggregation_method: str = "average",
        min_self_prob: float = 0.85,
        use_preprocessing: bool = True,  # NEW: Control EMA smoothing
        smooth_span: int = 100,
        covariance_type: str = "spherical",
        min_covar: float = 0.05,
        n_iter: int = 1000,
        random_state: int = 0,
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        # Ensemble configuration
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.min_features = min_features
        self.max_features = max_features
        self.feature_weight_strategy = feature_weight_strategy
        self.aggregation_method = aggregation_method

        # HMM parameters
        self.min_self_prob = min_self_prob
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_iter = n_iter

        # Preprocessing (NEW: Optional EMA smoothing)
        self.use_preprocessing = use_preprocessing
        self.smooth_span = smooth_span

        # Execution
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # State attributes for out-of-sample prediction
        self.ensemble_models_: list[hmm.GaussianHMM] | None = None
        self.ensemble_results_: list[dict] | None = None
        self.feature_subsets_: list[list[int]] | None = None
        self.feature_names_: list[str] | None = None
        self.preprocessing_params_: dict | None = None
        self.is_fitted_: bool = False

    # ======================================================================== #
    # TEMPORAL FEATURE PREPROCESSING (CRITICAL FOR OUT-OF-SAMPLE)
    # ======================================================================== #

    def _preprocess_features_fit(self, features: pd.DataFrame) -> np.ndarray:
        """
        Fit EWM smoother on TRAINING data (if enabled) and store parameters.

        CRITICAL: When preprocessing disabled, still stores last N observations
        for consistent API between train/test.
        """
        if not self.use_preprocessing:
            # No smoothing - store raw features for validation
            self.preprocessing_params_ = {
                "use_preprocessing": False,
                "last_train_values": features.iloc[-self.smooth_span :].copy(),
            }
            return features.values

        # Apply EMA smoothing
        features_smooth_df = features.ewm(span=self.smooth_span, min_periods=int(self.smooth_span / 4)).mean()

        # Store preprocessing state for test set alignment
        self.preprocessing_params_ = {
            "use_preprocessing": True,
            "smooth_span": self.smooth_span,
            "min_periods": int(self.smooth_span / 4),
            "last_train_values": features.iloc[-self.smooth_span :].copy(),
        }

        return features_smooth_df.values

    def _preprocess_features_transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        Apply FITTED preprocessing to test data WITHOUT look-ahead bias.

        CRITICAL: When preprocessing disabled, returns raw features.
        When enabled, uses last N training observations to initialize EWM.
        """
        if self.preprocessing_params_ is None:
            raise ValueError("Must fit model before transforming test features")

        # Validate feature alignment
        if features.shape[1] != len(self.feature_names_):
            raise ValueError(f"Feature count mismatch: expected {len(self.feature_names_)}, got {features.shape[1]}")

        # No preprocessing path - return raw features
        if not self.preprocessing_params_["use_preprocessing"]:
            return features.values

        # EMA smoothing path - concatenate with train window for proper continuation
        last_train = self.preprocessing_params_["last_train_values"]
        combined = pd.concat([last_train, features], axis=0)

        # Apply EWM with same parameters as training
        combined_smooth = combined.ewm(
            span=self.preprocessing_params_["smooth_span"], min_periods=self.preprocessing_params_["min_periods"]
        ).mean()

        # Extract only test portion (remove prepended train data)
        features_smooth = combined_smooth.iloc[len(last_train) :].values

        return features_smooth

    # ======================================================================== #
    # FEATURE SUBSET GENERATION
    # ======================================================================== #

    def _generate_feature_subsets(self, n_features: int) -> list[list[int]]:
        """Generate random feature subsets for ensemble diversity."""
        min_cols_adj = max(1, self.min_features)
        max_cols_adj = min(n_features, self.max_features)
        possible_counts = np.arange(min_cols_adj, max_cols_adj + 1)

        # Calculate sampling weights
        if self.feature_weight_strategy == "linear":
            weights = np.arange(len(possible_counts), 0, -1).astype(float)
        elif self.feature_weight_strategy == "exponential":
            weights = np.exp(-0.5 * np.arange(len(possible_counts)))
        elif self.feature_weight_strategy == "quadratic":
            weights = (len(possible_counts) - np.arange(len(possible_counts))) ** 2
        elif self.feature_weight_strategy == "uniform":
            weights = np.ones(len(possible_counts))
        else:
            raise ValueError(f"Unknown weight strategy: {self.feature_weight_strategy}")

        weights = weights / weights.sum()

        # Generate subsets with different random states
        feature_subsets = []
        for i in range(self.n_estimators):
            rng = np.random.RandomState(self.random_state + i)
            n_cols = rng.choice(possible_counts, p=weights)
            selected_indices = rng.choice(n_features, size=n_cols, replace=False)
            feature_subsets.append(selected_indices.tolist())

        if self.verbose >= 1:
            subset_sizes = [len(s) for s in feature_subsets]
            print(f"  Feature range: {min(subset_sizes)}-{max(subset_sizes)} columns per model")
            print(f"  Average features per model: {np.mean(subset_sizes):.2f}")

        return feature_subsets

    # ======================================================================== #
    # HMM TRAINING (STORES FITTED MODELS)
    # ======================================================================== #

    def _fit_single_hmm(
        self, features_smooth: np.ndarray, feature_subset_cols: list[int], random_state: int
    ) -> dict | None:
        """
        Fit single HMM and RETURN THE FITTED MODEL OBJECT.

        CRITICAL: Returns the trained model for out-of-sample prediction.
        """
        features_subset = features_smooth[:, feature_subset_cols]

        # Initialize persistent HMM
        hmm_model = self._PersistentHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            random_state=random_state,
            min_self_prob=self.min_self_prob,
        )

        # Fit model with convergence monitoring
        try:
            hmm_model.fit(features_subset)
        except Exception as e:
            if self.verbose >= 2:
                print(f"Model {random_state} failed: {e}")
            return None

        # Generate train predictions
        states = hmm_model.predict(features_subset)
        probs = hmm_model.predict_proba(features_subset)

        # Convergence validation
        converged = hmm_model.monitor_.converged
        log_likelihood = hmm_model.score(features_subset)
        n_iter = hmm_model.monitor_.iter

        # Validate log-likelihood monotonicity
        likelihood_history = list(hmm_model.monitor_.history)
        likelihood_stable = True
        if len(likelihood_history) >= 2 and likelihood_history[-1] < likelihood_history[-2]:
            likelihood_stable = False

        return {
            "model": hmm_model,
            "states": states,
            "probs": probs,
            "converged": converged,
            "likelihood_stable": likelihood_stable,
            "log_likelihood": log_likelihood,
            "n_iter": n_iter,
            "model_id": random_state,
            "feature_subset": feature_subset_cols,
        }

    class _PersistentHMM(hmm.GaussianHMM):
        """Custom HMM enforcing minimum self-transition probability."""

        def __init__(self, min_self_prob: float = 0.85, **kwargs):
            super().__init__(**kwargs)
            self.min_self_prob = min_self_prob

        def _init(self, X, lengths=None) -> None:
            """Initialize transition matrix with persistence constraint."""
            super()._init(X, lengths)
            k = self.n_components

            if k == 1:
                self.transmat_ = np.array([[1.0]])
                return

            off_diag = (1 - self.min_self_prob) / (k - 1)
            self.transmat_ = np.full((k, k), off_diag)
            np.fill_diagonal(self.transmat_, self.min_self_prob)

        def _do_mstep(self, stats: dict) -> None:
            """M-step with transition matrix constraint enforcement."""
            super()._do_mstep(stats)
            k = self.n_components

            for i in range(k):
                if self.transmat_[i, i] < self.min_self_prob:
                    deficit = self.min_self_prob - self.transmat_[i, i]
                    off_diag_sum = 1.0 - self.transmat_[i, i]

                    if off_diag_sum > deficit and k > 1:
                        scale = (off_diag_sum - deficit) / off_diag_sum
                        for j in range(k):
                            if i != j:
                                self.transmat_[i, j] *= scale
                    else:
                        for j in range(k):
                            if i != j:
                                self.transmat_[i, j] = (1 - self.min_self_prob) / max(k - 1, 1)

                    self.transmat_[i, i] = self.min_self_prob

            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            self.transmat_ = self.transmat_ / np.maximum(row_sums, 1e-10)

    # ======================================================================== #
    # CORE FIT METHOD (TRAINING ONLY)
    # ======================================================================== #

    def fit(self, features: pd.DataFrame) -> "HMMEnsembleRegimeDetector":
        """
        Fit ensemble HMM on TRAINING features.

        CRITICAL: Only call this on train split. Use predict() for test data.
        """
        if features.empty:
            raise ValueError("features cannot be empty")

        if self.verbose >= 1:
            print(f"\n{'=' * 70}")
            print("ENSEMBLE HMM TRAINING")
            print(f"{'=' * 70}")
            print(f"Feature Matrix: {features.shape}")
            preprocessing_status = "ENABLED (EMA)" if self.use_preprocessing else "DISABLED (Raw)"
            print(f"Preprocessing: {preprocessing_status}")

        # Step 1: Fit preprocessing on train data
        if self.verbose >= 1:
            status = (
                "Fitting EWM smoother on TRAIN data" if self.use_preprocessing else "Using RAW features (no smoothing)"
            )
            print(f"\nStep 1: {status}")

        self.feature_names_ = features.columns.tolist()
        features_smooth_train = self._preprocess_features_fit(features)

        # Step 2: Generate feature subsets
        if self.verbose >= 1:
            print(f"\nStep 2: Generating {self.n_estimators} random feature subsets")

        self.feature_subsets_ = self._generate_feature_subsets(features.shape[1])

        # Step 3: Fit ensemble
        if self.verbose >= 1:
            print("\nStep 3: Training ensemble HMM models")

        verbose_level = 5 if self.verbose >= 2 else 0

        raw_results = Parallel(n_jobs=self.n_jobs, verbose=verbose_level)(
            delayed(self._fit_single_hmm)(
                features_smooth=features_smooth_train,
                feature_subset_cols=self.feature_subsets_[i],
                random_state=self.random_state + i,
            )
            for i in range(self.n_estimators)
        )

        # Filter valid models
        valid_results = [r for r in raw_results if r is not None and r["converged"] and r["likelihood_stable"]]

        if len(valid_results) == 0:
            raise RuntimeError("All models failed convergence. Increase n_iter or reduce min_self_prob.")

        # Store fitted models AND training results
        self.ensemble_models_ = [r["model"] for r in valid_results]
        self.ensemble_results_ = valid_results

        # Diagnostics
        n_failed = sum(1 for r in raw_results if r is None)
        n_not_converged = sum(1 for r in raw_results if r is not None and not r["converged"])
        n_unstable = sum(1 for r in raw_results if r is not None and r["converged"] and not r["likelihood_stable"])
        n_valid = len(valid_results)

        if self.verbose >= 1:
            convergence_rate = 100 * n_valid / self.n_estimators
            print("\nModel Filtering Results:")
            print(f"  Total models:               {self.n_estimators}")
            print(f"  Failed (exceptions):        {n_failed}")
            print(f"  Non-converged (iter limit): {n_not_converged}")
            print(f"  Unstable (likelihood drop): {n_unstable}")
            print(f"  Valid (kept):               {n_valid} ({convergence_rate:.1f}%)")

            if convergence_rate < 50:
                print("\n  ⚠ WARNING: Low convergence rate")

            if n_unstable > 0.1 * self.n_estimators:
                print("\n  ⚠ WARNING: High instability rate - increase min_covar")

        self.is_fitted_ = True
        return self

    # ======================================================================== #
    # OUT-OF-SAMPLE PREDICTION (CRITICAL METHOD)
    # ======================================================================== #

    def predict(
        self, features: pd.DataFrame, df: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray] | pd.DataFrame:
        """
        Predict regimes using FITTED models (train or out-of-sample data).

        CRITICAL: This method does NOT retrain. It applies stored models to
        features with proper temporal preprocessing (if enabled).

        Parameters
        ----------
        features : pd.DataFrame
            Features for prediction (train or test)
        df : pd.DataFrame, optional
            Original dataframe to augment with regime predictions

        Returns
        -------
        If df is None:
            Tuple[np.ndarray, np.ndarray]
                (ensemble_states, ensemble_probs)
        If df is provided:
            pd.DataFrame
                Augmented dataframe with regime columns

        Examples
        --------
        >>> # Out-of-sample prediction
        >>> detector.fit(features_train)
        >>> states_test, probs_test = detector.predict(features_test)
        >>>
        >>> # With dataframe augmentation
        >>> df_test_with_regimes = detector.predict(features_test, df_test)
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")

        # Validate feature alignment
        if features.shape[1] != len(self.feature_names_):
            raise ValueError(f"Feature count mismatch: expected {len(self.feature_names_)}, got {features.shape[1]}")

        if df is not None and len(features) != len(df):
            raise ValueError(f"Length mismatch: features={len(features)}, df={len(df)}")

        if self.verbose >= 1:
            print(f"\n{'=' * 70}")
            print("PREDICTION (FITTED MODELS)")
            print(f"{'=' * 70}")
            print(f"Feature Matrix: {features.shape}")

        # Step 1: Apply FITTED preprocessing (no refitting)
        if self.verbose >= 1:
            status = (
                "Applying fitted EWM smoother (temporal causality preserved)"
                if self.use_preprocessing
                else "Using RAW features (no smoothing)"
            )
            print(f"\nStep 1: {status}")

        features_smooth = self._preprocess_features_transform(features)

        # Step 2: Generate predictions from STORED models
        if self.verbose >= 1:
            print(f"\nStep 2: Generating predictions from {len(self.ensemble_models_)} fitted models")

        n_samples = features_smooth.shape[0]
        all_probs = np.zeros((len(self.ensemble_models_), n_samples, self.n_components))

        for i, (model, result) in enumerate(zip(self.ensemble_models_, self.ensemble_results_, strict=True)):
            feature_subset = result["feature_subset"]
            features_subset = features_smooth[:, feature_subset]

            # Predict using FITTED model (no retraining)
            all_probs[i] = model.predict_proba(features_subset)

        # Step 3: Aggregate predictions
        if self.verbose >= 1:
            print(f"\nStep 3: Aggregating via '{self.aggregation_method}'")

        if self.aggregation_method == "average":
            ensemble_probs = all_probs.mean(axis=0)
        elif self.aggregation_method == "convergence_weighted":
            log_likelihoods = np.array([r["log_likelihood"] for r in self.ensemble_results_])
            weights = np.exp(log_likelihoods - log_likelihoods.max())
            weights = weights / weights.sum()
            ensemble_probs = (all_probs.T @ weights).T
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation_method}")

        ensemble_states = ensemble_probs.argmax(axis=1)

        # Validation
        self._validate_regimes(ensemble_states, ensemble_probs)

        # Return format
        if df is None:
            return ensemble_states, ensemble_probs

        # Augment dataframe
        df_result = df.copy()
        for i in range(self.n_components):
            df_result[f"prob_regime_{i}"] = ensemble_probs[:, i]
        df_result["regime"] = ensemble_states

        return df_result

    # ======================================================================== #
    # CONVENIENCE METHOD (TRAIN ONLY)
    # ======================================================================== #

    def fit_predict(
        self, features: pd.DataFrame, df: pd.DataFrame | None = None
    ) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
        """
        Fit ensemble and return TRAINING predictions.

        WARNING: DO NOT use this for test data. Use fit() then predict() instead.
        """
        self.fit(features)
        return self.predict(features, df)

    # ======================================================================== #
    # VALIDATION & DIAGNOSTICS
    # ======================================================================== #

    def _validate_regimes(self, states: np.ndarray, probs: np.ndarray) -> None:
        """Validate regime detection quality and print diagnostics."""
        if self.verbose < 1:
            return

        print("\n" + "=" * 70)
        print("REGIME VALIDATION")
        print("=" * 70)

        state_series = pd.Series(states)
        transitions = (state_series.diff() != 0).sum()
        avg_duration = len(states) / max(transitions, 1)

        print(f"\nTransitions: {transitions}")
        print(f"Average Duration: {avg_duration:.1f} periods")
        print(f"Status: {'✓ PASS' if avg_duration >= 15 else '✗ FAIL - Too volatile'}")

        max_probs = probs.max(axis=1)
        print(f"\nEnsemble Confidence (mean max prob): {max_probs.mean():.3f}")
        print(f"Ensemble Uncertainty (std of probs): {probs.std(axis=1).mean():.3f}")

        print("\nRegime Distribution:")
        for i in range(self.n_components):
            count = (states == i).sum()
            pct = 100 * count / len(states)
            print(f"  Regime {i}: {count:5d} periods ({pct:5.2f}%)")

    def get_convergence_stats(self) -> pd.DataFrame:
        """Extract ensemble convergence statistics for model diagnostics."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        stats = pd.DataFrame(
            [
                {
                    "model_id": r["model_id"],
                    "converged": r["converged"],
                    "likelihood_stable": r["likelihood_stable"],
                    "log_likelihood": r["log_likelihood"],
                    "n_iter": r["n_iter"],
                    "n_features": len(r["feature_subset"]),
                }
                for r in self.ensemble_results_
            ]
        )

        return stats

    # ======================================================================== #
    # MODEL PERSISTENCE
    # ======================================================================== #

    def save(self, filepath: str) -> None:
        """Serialize fitted model for deployment."""
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model")

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "ensemble_models": self.ensemble_models_,
                    "ensemble_results": self.ensemble_results_,
                    "feature_subsets": self.feature_subsets_,
                    "feature_names": self.feature_names_,
                    "preprocessing_params": self.preprocessing_params_,
                    "hyperparameters": {
                        "n_components": self.n_components,
                        "n_estimators": len(self.ensemble_models_),
                        "min_self_prob": self.min_self_prob,
                        "aggregation_method": self.aggregation_method,
                        "use_preprocessing": self.use_preprocessing,  # NEW: Persist flag
                        "smooth_span": self.smooth_span,
                    },
                },
                f,
            )

        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, verbose: int = 1) -> "HMMEnsembleRegimeDetector":
        """Load fitted model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        detector = cls(**data["hyperparameters"], verbose=verbose)
        detector.ensemble_models_ = data["ensemble_models"]
        detector.ensemble_results_ = data["ensemble_results"]
        detector.feature_subsets_ = data["feature_subsets"]
        detector.feature_names_ = data["feature_names"]
        detector.preprocessing_params_ = data["preprocessing_params"]
        detector.is_fitted_ = True

        preprocessing_status = "ENABLED" if detector.use_preprocessing else "DISABLED"
        print(f"✓ Model loaded from {filepath}")
        print(f"  - {len(detector.ensemble_models_)} HMM models")
        print(f"  - {len(detector.feature_names_)} features")
        print(f"  - {detector.n_components} regimes")
        print(f"  - Preprocessing: {preprocessing_status}")

        return detector
