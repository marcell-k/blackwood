import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from blackwood.config import RANDOM_STATE
from blackwood.regime.features import RegimeFeatureEngineer

# Feature sets
GMM_FEATURES = ["atr_40_4h", "realized_vol_20_4h", "atr_40_1h", "atr_14_15min", "realized_vol_20_15min"]
RF_FEATURES = [
    "vol_of_vol_20_15min",
    "vol_of_vol_40_15min",
    "adp_atr_15min",
    "atr_5_1h",
    "atr_14_1h",
    "atr_21_1h",
    "atr_40_1h",
    "realized_vol_20_1h",
    "vol_of_vol_20_1h",
    "vol_of_vol_40_1h",
    "adp_atr_1h",
    "vol_of_vol_20_4h",
    "vol_of_vol_40_4h",
    "adp_atr_4h",
]


class RegimeDetector:
    def __init__(
        self,
        gmm_features=GMM_FEATURES,
        rf_features=RF_FEATURES,
        n_components: int = 3,
        confidence_threshold: float = 0.6,
        random_state: int = RANDOM_STATE,
    ):
        self.gmm_features = gmm_features
        self.rf_features = rf_features
        self.n_components = n_components
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", n_init=10, random_state=random_state
        )
        self.hmm = GaussianHMM(
            n_components=n_components, covariance_type="diag", n_iter=2000, tol=1e-4, random_state=random_state
        )
        self.encoder = OneHotEncoder(sparse_output=False)
        self.rf = RandomForestClassifier(
            n_estimators=50, max_depth=3, min_samples_leaf=200, max_features="sqrt", random_state=random_state
        )

    def fit(self, X: pd.DataFrame) -> "RegimeDetector":
        # GMM: soft cluster assignments
        gmm_data = X[self.gmm_features].to_numpy()
        gmm_scaled = self.scaler.fit_transform(gmm_data)
        self.gmm.fit(gmm_scaled)
        gmm_probs = np.clip(self.gmm.predict_proba(gmm_scaled), 1e-6, 1.0 - 1e-6)

        # HMM: sequence decoding on GMM posteriors
        self.hmm.fit(gmm_probs)
        full_regimes = self.hmm.predict(gmm_probs)

        # RF: transition prediction (current regime + features -> next regime)
        current_regimes = full_regimes[:-1]
        next_regimes = full_regimes[1:]
        regime_ohe = self.encoder.fit_transform(current_regimes.reshape(-1, 1))

        # Precompute one-hot bases for all possible regimes
        self.ohe_bases = self.encoder.transform(np.arange(self.n_components).reshape(-1, 1))

        rf_data = X[self.rf_features].to_numpy()[:-1]
        transition_X = np.hstack([regime_ohe, rf_data])
        self.rf.fit(transition_X, next_regimes)

        # In-sample forward regimes (causal, RF-based, no gating)
        pred_transitions = self.rf.predict(transition_X)
        self._train_forward_regimes = pd.Series(pred_transitions, index=X.index[1:])
        self._last_regime = int(pred_transitions[-1])

        # Store for diagnostics
        self._transition_X = transition_X
        self._transition_y = next_regimes
        self._current_regimes = current_regimes
        self._train_index = X.index

        return self

    def apply_forward_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the fitted DataFrame with in-sample forward regimes (drops first row)."""
        if not df.index.equals(self._train_index):
            raise ValueError("DataFrame index must match the fitted feature index")
        result = df.loc[self._train_forward_regimes.index].copy()
        result["regime"] = self._train_forward_regimes.values.astype(int)
        return result

    def forward_predict_regimes(self, X: pd.DataFrame, use_confidence_gating: bool = False) -> pd.Series:
        """
        Causally predict regimes for new (out-of-sample) data, continuing from the last training regime.
        If use_confidence_gating=True, stays in current regime when GMM posterior is low.
        """
        if not hasattr(self, "_last_regime"):
            raise RuntimeError("Detector must be fitted first.")
        if X.empty:
            return pd.Series(dtype=int, index=X.index)

        n_rows = len(X)
        n_comp = self.n_components
        rf_feats = X[self.rf_features].to_numpy()

        # Confidence computation (vectorized)
        if use_confidence_gating:
            gmm_feats = X[self.gmm_features].to_numpy()
            gmm_scaled = self.scaler.transform(gmm_feats)
            posteriors = np.clip(self.gmm.predict_proba(gmm_scaled), 1e-6, 1.0 - 1e-6)
            confidences = posteriors.max(axis=1)
        else:
            confidences = np.full(n_rows, 1.0)

        # Precompute RF predictions for every row and every possible current regime
        ohe_all = np.tile(self.ohe_bases, (n_rows, 1))  # (n_rows * n_comp, n_comp)
        rf_all = np.repeat(rf_feats, n_comp, axis=0)  # (n_rows * n_comp, n_rf)
        input_all = np.hstack([ohe_all, rf_all])
        all_predictions = self.rf.predict(input_all)
        next_pred = all_predictions.reshape(n_rows, n_comp)  # next_pred[i, current]

        # Sequential pass (lightweight)
        regimes = np.empty(n_rows, dtype=int)
        current_regime = self._last_regime
        for i in range(n_rows):
            if confidences[i] < self.confidence_threshold:
                next_regime = current_regime
            else:
                next_regime = int(next_pred[i, current_regime])
            regimes[i] = next_regime
            current_regime = next_regime

        return pd.Series(regimes, index=X.index)

    def predict_next_regime(self, current_regime: int, rf_features: np.ndarray, regime_prob: float) -> int:
        """Single-step prediction with confidence gating (mirrors forward_predict_regimes logic)."""
        if regime_prob < self.confidence_threshold:
            return current_regime
        ohe = self.ohe_bases[current_regime]
        X_input = np.hstack([ohe, rf_features])
        return int(self.rf.predict(X_input[None, :])[0])

    def print_diagnostics(self):
        pred = self.rf.predict(self._transition_X)
        print("RF confusion matrix:")
        print(confusion_matrix(self._transition_y, pred))
        print("\nHMM transition matrix:")
        print(self.hmm.transmat_)
        for r in range(self.n_components):
            mask = self._current_regimes == r
            if mask.any():
                probs = self.rf.predict_proba(self._transition_X[mask]).mean(axis=0)
                print(f"\nRF avg transition probs from regime {r}:")
                print(probs)


def fit_and_label_regimes(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    timeframes: list[str] | None = None,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    features_to_exclude: list[str] | None = None,
    verbose: bool = True,
    **detector_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit regime detector on train data and label both train and OOS with forward regimes.

    Parameters
    ----------
    features_to_exclude : list[str] | None
        Feature names to drop before fitting (e.g. from PCA/RMT analysis)
    verbose : bool
        If True, print diagnostics and show regime plots
    """
    if timeframes is None:
        timeframes = ["15min", "1h", "4h"]

    # Feature engineering
    X_train = RegimeFeatureEngineer(train, timeframes=timeframes).run()
    X_oos = RegimeFeatureEngineer(oos, timeframes=timeframes).run()

    # Compute clipping bounds from train only (prevent look-ahead bias)
    clip_lower = X_train.quantile(clip_quantiles[0])
    clip_upper = X_train.quantile(clip_quantiles[1])

    # Apply clipping
    X_train_clipped = X_train.clip(lower=clip_lower, upper=clip_upper, axis=1)
    X_oos_clipped = X_oos.clip(lower=clip_lower, upper=clip_upper, axis=1)

    # Drop excluded features
    if features_to_exclude:
        cols_to_drop = [c for c in features_to_exclude if c in X_train_clipped.columns]
        X_train_clipped = X_train_clipped.drop(columns=cols_to_drop)
        X_oos_clipped = X_oos_clipped.drop(columns=cols_to_drop)
        if verbose:
            print(f"Dropped {len(cols_to_drop)} features: {cols_to_drop}")
            print(f"Remaining features: {X_train_clipped.shape[1]}")

    # Fit detector
    detector = RegimeDetector(**detector_kwargs)
    detector.fit(X_train_clipped)

    # Label regimes (train drops first row, OOS continues from last train regime)
    train_labeled = detector.apply_forward_regimes(train)
    oos_regimes = detector.forward_predict_regimes(X_oos_clipped)

    oos_labeled = oos.copy()
    oos_labeled["regime"] = oos_regimes.astype(int)

    if verbose:
        from blackwood.regime.analysis import analyze_regime_statistics
        from blackwood.visualization.regime import plot_regime_candlesticks

        detector.print_diagnostics()
        print(f"\nTrain regime distribution:\n{train_labeled['regime'].value_counts().sort_index()}")
        print(f"\nOOS regime distribution:\n{oos_labeled['regime'].value_counts().sort_index()}")

        plot_regime_candlesticks(
            train_labeled[-4000:], mark_transitions=False, per_bar_shading=True, use_session_hours=False
        )
        plot_regime_candlesticks(
            oos_labeled[-4000:], mark_transitions=False, per_bar_shading=True, use_session_hours=False
        )

        n_components = detector_kwargs.get("n_components", 3)
        regime_directions = {i: 1 for i in range(n_components)}
        analyze_regime_statistics(train_labeled, regime_directions=regime_directions)
        analyze_regime_statistics(oos_labeled, regime_directions=regime_directions)

    return train_labeled, oos_labeled
