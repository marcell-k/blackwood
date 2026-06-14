# Standard library imports

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from src.meta_labeling.utils import probability_to_bet_size


class ProbabilityCalibrator:
    """Post-hoc probability calibration: Platt, Beta, Temperature (base methods)
    or Ensemble, Stacking (meta-methods).
    """

    VALID_BASE = {"platt", "beta", "temperature"}
    VALID_META = {"ensemble", "stacking"}

    def __init__(
        self,
        method: str = "platt",
        base_methods: tuple = ("platt", "beta", "temperature"),
        verbose: bool = False,
    ):
        self.method = method.lower()
        self.base_methods = base_methods
        self.verbose = verbose
        self.model = None
        self.T = None
        self._fitted = False
        self.base_calibrators = {}
        self.meta_model = None

        if self.method not in self.VALID_BASE | self.VALID_META:
            raise ValueError(f"method must be one of: {self.VALID_BASE | self.VALID_META}")

        if self.method in self.VALID_META:
            invalid = set(base_methods) - self.VALID_BASE
            if invalid:
                raise ValueError(f"base_methods contains invalid methods: {invalid}")

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute logit: log(p / (1-p)), with clipping for numerical stability."""
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def _log(self, message: str) -> None:
        """Print message if verbose=True."""
        if self.verbose:
            print(message)

    def fit(
        self,
        y_cal: np.ndarray,
        p_cal: np.ndarray,
        y_dev: np.ndarray | None = None,
        p_dev: np.ndarray | None = None,
    ) -> ProbabilityCalibrator:
        """Fit calibration mapping on calibration set."""
        y_cal = np.asarray(y_cal, dtype=int).ravel()
        p_cal = np.asarray(p_cal, dtype=float).ravel()

        if y_cal.shape[0] != p_cal.shape[0]:
            raise ValueError("y_cal and p_cal must have same length")

        if self.method == "stacking" and (y_dev is None or p_dev is None):
            split_idx = int(len(y_cal) * 0.7)
            if split_idx <= 0 or split_idx >= len(y_cal):
                raise ValueError("Stacking requires enough samples for an internal 70/30 split.")
            y_dev = y_cal[split_idx:]
            p_dev = p_cal[split_idx:]
            y_cal = y_cal[:split_idx]
            p_cal = p_cal[:split_idx]

        if self.method in self.VALID_META:
            return self._fit_meta(y_cal, p_cal, y_dev, p_dev)
        return self._fit_base(y_cal, p_cal)

    def _fit_base(self, y_cal: np.ndarray, p_cal: np.ndarray) -> ProbabilityCalibrator:
        """Fit base calibration method (platt, beta, or temperature)."""
        logit_p = self._logit(p_cal)
        lr_kwargs = {"C": 1e6, "solver": "lbfgs", "random_state": 42, "max_iter": 1000}

        if self.method == "platt":
            self.model = LogisticRegression(**lr_kwargs)
            self.model.fit(logit_p.reshape(-1, 1), y_cal)

        elif self.method == "beta":
            X = np.column_stack([logit_p, self._logit(1 - p_cal)])
            self.model = LogisticRegression(**lr_kwargs)
            self.model.fit(X, y_cal)

        elif self.method == "temperature":

            def nll(tlog):
                T = np.exp(tlog[0])
                p = 1.0 / (1.0 + np.exp(-logit_p / T))
                return log_loss(y_cal, np.clip(p, 1e-12, 1 - 1e-12))

            res = minimize(nll, x0=[0.0], method="L-BFGS-B", bounds=[(-5, 5)])
            self.T = np.exp(res.x[0])
            self.model = "temperature"

        self._fitted = True
        return self

    def _fit_meta(
        self,
        y_cal: np.ndarray,
        p_cal: np.ndarray,
        y_dev: np.ndarray | None,
        p_dev: np.ndarray | None,
    ) -> ProbabilityCalibrator:
        """Fit meta-calibration method (ensemble or stacking)."""
        if self.method == "stacking":
            if y_dev is None or p_dev is None:
                raise ValueError("Stacking requires dev data (y_dev, p_dev).")
            y_dev = np.asarray(y_dev, dtype=int).ravel()
            p_dev = np.asarray(p_dev, dtype=float).ravel()

        self._log(f"\nFitting {self.method} calibrator...")
        self._log(f"Stage 1: Training {len(self.base_methods)} base calibrators ({len(y_cal)} samples)")

        for base_method in self.base_methods:
            cal = ProbabilityCalibrator(method=base_method, verbose=self.verbose)
            cal.fit(y_cal, p_cal)
            self.base_calibrators[base_method] = cal
            self._log(f"  {base_method} fitted")

        if self.method == "stacking":
            self._log(f"\nStage 2: Training meta-model on dev data ({len(y_dev)} samples)")
            meta_features = [self._logit(p_dev)]
            for cal in self.base_calibrators.values():
                meta_features.append(self._logit(cal.transform(p_dev)))

            self.meta_model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42, max_iter=1000)
            self.meta_model.fit(np.column_stack(meta_features), y_dev)

            feature_names = ["original"] + list(self.base_calibrators.keys())
            self._log("\nLearned meta-model weights:")
            for name, w in zip(feature_names, self.meta_model.coef_[0]):
                self._log(f"  {name:<12}: {w:+.4f}")
            self._log(f"  Intercept: {self.meta_model.intercept_[0]:+.4f}")

        self._log(f"{self.method.capitalize()} calibrator fitted successfully")
        self._fitted = True
        return self

    def transform(self, p_infer: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to new probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before transform")

        p = np.asarray(p_infer, dtype=float).ravel()

        if self.method == "ensemble":
            preds = [cal.transform(p) for cal in self.base_calibrators.values()]
            return np.mean(preds, axis=0)

        if self.method == "stacking":
            meta_features = [self._logit(p)]
            for cal in self.base_calibrators.values():
                meta_features.append(self._logit(cal.transform(p)))
            return self.meta_model.predict_proba(np.column_stack(meta_features))[:, 1]

        # Base methods
        logit_p = self._logit(p)
        if self.method == "platt":
            return self.model.predict_proba(logit_p.reshape(-1, 1))[:, 1]

        if self.method == "beta":
            X = np.column_stack([logit_p, self._logit(1 - p)])
            return self.model.predict_proba(X)[:, 1]

        if self.method == "temperature":
            return 1.0 / (1.0 + np.exp(-logit_p / self.T))

    @staticmethod
    def calculate_prob_array(X: pd.DataFrame, model) -> np.ndarray:
        """Calculate model probabilities as float32 numpy array."""
        feature_cols = [c for c in X.columns if c not in ["EntryTime", "ExitTime"]]
        if not feature_cols:
            raise ValueError("No feature columns available in X")
        return model.predict_proba(X[feature_cols])[:, 1].astype(np.float32)

    @staticmethod
    def attach_gate_and_bet_columns(
        df: pd.DataFrame,
        probs: pd.Series | np.ndarray,
        entry_times: pd.Series | pd.Index | np.ndarray,
        timezone: str = "Europe/Berlin",
        gate_col: str = "gate_col",
        bet_col: str = "bet_col",
        fill_value: float = 0.0,
        time: str = "9:00",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Attach calibrated probability gate and Kelly-like bet size columns.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC data to attach columns to.
        probs : array-like
            Calibrated probabilities at entry times.
        entry_times : array-like
            Timestamps corresponding to probability values.
        timezone : str
            Timezone for time snapping (e.g., 'Europe/Berlin').
        gate_col, bet_col : str
            Column names for gate (probability) and bet size.
        fill_value : float
            Fill value for unaligned bars.
        time : str
            Time-of-day snap (HH:MM format).

        Returns
        -------
        tuple
            (df_out, gate_series, bet_series)

        """
        prob_values = np.asarray(probs, dtype=np.float32).ravel()
        entry_index = pd.Index(np.asarray(entry_times))

        if prob_values.shape[0] != entry_index.shape[0]:
            raise ValueError(f"len(probs)={prob_values.shape[0]} != len(entry_times)={entry_index.shape[0]}")

        gate_series = pd.Series(prob_values, index=entry_index, name=gate_col, dtype=np.float32)
        bet_series = probability_to_bet_size(gate_series).astype(np.float32).rename(bet_col)

        # Parse time-of-day snap
        parts = time.strip().split(":")
        snap_td = pd.Timedelta(hours=int(parts[0]), minutes=int(parts[1]))

        # Convert entry times to aligned datetime index
        dt = pd.to_datetime(entry_index, utc=True)
        dt_idx = dt.tz_convert(timezone).normalize() + snap_td

        # Create aligned series and reindex to df
        map_gate = pd.Series(gate_series.values, index=dt_idx, name=gate_col)
        map_bet = pd.Series(bet_series.values, index=dt_idx, name=bet_col)

        gate_aligned = map_gate.reindex(df.index).fillna(fill_value).to_numpy(dtype=np.float32)
        bet_aligned = map_bet.reindex(df.index).fillna(fill_value).to_numpy(dtype=np.float32)

        df_out = df.copy()
        df_out[gate_col] = gate_aligned
        df_out[bet_col] = bet_aligned

        return df_out, gate_series, bet_series

    @classmethod
    def build_calibrated_dataframe(
        cls,
        df_cal: pd.DataFrame,
        X_cal: pd.DataFrame,
        model,
        calibrator: ProbabilityCalibrator | None = None,
        gate_col: str = "gate_col",
        bet_col: str = "bet_col",
        fill_value: float = 0.0,
        entry_time_col: str = "EntryTime",
        timezone: str = "Europe/Berlin",
        time: str = "9:00",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Create calibrated gate/bet columns from model probabilities.

        If calibrator is provided, probabilities are transformed with it.
        """
        if entry_time_col not in X_cal.columns:
            raise ValueError(f"X_cal must contain '{entry_time_col}' for alignment")

        p_raw = cls.calculate_prob_array(X_cal, model)
        p_calibrated = calibrator.transform(p_raw) if calibrator is not None else p_raw

        return cls.attach_gate_and_bet_columns(
            df=df_cal,
            probs=p_calibrated,
            entry_times=X_cal[entry_time_col],
            timezone=timezone,
            gate_col=gate_col,
            bet_col=bet_col,
            fill_value=fill_value,
            time=time,
        )

    @classmethod
    def fit_from_calibration_data(
        cls,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        model,
        method: str = "platt",
        save_path: str | None = None,
    ) -> ProbabilityCalibrator:
        """Fit a calibrator from features and labels, optionally save to disk."""
        y_cal = np.asarray(y_cal, dtype=int).ravel()
        p_cal = cls.calculate_prob_array(X_cal, model)

        calibrator = cls(method=method)
        calibrator.fit(y_cal, p_cal)

        if save_path:
            import pickle

            with open(save_path, "wb") as f:
                pickle.dump(calibrator, f)
            print(f"Calibrator saved to {save_path}")

        return calibrator

    @classmethod
    def select_best_calibration(
        cls,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        model,
        methods: tuple = ("platt", "beta", "temperature", "ensemble", "stacking"),
    ) -> tuple[list, str]:
        """Select best calibration method by calibration-set Brier score.

        Returns
        -------
        tuple
            (scores_list, best_method_name)

        """
        y_cal = np.asarray(y_cal, dtype=int).ravel()
        p_cal = cls.calculate_prob_array(X_cal, model)

        def score_row(name, p_vals):
            brier = brier_score_loss(y_cal, p_vals)
            ll = log_loss(y_cal, np.clip(p_vals, 1e-12, 1 - 1e-12))
            return (name, brier, ll)

        scores = [score_row("original", p_cal)]

        for method in methods:
            calibrator = cls(method=method)
            calibrator.fit(y_cal, p_cal)
            scores.append(score_row(method, calibrator.transform(p_cal)))

        best = min(scores, key=lambda x: x[1])[0]
        return scores, best

    @staticmethod
    def print_calibration_scores(scores: list) -> None:
        """Print calibration scores table (method, Brier score, log loss)."""
        print("=" * 38)
        print("Calibration Scores")
        print("-" * 38)
        print(f"{'Method':<12} {'Cal Brier':>12} {'Cal LL':>12}")
        print("-" * 38)
        for method, brier, ll in scores:
            print(f"{method:<12} {brier:>12.6f} {ll:>12.6f}")
        print("=" * 38)

    @classmethod
    def full_calibration_workflow(
        cls,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        model,
        methods: tuple = ("platt", "beta", "temperature", "ensemble", "stacking"),
    ) -> tuple[np.ndarray, str, list]:
        """Complete calibration workflow: select best method, calibrate, return probs."""
        y_cal = np.asarray(y_cal, dtype=int).ravel()
        scores, best = cls.select_best_calibration(X_cal, y_cal, model, methods)

        print(f"Best method (by calibration Brier): {best}")
        cls.print_calibration_scores(scores)

        if best == "original":
            return cls.calculate_prob_array(X_cal, model), best, scores

        calibrator = cls(method=best)
        p_cal = cls.calculate_prob_array(X_cal, model)
        calibrator.fit(y_cal, p_cal)
        return calibrator.transform(p_cal), best, scores

    @classmethod
    def fit_production_calibrator(
        cls,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        model,
        method: str = "platt",
        save_path: str | None = None,
    ) -> ProbabilityCalibrator:
        """Fit production calibrator from calibration features and labels."""
        print(f"\n{'=' * 60}\nFITTING PRODUCTION CALIBRATOR\n{'=' * 60}")
        return cls.fit_from_calibration_data(X_cal, y_cal, model, method=method, save_path=save_path)

    def save_state(self, output_dir: str = "models") -> None:
        """Save calibrator state to directory (cross-environment compatible)."""
        import json
        import os

        import joblib

        os.makedirs(output_dir, exist_ok=True)

        state = {
            "method": self.method,
            "base_methods": list(self.base_methods) if self.base_methods else None,
            "fitted": self._fitted,
            "fitted_models": {},
        }

        if self.method in self.VALID_META:
            for name, cal in self.base_calibrators.items():
                if cal.model is not None and cal.method != "temperature":
                    joblib.dump(
                        cal.model,
                        os.path.join(output_dir, f"{name}_model.joblib"),
                    )
                if cal.T is not None:
                    state["fitted_models"][name] = {"T": cal.T}
            if self.method == "stacking" and self.meta_model:
                joblib.dump(
                    self.meta_model,
                    os.path.join(output_dir, "stacking_meta_model.joblib"),
                )
        else:
            if self.model is not None and self.method != "temperature":
                joblib.dump(
                    self.model,
                    os.path.join(output_dir, f"{self.method}_model.joblib"),
                )
            if self.T is not None:
                state["T"] = self.T

        with open(os.path.join(output_dir, "calibrator_state.json"), "w") as f:
            json.dump(state, f, indent=2)
        print(f"Calibrator saved to {output_dir}/")

    @classmethod
    def load_state(cls, input_dir: str = "models") -> ProbabilityCalibrator:
        """Load calibrator from state directory."""
        import json
        import os

        import joblib

        with open(os.path.join(input_dir, "calibrator_state.json")) as f:
            state = json.load(f)

        calibrator = cls(
            method=state["method"],
            base_methods=tuple(state["base_methods"] or ()),
        )
        calibrator._fitted = state.get("fitted", True)

        if state["method"] in cls.VALID_META:
            for name in state["base_methods"]:
                base = cls(method=name)
                model_path = os.path.join(input_dir, f"{name}_model.joblib")
                if os.path.exists(model_path):
                    base.model = joblib.load(model_path)

                fitted_models = state.get("fitted_models", {})
                if name in fitted_models and "T" in fitted_models[name]:
                    base.T = fitted_models[name]["T"]
                    base.model = "temperature"

                base._fitted = True
                calibrator.base_calibrators[name] = base

            if state["method"] == "stacking":
                meta_path = os.path.join(input_dir, "stacking_meta_model.joblib")
                if os.path.exists(meta_path):
                    calibrator.meta_model = joblib.load(meta_path)
        else:
            model_path = os.path.join(input_dir, f"{state['method']}_model.joblib")
            if os.path.exists(model_path):
                calibrator.model = joblib.load(model_path)
            if "T" in state:
                calibrator.T = state["T"]
                calibrator.model = "temperature"

        print(f"Calibrator loaded from {input_dir}/")
        return calibrator
