from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


def _covariance_frame(covariance: np.ndarray, columns: pd.Index) -> pd.DataFrame:
    """Build a covariance DataFrame aligned to the given columns."""
    return pd.DataFrame(covariance, index=columns, columns=columns)


def _eps_diagonal_covariance(
    n_assets: int,
    eps: float,
    columns: pd.Index,
) -> pd.DataFrame:
    """Build an epsilon-scaled identity covariance matrix aligned to columns."""
    return _covariance_frame(np.eye(n_assets) * eps, columns)


class RiskModel(ABC):
    @abstractmethod
    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return a DAILY covariance matrix aligned to returns.columns."""
        raise NotImplementedError


@dataclass
class SampleCovariance(RiskModel):
    ddof: int = 1

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        r = returns.fillna(0.0)
        return r.cov(ddof=self.ddof)


@dataclass
class LegacySampleCovariance(RiskModel):
    """
    Legacy sample covariance estimator matching historical `returns.cov()` usage.

    Uses pairwise complete observations (pandas default) instead of filling NaNs.
    """

    ddof: int = 1
    min_periods: int | None = None

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.cov(ddof=self.ddof, min_periods=self.min_periods)


@dataclass
class GARCH11RiskModel(RiskModel):
    horizon: int = 1
    corr_source: str = "sample"  # "sample" only here
    eps: float = 1e-12

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        returns_matrix, columns = self._prepare_returns_matrix(returns)
        n_assets = returns_matrix.shape[1]

        forecast_variances = np.asarray(
            [
                self._fit_and_forecast_variance(
                    returns_matrix[:, asset_idx],
                    h=self.horizon,
                )
                for asset_idx in range(n_assets)
            ],
            dtype=float,
        )

        volatilities = np.sqrt(np.maximum(forecast_variances, self.eps))

        correlation = self._estimate_correlation(returns_matrix)
        correlation = self._project_to_corr_psd(correlation)

        covariance = np.outer(volatilities, volatilities) * correlation
        return _covariance_frame(covariance, columns)

    def _prepare_returns_matrix(
        self,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.Index]:
        """Fill missing values and demean returns by asset."""
        filled_returns = returns.fillna(0.0)
        returns_matrix = filled_returns.to_numpy(dtype=float)
        returns_matrix = returns_matrix - returns_matrix.mean(axis=0, keepdims=True)
        return returns_matrix, filled_returns.columns

    def _estimate_correlation(self, matrix: np.ndarray) -> np.ndarray:
        """Estimate a correlation matrix from demeaned returns."""
        if self.corr_source == "sample":
            return np.corrcoef(matrix, rowvar=False)
        raise ValueError(f"Unknown corr_source={self.corr_source}")

    def _garch_filter(
        self,
        omega: float,
        alpha: float,
        beta: float,
        x2: np.ndarray,
        var0: float,
    ) -> np.ndarray:
        """Run GARCH(1,1) variance recursion with lfilter."""
        t_obs = x2.size
        s = np.empty(t_obs, dtype=float)
        s[0] = var0

        if t_obs == 1:
            return s

        u = omega + alpha * x2[:-1]  # length T-1, corresponds to x2[t-1]
        u = np.float64(u)
        u[0] += beta * var0  # incorporate initial condition

        s[1:] = lfilter([1.0], [1.0, -beta], u)
        return s

    def _fit_and_forecast_variance(self, x: np.ndarray, h: int) -> float:
        """Fit GARCH(1,1) by MLE and forecast h-step variance."""
        penalty_loss = 1e100
        stationarity_cap = 0.999

        x = np.asarray(x, dtype=float)
        n_obs = x.size
        if n_obs < 20:
            return float(np.var(x, ddof=1))

        x2 = x**2
        var0 = float(np.var(x, ddof=1))
        var0 = max(var0, self.eps)

        def nll(params: np.ndarray) -> float:
            omega, alpha, beta = params

            # Hard constraints (return huge loss)
            if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= stationarity_cap:
                return penalty_loss

            s = self._garch_filter(omega, alpha, beta, x2, var0)

            if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
                return penalty_loss

            # Gaussian negative log-likelihood (up to constant)
            return 0.5 * float(np.sum(np.log(s) + x2 / s))

        x0 = np.array([0.01 * var0, 0.05, 0.90], dtype=float)
        bounds = [(self.eps, 10.0 * var0), (0.0, 1.0), (0.0, 1.0)]

        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            return var0

        omega, alpha, beta = (float(res.x[0]), float(res.x[1]), float(res.x[2]))

        # One final filter pass to get s_T without Python loops
        s = self._garch_filter(omega, alpha, beta, x2, var0)
        s_T = float(s[-1])

        s_1 = omega + alpha * float(x2[-1]) + beta * s_T
        s_1 = max(s_1, self.eps)

        ab = alpha + beta
        if h <= 1:
            return s_1

        s_bar = omega / max(1.0 - ab, self.eps)
        s_h = s_bar + (ab ** (h - 1)) * (s_1 - s_bar)
        return float(max(s_h, self.eps))

    def _project_to_corr_psd(self, R: np.ndarray) -> np.ndarray:
        """Project a matrix to the nearest clipped PSD correlation."""
        R = np.asarray(R, dtype=float)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(R, 1.0)

        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-8, None)
        R_psd = V @ np.diag(w) @ V.T

        d = np.sqrt(np.maximum(np.diag(R_psd), 1e-12))
        R_corr = R_psd / np.outer(d, d)
        np.fill_diagonal(R_corr, 1.0)
        return np.clip(R_corr, -1.0, 1.0)


@dataclass
class EWMARiskModel(RiskModel):
    """
    EWMA (RiskMetrics-style) covariance:
        S_t = (1-λ) * r_{t-1} r_{t-1}^T + λ * S_{t-1}

    Vectorized closed-form for the *latest* estimate (no look-ahead):
        S_T = λ^m S_0 + (1-λ) * Σ_{i=0}^{m-1} λ^{m-1-i} x_i x_i^T
    where x_i = r_i (demeaned) and m = number of updates = T-1.

    - Uses only returns up to r_{T-2} for the latest estimate.
    - No Python loop; uses matrix multiplication.
    """

    lam: float = 0.94
    ddof: int = 0
    eps: float = 1e-12
    init: str = "sample"  # "sample" or "diag"

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        filled_returns = returns.fillna(0.0)
        columns = filled_returns.columns
        returns_matrix = filled_returns.to_numpy(dtype=float)

        t_obs, n_assets = returns_matrix.shape
        if t_obs < 2:
            return _eps_diagonal_covariance(n_assets, self.eps, columns)

        lam = float(self.lam)
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1)")

        # No look-ahead: latest EWMA uses r[0..T-2], so build X from r[:-1]
        X = returns_matrix[:-1, :]
        m = X.shape[0]  # number of updates = T-1
        if m == 0:
            return _eps_diagonal_covariance(n_assets, self.eps, columns)

        # Demean using only the information actually used in the estimate (X)
        X = X - X.mean(axis=0, keepdims=True)

        S0 = self._initialize_covariance(X, n_assets)

        # Weights: w_i = (1-lam) * lam^(m-1-i), i=0..m-1
        # Recent rows get larger weight.
        exponents = (m - 1) - np.arange(m, dtype=float)
        w = (1.0 - lam) * np.power(lam, exponents)

        # Compute Σ w_i x_i x_i^T via (X * sqrt(w))^T (X * sqrt(w))
        sw = np.sqrt(w, dtype=float)
        Xw = X * sw[:, None]
        cov_est = Xw.T @ Xw

        # Add the decayed initial state: lam^m * S0
        cov_est += (lam**m) * S0

        # Numerical safety
        cov_est = np.nan_to_num(cov_est, nan=0.0, posinf=0.0, neginf=0.0)
        cov_est = 0.5 * (cov_est + cov_est.T)

        d = np.diag(cov_est).copy()
        d = np.maximum(d, self.eps)
        np.fill_diagonal(cov_est, d)

        return _covariance_frame(cov_est, columns)

    def _initialize_covariance(self, X: np.ndarray, n_assets: int) -> np.ndarray:
        if self.init == "sample":
            cov0 = np.cov(X, rowvar=False, ddof=self.ddof)
            if not np.all(np.isfinite(cov0)):
                cov0 = np.eye(n_assets) * self.eps
            return np.asarray(cov0, dtype=float)

        if self.init == "diag":
            diag_var = np.var(X, axis=0, ddof=self.ddof)
            diag_var = np.where(np.isfinite(diag_var), diag_var, self.eps)
            return np.diag(np.maximum(diag_var, self.eps))

        raise ValueError(f"Unknown init={self.init}")
