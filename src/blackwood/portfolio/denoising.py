from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.covariance import LedoitWolf

_EPS = 1e-12
_MP_MIN_ASSETS = 10
_MP_FALLBACK_Q = 0.8
_AUTO_FALLBACK_Q = 0.5


def denoise_covariance(
    cov_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "marcenko_pastur",
) -> pd.DataFrame:
    """
    Denoise a daily covariance matrix.

    Supported methods:
      - "shrinkage": Ledoit-Wolf (2003) linear shrinkage
      - "nonlinear": approximate eigenvalue shrinkage (oracle-style; not a full LW-2017 impl)
      - "marcenko_pastur": MP eigenvalue filtering on correlation matrix
      - "auto": chooses between shrinkage and MP based on N, T, q=N/T

    Notes:
      - This function returns a daily covariance matrix.
      - Annualization should be done once by the caller (e.g., * 252).

    """
    method = (method or "marcenko_pastur").lower()

    n_assets = cov_matrix.shape[0]
    n_obs = len(returns)
    q = (n_assets / n_obs) if n_obs > 0 else np.inf
    index = cov_matrix.index

    if method == "shrinkage":
        return _denoise_shrinkage(returns, index)

    if method == "nonlinear":
        return _denoise_nonlinear(returns, index)

    if method == "marcenko_pastur":
        if _use_shrinkage_for_mp(n_assets, q):
            return _denoise_shrinkage(returns, index)
        return _denoise_marcenko_pastur(cov_matrix, n_assets, q)

    if method == "auto":
        if _use_shrinkage_for_auto(n_assets, q):
            return _denoise_shrinkage(returns, index)
        return _denoise_marcenko_pastur(cov_matrix, n_assets, q)

    raise ValueError(f"Unknown denoising method: {method!r}")


def _use_shrinkage_for_mp(n_assets: int, q: float) -> bool:
    """Return whether MP should fallback to shrinkage."""
    return (q > _MP_FALLBACK_Q) or (n_assets < _MP_MIN_ASSETS)


def _use_shrinkage_for_auto(n_assets: int, q: float) -> bool:
    """Return whether auto mode should choose shrinkage."""
    return (n_assets < _MP_MIN_ASSETS) or (q > _AUTO_FALLBACK_Q)


def _clean_returns(returns: pd.DataFrame) -> np.ndarray:
    """Replace inf values and resolve missing rows for denoising estimators."""
    r = returns.replace([np.inf, -np.inf], np.nan)

    # Prefer dropping NaNs to avoid biasing correlations toward 0.
    r_drop = r.dropna(how="any")
    r = r_drop if len(r_drop) >= max(20, int(0.5 * len(r))) else r.fillna(0.0)

    return r.to_numpy(dtype=float)


def _denoise_shrinkage(returns: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Apply Ledoit-Wolf (2003) linear shrinkage."""
    X = _clean_returns(returns)
    lw = LedoitWolf()
    lw.fit(X)
    return pd.DataFrame(lw.covariance_, index=index, columns=index)


def _denoise_nonlinear(returns: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """
    Approximate nonlinear eigenvalue shrinkage.

    This is a pragmatic eigenvalue shrink procedure to stabilize S; it is NOT a full
    Ledoit-Wolf (2017) nonlinear shrinkage implementation.
    """
    X = _clean_returns(returns)
    t_obs, n_assets = X.shape
    if t_obs <= 2 or n_assets == 0:
        return pd.DataFrame(np.eye(n_assets), index=index, columns=index)

    # Sample covariance (daily)
    sample_cov = np.cov(X.T, bias=False)

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(sample_cov)
    evals = np.maximum(evals, 1e-10)

    # Compute simple repulsion term d_i = sum_{j≠i} 1/(λ_i - λ_j)^2
    diff = evals[:, None] - evals[None, :]
    np.fill_diagonal(diff, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sum(1.0 / (diff**2), axis=1) - 1.0
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    shrink = 1.0 + ((t_obs - n_assets - 1.0) / max(t_obs, 1.0)) * d
    shrink = np.clip(shrink, 0.1, 10.0)
    evals_shrunk = evals / shrink

    cov = evecs @ np.diag(evals_shrunk) @ evecs.T
    cov = _shift_to_psd(cov, tolerance=1e-10, jitter=1e-8)

    return pd.DataFrame(cov, index=index, columns=index)


def _denoise_marcenko_pastur(
    cov_matrix: pd.DataFrame,
    n_assets: int,
    q: float,
) -> pd.DataFrame:
    """Apply Marcenko-Pastur eigenvalue filtering on the correlation matrix."""
    cov = cov_matrix.to_numpy(dtype=float)

    vols = np.sqrt(np.maximum(np.diag(cov), _EPS))
    inv_vols = 1.0 / np.maximum(vols, _EPS)
    D_inv = np.diag(inv_vols)

    corr = D_inv @ cov @ D_inv
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    evals, evecs = np.linalg.eigh(corr)
    evals = np.maximum(evals, 0.0)

    q_fit, sigma_sq = _fit_marcenko_pastur(evals, q)

    lambda_plus = sigma_sq * (1.0 + np.sqrt(q_fit)) ** 2

    signal = evals > lambda_plus
    if signal.sum() == 0:
        n_keep = max(1, n_assets // 10)
        top_idx = np.argsort(evals)[-n_keep:]
        signal = np.zeros(n_assets, dtype=bool)
        signal[top_idx] = True

    noise_evals = evals[~signal]
    lam_shrink = float(np.mean(noise_evals)) if noise_evals.size else 1.0
    lam_shrink = max(lam_shrink, 1e-6)

    evals_clean = np.where(signal, evals, lam_shrink)

    corr_d = evecs @ np.diag(evals_clean) @ evecs.T
    np.fill_diagonal(corr_d, 1.0)
    corr_d = _shift_to_psd(corr_d, tolerance=1e-8, jitter=1e-6)

    corr_d = np.clip(corr_d, -1.0, 1.0)

    D = np.diag(vols)
    cov_d = D @ corr_d @ D
    return pd.DataFrame(cov_d, index=cov_matrix.index, columns=cov_matrix.columns)


def _shift_to_psd(matrix: np.ndarray, tolerance: float, jitter: float) -> np.ndarray:
    """Shift a symmetric matrix to PSD when eigenvalues are slightly negative."""
    min_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if min_eigenvalue < -tolerance:
        return matrix + np.eye(matrix.shape[0]) * (abs(min_eigenvalue) + jitter)
    return matrix


def _fit_marcenko_pastur(evals: np.ndarray, q_init: float) -> tuple[float, float]:
    """Fit MP parameters (q, sigma^2) by matching KDE to MP density."""
    evals = np.asarray(evals, dtype=float)
    n = evals.size

    if n < 5:
        return float(q_init), float(np.var(evals))

    # Trim to central 80% to reduce outlier impact
    lo = max(0, int(0.1 * n))
    hi = min(n, int(0.9 * n))
    x = np.sort(evals)[lo:hi]
    x = x[x > 1e-10]
    if x.size < 5:
        return float(q_init), float(np.var(evals))

    kde = gaussian_kde(x)

    def mp_pdf(lam: np.ndarray, q: float, sigma_sq: float) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        lam_minus = sigma_sq * (1.0 - np.sqrt(q)) ** 2
        lam_plus = sigma_sq * (1.0 + np.sqrt(q)) ** 2

        out = np.zeros_like(lam)
        mask = (lam >= lam_minus) & (lam <= lam_plus) & (lam > 1e-10)
        if not np.any(mask):
            return out

        lv = lam[mask]
        rad = (lam_plus - lv) * (lv - lam_minus)
        rad = np.maximum(rad, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[mask] = (q / (2.0 * np.pi * sigma_sq * lv)) * np.sqrt(rad)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def objective(params: np.ndarray) -> float:
        q, sigma_sq = params
        grid = np.linspace(x.min(), x.max(), 150)
        emp = kde(grid)
        theo = mp_pdf(grid, q, sigma_sq)
        return float(np.mean((emp - theo) ** 2))

    x0 = np.array([float(q_init), float(np.var(x))], dtype=float)
    bounds = [(0.01, 10.0), (0.01, float(np.max(evals) + 1e-6))]

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 150},
    )
    q_fit, sigma_sq_fit = res.x
    return float(q_fit), float(sigma_sq_fit)
