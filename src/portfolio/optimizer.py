import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import norm, probplot, t
from sklearn.metrics import silhouette_score
from src.portfolio.denoising import denoise_covariance
from src.portfolio.risk_models import GARCH11RiskModel, RiskModel


def _solver_candidates(primary: str, *fallbacks: str) -> list[str]:
    """Return unique solver candidates preserving order."""
    ordered = [primary, *fallbacks]
    seen: set[str] = set()
    candidates: list[str] = []
    for solver in ordered:
        normalized = str(solver).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)
    return candidates


def _solver_settings(solver: str) -> dict[str, Any]:
    """Return conservative settings to reduce inaccurate solutions."""
    name = str(solver).upper()
    if name == "OSQP":
        return {
            "eps_abs": 1e-8,
            "eps_rel": 1e-8,
            "max_iter": 200_000,
            "polish": True,
        }
    if name == "SCS":
        return {
            "eps": 1e-6,
            "max_iters": 20_000,
        }
    if name == "CLARABEL":
        return {
            "max_iter": 10_000,
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
        }
    return {}


def _solve_with_fallback(
    problem: cp.Problem,
    solvers: list[str],
    *,
    warm_start: bool = False,
    ignore_dpp: bool = False,
) -> str | None:
    """
    Solve with a solver cascade.

    Returns the final status, preferring an OPTIMAL solution.
    """
    last_status: str | None = None

    for solver in solvers:
        solve_kwargs: dict[str, Any] = {"solver": solver, "warm_start": warm_start}
        if ignore_dpp:
            solve_kwargs["ignore_dpp"] = True
        solve_kwargs.update(_solver_settings(solver))

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Solution may be inaccurate.*",
                    category=UserWarning,
                )
                problem.solve(**solve_kwargs)
        except Exception:
            continue

        last_status = problem.status
        if last_status == cp.OPTIMAL:
            return last_status

    return last_status


class OptimizationStrategy(ABC):
    """
    Abstract base class for portfolio optimization strategies with covariance denoising.
    """

    def __init__(
        self,
        max_weight: float = 0.9,
        use_denoising: bool = True,
        denoising_method: str = "marcenko_pastur",
        risk_model: RiskModel | None = None,
    ):
        self.max_weight = max_weight
        self.use_denoising = use_denoising
        self.denoising_method = denoising_method
        self.risk_model = risk_model or GARCH11RiskModel(horizon=21)

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Risk model produces DAILY covariance
        cov = self.risk_model.covariance(returns)

        # Denoisers return DAILY covariance (annualize once here)
        if self.use_denoising:
            cov = denoise_covariance(cov, returns, method=self.denoising_method)

        return cov * 252

    @abstractmethod
    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class TangencyStrategy(OptimizationStrategy):
    """
    Maximum Sharpe Ratio Portfolio (Tangency Portfolio), long-only with per-asset cap.

    Performance: caches the CVXPY problem and updates Parameters each call to avoid
    repeated problem construction overhead in backtests.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        max_weight: float = 0.5,
        use_denoising: bool = True,
        solver: str = "OSQP",
        risk_model: RiskModel | None = None,
        eps: float = 1e-8,
        annualization: int = 252,
        warm_start: bool = True,
    ):
        super().__init__(
            max_weight=max_weight,
            use_denoising=use_denoising,
            risk_model=risk_model,
        )
        self.risk_free_rate = float(risk_free_rate)
        self.solver = solver
        self.eps = float(eps)
        self.annualization = int(annualization)
        self.warm_start = bool(warm_start)

        # Cached CVXPY objects (rebuilt if asset count changes)
        self._n_assets: int | None = None
        self._w: cp.Variable | None = None
        self._risk_factor: cp.Parameter | None = None
        self._mu_excess: cp.Parameter | None = None
        self._problem: cp.Problem | None = None

    @staticmethod
    def _inv_vol_fallback(cov: np.ndarray, index: pd.Index, eps: float) -> pd.Series:
        """Return inverse-volatility weights for fallback cases."""
        vol = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        inv = 1.0 / (vol + eps)
        w = inv / (inv.sum() + eps)
        return pd.Series(w, index=index)

    def _build_problem(self, n_assets: int) -> None:
        """Build and cache the CVXPY problem for a fixed asset count."""
        w = cp.Variable(n_assets)
        risk_factor = cp.Parameter((n_assets, n_assets))
        mu_excess = cp.Parameter(n_assets)

        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(risk_factor @ w)),
            [
                mu_excess @ w == 1.0,
                w >= 0.0,
                w <= self.max_weight,
            ],
        )

        self._n_assets = n_assets
        self._w = w
        self._risk_factor = risk_factor
        self._mu_excess = mu_excess
        self._problem = problem

    def _ensure_problem(
        self,
        n_assets: int,
    ) -> tuple[cp.Variable, cp.Parameter, cp.Parameter, cp.Problem]:
        """Return cached optimization objects, rebuilding if shape changed."""
        if self._problem is None or self._n_assets != n_assets:
            self._build_problem(n_assets)

        if self._w is None or self._risk_factor is None or self._mu_excess is None or self._problem is None:
            raise RuntimeError("TangencyStrategy optimization cache is not initialized")
        return self._w, self._risk_factor, self._mu_excess, self._problem

    @staticmethod
    def _covariance_factor(cov: np.ndarray) -> np.ndarray:
        """Return a matrix L such that L.T @ L approximates covariance."""
        eigvals, eigvecs = np.linalg.eigh(cov)
        clipped = np.clip(eigvals, 0.0, np.inf)
        sqrt_diag = np.sqrt(clipped)
        return (eigvecs * sqrt_diag[np.newaxis, :]).T

    def _expected_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate annualized expected returns using geometric mean."""
        log1p_r = np.log1p(returns.fillna(0.0).to_numpy(dtype=float))
        return np.expm1(log1p_r.mean(axis=0) * self.annualization)

    def _normalize_weights(
        self,
        raw_weights: np.ndarray,
        cov: np.ndarray,
        idx: pd.Index,
    ) -> pd.Series:
        """Clip and normalize solved weights; fallback if invalid."""
        clipped = np.clip(raw_weights, 0.0, self.max_weight)
        total = clipped.sum()
        if (not np.isfinite(total)) or total <= self.eps:
            return self._inv_vol_fallback(cov, idx, self.eps)
        return pd.Series(clipped / total, index=idx)

    def _solve_cached_problem(
        self,
        cov: np.ndarray,
        mu_excess: np.ndarray,
    ) -> tuple[np.ndarray | None, str | None]:
        """Solve the cached CVXPY problem and return weights and status."""
        w, risk_factor, mu_excess_param, problem = self._ensure_problem(len(mu_excess))

        risk_factor.value = self._covariance_factor(cov)
        mu_excess_param.value = mu_excess

        status = _solve_with_fallback(
            problem,
            _solver_candidates(self.solver, "CLARABEL", "SCS"),
            warm_start=self.warm_start,
        )

        if w.value is None:
            return None, status
        return np.asarray(w.value, dtype=float), status

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        del kwargs
        mu = self._expected_returns(returns)
        cov_df = self.estimate_covariance(returns)
        cov = cov_df.to_numpy()
        idx = cov_df.index

        mu_excess = mu - self.risk_free_rate
        if np.all(mu_excess <= 0.0):
            return self._inv_vol_fallback(cov, idx, self.eps)

        solved_weights, status = self._solve_cached_problem(cov, mu_excess)
        if solved_weights is None or status != cp.OPTIMAL:
            return self._inv_vol_fallback(cov, idx, self.eps)

        return self._normalize_weights(solved_weights, cov, idx)


class EqualStrategy(OptimizationStrategy):
    """Equal weighted Optimization with risk aversion parameter."""

    def __init__(self, risk_model: RiskModel | None = None):
        super().__init__(risk_model=risk_model)

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)


class MVOStrategy(OptimizationStrategy):
    """Mean-Variance Optimization with risk aversion parameter."""

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = 0.9,
        use_denoising: bool = True,
        solver: str = "OSQP",
        risk_model: RiskModel | None = None,
        eps: float = 1e-10,
    ):
        super().__init__(max_weight=max_weight, use_denoising=use_denoising, risk_model=risk_model)
        self.risk_aversion = float(risk_aversion)
        self.solver = solver
        self.eps = float(eps)

        self._n_assets: int | None = None
        self._weights: cp.Variable | None = None
        self._mu: cp.Parameter | None = None
        self._risk_factor: cp.Parameter | None = None
        self._problem: cp.Problem | None = None

    def _stabilized_covariance(self, returns: pd.DataFrame, assets: pd.Index) -> np.ndarray:
        """Return covariance matrix aligned to assets with diagonal jitter."""
        cov_df = self.estimate_covariance(returns)
        aligned = cov_df.reindex(index=assets, columns=assets).fillna(0.0)
        cov = aligned.to_numpy(dtype=float, copy=True)
        cov.flat[:: cov.shape[0] + 1] += self.eps
        return cov

    @staticmethod
    def _covariance_factor(cov: np.ndarray) -> np.ndarray:
        """Return matrix L where L.T @ L approximates covariance."""
        eigvals, eigvecs = np.linalg.eigh(cov)
        clipped = np.clip(eigvals, 0.0, np.inf)
        sqrt_diag = np.sqrt(clipped)
        return (eigvecs * sqrt_diag[np.newaxis, :]).T

    def _build_problem(self, n_assets: int) -> None:
        """Build and cache CVXPY problem for a fixed number of assets."""
        weights = cp.Variable(n_assets)
        mu = cp.Parameter(n_assets)
        risk_factor = cp.Parameter((n_assets, n_assets))

        objective = cp.Maximize(
            mu @ weights - 0.5 * self.risk_aversion * cp.sum_squares(risk_factor @ weights),
        )
        constraints = [
            cp.sum(weights) == 1.0,
            weights >= 0.0,
            weights <= self.max_weight,
        ]

        self._n_assets = n_assets
        self._weights = weights
        self._mu = mu
        self._risk_factor = risk_factor
        self._problem = cp.Problem(objective, constraints)

    def _ensure_problem(self, n_assets: int) -> tuple[cp.Variable, cp.Parameter, cp.Parameter, cp.Problem]:
        """Return cached optimization objects, rebuilding when size changes."""
        if self._problem is None or self._n_assets != n_assets:
            self._build_problem(n_assets)

        if self._weights is None or self._mu is None or self._risk_factor is None or self._problem is None:
            raise RuntimeError("MVOStrategy optimization cache is not initialized")

        return self._weights, self._mu, self._risk_factor, self._problem

    def _solve_problem(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
        """Solve cached CVXPY problem and return raw weights if available."""
        weights, mu_param, risk_factor_param, problem = self._ensure_problem(len(mu))
        mu_param.value = mu
        risk_factor_param.value = self._covariance_factor(cov)

        status = _solve_with_fallback(
            problem,
            _solver_candidates(self.solver, "CLARABEL", "SCS"),
            warm_start=True,
        )
        if status != cp.OPTIMAL or weights.value is None:
            return None

        return np.asarray(weights.value, dtype=float).ravel()

    def _normalize_weights(self, weights: np.ndarray | None, n_assets: int, index: pd.Index) -> pd.Series:
        """Clip and normalize raw solver output."""
        if weights is None:
            return pd.Series(np.full(n_assets, 1.0 / n_assets), index=index)
        clipped = np.clip(weights, 0.0, self.max_weight)
        total = clipped.sum()
        if total <= 0.0:
            return pd.Series(np.full(n_assets, 1.0 / n_assets), index=index)
        return pd.Series(clipped / total, index=index)

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        del kwargs
        log_returns = np.log1p(returns.fillna(0.0))
        expected_returns = np.expm1(log_returns.mean() * 252.0)
        assets = expected_returns.index
        mu = expected_returns.to_numpy(dtype=float)
        cov = self._stabilized_covariance(returns, assets)

        raw_weights = self._solve_problem(mu, cov)
        n_assets = len(expected_returns)

        return self._normalize_weights(raw_weights, n_assets, assets)


class MinimumVarianceStrategy(OptimizationStrategy):
    """
    Global Minimum Variance Portfolio (GMV).

    minimize: w^T Σ w
    subject to: sum(w)=1, 0 ≤ w ≤ max_weight
    """

    def __init__(
        self,
        max_weight: float = 0.9,
        use_denoising: bool = True,
        solver: str = "OSQP",
        risk_model: RiskModel | None = None,
        ridge: float = 1e-10,
        warm_start: bool = True,
    ):
        super().__init__(max_weight=max_weight, use_denoising=use_denoising, risk_model=risk_model)
        self.solver = solver
        self.ridge = float(ridge)
        self.warm_start = bool(warm_start)

        # Cache: n_assets -> (Sigma_param, weights_var, problem)
        self._problem_cache: dict[int, tuple[cp.Parameter, cp.Variable, cp.Problem]] = {}

    def _get_or_build_problem(self, n_assets: int) -> tuple[cp.Parameter, cp.Variable, cp.Problem]:
        cached = self._problem_cache.get(n_assets)
        if cached is not None:
            return cached

        Sigma = cp.Parameter((n_assets, n_assets), PSD=True, name="Sigma")
        w = cp.Variable(n_assets, name="w")

        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= self.max_weight,
        ]
        problem = cp.Problem(objective, constraints)

        self._problem_cache[n_assets] = (Sigma, w, problem)
        return Sigma, w, problem

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        del kwargs

        cov_matrix = self.estimate_covariance(returns)
        covariance = np.asarray(cov_matrix.to_numpy(), dtype=float)

        n_assets = covariance.shape[0]
        if n_assets == 0:
            return pd.Series(dtype=float)

        # Numerical stabilization (keeps Σ symmetric, helps PSD)
        covariance = 0.5 * (covariance + covariance.T)
        if self.ridge > 0:
            covariance = covariance + self.ridge * np.eye(n_assets)

        Sigma, w, problem = self._get_or_build_problem(n_assets)

        Sigma.value = covariance
        status = _solve_with_fallback(
            problem,
            _solver_candidates(self.solver, "CLARABEL", "SCS"),
            warm_start=self.warm_start,
            ignore_dpp=True,
        )
        if status != cp.OPTIMAL or w.value is None:
            return pd.Series(np.full(n_assets, 1.0 / n_assets), index=cov_matrix.index)

        optimal = np.asarray(w.value, dtype=float)
        s = optimal.sum()
        if not np.isfinite(s) or s <= 0.0:
            return pd.Series(np.full(n_assets, 1.0 / n_assets), index=cov_matrix.index)

        optimal = optimal / s
        optimal = np.clip(optimal, 0.0, self.max_weight)
        clipped_sum = optimal.sum()
        if not np.isfinite(clipped_sum) or clipped_sum <= 0.0:
            return pd.Series(np.full(n_assets, 1.0 / n_assets), index=cov_matrix.index)
        optimal = optimal / clipped_sum

        return pd.Series(optimal, index=cov_matrix.index)


class NCOStrategy(OptimizationStrategy):
    """Nested Clustered Optimization with normalized weight constraints."""

    def __init__(
        self,
        n_clusters: int | None = None,
        linkage_method: str = "ward",
        risk_free_rate: float = 0.0,
        max_weight: float = 0.9,
        max_weight_intra: float | None = None,
        max_weight_inter: float | None = None,
        use_denoising: bool = False,
        solver: str = "OSQP",
        risk_model=None,
        eps: float = 1e-12,
        annualization: int = 252,
    ):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.risk_free_rate = float(risk_free_rate)
        self.max_weight_intra = float(max_weight if max_weight_intra is None else max_weight_intra)
        self.max_weight_inter = float(max_weight if max_weight_inter is None else max_weight_inter)
        self.annualization = annualization
        self.use_denoising = use_denoising
        self.solver = solver
        self.risk_model = risk_model
        self.eps = float(eps)

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute NCO portfolio weights."""
        resolved_n_clusters = kwargs.get("n_clusters", self.n_clusters)
        resolved_risk_free_rate = kwargs.get("risk_free_rate", self.risk_free_rate)

        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")
        if returns.shape[1] < 2:
            raise ValueError(f"Need at least 2 assets, got {returns.shape[1]}")

        clean_returns = returns.fillna(0.0)
        asset_names = clean_returns.columns
        returns_values = clean_returns.to_numpy(dtype=np.float64)

        expected_returns = self._annualized_expected_returns(returns_values)
        cov_values = self._estimate_covariance(returns_values)
        clusters = self._perform_clustering(cov_values, resolved_n_clusters)

        intra_weights, cluster_returns = self._optimize_intra_cluster(
            returns_values=returns_values,
            er_values=expected_returns,
            cov_values=cov_values,
            clusters=clusters,
            risk_free_rate=resolved_risk_free_rate,
        )
        inter_weights = self._optimize_inter_cluster(
            cluster_returns=cluster_returns,
            risk_free_rate=resolved_risk_free_rate,
        )

        final_weights = self._combine_weights(
            intra_weights=intra_weights,
            inter_weights=inter_weights,
            clusters=clusters,
            n_assets=len(asset_names),
        )
        return pd.Series(final_weights, index=asset_names)

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Public interface for covariance estimation (pandas compatible)."""
        return returns.cov() * self.annualization

    def _estimate_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Annualized covariance matrix from daily returns (numpy)."""
        return np.cov(returns, rowvar=False, ddof=1) * self.annualization

    def _annualized_expected_returns(self, returns: np.ndarray) -> np.ndarray:
        """Annualized expected returns using log-return aggregation."""
        log_returns = np.log1p(returns)
        mean_log = np.mean(log_returns, axis=0)
        return np.expm1(mean_log * self.annualization)

    @staticmethod
    def _correlation_distance(cov: np.ndarray) -> np.ndarray:
        """Correlation-based distance matrix: d = sqrt((1 - corr) / 2)."""
        vol = np.sqrt(np.diag(cov))
        vol = np.maximum(vol, 1e-10)
        inv_vol = 1.0 / vol

        corr = cov * np.outer(inv_vol, inv_vol)
        np.clip(corr, -1.0, 1.0, out=corr)

        dist = np.sqrt(0.5 * (1.0 - corr))
        np.clip(dist, 0.0, 1.0, out=dist)

        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)
        return dist

    def _perform_clustering(
        self,
        cov: np.ndarray,
        n_clusters: int | None,
    ) -> dict[int, np.ndarray]:
        """Hierarchical clustering on correlation distance matrix."""
        n_assets = cov.shape[0]

        dist = self._correlation_distance(cov)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)

        if n_clusters is None:
            n_clusters = self._auto_determine_clusters(link, dist, n_assets)

        labels = fcluster(link, n_clusters, criterion="maxclust")
        clusters: dict[int, np.ndarray] = {}
        for cluster_id in range(1, n_clusters + 1):
            mask = labels == cluster_id
            if np.any(mask):
                clusters[cluster_id] = np.where(mask)[0]
        return clusters

    def _auto_determine_clusters(
        self,
        link: np.ndarray,
        distance_matrix: np.ndarray,
        n_assets: int,
    ) -> int:
        """Auto-determine optimal k using silhouette score."""
        max_k = max(2, min(int(np.sqrt(n_assets)), max(2, n_assets // 3)))
        if max_k <= 2:
            return 2

        all_labels = np.column_stack(
            [fcluster(link, k, criterion="maxclust") for k in range(2, max_k + 1)],
        )

        scores = np.empty(max_k - 1, dtype=np.float64)
        for idx in range(max_k - 1):
            labels = all_labels[:, idx]
            if len(np.unique(labels)) < 2:
                scores[idx] = -1.0
                continue
            scores[idx] = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed",
            )

        best_k = np.argmax(scores) + 2
        return int(max(2, min(best_k, max(2, n_assets // 2))))

    def _optimize_intra_cluster(
        self,
        returns_values: np.ndarray,
        er_values: np.ndarray,
        cov_values: np.ndarray,
        clusters: dict[int, np.ndarray],
        risk_free_rate: float,
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Optimize weights within each cluster."""
        n_obs = returns_values.shape[0]
        cluster_ids = list(clusters.keys())

        cluster_returns = np.empty((n_obs, len(cluster_ids)), dtype=np.float64)
        intra_weights: dict[int, np.ndarray] = {}

        for idx, cluster_id in enumerate(cluster_ids):
            asset_idx = clusters[cluster_id]
            er_cluster = er_values[asset_idx]
            cov_cluster = cov_values[np.ix_(asset_idx, asset_idx)]
            returns_cluster = returns_values[:, asset_idx]

            weights = self._optimize_max_sharpe(
                mu=er_cluster,
                cov=cov_cluster,
                risk_free_rate=risk_free_rate,
                max_weight=self.max_weight_intra,
            )
            intra_weights[cluster_id] = weights
            cluster_returns[:, idx] = returns_cluster @ weights

        return intra_weights, cluster_returns

    def _optimize_inter_cluster(
        self,
        cluster_returns: np.ndarray,
        risk_free_rate: float,
    ) -> np.ndarray:
        """Optimize weights across clusters."""
        log_returns = np.log1p(cluster_returns)
        expected_returns = np.expm1(
            np.mean(log_returns, axis=0) * self.annualization,
        )
        cov_matrix = np.cov(cluster_returns, rowvar=False, ddof=1) * self.annualization
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])

        return self._optimize_max_sharpe(
            mu=expected_returns,
            cov=cov_matrix,
            risk_free_rate=risk_free_rate,
            max_weight=self.max_weight_inter,
        )

    @staticmethod
    def _fallback_weights(cov: np.ndarray, max_weight: float) -> np.ndarray:
        """Return inverse-volatility weights clipped to `max_weight`."""
        vol = np.sqrt(np.maximum(np.diag(cov), 1e-20))
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        clipped = np.minimum(weights, max_weight)
        return clipped / clipped.sum()

    def _optimize_max_sharpe(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        risk_free_rate: float,
        max_weight: float,
    ) -> np.ndarray:
        """Maximize Sharpe ratio with fallback to inverse-volatility weights."""
        excess_mu = mu - risk_free_rate
        n_assets = len(mu)

        if n_assets == 0:
            raise ValueError("Cannot optimize empty portfolio")
        if not (0.0 < max_weight <= 1.0):
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if max_weight * n_assets < 1.0 - self.eps:
            raise ValueError(f"Infeasible: n={n_assets}, max_weight={max_weight}")

        if not np.isfinite(cov).all() or not np.isfinite(excess_mu).all():
            return self._fallback_weights(cov, max_weight)
        if excess_mu.max() <= 0:
            return self._fallback_weights(cov, max_weight)

        cov = 0.5 * (cov + cov.T)
        min_eig = float(np.min(np.linalg.eigvalsh(cov)))
        if min_eig < 0.0:
            cov = cov + ((-min_eig) + self.eps) * np.eye(n_assets)

        weights_var = cp.Variable(n_assets)
        portfolio_var = cp.quad_form(weights_var, cov)
        scale = cp.sum(weights_var)

        constraints = [
            excess_mu @ weights_var == 1.0,
            weights_var >= 0.0,
            weights_var <= max_weight * scale,
        ]
        problem = cp.Problem(cp.Minimize(portfolio_var), constraints)

        status = _solve_with_fallback(
            problem,
            _solver_candidates(self.solver, "CLARABEL", "SCS"),
            ignore_dpp=True,
        )
        if status != cp.OPTIMAL:
            return self._fallback_weights(cov, max_weight)

        raw_weights = weights_var.value
        if raw_weights is None or not np.isfinite(raw_weights).all():
            return self._fallback_weights(cov, max_weight)

        optimized = np.maximum(raw_weights, 0.0)
        total = optimized.sum()
        if total <= self.eps:
            return self._fallback_weights(cov, max_weight)
        return optimized / total

    def _combine_weights(
        self,
        intra_weights: dict[int, np.ndarray],
        inter_weights: np.ndarray,
        clusters: dict[int, np.ndarray],
        n_assets: int,
    ) -> np.ndarray:
        """Combine intra and inter-cluster weights hierarchically."""
        final_weights = np.zeros(n_assets, dtype=np.float64)
        cluster_ids = list(clusters.keys())

        for idx, cluster_id in enumerate(cluster_ids):
            asset_idx = clusters[cluster_id]
            final_weights[asset_idx] = intra_weights[cluster_id] * inter_weights[idx]

        total = final_weights.sum()
        if total > 0:
            final_weights /= total
        return final_weights


class RiskParityStrategy(OptimizationStrategy):
    """Risk parity portfolio with equal or custom risk budgets (SLSQP + analytic gradient)."""

    def __init__(
        self,
        risk_budgets: np.ndarray | None = None,
        max_weight: float = 0.9,
        use_denoising: bool = True,
        risk_model: RiskModel | None = None,
        eps: float = 1e-12,
        warm_start: bool = True,
    ):
        super().__init__(
            max_weight=max_weight,
            use_denoising=use_denoising,
            risk_model=risk_model,
        )
        self.risk_budgets = risk_budgets
        self.eps = float(eps)
        self.warm_start = bool(warm_start)
        self._last_weights: pd.Series | None = None

    def _stabilized_covariance(self, cov_df: pd.DataFrame) -> np.ndarray:
        """Return symmetric covariance matrix with diagonal jitter."""
        cov = np.asarray(cov_df.values, dtype=np.float64, order="C")
        cov = 0.5 * (cov + cov.T)
        cov.flat[:: cov.shape[0] + 1] += self.eps
        return cov

    def _normalized_budgets(self, n_assets: int) -> np.ndarray:
        """Return normalized risk budgets for the current asset count."""
        budgets = self.risk_budgets
        if budgets is None:
            return np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

        normalized = np.asarray(budgets, dtype=np.float64).reshape(-1)
        if normalized.size != n_assets:
            raise ValueError(f"risk_budgets length {normalized.size} != n_assets {n_assets}")

        budget_sum = normalized.sum()
        if not np.isfinite(budget_sum) or budget_sum <= 0.0:
            raise ValueError("risk_budgets must sum to a positive finite value")
        return normalized / budget_sum

    def _normalize_weights(self, weights: np.ndarray, n_assets: int) -> np.ndarray:
        """Clip weights to bounds and normalize to sum to one."""
        clipped = np.clip(weights, 0.0, self.max_weight)
        total = clipped.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("normalized weights sum is non-finite or non-positive")
        return clipped / total

    def _initial_weights(self, cov: np.ndarray, columns: pd.Index) -> np.ndarray:
        """Build warm-start or inverse-volatility initial weights."""
        n_assets = len(columns)

        if self.warm_start and self._last_weights is not None:
            warm_weights = self._last_weights.reindex(columns).to_numpy(
                dtype=np.float64,
                copy=True,
            )
            if warm_weights.size == n_assets and np.all(np.isfinite(warm_weights)):
                return self._normalize_weights(warm_weights, n_assets)

        vol = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        inv_vol = np.where(vol > 1e-12, 1.0 / vol, 0.0)
        inv_sum = inv_vol.sum()
        if not np.isfinite(inv_sum) or inv_sum <= 0.0:
            raise RuntimeError("failed to initialize inverse-volatility weights")
        return self._normalize_weights(inv_vol / inv_sum, n_assets)

    def _objective_and_gradient(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        budgets: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return risk-budget objective and analytic gradient."""
        weights = np.asarray(weights, dtype=np.float64, order="C")

        cov_times_w = cov @ weights
        portfolio_var = float(weights @ cov_times_w)
        portfolio_var = max(portfolio_var, self.eps)
        portfolio_vol = np.sqrt(portfolio_var)

        weighted_cov = weights * cov_times_w
        risk_contrib = weighted_cov / portfolio_vol
        target_contrib = budgets * portfolio_vol

        error = risk_contrib - target_contrib
        objective = float(error @ error)

        weights_error = weights * error
        cov_weights_error = cov @ weights_error

        weighted_cov_dot_error = float(weighted_cov @ error)
        budget_dot_error = float(budgets @ error)

        grad_main = (cov_times_w * error + cov_weights_error) / portfolio_vol
        grad_rank = -cov_times_w * (weighted_cov_dot_error / (portfolio_vol**3))
        grad_target = -cov_times_w * (budget_dot_error / portfolio_vol)
        gradient = 2.0 * (grad_main + grad_rank + grad_target)

        return objective, gradient

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        del kwargs
        cov_df = self.estimate_covariance(returns)
        columns = cov_df.columns
        n_assets = len(columns)

        cov = self._stabilized_covariance(cov_df)
        budgets = self._normalized_budgets(n_assets)
        initial_weights = self._initial_weights(cov, columns)

        bounds = [(0.0, float(self.max_weight))] * n_assets
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

        def objective(w):
            return self._objective_and_gradient(w, cov, budgets)[0]
        def gradient(w):
            return self._objective_and_gradient(w, cov, budgets)[1]

        result = minimize(
            fun=objective,
            x0=initial_weights,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"risk parity optimization failed: {result.message}")

        weights = np.asarray(result.x, dtype=np.float64)
        if weights.shape != (n_assets,) or not np.all(np.isfinite(weights)):
            raise RuntimeError("optimizer returned invalid weights")
        normalized_weights = self._normalize_weights(weights, n_assets)

        output = pd.Series(normalized_weights, index=columns)
        self._last_weights = output
        return output


class TailRiskParityStrategy(OptimizationStrategy):
    """
    Tail-aware risk parity using CVaR, worst-k, or Ulcer Index risk contributions.
    Unlike RiskParityStrategy (volatility-based), this class budgets risk using
    measures that distinguish between upside and downside, penalizing tail events.
    Risk measures:
    - 'cvar': Euler-decomposed CVaR (Expected Shortfall) at given confidence level
    - 'worst_k': Average loss contribution over k worst portfolio days
    - 'ulcer': Inverse Ulcer Index weighting (drawdown-based, path-dependent)
    """

    def __init__(
        self,
        risk_measure: str = "cvar",
        risk_budgets: np.ndarray | None = None,
        confidence: float = 0.95,
        worst_k: int = 10,
        max_weight: float = 0.9,
        use_denoising: bool = True,
        risk_model: RiskModel | None = None,
    ):
        super().__init__(
            max_weight=max_weight,
            use_denoising=use_denoising,
            risk_model=risk_model,
        )
        self.risk_measure = risk_measure
        self.risk_budgets = risk_budgets
        self.confidence = confidence
        self.worst_k = worst_k
        valid_measures = {"cvar", "worst_k", "ulcer"}
        if self.risk_measure not in valid_measures:
            raise ValueError(f"risk_measure must be one of {valid_measures}, got '{self.risk_measure}'")

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        n_assets = returns.shape[1]
        returns_clean = returns.fillna(0)
        ret_values = returns_clean.values
        risk_budgets = self.risk_budgets
        if risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets

        # Initial guess: inverse volatility
        cov_denoised = self.estimate_covariance(returns)
        volatilities = np.sqrt(np.diag(cov_denoised.values))
        inv_vol = np.where(volatilities > 1e-8, 1 / volatilities, 0)
        w_init = inv_vol / np.sum(inv_vol) if np.sum(inv_vol) > 0 else np.ones(n_assets) / n_assets

        def objective(w):
            risk_fracs = self._compute_risk_fractions(w, ret_values)
            if risk_fracs is None:
                return 1e6
            return np.sum((risk_fracs - risk_budgets) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, self.max_weight) for _ in range(n_assets))
        result = minimize(
            objective,
            w_init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not result.success:
            # fallback to inverse-vol
            cov = self.estimate_covariance(returns)
            vols = np.sqrt(np.diag(cov.values))
            inv_vol = np.where(vols > 1e-8, 1 / vols, 0)
            w = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else np.ones(n_assets) / n_assets
            return pd.Series(w, index=returns.columns)
        w = np.clip(result.x, 0.0, self.max_weight)
        w_sum = w.sum()
        return pd.Series(w / w_sum if w_sum > 1e-10 else np.ones(n_assets) / n_assets, index=returns.columns)

    def _compute_risk_fractions(self, w: np.ndarray, ret_values: np.ndarray) -> np.ndarray | None:
        if self.risk_measure == "cvar":
            return self._cvar_risk_fractions(w, ret_values)
        elif self.risk_measure == "worst_k":
            return self._worst_k_risk_fractions(w, ret_values)
        else:  # ulcer
            return self._ulcer_risk_fractions(w, ret_values)

    def _cvar_risk_fractions(self, w: np.ndarray, ret_values: np.ndarray) -> np.ndarray | None:
        r_p = ret_values @ w
        n = len(r_p)
        alpha = 1 - self.confidence
        num_tail = max(1, int(n * alpha))

        tail_indices = np.argpartition(r_p, num_tail - 1)[:num_tail]
        mean_tail = np.mean(ret_values[tail_indices], axis=0)
        component_cvar = -w * mean_tail

        total_cvar = component_cvar.sum()
        if total_cvar <= 1e-12:
            return None
        return component_cvar / total_cvar

    def _worst_k_risk_fractions(self, w: np.ndarray, ret_values: np.ndarray) -> np.ndarray | None:
        r_p = ret_values @ w
        k = min(self.worst_k, len(r_p))
        if k == 0:
            return None

        worst_indices = np.argpartition(r_p, k)[:k]
        mean_worst = np.mean(ret_values[worst_indices], axis=0)
        component_risk = -w * mean_worst

        total_risk = component_risk.sum()
        if total_risk <= 1e-12:
            return None
        return component_risk / total_risk

    def _ulcer_risk_fractions(self, w: np.ndarray, ret_values: np.ndarray) -> np.ndarray | None:
        # Broadcast weights over time dimension: (T, N)
        weighted_returns = w[None, :] * ret_values

        # Equity curves for all assets simultaneously
        equity = np.cumprod(1 + weighted_returns, axis=0)

        # Running maximums
        running_max = np.maximum.accumulate(equity, axis=0)

        # Defensive drawdown (handles total ruin where running_max == 0)
        drawdown = np.where(running_max > 0, 1 - equity / running_max, 1.0)

        # Per-asset Ulcer Index
        ulcer_indices = np.sqrt(np.mean(drawdown**2, axis=0))

        total_ulcer = ulcer_indices.sum()
        if total_ulcer <= 1e-12:
            return None
        return ulcer_indices / total_ulcer


class HRPStrategy(OptimizationStrategy):
    """Hierarchical Risk Parity using correlation matrix clustering."""

    def __init__(
        self,
        linkage_method: str = "ward",
        max_weight: float = 0.9,
        use_denoising: bool = True,
        risk_model: RiskModel | None = None,
        eps: float = 1e-12,
    ):
        super().__init__(max_weight=max_weight, use_denoising=use_denoising, risk_model=risk_model)
        self.linkage_method = linkage_method
        self.eps = float(eps)

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute HRP weights aligned to the original asset order."""
        del kwargs
        covariance = self.estimate_covariance(returns)
        covariance_values = covariance.to_numpy(dtype=np.float64, copy=False)

        sorted_idx = self._sorted_asset_indices(covariance_values)
        covariance_sorted = covariance_values[np.ix_(sorted_idx, sorted_idx)]
        sorted_weights = self._recursive_bisection_numpy(covariance_sorted)

        return self._weights_in_original_order(
            sorted_weights=sorted_weights,
            sorted_idx=sorted_idx,
            assets=covariance.index,
        )

    def _sorted_asset_indices(self, covariance_values: np.ndarray) -> list[int]:
        """Return quasi-diagonalized leaf ordering from covariance matrix."""
        correlation = self._covariance_to_correlation(covariance_values, eps=self.eps)
        distance = self._correlation_to_distance(correlation)
        condensed_distance = squareform(distance, checks=False)
        link = linkage(condensed_distance, method=self.linkage_method)
        return self._quasi_diagonalization(link)

    @staticmethod
    def _covariance_to_correlation(covariance_values: np.ndarray, eps: float) -> np.ndarray:
        """Convert covariance matrix to correlation matrix with finite diagonal stabilization."""
        diagonal = np.diag(covariance_values)
        volatility = np.sqrt(np.maximum(diagonal, eps))
        inv_volatility = 1.0 / volatility
        correlation = (covariance_values * inv_volatility[None, :]) * inv_volatility[:, None]
        np.clip(correlation, -1.0, 1.0, out=correlation)
        return correlation

    @staticmethod
    def _correlation_to_distance(correlation: np.ndarray) -> np.ndarray:
        """Convert correlation matrix into HRP distance matrix."""
        return np.sqrt(0.5 * (1.0 - correlation))

    @staticmethod
    def _quasi_diagonalization(link: np.ndarray) -> list[int]:
        """Produce the leaf ordering implied by the hierarchical clustering linkage."""
        children = np.asarray(link[:, :2], dtype=np.int64)
        n_assets = int(link[-1, 3])

        order = [int(children[-1, 0]), int(children[-1, 1])]
        cursor = 0
        while cursor < len(order):
            node = order[cursor]
            if node >= n_assets:
                left, right = children[node - n_assets]
                order[cursor : cursor + 1] = [int(left), int(right)]
            else:
                cursor += 1
        return order

    def _recursive_bisection_numpy(self, cov_sorted: np.ndarray) -> np.ndarray:
        """HRP recursive bisection."""
        n_assets = cov_sorted.shape[0]
        weights = np.ones(n_assets, dtype=np.float64)

        # Precompute inverse diagonal once; used for IVP inside cluster variance
        diag = np.diag(cov_sorted)
        inv_diag = 1.0 / np.maximum(diag, self.eps)

        # Worklist of (start, end) half-open intervals in the sorted order
        stack: list[tuple[int, int]] = [(0, n_assets)]

        while stack:
            start, end = stack.pop()
            size = end - start
            if size <= 1:
                continue

            mid = start + (size // 2)
            left_start, left_end = start, mid
            right_start, right_end = mid, end

            left_var = self._cluster_variance_interval(cov_sorted, inv_diag, left_start, left_end)
            right_var = self._cluster_variance_interval(cov_sorted, inv_diag, right_start, right_end)

            alpha = 1.0 - (left_var / (left_var + right_var))
            weights[left_start:left_end] *= alpha
            weights[right_start:right_end] *= 1.0 - alpha

            stack.append((left_start, left_end))
            stack.append((right_start, right_end))

        return weights

    @staticmethod
    def _cluster_variance_interval(
        cov_sorted: np.ndarray,
        inv_diag: np.ndarray,
        start: int,
        end: int,
    ) -> float:
        """
        Cluster variance for a contiguous interval [start, end) in sorted space,
        using inverse-variance portfolio weights computed from the diagonal only.
        """
        idx = slice(start, end)
        inv = inv_diag[idx]
        w = inv / inv.sum()

        cov_sub = cov_sorted[idx, idx]
        # variance = w.T @ cov_sub @ w
        return float(w @ (cov_sub @ w))

    @staticmethod
    def _weights_in_original_order(sorted_weights: np.ndarray, sorted_idx: list[int], assets: pd.Index) -> pd.Series:
        """Map sorted weights back to the original asset index."""
        weights = np.zeros(len(assets), dtype=np.float64)
        weights[np.asarray(sorted_idx, dtype=int)] = sorted_weights
        return pd.Series(weights, index=assets, dtype=np.float64)


class EnsembleStrategy(OptimizationStrategy):
    """
    Production-grade ensemble combining multiple portfolio optimization strategies.

    Supports three robust methods:
    - 'equal': 1/N weighting (DeMiguel et al. 2009) - estimation-error robust baseline
    - 'minvar': Minimum variance across strategies - optimal diversification
    - 'hrp': Hierarchical Risk Parity - correlation-based clustering
    """

    def __init__(
        self,
        strategies: dict[str, OptimizationStrategy],
        ensemble_optimizer: str = "minvar",  # 'equal', 'minvar', or 'hrp'
        lookback_window: int = 252,
        min_strategy_weight: float = 0.05,
        max_strategy_weight: float = 0.95,
        cov_shrinkage: float = 0.0,
        hrp_linkage: str = "ward",  # Linkage method for HRP optimizer
        max_weight: float = 0.9,
        risk_model: RiskModel | None = None,
    ):
        super().__init__(max_weight=max_weight, risk_model=risk_model)
        self.strategies = strategies
        self.ensemble_optimizer = ensemble_optimizer
        self.lookback_window = lookback_window
        self.min_strategy_weight = min_strategy_weight
        self.max_strategy_weight = max_strategy_weight
        self.cov_shrinkage = cov_shrinkage
        self.hrp_linkage = hrp_linkage

        # Validate inputs
        valid_optimizers = {"equal", "minvar", "hrp"}
        if not strategies:
            raise ValueError("strategies cannot be empty")
        if self.ensemble_optimizer not in valid_optimizers:
            raise ValueError(f"ensemble_optimizer must be {valid_optimizers}, got '{self.ensemble_optimizer}'")
        if self.lookback_window <= 0:
            raise ValueError("lookback_window must be > 0")
        if not (0.0 <= self.min_strategy_weight <= self.max_strategy_weight <= 1.0):
            raise ValueError("strategy weights must satisfy 0 <= min_strategy_weight <= max_strategy_weight <= 1")

        if not (0 <= self.cov_shrinkage <= 1):
            raise ValueError("cov_shrinkage must be in [0, 1]")

        # Initialize HRP instance for strategy-level optimization
        if self.ensemble_optimizer == "hrp":
            self._hrp_instance = HRPStrategy(
                linkage_method=self.hrp_linkage,
                max_weight=self.max_strategy_weight,
                use_denoising=False,  # Strategy returns too short for meaningful denoising
                risk_model=self.risk_model,
            )
        self._strategy_projection_cache: dict[int, tuple[np.ndarray, np.ndarray, bool]] = {}
        self._asset_projection_cache: dict[int, tuple[np.ndarray, np.ndarray, bool]] = {}

    @staticmethod
    def _bounds_feasible(lower: np.ndarray, upper: np.ndarray, target_sum: float = 1.0) -> bool:
        tol = 1e-12
        return (lower.sum() - tol) <= target_sum <= (upper.sum() + tol)

    @staticmethod
    def _project_to_bounded_simplex(
        values: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target_sum: float = 1.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Project values onto {x | sum(x)=target_sum, lower<=x<=upper}."""
        values = np.asarray(values, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if np.any(lower > upper):
            raise ValueError("lower bounds cannot exceed upper bounds")
        if not EnsembleStrategy._bounds_feasible(lower, upper, target_sum=target_sum):
            raise ValueError("infeasible bounds for target_sum")

        left = np.min(values - upper)
        right = np.max(values - lower)
        x = np.clip(values, lower, upper)
        if abs(x.sum() - target_sum) <= tol:
            return x

        for _ in range(max_iter):
            mid = 0.5 * (left + right)
            x = np.clip(values - mid, lower, upper)
            total = x.sum()
            if abs(total - target_sum) <= tol:
                return x
            if total > target_sum:
                left = mid
            else:
                right = mid
        return np.clip(values - 0.5 * (left + right), lower, upper)

    def _build_strategy_returns_df(
        self,
        lookback_returns: pd.DataFrame,
        sub_weights: dict[str, pd.Series],
    ) -> pd.DataFrame:
        strategy_names = list(sub_weights.keys())
        filled_returns = lookback_returns.fillna(0.0)
        asset_names = filled_returns.columns
        ret_matrix = filled_returns.to_numpy(dtype=float)

        aligned_weight_rows: list[np.ndarray] = []
        all_aligned = True
        for name in strategy_names:
            weights = sub_weights[name]
            if not weights.index.equals(asset_names):
                all_aligned = False
                break
            aligned_weight_rows.append(weights.to_numpy(dtype=float, copy=False))

        if all_aligned:
            weight_matrix = np.vstack(aligned_weight_rows)
        else:
            weight_matrix = np.vstack(
                [
                    sub_weights[name].reindex(asset_names, fill_value=0.0).to_numpy(dtype=float)
                    for name in strategy_names
                ]
            )
        strategy_returns = ret_matrix @ weight_matrix.T
        return pd.DataFrame(strategy_returns, index=filled_returns.index, columns=strategy_names)

    def _get_strategy_projection_inputs(
        self,
        n_strats: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cached = self._strategy_projection_cache.get(n_strats)
        if cached is not None:
            return cached

        lower = np.full(n_strats, self.min_strategy_weight, dtype=float)
        upper = np.full(n_strats, self.max_strategy_weight, dtype=float)
        feasible = self._bounds_feasible(lower, upper)
        cached = (lower, upper, feasible)
        self._strategy_projection_cache[n_strats] = cached
        return cached

    def _get_asset_projection_inputs(
        self,
        n_assets: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cached = self._asset_projection_cache.get(n_assets)
        if cached is not None:
            return cached

        lower = np.zeros(n_assets, dtype=float)
        upper = np.full(n_assets, self.max_weight, dtype=float)
        feasible = self._bounds_feasible(lower, upper)
        cached = (lower, upper, feasible)
        self._asset_projection_cache[n_assets] = cached
        return cached

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute ensemble portfolio weights.

        1. Compute asset weights from each sub-strategy (same returns window)
        2. Compute strategy-level weights via ensemble_optimizer
        3. Blend: w_final = Σ_s (w_s * sub_strategy_weights_s)
        4. Apply asset constraints
        """
        sub_weights = {name: strategy.compute_weights(returns, **kwargs) for name, strategy in self.strategies.items()}

        if self.ensemble_optimizer == "equal":
            strategy_weights = self._equal_weights()
        elif self.ensemble_optimizer == "minvar":
            strategy_weights = self._minvar_weights(returns, sub_weights)
        else:  # 'hrp'
            strategy_weights = self._hrp_weights(returns, sub_weights)

        final_weights = self._blend_weights(sub_weights, strategy_weights)
        n_assets = len(final_weights)
        if n_assets == 0:
            raise ValueError("sub-strategy weights produced an empty asset universe")
        lower, upper, feasible = self._get_asset_projection_inputs(n_assets)
        if feasible:
            projected = self._project_to_bounded_simplex(
                final_weights.to_numpy(dtype=float),
                lower=lower,
                upper=upper,
            )
            final_weights = pd.Series(projected, index=final_weights.index)
        else:
            # Infeasible cap (e.g. one asset with max_weight < 1): normalize defensively.
            final_weights = final_weights.clip(lower=0.0)
            if final_weights.sum() > 1e-10:
                final_weights /= final_weights.sum()
            else:
                final_weights = pd.Series(1.0 / n_assets, index=final_weights.index)

        return final_weights

    def _equal_weights(self) -> pd.Series:
        """1/N equal weighting - O(1), robust baseline [DeMiguel2009]."""
        n_strats = len(self.strategies)
        return pd.Series(1.0 / n_strats, index=self.strategies.keys())

    def _minvar_weights(self, returns: pd.DataFrame, sub_weights: dict[str, pd.Series]) -> pd.Series:
        """
        Minimum variance strategy weights using closed-form solution.

        Process:
        1. Build strategy return matrix over lookback_window
        2. Estimate covariance with shrinkage
        3. Closed-form: w = Σ⁻¹1 / (1ᵀΣ⁻¹1)
        4. Apply bounds and renormalize

        Complexity: O(S³) where S = # strategies (negligible for S≤10)
        """
        lookback_returns = returns.iloc[-self.lookback_window :]

        if len(lookback_returns) < 20:
            return self._equal_weights()
        strategy_names = list(sub_weights.keys())
        strategy_returns_df = self._build_strategy_returns_df(lookback_returns, sub_weights)

        cov_raw = self.estimate_covariance(strategy_returns_df)
        n_strats = len(strategy_names)

        shrinkage_target = np.eye(n_strats) * np.trace(cov_raw.values) / n_strats
        cov_shrunk = (1 - self.cov_shrinkage) * cov_raw.values + self.cov_shrinkage * shrinkage_target

        eigvals = np.linalg.eigvalsh(cov_shrunk)
        if eigvals[0] < 1e-8:
            cov_shrunk += np.eye(n_strats) * (1e-6 - eigvals[0])

        try:
            inv_cov = np.linalg.inv(cov_shrunk)
            ones = np.ones(n_strats)
            raw_weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        except np.linalg.LinAlgError:
            return self._equal_weights()

        lower, upper, feasible = self._get_strategy_projection_inputs(n_strats)
        if not feasible:
            return self._equal_weights()
        weights = self._project_to_bounded_simplex(raw_weights, lower=lower, upper=upper)

        return pd.Series(weights, index=strategy_names)

    def _hrp_weights(self, returns: pd.DataFrame, sub_weights: dict[str, pd.Series]) -> pd.Series:
        """
        HRP strategy weights via composition with HRPStrategy.

        Process:
        1. Build strategy return matrix over lookback_window
        2. Treat strategies as "assets" and compute HRP weights
        3. Apply bounds and renormalize

        Complexity: O(S² log S) for S strategies
        """
        lookback_returns = returns.iloc[-self.lookback_window :]

        if len(lookback_returns) < 20:
            return self._equal_weights()
        strategy_names = list(sub_weights.keys())
        strategy_returns_df = self._build_strategy_returns_df(lookback_returns, sub_weights)

        raw_weights = self._hrp_instance.compute_weights(strategy_returns_df)

        n_strats = len(strategy_names)
        lower, upper, feasible = self._get_strategy_projection_inputs(n_strats)
        if not feasible:
            return self._equal_weights()
        raw_vector = raw_weights.reindex(strategy_names, fill_value=0.0).to_numpy(dtype=float)
        weights = self._project_to_bounded_simplex(raw_vector, lower=lower, upper=upper)

        return pd.Series(weights, index=strategy_names)

    def _blend_weights(self, sub_weights: dict[str, pd.Series], strategy_weights: pd.Series) -> pd.Series:
        r"""Hierarchical blending with index-safe alignment across strategy universes."""
        strategy_names = list(sub_weights.keys())
        first_weights = sub_weights[strategy_names[0]]
        all_aligned = all(sub_weights[name].index.equals(first_weights.index) for name in strategy_names[1:])

        if all_aligned:
            weights_matrix = np.column_stack(
                [sub_weights[name].to_numpy(dtype=float, copy=False) for name in strategy_names],
            )
            aligned_strategy_weights = strategy_weights.reindex(strategy_names, fill_value=0.0)
            final_values = weights_matrix @ aligned_strategy_weights.to_numpy(dtype=float, copy=False)
            return pd.Series(final_values, index=first_weights.index)

        weights_table = pd.concat(
            [w.rename(name) for name, w in sub_weights.items()],
            axis=1,
        ).fillna(0.0)
        aligned_strategy_weights = strategy_weights.reindex(weights_table.columns, fill_value=0.0)
        final_values = weights_table.to_numpy(dtype=float) @ aligned_strategy_weights.to_numpy(dtype=float)
        return pd.Series(final_values, index=weights_table.index)

    def diagnose(self, returns: pd.DataFrame) -> dict:
        """
        Diagnostic info for ensemble analysis.

        Returns:
            Dict with strategy_weights, strategy_corr, strategy_vol, concentration_hhi

        """
        sub_weights = {name: strat.compute_weights(returns) for name, strat in self.strategies.items()}

        if self.ensemble_optimizer == "minvar":
            strategy_weights = self._minvar_weights(returns, sub_weights)
        elif self.ensemble_optimizer == "hrp":
            strategy_weights = self._hrp_weights(returns, sub_weights)
        else:
            strategy_weights = self._equal_weights()

        lookback_returns = returns.iloc[-self.lookback_window :]
        strategy_df = self._build_strategy_returns_df(lookback_returns, sub_weights)

        return {
            "strategy_weights": strategy_weights,
            "strategy_corr": strategy_df.corr(),
            "strategy_vol": strategy_df.std() * np.sqrt(252),
            "concentration_hhi": (strategy_weights**2).sum(),
            "equal_hhi": 1.0 / len(self.strategies),
        }

    def __repr__(self) -> str:
        names = list(self.strategies.keys())
        return (
            f"EnsembleStrategy(optimizer='{self.ensemble_optimizer}', "
            f"strategies={names[:3]}{'...' if len(names) > 3 else ''}, "
            f"lookback={self.lookback_window}, shrinkage={self.cov_shrinkage})"
        )


@dataclass
class EnsemblePositionSizer:
    """Position-sizing wrapper around an EnsembleStrategy using Optimal f."""

    ensemble: EnsembleStrategy
    lookback_window: int = 252
    min_window_len: int = 60

    # Parameters passed through to OptimalFCalculator.calculate_optimal_f
    sigma_bounds: list[float] | None | str = "use_instance"
    sigma_bounds_margin: tuple[float, float] | None = None
    f_min: float = 0.0
    f_max: float = 5.0
    f_step: float = 0.01
    n_mc: int = 200_000
    seed: int = 89
    max_memory_mb: int = 500

    def compute_targets(
        self,
        returns: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        r"""
        Compute asset weights and portfolio-level 'risk % of equity' at a rebalance date.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset return matrix with DatetimeIndex (daily or whatever your
            base frequency is), columns = assets. Assumed pre-normalized to
            common annual volatility if that is your convention.
        as_of : pd.Timestamp, optional
            Rebalance date. If provided, only data up to and including this
            date is used. If None, the last index value in `returns` is used.

        Returns
        -------
        dict
            {
                "weights": pd.Series,
                    Final *relative* asset weights (sum to 1)
                "risk_fraction": float,
                    Fraction of equity suggested to risk per period (0–∞).
                "risk_percent": float,
                    Same as risk_fraction but expressed in percent (0–∞).
                "optimal_f": float,
                    Raw optimal f from OptimalFCalculator.
                "largest_loss": float,
                    Magnitude of largest empirical loss in the ensemble window.
                "optf_result": Dict[str, Any],
                    Full dict returned by OptimalFCalculator.calculate_optimal_f.
            }

        Notes
        -----
        - This method is intentionally *stateless* and uses only historical
          data up to `as_of`. There is no look-ahead.
        - It assumes `returns` are clean and aligned; NaN handling should be
          done upstream, consistent with your backtest.

        """
        if returns.empty:
            raise ValueError("returns is empty")

        # 1) Determine end index for the lookback window
        if as_of is None:
            as_of = returns.index[-1]
        else:
            # Ensure as_of is in the index (or adjust if needed)
            if as_of not in returns.index:
                # Forward-fill style alignment: use the last available date <= as_of
                # This keeps the logic robust if as_of is a non-trading day.
                loc = returns.index.searchsorted(as_of, side="right") - 1
                if loc < 0:
                    raise ValueError("as_of is before the first return index")
                as_of = returns.index[loc]

        # Slice history up to as_of (inclusive)
        hist = returns.loc[:as_of]

        # 2) Extract lookback window (last lookback_window observations)
        window = hist.iloc[-self.lookback_window :]

        if len(window) < self.min_window_len:
            # Not enough data to estimate both optimizer and Optimal f reliably
            # Fallback: equal-weight & zero extra risk (risk_fraction = 0)
            assets = returns.columns
            fallback_weights = pd.Series(1.0 / len(assets), index=assets)

            return {
                "weights": fallback_weights,
                "risk_fraction": 0.0,
                "risk_percent": 0.0,
                "optimal_f": 0.0,
                "largest_loss": np.nan,
                "optf_result": {},
            }

        # 3) Cross-sectional allocation via EnsembleStrategy (no look-ahead)
        #    We assume the same `window` is passed here as in your backtest
        #    for a given rebalance date.
        asset_weights = self.ensemble.compute_weights(window)

        # Ensure weights align to columns of `returns` (fill missing with 0)
        asset_weights = asset_weights.reindex(returns.columns).fillna(0.0)
        if asset_weights.sum() > 0:
            asset_weights /= asset_weights.sum()

        # 4) Construct ensemble portfolio return series over the window
        #    This is fully vectorized: T×N matrix times N-vector of weights.
        #    Result: T-vector of ensemble returns.
        ensemble_returns = (window * asset_weights).sum(axis=1)

        # 5) Feed ensemble returns into OptimalFCalculator
        # plt.hist(ensemble_returns, bins=50)
        # plt.show()
        optf_calc = OptimalFCalculator(portfolio_returns=ensemble_returns)

        optf_result = optf_calc.calculate_optimal_f(
            sigma_bounds=self.sigma_bounds,
            sigma_bounds_margin=self.sigma_bounds_margin,
            f_min=self.f_min,
            f_max=self.f_max,
            f_step=self.f_step,
            n_mc=self.n_mc,
            seed=self.seed,
            max_memory_mb=self.max_memory_mb,
            fraction=0.2,
        )

        optimal_f = float(optf_result["optimal_f"])
        largest_loss = float(optf_result["largest_loss"])

        # 6) Map optimal_f to fraction and percent of equity to risk
        #    risk_fraction = f / |min_return|; risk_percent = 100 * risk_fraction
        if largest_loss > 0 and optimal_f > 0:
            risk_fraction = optimal_f / largest_loss
            risk_percent = 100.0 * risk_fraction
        else:
            # Cases:
            # - all returns >= 0  → optimal_f might be f_max with infinite growth
            # - largest_loss == 0 → no historical loss (very unusual)
            # For robustness we cap at 0 additional risk in those edge cases.
            risk_fraction = 0.0
            risk_percent = 0.0

        return {
            "weights": asset_weights,
            "risk_fraction": float(risk_fraction),
            "risk_percent": float(risk_percent),
            "optimal_f": optimal_f,
            "largest_loss": largest_loss,
            "optf_result": optf_result,
        }


class OptimalFCalculator(OptimizationStrategy):
    def __init__(
        self,
        portfolio_returns: pd.Series,
        exclude_zeros: bool = False,
        use_denoising: bool = True,
        risk_model: RiskModel | None = None,
    ):
        super().__init__(use_denoising=use_denoising, risk_model=risk_model)

        self.portfolio_returns = portfolio_returns.sort_index()
        self.exclude_zeros = exclude_zeros

        # Fit distribution and calculate sigma bounds on initialization
        self.t_params = self._fit_distribution()
        self.sigma_bounds = self._calculate_empirical_sigma_bounds()

    def _fit_distribution(self) -> dict[str, float]:
        """
        Fit Student-t distribution to portfolio returns.

        Returns
        -------
        dict
            Dictionary with t_df, t_loc, t_scale, mean_emp, std_emp, min_return, max_return

        """
        from scipy.stats import t

        returns = self.portfolio_returns.copy()

        if self.exclude_zeros:
            returns = returns.loc[returns != 0]

        ret = returns.to_numpy()

        # Fit Student-t distribution
        df_t, loc_t, scale_t = t.fit(ret)

        # Calculate empirical statistics
        mean_emp = np.mean(ret)
        std_emp = np.std(ret, ddof=1)

        return {
            "t_df": df_t,
            "t_loc": loc_t,
            "t_scale": scale_t,
            "mean_emp": mean_emp,
            "std_emp": std_emp,
            "min_return": ret.min(),
            "max_return": ret.max(),
        }

    @staticmethod
    def _calculate_std_away(value: float, df: float, loc: float, scale: float) -> float:
        """
        Calculate how many standard deviations a value is from the Student-t mean.

        Parameters
        ----------
        value : float
            The value to evaluate (e.g., empirical min or max return)
        df : float
            Degrees of freedom from Student-t fit
        loc : float
            Location parameter (mean) from Student-t fit
        scale : float
            Scale parameter from Student-t fit

        Returns
        -------
        float
            Number of standard deviations away. Negative means below mean.

        """
        if df <= 2:
            raise ValueError(f"Student-t std is undefined for df <= 2. Got df={df}")

        # Theoretical std of fitted distribution
        sigma_t = scale * np.sqrt(df / (df - 2.0))

        # Z-score
        z_score = (value - loc) / sigma_t

        return z_score

    def _calculate_empirical_sigma_bounds(self) -> list[float]:
        """
        Calculate empirical sigma bounds from fitted distribution.

        Returns
        -------
        list[float]
            [z_min, z_max] based on observed min/max returns

        """
        min_ret = self.t_params["min_return"]
        max_ret = self.t_params["max_return"]
        df = self.t_params["t_df"]
        loc = self.t_params["t_loc"]
        scale = self.t_params["t_scale"]
        try:
            # Preferred: t-based sigma bounds when df > 2
            z_min = self._calculate_std_away(
                value=min_ret,
                df=df,
                loc=loc,
                scale=scale,
            )
            z_max = self._calculate_std_away(
                value=max_ret,
                df=df,
                loc=loc,
                scale=scale,
            )
        except ValueError:
            # Fallback: Normal-style z-scores using empirical mean/std
            mean_emp = self.t_params["mean_emp"]
            std_emp = self.t_params["std_emp"]
            if std_emp <= 0:
                # Completely degenerate empirical distribution
                z_min = 0.0
                z_max = 0.0
            else:
                z_min = (min_ret - mean_emp) / std_emp
                z_max = (max_ret - mean_emp) / std_emp

        return [float(z_min), float(z_max)]

    @staticmethod
    def _sigma_bounds_to_returns(sigma_bounds: list[float], df: float, loc: float, scale: float) -> tuple[float, float]:
        """
        Convert sigma bounds (z-scores) to actual return bounds.

        Parameters
        ----------
        sigma_bounds : list[float]
            [lower_sigma, upper_sigma] e.g., [-3.5, 5.7]
        df, loc, scale : float
            Student-t distribution parameters

        Returns
        -------
        tuple[float, float]
            (lower_return_bound, upper_return_bound)

        """
        if df <= 2:
            raise ValueError(f"Student-t std is undefined for df <= 2. Got df={df}")

        sigma_t = scale * np.sqrt(df / (df - 2.0))

        lower_bound = loc + sigma_bounds[0] * sigma_t
        upper_bound = loc + sigma_bounds[1] * sigma_t

        return lower_bound, upper_bound

    def calculate_optimal_f(
        self,
        sigma_bounds: list[float] | None = "use_instance",
        sigma_bounds_margin: tuple[float, float] | None = None,
        f_min: float = 0.0,
        f_max: float = 5.0,
        f_step: float = 0.01,
        n_mc: int = 200_000,
        seed: int = 42,
        max_memory_mb: int = 500,
        fraction: float = 0.5,
    ) -> dict[str, Any]:
        """
        Calculate Optimal f using Student-t distribution via Monte Carlo.

        Parameters
        ----------
        sigma_bounds : list[float] or "use_instance" or None, default "use_instance"
            Custom [lower_sigma, upper_sigma].
            - "use_instance": uses self.sigma_bounds (default)
            - None: unbounded (no clipping)
            - list: custom bounds
        sigma_bounds_margin : tuple[float, float], optional
            (lower_margin, upper_margin) to add to sigma bounds, e.g., (-1, 2)
        f_min, f_max, f_step : float
            Grid parameters for f values
        n_mc : int
            Number of Monte Carlo samples
        seed : int
            Random seed for reproducibility
        max_memory_mb : int
            Maximum memory for batch processing

        Returns
        -------
        dict
            Optimal f results including optimal_f, growth_rate, TWR_per_period, etc.

        """
        # Determine which sigma bounds to use
        use_bounds = True
        if sigma_bounds == "use_instance":
            sigma_bounds = self.sigma_bounds.copy()
        elif sigma_bounds is None:
            use_bounds = False
            sigma_bounds = None
        else:
            sigma_bounds = sigma_bounds.copy()

        # Apply margin if provided and bounds are being used
        if use_bounds and sigma_bounds_margin is not None:
            sigma_bounds = [sigma_bounds[0] + sigma_bounds_margin[0], sigma_bounds[1] + sigma_bounds_margin[1]]

        rng = np.random.default_rng(seed)

        # Generate samples
        t_samples = rng.standard_t(self.t_params["t_df"], size=n_mc)
        r_samples = self.t_params["t_loc"] + (self.t_params["t_scale"] * t_samples)

        # Apply sigma bounds clipping only if use_bounds is True
        if use_bounds:
            df = self.t_params["t_df"]
            loc = self.t_params["t_loc"]
            scale = self.t_params["t_scale"]

            if df > 2:
                # Use t-based sigma
                lower_bound, upper_bound = self._sigma_bounds_to_returns(
                    sigma_bounds,
                    df,
                    loc,
                    scale,
                )
            else:
                # Fallback: Normal-style bounds using empirical mean/std
                mean_emp = self.t_params["mean_emp"]
                std_emp = self.t_params["std_emp"]
                if std_emp > 0:
                    lower_bound = mean_emp + sigma_bounds[0] * std_emp
                    upper_bound = mean_emp + sigma_bounds[1] * std_emp
                else:
                    # Degenerate: no dispersion; no clipping effectively
                    lower_bound = min(self.t_params["min_return"], 0.0)
                    upper_bound = max(self.t_params["max_return"], 0.0)

            r_samples = np.clip(r_samples, lower_bound, upper_bound)
        else:
            lower_bound = upper_bound = None

        # Define normalization factor
        r_min_emp = self.t_params["min_return"]

        if r_min_emp >= 0:
            return {
                "optimal_f": f_max,
                "growth_rate": np.inf,
                "TWR_per_period": np.inf,
                "GAT_return": np.inf,
                "largest_loss": 0.0,
                "worst_simulated_return": 0.0,
                "f_values": np.array([f_max]),
                "growth_curve": np.array([np.inf]),
                "bounds_applied": use_bounds,
                "sigma_bounds": sigma_bounds if use_bounds else None,
                "return_bounds": (lower_bound, upper_bound) if use_bounds else None,
            }

        largest_loss_mag = abs(r_min_emp)

        # Check simulated worst case
        r_min_sim = np.min(r_samples)
        if r_min_sim < 0:
            max_safe_f_sim = largest_loss_mag / abs(r_min_sim) * 0.99
            if max_safe_f_sim < f_max:
                f_max = max_safe_f_sim

        # Create grid
        f_values = np.arange(f_min, f_max + f_step / 100.0, f_step)
        if len(f_values) == 0:
            f_values = np.array([0.0])

        # Batch processing
        bytes_per_col = n_mc * 8
        cols_per_chunk = max(1, (max_memory_mb * 1024 * 1024) // bytes_per_col)

        expected_growth = []
        R_norm = r_samples / largest_loss_mag

        for i in range(0, len(f_values), cols_per_chunk):
            f_chunk = f_values[i : i + cols_per_chunk]
            HPR = 1.0 + (R_norm[:, np.newaxis] * f_chunk[np.newaxis, :])

            valid_mask = HPR > 1e-9
            log_utility = np.full_like(HPR, -np.inf)
            np.log(HPR, out=log_utility, where=valid_mask)

            chunk_means = np.mean(log_utility, axis=0)
            expected_growth.append(chunk_means)

        expected_growth = np.concatenate(expected_growth)

        # Find optima
        best_idx = np.argmax(expected_growth)
        f_star = f_values[best_idx] * fraction
        g_star = expected_growth[best_idx]

        twr_per_period = np.exp(g_star) if g_star > -700 else 0.0

        return {
            "optimal_f": float(f_star),
            "growth_rate": float(g_star),
            "TWR_per_period": float(twr_per_period),
            "GAT_return": float(twr_per_period - 1.0),
            "largest_loss": largest_loss_mag,
            "worst_simulated_return": float(r_min_sim),
            "f_values": f_values,
            "growth_curve": expected_growth,
            "bounds_applied": use_bounds,
            "sigma_bounds": sigma_bounds if use_bounds else None,
            "return_bounds": (lower_bound, upper_bound) if use_bounds else None,
        }

    def calculate_rolling_optimal_f(
        self,
        lookback: int = 252,
        rebalance_freq: str = "ME",
        sigma_bounds_margin: tuple[float, float] | None = None,
        min_df: float = 2.5,
        f_step: float = 0.001,
        n_mc: int = 100_000,
        verbose: bool = True,
        anchored: bool = False,  # ➤ NEW
    ) -> pd.DataFrame:
        """
        Compute rolling or anchored Student-t Optimal f on daily returns.

        Parameters
        ----------
        anchored : bool, default False
            If True, use an anchored window (from start to each rebalance date).
            If False, use traditional rolling window with lookback.

        """
        ret = self.portfolio_returns.sort_index().astype(float)

        # Get rebalancing dates
        rebal_dates = ret.resample(rebalance_freq).last().index
        rebal_dates = rebal_dates[rebal_dates.isin(ret.index)]

        records = []

        for d in rebal_dates:
            end_loc = ret.index.get_loc(d)

            start_loc = max(0, end_loc - lookback + 1) if not anchored else 0

            window = ret.iloc[start_loc : end_loc + 1]

            if len(window) < 60:
                continue

            w_vals = window.to_numpy()

            # ---- Fit Student-t --------------------------------------------------
            mu_norm = np.mean(w_vals)
            sigma_norm = np.std(w_vals, ddof=1)

            try:
                df_t, loc_t, scale_t = t.fit(w_vals)
            except Exception as e:
                if verbose:
                    print(f"Skipping {d}: Fit failed - {e}")
                continue

            if df_t <= min_df:
                if verbose:
                    print(f"Skipping {d}: df={df_t:.2f} <= {min_df}")
                continue

            # ---- Sigma bounds --------------------------------------------------
            try:
                z_min = self._calculate_std_away(w_vals.min(), df_t, loc_t, scale_t)
                z_max = self._calculate_std_away(w_vals.max(), df_t, loc_t, scale_t)
                window_sigma_bounds = [float(z_min), float(z_max)]
            except Exception as e:
                if verbose:
                    print(f"Skipping {d}: {e}")
                continue

            if sigma_bounds_margin is not None:
                window_sigma_bounds[0] += sigma_bounds_margin[0]
                window_sigma_bounds[1] += sigma_bounds_margin[1]

            # ---- Optimal-f Monte Carlo -----------------------------------------
            try:
                rng = np.random.default_rng(42)
                t_samples = rng.standard_t(df_t, size=n_mc)
                r_samples = loc_t + (scale_t * t_samples)

                lower_bound, upper_bound = self._sigma_bounds_to_returns(window_sigma_bounds, df_t, loc_t, scale_t)
                r_samples = np.clip(r_samples, lower_bound, upper_bound)

                largest_loss_mag = abs(w_vals.min())
                r_min_sim = np.min(r_samples)

                f_max = 5.0
                if r_min_sim < 0:
                    max_safe_f_sim = largest_loss_mag / abs(r_min_sim) * 0.99
                    if max_safe_f_sim < f_max:
                        f_max = max_safe_f_sim

                f_values = np.arange(0.0, f_max + f_step / 100.0, f_step)
                if len(f_values) == 0:
                    f_values = np.array([0.0])

                R_norm = r_samples / largest_loss_mag
                HPR = 1.0 + (R_norm[:, np.newaxis] * f_values[np.newaxis, :])

                valid_mask = HPR > 1e-9
                log_utility = np.full_like(HPR, -np.inf)
                np.log(HPR, out=log_utility, where=valid_mask)

                expected_growth = np.mean(log_utility, axis=0)

                best_idx = np.argmax(expected_growth)
                f_star = f_values[best_idx]
                g_star = expected_growth[best_idx]

                twr_per_period = np.exp(g_star) if g_star > -700 else 0.0

            except Exception as e:
                if verbose:
                    print(f"Skipping {d}: optimal_f calculation failed - {e}")
                continue

            equity_per_unit_pct = (f_star / abs(largest_loss_mag)) * 100 if f_star > 1e-6 else np.nan

            records.append(
                {
                    "date": d,
                    "optimal_f": f_star,
                    "growth_rate": g_star,
                    "TWR_per_day": twr_per_period,
                    "GAT_return": twr_per_period - 1.0,
                    "Risk%": equity_per_unit_pct,
                    "mu_emp": mu_norm,
                    "sigma_emp": sigma_norm,
                    "df_t": df_t,
                    "largest_loss": largest_loss_mag,
                    "window_len": len(window),
                    "sigma_bounds": str(window_sigma_bounds),
                }
            )

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("date")

    def plot_distribution_fit(self, figsize: tuple[int, int] = (14, 6)) -> None:
        """
        Plot histogram with fitted distributions and Q-Q plot.

        Parameters
        ----------
        figsize : tuple[int, int], default (14, 6)
            Figure size (width, height)

        """
        returns = self.portfolio_returns.copy()
        if self.exclude_zeros:
            returns = returns.loc[returns != 0]
        ret = returns.to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Histogram with fitted distributions
        x_min, x_max = ret.min(), ret.max()
        x_axis = np.linspace(x_min - 0.01, x_max + 0.01, 1000)

        t_pdf = t.pdf(x_axis, df=self.t_params["t_df"], loc=self.t_params["t_loc"], scale=self.t_params["t_scale"])
        norm_pdf = norm.pdf(x_axis, loc=self.t_params["mean_emp"], scale=self.t_params["std_emp"])

        axes[0].hist(
            ret, bins="auto", density=True, alpha=0.6, color="steelblue", edgecolor="black", label="Empirical Returns"
        )
        axes[0].plot(x_axis, t_pdf, "r-", linewidth=2.5, label=f"Student-t (df={self.t_params['t_df']:.2f})")
        axes[0].plot(x_axis, norm_pdf, "g--", linewidth=2, label="Normal")
        axes[0].set_title("Distribution Fit: Original Scale", fontweight="bold")
        axes[0].set_xlabel("Daily Return")
        axes[0].set_ylabel("Probability Density")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Plot 2: Q-Q plot
        probplot(
            ret, dist=t, sparams=(self.t_params["t_df"], self.t_params["t_loc"], self.t_params["t_scale"]), plot=axes[1]
        )
        axes[1].set_title("Q-Q Plot: Student-t Fit", fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()

    def print_summary(self) -> None:
        """Print summary of fitted distribution and sigma bounds."""
        print("=" * 70)
        print("OPTIMAL F CALCULATOR SUMMARY")
        print("=" * 70)
        print("\nStudent-t Distribution Parameters:")
        print(f"  Degrees of Freedom (df):  {self.t_params['t_df']:.4f}")
        print(f"  Location (loc):           {self.t_params['t_loc']:.6f}")
        print(f"  Scale:                    {self.t_params['t_scale']:.6f}")
        print("\nEmpirical Statistics:")
        print(f"  Mean:                     {self.t_params['mean_emp']:.6f}")
        print(f"  Std Dev:                  {self.t_params['std_emp']:.6f}")
        print(f"  Min Return:               {self.t_params['min_return']:.6f}")
        print(f"  Max Return:               {self.t_params['max_return']:.6f}")
        print("\nSigma Bounds:")
        print(f"  Z-score Min:              {self.sigma_bounds[0]:.2f} σ")
        print(f"  Z-score Max:              {self.sigma_bounds[1]:.2f} σ")

        # Calculate theoretical std
        if self.t_params["t_df"] > 2:
            sigma_t = self.t_params["t_scale"] * np.sqrt(self.t_params["t_df"] / (self.t_params["t_df"] - 2))
            print(f"\nTheoretical Std Dev (Student-t):  {sigma_t:.6f}")
        print("=" * 70)
