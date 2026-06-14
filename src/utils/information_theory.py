import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
from src.config import RANDOM_STATE


class InformationTheory:
    """
    Centralized information-theoretic calculations for trading features.

    Provides:
        - Shannon entropy
        - Mutual information (MI)
        - Normalized mutual information (NMI)
        - Kullback-Leibler divergence
        - Jensen-Shannon divergence
    """

    def __init__(self, random_state: int = RANDOM_STATE, n_neighbors: int = 5):
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    @staticmethod
    def compute_entropy(data: np.ndarray, bins: int | None = None, base: float = 2.0) -> float:
        """Shannon entropy: H(X) = -Σ p(x) log p(x)"""
        if len(data) == 0:
            return 0.0

        bins = bins or int(np.sqrt(len(data)))
        counts, _ = np.histogram(data, bins=bins)
        counts = counts[counts > 0]

        probs = counts / counts.sum()
        return entropy(probs, base=base)

    @staticmethod
    def compute_joint_entropy(X: np.ndarray, Y: np.ndarray, bins: int | None = None, base: float = 2.0) -> float:
        """Joint entropy: H(X,Y) = -ΣΣ p(x,y) log p(x,y)"""
        if len(X) != len(Y) or len(X) == 0:
            return 0.0

        bins = bins or int(np.sqrt(len(X)))
        hist, _, _ = np.histogram2d(X, Y, bins=bins)
        hist = hist[hist > 0]

        probs = hist / hist.sum()
        return entropy(probs, base=base)

    def compute_mutual_information(self, X: np.ndarray, Y: np.ndarray, bins: int | None = None) -> float:
        """Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)"""
        H_X = self.compute_entropy(X, bins=bins)
        H_Y = self.compute_entropy(Y, bins=bins)
        H_XY = self.compute_joint_entropy(X, Y, bins=bins)

        return max(0.0, H_X + H_Y - H_XY)

    def compute_normalized_mutual_information(self, X: np.ndarray, Y: np.ndarray, bins: int | None = None) -> float:
        """NMI: 2*I(X;Y) / (H(X) + H(Y)) - Returns value in [0,1]"""
        H_X = self.compute_entropy(X, bins=bins)
        H_Y = self.compute_entropy(Y, bins=bins)

        if H_X + H_Y == 0:
            return 1.0 if np.array_equal(X, Y) else 0.0

        MI = self.compute_mutual_information(X, Y, bins=bins)
        return 2.0 * MI / (H_X + H_Y)

    @staticmethod
    def compute_kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Kullback-Leibler: KL(P||Q) = Σ P(i) log(P(i)/Q(i))"""
        P = np.asarray(P) + epsilon
        Q = np.asarray(Q) + epsilon
        P = P / P.sum()
        Q = Q / Q.sum()

        return np.sum(P * np.log(P / Q))

    @staticmethod
    def compute_jensen_shannon_divergence(P: np.ndarray, Q: np.ndarray, base: float = 2.0) -> float:
        """Jensen-Shannon Divergence - Symmetric, bounded metric [0,1]"""
        return jensenshannon(P, Q, base=base)

    def compute_jsd_from_data(self, X: np.ndarray, Y: np.ndarray, bins: int | None = None) -> float:
        """Compute JSD between two datasets by binning into distributions."""
        bins = bins or int(np.sqrt(min(len(X), len(Y))))

        all_data = np.concatenate([X, Y])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)

        P, _ = np.histogram(X, bins=bin_edges, density=True)
        Q, _ = np.histogram(Y, bins=bin_edges, density=True)

        P = P + 1e-10
        Q = Q + 1e-10
        P = P / P.sum()
        Q = Q / Q.sum()

        return self.compute_jensen_shannon_divergence(P, Q)

    def compute_nmi_matrix(self, X: pd.DataFrame, rank_gaussian: bool = True, min_len: int = 24) -> pd.DataFrame:
        """Compute pairwise NMI matrix using k-NN MI estimation."""
        Xc = X.select_dtypes(include=[np.number]).copy()
        valid_cols = [
            c
            for c in Xc.columns
            if (Xc[c].dropna().size >= min_len and not pd.isna(Xc[c].std(ddof=1)) and Xc[c].std(ddof=1) > 0)
        ]

        Xc = Xc[valid_cols].ffill().bfill()

        if Xc.isna().any().any():
            Xc = Xc.fillna(Xc.median(numeric_only=True))

        if rank_gaussian:
            Xc = rank_gaussian_transform(Xc)

        # Vectorized MI computation
        features = Xc.columns
        n = len(features)
        Z = Xc.values
        M = np.zeros((n, n), dtype=float)

        for j in range(n):
            y = Z[:, j]
            M[:, j] = mutual_info_regression(Z, y, n_neighbors=self.n_neighbors, random_state=self.random_state)

        # Symmetrize and normalize
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)

        bins = int(np.sqrt(len(Z)))
        entropies = np.array([self.compute_entropy(Z[:, i], bins=bins) for i in range(n)])

        entropy_sum = entropies[:, None] + entropies
        entropy_sum = np.where(entropy_sum > 0, entropy_sum, 1.0)
        M = 2.0 * M / entropy_sum
        M = np.clip(M, 0.0, 1.0)

        return pd.DataFrame(M, index=features, columns=features)


def rank_gaussian_transform(X: pd.DataFrame) -> pd.DataFrame:
    u = X.rank(method="average", pct=True).clip(1e-6, 1 - 1e-6)
    return pd.DataFrame(stats.norm.ppf(u), index=X.index, columns=X.columns)
