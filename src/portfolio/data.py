# ============================================================================
# Standard library
# ============================================================================
import glob
import itertools
import os

import matplotlib.pyplot as plt

# ============================================================================
# Third-party libraries
# ============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression

# ============================================================================
# Local application imports
# ============================================================================
from src.visualization.style import DEFAULT_STYLE


def load_strategy_files(
    directory_path: str, start_date: str | pd.Timestamp | None = None
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """
    Load all strategy equity and trades files from directory with optional date filtering

    Parameters
    ----------
    directory_path : str
        Path to directory containing *_equity.csv and *_trades.csv files
    start_date : str or pd.Timestamp, optional
        Filter data to only include dates >= start_date
        Examples: '2023-01-01', '2024-06-15', pd.Timestamp('2023-01-01')

    Returns
    -------
    equity_dict : Dict[str, pd.Series]
        Strategy name -> equity curve (datetime indexed, filtered if start_date provided)
    trades_dict : Dict[str, pd.DataFrame]
        Strategy name -> trades DataFrame (filtered if start_date provided)

    """
    equity_dict = {}
    trades_dict = {}

    # Convert start_date to timezone-aware Timestamp if provided
    if start_date is not None:
        start_date = pd.Timestamp(start_date, tz="UTC")
        print(f"Filtering data from: {start_date}\n")

    # Get all equity CSV files
    equity_files = glob.glob(os.path.join(directory_path, "*_equity.csv"))

    print(f"Found {len(equity_files)} equity files\n")

    for eq_file in equity_files:
        # Extract strategy name from filename
        strategy_name = os.path.basename(eq_file).replace("_equity.csv", "")

        # Load equity data
        eq_df = pd.read_csv(eq_file)

        # Auto-detect date and equity columns
        date_col = eq_df.columns[0]
        equity_col = eq_df.columns[1]

        # Convert to datetime with UTC to handle mixed timezones
        eq_df[date_col] = pd.to_datetime(eq_df[date_col], utc=True)
        equity_series = pd.Series(eq_df[equity_col].values, index=eq_df[date_col], name=strategy_name).sort_index()

        # Filter equity by start_date if provided
        if start_date is not None:
            original_len = len(equity_series)
            equity_series = equity_series[equity_series.index >= start_date]
            filtered_len = len(equity_series)
            # print(f"  {strategy_name} equity: {original_len:,} -> {filtered_len:,} points")

        equity_dict[strategy_name] = equity_series

        # Load corresponding trades file
        trades_file = eq_file.replace("_equity.csv", "_trades.csv")

        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)

            # Convert date columns to datetime with UTC
            for col in trades_df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    trades_df[col] = pd.to_datetime(trades_df[col], utc=True, errors="coerce")

            # Filter trades by EntryTime if start_date provided
            if start_date is not None and "EntryTime" in trades_df.columns:
                original_trades = len(trades_df)
                trades_df = trades_df[trades_df["EntryTime"] >= start_date].reset_index(drop=True)
                filtered_trades = len(trades_df)
                # print(f"  {strategy_name} trades: {original_trades} -> {filtered_trades} trades")

            trades_dict[strategy_name] = trades_df

            # if start_date is None:
            # print(f"✓ {strategy_name}: {len(equity_series):,} equity points, {len(trades_df)} trades")
        else:
            print(f"✗ {strategy_name}: {len(equity_series):,} equity points, NO TRADES FILE")

    return equity_dict, trades_dict


def normalize_equity_curves(
    equity_dict: dict[str, pd.Series],
    target_vol: float,
    lookback_days: int = 252,
) -> dict[str, pd.Series]:
    """
    Normalize equity curves to target annual volatility.

    Parameters
    ----------
    equity_dict : Dict[str, pd.Series]
        Strategy name → equity curve (UTC-aware index)
    target_vol : float
        Target annual volatility (e.g., 0.10 for 10%)
    lookback_days : int
        Rolling window for volatility calculation

    Returns
    -------
    normalized_equity : Dict[str, pd.Series]
        Strategy name → normalized equity curve (UTC-aware index)

    """
    normalized_equity_dict = {}

    for strategy_name, equity_series in equity_dict.items():
        # Calculate log returns
        log_returns = np.log(equity_series / equity_series.shift(1)).dropna()

        # Calculate rolling annualized volatility
        rolling_vol = log_returns.rolling(window=lookback_days).std() * np.sqrt(252)

        # Floor vol to avoid division by zero
        rolling_vol = rolling_vol.clip(lower=1e-6)

        # Calculate scaling factors
        scale_factors = target_vol / rolling_vol

        # Apply scaling to log returns
        scaled_log_returns = log_returns * scale_factors

        # Reconstruct normalized equity curve
        first_valid_idx = scaled_log_returns.index[0]
        initial_equity = equity_series.loc[:first_valid_idx].iloc[-1]

        normalized_equity = initial_equity * np.exp(scaled_log_returns.cumsum())

        # Align with original index (forward-fill warm-up period)
        normalized_equity = pd.Series(
            data=normalized_equity.values, index=scaled_log_returns.index, name=equity_series.name
        )
        normalized_equity = normalized_equity.reindex(equity_series.index, method="ffill")

        normalized_equity_dict[strategy_name] = normalized_equity

    return normalized_equity_dict


class EntropyCore:
    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    # ========================================================================
    # NUMBA-OPTIMIZED STATIC METHODS (Hot Path Functions)
    # ========================================================================

    @staticmethod
    @njit
    def _uniform_bin_1d(arr: np.ndarray, n_bins: int) -> np.ndarray:
        x_min = arr.min()
        x_max = arr.max()

        # Edge case: constant array (e.g., all zeros)
        if x_max == x_min:
            return np.zeros(len(arr), dtype=np.int64)

        bin_width = (x_max - x_min) / n_bins
        bins = np.floor((arr - x_min) / bin_width).astype(np.int64)

        # Clamp upper bin edge (floating point precision)
        bins[bins >= n_bins] = n_bins - 1
        return bins

    @staticmethod
    @njit
    def _entropy_from_bins(binned: np.ndarray, n_bins: int) -> float:
        """
        Shannon entropy H(X) from pre-binned discrete data.
        """
        counts = np.bincount(binned, minlength=n_bins)
        probs = counts / binned.size

        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * np.log2(p)
        return h

    @staticmethod
    @njit
    def _joint_entropy_from_bins(x_bins: np.ndarray, y_bins: np.ndarray, n_bins: int) -> float:
        """
        Joint entropy H(X,Y) using 2D histogram encoding.
        """
        joint_idx = x_bins * n_bins + y_bins
        counts = np.bincount(joint_idx, minlength=n_bins * n_bins)
        probs = counts / x_bins.size

        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * np.log2(p)
        return h

    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================

    def discrete_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Mutual Information using uniform binning (deterministic, O(n)).
        """
        x_binned = self._uniform_bin_1d(x, self.n_bins)
        y_binned = self._uniform_bin_1d(y, self.n_bins)

        h_x = self._entropy_from_bins(x_binned, self.n_bins)
        h_y = self._entropy_from_bins(y_binned, self.n_bins)
        h_xy = self._joint_entropy_from_bins(x_binned, y_binned, self.n_bins)

        mi = h_x + h_y - h_xy
        return max(0.0, mi)  # Numerical stability (should never be negative)

    def rolling_mi(self, x: pd.Series, y: pd.Series, window: int, stride: int = 1) -> pd.Series:
        """
        Vectorized rolling mutual information.
        """
        x_arr = x.values
        y_arr = y.values
        dates = x.index

        mi_vals = []
        mi_idx = []

        for end in range(window, len(x_arr) + 1, stride):
            start = end - window
            mi = self.discrete_mi(x_arr[start:end], y_arr[start:end])
            mi_vals.append(mi)
            mi_idx.append(dates[end - 1])  # Window end date

        return pd.Series(mi_vals, index=mi_idx, name=f"MI_{x.name}_vs_{y.name}")

    def transfer_entropy(self, source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """
        Transfer Entropy: TE(source → target) measuring directional information flow.
        """
        # Align time series with lag
        y_t = target[lag:]
        y_tm1 = target[:-lag]
        x_tm1 = source[:-lag]

        # Discretize each variable
        y_t_b = self._quantile_bin(y_t)
        y_tm1_b = self._quantile_bin(y_tm1)
        x_tm1_b = self._quantile_bin(x_tm1)

        # Compute entropies
        h_yy = self._joint_entropy_discrete(y_t_b, y_tm1_b)
        h_yx = self._joint_entropy_discrete(y_tm1_b, x_tm1_b)
        h_y = self._entropy_discrete(y_tm1_b)
        h_yyx = self._joint_entropy_3d(y_t_b, y_tm1_b, x_tm1_b)

        te = h_yy + h_yx - h_y - h_yyx
        return max(0.0, te)

    # ========================================================================
    # HELPER METHODS (TE Support)
    # ========================================================================

    def _quantile_bin(self, arr: np.ndarray) -> np.ndarray:
        """Quantile-based binning with duplicate handling."""
        quantiles = np.percentile(arr, np.linspace(0, 100, self.n_bins + 1))
        if np.allclose(quantiles, quantiles[0]):  # All values identical
            return np.zeros(len(arr), dtype=np.int64)
        bins = np.searchsorted(quantiles[1:-1], arr, side="right")
        return bins.astype(np.int64)

    def _entropy_discrete(self, labels: np.ndarray) -> float:
        """Shannon entropy for discrete labels using scipy."""
        if labels.size == 0:
            return 0.0
        counts = np.bincount(labels)
        probs = counts[counts > 0] / labels.size
        return entropy(probs, base=2)

    def _joint_entropy_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """Joint entropy H(X,Y) for discrete arrays."""
        joint_idx = x * self.n_bins + y
        return self._entropy_discrete(joint_idx)

    def _joint_entropy_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Joint entropy H(X,Y,Z) for 3 discrete variables."""
        encoded = x * (self.n_bins**2) + y * self.n_bins + z
        return self._entropy_discrete(encoded)


class RelationshipAnalyzer:
    """
    Orchestrates information-theoretic analysis with production-grade visualizations.

    **Features**:
    - Static MI matrix (global dependencies)
    - Rolling MI/correlation (regime detection)
    - Transfer entropy (directional causality)
    - Correlation heatmaps (diversification breakdown)

    **Usage Pattern**:
        analyzer = RelationshipAnalyzer(df, columns=['SPX', 'VIX', 'Gold'])
        analyzer.run_full_analysis(window=252, stride=5)
    """

    def __init__(self, df: pd.DataFrame, columns: list[str], n_bins: int = 5, entropy_core: EntropyCore | None = None):
        """
        Args:
            df: DataFrame with datetime index and numeric columns
            columns: Column names to analyze
            n_bins: Discretization bins for MI/TE (5=quintiles, 10=deciles)
            entropy_core: Custom EntropyCore instance (optional, uses default if None)

        """
        self.df = df[columns].dropna()
        self.columns = columns
        self.core = entropy_core or EntropyCore(n_bins=n_bins)
        self.pairs = list(itertools.combinations(columns, 2))

    # ========================================================================
    # COMPUTATION METHODS
    # ========================================================================

    def compute_static_mi_matrix(self, method: str = "discrete") -> pd.DataFrame:
        """
        Computes mutual information matrix for all column pairs.
        """
        mi_results = []

        for col_a, col_b in self.pairs:
            x = self.df[col_a].values
            y = self.df[col_b].values

            if method == "discrete":
                mi = self.core.discrete_mi(x, y)
            elif method == "knn":
                mi = self._knn_mi(x, y, n_seeds=30)
            else:
                raise ValueError(f"Unknown method: {method}")

            mi_results.append({"Pair": f"{col_a} vs {col_b}", "MI": mi, "Correlation": np.corrcoef(x, y)[0, 1]})

        return pd.DataFrame(mi_results).sort_values("MI", ascending=False)

    def compute_rolling_analysis(self, window: int = 252, stride: int = 5) -> dict[str, pd.DataFrame]:
        """
        Computes rolling MI and correlation for all pairs.
        """
        mi_data = {}
        corr_data = {}

        for col_a, col_b in self.pairs:
            pair_name = f"{col_a}_vs_{col_b}"

            # Rolling MI
            mi_series = self.core.rolling_mi(self.df[col_a], self.df[col_b], window=window, stride=stride)
            mi_data[pair_name] = mi_series

            # Rolling correlation
            corr_series = self.df[col_a].rolling(window).corr(self.df[col_b])
            corr_data[pair_name] = corr_series

        return {"mi": pd.DataFrame(mi_data), "corr": pd.DataFrame(corr_data)}

    def compute_transfer_entropy_matrix(self, window: int = 126, lag: int = 1) -> pd.DataFrame:
        """
        Computes rolling transfer entropy for all pairs (bidirectional).
        """
        results = []

        for col_a, col_b in self.pairs:
            src = self.df[col_a].values
            tgt = self.df[col_b].values

            # Rolling TE in both directions
            te_fwd_vals = []
            te_bwd_vals = []

            for i in range(window, len(src)):
                s = slice(i - window, i)
                te_fwd_vals.append(self.core.transfer_entropy(src[s], tgt[s], lag))
                te_bwd_vals.append(self.core.transfer_entropy(tgt[s], src[s], lag))

            results.append(
                {"Source": col_a, "Target": col_b, "TE_mean": np.mean(te_fwd_vals), "TE_std": np.std(te_fwd_vals)}
            )
            results.append(
                {"Source": col_b, "Target": col_a, "TE_mean": np.mean(te_bwd_vals), "TE_std": np.std(te_bwd_vals)}
            )

        return pd.DataFrame(results).sort_values("TE_mean", ascending=False)

    # ========================================================================
    # VISUALIZATION METHODS (Return Figures, Don't Call plt.show())
    # ========================================================================
    def _apply_style(self, fig: plt.Figure):
        DEFAULT_STYLE.apply_mpl(fig=fig)
        return fig

    def plot_static_correlation_heatmap(self) -> plt.Figure:
        """
        Global correlation heatmap for full period.

        Returns:
            matplotlib Figure object

        """
        fig, ax = plt.subplots(figsize=(10, 8))

        corr_matrix = self.df.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"label": "Correlation Coefficient"},
        )

        ax.set_title("Static Correlation Matrix (Full Period)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._apply_style(fig)

    def plot_rolling_comparison(self, rolling_data: dict[str, pd.DataFrame], window: int) -> plt.Figure:
        """
        2-column plot: Rolling correlation vs Rolling MI for each pair.

        Args:
            rolling_data: Output from compute_rolling_analysis()
            window: Window size (for title)

        Returns:
            matplotlib Figure object

        """
        mi_df = rolling_data["mi"]
        corr_df = rolling_data["corr"]

        n_pairs = len(self.pairs)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(18, 4 * n_pairs), sharex=True)

        # Handle single pair case
        if n_pairs == 1:
            axes = axes[None, :]

        for i, (col_a, col_b) in enumerate(self.pairs):
            pair_name = f"{col_a}_vs_{col_b}"

            # Left panel: Rolling correlation
            corr_series = corr_df[pair_name].dropna()
            axes[i, 0].plot(corr_series, color="steelblue", linewidth=1.5)
            axes[i, 0].axhline(0, linestyle="--", color="black", alpha=0.5, linewidth=1)
            axes[i, 0].axhline(0.5, linestyle=":", color="gray", alpha=0.3)
            axes[i, 0].axhline(-0.5, linestyle=":", color="gray", alpha=0.3)
            axes[i, 0].set_title(f"Rolling Correlation ({window}d): {col_a} vs {col_b}")
            axes[i, 0].set_ylabel("Correlation")
            axes[i, 0].set_ylim(-1, 1)
            axes[i, 0].grid(alpha=0.3)

            # Right panel: Rolling MI
            mi_series = mi_df[pair_name].dropna()
            axes[i, 1].plot(mi_series, alpha=0.7, label="MI", color="darkorange", linewidth=1.5)
            axes[i, 1].plot(mi_series.rolling(20).mean(), linestyle="--", label="MA(20)", color="darkred", linewidth=2)
            axes[i, 1].set_title(f"Rolling MI ({self.core.n_bins} bins): {col_a} vs {col_b}")
            axes[i, 1].set_ylabel("MI (bits)")
            axes[i, 1].legend()
            axes[i, 1].grid(alpha=0.3)

        plt.tight_layout()
        return self._apply_style(fig)

    def plot_correlation_regime_heatmap(self, window: int = 252, top_n: int = 10) -> plt.Figure:
        """
        Dual-panel: (1) Top N most variable rolling correlations, (2) Heatmap over time.
        """
        # Compute all rolling correlations
        rolling_corrs = {}
        for col_a, col_b in self.pairs:
            corr = self.df[col_a].rolling(window).corr(self.df[col_b])
            rolling_corrs[f"{col_a} vs {col_b}"] = corr

        corr_df = pd.DataFrame(rolling_corrs).dropna()

        # Identify most variable pairs
        corr_std = corr_df.std().sort_values(ascending=False)
        top_pairs = corr_std.head(top_n).index.tolist()

        # Create figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.3)

        # ====================================================================
        # Panel 1: Top N Most Variable Pairs (Line Plot)
        # ====================================================================
        ax1 = fig.add_subplot(gs[0])

        for label in top_pairs:
            ax1.plot(corr_df[label], label=label, alpha=0.8, linewidth=2)

        ax1.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
        ax1.axhline(-0.5, color="gray", linestyle=":", alpha=0.3)

        ax1.set_title(
            f"Top {top_n} Most Variable Rolling Correlations (Window={window} days)", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Correlation Coefficient", fontsize=12)
        ax1.set_ylim(-1.05, 1.05)
        ax1.legend(loc="upper left", fontsize=9, ncol=2)
        ax1.grid(alpha=0.3)

        # ====================================================================
        # Panel 2: Heatmap of All Correlations Over Time
        # ====================================================================
        ax2 = fig.add_subplot(gs[1])

        # Downsample for visualization
        corr_df_sampled = corr_df.iloc[::20, :]

        im = ax2.imshow(
            corr_df_sampled.T.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
        )

        # X-axis: dates
        x_positions = np.arange(0, len(corr_df_sampled), max(1, len(corr_df_sampled) // 10))
        x_labels = [corr_df_sampled.index[i].strftime("%Y-%m") for i in x_positions]
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, rotation=45, ha="right")

        # Y-axis: pair names (show subset to avoid crowding)
        n_pairs = len(corr_df_sampled.columns)
        y_step = max(1, n_pairs // 20)
        y_positions = np.arange(0, n_pairs, y_step)
        y_labels = [corr_df_sampled.columns[i] for i in y_positions]
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(y_labels, fontsize=7)

        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Asset Pairs", fontsize=12)
        ax2.set_title("Correlation Heatmap Over Time", fontsize=12, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation", rotation=270, labelpad=20)

        plt.tight_layout()
        return self._apply_style(fig)

    def plot_transfer_entropy(self, window: int = 126, lag: int = 1) -> plt.Figure:
        """
        Plots bidirectional transfer entropy for first pair (can extend to all).

        Args:
            window: Rolling TE window
            lag: Prediction lag

        Returns:
            matplotlib Figure with 2 subplots (directional TE, net flow)

        """
        # Use first pair as example (extend to all pairs if needed)
        col_a, col_b = self.pairs[0]
        src = self.df[col_a].values
        tgt = self.df[col_b].values

        # Compute rolling TE
        te_fwd = []
        te_bwd = []
        dates = []

        for i in range(window, len(src)):
            s = slice(i - window, i)
            te_fwd.append(self.core.transfer_entropy(src[s], tgt[s], lag))
            te_bwd.append(self.core.transfer_entropy(tgt[s], src[s], lag))
            dates.append(self.df.index[i])

        te_fwd_series = pd.Series(te_fwd, index=dates)
        te_bwd_series = pd.Series(te_bwd, index=dates)
        te_net = te_fwd_series - te_bwd_series

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Panel 1: Directional TE
        ax1.plot(te_fwd_series.rolling(10).mean(), label=f"{col_a} → {col_b}", linewidth=2)
        ax1.plot(te_bwd_series.rolling(10).mean(), label=f"{col_b} → {col_a}", linewidth=2)
        ax1.set_title(f"Directional Transfer Entropy (Window={window}, Lag={lag})")
        ax1.set_ylabel("TE (bits)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Panel 2: Net flow
        net_sm = te_net.rolling(10).mean()
        ax2.fill_between(net_sm.index, net_sm, 0, where=net_sm >= 0, alpha=0.3, color="green")
        ax2.fill_between(net_sm.index, net_sm, 0, where=net_sm < 0, alpha=0.3, color="red")
        ax2.axhline(0, linestyle="--", alpha=0.5, color="black")
        ax2.set_title("Net Information Flow (Positive = Forward Dominance)")
        ax2.set_ylabel("Net TE (bits)")
        ax2.set_xlabel("Date")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return self._apply_style(fig)

    # ========================================================================
    # ORCHESTRATION
    # ========================================================================

    def run_full_analysis(self, window: int = 252, stride: int = 5, show_plots: bool = True) -> dict[str, any]:
        """
        Runs complete analysis pipeline and generates all visualizations.

        **Workflow**:
        1. Static correlation heatmap
        2. Static MI matrix
        3. Rolling MI vs correlation comparison
        4. Correlation regime heatmap
        5. Transfer entropy (optional, expensive)
        """
        print("=" * 70)
        print("INFORMATION THEORY ANALYSIS")
        print("=" * 70)

        results = {}

        # ====================================================================
        # 1. Static Correlation
        # ====================================================================
        print("\n[1/5] Computing static correlation heatmap...")
        fig_corr = self.plot_static_correlation_heatmap()
        results["fig_static_corr"] = fig_corr
        if show_plots:
            plt.show()

        # ====================================================================
        # 2. Static MI
        # ====================================================================
        print("[2/5] Computing static mutual information matrix...")
        mi_df = self.compute_static_mi_matrix(method="discrete")
        results["static_mi"] = mi_df
        print("\n=== Static Mutual Information ===")
        print(mi_df.to_string(index=False))

        # ====================================================================
        # 3. Rolling Analysis
        # ====================================================================
        print(f"\n[3/5] Computing rolling analysis (window={window}, stride={stride})...")
        rolling_data = self.compute_rolling_analysis(window=window, stride=stride)
        results["rolling_data"] = rolling_data

        fig_rolling = self.plot_rolling_comparison(rolling_data, window)
        results["fig_rolling"] = fig_rolling
        if show_plots:
            plt.show()

        # ====================================================================
        # 4. Correlation Regime Heatmap
        # ====================================================================
        print("[4/5] Generating correlation regime heatmap...")
        fig_regime = self.plot_correlation_regime_heatmap(window=window, top_n=10)
        results["fig_regime"] = fig_regime
        if show_plots:
            plt.show()

        # Print regime statistics
        corr_df = rolling_data["corr"]
        corr_std = corr_df.std().sort_values(ascending=False)

        print("\n=== Correlation Variability (Top 10 Pairs) ===")
        variability_df = pd.DataFrame(
            {
                "Pair": corr_std.head(10).index,
                "Std Dev": corr_std.head(10).values,
                "Current": corr_df[corr_std.head(10).index].iloc[-1].values,
                "Mean": corr_df[corr_std.head(10).index].mean().values,
            }
        )
        print(variability_df.to_string(index=False))

        # ====================================================================
        # 5. Transfer Entropy (Optional - Expensive)
        # ====================================================================
        print("\n[5/5] Computing transfer entropy (this may take time)...")
        te_df = self.compute_transfer_entropy_matrix(window=window // 2)
        results["transfer_entropy"] = te_df
        print("\n=== Transfer Entropy Matrix ===")
        print(te_df.head(10).to_string(index=False))

        if len(self.pairs) > 0:
            fig_te = self.plot_transfer_entropy(window=window // 2)
            results["fig_te"] = fig_te
            if show_plots:
                plt.show()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)

        return results

    # ========================================================================
    # UTILITY
    # ========================================================================

    @staticmethod
    def _knn_mi(x: np.ndarray, y: np.ndarray, n_seeds: int = 30) -> float:
        """
        kNN-based MI with random seed averaging (sklearn wrapper).

        **Performance**: ~50x slower than discrete MI.
        Use only when high accuracy is needed for non-uniform distributions.
        """
        X = x.reshape(-1, 1)
        return np.mean([mutual_info_regression(X, y, random_state=s)[0] for s in range(n_seeds)])
