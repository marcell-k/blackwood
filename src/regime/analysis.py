import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from src.visualization.style import DEFAULT_STYLE


def analyze_regime_statistics(df: pd.DataFrame, regime_directions: dict[int, int] | None = None) -> None:
    r"""
    Comprehensive regime analysis with directional position control.
    Enhanced with transition matrix, tail risk metrics, and continuous equity curve.

    CHANGES FROM ORIGINAL:
    - Removed Plot 7 (Rolling Sharpe Ratio)
    - Changed Plots 10, 11 from boxplots to white scatter points with jitter
    - Fixed Plot 8 cumulative returns to show continuous equity curve
    - LATEST: Volatility moved to Row 2 Col 1, Continuous Equity spans full Row 3
    """
    # Default to long positions for all regimes if not specified
    if regime_directions is None:
        unique_regimes = sorted(df["regime"].unique())
        regime_directions = {int(r): 1 for r in unique_regimes}

    # Validate regime_directions
    for regime in df["regime"].unique():
        if int(regime) not in regime_directions:
            regime_directions[int(regime)] = 1  # Default to long

    df_work = df.assign(
        returns=df["Close"].pct_change(),
        regime_tradeable=df["regime"].shift(1),
    )

    # Calculate directional returns (vectorized)
    df_work = df_work.assign(
        direction=df_work["regime_tradeable"].map(regime_directions),
        directional_returns=lambda x: x["returns"] * x["direction"],
    )

    # Rolling volatility (annualized, 20-day window)
    df_work = df_work.assign(volatility=df_work["returns"].rolling(20).std() * np.sqrt(252))

    def calculate_max_drawdown_per_period(returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0 or returns.isna().all():
            return 0.0

        cum_returns = (1 + returns.fillna(0)).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def identify_regime_periods_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        r"""
        OPTIMIZED: Identify continuous regime periods using groupby aggregation.
        """
        regime_changes = (df["regime_tradeable"] != df["regime_tradeable"].shift(1)).cumsum()

        period_stats = (
            df.groupby(regime_changes)
            .agg(
                regime=("regime_tradeable", "first"),
                direction=("direction", "first"),
                start_date=("regime_tradeable", lambda x: x.index[0]),
                end_date=("regime_tradeable", lambda x: x.index[-1]),
                duration=("regime_tradeable", "size"),
            )
            .reset_index(drop=True)
        )

        mdd_by_period = (
            df.groupby(regime_changes)["directional_returns"]
            .apply(calculate_max_drawdown_per_period)
            .reset_index(drop=True)
        )

        period_stats["max_drawdown"] = mdd_by_period
        period_stats = period_stats[period_stats["regime"].notna()].reset_index(drop=True)

        return period_stats

    def calculate_regime_transition_matrix(df: pd.DataFrame, forward_periods: int = 5) -> pd.DataFrame:
        r"""
        Calculate regime transition probabilities.

        Transition probability: \( P_{ij} = P(R_{t+1} = j | R_t = i) \)

        Uses regime_tradeable throughout to avoid look-ahead in regime assignment.
        """
        df_transitions = df[["regime_tradeable"]].copy()
        df_transitions["next_regime"] = df_transitions["regime_tradeable"].shift(-1)

        # Remove rows with NaN regimes
        df_transitions = df_transitions.dropna(subset=["regime_tradeable", "next_regime"])

        # Get unique regimes (sorted)
        unique_regimes = sorted(df_transitions["regime_tradeable"].unique())

        # VECTORIZED: Count transitions using crosstab
        transition_counts = pd.crosstab(
            df_transitions["regime_tradeable"],
            df_transitions["next_regime"],
            normalize="index",  # Row-wise normalization gives probabilities
        )

        # Ensure all regimes are represented (fill missing with 0)
        for regime in unique_regimes:
            if regime not in transition_counts.index:
                transition_counts.loc[regime] = 0
            if regime not in transition_counts.columns:
                transition_counts[regime] = 0

        transition_counts = transition_counts.sort_index().sort_index(axis=1)

        return transition_counts

    def regime_statistics(df: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
        r"""
        Calculate comprehensive regime statistics with tail risk metrics.

        Tail Risk Metrics:
        - VaR (5%): \( \text{VaR}_{0.05} = Q_{0.05}(R) \) where Q is the quantile function
        - CVaR: \( \text{CVaR}_{0.05} = E[R | R \leq \text{VaR}_{0.05}] \)
        - Skewness: \( \gamma_1 = E\left[\left(\frac{R - \mu}{\sigma}\right)^3\right] \)
        - Kurtosis: \( \gamma_2 = E\left[\left(\frac{R - \mu}{\sigma}\right)^4\right] - 3 \) (excess kurtosis)
        """
        stats_list = []

        for regime in sorted(df["regime_tradeable"].dropna().unique()):
            regime_mask = df["regime_tradeable"] == regime
            regime_data = df[regime_mask]

            direction = regime_directions.get(int(regime), 1)
            direction_label = "LONG" if direction == 1 else "SHORT" if direction == -1 else "FLAT"

            # Directional returns for statistics
            dir_returns = regime_data["directional_returns"].dropna()

            # Return statistics (annualized)
            mean_return = dir_returns.mean() * 252
            std_return = dir_returns.std() * np.sqrt(252)

            # Volatility statistics
            mean_vol = regime_data["volatility"].mean()
            std_vol = regime_data["volatility"].std()

            # Maximum drawdown
            regime_periods = periods[periods["regime"] == regime]
            max_dd = regime_periods["max_drawdown"].min() if len(regime_periods) > 0 else 0.0

            # Regime duration
            avg_duration = regime_periods["duration"].mean() if len(regime_periods) > 0 else 0.0
            num_periods = len(regime_periods)

            # Tail risk metrics (vectorized)
            var_5pct = np.percentile(dir_returns, 5) * np.sqrt(252) if len(dir_returns) > 0 else 0.0

            var_threshold_daily = np.percentile(dir_returns, 5) if len(dir_returns) > 0 else 0.0
            tail_returns = dir_returns[dir_returns <= var_threshold_daily]
            cvar_5pct = tail_returns.mean() * np.sqrt(252) if len(tail_returns) > 0 else 0.0

            skewness = skew(dir_returns, nan_policy="omit") if len(dir_returns) > 2 else 0.0
            kurtosis_value = kurtosis(dir_returns, nan_policy="omit") if len(dir_returns) > 3 else 0.0

            stats_list.append(
                {
                    "Regime": regime,
                    "Direction": direction_label,
                    "Observations": regime_mask.sum(),
                    "Num_Periods": num_periods,
                    "Avg_Return_Ann": mean_return,
                    "Std_Return_Ann": std_return,
                    "Sharpe_Ratio": mean_return / std_return if std_return != 0 else 0,
                    "Avg_Volatility": mean_vol,
                    "Std_Volatility": std_vol,
                    "Max_Drawdown": max_dd,
                    "Avg_Duration_Days": avg_duration,
                    "VaR_5pct_Ann": var_5pct,
                    "CVaR_5pct_Ann": cvar_5pct,
                    "Skewness": skewness,
                    "Kurtosis": kurtosis_value,
                }
            )

        return pd.DataFrame(stats_list)

    def calculate_continuous_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Calculate continuous equity curve from directional returns.

        Formula: \( \text{Equity}_t = \prod_{i=1}^{t}(1 + r_i^{\text{dir}}) \)

        No resets at regime changes - shows true cumulative P&L.

        Args:
            df: Working DataFrame with directional_returns

        Returns:
            DataFrame with continuous_equity column
        """
        df_equity = df.copy()

        # VECTORIZED: Cumulative product of (1 + returns)
        # fillna(0) handles NaN returns (treat as 0% return, i.e., multiplier = 1)
        df_equity["continuous_equity"] = (1 + df_equity["directional_returns"].fillna(0)).cumprod()

        return df_equity

    # ============================================================================
    # EXECUTE ANALYSIS
    # ============================================================================

    periods = identify_regime_periods_vectorized(df_work)
    regime_stats = regime_statistics(df_work, periods)

    transition_probs = calculate_regime_transition_matrix(df_work, forward_periods=5)

    # Continuous equity curve
    df_equity = calculate_continuous_equity_curve(df_work)

    # ============================================================================
    # PRINT STATISTICS
    # ============================================================================

    print("\n" + "=" * 100)
    print("REGIME STATISTICS (DIRECTIONAL - NO LOOK-AHEAD BIAS)")
    print("=" * 100)
    print(regime_stats.to_string(index=False))

    print("\n" + "=" * 100)
    print("REGIME TRANSITION PROBABILITY MATRIX")
    print("=" * 100)
    print("Rows: Current Regime | Columns: Next Regime (1-bar forward)")
    print(transition_probs.to_string())

    # ============================================================================
    # VISUALIZATION SECTION - CORRECTED LAYOUT
    # ============================================================================

    fig = plt.figure(figsize=(20, 16))

    style = DEFAULT_STYLE  # Fixed typo

    colors = [
        "#00E5FF",  # Bright cyan - Regime 0
        "#FF6B9D",  # Bright pink - Regime 1
        "#69F0AE",  # Bright mint green - Regime 2
        "#FFD740",  # Bright amber - Regime 3
        "#B388FF",  # Bright purple - Regime 4
    ]

    price_line_color = "#E0E0E0"

    # ===== ROW 1: Regime Timeline, Price, Transition Heatmap =====

    # 1. Regime Timeline
    ax1 = plt.subplot(4, 3, 1)

    regime_colors_scatter = df_work["regime"].map(
        {0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3], 4: colors[4]}
    )
    ax1.scatter(df_work.index, df_work["regime"], c=regime_colors_scatter, alpha=0.8, s=15, edgecolors="none")
    ax1.set_ylabel("Regime", fontsize=11, fontweight="bold")
    ax1.set_title("Regime Classification Over Time", fontweight="bold", fontsize=12)
    ax1.set_yticks(sorted(df_work["regime"].unique()))

    for regime in sorted(df_work["regime"].unique()):
        direction = regime_directions.get(int(regime), 1)
        direction_text = "L" if direction == 1 else "S" if direction == -1 else "F"
        ax1.text(
            0.02,
            int(regime) / max(df_work["regime"]) + 0.05,
            direction_text,
            transform=ax1.transAxes,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=colors[int(regime)], edgecolor="white", linewidth=1.5, alpha=0.9),
        )

    style.apply_mpl(fig, ax1)  # APPLY STYLE AFTER ALL ELEMENTS

    # 2. Price with Regime Background
    ax2 = plt.subplot(4, 3, 2)

    ax2.plot(df_work.index, df_work["Close"], color=price_line_color, linewidth=2, label="Close Price", zorder=10)

    for regime in sorted(df_work["regime_tradeable"].dropna().unique()):
        regime_mask = df_work["regime_tradeable"] == regime
        if regime_mask.sum() > 0:
            direction = regime_directions.get(int(regime), 1)
            direction_label = "L" if direction == 1 else "S"
            ax2.fill_between(
                df_work.index,
                df_work["Close"].min(),
                df_work["Close"].max(),
                where=regime_mask,
                alpha=0.25,
                color=colors[int(regime)],
                label=f"R{int(regime)} ({direction_label})",
            )

    ax2.set_ylabel("Price", fontsize=11, fontweight="bold")
    ax2.set_title("Price with Tradeable Regime Overlay", fontweight="bold", fontsize=12)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    style.apply_mpl(fig, ax2)  # APPLY STYLE AFTER ALL ELEMENTS

    # 3. Transition Probability Heatmap
    ax3 = plt.subplot(4, 3, 3)

    im = ax3.imshow(transition_probs.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax3.set_xticks(np.arange(len(transition_probs.columns)))
    ax3.set_yticks(np.arange(len(transition_probs.index)))
    ax3.set_xticklabels([f"R{int(r)}" for r in transition_probs.columns])
    ax3.set_yticklabels([f"R{int(r)}" for r in transition_probs.index])

    # for i in range(len(transition_probs.index)):
    #     for j in range(len(transition_probs.columns)):
    #         text = ax3.text(
    #             j,
    #             i,
    #             f"{transition_probs.values[i, j]:.2f}",
    #             ha="center",
    #             va="center",
    #             color="white",
    #             fontweight="bold",
    #             fontsize=9,
    #         )

    ax3.set_xlabel("Next Regime", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Current Regime", fontsize=11, fontweight="bold")
    ax3.set_title("Regime Transition Probabilities", fontweight="bold", fontsize=12)

    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Probability", rotation=270, labelpad=15, fontweight="bold")

    style.apply_mpl(fig, ax3)  # APPLY STYLE AFTER ALL ELEMENTS
    # Additional colorbar styling
    cbar.ax.tick_params(colors=style.font_color)
    cbar.ax.yaxis.label.set_color(style.font_color)

    # ===== ROW 2: Volatility Distribution, VaR/CVaR, Distribution Shape =====

    # 4. Volatility Distribution
    ax4 = plt.subplot(4, 3, 4)

    for regime in sorted(df_work["regime"].unique()):
        regime_vol = df_work[df_work["regime"] == regime]["volatility"].dropna() * 100
        ax4.hist(
            regime_vol,
            bins=40,
            alpha=0.7,
            label=f"Regime {int(regime)}",
            color=colors[int(regime)],
            density=True,
            edgecolor=colors[int(regime)],
            linewidth=0.5,
        )

    ax4.set_xlabel("Annualized Volatility (%)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax4.set_title("Volatility Distribution by Regime", fontweight="bold", fontsize=12)
    ax4.legend(fontsize=8, framealpha=0.9)

    style.apply_mpl(fig, ax4)  # APPLY STYLE AFTER ALL ELEMENTS

    # 5. VaR and CVaR Comparison
    ax5 = plt.subplot(4, 3, 5)

    regimes_sorted = regime_stats["Regime"].values
    # var_values = regime_stats["VaR_5pct_Ann"].values * 100
    # cvar_values = regime_stats["CVaR_5pct_Ann"].values * 100
    #
    x_pos = np.arange(len(regimes_sorted))
    # width = 0.35

    # bars1 = ax5.bar(
    #     x_pos - width / 2,
    #     var_values,
    #     width,
    #     label="VaR (5%)",
    #     color=colors[0],
    #     alpha=0.8,
    #     edgecolor="white",
    #     linewidth=1.5,
    # )
    # bars2 = ax5.bar(
    #     x_pos + width / 2,
    #     cvar_values,
    #     width,
    #     label="CVaR (5%)",
    #     color=colors[1],
    #     alpha=0.8,
    #     edgecolor="white",
    #     linewidth=1.5,
    # )

    ax5.set_ylabel("Annualized Tail Risk (%)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Regime", fontsize=11, fontweight="bold")
    ax5.set_title("Tail Risk: VaR vs CVaR (5th Percentile)", fontweight="bold", fontsize=12)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f"R{int(r)}" for r in regimes_sorted])
    ax5.legend(fontsize=9, framealpha=0.9)
    ax5.axhline(0, linestyle="--", linewidth=1, alpha=0.5)

    style.apply_mpl(fig, ax5)  # APPLY STYLE AFTER ALL ELEMENTS

    # 6. Skewness and Kurtosis
    ax6 = plt.subplot(4, 3, 6)

    skew_values = regime_stats["Skewness"].values
    kurt_values = regime_stats["Kurtosis"].values

    for i, regime in enumerate(regimes_sorted):
        direction = regime_directions.get(int(regime), 1)
        direction_label = "L" if direction == 1 else "S"
        ax6.scatter(
            skew_values[i],
            kurt_values[i],
            s=200,
            color=colors[int(regime)],
            alpha=0.8,
            edgecolors="white",
            linewidth=2,
            label=f"R{int(regime)} ({direction_label})",
        )

        ax6.text(
            skew_values[i],
            kurt_values[i],
            f"R{int(regime)}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax6.set_xlabel("Skewness", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Excess Kurtosis", fontsize=11, fontweight="bold")
    ax6.set_title("Return Distribution Shape by Regime", fontweight="bold", fontsize=12)
    ax6.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
    ax6.axvline(0, linestyle="--", linewidth=1, alpha=0.5)
    ax6.legend(fontsize=8, framealpha=0.9, loc="best")
    ax6.grid(True, alpha=0.3)

    style.apply_mpl(fig, ax6)  # APPLY STYLE AFTER ALL ELEMENTS

    # ===== ROW 3: CONTINUOUS EQUITY CURVE SPANNING ALL 3 COLUMNS =====

    # 7. Continuous Equity Curve
    ax7 = plt.subplot(4, 3, (7, 9))

    for period_id in periods.index:
        period = periods.loc[period_id]
        regime = period["regime"]

        period_mask = (df_equity.index >= period["start_date"]) & (df_equity.index <= period["end_date"])
        period_data = df_equity[period_mask]

        if len(period_data) > 0 and pd.notna(regime):
            ax7.plot(
                period_data.index, period_data["continuous_equity"], color=colors[int(regime)], linewidth=2.5, alpha=0.9
            )

    ax7.set_ylabel("Cumulative Equity", fontsize=12, fontweight="bold")
    ax7.set_title("Continuous Equity Curve by Regime", fontweight="bold", fontsize=13)
    ax7.axhline(1, linestyle="--", linewidth=1.5, alpha=0.6, label="Breakeven")
    ax7.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch

    legend_elements = []
    for r in sorted(df_equity["regime_tradeable"].dropna().unique()):
        direction = regime_directions.get(int(r), 1)
        direction_label = "LONG" if direction == 1 else "SHORT"
        legend_elements.append(
            Patch(facecolor=colors[int(r)], edgecolor="white", label=f"Regime {int(r)} ({direction_label})")
        )
    ax7.legend(handles=legend_elements, fontsize=9, framealpha=0.9, loc="best", ncol=len(legend_elements))

    style.apply_mpl(fig, ax7)  # APPLY STYLE AFTER ALL ELEMENTS

    # ===== ROW 4: WHITE SCATTER PLOTS =====

    # 10. Return Distribution
    ax10 = plt.subplot(4, 3, 10)

    unique_regimes = sorted(df_work["regime_tradeable"].dropna().unique())
    np.random.seed(42)

    for i, regime in enumerate(unique_regimes):
        regime_returns = df_work[df_work["regime_tradeable"] == regime]["directional_returns"].dropna() * 100

        if len(regime_returns) > 0:
            jitter = np.random.uniform(-0.2, 0.2, size=len(regime_returns))
            x_positions = np.full(len(regime_returns), i) + jitter

            ax10.scatter(x_positions, regime_returns, color="white", edgecolors="black", alpha=0.6, s=20, linewidth=0.5)

            median_val = regime_returns.median()
            ax10.plot(
                [i - 0.3, i + 0.3],
                [median_val, median_val],
                color=colors[int(regime)],
                linewidth=3,
                alpha=0.9,
                zorder=10,
            )

            mean_val = regime_returns.mean()
            ax10.scatter(
                [i],
                [mean_val],
                color=colors[int(regime)],
                marker="D",
                s=100,
                edgecolors="white",
                linewidth=2,
                zorder=11,
            )

    direction_labels = []
    for r in unique_regimes:
        direction = regime_directions.get(int(r), 1)
        direction_label = "L" if direction == 1 else "S"
        direction_labels.append(f"R{int(r)} ({direction_label})")

    ax10.set_xticks(range(len(unique_regimes)))
    ax10.set_xticklabels(direction_labels)
    ax10.set_ylabel("Daily Directional Returns (%)", fontsize=11, fontweight="bold")
    ax10.set_title("Directional Return Distribution (Median=Line, Mean=Diamond)", fontweight="bold", fontsize=12)
    ax10.axhline(0, linestyle="--", linewidth=1.5, alpha=0.6)
    ax10.grid(True, alpha=0.3, axis="y")

    style.apply_mpl(fig, ax10)  # APPLY STYLE AFTER ALL ELEMENTS

    # 11. Drawdown Distribution
    ax11 = plt.subplot(4, 3, 11)

    np.random.seed(43)

    regimes_with_dd = []
    for _, regime in enumerate(unique_regimes):
        regime_period_mdd = periods[periods["regime"] == regime]["max_drawdown"].values * 100

        if len(regime_period_mdd) > 0:
            regimes_with_dd.append(regime)

            jitter = np.random.uniform(-0.2, 0.2, size=len(regime_period_mdd))
            x_positions = np.full(len(regime_period_mdd), len(regimes_with_dd) - 1) + jitter

            ax11.scatter(
                x_positions, regime_period_mdd, color="white", edgecolors="black", alpha=0.6, s=20, linewidth=0.5
            )

            median_val = np.median(regime_period_mdd)
            ax11.plot(
                [len(regimes_with_dd) - 1 - 0.3, len(regimes_with_dd) - 1 + 0.3],
                [median_val, median_val],
                color=colors[int(regime)],
                linewidth=3,
                alpha=0.9,
                zorder=10,
            )

            mean_val = np.mean(regime_period_mdd)
            ax11.scatter(
                [len(regimes_with_dd) - 1],
                [mean_val],
                color=colors[int(regime)],
                marker="D",
                s=100,
                edgecolors="white",
                linewidth=2,
                zorder=11,
            )

    dd_labels = []
    for r in regimes_with_dd:
        direction = regime_directions.get(int(r), 1)
        direction_label = "L" if direction == 1 else "S"
        dd_labels.append(f"R{int(r)} ({direction_label})")

    ax11.set_xticks(range(len(regimes_with_dd)))
    ax11.set_xticklabels(dd_labels)
    ax11.set_ylabel("Max Drawdown per Period (%)", fontsize=11, fontweight="bold")
    ax11.set_title("Drawdown Distribution by Period (Median=Line, Mean=Diamond)", fontweight="bold", fontsize=12)
    ax11.axhline(0, linestyle="--", linewidth=1.5, alpha=0.6)
    ax11.grid(True, alpha=0.3, axis="y")

    style.apply_mpl(fig, ax11)  # APPLY STYLE AFTER ALL ELEMENTS

    # 12. Max Drawdown in Longest Period
    ax12 = plt.subplot(4, 3, 12)

    longest_period_mdd = []
    regime_labels_bar = []

    for regime in sorted(df_work["regime_tradeable"].dropna().unique()):
        regime_periods_subset = periods[periods["regime"] == regime]

        if len(regime_periods_subset) > 0:
            longest_period = regime_periods_subset.loc[regime_periods_subset["duration"].idxmax()]
            longest_period_mdd.append(longest_period["max_drawdown"] * 100)
            direction = regime_directions.get(int(regime), 1)
            direction_label = "L" if direction == 1 else "S"
            regime_labels_bar.append(f"R{int(regime)} ({direction_label})")

    bars = ax12.bar(
        regime_labels_bar,
        longest_period_mdd,
        color=colors[: len(longest_period_mdd)],
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
    )

    ax12.set_ylabel("Maximum Drawdown (%)", fontsize=11, fontweight="bold")
    ax12.set_title("Max DD in Longest Period", fontweight="bold", fontsize=12)

    for bar, val in zip(bars, longest_period_mdd, strict=True):
        height = bar.get_height()
        ax12.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="white", linewidth=0.5, alpha=0.7),
        )

    style.apply_mpl(fig, ax12)  # APPLY STYLE AFTER ALL ELEMENTS

    plt.tight_layout()
    plt.show()
