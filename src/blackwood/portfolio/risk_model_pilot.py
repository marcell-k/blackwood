from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

from blackwood.portfolio.core import PortfolioBacktester
from blackwood.portfolio.optimizer import MinimumVarianceStrategy
from blackwood.portfolio.risk_models import (
    EWMARiskModel,
    GARCH11RiskModel,
    RiskModel,
    SampleCovariance,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TRADING_DAYS = 252.0


@dataclass(frozen=True)
class PilotConfig:
    optimizer: str = "MinimumVarianceStrategy"
    rebalance_freq: list[str] = field(default_factory=lambda: ["W-FRI", "ME"])
    risk_model: list[str] = field(default_factory=lambda: ["sample", "ewma", "garch"])
    garch_horizon: list[int] = field(default_factory=lambda: [1, 5, 21])
    lookback_periods: int = 252
    use_denoising: bool = False
    seed: int = 89

    initial_capital: float = 10_000_000.0
    target_vol: float = 0.10
    apply_leverage: bool = False
    vol_method: str = "arithmetic"
    min_vol_threshold: float = 1e-4
    min_vol_floor: float = 0.05
    max_scale_factor: float = 10.0
    warmup_periods: int = 0
    ramp_periods: int = 0
    min_obs_for_full_trust: int = 0
    verbose: bool = False
    max_drawdown_worsen_tolerance: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilotConfig:
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(payload) - known_fields
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown config field(s): {unknown_text}")

        config = cls(**payload)
        config._validate()
        return config

    def _validate(self) -> None:
        if self.optimizer != "MinimumVarianceStrategy":
            raise ValueError("optimizer must be 'MinimumVarianceStrategy' for this pilot.")
        if not self.rebalance_freq:
            raise ValueError("rebalance_freq must contain at least one frequency.")
        if self.lookback_periods <= 0:
            raise ValueError("lookback_periods must be > 0.")
        if any(h <= 0 for h in self.garch_horizon):
            raise ValueError("garch_horizon values must be > 0.")
        if not self.risk_model:
            raise ValueError("risk_model must contain at least one model name.")
        unsupported = set(self.risk_model) - {"sample", "ewma", "garch"}
        if unsupported:
            unsupported_text = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported risk_model entries: {unsupported_text}")


@dataclass(frozen=True)
class PilotRunSpec:
    run_id: str
    rebalance_freq: str
    risk_model_name: str
    garch_horizon: int | None

    @property
    def run_label(self) -> str:
        if self.garch_horizon is None:
            return f"{self.rebalance_freq} | {self.risk_model_name}"
        return f"{self.rebalance_freq} | {self.risk_model_name}(h={self.garch_horizon})"


@dataclass(frozen=True)
class PilotArtifacts:
    metrics_csv: Path
    rank_csv: Path
    delta_csv: Path
    kpi_plot: Path
    garch_uplift_plot: Path
    summary_note: Path


def load_pilot_config(config_path: str | Path | None = None) -> PilotConfig:
    if config_path is None:
        cfg = PilotConfig()
        cfg._validate()
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text())
    else:
        raise ValueError("Config file must be JSON or YAML.")

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Config content must deserialize to a dictionary.")

    return PilotConfig.from_dict(payload)


def load_returns_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Returns CSV not found: {path}")

    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns CSV index must parse as DatetimeIndex.")
    if returns.empty:
        raise ValueError("Returns CSV is empty.")

    returns = returns.sort_index()
    returns = returns.apply(pd.to_numeric, errors="coerce")
    return returns


def build_pilot_run_specs(config: PilotConfig) -> list[PilotRunSpec]:
    runs: list[PilotRunSpec] = []
    for freq in config.rebalance_freq:
        if "sample" in config.risk_model:
            runs.append(
                PilotRunSpec(
                    run_id=f"{freq}__sample",
                    rebalance_freq=freq,
                    risk_model_name="sample",
                    garch_horizon=None,
                )
            )
        if "ewma" in config.risk_model:
            runs.append(
                PilotRunSpec(
                    run_id=f"{freq}__ewma",
                    rebalance_freq=freq,
                    risk_model_name="ewma",
                    garch_horizon=None,
                )
            )
        if "garch" in config.risk_model:
            for horizon in config.garch_horizon:
                runs.append(
                    PilotRunSpec(
                        run_id=f"{freq}__garch_h{horizon}",
                        rebalance_freq=freq,
                        risk_model_name="garch",
                        garch_horizon=horizon,
                    )
                )
    return runs


def _instantiate_risk_model(spec: PilotRunSpec) -> RiskModel:
    if spec.risk_model_name == "sample":
        return SampleCovariance()
    if spec.risk_model_name == "ewma":
        return EWMARiskModel(lam=0.94, ddof=0, init="sample")
    if spec.risk_model_name == "garch":
        if spec.garch_horizon is None:
            raise ValueError("GARCH spec missing horizon.")
        return GARCH11RiskModel(horizon=spec.garch_horizon, corr_source="sample")
    raise ValueError(f"Unsupported risk model name: {spec.risk_model_name}")


def _compute_rebalance_turnover(weights_history: pd.DataFrame) -> tuple[float, float, int]:
    if weights_history.empty:
        return np.nan, np.nan, 0

    ordered = weights_history.sort_index().fillna(0.0)
    if ordered.shape[0] < 2:
        return np.nan, np.nan, int(ordered.shape[0])

    rebalance_turnover = 0.5 * ordered.diff().abs().sum(axis=1).dropna()
    if rebalance_turnover.empty:
        return np.nan, np.nan, int(ordered.shape[0])

    return (
        float(rebalance_turnover.mean()),
        float(rebalance_turnover.sum()),
        int(ordered.shape[0]),
    )


def _build_metrics_row(
    spec: PilotRunSpec,
    risk_model: RiskModel,
    results: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    weights_history = results.get("weights_history", pd.DataFrame())
    avg_turnover, total_turnover, n_rebalances = _compute_rebalance_turnover(weights_history)

    weight_sum_error = np.nan
    has_nan_weights = False
    if isinstance(weights_history, pd.DataFrame) and not weights_history.empty:
        row_sums = weights_history.sum(axis=1)
        weight_sum_error = float((row_sums - 1.0).abs().max())
        has_nan_weights = bool(weights_history.isna().any().any())

    daily_returns = results["daily_returns"]
    oos_sharpe_annualized = np.nan
    daily_std = float(daily_returns.std(ddof=1))
    if daily_std > 0:
        oos_sharpe_annualized = float((daily_returns.mean() / daily_std) * np.sqrt(_TRADING_DAYS))

    row = {
        "run_id": spec.run_id,
        "run_label": spec.run_label,
        "rebalance_freq": spec.rebalance_freq,
        "risk_model": spec.risk_model_name,
        "risk_model_class": type(risk_model).__name__,
        "garch_horizon": spec.garch_horizon,
        "use_denoising": False,
        "lookback_periods": results.get("lookback_periods_dict"),
        "primary_kpi_sharpe": oos_sharpe_annualized,
        "annualized_return": float(metrics["annualized_return"]),
        "annualized_volatility": float(metrics["annualized_volatility"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "avg_rebalance_turnover": avg_turnover,
        "total_rebalance_turnover": total_turnover,
        "rebalance_count": n_rebalances,
        "weight_sum_max_abs_error": weight_sum_error,
        "has_nan_weights": has_nan_weights,
    }
    return row


def _build_delta_table(
    metrics_df: pd.DataFrame,
    max_drawdown_worsen_tolerance: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for rebalance_freq, group in metrics_df.groupby("rebalance_freq", sort=False):
        baseline = group[group["risk_model"] == "sample"]
        if baseline.shape[0] != 1:
            raise ValueError(f"Expected exactly one sample baseline for rebalance_freq={rebalance_freq}.")

        baseline_row = baseline.iloc[0]
        baseline_sharpe = float(baseline_row["primary_kpi_sharpe"])
        baseline_mdd_abs = abs(float(baseline_row["max_drawdown"]))

        for _, row in group.iterrows():
            sharpe_delta = float(row["primary_kpi_sharpe"]) - baseline_sharpe
            max_drawdown_worsen = abs(float(row["max_drawdown"])) - baseline_mdd_abs
            is_better = (sharpe_delta > 0.0) and (max_drawdown_worsen <= max_drawdown_worsen_tolerance)

            rows.append(
                {
                    "run_id": row["run_id"],
                    "run_label": row["run_label"],
                    "rebalance_freq": rebalance_freq,
                    "risk_model": row["risk_model"],
                    "garch_horizon": row["garch_horizon"],
                    "baseline_run_id": baseline_row["run_id"],
                    "baseline_sharpe": baseline_sharpe,
                    "baseline_max_drawdown": float(baseline_row["max_drawdown"]),
                    "delta_primary_kpi_sharpe": sharpe_delta,
                    "delta_annualized_return": float(row["annualized_return"])
                    - float(baseline_row["annualized_return"]),
                    "delta_annualized_volatility": float(row["annualized_volatility"])
                    - float(baseline_row["annualized_volatility"]),
                    "delta_max_drawdown": float(row["max_drawdown"]) - float(baseline_row["max_drawdown"]),
                    "max_drawdown_worsen_abs": max_drawdown_worsen,
                    "is_better_by_rule": bool(is_better),
                }
            )

    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def _plot_kpi_by_run(metrics_df: pd.DataFrame, out_path: Path) -> None:
    order = metrics_df.sort_values("primary_kpi_sharpe", ascending=False)
    colors = {
        "sample": "#4C78A8",
        "ewma": "#F58518",
        "garch": "#54A24B",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        order["run_label"],
        order["primary_kpi_sharpe"],
        color=[colors.get(name, "#999999") for name in order["risk_model"]],
    )
    ax.set_title("Out-of-Sample Annualized Sharpe by Run")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_xlabel("Run")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_garch_uplift_by_horizon(delta_df: pd.DataFrame, out_path: Path) -> None:
    garch = delta_df[delta_df["risk_model"] == "garch"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    if garch.empty:
        ax.text(0.5, 0.5, "No GARCH runs available.", ha="center", va="center")
        ax.set_axis_off()
    else:
        for rebalance_freq, group in garch.groupby("rebalance_freq", sort=False):
            group = group.sort_values("garch_horizon")
            ax.plot(
                group["garch_horizon"].astype(int),
                group["delta_primary_kpi_sharpe"],
                marker="o",
                label=str(rebalance_freq),
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_title("GARCH Sharpe Uplift vs Sample Baseline by Horizon")
        ax.set_xlabel("GARCH Horizon (days)")
        ax.set_ylabel("Sharpe Uplift vs Baseline")
        ax.grid(alpha=0.2)
        ax.legend(title="Rebalance Freq")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary_note(
    metrics_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    out_path: Path,
) -> None:
    top = rank_df.iloc[0]
    weekly = delta_df[(delta_df["rebalance_freq"] == "W-FRI") & (delta_df["risk_model"] == "garch")]
    monthly = delta_df[(delta_df["rebalance_freq"] == "ME") & (delta_df["risk_model"] == "garch")]
    weekly_best = float(weekly["delta_primary_kpi_sharpe"].max()) if not weekly.empty else np.nan
    monthly_best = float(monthly["delta_primary_kpi_sharpe"].max()) if not monthly.empty else np.nan

    mismatch_confirmed = False
    if np.isfinite(weekly_best) and np.isfinite(monthly_best):
        mismatch_confirmed = (weekly_best > 0.0) and (monthly_best <= 0.0)

    lines = [
        "# Risk Model Pilot Summary",
        "",
        f"- Total runs: {len(metrics_df)}",
        f"- Top run: `{top['run_id']}` ({top['run_label']})",
        f"- Top annualized Sharpe: {float(top['primary_kpi_sharpe']):.6f}",
        "",
        "## Horizon/Frequency Diagnostic",
        f"- Best GARCH Sharpe uplift at weekly rebalance (`W-FRI`): {weekly_best:.6f}",
        f"- Best GARCH Sharpe uplift at monthly rebalance (`ME`): {monthly_best:.6f}",
        f"- Horizon-frequency mismatch confirmed: `{mismatch_confirmed}`",
    ]
    out_path.write_text("\n".join(lines))


def run_pilot_experiment(
    returns: pd.DataFrame,
    output_dir: str | Path,
    config: PilotConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PilotArtifacts]:
    cfg = config or PilotConfig()
    cfg._validate()

    if returns.empty:
        raise ValueError("returns is empty.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex.")

    np.random.seed(cfg.seed)
    returns_clean = returns.sort_index().copy()

    runs = build_pilot_run_specs(cfg)
    rows: list[dict[str, Any]] = []

    for spec in runs:
        risk_model = _instantiate_risk_model(spec)
        strategy = MinimumVarianceStrategy(
            use_denoising=cfg.use_denoising,
            risk_model=risk_model,
        )
        backtester = PortfolioBacktester(
            strategy=strategy,
            target_vol=cfg.target_vol,
            apply_leverage=cfg.apply_leverage,
        )
        results = backtester.backtest(
            returns=returns_clean,
            rebalance_freq=spec.rebalance_freq,
            lookback_periods=cfg.lookback_periods,
            initial_capital=cfg.initial_capital,
            vol_method=cfg.vol_method,
            min_vol_threshold=cfg.min_vol_threshold,
            min_vol_floor=cfg.min_vol_floor,
            max_scale_factor=cfg.max_scale_factor,
            warmup_periods=cfg.warmup_periods,
            ramp_periods=cfg.ramp_periods,
            min_obs_for_full_trust=cfg.min_obs_for_full_trust,
            verbose=cfg.verbose,
        )
        metrics = backtester.get_performance_metrics(results)
        row = _build_metrics_row(spec=spec, risk_model=risk_model, results=results, metrics=metrics)
        row["use_denoising"] = cfg.use_denoising
        row["seed"] = cfg.seed
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)

    rank_df = (
        metrics_df.sort_values(
            ["primary_kpi_sharpe", "annualized_return", "max_drawdown"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
        .copy()
    )
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))

    delta_df = _build_delta_table(
        metrics_df,
        max_drawdown_worsen_tolerance=cfg.max_drawdown_worsen_tolerance,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "metrics_table.csv"
    rank_csv = out_dir / "rank_table.csv"
    delta_csv = out_dir / "delta_vs_baseline.csv"
    kpi_plot = out_dir / "kpi_by_run.png"
    uplift_plot = out_dir / "garch_sharpe_uplift_by_horizon.png"
    summary_note = out_dir / "pilot_summary.md"

    metrics_df.to_csv(metrics_csv, index=False)
    rank_df.to_csv(rank_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    _plot_kpi_by_run(metrics_df, kpi_plot)
    _plot_garch_uplift_by_horizon(delta_df, uplift_plot)
    _write_summary_note(metrics_df, rank_df, delta_df, summary_note)

    artifacts = PilotArtifacts(
        metrics_csv=metrics_csv,
        rank_csv=rank_csv,
        delta_csv=delta_csv,
        kpi_plot=kpi_plot,
        garch_uplift_plot=uplift_plot,
        summary_note=summary_note,
    )
    return metrics_df, rank_df, delta_df, artifacts


def run_pilot_from_files(
    returns_csv: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PilotArtifacts]:
    config = load_pilot_config(config_path)
    returns = load_returns_csv(returns_csv)
    return run_pilot_experiment(returns=returns, output_dir=output_dir, config=config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the risk-model horizon vs rebalance-frequency pilot experiment.")
    parser.add_argument(
        "--returns-csv",
        required=True,
        help="Path to returns CSV (DatetimeIndex in first column).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV tables and plots will be written.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON pilot config file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _, rank_df, _, artifacts = run_pilot_from_files(
        returns_csv=args.returns_csv,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    top = rank_df.iloc[0]
    print(
        f"Pilot complete. Top run: {top['run_id']} ({top['run_label']}) Sharpe={float(top['primary_kpi_sharpe']):.6f}"
    )
    print(f"Artifacts written to: {Path(args.output_dir).resolve()}")
    print(f"Summary note: {artifacts.summary_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
