from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from backtesting import Backtest, Strategy

    from blackwood.optimization.optimization import OptimizationResult

type CPCVPaths = dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]]
type ParameterSpace = dict[str, tuple[Any, ...] | list[int | float]]
type CommissionSpec = float | tuple[float, float] | Callable[..., float]
type ReturnsFrame = pd.DataFrame  # index=DatetimeIndex, columns=strategy/asset names
type BacktestStats = dict[str, Any]


class ParamStabilityRow(TypedDict):
    """Per-parameter stability metrics, as built in robustness/parameter_stability.py."""

    mean_all: float
    std_all: float
    cv_all: float
    mean_top: float
    std_top: float
    cv_top: float
    mode_top: float | None


type StabilityMetrics = dict[str, ParamStabilityRow]


class PathResult(TypedDict):
    """Per-CPCV-path result, as built by CPCVAnalyzer._run_wfo_on_paths."""

    path_id: int
    n_folds: int
    final_equity: float
    path_metrics: dict[str, float]
    parameters: list[list[Any]]


type PathResults = dict[int, PathResult]


class RiskRulesConfig(TypedDict):
    """Normalized risk-rules config, as returned by PortfolioBacktester._normalize_risk_rules."""

    enabled: bool
    intraday_mode: bool
    day_boundary: str
    close_daily_loss_threshold: float
    half_risk_on_drawdown: float
    half_risk_off_drawdown: float
    half_risk_multiplier: float
    timeframe: str


class WFOResult(NamedTuple):
    """Return value of WalkForwardOptimizer.run_wfo (replaces the old bare 7-tuple)."""

    trades: pd.DataFrame
    stats_train: list[BacktestStats]
    stats_test: list[BacktestStats]
    parameters: list[list[Any]]
    backtests: list[Backtest]
    param_names: list[str]
    optimize_results: list[Any]


class BacktestFunc(Protocol):
    """Callable shape accepted as `bt_func` by GridOptimizer/OptunaOptimizer/CPCV workers."""

    def __call__(self, df: pd.DataFrame, strat: type[Strategy], cash: float, **params: Any) -> BacktestStats: ...


class Optimizer(Protocol):
    """Shared contract implemented by SamboOptimizer, OptunaOptimizer, GridOptimizer, PortfolioOptimizer."""

    def optimize(
        self,
        df: pd.DataFrame,
        strategy_class: type[Strategy],
        cash: float,
        **kwargs: Any,
    ) -> OptimizationResult: ...


class RiskModelProtocol(Protocol):
    """Structural counterpart to portfolio.risk_models.RiskModel, for duck-typed risk models."""

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame: ...
