# Blackwood

A quantitative trading research framework built on [backtesting.py](https://github.com/kernc/backtesting.py). Covers the full pipeline from data loading and feature engineering through strategy optimization, meta-labeling, portfolio construction, and robustness validation.

## Structure

```
src/
├── config.py              # Global constants: instruments, spreads, commissions, timezones
├── data/
│   ├── loaders.py         # OHLCV loading, news integration, backtest setup
│   ├── splitters.py       # CPCV and walk-forward train/test splits
│   └── bootstrap.py       # Block-bootstrap resampling utilities
├── indicators/
│   ├── core.py            # News event processing
│   ├── cycle.py           # Ehler dominant-cycle detection, adaptive ATR
│   ├── zone.py            # Support/resistance zones
│   └── zigzag_pure.py     # ZigZag pivot implementation
├── strategies/
│   ├── base.py            # BaseTemplateStrategy and MetaLabeling strategy classes
│   ├── tools.py           # Entry/exit helpers
│   └── wyckoff.py         # Wyckoff fractal, spring, and upthrust detection
├── meta_labeling/
│   ├── features.py        # Feature engineering for meta-labels
│   ├── feature_selection.py
│   ├── models.py          # XGBoost binary classifier with Optuna tuning
│   ├── rf_model.py        # Random Forest meta-labeler
│   ├── calibration.py     # Probability calibration
│   ├── evaluation.py      # Classification metrics and diagnostics
│   ├── selection.py       # Model selection utilities
│   └── utils.py
├── regime/
│   ├── features.py        # Multi-timeframe volatility features
│   ├── models.py          # GMM + HMM + Random Forest regime detector
│   ├── analysis.py        # Regime transition analysis
│   └── pipeline.py        # End-to-end regime detection pipeline
├── portfolio/
│   ├── core.py            # Portfolio core types
│   ├── data.py            # Portfolio data helpers
│   ├── risk_models.py     # SampleCovariance, EWMA, GARCH(1,1), DCC-GARCH
│   ├── denoising.py       # Marchenko-Pastur covariance denoising
│   ├── optimizer.py       # MVO, HRP, CVaR, risk-parity via CVXPY
│   ├── risk_allocator.py  # Kelly / fractional-Kelly sizing
│   ├── analyzer.py        # Portfolio analytics
│   ├── visualization.py   # Portfolio charts
│   └── utils.py
├── optimization/
│   ├── optimization.py    # SAMBO and Optuna hyperparameter search
│   ├── walk_forward.py    # Walk-Forward Optimizer with WFE reporting
│   └── cross_validation.py # CPCV cross-validation
├── evaluation/
│   ├── analyzers.py       # Strategy analytics and equity-curve diagnostics
│   ├── monte_carlo.py     # IID, Stationary Bootstrap, POT-GPD, GH/Student-t MC
│   ├── robustness.py      # Robustness checks
│   └── utils.py           # Equity-curve merging, performance metrics
├── robustness/
│   ├── parameter_stability.py  # Optuna-based parameter stability analysis
│   ├── stability_pipeline.py   # 5-phase CPCV → bootstrap → OOS pipeline
│   └── ranking_display.py      # Composite scoring and rank display
├── metrics/
│   └── core.py            # Sharpe, Calmar, drawdown, and information-theory metrics
├── utils/
│   ├── benchmark.py       # Benchmark comparison helpers
│   ├── debug.py           # Debugging utilities
│   └── information_theory.py  # Entropy and mutual-information measures
└── visualization/
    ├── core.py            # Core plotting helpers
    ├── regime.py          # Regime visualization
    ├── style.py           # Plot style defaults
    └── tests.py           # Visual test output helpers
```

## Key Concepts

**Walk-Forward Optimization** — `WalkForwardOptimizer` splits data into rolling IS/OOS windows, optimizes hyperparameters with SAMBO on each IS fold, and reports Walk-Forward Efficiency (WFE) across all folds.

**CPCV** — Combinatorial Purged Cross-Validation splits prevent leakage between overlapping time-series folds. Used throughout optimization and stability analysis.

**Meta-Labeling** — A secondary XGBoost or Random Forest model gates primary strategy signals, outputting calibrated probabilities that drive position sizing. The `MetaLabeling` base strategy class integrates gate and bet columns directly into backtesting.py.

**Regime Detection** — A three-stage classifier (GMM for unsupervised clustering → HMM for sequence smoothing → Random Forest for supervised refinement) labels market regimes from multi-timeframe volatility features.

**Portfolio Optimization** — CVXPY-based solvers support Mean-Variance, HRP, CVaR minimization, and risk-parity. Risk models include sample covariance, EWMA, GARCH(1,1), and DCC-GARCH with optional Marchenko-Pastur denoising.

**Robustness Pipeline** — Five-phase process: CPCV execution → dual-filter selection (performance + proximity stability) → block-bootstrap validation → OOS holdout → composite scoring and ranking.

**Monte Carlo** — Four simulation methods (IID, Stationary Bootstrap, POT-GPD semi-parametric, GH/Student-t parametric) stress-test equity curves and estimate tail-risk metrics.

## Installation

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/marcell-k/blackwood.git
cd blackwood
uv sync
```

Install dev dependencies (linting, type-checking, tests):

```bash
uv sync --extra dev
```

**Note:** several modules import optional heavy dependencies (`cvxpy`, `xgboost`, `optuna`, `hmmlearn`, `numba`). Install these separately if you use those modules:

```bash
uv add cvxpy xgboost optuna hmmlearn numba scikit-learn scipy
```

## Configuration

`src/config.py` holds all global constants:

| Constant | Default | Description |
|---|---|---|
| `IS_MONTHS` | 10 | In-sample window length (months) |
| `OOS_MONTHS` | 3 | Out-of-sample window length (months) |
| `KELLY_FRACTION` | 0.1 | Fractional Kelly multiplier |
| `CASH` | 10,000,000 | Starting equity |
| `MARGIN` | 0.01 | Margin requirement |
| `ANNUAL_TRADING_DAYS` | 252 | Used for annualising metrics |
| `RANDOM_STATE` | 89 | Global RNG seed |
| `SPLIT_TIME` | 2024-01-01 | Train/test split date |

`BROKER_SPREADS` and `BROKER_COMMISSION` cover crypto, indices, FX, metals, and energy instruments.

## Usage Examples

```python
from backtesting import Backtest
from src.data.loaders import load_ohlcv
from src.optimization.walk_forward import WalkForwardOptimizer
from src.strategies.wyckoff import WyckoffStrategy  # your concrete strategy

df = load_ohlcv("EURUSD", timeframe="1h")

wfo = WalkForwardOptimizer(
    train_dfs=[df_train],
    test_dfs=[df_test],
    is_months=10,
    oos_months=3,
)
results = wfo.run(WyckoffStrategy, param_space={...})
```

```python
from src.evaluation.monte_carlo import MonteCarloSimulator

mc = MonteCarloSimulator(data=equity_curve, n_simulations=10_000)
mc.run(method="stationary_bootstrap")
mc.plot()
```

```python
from src.portfolio.optimizer import HRPOptimizer
from src.portfolio.risk_models import EWMARiskModel

risk_model = EWMARiskModel(span=60)
optimizer = HRPOptimizer(risk_model=risk_model)
weights = optimizer.optimize(returns_df)
```

## Running Tests

```bash
pytest src/portfolio/tests/
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
