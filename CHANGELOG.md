# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

---

## [0.1.0] - 2026-06-14

Initial public release.

### Added

**Data**
- OHLCV loading with DukasCopy and MT5 support (`src/data/loaders.py`)
- CPCV and walk-forward train/test splitters (`src/data/splitters.py`)
- Block-bootstrap resampling (`src/data/bootstrap.py`)

**Indicators**
- Ehlers dominant-cycle detection and adaptive ATR (`src/indicators/cycle.py`)
- Support/resistance zone detection (`src/indicators/zone.py`)
- ZigZag pivot implementation (`src/indicators/zigzag_pure.py`)
- News event processing (`src/indicators/core.py`)

**Strategies**
- `BaseTemplateStrategy` and `MetaLabeling` base classes (`src/strategies/base.py`)
- Wyckoff fractal, spring, and upthrust detection (`src/strategies/wyckoff.py`)

**Meta-Labeling**
- XGBoost binary classifier with Optuna tuning (`src/meta_labeling/models.py`)
- Random Forest meta-labeler (`src/meta_labeling/rf_model.py`)
- Feature engineering, selection, calibration, and evaluation

**Regime Detection**
- Three-stage GMM → HMM → Random Forest classifier (`src/regime/models.py`)
- Multi-timeframe volatility features (`src/regime/features.py`)
- Regime transition analysis and end-to-end pipeline

**Portfolio**
- CVXPY-based MVO, HRP, CVaR, and risk-parity optimizers (`src/portfolio/optimizer.py`)
- Risk models: sample covariance, EWMA, GARCH(1,1), DCC-GARCH (`src/portfolio/risk_models.py`)
- Marchenko-Pastur covariance denoising (`src/portfolio/denoising.py`)
- Fractional-Kelly position sizing (`src/portfolio/risk_allocator.py`)
- Portfolio analytics and visualization

**Optimization**
- SAMBO and Optuna hyperparameter search (`src/optimization/optimization.py`)
- Walk-Forward Optimizer with WFE reporting (`src/optimization/walk_forward.py`)
- CPCV cross-validation (`src/optimization/cross_validation.py`)

**Evaluation**
- Four Monte Carlo methods: IID, Stationary Bootstrap, POT-GPD, GH/Student-t (`src/evaluation/monte_carlo.py`)
- Strategy analytics and equity-curve diagnostics (`src/evaluation/analyzers.py`)
- Robustness checks (`src/evaluation/robustness.py`)

**Robustness**
- Five-phase CPCV → bootstrap → OOS pipeline (`src/robustness/stability_pipeline.py`)
- Optuna-based parameter stability analysis (`src/robustness/parameter_stability.py`)
- Composite scoring and rank display (`src/robustness/ranking_display.py`)

**Metrics**
- Sharpe, Calmar, drawdown, and information-theory metrics (`src/metrics/core.py`)

**Visualization**
- Core plotting, regime charts, and style defaults

[Unreleased]: https://github.com/marcell-k/blackwood/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marcell-k/blackwood/releases/tag/v0.1.0
