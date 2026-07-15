import ast
import inspect
import textwrap

from blackwood.portfolio.optimizer import (
    EnsembleStrategy,
    EqualStrategy,
    HRPStrategy,
    MinimumVarianceStrategy,
    MVOStrategy,
    NCOStrategy,
    OptimalFCalculator,
    RiskParityStrategy,
    TailRiskParityStrategy,
    TangencyStrategy,
)
from blackwood.portfolio.risk_models import LegacySampleCovariance


def _calls_estimate_covariance(method) -> bool:
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "estimate_covariance"
        ):
            return True
    return False


def test_all_optimizer_strategies_accept_risk_model_parameter():
    strategies = [
        TangencyStrategy,
        EqualStrategy,
        MVOStrategy,
        MinimumVarianceStrategy,
        NCOStrategy,
        RiskParityStrategy,
        TailRiskParityStrategy,
        HRPStrategy,
        EnsembleStrategy,
        OptimalFCalculator,
    ]

    for strategy in strategies:
        assert "risk_model" in inspect.signature(strategy).parameters


def test_covariance_paths_route_through_estimate_covariance():
    methods = [
        TangencyStrategy.compute_weights,
        MVOStrategy.compute_weights,
        MinimumVarianceStrategy.compute_weights,
        NCOStrategy.compute_weights,
        NCOStrategy._optimize_inter_cluster,
        RiskParityStrategy.compute_weights,
        TailRiskParityStrategy.compute_weights,
        HRPStrategy.compute_weights,
        EnsembleStrategy._minvar_weights,
    ]

    for method in methods:
        assert _calls_estimate_covariance(method)


def test_default_optimizer_risk_model_is_legacy_pairwise_sample_covariance():
    strategy = TangencyStrategy()
    assert isinstance(strategy.risk_model, LegacySampleCovariance)
