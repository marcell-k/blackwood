import inspect

import numpy as np
import pandas as pd
import pytest
from src.portfolio.risk_models import (
    EWMARiskModel,
    GARCH11RiskModel,
    LegacySampleCovariance,
    SampleCovariance,
)


def _make_returns() -> pd.DataFrame:
    rng = np.random.default_rng(20240218)
    arr = rng.normal(0.0004, 0.01, size=(80, 4))
    arr[3, 1] = np.nan
    arr[17, 3] = np.nan
    arr[40, 0] = np.nan
    idx = pd.bdate_range("2024-01-02", periods=80, tz="UTC")
    return pd.DataFrame(arr, index=idx, columns=["A", "B", "C", "D"])


EXPECTED_SAMPLE_COV = np.array(
    [
        [
            1.0797298154736561e-04,
            -1.0354037963115247e-05,
            -9.9156460582758511e-06,
            -1.2051531540993854e-05,
        ],
        [
            -1.0354037963115247e-05,
            9.5030213410581067e-05,
            -1.8086734199865284e-06,
            3.8879181798449718e-06,
        ],
        [
            -9.9156460582758511e-06,
            -1.8086734199865284e-06,
            7.4582470548742845e-05,
            2.2178112507367826e-05,
        ],
        [
            -1.2051531540993854e-05,
            3.8879181798449718e-06,
            2.2178112507367826e-05,
            1.2789507281379944e-04,
        ],
    ],
    dtype=float,
)

EXPECTED_GARCH_COV = np.array(
    [
        [
            7.4217960903696304e-05,
            -5.1732879713100231e-06,
            -6.8227191702074297e-06,
            -9.7147610195131816e-06,
        ],
        [
            -5.1732879713100231e-06,
            3.4512910298276073e-05,
            -9.0460631010636225e-07,
            2.2780852575568895e-06,
        ],
        [
            -6.8227191702074263e-06,
            -9.0460631010636225e-07,
            5.1370737452825752e-05,
            1.7896043120845256e-05,
        ],
        [
            -9.7147610195131799e-06,
            2.2780852575568967e-06,
            1.7896043120845263e-05,
            1.2090371595293300e-04,
        ],
    ],
    dtype=float,
)

EXPECTED_LEGACY_SAMPLE_COV = np.array(
    [
        [
            1.0935306233198729e-04,
            -1.0340325253337351e-05,
            -1.0053794403411288e-05,
            -1.2400988865727667e-05,
        ],
        [
            -1.0340325253337351e-05,
            9.6080330596644951e-05,
            -1.5521522107695216e-06,
            4.6621402292449890e-06,
        ],
        [
            -1.0053794403411288e-05,
            -1.5521522107695216e-06,
            7.4582470548742832e-05,
            2.2509720584712536e-05,
        ],
        [
            -1.2400988865727667e-05,
            4.6621402292449890e-06,
            2.2509720584712536e-05,
            1.2953135000941542e-04,
        ],
    ],
    dtype=float,
)

EXPECTED_EWMA_COV = np.array(
    [
        [
            1.0660468025296352e-04,
            -1.8323823850300528e-05,
            -2.2833446147318779e-05,
            -3.5860743744638466e-05,
        ],
        [
            -1.8323823850300528e-05,
            5.8055465004342600e-05,
            6.2156166205630990e-06,
            2.2147570775056616e-05,
        ],
        [
            -2.2833446147318779e-05,
            6.2156166205630990e-06,
            8.5788980596393947e-05,
            5.3281783592625838e-05,
        ],
        [
            -3.5860743744638466e-05,
            2.2147570775056616e-05,
            5.3281783592625838e-05,
            1.4169556175351838e-04,
        ],
    ],
    dtype=float,
)


def test_risk_model_constructor_signatures_unchanged():
    expected_garch_signature = (
        "(horizon: int = 1, corr_source: str = 'sample', "
        "eps: float = 1e-12) -> None"
    )
    assert str(inspect.signature(GARCH11RiskModel)) == expected_garch_signature
    assert (
        str(inspect.signature(EWMARiskModel))
        == "(lam: float = 0.94, ddof: int = 0, eps: float = 1e-12, init: str = 'sample') -> None"
    )
    assert str(inspect.signature(SampleCovariance)) == "(ddof: int = 1) -> None"
    assert (
        str(inspect.signature(LegacySampleCovariance))
        == "(ddof: int = 1, min_periods: int | None = None) -> None"
    )


def test_sample_covariance_regression():
    returns = _make_returns()
    cov = SampleCovariance(ddof=1).covariance(returns)
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (4, 4)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    np.testing.assert_allclose(cov.values, EXPECTED_SAMPLE_COV, rtol=0.0, atol=1e-12)


def test_garch_covariance_regression():
    returns = _make_returns()
    cov = GARCH11RiskModel(horizon=2).covariance(returns)
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (4, 4)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    np.testing.assert_allclose(cov.values, EXPECTED_GARCH_COV, rtol=0.0, atol=1e-12)


def test_legacy_sample_covariance_matches_old_pairwise_covariance():
    returns = _make_returns()
    cov = LegacySampleCovariance(ddof=1).covariance(returns)
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (4, 4)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    np.testing.assert_allclose(cov.values, EXPECTED_LEGACY_SAMPLE_COV, rtol=0.0, atol=1e-12)


def test_ewma_covariance_regression():
    returns = _make_returns()
    cov = EWMARiskModel(lam=0.94, ddof=0, init="sample").covariance(returns)
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (4, 4)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    np.testing.assert_allclose(cov.values, EXPECTED_EWMA_COV, rtol=0.0, atol=1e-12)


def test_ewma_short_history_returns_eps_diagonal():
    returns = pd.DataFrame([[0.01, -0.02]], columns=["A", "B"])
    model = EWMARiskModel(eps=1e-8)
    cov = model.covariance(returns)
    np.testing.assert_allclose(cov.values, np.eye(2) * 1e-8, rtol=0.0, atol=0.0)


def test_ewma_invalid_lambda_raises():
    returns = _make_returns()
    with pytest.raises(ValueError, match="lam must be in \\(0, 1\\)"):
        EWMARiskModel(lam=1.0).covariance(returns)


def test_garch_invalid_corr_source_raises():
    returns = _make_returns()
    with pytest.raises(ValueError, match="Unknown corr_source"):
        GARCH11RiskModel(corr_source="foo").covariance(returns)


def test_projected_correlation_has_unit_diagonal_and_is_bounded():
    model = GARCH11RiskModel()
    raw = np.array(
        [
            [1.0, 1.4, -1.6],
            [1.4, 1.0, 0.8],
            [-1.6, 0.8, 1.0],
        ],
        dtype=float,
    )
    corr = model._project_to_corr_psd(raw)
    np.testing.assert_allclose(np.diag(corr), np.ones(3), rtol=0.0, atol=1e-12)
    assert np.all(corr <= 1.0)
    assert np.all(corr >= -1.0)
