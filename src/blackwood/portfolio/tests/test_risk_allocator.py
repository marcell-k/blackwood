import numpy as np
import pandas as pd
from blackwood.config import LIVE_START, TEST_END, TRAIN_END
from blackwood.data.splitters import split_train_test_live
from blackwood.portfolio.core import PortfolioBacktester
from blackwood.portfolio.risk_allocator import AllocatorConfig, CentralRiskAllocator


class DummyEqualStrategy:
    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)


def _make_returns(
    n: int = 400,
    start: str = "2023-01-02",
    cols: tuple[str, ...] = ("S1", "S2", "S3"),
    seed: int = 7,
) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n, tz="UTC")
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0004, scale=0.01, size=(n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=list(cols))


def test_split_train_test_live_integrity():
    idx = pd.date_range("2023-12-29", "2025-12-02", freq="D", tz="UTC")
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)

    train_df, test_df, live_df = split_train_test_live(
        df=df,
        train_end=TRAIN_END,
        test_end=TEST_END,
    )

    assert not train_df.empty
    assert not test_df.empty
    assert not live_df.empty

    assert train_df.index.max() <= pd.Timestamp(TRAIN_END, tz="UTC")
    assert test_df.index.min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert test_df.index.max() <= pd.Timestamp(TEST_END, tz="UTC")
    assert live_df.index.min() >= pd.Timestamp(LIVE_START, tz="UTC")

    train_set = set(train_df.index)
    test_set = set(test_df.index)
    live_set = set(live_df.index)
    assert train_set.isdisjoint(test_set)
    assert train_set.isdisjoint(live_set)
    assert test_set.isdisjoint(live_set)


def test_compute_multipliers_no_lookahead():
    returns = _make_returns(n=420, seed=19)
    train_returns = returns.loc[returns.index <= pd.Timestamp("2023-12-29", tz="UTC")]
    as_of = pd.Timestamp("2024-07-15", tz="UTC")

    cfg = AllocatorConfig(rolling_window=30, min_history=30, alpha=0.7)

    allocator_a = CentralRiskAllocator(cfg)
    allocator_a.fit(train_returns)
    m_a = allocator_a.compute_multipliers(as_of=as_of, returns_hist=returns)

    returns_shifted_future = returns.copy()
    returns_shifted_future.loc[returns_shifted_future.index >= as_of, :] = 0.35

    allocator_b = CentralRiskAllocator(cfg)
    allocator_b.fit(train_returns)
    m_b = allocator_b.compute_multipliers(as_of=as_of, returns_hist=returns_shifted_future)

    pd.testing.assert_series_equal(m_a.sort_index(), m_b.sort_index(), atol=1e-12, rtol=0.0)


def test_overlay_constraints_sum_cap_floor_turnover():
    allocator = CentralRiskAllocator(
        AllocatorConfig(weight_floor=0.0, weight_cap=0.70, turnover_cap=0.10)
    )

    base = pd.Series([0.5, 0.3, 0.2], index=["S1", "S2", "S3"], dtype=float)
    multipliers = pd.Series([0.1, 3.0, 2.0], index=["S1", "S2", "S3"], dtype=float)
    adjusted = allocator.apply_overlay(base_weights=base, multipliers=multipliers)

    assert np.isclose(adjusted.sum(), 1.0, atol=1e-10)
    assert adjusted.max() <= 0.70 + 1e-10
    assert adjusted.min() >= -1e-12

    turnover = float(np.abs(adjusted - base).sum())
    assert turnover <= 0.10 + 1e-8


def test_compute_multipliers_cold_start_neutral():
    returns = _make_returns(n=220, seed=33)
    allocator = CentralRiskAllocator(AllocatorConfig(rolling_window=40, min_history=40))
    allocator.fit(returns.iloc[:120])

    as_of = returns.index[25]
    multipliers = allocator.compute_multipliers(as_of=as_of, returns_hist=returns.iloc[:26])

    assert np.allclose(multipliers.values, 1.0, atol=0.0, rtol=0.0)


def test_backtester_allocator_regression_logging():
    returns = _make_returns(n=780, start="2023-01-02", seed=91)
    strategy = DummyEqualStrategy()

    bt_no_allocator = PortfolioBacktester(strategy=strategy, apply_leverage=False)
    res_no_allocator = bt_no_allocator.backtest(
        returns=returns,
        rebalance_freq="W-FRI",
        lookback_periods=63,
        initial_capital=1_000_000,
    )

    assert "allocator_history" in res_no_allocator
    assert res_no_allocator["allocator_history"] == []

    allocator = CentralRiskAllocator(AllocatorConfig(rolling_window=63, min_history=63))
    bt_with_allocator = PortfolioBacktester(
        strategy=strategy,
        apply_leverage=False,
        central_allocator=allocator,
        allocator_freq="W-FRI",
    )
    res_with_allocator = bt_with_allocator.backtest(
        returns=returns,
        rebalance_freq="W-FRI",
        lookback_periods=63,
        initial_capital=1_000_000,
        allocator_start_date=pd.Timestamp("2024-01-01", tz="UTC"),
    )

    assert "allocator_history" in res_with_allocator
    assert len(res_with_allocator["allocator_history"]) > 0
    assert not res_with_allocator["allocator_history_df"].empty
    assert len(res_with_allocator["equity"]) == len(res_no_allocator["equity"])
    assert res_with_allocator["weights_history"].shape == res_no_allocator["weights_history"].shape


def test_allocator_runs_independently_from_base_rebalance_frequency():
    returns = _make_returns(n=780, start="2023-01-02", seed=123)
    strategy = DummyEqualStrategy()
    allocator = CentralRiskAllocator(AllocatorConfig(rolling_window=63, min_history=63))

    bt = PortfolioBacktester(
        strategy=strategy,
        apply_leverage=False,
        central_allocator=allocator,
        allocator_freq="W-FRI",
    )

    # Base optimizer rebalances monthly, allocator rebalances weekly.
    results = bt.backtest(
        returns=returns,
        rebalance_freq="ME",
        lookback_periods=63,
        initial_capital=1_000_000,
        allocator_start_date=pd.Timestamp("2024-01-01", tz="UTC"),
    )

    n_allocator = len(results["allocator_history"])
    n_base = len(results["scale_factors_by_date"])

    assert n_allocator > 0
    assert n_base > 0
    assert n_allocator > n_base
