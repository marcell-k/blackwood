import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Literal, cast

import pandas as pd
from backtesting import Backtest, Strategy

from blackwood.config import (
    CASH,
    DATA_DIR,
    MARGIN,
    NEWS_PATH,
    SPLIT_TIME,
)
from blackwood.indicators.core import process_news
from blackwood.presets.instruments import (
    ASSET_CLASSES,
    BROKER_COMMISSION,
    BROKER_SPREADS,
    NEWS_CURRENCIES,
    TIMEZONE_INSTRUMENT,
    US_OFFSET_INSTRUMENTS,
)
from blackwood.strategies.base import BuyAndHoldStrategy

_DATA_DIR = str(DATA_DIR)
_NEWS_PATH = str(NEWS_PATH)
_OHLCV_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


type SecurityData = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, tuple[float, float]]


@lru_cache(maxsize=256)
def _load_price_csv_cached(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    return df


@lru_cache(maxsize=4)
def _load_news_csv_cached(news_path: str) -> pd.DataFrame:
    return pd.read_csv(news_path, low_memory=False)


def load_security(
    security: str,
    resample_rule: str | None = None,
    timezone: str | None = None,
    use_offset: bool | None = True,
    split_time: str = SPLIT_TIME,
    base_path: str = _DATA_DIR,
    news_df: pd.DataFrame | None = None,
    news_n_bars: int = 1,
    news_impacts: Sequence[str] = ("red", "High Impact Expected"),
    indicator_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    use_news: bool = True,
    session_hours: Literal[False] | tuple[int, int] = False,
    is_stocks: bool = False,
    file_granularity: str | None = None,
) -> SecurityData:
    rule = resample_rule if resample_rule else "D"

    if is_stocks:
        if base_path.rstrip("/").endswith("H1"):
            file_granularity = "H1"
        elif base_path.rstrip("/").endswith("M15"):
            file_granularity = "M15"
        elif base_path.rstrip("/").endswith("M5"):
            file_granularity = "M5"
        else:
            file_granularity = "M15"
        default_tz = "US/Eastern"
        spread = 0.0006  # Spread: 0.02% | Commission: 0.01% | Swap-equivalent: 0.03% | Total ≈ 0.06%
        commission = (0, 0)
        currencies = ("USD", "")
    else:
        default_tz = TIMEZONE_INSTRUMENT.get(security)
        spread = BROKER_SPREADS.get(security) or 0.0
        commission = BROKER_COMMISSION.get(security) or (0.0, 0.0)
        currencies = NEWS_CURRENCIES.get(security, ("", ""))

    tz = timezone or default_tz
    if file_granularity is None:
        file_path = f"{base_path}/{security}.csv"
    else:
        file_path = f"{base_path}/{security}_{file_granularity}.csv"
    try:
        # Shallow copy protects cached source while avoiding repeat disk parsing.
        df = _load_price_csv_cached(file_path).copy(deep=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find data file: {file_path}") from exc

    is_us_future = security in US_OFFSET_INSTRUMENTS and use_offset
    is_intraday_agg = rule in ["1h", "2h", "4h"]

    if is_us_future and is_intraday_agg:
        df_naive = df.copy()
        naive_idx = pd.DatetimeIndex(df_naive.index).tz_convert(tz).tz_localize(None)
        df_naive.index = naive_idx

        origin_naive = pd.Timestamp("2000-01-01 09:30:00")
        df_resampled = df_naive.resample(rule, origin=origin_naive).agg(_OHLCV_AGG).dropna()  # type: ignore[arg-type]
        df = df_resampled
        df.index = pd.DatetimeIndex(df.index).tz_localize(tz, ambiguous="infer")
    else:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")
        df = df.resample(rule, origin="start_day").agg(_OHLCV_AGG).dropna()  # type: ignore[arg-type]
        df.index = pd.DatetimeIndex(df.index).tz_convert(tz)
    if indicator_fn is not None:
        df = indicator_fn(df)

    if session_hours is not False:
        start_hour, end_hour = session_hours
        idx = cast("pd.DatetimeIndex", df.index)
        df = df[(idx.hour >= start_hour) & (idx.hour <= end_hour)]

    if use_news:
        if news_df is None:
            news_df = _load_news_csv_cached(_NEWS_PATH)
        assert tz is not None, "timezone could not be resolved for security"
        df["News"] = process_news(
            df,
            news_df,
            n_bars=news_n_bars,
            timezone=tz,
            currencies=currencies,
            impacts=news_impacts,
        )

    split_date = pd.Timestamp(split_time).tz_localize(cast("pd.DatetimeIndex", df.index).tz)
    train = df.loc[df.index < split_date].copy()
    oos = df.loc[df.index >= split_date].copy()

    df.attrs["freq"] = rule
    train.attrs["freq"] = rule
    oos.attrs["freq"] = rule

    return df, train, oos, spread, commission


def run_batch_backtest(
    symbols: list[str] | None = None,
    strategy_cls: type[Strategy] = BuyAndHoldStrategy,
    timeframe: str | None = None,
    indicator_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    data: Literal["train", "oos", "df"] = "train",
    cash: float = CASH,
    margin: float = MARGIN,
    base_path: str = _DATA_DIR,
    is_stocks: bool = False,
    force_timezone: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    if symbols is None:
        if is_stocks:
            if base_path.rstrip("/").endswith("H1"):
                file_granularity = "H1"
            elif base_path.rstrip("/").endswith("M15"):
                file_granularity = "M15"
            elif base_path.rstrip("/").endswith("M5"):
                file_granularity = "M5"
            else:
                file_granularity = "M15"
            symbols = [
                fname.split(f"_{file_granularity}.csv")[0]
                for fname in os.listdir(base_path)
                if fname.endswith(f"_{file_granularity}.csv")
            ]
        else:
            symbols = list(BROKER_SPREADS.keys())

    results_list = []
    equity_curves = {}
    if verbose:
        print(f"\n{'=' * 33}")
        print(f"Running Batch: {timeframe} | {len(symbols)} Instruments")
        print(f"Dataset: {data.upper()}")
        print(f"{'=' * 33}")

    for symbol in symbols:
        try:
            df_full, train, oos, spread, commission = load_security(
                security=symbol,
                resample_rule=timeframe,
                base_path=base_path,
                indicator_fn=indicator_fn,
                is_stocks=is_stocks,
                timezone=force_timezone,
            )

            if data == "train":
                selected_data = train
            elif data == "oos":
                selected_data = oos
            else:
                selected_data = df_full

            if selected_data.empty:
                print(f"⚠ Skipping {symbol}: Empty dataset after split")
                continue

            bt = Backtest(
                selected_data,
                strategy_cls,
                cash=cash,
                exclusive_orders=False,
                trade_on_close=True,
                spread=spread * 1.1,
                commission=commission,
                margin=margin,
                hedging=False,
                finalize_trades=True,
            )
            stats = bt.run()
            equity_curves[symbol] = stats["_equity_curve"]["Equity"]
            has_sl = stats["_trades"]["SL"].notna().any()
            if has_sl:
                from blackwood.metrics.core import calculate_risk_reward_ratio

                stats["_trades"] = calculate_risk_reward_ratio(stats["_trades"])
            result = {
                "Symbol": symbol,
                "# Trades": int(stats["# Trades"]),
                "PF": round(stats["Profit Factor"], 2),
                "Sharpe": round(stats["Sharpe Ratio"], 2),
                "Calmar": round(stats["Calmar Ratio"], 2),
                "Max DD%": round(stats["Max. Drawdown [%]"], 2),
                "Win Rate%": round(stats["Win Rate [%]"], 2),
                "Exp %": round(stats["Expectancy [%]"], 3),
                "Return %": round(stats["Return [%]"], 2),
            }
            if has_sl:
                result["Mean RRR"] = round(stats["_trades"]["RiskRewardRatio"].mean(), 2)
            results_list.append(result)
        except FileNotFoundError:
            print(f" Error: File not found for {symbol}")
        except KeyError as e:
            print(f" Error: Configuration missing for {symbol} - {e}")
        except Exception as e:
            print(f" Error testing {symbol}: {e!s}")

    if not results_list:
        print("⚠ No valid results generated.")
        return pd.DataFrame(), {}

    results_df = pd.DataFrame(results_list)
    results_df.sort_values("Sharpe", ascending=False, inplace=True)
    if verbose:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.width", 120)
        print(f"\n{'─' * 92}")
        print(f"RESULTS: {timeframe} ({data.upper()})")
        print(f"{'─' * 92}")
        print(results_df.to_string(index=False))
        print(f"{'─' * 92}\n")

    return results_df, equity_curves


def _create_equal_weight_portfolio(equity_curves: dict[str, pd.Series]) -> pd.Series:
    """Create equal-weight portfolio from equity curves."""
    equity_df = pd.DataFrame(equity_curves).ffill()
    returns_df = equity_df.pct_change()
    portfolio_returns = returns_df.mean(axis=1, skipna=True)

    first_valid_idx = returns_df.notna().any(axis=1).idxmax()
    initial_capital = equity_df.loc[first_valid_idx].mean(skipna=True)
    portfolio_equity = initial_capital * (1 + portfolio_returns).cumprod()

    if not portfolio_equity.empty:
        portfolio_equity.iloc[0] = initial_capital

    return portfolio_equity


def run_full_suite(
    strategy_cls: type[Strategy],
    timeframes: list[str] | None = None,
    indicator_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    data: Literal["train", "oos", "df"] = "train",
    cash: float = CASH,
    margin: float = MARGIN,
    base_path: str = _DATA_DIR,
    symbols: Sequence[str] | None = None,
    is_stocks: bool = False,
    asset_classes: str | tuple[str, ...] = "All",
    exclude_assets: str | tuple[str, ...] = (),
    force_timezone: str | None = None,
    create_portfolios: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    if timeframes is None:
        timeframes = ["5min", "15min", "30min", "1h", "2h", "4h", "D", "W"]
    if isinstance(exclude_assets, str):
        exclude_assets = (exclude_assets,)

    if symbols is None:
        if is_stocks:
            pass
        elif asset_classes == "All":
            symbols = list(BROKER_SPREADS.keys())
        else:
            if isinstance(asset_classes, str):
                asset_classes = (asset_classes,)
            filtered_symbols = []
            for asset_class in asset_classes:
                if asset_class in ASSET_CLASSES:
                    filtered_symbols.extend(ASSET_CLASSES[asset_class])
                else:
                    print(f"⚠ Warning: Unknown asset class '{asset_class}', skipping")
            symbols = [s for s in filtered_symbols if s in BROKER_SPREADS]

        if symbols is not None and exclude_assets:
            excluded_set = set(exclude_assets)
            symbols = [s for s in symbols if s not in excluded_set]

        if symbols is None and not is_stocks:
            print(f"⚠ Warning: No valid symbols found for asset classes {asset_classes}")
            return pd.DataFrame(), {}

        if symbols is not None and not symbols and not is_stocks:
            print(f"⚠ Warning: No valid symbols found for asset classes {asset_classes}")
            return pd.DataFrame(), {}

    all_equity_curves = {}
    results_df = pd.DataFrame()
    for tf in timeframes:
        results_df, equity_curves = run_batch_backtest(
            symbols=list(symbols) if symbols else None,
            strategy_cls=strategy_cls,
            timeframe=tf,
            indicator_fn=indicator_fn,
            data=data,
            cash=cash,
            margin=margin,
            base_path=base_path,
            is_stocks=is_stocks,
            force_timezone=force_timezone,
            verbose=verbose,
        )
        all_equity_curves[tf] = equity_curves

    # Create portfolios and benchmarks if requested
    if create_portfolios:
        from blackwood.strategies.base import BuyAndHoldStrategy

        for tf in timeframes:
            equity_curves = all_equity_curves[tf]

            if len(equity_curves) < 2:
                continue  # Need at least 2 strategies for portfolio

            # Save original symbol list before adding portfolio
            original_symbols = list(equity_curves.keys())

            # 1. Create equal-weight portfolio from strategies
            portfolio_equity = _create_equal_weight_portfolio(equity_curves)
            all_equity_curves[tf]["Portfolio_EqualWeight"] = portfolio_equity

            # # 2. Run Buy & Hold backtests for all symbols
            bh_equity_curves = {}
            for symbol in original_symbols:
                try:
                    df_full, train, oos, spread, commission = load_security(
                        security=symbol,
                        resample_rule=tf,
                        base_path=base_path,
                        indicator_fn=indicator_fn,
                        is_stocks=is_stocks,
                        timezone=force_timezone,
                    )

                    # Use same data split as strategy
                    if data == "train":
                        selected_data = train
                    elif data == "oos":
                        selected_data = oos
                    else:
                        selected_data = df_full

                    bt = Backtest(
                        selected_data,
                        BuyAndHoldStrategy,
                        cash=cash,
                        spread=spread * 1.1,
                        commission=commission,
                        trade_on_close=True,
                        exclusive_orders=False,
                        finalize_trades=True,
                    )
                    bh_stats = bt.run()
                    bh_equity_curves[symbol] = bh_stats["_equity_curve"]["Equity"]
                except Exception as e:
                    print(f"⚠ Warning: Buy & Hold failed for {symbol}: {e}")

            # 3. Create equal-weight Buy & Hold benchmark portfolio
            if bh_equity_curves:
                bh_portfolio_equity = _create_equal_weight_portfolio(bh_equity_curves)
                all_equity_curves[tf]["Benchmark_BuyHold"] = bh_portfolio_equity

    return results_df, all_equity_curves


def run_full_suite_berlin_tz(
    strategy_cls: type[Strategy],
    timeframes: list[str] | None = None,
    indicator_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    data: Literal["train", "oos", "df"] = "train",
    cash: float = CASH,
    margin: float = MARGIN,
    base_path: str = _DATA_DIR,
    symbols: Sequence[str] | None = None,
    is_stocks: bool = False,
    asset_classes: str | tuple[str, ...] = "All",
    exclude_assets: str | tuple[str, ...] = (),
    create_portfolios: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    if timeframes is None:
        timeframes = ["5min", "15min", "30min", "1h", "2h", "4h", "D", "W"]
    return run_full_suite(
        strategy_cls=strategy_cls,
        timeframes=timeframes,
        indicator_fn=indicator_fn,
        data=data,
        cash=cash,
        margin=margin,
        base_path=base_path,
        symbols=symbols,
        is_stocks=is_stocks,
        asset_classes=asset_classes,
        exclude_assets=exclude_assets,
        force_timezone="Europe/Berlin",
        create_portfolios=create_portfolios,
        verbose=verbose,
    )
