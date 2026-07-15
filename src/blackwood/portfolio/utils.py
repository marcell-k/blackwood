import pandas as pd


def extend_returns_with_next_month_start(returns: pd.DataFrame) -> pd.DataFrame:
    last_date = returns.index.max()
    next_month_start = last_date + pd.offsets.MonthBegin(1)
    synthetic_row = pd.DataFrame(0.0, index=[next_month_start], columns=returns.columns)
    extended = pd.concat([returns, synthetic_row])
    extended = extended.loc[~extended.index.duplicated(keep="last")].sort_index()
    return extended


def _resample_df_daily(df: pd.DataFrame, rule: str = "D") -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    bool_cols = df.select_dtypes(include=["bool"]).columns
    agg_map = dict.fromkeys(bool_cols, "max")
    agg_map.update({col: "last" for col in df.columns if col not in bool_cols})
    return df.sort_index().resample(rule).agg(agg_map).dropna(how="all")


def _resample_event_list_daily(events: list, rule: str = "D") -> list:
    if not events:
        return events
    df = pd.DataFrame(events)
    if "date" not in df.columns:
        return events
    df["date"] = pd.to_datetime(df["date"])
    return _resample_df_daily(df.set_index("date"), rule).reset_index().to_dict("records")


def resample_all_results(all_results: dict, resample: bool = True, rule: str = "D") -> dict:
    if not resample:
        return all_results

    resampled_results = {}
    for name, payload in all_results.items():
        payload_copy = payload.copy()
        results = payload_copy.get("results", {}).copy()

        equity = results.get("equity")
        if isinstance(equity, pd.Series) and isinstance(equity.index, pd.DatetimeIndex) and not equity.empty:
            equity = pd.to_numeric(equity, errors="coerce")
            equity_daily = equity.sort_index().resample(rule).last().dropna()
            results["equity"] = equity_daily
            results["daily_returns"] = equity_daily.pct_change().fillna(0)
            # Update stored annualization factor to match resampled frequency so
            # downstream metric computations (e.g. get_performance_metrics) use the
            # correct per-bar factor rather than the original intraday factor.
            if "annualization_factor" in results and not equity_daily.empty:
                from blackwood.portfolio.core import PortfolioBacktester

                results["annualization_factor"] = max(
                    1, round(PortfolioBacktester._infer_annualization_factor(equity_daily.index))
                )

        for df_key in ("weights_history", "leverage_history_df", "allocator_history_df", "risk_rules_audit_df"):
            if isinstance(results.get(df_key), pd.DataFrame):
                results[df_key] = _resample_df_daily(results[df_key], rule)

        for list_key in ("leverage_history", "allocator_history", "risk_rules_audit"):
            if isinstance(results.get(list_key), list):
                results[list_key] = _resample_event_list_daily(results[list_key], rule)

        payload_copy["results"] = results
        resampled_results[name] = payload_copy

    return resampled_results


def resample_returns(returns: pd.DataFrame, resample: bool = True, rule: str = "D") -> pd.DataFrame:
    if not resample:
        return returns
    if returns is None or returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return returns
    gross_returns = 1.0 + returns.sort_index()
    return (gross_returns.resample(rule).prod(min_count=1) - 1.0).dropna(how="all")


def print_risk_rules_summary(label: str, results: dict) -> None:
    audit_df = results.get("risk_rules_audit_df", pd.DataFrame())
    if audit_df.empty:
        print(f"{label:<20} | RiskRules: disabled/no-events")
        return

    kill_days = (
        int(audit_df.loc[audit_df["kill_triggered_this_bar"], "utc_day"].nunique())
        if "kill_triggered_this_bar" in audit_df.columns
        else 0
    )
    half_risk_bars = int(audit_df["half_risk_active"].sum()) if "half_risk_active" in audit_df.columns else 0
    print(f"{label:<20} | RiskRules: kill_days={kill_days:>3} | half_risk_bars={half_risk_bars:>6}")
