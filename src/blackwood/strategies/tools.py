import pandas as pd


def aggregate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.iloc[:0]

    # Single global sort to establish consistent order for 'first' (min ExitBar) and 'last' (max ExitBar)
    sorted_trades = trades.sort_values(["EntryBar", "ExitBar"])

    # Vectorized computation of weights and weighted terms
    abs_size = sorted_trades["Size"].abs()
    sorted_trades = sorted_trades.assign(
        abs_size=abs_size,
        w_entry=abs_size * sorted_trades["EntryPrice"],
        w_exit=abs_size * sorted_trades["ExitPrice"],
        w_risk=abs_size * sorted_trades["PotentialRisk"],
        w_reward=abs_size * sorted_trades["RealizedReward"],
        w_rr=abs_size * sorted_trades["RiskRewardRatio"],
    )

    gb = sorted_trades.groupby("EntryBar", sort=False)

    base = gb[[col for col in trades.columns if col != "EntryBar"]].first()

    # Aggregations (all vectorized)
    agg = gb.agg(
        total_size=("Size", "sum"),
        total_pnl=("PnL", "sum"),
        total_commission=("Commission", "sum"),
        exit_bar=("ExitBar", "last"),
        abs_total_size=("abs_size", "sum"),
        sum_w_entry=("w_entry", "sum"),
        sum_w_exit=("w_exit", "sum"),
        sum_w_risk=("w_risk", "sum"),
        sum_w_reward=("w_reward", "sum"),
        sum_w_rr=("w_rr", "sum"),
    )

    # Combine and compute final metrics
    result = base.join(agg)

    # Weighted averages (safe division)
    denom = result["abs_total_size"]
    mask = denom > 0
    result["EntryPrice"] = 0.0
    result.loc[mask, "EntryPrice"] = result.loc[mask, "sum_w_entry"] / denom[mask]
    result["ExitPrice"] = 0.0
    result.loc[mask, "ExitPrice"] = result.loc[mask, "sum_w_exit"] / denom[mask]
    result["PotentialRisk"] = 0.0
    result.loc[mask, "PotentialRisk"] = result.loc[mask, "sum_w_risk"] / denom[mask]
    result["RealizedReward"] = 0.0
    result.loc[mask, "RealizedReward"] = result.loc[mask, "sum_w_reward"] / denom[mask]
    result["RiskRewardRatio"] = 0.0
    result.loc[mask, "RiskRewardRatio"] = result.loc[mask, "sum_w_rr"] / denom[mask]

    # Simple overrides
    result["Size"] = result["total_size"]
    result["PnL"] = result["total_pnl"]
    result["Commission"] = result["total_commission"]
    result["ExitBar"] = result["exit_bar"]

    # ReturnPct (safe division)
    position_value = result["Size"].abs() * result["EntryPrice"]
    result["ReturnPct"] = 0.0
    mask_pv = position_value != 0
    result.loc[mask_pv, "ReturnPct"] = result.loc[mask_pv, "PnL"] / position_value[mask_pv]

    # Drop temporaries
    result.drop(
        columns=[
            "abs_total_size",
            "sum_w_entry",
            "sum_w_exit",
            "sum_w_risk",
            "sum_w_reward",
            "sum_w_rr",
            "total_size",
            "total_pnl",
            "total_commission",
            "exit_bar",
        ],
        inplace=True,
    )

    # Restore EntryBar as column and original column order
    result = result.reset_index()
    result = result[trades.columns]

    return result
